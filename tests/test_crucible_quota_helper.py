# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for the privileged ext4 quota helper.

No test requires root, a quota-enabled filesystem, or quota utilities.  Kernel
and command boundaries are mocked; the SQLite/flock ledger is exercised for
real in a private temporary directory.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "crucible_quota_helper.py"
SPEC = importlib.util.spec_from_file_location("crucible_quota_helper", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
quota_helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quota_helper
SPEC.loader.exec_module(quota_helper)

LEASE_TOKEN = "T" * 43


@pytest.fixture
def config(tmp_path: Path):
    root = tmp_path / "quota-root"
    state = tmp_path / "state"
    root.mkdir(mode=0o700)
    state.mkdir(mode=0o700)
    return quota_helper.Config(
        allowed_root=root,
        ledger_path=state / "leases.sqlite3",
        project_id_min=700_000,
        project_id_max=700_010,
    )


@pytest.fixture
def mount(config):
    device = config.allowed_root.stat().st_dev
    return quota_helper.MountInfo(
        mount_id=42,
        device=f"{os.major(device)}:{os.minor(device)}",
        root="/",
        mountpoint=config.allowed_root,
        filesystem="ext4",
        source="/dev/loop-test",
        options=frozenset({"rw", "prjquota"}),
    )


@pytest.fixture
def tools():
    return quota_helper.Tools(
        setquota="/usr/sbin/setquota",
        tune2fs="/usr/sbin/tune2fs",
    )


@pytest.fixture
def filesystem_authority(config, mount):
    info = config.allowed_root.stat()
    filesystem_info = os.statvfs(config.allowed_root)
    return quota_helper.FilesystemAuthority(
        canonical_root=config.allowed_root,
        root_device_major=os.major(info.st_dev),
        root_device_minor=os.minor(info.st_dev),
        root_inode=str(info.st_ino),
        root_type=stat.S_IFMT(info.st_mode),
        root_fsid=str(filesystem_info.f_fsid),
        mount_id=mount.mount_id,
        mount_device=mount.device,
        mountpoint=mount.mountpoint,
        mount_root=mount.root,
        mount_filesystem=mount.filesystem,
        mount_source=mount.source,
        filesystem_uuid="12345678-1234-1234-1234-123456789abc",
    )


@pytest.fixture
def manager(
    config,
    mount,
    tools,
    filesystem_authority,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(quota_helper, "_read_mountinfo", lambda: (mount,))
    monkeypatch.setattr(
        quota_helper,
        "_filesystem_context",
        lambda _config, _tools: (
            config.allowed_root,
            mount,
            (mount,),
            filesystem_authority,
        ),
    )
    with quota_helper.Ledger(config) as ledger:
        yield quota_helper.QuotaManager(
            config,
            ledger,
            tools,
            config.allowed_root,
            mount,
            (mount,),
            filesystem_authority,
        )


def _nodes(path: Path):
    info = path.stat()
    return (
        quota_helper.Node(
            path=path,
            device=info.st_dev,
            inode=info.st_ino,
            mode=info.st_mode,
            links=info.st_nlink,
        ),
    )


def _quota(
    *,
    blocks: int = 0,
    soft_blocks: int = 0,
    hard_inodes: int = 0,
    soft_inodes: int = 0,
    bytes_used: int = 0,
    inodes: int = 0,
    valid: int | None = None,
):
    return quota_helper.Quota(
        hard_blocks=blocks,
        soft_blocks=soft_blocks,
        current_bytes=bytes_used,
        hard_inodes=hard_inodes,
        soft_inodes=soft_inodes,
        current_inodes=inodes,
        valid=(
            quota_helper.QIF_BLIMITS
            | quota_helper.QIF_SPACE
            | quota_helper.QIF_ILIMITS
            | quota_helper.QIF_INODES
            if valid is None
            else valid
        ),
    )


def _protective_quota(
    *, bytes_used: int = 0, inodes: int = 0, valid: int | None = None
):
    return _quota(
        blocks=2,
        hard_inodes=1_000_000,
        bytes_used=bytes_used,
        inodes=inodes,
        valid=valid,
    )


def _tune2fs_result():
    return subprocess.CompletedProcess(
        args=["/usr/sbin/tune2fs", "-l", "/dev/loop-test"],
        returncode=0,
        stdout=(
            "Filesystem UUID:          12345678-1234-1234-1234-123456789abc\n"
            "Filesystem features:      project ext_attr\n"
        ),
        stderr="",
    )


def _stub_reservation_boundaries(monkeypatch, manager, path: Path, events: list[str]):
    nodes = _nodes(path)
    monkeypatch.setattr(
        quota_helper,
        "_validate_roots",
        lambda requested, **_kwargs: tuple(requested),
    )
    monkeypatch.setattr(
        quota_helper,
        "_scan_tree",
        lambda _roots, **_kwargs: nodes,
    )
    monkeypatch.setattr(quota_helper, "_validate_fresh", lambda _nodes: None)
    monkeypatch.setattr(quota_helper, "_verify_project", lambda _nodes, _id: None)
    monkeypatch.setattr(quota_helper, "_process_identity", lambda _pid: ("12345", "S"))
    monkeypatch.setattr(quota_helper, "_boot_id", lambda: "boot-a")

    def set_quota(_tools, _mount, _project_id, _blocks, _inodes):
        row = manager.ledger.rows()[-1]
        assert row["state"] == quota_helper.LEASE_PREPARING
        events.append("hard-quota")

    monkeypatch.setattr(quota_helper, "_set_quota", set_quota)
    monkeypatch.setattr(
        quota_helper,
        "_verify_hard_quota",
        lambda *_args: events.append("verify-quota"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_project",
        lambda _node, _project_id: events.append("tag"),
    )


def test_reserve_persists_preparing_before_quota_and_activates(
    tmp_path: Path, monkeypatch, manager
):
    lease_path = manager.root / "job-a"
    lease_path.mkdir()
    events: list[str] = []
    _stub_reservation_boundaries(monkeypatch, manager, lease_path, events)

    token = manager.reserve(
        owner_host=socket.gethostname(),
        owner_pid=123,
        owner_start_token="12345",
        limit_bytes=4097,
        requested_paths=(lease_path,),
    )

    row = manager.ledger.row(token)
    assert row is not None
    assert row["state"] == quota_helper.LEASE_ACTIVE
    assert row["project_id"] == 700_000
    assert row["hard_blocks"] == 4
    assert row["hard_inodes"] == 1_000_000
    assert row["authority_root"] == str(manager.filesystem_authority.canonical_root)
    assert row["root_inode"] == manager.filesystem_authority.root_inode
    assert row["root_fsid"] == manager.filesystem_authority.root_fsid
    assert row["mount_device"] == manager.filesystem_authority.mount_device
    assert row["mount_root"] == manager.filesystem_authority.mount_root
    assert row["filesystem_uuid"] == manager.filesystem_authority.filesystem_uuid
    assert events[0:2] == ["hard-quota", "verify-quota"]
    assert events.index("hard-quota") < events.index("tag")


def test_reserve_failure_leaves_durable_releasing_tombstone(
    monkeypatch, manager
):
    lease_path = manager.root / "job-failed"
    lease_path.mkdir()
    events: list[str] = []
    _stub_reservation_boundaries(monkeypatch, manager, lease_path, events)

    def fail_quota(*_args):
        raise quota_helper.QuotaHelperError("setquota failed")

    monkeypatch.setattr(quota_helper, "_set_quota", fail_quota)
    with pytest.raises(quota_helper.QuotaHelperError, match="setquota failed"):
        manager.reserve(
            owner_host=socket.gethostname(),
            owner_pid=123,
            owner_start_token="12345",
            limit_bytes=1024,
            requested_paths=(lease_path,),
        )

    row = manager.ledger.rows()[-1]
    assert row["state"] == quota_helper.LEASE_RELEASING
    assert "setquota failed" in row["last_error"]


def test_reserve_rejects_sub_kibibyte_limit_before_mutation(manager):
    with pytest.raises(quota_helper.QuotaHelperError, match="at least 1024"):
        manager.reserve(
            owner_host=socket.gethostname(),
            owner_pid=123,
            owner_start_token="12345",
            limit_bytes=1023,
            requested_paths=(manager.root / "unused",),
        )
    assert manager.ledger.rows() == []


def test_ledger_never_reuses_retired_project_ids(
    config,
    filesystem_authority,
):
    with quota_helper.Ledger(config) as ledger:
        first = ledger.create_preparing(
            token="first",
            owner_host="host",
            owner_pid=1,
            owner_start_token="1",
            owner_boot_id="boot",
            limit_bytes=1,
            hard_blocks=1,
            filesystem_authority=filesystem_authority,
            paths=(config.allowed_root / "one",),
        )
        ledger.set_state("first", quota_helper.LEASE_RETIRED)
    with quota_helper.Ledger(config) as ledger:
        second = ledger.create_preparing(
            token="second",
            owner_host="host",
            owner_pid=1,
            owner_start_token="1",
            owner_boot_id="boot",
            limit_bytes=1,
            hard_blocks=1,
            filesystem_authority=filesystem_authority,
            paths=(config.allowed_root / "two",),
        )
    assert (first, second) == (700_000, 700_001)


def test_ledger_rejects_changed_project_id_range(config):
    with quota_helper.Ledger(config):
        pass
    changed = quota_helper.Config(
        allowed_root=config.allowed_root,
        ledger_path=config.ledger_path,
        project_id_min=config.project_id_min + 1,
        project_id_max=config.project_id_max,
    )
    with pytest.raises(quota_helper.QuotaHelperError, match="metadata mismatch"):
        with quota_helper.Ledger(changed):
            pass


def test_ledger_refuses_group_writable_state_file(config):
    config.ledger_path.touch(mode=0o660)
    config.ledger_path.chmod(0o660)
    with pytest.raises(quota_helper.QuotaHelperError, match="state file must be private"):
        with quota_helper.Ledger(config):
            pass


def _insert_lease(
    manager,
    path: Path,
    *,
    token: str = LEASE_TOKEN,
    state: str = "ACTIVE",
):
    project_id = manager.ledger.create_preparing(
        token=token,
        owner_host=socket.gethostname(),
        owner_pid=321,
        owner_start_token="987",
        owner_boot_id="boot-a",
        limit_bytes=2048,
        hard_blocks=2,
        filesystem_authority=manager.filesystem_authority,
        paths=(path,),
    )
    manager.ledger.set_state(token, state)
    return project_id


def _durable_state(manager, token: str = LEASE_TOKEN) -> str:
    row = manager.ledger.db.execute(
        "SELECT state FROM leases WHERE token = ?", (token,)
    ).fetchone()
    assert row is not None
    return str(row["state"])


def _replacement_mount(manager):
    return replace(
        manager.mount,
        mount_id=manager.mount.mount_id + 1,
        source="/dev/hostile-replacement",
    )


def _set_current_filesystem_authority(
    monkeypatch: pytest.MonkeyPatch,
    manager,
    authority,
    *,
    mount=None,
) -> None:
    current_mount = manager.mount if mount is None else mount
    monkeypatch.setattr(
        quota_helper,
        "_filesystem_context",
        lambda _config, _tools: (
            manager.root,
            current_mount,
            (current_mount,),
            authority,
        ),
    )


def _drifted_filesystem_authority(manager, drift: str):
    authority = manager.filesystem_authority
    if drift == "device":
        major = authority.root_device_major + 1
        return replace(
            authority,
            root_device_major=major,
            mount_device=f"{major}:{authority.root_device_minor}",
        )
    if drift == "fsid":
        return replace(authority, root_fsid=str(int(authority.root_fsid) + 1))
    if drift == "uuid":
        return replace(
            authority,
            filesystem_uuid="abcdefab-cdef-abcd-efab-cdefabcdefab",
        )
    if drift == "inode":
        return replace(authority, root_inode=str(int(authority.root_inode) + 1))
    raise AssertionError(f"unknown drift: {drift}")


def test_release_is_successful_but_retains_cap_while_path_exists(
    monkeypatch, manager
):
    path = manager.root / "still-mounted"
    path.mkdir()
    _insert_lease(manager, path)
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("usage must not be queried while path exists"),
    )

    manager.release(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


def test_unknown_release_token_is_idempotent(monkeypatch, manager):
    monkeypatch.setattr(
        manager,
        "_try_retire",
        lambda _token: pytest.fail("unknown token must not trigger cleanup"),
    )
    manager.release("does-not-exist")
    manager.release("")


def test_release_retains_cap_for_open_unlinked_usage(monkeypatch, manager):
    path = manager.root / "gone"
    project_id = _insert_lease(manager, path)
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda _mount, actual_id: (
            _protective_quota(bytes_used=4096, inodes=1)
            if actual_id == project_id
            else pytest.fail("wrong project")
        ),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("nonzero usage must keep the cap"),
    )

    manager.release(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


def test_release_retires_with_the_original_protective_cap_quarantined(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "gone"
    project_id = _insert_lease(manager, path)
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda _mount, actual_id: (
            _protective_quota()
            if actual_id == project_id
            else pytest.fail("wrong project")
        ),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    manager.release(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RETIRED


@pytest.mark.parametrize(
    "state",
    (quota_helper.LEASE_RELEASING, quota_helper.LEASE_RETIRED),
)
def test_retire_rejects_current_mount_substitution_before_path_or_quota_access(
    state: str,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(manager, manager.root / "absent", state=state)
    replacement = _replacement_mount(manager)
    replacement_authority = replace(
        manager.filesystem_authority,
        mount_id=replacement.mount_id,
        mount_source=replacement.source,
    )
    _set_current_filesystem_authority(
        monkeypatch,
        manager,
        replacement_authority,
        mount=replacement,
    )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("mount substitution must stop quota access"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="filesystem authority changed"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == state


def test_retire_rejects_changed_root_device_binding_before_path_or_quota_access(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    different_major = manager.filesystem_authority.root_device_major + 1
    replacement = replace(
        manager.mount,
        device=f"{different_major}:{manager.filesystem_authority.root_device_minor}",
    )
    replacement_authority = replace(
        manager.filesystem_authority,
        root_device_major=different_major,
        mount_device=replacement.device,
    )
    _set_current_filesystem_authority(
        monkeypatch,
        manager,
        replacement_authority,
        mount=replacement,
    )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("device mismatch must stop quota access"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="filesystem authority changed"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


@pytest.mark.parametrize("drift", ("device", "fsid", "uuid", "inode"))
@pytest.mark.parametrize("timing", ("after-commit", "already-retired"))
def test_retire_rejects_full_filesystem_identity_drift_at_every_proof_boundary(
    drift: str,
    timing: str,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    initial_state = (
        quota_helper.LEASE_RELEASING
        if timing == "after-commit"
        else quota_helper.LEASE_RETIRED
    )
    _insert_lease(manager, manager.root / "absent", state=initial_state)
    drifted = _drifted_filesystem_authority(manager, drift)
    original_set_state = manager.ledger.set_state
    committed = timing == "already-retired"
    quota_calls = 0

    def set_state(token: str, state: str, error: str | None = None) -> None:
        nonlocal committed
        original_set_state(token, state, error)
        committed = state == quota_helper.LEASE_RETIRED

    def current_context(_config, _tools):
        authority = drifted if committed else manager.filesystem_authority
        current_mount = manager.mount
        if drift == "device" and committed:
            current_mount = replace(manager.mount, device=authority.mount_device)
        return manager.root, current_mount, (current_mount,), authority

    def get_quota(*_args):
        nonlocal quota_calls
        quota_calls += 1
        return _protective_quota()

    monkeypatch.setattr(quota_helper, "_filesystem_context", current_context)
    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", set_state)

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="current filesystem authority changed",
    ):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RETIRED
    assert quota_calls == (1 if timing == "after-commit" else 0)


@pytest.mark.parametrize(
    ("column", "value", "error"),
    (
        ("mount_id", 0, "malformed durable mount ID"),
        ("mount_id", 41, "durable filesystem authority"),
        ("mount_source", "", "malformed durable mount source"),
        ("mount_source", "/dev/other", "durable filesystem authority"),
    ),
)
def test_retire_rejects_malformed_or_mismatched_durable_mount_evidence(
    column: str,
    value: object,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    with manager.ledger.transaction() as db:
        statements = {
            "mount_id": "UPDATE leases SET mount_id = ? WHERE token = ?",
            "mount_source": "UPDATE leases SET mount_source = ? WHERE token = ?",
        }
        db.execute(
            statements[column],
            (value, LEASE_TOKEN),
        )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("invalid durable mount must stop quota access"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match=error):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_retire_rejects_legacy_rows_without_durable_mount_evidence(manager) -> None:
    with pytest.raises(quota_helper.QuotaHelperError, match="legacy evidence rejected"):
        manager._durable_filesystem_authority({})


def test_schema_migration_never_invents_authority_for_legacy_leases(
    config,
    mount,
    tools,
    filesystem_authority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite3.connect(config.ledger_path) as db:
        db.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE leases (
                token TEXT PRIMARY KEY,
                project_id INTEGER NOT NULL UNIQUE,
                state TEXT NOT NULL,
                owner_host TEXT NOT NULL,
                owner_pid INTEGER NOT NULL,
                owner_start_token TEXT NOT NULL,
                owner_boot_id TEXT NOT NULL,
                limit_bytes INTEGER NOT NULL,
                hard_blocks INTEGER NOT NULL,
                hard_inodes INTEGER NOT NULL,
                mount_id INTEGER NOT NULL,
                mount_source TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_error TEXT
            );
            CREATE TABLE lease_paths (
                token TEXT NOT NULL REFERENCES leases(token),
                path TEXT NOT NULL,
                PRIMARY KEY (token, path)
            );
            """
        )
        db.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", "1"),
                ("project_id_min", "700000"),
                ("project_id_max", "700010"),
                ("inode_hard_limit", "1000000"),
                ("next_project_id", "700001"),
            ),
        )
        db.execute(
            """
            INSERT INTO leases VALUES (
                ?, 700000, 'RELEASING', ?, 321, '987', 'boot-a',
                2048, 2, 1000000, ?, ?, 0.0, 0.0, NULL
            )
            """,
            (LEASE_TOKEN, socket.gethostname(), mount.mount_id, mount.source),
        )
        db.execute(
            "INSERT INTO lease_paths(token, path) VALUES (?, ?)",
            (LEASE_TOKEN, str(config.allowed_root / "absent")),
        )
    config.ledger_path.chmod(0o600)
    monkeypatch.setattr(
        quota_helper,
        "_filesystem_context",
        lambda _config, _tools: (
            config.allowed_root,
            mount,
            (mount,),
            filesystem_authority,
        ),
    )

    with quota_helper.Ledger(config) as ledger:
        schema = ledger.db.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()["value"]
        row = ledger.row(LEASE_TOKEN)
        assert row is not None
        assert schema == "2"
        assert all(row[name] is None for name, _kind in quota_helper._AUTHORITY_COLUMNS)
        manager = quota_helper.QuotaManager(
            config,
            ledger,
            tools,
            config.allowed_root,
            mount,
            (mount,),
            filesystem_authority,
        )
        monkeypatch.setattr(
            quota_helper,
            "_get_quota",
            lambda *_args: pytest.fail("legacy evidence must stop quota access"),
        )
        with pytest.raises(
            quota_helper.QuotaHelperError,
            match="incomplete durable filesystem authority",
        ):
            manager.retire(LEASE_TOKEN)


def test_retire_rejects_one_missing_durable_authority_field_before_quota(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    with manager.ledger.transaction() as db:
        db.execute(
            "UPDATE leases SET root_fsid = NULL WHERE token = ?",
            (LEASE_TOKEN,),
        )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("missing evidence must stop quota access"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("missing evidence must stop quota mutation"),
    )

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="incomplete durable filesystem authority",
    ):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_retire_rejects_malformed_durable_path_before_quota_access(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    with manager.ledger.transaction() as db:
        db.execute(
            "UPDATE lease_paths SET path = ? WHERE token = ?",
            (str(manager.root.parent / "outside"), LEASE_TOKEN),
        )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("invalid durable path must stop quota access"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="durable root/path"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_retire_requires_a_known_releasing_tombstone(manager) -> None:
    path = manager.root / "active"
    _insert_lease(manager, path, token=LEASE_TOKEN)

    with pytest.raises(quota_helper.QuotaHelperError, match="must be RELEASING"):
        manager.retire(LEASE_TOKEN)
    with pytest.raises(quota_helper.QuotaHelperError, match="known lease"):
        manager.retire("U" * 43)
    with pytest.raises(quota_helper.QuotaHelperError, match="exact shape"):
        manager.retire("short")
    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_ACTIVE


def test_retire_fails_while_the_releasing_path_remains(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    _insert_lease(
        manager,
        path,
        token=LEASE_TOKEN,
        state=quota_helper.LEASE_RELEASING,
    )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("path presence must stop before quota query"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="not yet complete"):
        manager.retire(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


@pytest.mark.parametrize("replacement_kind", ("symlink", "renamed-replacement"))
def test_retire_treats_symlink_and_renamed_replacement_as_present(
    replacement_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "leased-path"
    path.mkdir()
    _insert_lease(
        manager,
        path,
        token=LEASE_TOKEN,
        state=quota_helper.LEASE_RELEASING,
    )
    if replacement_kind == "symlink":
        path.rmdir()
        path.symlink_to(manager.root / "missing-target", target_is_directory=True)
    else:
        path.rename(manager.root / "renamed-project-inode")
        path.mkdir()
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("replacement presence must stop quota query"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="not yet complete"):
        manager.retire(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


@pytest.mark.parametrize(
    ("bytes_used", "inodes"),
    ((0, 1), (4096, 1)),
    ids=("renamed-current-inode", "open-unlinked-charged-inode"),
)
def test_retire_retains_cap_while_project_usage_is_charged(
    bytes_used: int,
    inodes: int,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "absent"
    project_id = _insert_lease(
        manager,
        path,
        token=LEASE_TOKEN,
        state=quota_helper.LEASE_RELEASING,
    )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda _mount, actual_id: (
            _protective_quota(bytes_used=bytes_used, inodes=inodes)
            if actual_id == project_id
            else pytest.fail("wrong project")
        ),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("charged usage must retain the hard cap"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="not yet complete"):
        manager.retire(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


@pytest.mark.parametrize(
    ("observed_quota", "error"),
    (
        (_quota(blocks=3, hard_inodes=1_000_000), "cap drifted"),
        (_quota(blocks=2, hard_inodes=999_999), "cap drifted"),
        (
            _protective_quota(valid=quota_helper.QIF_SPACE | quota_helper.QIF_INODES),
            "complete quota evidence",
        ),
    ),
    ids=("block-cap-drift", "inode-cap-drift", "incomplete-validity"),
)
def test_retire_rejects_cap_drift_or_incomplete_quota_evidence(
    observed_quota,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "absent"
    _insert_lease(
        manager,
        path,
        token=LEASE_TOKEN,
        state=quota_helper.LEASE_RELEASING,
    )
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: observed_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match=error):
        manager.retire(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


def test_retire_rejects_a_non_durable_retired_transition(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "absent"
    _insert_lease(
        manager,
        path,
        token=LEASE_TOKEN,
        state=quota_helper.LEASE_RELEASING,
    )
    quotas = iter((_protective_quota(), _protective_quota()))
    original_set_state = manager.ledger.set_state

    def drop_retired_state(token: str, state: str, error: str | None = None) -> None:
        if state != quota_helper.LEASE_RETIRED:
            original_set_state(token, state, error)

    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: next(quotas))
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", drop_retired_state)

    with pytest.raises(quota_helper.QuotaHelperError, match="durable RETIRED"):
        manager.retire(LEASE_TOKEN)

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


def test_retire_proves_state_while_quarantining_the_original_protective_cap(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "absent"
    project_id = _insert_lease(
        manager,
        path,
        token=LEASE_TOKEN,
        state=quota_helper.LEASE_RELEASING,
    )
    quotas = iter((_protective_quota(), _protective_quota()))
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: next(quotas))
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    manager.retire(LEASE_TOKEN)

    assert project_id == 700_000
    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RETIRED


def test_retire_revalidates_an_already_retired_row_without_state_or_quota_mutation(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RETIRED,
    )
    quota_calls = 0

    def get_quota(*_args):
        nonlocal quota_calls
        quota_calls += 1
        return _protective_quota()

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(
        manager.ledger,
        "set_state",
        lambda *_args: pytest.fail("already RETIRED must not rewrite state"),
    )

    manager.retire(LEASE_TOKEN)

    assert quota_calls == 2
    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


def test_retire_rejects_path_reappearance_for_an_already_retired_row(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "reappeared"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_RETIRED)
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: pytest.fail("path presence must stop quota access"),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="not yet complete"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


@pytest.mark.parametrize("persist_before_fault", (False, True))
def test_retire_never_mutates_quota_across_state_commit_faults(
    persist_before_fault: bool,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    original_set_state = manager.ledger.set_state
    quota_calls = 0

    def get_quota(*_args):
        nonlocal quota_calls
        quota_calls += 1
        return _protective_quota()

    def fault_state_commit(
        token: str,
        state: str,
        error: str | None = None,
    ) -> None:
        if state != quota_helper.LEASE_RETIRED:
            original_set_state(token, state, error)
            return
        if persist_before_fault:
            original_set_state(token, state, error)
        raise quota_helper.QuotaHelperError("injected state-commit fault")

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", fault_state_commit)

    with pytest.raises(quota_helper.QuotaHelperError, match="state-commit fault"):
        manager.retire(LEASE_TOKEN)

    expected = (
        quota_helper.LEASE_RETIRED
        if persist_before_fault
        else quota_helper.LEASE_RELEASING
    )
    assert quota_calls == 1
    assert _durable_state(manager) == expected


def test_retire_never_mutates_quota_when_post_commit_row_readback_faults(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    original_row = manager.ledger.row
    original_set_state = manager.ledger.set_state
    committed = False

    def set_state(token: str, state: str, error: str | None = None) -> None:
        nonlocal committed
        original_set_state(token, state, error)
        committed = state == quota_helper.LEASE_RETIRED

    def fault_row_readback(token: str):
        if committed:
            raise quota_helper.QuotaHelperError("injected row-readback fault")
        return original_row(token)

    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: _protective_quota())
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", set_state)
    monkeypatch.setattr(manager.ledger, "row", fault_row_readback)

    with pytest.raises(quota_helper.QuotaHelperError, match="row-readback fault"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


def test_retire_revalidates_mount_identity_after_the_state_commit(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    original_set_state = manager.ledger.set_state
    replacement = _replacement_mount(manager)
    replacement_authority = replace(
        manager.filesystem_authority,
        mount_id=replacement.mount_id,
        mount_source=replacement.source,
    )
    committed = False

    def set_state(token: str, state: str, error: str | None = None) -> None:
        nonlocal committed
        original_set_state(token, state, error)
        committed = state == quota_helper.LEASE_RETIRED

    monkeypatch.setattr(
        quota_helper,
        "_filesystem_context",
        lambda _config, _tools: (
            manager.root,
            replacement if committed else manager.mount,
            (replacement if committed else manager.mount,),
            replacement_authority if committed else manager.filesystem_authority,
        ),
    )
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: _protective_quota())
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", set_state)

    with pytest.raises(quota_helper.QuotaHelperError, match="filesystem authority changed"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


def test_retire_revalidates_path_absence_after_the_state_commit(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "reappears-after-commit"
    _insert_lease(manager, path, state=quota_helper.LEASE_RELEASING)
    original_set_state = manager.ledger.set_state
    quota_calls = 0

    def get_quota(*_args):
        nonlocal quota_calls
        quota_calls += 1
        return _protective_quota()

    def set_state(token: str, state: str, error: str | None = None) -> None:
        original_set_state(token, state, error)
        if state == quota_helper.LEASE_RETIRED:
            path.mkdir()

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", set_state)

    with pytest.raises(quota_helper.QuotaHelperError, match="path reappeared"):
        manager.retire(LEASE_TOKEN)

    assert quota_calls == 1
    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


def test_retire_rejects_durable_path_evidence_change_after_the_state_commit(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    original_path = manager.root / "original-absent"
    _insert_lease(manager, original_path, state=quota_helper.LEASE_RELEASING)
    original_set_state = manager.ledger.set_state
    quota_calls = 0

    def get_quota(*_args):
        nonlocal quota_calls
        quota_calls += 1
        return _protective_quota()

    def set_state(token: str, state: str, error: str | None = None) -> None:
        original_set_state(token, state, error)
        if state == quota_helper.LEASE_RETIRED:
            with manager.ledger.transaction() as db:
                db.execute(
                    "UPDATE lease_paths SET path = ? WHERE token = ?",
                    (str(manager.root / "different-absent"), token),
                )

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )
    monkeypatch.setattr(manager.ledger, "set_state", set_state)

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="durable retirement evidence changed",
    ):
        manager.retire(LEASE_TOKEN)

    assert quota_calls == 1
    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


@pytest.mark.parametrize(
    ("post_commit_observation", "error"),
    (
        (
            quota_helper.QuotaHelperError("injected quota-readback fault"),
            "quota-readback fault",
        ),
        (_quota(blocks=3, hard_inodes=1_000_000), "cap drifted"),
        (_protective_quota(bytes_used=1024, inodes=1), "usage reappeared"),
        (
            _protective_quota(valid=quota_helper.QIF_SPACE),
            "complete quota evidence",
        ),
    ),
    ids=("read-fault", "cap-drift", "usage", "incomplete-validity"),
)
def test_retire_revalidates_exact_quota_evidence_after_the_state_commit(
    post_commit_observation,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    observations = iter((_protective_quota(), post_commit_observation))

    def get_quota(*_args):
        observation = next(observations)
        if isinstance(observation, BaseException):
            raise observation
        return observation

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match=error):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


def test_reconcile_revalidates_quarantine_for_already_retired_rows(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RETIRED,
    )
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: _quota(blocks=3, hard_inodes=1_000_000),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement must never mutate quota"),
    )

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="refusing to overwrite partial or nonzero quota limits",
    ):
        manager.reconcile(socket.gethostname())

    assert _durable_state(manager) == quota_helper.LEASE_RETIRED


def test_reconcile_establishes_the_exact_original_cap_from_wholly_unset_limits(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    project_id = _insert_lease(
        manager,
        path,
        state=quota_helper.LEASE_RELEASING,
    )
    observations = iter((_quota(), _quota(), _protective_quota()))
    mutations: list[tuple[int, int, int]] = []
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: next(observations))
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda _tools, _mount, actual_id, blocks, inodes: mutations.append(
            (actual_id, blocks, inodes)
        ),
    )

    manager.reconcile(socket.gethostname())

    assert mutations == [(project_id, 2, 1_000_000)]
    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


@pytest.mark.parametrize(
    "observed",
    (
        _quota(blocks=1, hard_inodes=1_000_000),
        _quota(blocks=2, hard_inodes=999_999),
        _quota(blocks=0, hard_inodes=1_000_000),
        _quota(blocks=2, hard_inodes=0),
        _quota(soft_blocks=1),
        _quota(soft_inodes=1),
        _quota(blocks=2, soft_blocks=1, hard_inodes=1_000_000),
    ),
)
def test_reconcile_never_overwrites_partial_nonzero_or_lower_caps(
    observed,
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_RELEASING)
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: observed)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("partial quota evidence must not be overwritten"),
    )

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="refusing to overwrite partial or nonzero quota limits",
    ):
        manager.reconcile(socket.gethostname())

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_reconcile_requires_complete_zero_limit_evidence_before_repair(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_RELEASING)
    monkeypatch.setattr(
        quota_helper,
        "_get_quota",
        lambda *_args: _quota(valid=quota_helper.QIF_SPACE),
    )
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("incomplete evidence must not authorize repair"),
    )

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="complete quota evidence",
    ):
        manager.reconcile(socket.gethostname())

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_reconcile_rechecks_all_zero_limits_at_the_repair_boundary(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_RELEASING)
    observations = iter(
        (
            _quota(),
            _quota(blocks=1, hard_inodes=1_000_000),
        )
    )
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: next(observations))
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("boundary drift must stop quota mutation"),
    )

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="refusing to overwrite partial or nonzero quota limits",
    ):
        manager.reconcile(socket.gethostname())

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_reconcile_revalidates_filesystem_after_zero_limit_observation(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_RELEASING)
    drifted = _drifted_filesystem_authority(manager, "uuid")

    def observe_then_replace_filesystem(*_args):
        _set_current_filesystem_authority(monkeypatch, manager, drifted)
        return _quota()

    monkeypatch.setattr(quota_helper, "_get_quota", observe_then_replace_filesystem)
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("filesystem drift must stop quota mutation"),
    )

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="current filesystem authority changed",
    ):
        manager.reconcile(socket.gethostname())

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_quarantine_repair_effect_then_error_is_idempotent_on_retry(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_RELEASING)
    cap_established = False
    mutation_calls = 0

    def get_quota(*_args):
        return _protective_quota() if cap_established else _quota()

    def effect_then_error(*_args) -> None:
        nonlocal cap_established, mutation_calls
        mutation_calls += 1
        cap_established = True
        raise quota_helper.QuotaHelperError("injected effect-then-error")

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(quota_helper, "_set_quota", effect_then_error)

    with pytest.raises(quota_helper.QuotaHelperError, match="effect-then-error"):
        manager.reconcile(socket.gethostname())
    manager.reconcile(socket.gethostname())

    assert mutation_calls == 1
    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_quarantine_repair_no_effect_then_error_retries_the_same_exact_cap(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    project_id = _insert_lease(
        manager,
        path,
        state=quota_helper.LEASE_RELEASING,
    )
    cap_established = False
    mutations: list[tuple[int, int, int]] = []

    def get_quota(*_args):
        return _protective_quota() if cap_established else _quota()

    def repair(_tools, _mount, actual_id, blocks, inodes) -> None:
        nonlocal cap_established
        mutations.append((actual_id, blocks, inodes))
        if len(mutations) == 1:
            raise quota_helper.QuotaHelperError("injected no-effect error")
        cap_established = True

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(quota_helper, "_set_quota", repair)

    with pytest.raises(quota_helper.QuotaHelperError, match="no-effect error"):
        manager.reconcile(socket.gethostname())
    manager.reconcile(socket.gethostname())

    assert mutations == [
        (project_id, 2, 1_000_000),
        (project_id, 2, 1_000_000),
    ]
    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_quarantine_repair_no_effect_readback_retries_the_same_exact_cap(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    path = manager.root / "still-present"
    path.mkdir()
    project_id = _insert_lease(
        manager,
        path,
        state=quota_helper.LEASE_RELEASING,
    )
    cap_established = False
    mutations: list[tuple[int, int, int]] = []

    def get_quota(*_args):
        return _protective_quota() if cap_established else _quota()

    def repair(_tools, _mount, actual_id, blocks, inodes) -> None:
        nonlocal cap_established
        mutations.append((actual_id, blocks, inodes))
        if len(mutations) == 2:
            cap_established = True

    monkeypatch.setattr(quota_helper, "_get_quota", get_quota)
    monkeypatch.setattr(quota_helper, "_set_quota", repair)

    with pytest.raises(
        quota_helper.QuotaHelperError,
        match="quarantine repair verification failed",
    ):
        manager.reconcile(socket.gethostname())
    manager.reconcile(socket.gethostname())

    assert mutations == [
        (project_id, 2, 1_000_000),
        (project_id, 2, 1_000_000),
    ]
    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_retire_never_uses_reconciliation_repair_authority(
    monkeypatch: pytest.MonkeyPatch,
    manager,
) -> None:
    _insert_lease(
        manager,
        manager.root / "absent",
        state=quota_helper.LEASE_RELEASING,
    )
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: _quota())
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("retirement has no quota-mutation authority"),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="cap drifted"):
        manager.retire(LEASE_TOKEN)

    assert _durable_state(manager) == quota_helper.LEASE_RELEASING


def test_reconcile_keeps_exact_live_owner(monkeypatch, manager):
    path = manager.root / "active"
    path.mkdir()
    _insert_lease(manager, path)
    monkeypatch.setattr(quota_helper, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(quota_helper, "_process_identity", lambda _pid: ("987", "S"))

    manager.reconcile(socket.gethostname())

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_ACTIVE


@pytest.mark.parametrize(
    ("identity", "boot_id"),
    [
        (None, "boot-a"),
        (("different", "S"), "boot-a"),
        (("987", "S"), "new-boot"),
    ],
)
def test_reconcile_tombstones_non_exact_owner(
    monkeypatch, manager, identity, boot_id
):
    path = manager.root / "active"
    path.mkdir()
    _insert_lease(manager, path)
    monkeypatch.setattr(quota_helper, "_boot_id", lambda: boot_id)
    monkeypatch.setattr(quota_helper, "_process_identity", lambda _pid: identity)
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: _protective_quota())
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("an existing cap must not be rewritten"),
    )

    manager.reconcile(socket.gethostname())

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


def test_reconcile_always_tombstones_interrupted_preparing(monkeypatch, manager):
    path = manager.root / "preparing"
    path.mkdir()
    _insert_lease(manager, path, state=quota_helper.LEASE_PREPARING)
    monkeypatch.setattr(
        quota_helper,
        "_owner_is_alive",
        lambda **_kwargs: pytest.fail("PREPARING must not trust the owner"),
    )
    monkeypatch.setattr(quota_helper, "_get_quota", lambda *_args: _protective_quota())
    monkeypatch.setattr(
        quota_helper,
        "_set_quota",
        lambda *_args: pytest.fail("an existing cap must not be rewritten"),
    )

    manager.reconcile(socket.gethostname())

    assert manager.ledger.row(LEASE_TOKEN)["state"] == quota_helper.LEASE_RELEASING


def test_scan_rejects_external_hardlink(config):
    root = config.allowed_root / "job"
    root.mkdir()
    original = root / "data"
    original.write_bytes(b"content")
    os.link(original, config.allowed_root / "outside-link")

    with pytest.raises(quota_helper.QuotaHelperError, match="hard link outside"):
        quota_helper._scan_tree((root,), expected_device=root.stat().st_dev)


def test_scan_accepts_hardlinks_contained_in_roots(config):
    root = config.allowed_root / "job"
    root.mkdir()
    original = root / "data"
    original.write_bytes(b"content")
    os.link(original, root / "contained-link")

    nodes = quota_helper._scan_tree((root,), expected_device=root.stat().st_dev)

    assert len(nodes) == 3


def test_canonical_path_rejects_symlink_component(config):
    real = config.allowed_root / "real"
    real.mkdir()
    alias = config.allowed_root / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(quota_helper.QuotaHelperError, match="canonical"):
        quota_helper._canonical_no_symlinks(alias, require_directory=True)


def test_validate_roots_rejects_nested_mount(config, mount):
    job = config.allowed_root / "job"
    nested_path = job / "nested"
    nested_path.mkdir(parents=True)
    nested = quota_helper.MountInfo(
        mount_id=43,
        device=mount.device,
        root=mount.root,
        mountpoint=nested_path,
        filesystem="ext4",
        source="/dev/loop-other",
        options=mount.options,
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="nested mount"):
        quota_helper._validate_roots(
            (job,), root=config.allowed_root, mount=mount, mounts=(mount, nested)
        )


def test_filesystem_context_requires_explicit_prjquota(
    monkeypatch, config, mount, tools
):
    accounting_only = quota_helper.MountInfo(
        mount_id=mount.mount_id,
        device=mount.device,
        root=mount.root,
        mountpoint=mount.mountpoint,
        filesystem=mount.filesystem,
        source=mount.source,
        options=frozenset({"rw", "project"}),
    )
    monkeypatch.setattr(
        quota_helper,
        "_canonical_no_symlinks",
        lambda _path, **_kwargs: config.allowed_root,
    )
    monkeypatch.setattr(
        quota_helper,
        "_read_mountinfo",
        lambda: (accounting_only,),
    )

    with pytest.raises(quota_helper.QuotaHelperError, match="prjquota"):
        quota_helper._filesystem_context(config, tools)


def test_mountinfo_parser_preserves_the_mount_root_field(tmp_path: Path) -> None:
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "42 1 7:42 /subtree /data rw,relatime - ext4 "
        "/dev/loop-test rw,prjquota\n",
        encoding="utf-8",
    )

    (mount,) = quota_helper._read_mountinfo(mountinfo)

    assert mount.root == "/subtree"
    assert mount.mountpoint == Path("/data")


def test_filesystem_authority_uses_and_exactly_closes_a_nofollow_root_fd(
    config,
    mount,
    tools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open
    real_close = os.close
    opened_flags: list[int] = []
    closed: list[int] = []

    def open_root(path, flags, *args):
        opened_flags.append(flags)
        return real_open(path, flags, *args)

    def close_root(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(quota_helper.os, "open", open_root)
    monkeypatch.setattr(quota_helper.os, "close", close_root)
    monkeypatch.setattr(quota_helper, "_read_mountinfo", lambda: (mount,))
    monkeypatch.setattr(quota_helper, "_run", lambda *_args, **_kwargs: _tune2fs_result())

    root, current_mount, _mounts, authority = quota_helper._filesystem_context(
        config,
        tools,
    )

    assert root == config.allowed_root
    assert current_mount == mount
    assert authority.root_inode == str(config.allowed_root.stat().st_ino)
    assert authority.root_fsid == str(os.statvfs(config.allowed_root).f_fsid)
    assert authority.filesystem_uuid == "12345678-1234-1234-1234-123456789abc"
    assert len(opened_flags) == 1
    assert opened_flags[0] & os.O_NOFOLLOW
    assert opened_flags[0] & os.O_DIRECTORY
    assert len(closed) == 1


@pytest.mark.parametrize("fault", ("fstat", "fstatvfs", "mountinfo", "tune2fs"))
def test_filesystem_authority_capture_closes_its_root_fd_on_every_fault(
    fault: str,
    config,
    mount,
    tools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = os.close
    closed: list[int] = []

    def close_root(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(quota_helper.os, "close", close_root)
    monkeypatch.setattr(quota_helper, "_read_mountinfo", lambda: (mount,))
    monkeypatch.setattr(quota_helper, "_run", lambda *_args, **_kwargs: _tune2fs_result())
    if fault == "fstat":
        monkeypatch.setattr(
            quota_helper.os,
            "fstat",
            lambda _descriptor: (_ for _ in ()).throw(OSError("fstat fault")),
        )
    elif fault == "fstatvfs":
        monkeypatch.setattr(
            quota_helper.os,
            "fstatvfs",
            lambda _descriptor: (_ for _ in ()).throw(OSError("fstatvfs fault")),
        )
    elif fault == "mountinfo":
        monkeypatch.setattr(
            quota_helper,
            "_read_mountinfo",
            lambda: (_ for _ in ()).throw(
                quota_helper.QuotaHelperError("mountinfo fault")
            ),
        )
    else:
        monkeypatch.setattr(
            quota_helper,
            "_run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                quota_helper.QuotaHelperError("tune2fs fault")
            ),
        )

    with pytest.raises(quota_helper.QuotaHelperError):
        quota_helper._filesystem_context(config, tools)

    assert len(closed) == 1


def test_filesystem_authority_capture_surfaces_one_close_fault_without_retry(
    config,
    mount,
    tools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_close = os.close
    close_calls = 0

    def fail_after_close(descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(descriptor)
        raise OSError("injected close fault")

    monkeypatch.setattr(quota_helper.os, "close", fail_after_close)
    monkeypatch.setattr(quota_helper, "_read_mountinfo", lambda: (mount,))
    monkeypatch.setattr(quota_helper, "_run", lambda *_args, **_kwargs: _tune2fs_result())

    with pytest.raises(quota_helper.QuotaHelperError, match="cannot close"):
        quota_helper._filesystem_context(config, tools)

    assert close_calls == 1


def test_set_project_preserves_flags_and_sets_inherit_only_for_directory(
    tmp_path: Path, monkeypatch
):
    directory = tmp_path / "dir"
    directory.mkdir()
    info = directory.stat()
    node = quota_helper.Node(
        directory, info.st_dev, info.st_ino, info.st_mode, info.st_nlink
    )
    initial = quota_helper.FsXattr(
        xflags=0x00080000,
        extsize=1,
        nextents=2,
        project_id=0,
        cowextsize=3,
        pad=b"\0" * 8,
    )
    written: list[bytes] = []

    def ioctl(_fd, operation, argument, *_mutate):
        if operation == quota_helper.FS_IOC_FSGETXATTR:
            argument[:] = initial.pack()
        elif operation == quota_helper.FS_IOC_FSSETXATTR:
            written.append(argument)
        else:
            pytest.fail(f"unexpected ioctl {operation}")
        return 0

    monkeypatch.setattr(quota_helper.fcntl, "ioctl", ioctl)

    quota_helper._set_project(node, 765_432)

    updated = quota_helper.FsXattr.unpack(written[0])
    assert updated.project_id == 765_432
    assert updated.xflags == initial.xflags | quota_helper.FS_XFLAG_PROJINHERIT
    assert updated.extsize == initial.extsize
    assert updated.cowextsize == initial.cowextsize


def test_root_gate_and_process_zombie_are_fail_closed(monkeypatch):
    monkeypatch.setattr(quota_helper.os, "geteuid", lambda: 1000)
    with pytest.raises(quota_helper.QuotaHelperError, match="must run as root"):
        quota_helper._assert_root()

    stat_line = "12 (worker with ) paren) Z " + " ".join(
        ["0"] * 18 + ["98765"] + ["0"] * 10
    )
    original_read_text = quota_helper.Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/12/stat":
            return stat_line
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(quota_helper.Path, "read_text", read_text)
    assert quota_helper._process_identity(12) is None


def test_secure_script_rejects_a_direct_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "helper"
    target.write_text("helper", encoding="utf-8")
    alias = tmp_path / "helper-link"
    alias.symlink_to(target)
    monkeypatch.setattr(quota_helper, "__file__", str(alias))

    with pytest.raises(quota_helper.QuotaHelperError, match="not a symlink"):
        quota_helper._assert_secure_script()


@pytest.mark.parametrize("alias_root", ("/proc/self/fd", "/dev/fd"))
def test_secure_script_authenticates_only_exact_descriptor_aliases(
    alias_root: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory_info = os.stat_result(
        (stat.S_IFDIR | 0o500, 41, 7, 1, 0, 0, 0, 0, 0, 0)
    )
    link_info = os.stat_result(
        (stat.S_IFLNK | 0o777, 43, 7, 1, 0, 0, 0, 0, 0, 0)
    )
    executable_info = os.stat_result(
        (stat.S_IFREG | 0o755, 47, 8, 1, 0, 0, 1024, 0, 0, 0)
    )
    alias = f"{alias_root}/9"

    def lstat(path: object) -> os.stat_result:
        path_text = os.fspath(path)
        if path_text == alias:
            return link_info
        if path_text == "/proc/self/fd":
            return directory_info
        if path_text == "/dev/fd":
            return link_info
        raise AssertionError(f"unexpected lstat path: {path_text}")

    def followed_stat(
        path: object, *, follow_symlinks: bool = True
    ) -> os.stat_result:
        assert follow_symlinks
        path_text = os.fspath(path)
        if path_text in {"/proc/self/fd", "/dev/fd"}:
            return directory_info
        if path_text == alias:
            return executable_info
        raise AssertionError(f"unexpected stat path: {path_text}")

    def fstat(descriptor: int) -> os.stat_result:
        if descriptor != 9:
            raise AssertionError(f"unexpected descriptor: {descriptor}")
        return executable_info

    monkeypatch.setattr(quota_helper, "__file__", alias)
    monkeypatch.setattr(quota_helper.os, "lstat", lstat)
    monkeypatch.setattr(quota_helper.os, "stat", followed_stat)
    monkeypatch.setattr(quota_helper.os, "fstat", fstat)

    quota_helper._assert_secure_script()


def test_real_helper_accepts_exact_linux_descriptor_execution(tmp_path: Path) -> None:
    unshare = shutil.which("unshare")
    if unshare is None:
        pytest.skip("util-linux unshare is unavailable")
    helper = tmp_path / "crucible-quota-helper"
    helper.write_bytes(SCRIPT.read_bytes())
    helper.chmod(0o755)
    descriptor = os.open(helper, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        alias = f"/proc/self/fd/{descriptor}"
        completed = subprocess.run(
            (
                unshare,
                "-Ur",
                alias,
                "--config",
                str(tmp_path / "intentionally-missing.json"),
                "probe",
            ),
            check=False,
            capture_output=True,
            close_fds=True,
            pass_fds=(descriptor,),
            timeout=10.0,
        )
    finally:
        os.close(descriptor)
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if "unshare failed" in stderr:
        pytest.skip(f"unprivileged user namespaces are unavailable: {stderr.strip()}")
    assert completed.returncode == 1
    assert "helper executable" not in stderr
    assert "configuration does not exist" in stderr


def test_real_helper_accepts_linux_fexecve_script_alias(tmp_path: Path) -> None:
    unshare = shutil.which("unshare")
    if unshare is None:
        pytest.skip("util-linux unshare is unavailable")
    if os.execve not in os.supports_fd:
        pytest.skip("fd-form execve is unavailable")
    helper = tmp_path / "crucible-quota-helper"
    helper.write_bytes(SCRIPT.read_bytes())
    helper.chmod(0o755)
    driver = """
import os
import sys

descriptor = os.open(sys.argv[1], os.O_RDONLY)
os.set_inheritable(descriptor, True)
os.execve(
    descriptor,
    (
        sys.argv[1],
        "--config",
        sys.argv[2],
        "probe",
    ),
    dict(os.environ),
)
"""
    completed = subprocess.run(
        (
            unshare,
            "-Ur",
            sys.executable,
            "-I",
            "-S",
            "-c",
            driver,
            str(helper),
            str(tmp_path / "intentionally-missing.json"),
        ),
        check=False,
        capture_output=True,
        close_fds=True,
        timeout=10.0,
    )
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if "unshare failed" in stderr:
        pytest.skip(f"unprivileged user namespaces are unavailable: {stderr.strip()}")
    assert completed.returncode == 1
    assert "helper executable" not in stderr
    assert "configuration does not exist" in stderr


def test_setquota_uses_argv_and_kib_hard_limit(monkeypatch, tools, mount):
    commands = []
    monkeypatch.setattr(
        quota_helper,
        "_run",
        lambda command, **_kwargs: commands.append(command),
    )

    quota_helper._set_quota(tools, mount, 700_123, 17, 456)

    assert commands == [
        [
            "/usr/sbin/setquota",
            "-P",
            "700123",
            "0",
            "17",
            "0",
            "456",
            str(mount.mountpoint),
        ]
    ]


def test_project_id_exhaustion_is_durable(config, filesystem_authority):
    tiny = quota_helper.Config(
        allowed_root=config.allowed_root,
        ledger_path=config.ledger_path,
        project_id_min=10,
        project_id_max=10,
    )
    # Replace any fixture-range database with a fresh one for this test.
    if tiny.ledger_path.exists():
        tiny.ledger_path.unlink()
    with quota_helper.Ledger(tiny) as ledger:
        ledger.create_preparing(
            token="only",
            owner_host="host",
            owner_pid=1,
            owner_start_token="1",
            owner_boot_id="boot",
            limit_bytes=1,
            hard_blocks=1,
            filesystem_authority=filesystem_authority,
            paths=(tiny.allowed_root / "one",),
        )
        with pytest.raises(quota_helper.QuotaHelperError, match="exhausted"):
            ledger.create_preparing(
                token="overflow",
                owner_host="host",
                owner_pid=1,
                owner_start_token="1",
                owner_boot_id="boot",
                limit_bytes=1,
                hard_blocks=1,
                filesystem_authority=filesystem_authority,
                paths=(tiny.allowed_root / "two",),
            )
        next_id = ledger.db.execute(
            "SELECT value FROM metadata WHERE key='next_project_id'"
        ).fetchone()["value"]
    assert next_id == "11"
    with sqlite3.connect(tiny.ledger_path) as db:
        assert db.execute("SELECT count(*) FROM leases").fetchone()[0] == 1
