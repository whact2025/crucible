# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hostile lifecycle tests for Kimi's process-isolated source mutators."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.tools import _isolated_mutation_host as host_module
from crucible.tools._isolated_mutation_host import (
    IsolatedMutationHost,
    MutationHostLifecycleError,
    MutationHostProtocolError,
    MutationHostQualificationError,
    MutationHostTimeoutError,
    ProcessHostedToolSet,
    _ProjectImportBlocker,
    build_isolated_toolset_spec,
)
from crucible.tools._dispatch_budget import (
    ProgressBoundedMutatingToolSet,
    ToolDispatchBudgetExceeded,
)
from crucible.tools import ToolContext, composite
from crucible.tools.profiles import (
    build_autonomous_implementer_profile,
    build_exact_create_recovery_profile,
    build_implementer_profile,
)
from crucible.tools.source import build_source_tools
from crucible.tools.write import build_exact_create_tools, build_write_tools


_FAKE_DELETE_JOURNALS: dict[
    tuple[str, str],
    "_FakeDeleteJournal",
] = {}


class _FakeDeleteJournal:
    """Narrow host-protocol double; journal CAS has separate real tests."""

    def __init__(self, root: Path, journal_dir: Path, transaction_id: str):
        self.root = root
        opened = root.stat()
        self.root_identity = (opened.st_dev, opened.st_ino)
        self.journal_dir = journal_dir
        self.transaction_id = transaction_id
        self.receipt: dict[str, object] | None = None
        self.lock = threading.Lock()

    def _assert_root(self, descriptor: int | None) -> None:
        assert descriptor is not None
        held = os.fstat(descriptor)
        assert (held.st_dev, held.st_ino) == self.root_identity

    def arm_pending_delete_quarantine(
        self,
        *,
        token: str,
        target_relative: str,
        parent_chain: object | None = None,
        boot_id: object | None = None,
        workspace_root_descriptor: int | None = None,
    ) -> dict[str, object]:
        self._assert_root(workspace_root_descriptor)
        target = Path(target_relative)
        quarantine_name = (
            f".{target.name}.crucible-delete-{token}.quarantine"
        )
        quarantine = target.parent / quarantine_name
        assert isinstance(parent_chain, list) and parent_chain
        assert all(
            isinstance(identity, list) and len(identity) == 6
            for identity in parent_chain
        )
        assert isinstance(boot_id, str) and boot_id
        with self.lock:
            candidate: dict[str, object] = {
                "token": token,
                "target_relative": target.as_posix(),
                "quarantine_relative": quarantine.as_posix(),
                "parent_chain": parent_chain,
                "state": "reserved",
                "quarantine_device": None,
                "quarantine_inode": None,
                "quarantine_mode": None,
                "quarantine_mount_id": None,
                "child_pid": None,
                "child_start_ticks": None,
                "boot_id": boot_id,
            }
            if self.receipt is None:
                assert not (self.root / quarantine).exists()
                self.receipt = candidate
            else:
                assert self.receipt == candidate
            return dict(self.receipt)

    def bind_pending_delete_quarantine(
        self,
        expected: dict[str, object],
        *,
        quarantine_device: int,
        quarantine_inode: int,
        quarantine_mode: int,
        quarantine_mount_id: int,
        child_pid: int,
        child_start_ticks: int,
        boot_id: str,
        workspace_root_descriptor: int | None = None,
        parent_descriptor: int | None = None,
    ) -> dict[str, object]:
        self._assert_root(workspace_root_descriptor)
        with self.lock:
            assert parent_descriptor is not None
            held_parent = os.fstat(parent_descriptor)
            expected_parent = expected["parent_chain"][-1]
            assert [
                held_parent.st_dev,
                held_parent.st_ino,
                stat.S_IFMT(held_parent.st_mode),
                held_parent.st_uid,
                held_parent.st_gid,
            ] == expected_parent[:5]
            quarantine_name = Path(
                str(expected["quarantine_relative"])
            ).name
            observed = os.stat(
                quarantine_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            assert (observed.st_dev, observed.st_ino, observed.st_mode) == (
                quarantine_device,
                quarantine_inode,
                quarantine_mode,
            )
            assert quarantine_mount_id > 0
            assert (
                child_pid > 1
                and child_start_ticks > 0
                and boot_id == expected["boot_id"]
            )
            proposed = {
                **expected,
                "state": "bound",
                "quarantine_device": quarantine_device,
                "quarantine_inode": quarantine_inode,
                "quarantine_mode": quarantine_mode,
                "quarantine_mount_id": quarantine_mount_id,
                "child_pid": child_pid,
                "child_start_ticks": child_start_ticks,
                "boot_id": boot_id,
            }
            if self.receipt != proposed:
                assert self.receipt == expected
                self.receipt = proposed
            return dict(self.receipt)

    def clear_pending_delete_quarantine(
        self,
        expected: dict[str, object],
        *,
        workspace_root_descriptor: int | None = None,
        parent_descriptor: int | None = None,
    ) -> None:
        self._assert_root(workspace_root_descriptor)
        with self.lock:
            quarantine_name = Path(
                str(expected["quarantine_relative"])
            ).name
            if parent_descriptor is None:
                assert not (
                    self.root / str(expected["quarantine_relative"])
                ).exists()
            else:
                held_parent = os.fstat(parent_descriptor)
                expected_parent = expected["parent_chain"][-1]
                assert [
                    held_parent.st_dev,
                    held_parent.st_ino,
                    stat.S_IFMT(held_parent.st_mode),
                    held_parent.st_uid,
                    held_parent.st_gid,
                ] == expected_parent[:5]
                with pytest.raises(FileNotFoundError):
                    os.stat(
                        quarantine_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
            if self.receipt is None:
                return
            assert self.receipt == expected
            self.receipt = None


@pytest.fixture(autouse=True)
def _fake_delete_journal_resolver(monkeypatch: pytest.MonkeyPatch):
    from crucible.workflows import _mutation_journal

    _FAKE_DELETE_JOURNALS.clear()

    def resolve(journal_dir: str | Path, transaction_id: str):
        return _FAKE_DELETE_JOURNALS[(str(journal_dir), transaction_id)]

    monkeypatch.setattr(
        _mutation_journal,
        "resolve_live_mutation_journal",
        resolve,
    )
    yield
    _FAKE_DELETE_JOURNALS.clear()


def _text_authoring_toolset(
    root: Path,
    *,
    scope: tuple[str, ...],
    include_delete: bool = False,
):
    journal_dir = root / ".crucible-test-journal"
    journal_dir.mkdir(exist_ok=True)
    transaction_id = hashlib.sha256(
        os.fsencode(str(root.resolve()))
    ).hexdigest()[:32]
    _FAKE_DELETE_JOURNALS[(str(journal_dir), transaction_id)] = (
        _FakeDeleteJournal(root, journal_dir, transaction_id)
    )
    context = ToolContext(
        workspace_root=root,
        repo_root=root,
        write_scope=scope,
        mutation_journal_dir=journal_dir,
        mutation_journal_transaction_id=transaction_id,
    )
    write_names = {"write_source", "apply_patch"}
    if include_delete:
        write_names.add("delete_source")
    return composite(
        build_source_tools(context),
        build_write_tools(context).filter_to(write_names),
    )


def test_project_import_blocker_rejects_only_crucible_namespace() -> None:
    blocker = _ProjectImportBlocker()

    assert blocker.find_spec("json") is None
    with pytest.raises(
        MutationHostQualificationError,
        match="project imports are sealed",
    ):
        blocker.find_spec("crucible")
    with pytest.raises(
        MutationHostQualificationError,
        match="project imports are sealed",
    ):
        blocker.find_spec("crucible.tools")


def test_project_import_seal_rejects_preload_and_retention_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sys = SimpleNamespace(modules={"crucible": object()}, meta_path=[])
    monkeypatch.setattr(host_module, "sys", fake_sys)

    with pytest.raises(
        MutationHostQualificationError,
        match="outside its self-contained closure",
    ):
        host_module._seal_child_project_imports()

    fake_sys.modules.clear()
    blocker = host_module._seal_child_project_imports()
    assert fake_sys.meta_path == [blocker]
    host_module._verify_child_project_import_seal(blocker)

    fake_sys.modules["crucible.injected"] = object()
    with pytest.raises(
        MutationHostQualificationError,
        match="loaded project modules after sealing",
    ):
        host_module._verify_child_project_import_seal(blocker)

    fake_sys.modules.clear()
    fake_sys.meta_path.insert(0, object())
    with pytest.raises(
        MutationHostQualificationError,
        match="blocker retention is uncertain",
    ):
        host_module._verify_child_project_import_seal(blocker)


def _host(
    root: Path,
    *,
    scope: tuple[str, ...] = ("target.txt",),
    deadline: float = 10.0,
    before_start_ack=None,
    before_publish_ack=None,
    before_quarantine_ack=None,
) -> IsolatedMutationHost:
    toolset = _text_authoring_toolset(
        root,
        scope=scope,
        include_delete=True,
    )
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset(
            {
                "write_source",
                "apply_patch",
                "delete_source",
            }
        ),
    )
    return IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + deadline,
        _before_start_ack=before_start_ack,
        _before_publish_ack=before_publish_ack,
        _before_quarantine_ack=before_quarantine_ack,
    )


def _assert_process_group_gone(pgid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def _assert_descriptor_open(descriptor: int) -> None:
    os.fstat(descriptor)


def _assert_descriptor_closed(descriptor: int) -> None:
    with pytest.raises(OSError) as caught:
        os.fstat(descriptor)
    assert caught.value.errno == errno.EBADF


def _copy_mutation_host_source(package_root: Path, source: Path) -> Path:
    """Create the minimal exact package layout used by source-hostile tests."""

    target = package_root / "tools" / "_isolated_mutation_host.py"
    target.parent.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (target.parent / "__init__.py").write_text("", encoding="utf-8")
    target.write_bytes(source.read_bytes())
    return target


def test_canonical_text_writes_execute_in_reapable_child(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    host.start()
    pid = host.pid
    assert pid is not None
    assert pid != os.getpid()

    created = host.dispatch(
        "write_source",
        {
            "file_path": "target.txt",
            "content": "alpha\n",
        },
    )
    digest = hashlib.sha256(b"alpha\n").hexdigest()
    patched = host.dispatch(
        "apply_patch",
        {
            "file_path": "target.txt",
            "before": "alpha",
            "after": "beta",
        },
    )
    stale = host.dispatch(
        "write_source",
        {
            "file_path": "target.txt",
            "content": "wrong\n",
            "expected_sha256": digest,
        },
    )
    host.stop()

    assert created == "OK: wrote 1 lines to target.txt"
    assert "OK: replaced" in patched
    assert "stale expected_sha256" in stale
    assert (tmp_path / "target.txt").read_text(encoding="utf-8") == "beta\n"
    assert host.ever_started_write is True
    assert host.is_reaped is True
    _assert_process_group_gone(pid)
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))


def test_nested_publication_is_descriptor_relative(tmp_path: Path) -> None:
    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    with IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    ) as host:
        result = host.dispatch(
            "write_source",
            {
                "file_path": "nested/deep/target.txt",
                "content": "descriptor rooted\n",
            },
        )
    assert result == "OK: wrote 1 lines to nested/deep/target.txt"
    assert (tmp_path / "nested/deep/target.txt").read_text() == "descriptor rooted\n"


def test_symlinked_parent_cannot_redirect_descriptor_relative_write(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    toolset = _text_authoring_toolset(tmp_path, scope=("linked",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    with pytest.raises(MutationHostProtocolError):
        host.dispatch(
            "write_source",
            {"file_path": "linked/target.txt", "content": "must not redirect\n"},
        )
    assert host.is_reaped is True
    assert not (real / "target.txt").exists()


def test_host_uses_posix_spawn_process_group_not_popen_fork_exec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    observed: list[dict[str, Any]] = []
    real_spawn = host_mod._atomic_posix_spawn_closefrom
    secret = "must-not-reach-isolated-mutation-child"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)

    class CredentialGuard(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            if key == "MOONSHOT_API_KEY":
                pytest.fail("provider credential value was loaded")
            return super().__getitem__(key)

        def items(self):
            pytest.fail("environment values must not be enumerated")

    monkeypatch.setattr(host_mod.os, "environ", CredentialGuard(os.environ))

    def recording_spawn(
        process,
        command,
        environment,
        duplicate_actions,
        *,
        closefrom,
    ):
        observation = {
            "argv": tuple(command),
            "environment": dict(environment),
            "duplicate_actions": tuple(duplicate_actions),
            "closefrom": closefrom,
        }
        observed.append(observation)
        return real_spawn(
            process,
            command,
            environment,
            duplicate_actions,
            closefrom=closefrom,
        )

    monkeypatch.setattr(host_mod, "_atomic_posix_spawn_closefrom", recording_spawn)
    monkeypatch.setattr(
        os,
        "posix_spawn",
        lambda *_args, **_kwargs: pytest.fail(
            "mutation host must not use snapshot-based os.posix_spawn"
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "mutation host must not enter subprocess fork_exec"
        ),
    )
    pid: int | None = None
    with _host(tmp_path) as host:
        pid = host.pid
        process = host._process  # noqa: SLF001
        assert process is not None
        assert len(process.raw_descriptors) == 2
        assert "OK: wrote" in host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "spawned\n",
            },
        )

    assert len(observed) == 1
    assert observed[0]["closefrom"] == 6
    assert observed[0]["duplicate_actions"]
    argv = observed[0]["argv"]
    assert argv[1:5] == ("-I", "-B", "-S", "-c")
    assert argv[-1] == "--child"
    assert "-m" not in argv
    environment = observed[0]["environment"]
    source_descriptor = int(environment["CRUCIBLE_ISOLATED_MUTATION_SOURCE_FD"])
    assert source_descriptor == 3
    assert int(environment["CRUCIBLE_ISOLATED_MUTATION_ROOT_FD"]) == 4
    mutation_roots = json.loads(environment["CRUCIBLE_ISOLATED_MUTATION_BOUND_ROOTS"])
    assert len(mutation_roots) == 1
    assert mutation_roots[0]["descriptor"] == 5
    duplicate_targets = {target for _source, target in observed[0]["duplicate_actions"]}
    assert {0, 1, 2, 3, 4, 5} <= duplicate_targets
    assert {target for target in duplicate_targets if target < 6} == set(range(6))
    assert secret not in repr(argv)
    assert secret not in repr(environment)
    assert pid is not None
    assert IsolatedMutationHost._process_group_exists(pid) is False


@pytest.mark.parametrize("tampering", ["extra-field", "wrong-session"])
def test_ready_frame_is_closed_and_attests_the_exact_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tampering: str,
) -> None:
    host = _host(tmp_path)
    real_read_frame = host._read_frame  # noqa: SLF001
    child_pids: list[int] = []

    def tampered_read_frame(deadline: float) -> dict[str, Any]:
        frame = real_read_frame(deadline)
        if frame.get("type") == "ready":
            pid = host.pid
            assert pid is not None
            child_pids.append(pid)
            if tampering == "extra-field":
                frame["unattested"] = True
            else:
                frame["sid"] = int(frame["sid"]) + 1
        return frame

    monkeypatch.setattr(host, "_read_frame", tampered_read_frame)
    with pytest.raises(
        MutationHostProtocolError,
        match="did not attest readiness",
    ):
        host.start()

    assert len(child_pids) == 1
    assert host.pid is None
    assert host.is_reaped is True
    assert host._private_root is None  # noqa: SLF001
    _assert_process_group_gone(child_pids[0])


def test_preopen_source_replacement_cannot_define_executable_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    production_source = Path(host_mod.__file__).resolve(strict=True)
    package_root = tmp_path / "source" / "crucible"
    source = _copy_mutation_host_source(package_root, production_source)
    monkeypatch.setattr(host_mod, "__file__", str(source))
    real_spawn = host_mod._posix_spawn_child
    replacement_identities: list[tuple[int, int]] = []

    def replacing_spawn(*args, **kwargs):
        original_identity = source.stat()
        replacement = source.with_suffix(".replacement")
        replacement.write_text(
            "raise RuntimeError('replacement source executed')\n",
            encoding="utf-8",
        )
        os.replace(replacement, source)
        replacement_identity = source.stat()
        replacement_identities.extend(
            [
                (original_identity.st_dev, original_identity.st_ino),
                (replacement_identity.st_dev, replacement_identity.st_ino),
            ]
        )
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(host_mod, "_posix_spawn_child", replacing_spawn)
    host = _host(tmp_path)
    try:
        host.start()
        result = host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "sealed source\n"},
        )
    finally:
        host.stop()

    assert len(replacement_identities) == 2
    assert replacement_identities[0] != replacement_identities[1]
    assert "replacement source executed" in source.read_text(encoding="utf-8")
    assert result == "OK: wrote 1 lines to target.txt"
    assert (tmp_path / "target.txt").read_text() == "sealed source\n"
    assert host.pid is None
    assert host.is_reaped is True
    assert host._private_root is None  # noqa: SLF001


def test_executable_source_package_rebind_is_not_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    production_source = Path(host_mod.__file__).resolve(strict=True)
    package_root = tmp_path / "source" / "crucible"
    source = _copy_mutation_host_source(package_root, production_source)
    retained_root = tmp_path / "retained-crucible"
    monkeypatch.setattr(host_mod, "__file__", str(source))
    real_spawn = host_mod._posix_spawn_child

    def rebinding_spawn(*args, **kwargs):
        package_root.rename(retained_root)
        _copy_mutation_host_source(package_root, production_source)
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(host_mod, "_posix_spawn_child", rebinding_spawn)
    host = _host(tmp_path)
    try:
        host.start()
        result = host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "bound before path rebind\n"},
        )
    finally:
        host.stop()

    assert retained_root.is_dir()
    assert package_root.is_dir()
    assert result == "OK: wrote 1 lines to target.txt"
    assert host.pid is None
    assert host.is_reaped is True
    assert host._private_root is None  # noqa: SLF001


def test_ready_child_uses_complete_self_contained_mutation_closure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    production_source = Path(host_mod.__file__).resolve(strict=True)
    production_root = production_source.parent.parent
    package_root = tmp_path / "source" / "crucible"
    source = _copy_mutation_host_source(package_root, production_source)
    dependency_paths = (
        Path("tools/_context.py"),
        Path("tools/_paths.py"),
        Path("tools/write/handlers.py"),
    )
    for relative in dependency_paths:
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((production_root / relative).read_bytes())

    monkeypatch.setattr(host_mod, "__file__", str(source))
    host = _host(tmp_path)
    host.start()
    try:
        source.write_text(
            "raise RuntimeError('host source replacement must never execute')\n",
            encoding="utf-8",
        )
        for relative in dependency_paths:
            (package_root / relative).write_text(
                "raise RuntimeError('replacement must never execute')\n",
                encoding="utf-8",
            )
        result = host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "sealed closure\n"},
        )
    finally:
        host.stop()

    assert result == "OK: wrote 1 lines to target.txt"
    assert (tmp_path / "target.txt").read_text() == "sealed closure\n"
    assert not tuple(package_root.rglob("*.pyc"))
    assert not tuple(package_root.rglob("__pycache__"))


def test_mutable_source_path_is_never_reopened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    production_source = Path(host_mod.__file__).resolve(strict=True)
    package_root = tmp_path / "fifo-source" / "crucible"
    source = _copy_mutation_host_source(package_root, production_source)
    source.unlink()
    os.mkfifo(source, mode=0o600)
    monkeypatch.setattr(host_mod, "__file__", str(source))
    host = _host(tmp_path, deadline=2.0)
    started = time.monotonic()
    try:
        host.start()
        assert time.monotonic() - started < 1.0
        assert "OK: wrote" in host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "never opened fifo\n"},
        )
    finally:
        host.stop()


def test_in_place_source_mutation_before_spawn_cannot_execute_unsealed_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    production_source = Path(host_mod.__file__).resolve(strict=True)
    package_root = tmp_path / "fifo-race-source" / "crucible"
    source = _copy_mutation_host_source(package_root, production_source)
    monkeypatch.setattr(host_mod, "__file__", str(source))
    original_identity = source.stat()
    real_spawn = host_mod._posix_spawn_child
    marker = tmp_path / "unattested-source-ran"

    def mutating_spawn(*args, **kwargs):
        source.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
            encoding="utf-8",
        )
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(host_mod, "_posix_spawn_child", mutating_spawn)
    host = _host(tmp_path, deadline=2.0)
    marker_was_absent = False
    try:
        host.start()
        assert "OK: wrote" in host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "immutable image\n"},
        )
        marker_was_absent = not marker.exists()
    finally:
        host.stop()
        marker.unlink(missing_ok=True)
    mutated_identity = source.stat()
    assert (mutated_identity.st_dev, mutated_identity.st_ino) == (
        original_identity.st_dev,
        original_identity.st_ino,
    )
    assert mutated_identity.st_mtime_ns != original_identity.st_mtime_ns
    assert marker_was_absent is True


def test_loaded_executable_source_image_is_write_sealed() -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    image = host_mod._LOADED_SOURCE_IMAGE
    assert image is not None
    assert (
        image.seals & host_mod._SOURCE_IMAGE_REQUIRED_SEALS
        == host_mod._SOURCE_IMAGE_REQUIRED_SEALS
    )
    before = os.pread(image.descriptor, 32, 0)
    with pytest.raises(OSError) as caught:
        os.pwrite(image.descriptor, b"unattested", 0)
    assert caught.value.errno == errno.EPERM
    assert os.pread(image.descriptor, 32, 0) == before


def test_replacement_loader_bytes_cannot_define_source_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    class ReplacementLoader:
        @staticmethod
        def get_source(_name: str) -> str:
            return "raise RuntimeError('replacement loader bytes executed')\n"

    monkeypatch.setattr(host_mod, "__loader__", ReplacementLoader())
    with pytest.raises(
        MutationHostQualificationError,
        match="trusted digest",
    ):
        host_mod._capture_loaded_source_image()


def test_exact_create_recovery_ignores_model_selected_path(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
    )
    toolset = build_exact_create_recovery_profile(
        context,
        target_path="declared.txt",
    )
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source"}),
    )
    with IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    ) as host:
        result = host.dispatch(
            "write_source",
            {
                "content": "bound\n",
                "file_path": "adjacent.txt",
            },
        )

    assert result == "OK: wrote 1 lines to declared.txt"
    assert (tmp_path / "declared.txt").read_text(encoding="utf-8") == "bound\n"
    assert not (tmp_path / "adjacent.txt").exists()


def test_exact_create_target_must_match_typed_exact_scope(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("safe.txt",),
        write_exact_scope=("safe.txt",),
    )
    mismatched = build_exact_create_tools(
        context,
        target_path="different.txt",
    )

    with pytest.raises(
        MutationHostQualificationError,
        match="mutating tools: write_source",
    ):
        build_isolated_toolset_spec(
            mismatched,
            expected_write_names=frozenset({"write_source"}),
        )


def test_broad_implementer_profile_fails_with_every_unsupported_mutator(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("target.txt",),
    )
    toolset = build_implementer_profile(context)

    with pytest.raises(
        MutationHostQualificationError,
        match=("build_project, create_rust_crate, materialize_artifact"),
    ):
        build_isolated_toolset_spec(
            toolset,
            expected_write_names=frozenset(toolset.write_names),
        )


def test_autonomous_implementer_profile_qualifies_for_process_host(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("target.txt",),
    )
    toolset = build_autonomous_implementer_profile(context)

    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset(toolset.write_names),
    )

    assert {binding.name for binding in spec.bindings} == {
        "write_source",
        "apply_patch",
        "delete_source",
    }


def test_recursive_delete_executes_in_reapable_child(tmp_path: Path) -> None:
    donor = tmp_path / "donor"
    (donor / "nested").mkdir(parents=True)
    (donor / "nested" / "obsolete.txt").write_text("obsolete\n")

    with _host(tmp_path, scope=("donor",)) as host:
        pid = host.pid
        result = host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    assert result == (
        "OK: recursively deleted 1 files and 2 directories from donor"
    )
    assert not donor.exists()
    assert host.started_write_names == ("delete_source",)
    assert pid is not None
    _assert_process_group_gone(pid)


def test_regular_file_delete_executes_in_sealed_child(tmp_path: Path) -> None:
    target = tmp_path / "obsolete.txt"
    target.write_text("obsolete\n")

    with _host(tmp_path, scope=("obsolete.txt",)) as host:
        result = host.dispatch(
            "delete_source",
            {"file_path": "obsolete.txt", "recursive": False},
        )

    assert result == "OK: deleted file obsolete.txt"
    assert not target.exists()
    assert next(iter(_FAKE_DELETE_JOURNALS.values())).receipt is None
    assert host.is_reaped is True


def test_delete_without_durable_journal_is_denied_before_child_start(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "kept.txt").write_text("kept\n")
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("donor",),
    )
    toolset = build_write_tools(context).filter_to({
        "write_source",
        "apply_patch",
        "delete_source",
    })
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset(toolset.write_names),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )

    result = host.dispatch(
        "delete_source",
        {"file_path": "donor", "recursive": True},
    )

    assert result == (
        "Error: delete_source requires active durable "
        "mutation-journal authority"
    )
    assert host.pid is None
    assert (donor / "kept.txt").read_text() == "kept\n"
    host.stop()


def test_delete_timeout_before_publish_ack_preserves_tree(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    target = donor / "kept.txt"
    target.write_text("kept\n")
    entered = threading.Event()
    release = threading.Event()

    def pause_before_publish_ack() -> None:
        entered.set()
        release.wait(5.0)

    host = _host(
        tmp_path,
        scope=("donor",),
        deadline=0.35,
        before_publish_ack=pause_before_publish_ack,
    )
    errors: list[BaseException] = []

    def dispatch() -> None:
        try:
            host.dispatch(
                "delete_source",
                {"file_path": "donor", "recursive": True},
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2.0)
    pid = host.pid
    assert pid is not None
    time.sleep(0.4)
    release.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert any(isinstance(exc, MutationHostTimeoutError) for exc in errors)
    assert target.read_text() == "kept\n"
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_delete_timeout_after_quarantine_restores_tree(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    target = donor / "kept.txt"
    target.write_text("kept\n")
    entered = threading.Event()
    release = threading.Event()

    def pause_after_quarantine() -> None:
        entered.set()
        release.wait(5.0)

    host = _host(
        tmp_path,
        scope=("donor",),
        deadline=0.35,
        before_quarantine_ack=pause_after_quarantine,
    )
    errors: list[BaseException] = []

    def dispatch() -> None:
        try:
            host.dispatch(
                "delete_source",
                {"file_path": "donor", "recursive": True},
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2.0)
    assert not donor.exists()
    pid = host.pid
    assert pid is not None
    time.sleep(0.4)
    release.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert any(isinstance(exc, MutationHostTimeoutError) for exc in errors)
    assert target.read_text() == "kept\n"
    assert not tuple(
        tmp_path.glob(".donor.crucible-delete-*.quarantine")
    )
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_delete_sigint_after_quarantine_reaps_then_restores(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    target = donor / "kept.txt"
    target.write_text("kept\n")
    observed_pids: list[int] = []
    host: IsolatedMutationHost

    def interrupt_after_quarantine() -> None:
        pid = host.pid
        assert pid is not None
        observed_pids.append(pid)
        assert not donor.exists()
        raise KeyboardInterrupt

    host = _host(
        tmp_path,
        scope=("donor",),
        before_quarantine_ack=interrupt_after_quarantine,
    )

    with pytest.raises(KeyboardInterrupt):
        host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    assert target.read_text() == "kept\n"
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))
    assert next(iter(_FAKE_DELETE_JOURNALS.values())).receipt is None
    assert host.is_reaped is True
    assert len(observed_pids) == 1
    _assert_process_group_gone(observed_pids[0])


def test_delete_replacement_after_preflight_never_traverses_symlink(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    donor = repo / "donor"
    donor.mkdir(parents=True)
    (donor / "original.txt").write_text("original\n")
    parked = repo / "parked-original"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("external\n")

    def replace_public_target() -> None:
        donor.rename(parked)
        donor.symlink_to(external, target_is_directory=True)

    with _host(
        repo,
        scope=("donor",),
        before_publish_ack=replace_public_target,
    ) as host:
        with pytest.raises(
            MutationHostLifecycleError,
            match="delete target changed before publication ACK",
        ):
            host.dispatch(
                "delete_source",
                {"file_path": "donor", "recursive": True},
            )

    assert donor.is_symlink()
    assert donor.resolve() == external.resolve()
    assert (parked / "original.txt").read_text() == "original\n"
    assert sentinel.read_text() == "external\n"
    assert not tuple(repo.glob(".donor.crucible-delete-*.quarantine"))
    assert next(iter(_FAKE_DELETE_JOURNALS.values())).receipt is None
    assert host.is_reaped is True


@pytest.mark.parametrize("barrier", ["publish", "quarantine"])
@pytest.mark.parametrize("rebind_level", ["root", "nested"])
def test_delete_ack_barrier_rebind_restores_only_retained_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    barrier: str,
    rebind_level: str,
) -> None:
    repo = tmp_path / "repo"
    target_relative = (
        "donor" if rebind_level == "root" else "nested/parent/donor"
    )
    original_target = repo / target_relative
    original_target.mkdir(parents=True)
    (original_target / "original.txt").write_text(
        "original\n",
        encoding="utf-8",
    )
    if rebind_level == "root":
        rebind_source = repo
        parked = tmp_path / "parked-repo"
        retained_target = parked / "donor"
        retained_parent = parked
    else:
        rebind_source = repo / "nested"
        parked = repo / "parked-nested"
        retained_target = parked / "parent" / "donor"
        retained_parent = parked / "parent"
    replacement_target = repo / target_relative
    observed_pids: list[int] = []
    recovery_reap_checks: list[int] = []
    host: IsolatedMutationHost

    def rebind() -> None:
        pid = host.pid
        assert pid is not None
        observed_pids.append(pid)
        quarantines = tuple(
            original_target.parent.glob(
                ".donor.crucible-delete-*.quarantine"
            )
        )
        assert len(quarantines) == 1
        if barrier == "publish":
            assert original_target.is_dir()
            assert tuple(quarantines[0].iterdir()) == ()
        else:
            assert not original_target.exists()
            assert (
                quarantines[0] / "payload" / "original.txt"
            ).read_text(encoding="utf-8") == "original\n"
        rebind_source.rename(parked)
        replacement_target.mkdir(parents=True)
        (replacement_target / "victim.txt").write_text(
            "replacement\n",
            encoding="utf-8",
        )

    real_recover = IsolatedMutationHost._recover_delete_quarantine_locked

    def prove_reaped_before_recovery(self, pending) -> None:
        assert observed_pids
        _assert_process_group_gone(observed_pids[-1])
        recovery_reap_checks.append(observed_pids[-1])
        real_recover(self, pending)

    monkeypatch.setattr(
        IsolatedMutationHost,
        "_recover_delete_quarantine_locked",
        prove_reaped_before_recovery,
    )
    hook = (
        {"before_publish_ack": rebind}
        if barrier == "publish"
        else {"before_quarantine_ack": rebind}
    )
    host = _host(
        repo,
        scope=(target_relative,),
        **hook,
    )

    with pytest.raises(
        (MutationHostQualificationError, MutationHostLifecycleError),
    ):
        host.dispatch(
            "delete_source",
            {"file_path": target_relative, "recursive": True},
        )

    assert observed_pids and recovery_reap_checks == observed_pids
    assert (replacement_target / "victim.txt").read_text(
        encoding="utf-8"
    ) == "replacement\n"
    assert (retained_target / "original.txt").read_text(
        encoding="utf-8"
    ) == "original\n"
    assert not tuple(
        retained_parent.glob(".donor.crucible-delete-*.quarantine")
    )
    assert next(iter(_FAKE_DELETE_JOURNALS.values())).receipt is None
    assert host.is_reaped is True


def test_delete_parent_chain_open_effect_error_remains_unsettled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    donor = repo / "nested" / "donor"
    donor.mkdir(parents=True)
    (donor / "original.txt").write_text("original\n", encoding="utf-8")
    host = _host(repo, scope=("nested/donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    real_open_chain = host_module._open_delete_parent_chain
    leaked: list[int] = []
    attempts = 0

    def effect_then_error(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        descriptor, _chain = real_open_chain(*args, **kwargs)
        leaked.append(descriptor)
        raise OSError(errno.EIO, "injected after parent-chain open")

    monkeypatch.setattr(
        host_module,
        "_open_delete_parent_chain",
        effect_then_error,
    )

    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "delete_source",
            {"file_path": "nested/donor", "recursive": True},
        )

    assert attempts == 1
    assert len(leaked) == 1
    os.fstat(leaked[0])
    assert (donor / "original.txt").read_text(encoding="utf-8") == (
        "original\n"
    )
    assert next(iter(_FAKE_DELETE_JOURNALS.values())).receipt is None
    assert host.is_reaped is False
    _assert_process_group_gone(pid)

    # Test-only resolution of the deliberately unknowable acquisition lets
    # the remaining ordinary host resources close without leaking a process.
    monkeypatch.setattr(
        host_module,
        "_open_delete_parent_chain",
        real_open_chain,
    )
    os.close(leaked[0])
    with host._state_lock:
        pending, authority = next(iter(host._delete_receipts.items()))
        authority.parent_acquisition_uncertain = False
        authority.parent_acquisition_in_progress = False
        host._pending_staging.discard(pending)
        del host._delete_receipts[pending]
    host.stop()
    assert host.is_reaped is True


def test_delete_parent_chain_close_effect_error_never_retries_reused_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    donor = repo / "nested" / "donor"
    donor.mkdir(parents=True)
    (donor / "original.txt").write_text("original\n", encoding="utf-8")
    sentinel = tmp_path / "sentinel-parent-chain"
    sentinel.mkdir()
    host = _host(repo, scope=("nested/donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    real_dup = host_module.os.dup
    real_close = host_module.os.close
    duplicated_root: int | None = None
    reused_descriptor: int | None = None
    close_attempts = 0

    def capture_dup(descriptor: int) -> int:
        nonlocal duplicated_root
        opened = real_dup(descriptor)
        if duplicated_root is None:
            duplicated_root = opened
        return opened

    def close_then_error(descriptor: int) -> None:
        nonlocal close_attempts, reused_descriptor
        if descriptor == duplicated_root:
            close_attempts += 1
            if close_attempts > 1:
                pytest.fail("detached parent-chain fd was retried")
            real_close(descriptor)
            reused_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            assert reused_descriptor == descriptor
            raise OSError(errno.EIO, "injected after chain-parent close")
        real_close(descriptor)

    monkeypatch.setattr(host_module.os, "dup", capture_dup)
    monkeypatch.setattr(host_module.os, "close", close_then_error)
    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "delete_source",
            {"file_path": "nested/donor", "recursive": True},
        )

    assert close_attempts == 1
    assert reused_descriptor == duplicated_root
    assert reused_descriptor is not None
    os.fstat(reused_descriptor)
    assert (donor / "original.txt").read_text(encoding="utf-8") == (
        "original\n"
    )
    assert host.is_reaped is False
    _assert_process_group_gone(pid)

    monkeypatch.setattr(host_module.os, "dup", real_dup)
    monkeypatch.setattr(host_module.os, "close", real_close)
    real_close(reused_descriptor)
    with host._state_lock:
        pending, authority = next(iter(host._delete_receipts.items()))
        authority.parent_acquisition_uncertain = False
        authority.parent_acquisition_in_progress = False
        host._pending_staging.discard(pending)
        del host._delete_receipts[pending]
    host.stop()
    assert host.is_reaped is True


def test_delete_qdir_close_effect_error_never_retries_reused_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "obsolete.txt").write_text("obsolete\n", encoding="utf-8")
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    host: IsolatedMutationHost
    target_descriptor: int | None = None
    reused_descriptor: int | None = None
    close_attempts = 0

    def capture_qdir() -> None:
        nonlocal target_descriptor
        with host._state_lock:
            authority = next(iter(host._delete_receipts.values()))
            target_descriptor = authority.quarantine_descriptor
        assert target_descriptor is not None

    host = _host(
        tmp_path,
        scope=("donor",),
        before_quarantine_ack=capture_qdir,
    )
    host.start()
    pid = host.pid
    assert pid is not None
    real_close = host_module.os.close

    def close_then_error(descriptor: int) -> None:
        nonlocal close_attempts, reused_descriptor
        if descriptor == target_descriptor:
            close_attempts += 1
            if close_attempts > 1:
                pytest.fail("detached qdir descriptor was retried")
            real_close(descriptor)
            reused_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            assert reused_descriptor == descriptor
            raise OSError(errno.EIO, "injected after qdir close effect")
        real_close(descriptor)

    monkeypatch.setattr(host_module.os, "close", close_then_error)
    with pytest.raises(
        MutationHostLifecycleError,
        match="descriptor lifecycle remains uncertain",
    ):
        host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    assert close_attempts == 1
    assert reused_descriptor == target_descriptor
    assert reused_descriptor is not None
    os.fstat(reused_descriptor)
    assert not donor.exists()
    assert next(iter(_FAKE_DELETE_JOURNALS.values())).receipt is None
    assert host.is_reaped is False
    _assert_process_group_gone(pid)
    with pytest.raises(
        MutationHostLifecycleError,
        match="descriptor lifecycle remains uncertain",
    ):
        host.stop()
    assert close_attempts == 1
    os.fstat(reused_descriptor)

    monkeypatch.setattr(host_module.os, "close", real_close)
    real_close(reused_descriptor)
    with host._state_lock:
        authority = next(iter(host._delete_receipts.values()))
        authority.descriptor_settlement_uncertain = False
    host.stop()
    assert host.is_reaped is True


def test_stop_serializes_with_inflight_delete_receipt_arm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "kept.txt").write_text("kept\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    journal = next(iter(_FAKE_DELETE_JOURNALS.values()))
    real_arm = journal.arm_pending_delete_quarantine
    arm_entered = threading.Event()
    release_arm = threading.Event()
    stop_entered = threading.Event()
    stop_returned = threading.Event()
    dispatch_errors: list[BaseException] = []
    stop_errors: list[BaseException] = []

    def blocked_arm(**kwargs):
        arm_entered.set()
        assert release_arm.wait(3.0)
        return real_arm(**kwargs)

    monkeypatch.setattr(journal, "arm_pending_delete_quarantine", blocked_arm)

    def dispatch() -> None:
        try:
            host.dispatch(
                "delete_source",
                {"file_path": "donor", "recursive": True},
            )
        except BaseException as exc:
            dispatch_errors.append(exc)

    def stop() -> None:
        stop_entered.set()
        try:
            host.stop()
        except BaseException as exc:
            stop_errors.append(exc)
        finally:
            stop_returned.set()

    dispatch_thread = threading.Thread(target=dispatch)
    stop_thread = threading.Thread(target=stop)
    dispatch_thread.start()
    assert arm_entered.wait(2.0)
    stop_thread.start()
    assert stop_entered.wait(1.0)
    assert not stop_returned.wait(0.1)
    release_arm.set()
    dispatch_thread.join(timeout=4.0)
    stop_thread.join(timeout=4.0)

    assert not dispatch_thread.is_alive()
    assert not stop_thread.is_alive()
    assert not stop_errors
    assert dispatch_errors
    assert journal.receipt is None
    assert (donor / "kept.txt").read_text() == "kept\n"
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_stop_serializes_with_inflight_delete_quarantine_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "kept.txt").write_text("kept\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    journal = next(iter(_FAKE_DELETE_JOURNALS.values()))
    real_prepare = host._prepare_delete_quarantine_locked  # noqa: SLF001
    prepare_entered = threading.Event()
    release_prepare = threading.Event()
    stop_entered = threading.Event()
    stop_returned = threading.Event()
    dispatch_errors: list[BaseException] = []
    stop_errors: list[BaseException] = []

    def blocked_prepare(pending) -> None:
        prepare_entered.set()
        assert release_prepare.wait(3.0)
        real_prepare(pending)

    monkeypatch.setattr(
        host,
        "_prepare_delete_quarantine_locked",
        blocked_prepare,
    )

    def dispatch() -> None:
        try:
            host.dispatch(
                "delete_source",
                {"file_path": "donor", "recursive": True},
            )
        except BaseException as exc:
            dispatch_errors.append(exc)

    def stop() -> None:
        stop_entered.set()
        try:
            host.stop()
        except BaseException as exc:
            stop_errors.append(exc)
        finally:
            stop_returned.set()

    dispatch_thread = threading.Thread(target=dispatch)
    stop_thread = threading.Thread(target=stop)
    dispatch_thread.start()
    assert prepare_entered.wait(2.0)
    stop_thread.start()
    assert stop_entered.wait(1.0)
    assert not stop_returned.wait(0.1)
    release_prepare.set()
    dispatch_thread.join(timeout=4.0)
    stop_thread.join(timeout=4.0)

    assert not dispatch_thread.is_alive()
    assert not stop_thread.is_alive()
    assert not stop_errors
    assert dispatch_errors
    assert journal.receipt is None
    assert (donor / "kept.txt").read_text() == "kept\n"
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_post_cas_bind_interrupt_reconciles_before_child_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "kept.txt").write_text("kept\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    journal = next(iter(_FAKE_DELETE_JOURNALS.values()))
    real_bind = journal.bind_pending_delete_quarantine
    bind_calls = 0

    def interrupt_after_bind(*args, **kwargs):
        nonlocal bind_calls
        bind_calls += 1
        result = real_bind(*args, **kwargs)
        if bind_calls == 1:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(
        journal,
        "bind_pending_delete_quarantine",
        interrupt_after_bind,
    )

    with pytest.raises(KeyboardInterrupt):
        host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    assert bind_calls == 2
    assert journal.receipt is None
    assert (donor / "kept.txt").read_text() == "kept\n"
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_post_cas_clear_interrupt_finishes_local_receipt_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "obsolete.txt").write_text("obsolete\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    journal = next(iter(_FAKE_DELETE_JOURNALS.values()))
    real_clear = journal.clear_pending_delete_quarantine
    clear_calls = 0

    def interrupt_after_clear(*args, **kwargs):
        nonlocal clear_calls
        clear_calls += 1
        result = real_clear(*args, **kwargs)
        if clear_calls == 1:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(
        journal,
        "clear_pending_delete_quarantine",
        interrupt_after_clear,
    )

    with pytest.raises(KeyboardInterrupt):
        host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    assert clear_calls == 2
    assert journal.receipt is None
    assert not donor.exists()
    assert not tuple(tmp_path.glob(".donor.crucible-delete-*.quarantine"))
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_persistent_bind_failure_reaps_child_and_retains_recovery_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "kept.txt").write_text("kept\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    journal = next(iter(_FAKE_DELETE_JOURNALS.values()))
    real_bind = journal.bind_pending_delete_quarantine

    def fail_bind(*_args, **_kwargs):
        raise RuntimeError("persistent bind storage failure")

    monkeypatch.setattr(
        journal,
        "bind_pending_delete_quarantine",
        fail_bind,
    )

    with pytest.raises(
        MutationHostLifecycleError,
        match="indeterminate after child death",
    ):
        host.dispatch(
            "delete_source",
            {"file_path": "donor", "recursive": True},
        )

    _assert_process_group_gone(pid)
    # The child is positively reaped, but the hardened host intentionally
    # reports incomplete settlement while exact receipt/root authority is
    # retained for a later recovery retry.
    assert host.is_reaped is False
    with host._state_lock:  # noqa: SLF001
        assert host._pending_staging  # noqa: SLF001
        assert host._delete_receipts  # noqa: SLF001
        assert host._repository_roots  # noqa: SLF001
        pending = next(iter(host._pending_staging))  # noqa: SLF001
        root = next(iter(host._repository_roots.values()))  # noqa: SLF001
        os.fstat(root.descriptor)
    assert pending.path.is_dir()
    assert not tuple(pending.path.iterdir())
    assert (donor / "kept.txt").read_text() == "kept\n"

    # The real recovery path is startup journal recovery.  Restore the narrow
    # fake's bind operation only to release test-owned descriptors/qdir.
    monkeypatch.setattr(
        journal,
        "bind_pending_delete_quarantine",
        real_bind,
    )
    host.stop()
    assert journal.receipt is None
    assert not pending.path.exists()


def test_external_stop_cannot_deadlock_post_clear_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "obsolete.txt").write_text("obsolete\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    pid = host.pid
    assert pid is not None
    journal = next(iter(_FAKE_DELETE_JOURNALS.values()))
    real_clear = journal.clear_pending_delete_quarantine
    clear_published = threading.Event()
    release_clear = threading.Event()
    stop_lock_held = threading.Event()
    dispatch_errors: list[BaseException] = []
    stop_errors: list[BaseException] = []
    clear_calls = 0

    def interrupt_after_clear(*args, **kwargs):
        nonlocal clear_calls
        clear_calls += 1
        result = real_clear(*args, **kwargs)
        if clear_calls == 1:
            clear_published.set()
            assert release_clear.wait(3.0)
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(
        journal,
        "clear_pending_delete_quarantine",
        interrupt_after_clear,
    )

    def dispatch() -> None:
        try:
            host.dispatch(
                "delete_source",
                {"file_path": "donor", "recursive": True},
            )
        except BaseException as exc:
            dispatch_errors.append(exc)

    def external_stop() -> None:
        try:
            # Prove the external path owns stop_lock while dispatch still owns
            # delete_lifecycle_lock. Correct code releases the latter before
            # attempting stop(), so the two paths cannot form AB/BA deadlock.
            with host._stop_lock:  # noqa: SLF001
                stop_lock_held.set()
                host.stop()
        except BaseException as exc:
            stop_errors.append(exc)

    dispatch_thread = threading.Thread(target=dispatch)
    stop_thread = threading.Thread(target=external_stop)
    dispatch_thread.start()
    assert clear_published.wait(2.0)
    stop_thread.start()
    assert stop_lock_held.wait(1.0)
    release_clear.set()
    dispatch_thread.join(timeout=4.0)
    stop_thread.join(timeout=4.0)

    assert not dispatch_thread.is_alive()
    assert not stop_thread.is_alive()
    assert not stop_errors
    assert len(dispatch_errors) == 1
    assert isinstance(dispatch_errors[0], KeyboardInterrupt)
    assert clear_calls == 2
    assert journal.receipt is None
    assert not donor.exists()
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_delete_recovery_uses_retained_parent_under_symlink_redirect(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    donor = repo / "nested" / "parent" / "donor"
    donor.mkdir(parents=True)
    (donor / "original.txt").write_text("original\n")
    host = _host(repo, scope=("nested/parent/donor",))
    host.start()
    binding = host._spec.binding_for("delete_source")
    pending = host._staging_path(
        binding,
        {"file_path": "nested/parent/donor"},
        "a" * 32,
    )
    assert pending is not None
    receipt_authority = host._arm_delete_staging(
        pending,
        binding,
        "a" * 32,
    )
    with host._state_lock:
        host._pending_staging.add(pending)
        host._delete_receipts[pending] = receipt_authority
    host._prepare_delete_quarantine(pending)
    quarantine = pending.path
    donor.rename(quarantine / "payload")

    parked = repo / "parked-nested"
    (repo / "nested").rename(parked)
    external = tmp_path / "external"
    external_quarantine = external / "parent" / quarantine.name
    external_quarantine.mkdir(parents=True, mode=0o700)
    external_sentinel = external_quarantine / "sentinel.txt"
    external_sentinel.write_text("external\n")
    (repo / "nested").symlink_to(external, target_is_directory=True)

    host._recover_delete_quarantine(pending)

    assert external_sentinel.read_text() == "external\n"
    retained = parked / "parent" / "donor"
    assert (retained / "original.txt").read_text() == "original\n"
    assert not (parked / "parent" / quarantine.name).exists()
    (repo / "nested").unlink()
    parked.rename(repo / "nested")
    host.stop()


def test_delete_recovery_rejects_bound_quarantine_substitution(
    tmp_path: Path,
) -> None:
    donor = tmp_path / "donor"
    donor.mkdir()
    (donor / "original.txt").write_text("original\n")
    host = _host(tmp_path, scope=("donor",))
    host.start()
    binding = host._spec.binding_for("delete_source")
    pending = host._staging_path(
        binding,
        {"file_path": "donor"},
        "b" * 32,
    )
    assert pending is not None
    receipt_authority = host._arm_delete_staging(
        pending,
        binding,
        "b" * 32,
    )
    with host._state_lock:
        host._pending_staging.add(pending)
        host._delete_receipts[pending] = receipt_authority
    host._prepare_delete_quarantine(pending)

    original_quarantine = pending.path.with_name(
        pending.path.name + ".parked"
    )
    pending.path.rename(original_quarantine)
    pending.path.mkdir(mode=0o700)
    replacement = pending.path / "replacement.txt"
    replacement.write_text("replacement\n")

    with pytest.raises(
        MutationHostLifecycleError,
        match="binding could not be proved",
    ):
        host._recover_delete_quarantine(pending)

    assert replacement.read_text() == "replacement\n"
    assert (donor / "original.txt").read_text() == "original\n"
    with host._state_lock:
        assert pending in host._pending_staging
        assert pending in host._delete_receipts

    replacement.unlink()
    pending.path.rmdir()
    original_quarantine.rename(pending.path)
    host.stop()


def test_progress_bounded_profile_routes_write_through_child_and_resets(
    tmp_path: Path,
) -> None:
    inner = build_autonomous_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.txt",),
        )
    )
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=2,
    )
    spec = build_isolated_toolset_spec(
        bounded,
        expected_write_names=frozenset(bounded.write_names),
    )

    with IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    ) as host:
        hosted = ProcessHostedToolSet(bounded, host)
        hosted.dispatch("grep_symbol", {"pattern": "absent-one"})
        hosted.dispatch("grep_symbol", {"pattern": "absent-two"})
        assert bounded.snapshot().consecutive_read_calls == 2
        result = hosted.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "isolated progress\n",
            },
        )
        assert host.ever_started_write is True

    assert result == "OK: wrote 1 lines to target.txt"
    assert (tmp_path / "target.txt").read_text() == "isolated progress\n"
    assert bounded.snapshot().write_calls == 1
    assert bounded.snapshot().consecutive_read_calls == 0


def test_progress_bounded_profile_does_not_credit_rejected_child_write(
    tmp_path: Path,
) -> None:
    inner = build_autonomous_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.txt",),
        )
    )
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=1,
    )
    spec = build_isolated_toolset_spec(
        bounded,
        expected_write_names=frozenset(bounded.write_names),
    )

    with IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    ) as host:
        hosted = ProcessHostedToolSet(bounded, host)
        hosted.dispatch("grep_symbol", {"pattern": "absent"})
        result = hosted.dispatch(
            "write_source",
            {
                "file_path": "outside.txt",
                "content": "not authorized\n",
            },
        )
        with pytest.raises(ToolDispatchBudgetExceeded):
            hosted.dispatch("grep_symbol", {"pattern": "still-absent"})

    assert result.startswith("Error")
    assert bounded.snapshot().write_calls == 0
    assert bounded.snapshot().consecutive_read_calls == 1


def test_progress_bounded_profile_does_not_credit_identical_child_write(
    tmp_path: Path,
) -> None:
    original = "unchanged\n"
    (tmp_path / "target.txt").write_text(original)
    inner = build_autonomous_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.txt",),
        )
    )
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=1,
    )
    spec = build_isolated_toolset_spec(
        bounded,
        expected_write_names=frozenset(bounded.write_names),
    )

    with IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    ) as host:
        hosted = ProcessHostedToolSet(bounded, host)
        hosted.dispatch("grep_symbol", {"pattern": "absent"})
        result = hosted.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": original,
                "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
            },
        )
        with pytest.raises(ToolDispatchBudgetExceeded):
            hosted.dispatch("grep_symbol", {"pattern": "still-absent"})

    assert result.startswith("Error: replacement")
    assert (tmp_path / "target.txt").read_text() == original
    assert bounded.snapshot().write_calls == 0
    assert bounded.snapshot().consecutive_read_calls == 1


def test_custom_mutating_toolset_fails_preflight_even_with_canonical_name(
    tmp_path: Path,
) -> None:
    del tmp_path

    class Custom:
        def __init__(self) -> None:
            self.write_names = {"write_source"}
            self.read_only_names: set[str] = set()

        def to_anthropic_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "name": "write_source",
                    "description": "custom",
                    "input_schema": {"type": "object"},
                }
            ]

        def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
            raise AssertionError((name, arguments))

    with pytest.raises(
        MutationHostQualificationError,
        match="mutating tools: write_source",
    ):
        build_isolated_toolset_spec(
            Custom(),
            expected_write_names=frozenset({"write_source"}),
        )


def test_timeout_after_fsynced_stage_reaps_and_never_publishes_late(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def pause_before_publish_ack() -> None:
        entered.set()
        release.wait(5.0)

    host = _host(
        tmp_path,
        deadline=0.35,
        before_publish_ack=pause_before_publish_ack,
    )
    errors: list[BaseException] = []

    def dispatch() -> None:
        try:
            host.dispatch(
                "write_source",
                {
                    "file_path": "target.txt",
                    "content": "must-never-appear",
                },
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2.0)
    pid = host.pid
    assert pid is not None
    time.sleep(0.4)
    release.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert any(isinstance(exc, MutationHostTimeoutError) for exc in errors)
    assert host.ever_started_write is True
    assert host.is_reaped is True
    _assert_process_group_gone(pid)
    assert not (tmp_path / "target.txt").exists()
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))
    time.sleep(0.1)
    assert not (tmp_path / "target.txt").exists()


def test_destination_change_during_publication_ack_fails_stale(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("original\n", encoding="utf-8")

    def concurrent_change() -> None:
        target.write_text("concurrent\n", encoding="utf-8")

    host = _host(
        tmp_path,
        before_publish_ack=concurrent_change,
    )
    with host:
        result = host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "replacement\n",
                "expected_sha256": hashlib.sha256(b"original\n").hexdigest(),
            },
        )

    assert "stale file changed during publication" in result
    assert target.read_text(encoding="utf-8") == "concurrent\n"
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))


def test_new_destination_appearing_during_publication_is_not_clobbered(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"

    def concurrent_create() -> None:
        target.write_text("concurrent\n", encoding="utf-8")

    host = _host(
        tmp_path,
        before_publish_ack=concurrent_create,
    )
    with host:
        result = host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "replacement\n",
            },
        )

    assert "destination appeared during publication" in result
    assert target.read_text(encoding="utf-8") == "concurrent\n"
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))


def test_external_cancellation_reaps_child_before_dispatch_unblocks(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def pause_before_publish_ack() -> None:
        entered.set()
        release.wait(5.0)

    host = _host(
        tmp_path,
        before_publish_ack=pause_before_publish_ack,
    )
    errors: list[BaseException] = []

    def dispatch() -> None:
        try:
            host.dispatch(
                "write_source",
                {
                    "file_path": "target.txt",
                    "content": "must-never-appear",
                },
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2.0)
    pid = host.pid
    assert pid is not None
    host.stop()
    _assert_process_group_gone(pid)
    release.set()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert errors
    assert any(
        isinstance(exc, (MutationHostProtocolError, MutationHostLifecycleError))
        for exc in errors
    )
    assert host.is_reaped is True
    assert not (tmp_path / "target.txt").exists()
    time.sleep(0.1)
    assert not (tmp_path / "target.txt").exists()


def test_hostile_child_death_is_reaped_without_late_mutation(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def pause_before_publish_ack() -> None:
        entered.set()
        release.wait(5.0)

    host = _host(
        tmp_path,
        before_publish_ack=pause_before_publish_ack,
    )
    errors: list[BaseException] = []

    def dispatch() -> None:
        try:
            host.dispatch(
                "write_source",
                {
                    "file_path": "target.txt",
                    "content": "must-never-appear",
                },
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert entered.wait(2.0)
    pid = host.pid
    assert pid is not None
    os.killpg(pid, signal.SIGKILL)
    release.set()
    thread.join(timeout=3.0)
    host.stop()

    assert not thread.is_alive()
    assert errors
    assert host.is_reaped is True
    _assert_process_group_gone(pid)
    assert not (tmp_path / "target.txt").exists()
    time.sleep(0.1)
    assert not (tmp_path / "target.txt").exists()


def test_failed_settlement_is_not_forgotten_and_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    host.start()
    pid = host.pid
    assert pid is not None
    real_settle = host._settle_process  # noqa: SLF001

    def fail_settlement(*_args, **_kwargs) -> None:
        raise MutationHostLifecycleError("injected settlement failure")

    monkeypatch.setattr(host, "_settle_process", fail_settlement)
    with pytest.raises(
        MutationHostLifecycleError,
        match="injected settlement failure",
    ):
        host.stop()

    assert host.pid == pid
    assert IsolatedMutationHost._process_group_exists(pid) is True

    monkeypatch.setattr(host, "_settle_process", real_settle)
    host.stop()
    assert host.pid is None
    _assert_process_group_gone(pid)


def test_sigint_during_wait_is_replayed_only_after_host_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A main-thread cancellation cannot strand killed child authority."""

    host = _host(tmp_path)
    host.start()
    pid = host.pid
    process = host._process  # noqa: SLF001
    assert pid is not None
    assert process is not None
    real_wait = process.wait
    wait_calls = 0

    def interrupt_first_wait(timeout=None):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            raise KeyboardInterrupt
        return real_wait(timeout=timeout)

    monkeypatch.setattr(process, "wait", interrupt_first_wait)

    with pytest.raises(KeyboardInterrupt):
        host.stop()

    assert wait_calls >= 2
    assert host.pid is None
    assert host.is_reaped is True
    _assert_process_group_gone(pid)


def test_repeated_sigint_retains_host_for_later_settlement_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A deterministic interrupt cannot create an unbounded cleanup loop."""

    host = _host(tmp_path)
    host.start()
    pid = host.pid
    process = host._process  # noqa: SLF001
    assert pid is not None
    assert process is not None
    real_wait = process.wait
    original_interrupt = KeyboardInterrupt()

    def always_interrupt(timeout=None):
        del timeout
        raise original_interrupt

    monkeypatch.setattr(process, "wait", always_interrupt)
    with pytest.raises(
        MutationHostLifecycleError,
        match="repeated SIGINT",
    ):
        host.stop()

    assert host.pid == pid
    monkeypatch.setattr(process, "wait", real_wait)
    with pytest.raises(KeyboardInterrupt) as replayed:
        host.stop()

    assert replayed.value is original_interrupt
    assert host.pid is None
    _assert_process_group_gone(pid)


def test_waitpid_eintr_is_retried_without_claiming_process_death(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A returning signal handler's EINTR is not settlement evidence."""

    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    host.start()
    pid = host.pid
    assert pid is not None
    real_waitpid = host_mod.os.waitpid
    interrupted = False

    def interrupt_once(wait_pid: int, options: int):
        nonlocal interrupted
        if wait_pid == pid and not interrupted:
            interrupted = True
            raise InterruptedError
        return real_waitpid(wait_pid, options)

    monkeypatch.setattr(host_mod.os, "waitpid", interrupt_once)
    host.stop()

    assert interrupted
    assert host.pid is None
    _assert_process_group_gone(pid)


def test_failed_descriptor_relative_root_cleanup_retains_exact_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    host.start()
    pid = host.pid
    private_root = host._private_root  # noqa: SLF001
    real_cleanup = host_mod._close_private_root_authority
    assert pid is not None
    assert private_root is not None

    def fail_cleanup(_authority) -> None:
        raise MutationHostLifecycleError("injected descriptor cleanup failure")

    monkeypatch.setattr(host_mod, "_close_private_root_authority", fail_cleanup)
    with pytest.raises(MutationHostLifecycleError, match="resource cleanup"):
        host.stop()

    assert host.pid == pid
    assert host.is_reaped is False
    assert host._private_root == private_root  # noqa: SLF001
    assert private_root.is_dir()

    monkeypatch.setattr(host_mod, "_close_private_root_authority", real_cleanup)
    host.stop()
    assert host.pid is None
    assert host.is_reaped is True
    assert private_root.exists() is False


def test_post_effect_root_open_never_closes_reused_numeric_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_open = host_mod.os.open
    real_close = host_mod.os.close
    target: list[int] = []
    sentinels: list[int] = []

    def injected_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if not target and Path(path) == tmp_path:
            target.append(descriptor)
            real_close(descriptor)
            while descriptor not in sentinels:
                sentinels.append(real_open("/dev/null", os.O_RDONLY | os.O_CLOEXEC))
            raise OSError(errno.EINTR, "injected post-effect root open error")
        return descriptor

    monkeypatch.setattr(host_mod.os, "open", injected_open)
    with pytest.raises(MutationHostLifecycleError, match="startup cleanup"):
        host.start()
    assert len(target) == 1
    assert host.is_reaped is False
    os.fstat(target[0])
    with pytest.raises(MutationHostLifecycleError, match="resource cleanup"):
        host.stop()
    os.fstat(target[0])
    monkeypatch.setattr(host_mod.os, "open", real_open)
    for descriptor in sentinels:
        real_close(descriptor)


def test_post_effect_stderr_open_is_sticky_and_never_closes_reused_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_open = host_mod.os.open
    real_close = host_mod.os.close
    target: list[int] = []
    sentinels: list[int] = []

    def effect_then_error(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if not target and os.fspath(path) == ".child-stderr":
            target.append(descriptor)
            real_close(descriptor)
            while descriptor not in sentinels:
                sentinels.append(real_open("/dev/null", os.O_RDONLY | os.O_CLOEXEC))
            raise OSError(errno.EINTR, "injected post-effect stderr open error")
        return descriptor

    monkeypatch.setattr(host_mod.os, "open", effect_then_error)
    with pytest.raises(MutationHostLifecycleError, match="startup cleanup"):
        host.start()
    assert len(target) == 1
    assert host.is_reaped is False
    os.fstat(target[0])
    with pytest.raises(MutationHostLifecycleError, match="resource cleanup"):
        host.stop()
    os.fstat(target[0])
    monkeypatch.setattr(host_mod.os, "open", real_open)
    for descriptor in sentinels:
        real_close(descriptor)


def test_post_effect_pipe_acquisition_is_published_and_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_call = host_mod._call_libc_pipe2
    acquired: list[int] = []

    def effect_then_error(cell) -> int:
        result = real_call(cell)
        acquired.extend((int(cell[0]), int(cell[1])))
        raise KeyboardInterrupt("injected after pipe2 effect")

    monkeypatch.setattr(host_mod, "_call_libc_pipe2", effect_then_error)
    with pytest.raises(KeyboardInterrupt, match="after pipe2 effect"):
        host.start()
    assert len(acquired) == 2
    assert host.is_reaped is True
    for descriptor in acquired:
        _assert_descriptor_closed(descriptor)


@pytest.mark.parametrize(
    "target_attribute",
    ["_request_read_descriptor", "_response_write_descriptor"],
)
def test_post_effect_pipe_close_error_never_retries_reused_integer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_attribute: str,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_prepare = host_mod._PosixSpawnProcess.prepare_parent_pipes
    real_close = host_mod.os.close
    target: list[int] = []
    sentinel: list[int] = []
    process_refs = []

    def prepare(process) -> None:
        descriptor = getattr(process, target_attribute)
        assert descriptor is not None
        target.append(descriptor)
        process_refs.append(process)
        real_prepare(process)

    def late_close_error(descriptor: int) -> None:
        if target and not sentinel and descriptor == target[0]:
            real_close(descriptor)
            while descriptor not in sentinel:
                opened = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
                sentinel.append(opened)
            raise OSError(errno.EINTR, "injected post-effect close error")
        real_close(descriptor)

    monkeypatch.setattr(host_mod._PosixSpawnProcess, "prepare_parent_pipes", prepare)
    monkeypatch.setattr(host_mod.os, "close", late_close_error)
    with pytest.raises(Exception):
        host.start()

    assert len(target) == 1
    assert getattr(process_refs[0], target_attribute) is None
    os.fstat(target[0])
    monkeypatch.setattr(host_mod.os, "close", real_close)
    host.stop()
    os.fstat(target[0])
    for descriptor in sentinel:
        real_close(descriptor)


def test_host_shutdown_does_not_close_shared_sealed_source_image(
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    image = host_mod._LOADED_SOURCE_IMAGE
    assert image is not None
    identity = os.fstat(image.descriptor)
    host = _host(tmp_path)
    host.start()
    host.stop()
    observed = os.fstat(image.descriptor)
    assert (observed.st_dev, observed.st_ino) == (identity.st_dev, identity.st_ino)


def test_parent_ipc_never_transfers_raw_descriptors_through_fdopen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    monkeypatch.setattr(
        host_mod.os,
        "fdopen",
        lambda *_args, **_kwargs: pytest.fail("raw IPC must not call fdopen"),
    )
    with _host(tmp_path) as host:
        assert "OK: wrote" in host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "raw pipes\n"},
        )


def test_pid_authority_is_published_at_live_spawn_result_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_call = host_mod._call_libc_posix_spawn
    spawned = threading.Event()
    release = threading.Event()
    child_pids: list[int] = []
    errors: list[BaseException] = []

    def paused_call(*args, **kwargs) -> int:
        pid = real_call(*args, **kwargs)
        child_pids.append(pid)
        spawned.set()
        assert release.wait(3.0)
        return pid

    monkeypatch.setattr(host_mod, "_call_libc_posix_spawn", paused_call)

    def start() -> None:
        try:
            host.start()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=start)
    thread.start()
    assert spawned.wait(3.0)
    try:
        assert host._process is not None  # noqa: SLF001
        assert host.pid == child_pids[0]
        assert host.is_reaped is False
    finally:
        release.set()
        thread.join(timeout=5.0)
        host.stop()
    assert not thread.is_alive()
    assert not errors
    assert host.is_reaped is True
    assert len(child_pids) == 1
    _assert_process_group_gone(child_pids[0])


def test_post_spawn_effect_error_reaps_result_cell_pid_without_zombie(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_call = host_mod._call_libc_posix_spawn
    spawned: list[int] = []

    def effect_then_error(*args, **kwargs) -> int:
        pid = real_call(*args, **kwargs)
        spawned.append(pid)
        raise KeyboardInterrupt("injected after posix_spawn effect")

    monkeypatch.setattr(host_mod, "_call_libc_posix_spawn", effect_then_error)
    with pytest.raises(KeyboardInterrupt, match="after posix_spawn effect"):
        host.start()
    assert len(spawned) == 1
    assert host.pid is None
    assert host.is_reaped is True
    _assert_process_group_gone(spawned[0])
    with pytest.raises(ChildProcessError):
        os.waitpid(spawned[0], os.WNOHANG)


def test_atomic_closefrom_excludes_fd_created_at_spawn_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_call = host_mod._call_libc_posix_spawn
    raced_descriptors: list[int] = []

    def racing_call(*args, **kwargs) -> int:
        descriptor = os.open("/dev/null", os.O_RDONLY)
        os.set_inheritable(descriptor, True)
        raced_descriptors.append(descriptor)
        return real_call(*args, **kwargs)

    monkeypatch.setattr(host_mod, "_call_libc_posix_spawn", racing_call)
    try:
        host.start()
        assert len(raced_descriptors) == 1
        descriptor = raced_descriptors[0]
        try:
            child_target = os.readlink(f"/proc/{host.pid}/fd/{descriptor}")
        except FileNotFoundError:
            child_target = ""
        assert child_target != "/dev/null"
    finally:
        host.stop()
        for descriptor in raced_descriptors:
            os.close(descriptor)


def test_repo_root_rename_rebind_cannot_redirect_mutation(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    retained = tmp_path / "retained-repo"
    host = _host(repo)
    host.start()
    repo.rename(retained)
    repo.mkdir()
    victim = repo / "target.txt"
    victim.write_text("must survive\n", encoding="utf-8")
    try:
        with pytest.raises(
            MutationHostQualificationError,
            match="mutation-root path was rebound",
        ):
            host.dispatch(
                "write_source",
                {"file_path": "target.txt", "content": "redirected\n"},
            )
        assert victim.read_text(encoding="utf-8") == "must survive\n"
        assert not (retained / "target.txt").exists()
    finally:
        host.stop()


def test_repo_root_rebind_before_descriptor_open_is_rejected(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    retained = tmp_path / "retained-repo"
    host = _host(repo)
    repo.rename(retained)
    repo.mkdir()
    victim = repo / "target.txt"
    victim.write_text("must survive\n", encoding="utf-8")
    with pytest.raises(
        MutationHostQualificationError,
        match="changed before descriptor binding",
    ):
        host.start()
    assert host.is_reaped is True
    assert victim.read_text(encoding="utf-8") == "must survive\n"
    assert not (retained / "target.txt").exists()


def test_repo_root_rebind_at_publication_barrier_cannot_redirect_write(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    retained = tmp_path / "retained-repo"

    def rebind() -> None:
        repo.rename(retained)
        repo.mkdir()
        (repo / "target.txt").write_text("victim\n", encoding="utf-8")

    host = _host(repo, before_publish_ack=rebind)
    with pytest.raises(
        MutationHostQualificationError,
        match="mutation-root path was rebound",
    ):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "must not publish\n"},
        )
    assert host.is_reaped is True
    assert (repo / "target.txt").read_text(encoding="utf-8") == "victim\n"
    assert not (retained / "target.txt").exists()
    assert not tuple(retained.glob(".*.crucible-*.tmp"))


def test_private_root_creation_bypasses_python_mkdir_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_mkdir = host_mod.os.mkdir
    attempted = False

    def effect_then_error(path, *args, **kwargs) -> None:
        nonlocal attempted
        if str(path).startswith(host_mod._PRIVATE_ROOT_PREFIX):
            attempted = True
            real_mkdir(path, *args, **kwargs)
            raise OSError(errno.EINTR, "injected after mkdir effect")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(host_mod.os, "mkdir", effect_then_error)
    host.start()
    host.stop()
    assert attempted is False
    assert host.is_reaped is True


def test_private_root_creation_fsync_effect_error_never_reports_startup_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_fsync = host_mod.os.fsync
    injected = False

    def creation_fsync_then_error(descriptor: int) -> None:
        nonlocal injected
        authority = host._private_root_authority  # noqa: SLF001
        pending = None if authority is None else authority.pending_creation
        if (
            not injected
            and pending is not None
            and pending.creation_unsynced
            and descriptor == pending.parent_descriptor
        ):
            injected = True
            real_fsync(descriptor)
            raise OSError(errno.EIO, "injected after private-root creation fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", creation_fsync_then_error)
    with pytest.raises(
        MutationHostLifecycleError,
        match="directory publication fsync outcome is uncertain",
    ):
        host.start()

    assert injected is True
    assert host.is_reaped is True


def test_private_root_removal_fsync_effect_error_requires_exact_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    host.start()
    private_root = host._private_root  # noqa: SLF001
    authority = host._private_root_authority  # noqa: SLF001
    assert private_root is not None
    assert authority is not None
    real_fsync = host_mod.os.fsync
    injected = False

    def removal_fsync_then_error(descriptor: int) -> None:
        nonlocal injected
        if (
            not injected
            and authority.root_removal_unsynced
            and descriptor == authority.parent_descriptor
        ):
            injected = True
            real_fsync(descriptor)
            raise OSError(errno.EIO, "injected after private-root removal fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", removal_fsync_then_error)
    with pytest.raises(MutationHostLifecycleError, match="resource cleanup"):
        host.stop()

    assert injected is True
    assert private_root.exists() is False
    assert host.is_reaped is False
    host.stop()
    assert host.is_reaped is True


def test_private_root_cleanup_tracks_owned_inode_across_path_rebind(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    host.start()
    private_root = host._private_root  # noqa: SLF001
    assert private_root is not None
    retained = private_root.with_name(private_root.name + "-retained")
    private_root.rename(retained)
    private_root.mkdir(mode=0o700)
    victim = private_root / "unrelated"
    victim.write_text("must survive\n", encoding="utf-8")
    try:
        host.stop()
        assert retained.exists() is False
        assert victim.read_text(encoding="utf-8") == "must survive\n"
        assert host.is_reaped is True
    finally:
        shutil.rmtree(retained, ignore_errors=True)
        shutil.rmtree(private_root, ignore_errors=True)


def test_private_root_moved_outside_held_parent_remains_uncertain(
    tmp_path: Path,
) -> None:
    host = _host(tmp_path)
    host.start()
    private_root = host._private_root  # noqa: SLF001
    assert private_root is not None
    outside = tmp_path / "moved-private-root"
    returned = private_root.with_name(private_root.name + "-returned")
    private_root.rename(outside)
    try:
        with pytest.raises(
            MutationHostLifecycleError,
            match="resource cleanup",
        ):
            host.stop()
        assert host.is_reaped is False
        outside.rename(returned)
        host.stop()
        assert returned.exists() is False
        assert host.is_reaped is True
    finally:
        shutil.rmtree(outside, ignore_errors=True)
        shutil.rmtree(returned, ignore_errors=True)


def test_is_reaped_requires_same_process_group_to_be_empty(tmp_path: Path) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    leader = os.fork()
    if leader == 0:
        os.setsid()
        member = os.fork()
        if member == 0:
            time.sleep(30.0)
            os._exit(0)
        os._exit(0)

    host = _host(tmp_path)
    process = host_mod._PosixSpawnProcess()
    process._spawn_pid_cell.value = leader  # noqa: SLF001
    host._process = process  # noqa: SLF001
    try:
        deadline = time.monotonic() + 3.0
        while process.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert host_mod.IsolatedMutationHost._process_group_exists(leader) is True
        assert host.is_reaped is False
    finally:
        host.stop()
    assert host_mod.IsolatedMutationHost._process_group_exists(leader) is False
    assert host.is_reaped is True


def test_process_group_inspection_oserror_is_a_typed_lifecycle_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    def fail_inspection(_pgid: int, _signal_number: int) -> None:
        raise OSError("injected process-table failure")

    monkeypatch.setattr(host_mod.os, "killpg", fail_inspection)
    with pytest.raises(
        MutationHostLifecycleError,
        match="could not inspect",
    ):
        IsolatedMutationHost._process_group_exists(12345)


def test_start_and_stop_are_serialized_across_the_spawn_attachment_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_spawn = host_mod._posix_spawn_child
    spawned = threading.Event()
    release = threading.Event()
    child_pids: list[int] = []
    errors: list[BaseException] = []

    def paused_spawn(*args, **kwargs):
        process = args[0]
        real_spawn(*args, **kwargs)
        assert process.pid is not None
        child_pids.append(process.pid)
        spawned.set()
        assert release.wait(2.0)

    monkeypatch.setattr(host_mod, "_posix_spawn_child", paused_spawn)

    def run(callable_) -> None:
        try:
            callable_()
        except BaseException as exc:
            errors.append(exc)

    start_thread = threading.Thread(target=run, args=(host.start,))
    stop_thread = threading.Thread(target=run, args=(host.stop,))
    start_thread.start()
    assert spawned.wait(2.0)
    stop_thread.start()
    time.sleep(0.05)
    stop_was_waiting = stop_thread.is_alive()
    release.set()
    start_thread.join(timeout=3.0)
    stop_thread.join(timeout=3.0)

    assert not errors
    assert stop_was_waiting
    assert not start_thread.is_alive()
    assert not stop_thread.is_alive()
    assert host.pid is None
    assert len(child_pids) == 1
    _assert_process_group_gone(child_pids[0])


def test_sigkill_before_atomic_replace_preserves_complete_old_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    marker = tmp_path / "before-replace"
    target.write_text("old-complete\n", encoding="utf-8")
    script = """
import os
import signal
import sys
from pathlib import Path
from crucible.tools.write import handlers

target = Path(sys.argv[1])
marker = Path(sys.argv[2])
real_replace = os.replace

def stop_before_replace(source, destination, *args, **kwargs):
    marker.write_text("ready", encoding="utf-8")
    os.kill(os.getpid(), signal.SIGSTOP)
    return real_replace(source, destination, *args, **kwargs)

handlers.os.replace = stop_before_replace
handlers._publish_text_atomic(
    target,
    "new-complete\\n",
    expected_current_sha256=__import__("hashlib").sha256(
        b"old-complete\\n"
    ).hexdigest(),
    publication_token="a" * 32,
)
"""
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", script, str(target), str(marker)],
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while not marker.exists() and process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("atomic writer never reached publication barrier")
        time.sleep(0.01)
    assert process.poll() is None
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=2.0)

    assert target.read_text(encoding="utf-8") == "old-complete\n"
    stage = tmp_path / (".target.txt.crucible-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp")
    stage.unlink()


@pytest.mark.parametrize("replacement_before_tools_import", [False, True])
def test_enforcing_loader_never_executes_preimport_replacement(
    tmp_path: Path,
    replacement_before_tools_import: bool,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    source_root = Path(host_mod.__file__).resolve(strict=True).parent
    package_root = tmp_path / "sealed-import" / "crucible"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    shutil.copytree(source_root, package_root / "tools")
    marker = tmp_path / "unattested-import-executed"
    copied_host = package_root / "tools" / "_isolated_mutation_host.py"
    replacement = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    script = [
        "import importlib",
        "import sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(package_root.parent)!r})",
    ]
    if replacement_before_tools_import:
        script.append(f"Path({str(copied_host)!r}).write_text({replacement!r})")
    script.append("import crucible.tools")
    if not replacement_before_tools_import:
        script.append(f"Path({str(copied_host)!r}).write_text({replacement!r})")
    script.append("importlib.import_module('crucible.tools._isolated_mutation_host')")
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", ";".join(script)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert marker.exists() is False
    assert completed.returncode != 0
    assert "trusted digest" in completed.stderr


def test_child_bootstrap_rejects_same_number_sealed_source_fd_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    marker = tmp_path / "unattested-child-executed"
    create = host_mod.ctypes.CDLL(None, use_errno=True).memfd_create
    create.restype = host_mod.ctypes.c_int
    replacement = int(create(b"hostile-sealed-source", 0x0001 | 0x0002))
    assert replacement >= 0
    payload = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    ).encode()
    os.write(replacement, payload)
    os.fsync(replacement)
    host_mod.fcntl.fcntl(
        replacement,
        host_mod._F_ADD_SEALS,
        host_mod._SOURCE_IMAGE_REQUIRED_SEALS,
    )
    host = _host(tmp_path)
    real_projection = host_mod._spawn_projection_targets
    swapped: list[int] = []

    def replace_attached_source(excluded, *, count):
        authority = host._source_authority  # noqa: SLF001
        assert authority is not None
        descriptor = authority.descriptor
        os.close(descriptor)
        os.dup2(replacement, descriptor, inheritable=False)
        swapped.append(descriptor)
        return real_projection(excluded, count=count)

    monkeypatch.setattr(host_mod, "_spawn_projection_targets", replace_attached_source)
    try:
        with pytest.raises(
            (MutationHostQualificationError, MutationHostProtocolError),
        ):
            host.start()
        assert swapped
        assert marker.exists() is False
        assert host.is_reaped is True
    finally:
        with suppress(OSError):
            os.close(replacement)


def test_nested_parent_rebind_cannot_split_named_and_descriptor_publication(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    nested = repo / "nested"
    nested.mkdir(parents=True)
    target = nested / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    retained = repo / "retained"

    def rebind() -> None:
        nested.rename(retained)
        nested.mkdir()
        (nested / "target.txt").write_text("old\n", encoding="utf-8")

    toolset = _text_authoring_toolset(repo, scope=("nested/target.txt",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
        _before_publish_ack=rebind,
    )
    with pytest.raises(
        MutationHostQualificationError,
        match="ancestor",
    ):
        host.dispatch(
            "write_source",
            {
                "file_path": "nested/target.txt",
                "content": "new\n",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
            },
        )
    assert (nested / "target.txt").read_text() == "old\n"
    assert (retained / "target.txt").read_text() == "old\n"
    assert not tuple(retained.glob(".*.crucible-*.tmp"))


@pytest.mark.parametrize(
    "attack",
    ["same-inode", "symlink", "fifo", "hardlink"],
)
def test_stage_object_substitution_never_publishes_or_deletes_unowned_entry(
    tmp_path: Path,
    attack: str,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("attacker\n", encoding="utf-8")
    staged: list[Path] = []

    def substitute() -> None:
        stage = next(tmp_path.glob(".*.crucible-*.tmp"))
        staged.append(stage)
        if attack == "same-inode":
            stage.write_text("attacker\n", encoding="utf-8")
            return
        stage.unlink()
        if attack == "symlink":
            stage.symlink_to(outside)
        elif attack == "fifo":
            os.mkfifo(stage, mode=0o600)
        else:
            os.link(outside, stage)

    host = _host(tmp_path, before_publish_ack=substitute)
    try:
        with pytest.raises(
            (MutationHostQualificationError, MutationHostLifecycleError),
        ):
            host.dispatch(
                "write_source",
                {"file_path": "target.txt", "content": "provider\n"},
            )
        assert not (tmp_path / "target.txt").exists()
        assert outside.read_text() == "attacker\n"
    finally:
        if staged:
            staged[0].unlink(missing_ok=True)
        host.stop()
    assert host.is_reaped is True


def test_child_death_rolls_back_parent_owned_nested_directory_chain(
    tmp_path: Path,
) -> None:
    toolset = _text_authoring_toolset(
        tmp_path,
        scope=("nested/deep/target.txt",),
    )
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    holder: dict[str, IsolatedMutationHost] = {}

    def kill_before_publication() -> None:
        host = holder["host"]
        assert host.pid is not None
        os.killpg(host.pid, signal.SIGKILL)

    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
        _before_publish_ack=kill_before_publication,
    )
    holder["host"] = host
    with pytest.raises(MutationHostProtocolError):
        host.dispatch(
            "write_source",
            {
                "file_path": "nested/deep/target.txt",
                "content": "never publish\n",
            },
        )
    assert not (tmp_path / "nested").exists()
    assert host.is_reaped is True


def test_memfd_effect_then_error_is_sticky_before_host_import(
    tmp_path: Path,
) -> None:
    import crucible.tools as tools_mod

    import_root = tmp_path / "fresh-import"
    import_root.mkdir()
    (import_root / "crucible").symlink_to(
        Path(tools_mod.__file__).resolve().parent.parent,
        target_is_directory=True,
    )
    script = "\n".join(
        [
            "import errno, json, os, sys",
            f"sys.path.insert(0, {str(import_root)!r})",
            "import crucible.tools as tools_mod",
            "real_cdll = tools_mod._ctypes.CDLL",
            "libc = real_cdll(None, use_errno=True)",
            "real_create = libc.memfd_create",
            "real_create.restype = tools_mod._ctypes.c_int",
            "leaked = []",
            "class EffectThenError:",
            "    restype = tools_mod._ctypes.c_int",
            "    def __call__(self, name, flags):",
            "        descriptor = int(real_create(name, flags))",
            "        leaked.append(descriptor)",
            "        raise OSError(errno.EINTR, 'effect then error')",
            "class FakeLibc:",
            "    memfd_create = EffectThenError()",
            "tools_mod._ctypes.CDLL = lambda *_a, **_k: FakeLibc()",
            "errors = []",
            "for _ in range(2):",
            "    try:",
            "        tools_mod._prepare_isolated_mutation_host_source()",
            "    except ImportError as exc:",
            "        errors.append(str(exc))",
            "print(json.dumps({'errors': errors, 'count': len(leaked), "
            "'open': [os.fstat(fd).st_ino > 0 for fd in leaked]}))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["count"] == 1
    assert observed["open"] == [True]
    assert len(observed["errors"]) == 2
    assert "memfd acquisition is uncertain" in observed["errors"][0]
    assert "remains uncertain" in observed["errors"][1]


def test_normal_tools_import_does_not_require_fcntl_or_memfd(tmp_path: Path) -> None:
    import crucible.tools as tools_mod

    import_root = tmp_path / "portable-import"
    import_root.mkdir()
    (import_root / "crucible").symlink_to(
        Path(tools_mod.__file__).resolve().parent.parent,
        target_is_directory=True,
    )
    script = "\n".join(
        [
            "import builtins, ctypes, importlib, json, os, sys",
            f"sys.path.insert(0, {str(import_root)!r})",
            "real_import = builtins.__import__",
            "real_cdll = ctypes.CDLL",
            "real_pread = os.pread",
            "def guarded_import(name, *args, **kwargs):",
            "    if name == 'fcntl':",
            "        raise ImportError('fcntl blocked')",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = guarded_import",
            "ctypes.CDLL = lambda *_a, **_k: (_ for _ in ()).throw("
            "RuntimeError('memfd blocked'))",
            "os.pread = lambda *_a, **_k: (_ for _ in ()).throw("
            "RuntimeError('pread blocked'))",
            "import crucible.tools as tools_mod",
            "host_name = 'crucible.tools._isolated_mutation_host'",
            "before = {'builder': hasattr(tools_mod, 'ToolSetBuilder'), "
            "'sealed': tools_mod._ISOLATED_MUTATION_HOST_DESCRIPTOR, "
            "'snapshotted': tools_mod._ISOLATED_MUTATION_HOST_PAYLOAD is not None, "
            "'registered': host_name in sys.modules}",
            "host_error = None",
            "try:",
            "    importlib.import_module(host_name)",
            "except ImportError as exc:",
            "    host_error = type(exc).__name__ + ': ' + str(exc)",
            "after_failure = {'registered': host_name in sys.modules, "
            "'builder': hasattr(tools_mod, 'ToolSetBuilder')}",
            "builtins.__import__ = real_import",
            "ctypes.CDLL = real_cdll",
            "os.pread = real_pread",
            "import crucible.backends.claude as claude_mod",
            "print(json.dumps({'before': before, 'error': host_error, "
            "'after_failure': after_failure, "
            "'claude': hasattr(claude_mod, 'ClaudeBackend')}))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["before"] == {
        "builder": True,
        "sealed": None,
        "snapshotted": False,
        "registered": False,
    }
    assert observed["error"].startswith("ImportError: isolated mutation host")
    assert observed["after_failure"] == {"registered": False, "builder": True}
    assert observed["claude"] is True


def test_lazy_composite_exports_preserve_exact_public_api_identity() -> None:
    import crucible.tools as tools_mod
    from crucible.tools._composite import CompositeToolSet, composite

    assert tools_mod.CompositeToolSet is CompositeToolSet
    assert tools_mod.composite is composite
    assert "CompositeToolSet" in tools_mod.__all__
    assert "composite" in tools_mod.__all__


def test_child_bootstrap_imports_runpy_only_after_source_validation() -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    bootstrap = host_mod._CHILD_BOOTSTRAP
    assert bootstrap.index("import runpy") > bootstrap.index("digest.hexdigest()")


@pytest.mark.parametrize("existing", [False, True])
def test_child_death_after_publication_syscall_rolls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bool,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    old_inode: int | None = None
    arguments: dict[str, str] = {
        "file_path": "target.txt",
        "content": "new\n",
    }
    if existing:
        target.write_text("old\n", encoding="utf-8")
        old_inode = target.stat().st_ino
        arguments["expected_sha256"] = hashlib.sha256(b"old\n").hexdigest()
    host = _host(tmp_path)
    killed = False
    if existing:
        real_exchange = host_mod._rename_exchange

        def exchange_then_kill(parent, first, second):
            nonlocal killed
            real_exchange(parent, first, second)
            if not killed:
                assert host.pid is not None
                os.killpg(host.pid, signal.SIGKILL)
                killed = True

        monkeypatch.setattr(host_mod, "_rename_exchange", exchange_then_kill)
    else:
        real_link = host_mod._link_descriptor_to_name

        def link_then_kill(source, parent, name):
            nonlocal killed
            real_link(source, parent, name)
            if not killed:
                assert host.pid is not None
                os.killpg(host.pid, signal.SIGKILL)
                killed = True

        monkeypatch.setattr(host_mod, "_link_descriptor_to_name", link_then_kill)

    with pytest.raises(MutationHostProtocolError):
        host.dispatch("write_source", arguments)

    if existing:
        assert target.read_text(encoding="utf-8") == "old\n"
        assert target.stat().st_ino == old_inode
    else:
        assert target.exists() is False
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))
    assert host.is_reaped is True


@pytest.mark.parametrize("existing", [False, True])
def test_ancestor_rebind_inside_publication_syscall_cannot_redirect_or_land(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bool,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    repo = tmp_path / "repo"
    nested = repo / "nested"
    retained = repo / "retained"
    attacker = repo / "attacker"
    nested.mkdir(parents=True)
    arguments: dict[str, str] = {
        "file_path": "nested/target.txt",
        "content": "new\n",
    }
    if existing:
        (nested / "target.txt").write_text("old\n", encoding="utf-8")
        arguments["expected_sha256"] = hashlib.sha256(b"old\n").hexdigest()
    toolset = _text_authoring_toolset(repo, scope=("nested/target.txt",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    rebound = False

    def rebind() -> None:
        nonlocal rebound
        if rebound:
            return
        nested.rename(retained)
        nested.mkdir()
        (nested / "victim.txt").write_text("attacker\n", encoding="utf-8")
        if existing:
            (nested / "target.txt").write_text("attacker target\n", encoding="utf-8")
        rebound = True

    if existing:
        real_exchange = host_mod._rename_exchange

        def rebind_then_exchange(parent, first, second):
            rebind()
            real_exchange(parent, first, second)

        monkeypatch.setattr(host_mod, "_rename_exchange", rebind_then_exchange)
    else:
        real_link = host_mod._link_descriptor_to_name

        def rebind_then_link(source, parent, name):
            rebind()
            real_link(source, parent, name)

        monkeypatch.setattr(host_mod, "_link_descriptor_to_name", rebind_then_link)

    with pytest.raises(MutationHostQualificationError, match="ancestor"):
        host.dispatch("write_source", arguments)

    assert (nested / "victim.txt").read_text(encoding="utf-8") == "attacker\n"
    if existing:
        assert (nested / "target.txt").read_text() == "attacker target\n"
        assert (retained / "target.txt").read_text() == "old\n"
    else:
        assert not (nested / "target.txt").exists()
        assert not (retained / "target.txt").exists()
    assert not tuple(retained.glob(".*.crucible-*.tmp"))
    assert host.is_reaped is False

    nested.rename(attacker)
    retained.rename(nested)
    host.stop()
    assert host.is_reaped is True
    assert (attacker / "victim.txt").read_text(encoding="utf-8") == "attacker\n"


def test_same_inode_alias_during_create_remains_owned_until_operator_settles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    alias = tmp_path / "attacker-alias.txt"
    real_link = host_mod._link_descriptor_to_name

    def alias_then_publish(source, parent, name):
        real_link(source, parent, alias.name)
        real_link(source, parent, name)

    monkeypatch.setattr(host_mod, "_link_descriptor_to_name", alias_then_publish)
    host = _host(tmp_path)
    with pytest.raises((MutationHostQualificationError, MutationHostLifecycleError)):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )
    assert not (tmp_path / "target.txt").exists()
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))
    assert alias.read_text(encoding="utf-8") == "new\n"
    assert host.is_reaped is False

    alias.unlink()
    host.stop()
    assert host.is_reaped is True


def test_post_publish_stat_failure_rolls_back_bound_ancestor_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    real_stat = host_mod.os.stat
    injected = False

    def fail_after_mkdir(path, *args, **kwargs):
        nonlocal injected
        observed = real_stat(path, *args, **kwargs)
        if (
            not injected
            and path == "nested"
            and kwargs.get("dir_fd") is not None
            and stat.S_ISDIR(observed.st_mode)
        ):
            injected = True
            raise OSError(errno.EIO, "injected after mkdir")
        return observed

    monkeypatch.setattr(host_mod.os, "stat", fail_after_mkdir)
    toolset = _text_authoring_toolset(tmp_path, scope=("nested/target.txt",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "write_source",
            {"file_path": "nested/target.txt", "content": "new\n"},
        )
    nested = tmp_path / "nested"
    assert injected is True
    assert nested.exists() is False
    assert host.is_reaped is True


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("attack", ["alias", "rebind", "transient_write"])
def test_final_commit_fsync_attack_never_returns_unvalidated_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bool,
    attack: str,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    repo = tmp_path / "repo"
    nested = repo / "nested"
    retained = repo / "retained"
    attacker = repo / "attacker"
    nested.mkdir(parents=True)
    target = nested / "target.txt"
    arguments: dict[str, str] = {
        "file_path": "nested/target.txt",
        "content": "new\n",
    }
    if existing:
        target.write_text("old\n", encoding="utf-8")
        arguments["expected_sha256"] = hashlib.sha256(b"old\n").hexdigest()
    toolset = _text_authoring_toolset(repo, scope=("nested/target.txt",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    real_fsync = host_mod.os.fsync
    alias = nested / "attacker-alias.txt"
    injected = False

    def attack_final_fsync(descriptor):
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and transaction.phase == "committed_unsynced"
        ):
            injected = True
            if attack == "alias":
                os.link(target, alias)
            elif attack == "rebind":
                nested.rename(retained)
                nested.mkdir()
                (nested / "victim.txt").write_text("attacker\n", encoding="utf-8")
            else:
                transient = nested / "transient-alias.txt"
                os.link(target, transient)
                transient.write_text("tampered\n", encoding="utf-8")
                transient.unlink()
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", attack_final_fsync)
    with pytest.raises(MutationHostQualificationError):
        host.dispatch("write_source", arguments)

    assert injected is True
    assert host.is_reaped is False
    if attack == "alias":
        assert target.read_text(encoding="utf-8") == "new\n"
        assert alias.read_text(encoding="utf-8") == "new\n"
        alias.unlink()
    elif attack == "rebind":
        assert (retained / "target.txt").read_text(encoding="utf-8") == "new\n"
        assert (nested / "victim.txt").read_text(encoding="utf-8") == "attacker\n"
        nested.rename(attacker)
        retained.rename(nested)
    else:
        assert target.read_text(encoding="utf-8") == "tampered\n"
        target.write_text("new\n", encoding="utf-8")
    host.stop()
    assert host.is_reaped is True
    assert (nested / "target.txt").read_text(encoding="utf-8") == "new\n"
    if attack == "rebind":
        assert (attacker / "victim.txt").read_text(encoding="utf-8") == "attacker\n"


def test_replacement_old_target_link_change_during_final_validation_is_sticky(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    host = _host(tmp_path)
    real_revalidate = host_mod._ParentPublicationTransaction.revalidate
    real_target_identity = (
        host_mod._ParentPublicationTransaction._target_descriptor_identity
    )
    injected = False
    report_extra_link = False

    def arm_post_chain_observation(transaction):
        nonlocal injected, report_extra_link
        real_revalidate(transaction)
        if transaction.phase == "committed_unsynced" and not injected:
            injected = True
            report_extra_link = True

    def observe_old_target_link_change(transaction):
        identity = real_target_identity(transaction)
        if report_extra_link:
            changed = list(identity)
            changed[3] = 1
            return tuple(changed)
        return identity

    monkeypatch.setattr(
        host_mod._ParentPublicationTransaction,
        "revalidate",
        arm_post_chain_observation,
    )
    monkeypatch.setattr(
        host_mod._ParentPublicationTransaction,
        "_target_descriptor_identity",
        observe_old_target_link_change,
    )
    with pytest.raises(MutationHostQualificationError):
        host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "new\n",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
            },
        )
    assert injected is True
    assert target.read_text(encoding="utf-8") == "new\n"
    assert host.is_reaped is False

    report_extra_link = False
    host.stop()
    assert host.is_reaped is True


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("attack", ["effect_error", "rebind"])
def test_rollback_fsync_uncertainty_never_claims_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bool,
    attack: str,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    repo = tmp_path / "repo"
    nested = repo / "nested"
    retained = repo / "retained"
    attacker = repo / "attacker"
    nested.mkdir(parents=True)
    target = nested / "target.txt"
    arguments: dict[str, str] = {
        "file_path": "nested/target.txt",
        "content": "new\n",
    }
    if existing:
        target.write_text("old\n", encoding="utf-8")
        arguments["expected_sha256"] = hashlib.sha256(b"old\n").hexdigest()
    toolset = _text_authoring_toolset(repo, scope=("nested/target.txt",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    killed = False
    if existing:
        real_exchange = host_mod._rename_exchange

        def exchange_then_kill(parent, first, second):
            nonlocal killed
            real_exchange(parent, first, second)
            if not killed:
                assert host.pid is not None
                os.killpg(host.pid, signal.SIGKILL)
                killed = True

        monkeypatch.setattr(host_mod, "_rename_exchange", exchange_then_kill)
    else:
        real_link = host_mod._link_descriptor_to_name

        def create_then_kill(source, parent, name):
            nonlocal killed
            real_link(source, parent, name)
            if not killed:
                assert host.pid is not None
                os.killpg(host.pid, signal.SIGKILL)
                killed = True

        monkeypatch.setattr(host_mod, "_link_descriptor_to_name", create_then_kill)
    real_fsync = host_mod.os.fsync
    injected = False

    def attack_rollback_fsync(descriptor):
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and transaction.phase == "rolled_back_unsynced"
        ):
            injected = True
            if attack == "effect_error":
                real_fsync(descriptor)
                raise OSError(errno.EIO, "injected after rollback fsync")
            nested.rename(retained)
            nested.mkdir()
            (nested / "victim.txt").write_text("attacker\n", encoding="utf-8")
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", attack_rollback_fsync)
    with pytest.raises((MutationHostLifecycleError, MutationHostQualificationError)):
        host.dispatch("write_source", arguments)

    assert injected is True
    assert host.is_reaped is False
    observed_parent = retained if attack == "rebind" else nested
    observed_target = observed_parent / "target.txt"
    if existing:
        assert observed_target.read_text(encoding="utf-8") == "old\n"
    else:
        assert observed_target.exists() is False
    assert not tuple(observed_parent.glob(".*.crucible-*.tmp"))
    if attack == "rebind":
        nested.rename(attacker)
        retained.rename(nested)
    host.stop()
    assert host.is_reaped is True
    if attack == "rebind":
        assert (attacker / "victim.txt").read_text(encoding="utf-8") == "attacker\n"


def test_replacement_rollback_rejects_transient_original_target_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    host = _host(tmp_path)
    real_exchange = host_mod._rename_exchange
    killed = False

    def exchange_then_kill(parent, first, second):
        nonlocal killed
        real_exchange(parent, first, second)
        if not killed:
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)
            killed = True

    monkeypatch.setattr(host_mod, "_rename_exchange", exchange_then_kill)
    real_fsync = host_mod.os.fsync
    injected = False

    def mutate_during_rollback_fsync(descriptor):
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and transaction.phase == "rolled_back_unsynced"
        ):
            injected = True
            alias = tmp_path / "transient-old-alias.txt"
            os.link(target, alias)
            alias.write_text("tampered\n", encoding="utf-8")
            alias.unlink()
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", mutate_during_rollback_fsync)
    with pytest.raises(MutationHostQualificationError, match="target content"):
        host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "new\n",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
            },
        )
    assert injected is True
    assert target.read_text(encoding="utf-8") == "tampered\n"
    assert host.is_reaped is False

    target.write_text("old\n", encoding="utf-8")
    host.stop()
    assert target.read_text(encoding="utf-8") == "old\n"
    assert host.is_reaped is True


def test_replacement_preserves_exact_existing_permission_mode(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o755)
    host = _host(tmp_path)
    result = host.dispatch(
        "write_source",
        {
            "file_path": "target.txt",
            "content": "new\n",
            "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
        },
    )
    host.stop()

    assert result == "OK: wrote 1 lines to target.txt"
    assert target.read_text(encoding="utf-8") == "new\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_effect_then_error_chmod_settles_only_the_exact_observed_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o755)
    host = _host(tmp_path)
    real_fchmod = host_mod.os.fchmod
    injected = False

    def effect_then_error(descriptor: int, mode: int) -> None:
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and descriptor == transaction.stage_descriptor
            and mode == 0o755
        ):
            injected = True
            real_fchmod(descriptor, mode)
            raise OSError(errno.EIO, "injected after chmod effect")
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(host_mod.os, "fchmod", effect_then_error)
    result = host.dispatch(
        "write_source",
        {
            "file_path": "target.txt",
            "content": "new\n",
            "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
        },
    )
    host.stop()

    assert injected is True
    assert result == "OK: wrote 1 lines to target.txt"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_no_effect_chmod_error_cannot_publish_the_wrong_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o755)
    host = _host(tmp_path)
    real_fchmod = host_mod.os.fchmod
    injected = False

    def no_effect_error(descriptor: int, mode: int) -> None:
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and descriptor == transaction.stage_descriptor
            and mode == 0o755
        ):
            injected = True
            raise OSError(errno.EIO, "injected before chmod effect")
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(host_mod.os, "fchmod", no_effect_error)
    with pytest.raises(
        MutationHostLifecycleError,
        match="permission update outcome is uncertain",
    ):
        host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "new\n",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
            },
        )

    assert injected is True
    assert target.read_text(encoding="utf-8") == "old\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    assert host.is_reaped is True


def test_created_ancestor_entries_are_fsynced_in_parent_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    real_fsync = host_mod.os.fsync
    directory_fsyncs: list[tuple[int, int]] = []

    def trace_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            directory_fsyncs.append((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", trace_fsync)
    toolset = _text_authoring_toolset(tmp_path, scope=("a",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    result = host.dispatch(
        "write_source",
        {"file_path": "a/b/target.txt", "content": "nested\n"},
    )
    host.stop()

    root_key = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    first_key = ((tmp_path / "a").stat().st_dev, (tmp_path / "a").stat().st_ino)
    final_key = (
        (tmp_path / "a/b").stat().st_dev,
        (tmp_path / "a/b").stat().st_ino,
    )
    assert result == "OK: wrote 1 lines to a/b/target.txt"
    assert directory_fsyncs.index(root_key) < directory_fsyncs.index(first_key)
    assert directory_fsyncs.index(first_key) < directory_fsyncs.index(final_key)


def test_ancestor_fsync_effect_error_rolls_back_without_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    real_fsync = host_mod.os.fsync
    injected = False
    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )

    def effect_then_error(descriptor: int) -> None:
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        pending = (
            None if transaction is None else transaction.pending_directory_creation
        )
        if (
            not injected
            and pending is not None
            and pending.final_name == "nested"
            and pending.creation_unsynced
            and descriptor == pending.parent_descriptor
        ):
            injected = True
            real_fsync(descriptor)
            raise OSError(errno.EIO, "injected after ancestor fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", effect_then_error)
    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "write_source",
            {"file_path": "nested/target.txt", "content": "new\n"},
        )

    assert injected is True
    assert not (tmp_path / "nested").exists()
    assert host.is_reaped is True


def test_ancestor_removal_fsync_effect_error_requires_exact_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    real_link = host_mod._link_descriptor_to_name
    real_fsync = host_mod.os.fsync
    killed = False
    injected = False

    def link_then_kill(source: int, parent: int, name: str) -> None:
        nonlocal killed
        real_link(source, parent, name)
        if not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    def removal_fsync_then_error(descriptor: int) -> None:
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        removing = (
            None
            if transaction is None
            else next(
                (
                    ancestor
                    for ancestor in transaction.ancestors
                    if ancestor.created and ancestor.removal_unsynced
                ),
                None,
            )
        )
        if (
            not injected
            and removing is not None
            and descriptor == removing.parent_descriptor
        ):
            injected = True
            real_fsync(descriptor)
            raise OSError(errno.EIO, "injected after ancestor removal fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod, "_link_descriptor_to_name", link_then_kill)
    monkeypatch.setattr(host_mod.os, "fsync", removal_fsync_then_error)
    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "write_source",
            {"file_path": "nested/target.txt", "content": "new\n"},
        )

    assert injected is True
    assert not (tmp_path / "nested").exists()
    assert host.is_reaped is False
    host.stop()
    assert host.is_reaped is True


def test_unpublished_stage_removal_is_fsynced_before_reap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    def abort_before_publication() -> None:
        raise RuntimeError("abort before publication")

    host = _host(
        tmp_path,
        before_publish_ack=abort_before_publication,
    )
    real_fsync = host_mod.os.fsync
    removal_fsyncs = 0

    def trace_removal_fsync(descriptor: int) -> None:
        nonlocal removal_fsyncs
        transaction = host._active_transaction  # noqa: SLF001
        if (
            transaction is not None
            and transaction.stage_removal_unsynced
            and descriptor == transaction.final_parent_descriptor
        ):
            removal_fsyncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", trace_removal_fsync)
    with pytest.raises(RuntimeError, match="abort before publication"):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )

    assert removal_fsyncs == 1
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))
    assert host.is_reaped is True


def test_unpublished_stage_removal_fsync_effect_error_requires_exact_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    def abort_before_publication() -> None:
        raise RuntimeError("abort before publication")

    host = _host(
        tmp_path,
        before_publish_ack=abort_before_publication,
    )
    real_fsync = host_mod.os.fsync
    removal_fsyncs = 0

    def effect_then_error(descriptor: int) -> None:
        nonlocal removal_fsyncs
        transaction = host._active_transaction  # noqa: SLF001
        if (
            transaction is not None
            and transaction.stage_removal_unsynced
            and descriptor == transaction.final_parent_descriptor
        ):
            removal_fsyncs += 1
            real_fsync(descriptor)
            if removal_fsyncs == 1:
                raise OSError(errno.EIO, "injected after stage-removal fsync")
            return
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", effect_then_error)
    with pytest.raises(MutationHostLifecycleError, match="cleanup was incomplete"):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )

    assert removal_fsyncs == 1
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))
    assert host.is_reaped is False
    host.stop()
    assert removal_fsyncs == 2
    assert host.is_reaped is True


def test_publication_ancestor_creation_bypasses_python_mkdir_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    real_mkdir = host_mod.os.mkdir
    attempted = False

    def substitute_after_mkdir(path, *args, **kwargs) -> None:
        nonlocal attempted
        if path == "nested" and kwargs.get("dir_fd") is not None:
            attempted = True
            real_mkdir(path, *args, **kwargs)
            raise OSError(errno.EIO, "injected after public mkdir")
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(host_mod.os, "mkdir", substitute_after_mkdir)
    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    result = host.dispatch(
        "write_source",
        {"file_path": "nested/target.txt", "content": "new\n"},
    )
    host.stop()

    assert attempted is False
    assert result == "OK: wrote 1 lines to nested/target.txt"
    assert (tmp_path / "nested/target.txt").read_text(encoding="utf-8") == "new\n"


@pytest.mark.parametrize("existing", [False, True])
def test_transient_write_restore_at_commit_checkpoint_never_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bool,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    arguments: dict[str, str] = {
        "file_path": "target.txt",
        "content": "new\n",
    }
    if existing:
        target.write_text("old\n", encoding="utf-8")
        arguments["expected_sha256"] = hashlib.sha256(b"old\n").hexdigest()
    host = _host(tmp_path)
    real_fsync = host_mod.os.fsync
    injected = False

    def transient_restore(descriptor: int) -> None:
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and transaction.phase == "committed_unsynced"
            and descriptor == transaction.final_parent_descriptor
        ):
            injected = True
            alias = tmp_path / "transient-new-alias.txt"
            os.link(target, alias)
            target.write_text("bad\n", encoding="utf-8")
            target.write_text("new\n", encoding="utf-8")
            alias.unlink()
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod.os, "fsync", transient_restore)
    with pytest.raises(MutationHostQualificationError, match="exact retry"):
        host.dispatch("write_source", arguments)

    assert injected is True
    assert target.read_text(encoding="utf-8") == "new\n"
    assert host.is_reaped is True


def test_transient_restore_at_rollback_checkpoint_requires_later_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    host = _host(tmp_path)
    real_exchange = host_mod._rename_exchange
    real_fsync = host_mod.os.fsync
    killed = False
    injected = False

    def exchange_then_kill(parent: int, first: str, second: str) -> None:
        nonlocal killed
        real_exchange(parent, first, second)
        if not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    def transient_restore(descriptor: int) -> None:
        nonlocal injected
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and transaction.phase == "rolled_back_unsynced"
            and descriptor == transaction.final_parent_descriptor
        ):
            injected = True
            alias = tmp_path / "transient-old-alias.txt"
            os.link(target, alias)
            target.write_text("bad\n", encoding="utf-8")
            target.write_text("old\n", encoding="utf-8")
            alias.unlink()
        real_fsync(descriptor)

    monkeypatch.setattr(host_mod, "_rename_exchange", exchange_then_kill)
    monkeypatch.setattr(host_mod.os, "fsync", transient_restore)
    with pytest.raises(MutationHostQualificationError, match="exact retry"):
        host.dispatch(
            "write_source",
            {
                "file_path": "target.txt",
                "content": "new\n",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
            },
        )

    assert injected is True
    assert target.read_text(encoding="utf-8") == "old\n"
    assert host.is_reaped is False
    host.stop()
    assert host.is_reaped is True


@pytest.mark.parametrize("rollback", [False, True])
def test_post_revalidate_ancestor_rebind_never_claims_commit_or_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rollback: bool,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    repo = tmp_path / "repo"
    nested = repo / "nested"
    retained = repo / "retained"
    attacker = repo / "attacker"
    nested.mkdir(parents=True)
    target = nested / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    toolset = _text_authoring_toolset(repo, scope=("nested/target.txt",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    real_revalidate = host_mod._ParentPublicationTransaction.revalidate
    real_exchange = host_mod._rename_exchange
    rebound = False
    killed = False

    def revalidate_then_rebind(transaction) -> None:
        nonlocal rebound
        real_revalidate(transaction)
        wanted = "rolled_back_unsynced" if rollback else "committed_unsynced"
        if not rebound and transaction.phase == wanted:
            rebound = True
            nested.rename(retained)
            nested.mkdir()
            (nested / "victim.txt").write_text("victim\n", encoding="utf-8")

    def exchange_then_kill(parent: int, first: str, second: str) -> None:
        nonlocal killed
        real_exchange(parent, first, second)
        if rollback and not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    monkeypatch.setattr(
        host_mod._ParentPublicationTransaction,
        "revalidate",
        revalidate_then_rebind,
    )
    monkeypatch.setattr(host_mod, "_rename_exchange", exchange_then_kill)
    with pytest.raises(MutationHostQualificationError, match="ancestor"):
        host.dispatch(
            "write_source",
            {
                "file_path": "nested/target.txt",
                "content": "new\n",
                "expected_sha256": hashlib.sha256(b"old\n").hexdigest(),
            },
        )

    assert rebound is True
    assert host.is_reaped is False
    nested.rename(attacker)
    retained.rename(nested)
    host.stop()
    assert host.is_reaped is True
    expected = "old\n" if rollback else "new\n"
    assert target.read_text(encoding="utf-8") == expected
    assert (attacker / "victim.txt").read_text(encoding="utf-8") == "victim\n"


@pytest.mark.parametrize("entry_kind", ["target", "stage", "ancestor"])
def test_python_delete_hook_cannot_substitute_unowned_cleanup_victim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entry_kind: str,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    unlink_called = False
    rmdir_called = False
    real_unlink = host_mod.os.unlink
    real_rmdir = host_mod.os.rmdir

    def forbidden_unlink(path, *args, **kwargs) -> None:
        nonlocal unlink_called
        if kwargs.get("dir_fd") is not None:
            unlink_called = True
            raise AssertionError("public cleanup reached os.unlink")
        real_unlink(path, *args, **kwargs)

    def forbidden_rmdir(path, *args, **kwargs) -> None:
        nonlocal rmdir_called
        if kwargs.get("dir_fd") is not None:
            rmdir_called = True
            raise AssertionError("public cleanup reached os.rmdir")
        real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(host_mod.os, "unlink", forbidden_unlink)
    monkeypatch.setattr(host_mod.os, "rmdir", forbidden_rmdir)
    if entry_kind == "ancestor":
        toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
        spec = build_isolated_toolset_spec(
            toolset,
            expected_write_names=frozenset({"write_source", "apply_patch"}),
        )
        host = IsolatedMutationHost(
            spec,
            invocation_deadline=time.monotonic() + 10.0,
        )
        path = "nested/target.txt"
    else:
        host = _host(tmp_path)
        path = "target.txt"
    real_link = host_mod._link_descriptor_to_name
    killed = False

    def link_then_kill(source: int, parent: int, name: str) -> None:
        nonlocal killed
        real_link(source, parent, name)
        if not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    monkeypatch.setattr(host_mod, "_link_descriptor_to_name", link_then_kill)
    with pytest.raises(MutationHostProtocolError):
        host.dispatch("write_source", {"file_path": path, "content": "new\n"})

    assert unlink_called is False
    assert rmdir_called is False
    assert not (tmp_path / path).exists()
    if entry_kind == "ancestor":
        assert not (tmp_path / "nested").exists()
    assert host.is_reaped is True


def test_same_uid_quarantine_boundary_swap_cannot_claim_false_settlement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The documented same-UID exclusion still fails closed on retained state."""

    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_link = host_mod._link_descriptor_to_name
    real_unlinkat = host_mod._unlinkat_direct
    killed = False
    injected = False
    escaped = tmp_path / "same-uid-escaped-new"
    victim_proof = tmp_path / "same-uid-victim-proof"

    def link_then_kill(source: int, parent: int, name: str) -> None:
        nonlocal killed
        real_link(source, parent, name)
        if not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    def swap_inside_unlink_boundary(
        parent: int,
        name: str,
        *,
        directory: bool,
    ) -> None:
        nonlocal injected
        if not injected and not directory:
            injected = True
            parent_path = Path(os.readlink(f"/proc/self/fd/{parent}"))
            quarantine = parent_path / name
            quarantine.rename(escaped)
            quarantine.write_text("unowned\n", encoding="utf-8")
            os.link(quarantine, victim_proof)
        real_unlinkat(parent, name, directory=directory)

    monkeypatch.setattr(host_mod, "_link_descriptor_to_name", link_then_kill)
    monkeypatch.setattr(host_mod, "_unlinkat_direct", swap_inside_unlink_boundary)
    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )

    assert injected is True
    assert host.is_reaped is False
    assert escaped.read_text(encoding="utf-8") == "new\n"
    assert victim_proof.read_text(encoding="utf-8") == "unowned\n"
    escaped.unlink()
    host.stop()
    assert host.is_reaped is True
    assert victim_proof.read_text(encoding="utf-8") == "unowned\n"


def test_bound_directory_close_error_never_reuses_numeric_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    state = host_mod._BoundDirectoryCreation(
        parent_descriptor=parent_descriptor,
        parent_identity=host_mod._stable_directory_identity(
            os.fstat(parent_descriptor)
        ),
        expected_mount_id=host_mod._descriptor_mount_id(parent_descriptor),
        final_name="owned",
        private_name=".owned-private",
        mode=0o755,
        token="a" * 32,
    )
    state.create()
    state.publish_and_sync()
    retained_numeric = state.descriptor
    assert retained_numeric is not None
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    real_close = host_mod.os.close
    sentinel_descriptor: int | None = None

    def close_then_reuse(descriptor: int) -> None:
        nonlocal sentinel_descriptor
        if descriptor == retained_numeric and sentinel_descriptor is None:
            real_close(descriptor)
            sentinel_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            sentinel.rmdir()
            assert sentinel_descriptor == retained_numeric
            raise OSError(errno.EIO, "injected after close effect")
        real_close(descriptor)

    monkeypatch.setattr(host_mod.os, "close", close_then_reuse)
    with pytest.raises(MutationHostLifecycleError, match="descriptor close"):
        state.rollback()
    assert state.descriptor is None
    assert state.descriptor_settlement_uncertain is True
    assert sentinel_descriptor == retained_numeric

    with pytest.raises(
        MutationHostLifecycleError, match="settlement remains uncertain"
    ):
        state.rollback()
    os.fstat(sentinel_descriptor)
    real_close(sentinel_descriptor)
    real_close(parent_descriptor)


def test_private_root_transfer_failure_has_one_descriptor_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_mount_id = host_mod._descriptor_mount_id
    real_close = host_mod.os.close
    injected = False
    retained_numeric: int | None = None
    sentinel_descriptor: int | None = None
    sentinel = tmp_path / "private-root-transfer-sentinel"
    sentinel.mkdir()

    def fail_transfer_mount(descriptor: int) -> int:
        nonlocal injected, retained_numeric
        authority = host._private_root_authority  # noqa: SLF001
        pending = None if authority is None else authority.pending_creation
        if (
            not injected
            and pending is not None
            and pending.descriptor == descriptor
            and pending.published
            and not pending.creation_unsynced
        ):
            injected = True
            retained_numeric = descriptor
            raise OSError(errno.EIO, "injected at private-root transfer")
        return real_mount_id(descriptor)

    def close_then_reuse(descriptor: int) -> None:
        nonlocal sentinel_descriptor
        real_close(descriptor)
        if descriptor == retained_numeric and sentinel_descriptor is None:
            sentinel_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            sentinel.rmdir()
            assert sentinel_descriptor == retained_numeric

    monkeypatch.setattr(host_mod, "_descriptor_mount_id", fail_transfer_mount)
    monkeypatch.setattr(host_mod.os, "close", close_then_reuse)
    with pytest.raises(
        MutationHostQualificationError,
        match="startup authority is unavailable",
    ) as caught:
        host.start()
    assert isinstance(caught.value.__cause__, OSError)
    assert "private-root transfer" in str(caught.value.__cause__)

    assert injected is True
    assert host.is_reaped is True
    assert sentinel_descriptor == retained_numeric
    os.fstat(sentinel_descriptor)
    real_close(sentinel_descriptor)


def test_publication_ancestor_transfer_failure_has_one_descriptor_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    real_mount_id = host_mod._descriptor_mount_id
    real_close = host_mod.os.close
    injected = False
    retained_numeric: int | None = None
    sentinel_descriptor: int | None = None
    sentinel = tmp_path / "ancestor-transfer-sentinel"
    sentinel.mkdir()

    def fail_transfer_mount(descriptor: int) -> int:
        nonlocal injected, retained_numeric
        transaction = host._active_transaction  # noqa: SLF001
        pending = (
            None if transaction is None else transaction.pending_directory_creation
        )
        if (
            not injected
            and pending is not None
            and pending.final_name == "nested"
            and pending.descriptor == descriptor
            and pending.published
            and not pending.creation_unsynced
        ):
            injected = True
            retained_numeric = descriptor
            raise OSError(errno.EIO, "injected at ancestor transfer")
        return real_mount_id(descriptor)

    def close_then_reuse(descriptor: int) -> None:
        nonlocal sentinel_descriptor
        real_close(descriptor)
        if descriptor == retained_numeric and sentinel_descriptor is None:
            sentinel_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            sentinel.rmdir()
            assert sentinel_descriptor == retained_numeric

    monkeypatch.setattr(host_mod, "_descriptor_mount_id", fail_transfer_mount)
    monkeypatch.setattr(host_mod.os, "close", close_then_reuse)
    with pytest.raises(OSError, match="ancestor transfer"):
        host.dispatch(
            "write_source",
            {"file_path": "nested/target.txt", "content": "new\n"},
        )

    assert injected is True
    assert host.is_reaped is True
    assert not (tmp_path / "nested").exists()
    assert sentinel_descriptor == retained_numeric
    os.fstat(sentinel_descriptor)
    real_close(sentinel_descriptor)


def test_created_ancestor_lost_authority_suppresses_missing_name_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    identity = host_mod._stable_directory_identity(os.fstat(descriptor))
    mount_id = host_mod._descriptor_mount_id(descriptor)
    root = host_mod._MutationRootAuthority(
        path=tmp_path,
        expected_identity=identity,
        expected_mount_id=mount_id,
        child_descriptor=999,
        descriptor=descriptor,
    )
    roots = host_mod._MutationRootAuthorities(roots=(root,))
    context = host_mod.MutationContextSpec(
        workspace_root=str(tmp_path),
        workspace_root_identity=identity,
        workspace_root_mount_id=mount_id,
        repo_root=str(tmp_path),
        repo_root_identity=identity,
        repo_root_mount_id=mount_id,
        workspace_relative_paths=False,
        write_scope=("nested",),
        write_exact_scope=(),
        write_deny_scope=(),
        mutation_journal_dir=None,
        mutation_journal_transaction_id=None,
        container_id=None,
        container_workdir=".",
    )
    creations: list[LostAuthorityCreation] = []

    class LostAuthorityCreation:
        def __init__(self, **settings: object) -> None:
            self.settings = settings
            self.descriptor = None
            self.identity = None
            self.operations: list[str] = []
            creations.append(self)

        def create(self) -> None:
            self.operations.append("create")

        def publish_and_sync(self) -> None:
            self.operations.append("publish_and_sync")

    attached: list[host_mod._ParentPublicationTransaction] = []
    monkeypatch.setattr(host_mod, "_BoundDirectoryCreation", LostAuthorityCreation)

    try:
        with pytest.raises(
            MutationHostLifecycleError,
            match="^created publication ancestor lost its authority$",
        ) as caught:
            host_mod._ParentPublicationTransaction.prepare(
                roots,
                context,
                "nested/target.txt",
                "authority-token",
                attach=attached.append,
            )
    finally:
        os.close(descriptor)
        root.descriptor = None

    failure = caught.value
    assert failure.__cause__ is None
    assert isinstance(failure.__context__, FileNotFoundError)
    assert failure.__suppress_context__ is True
    assert len(attached) == 1
    assert len(creations) == 1
    pending = creations[0]
    assert pending.operations == ["create", "publish_and_sync"]
    assert pending.settings["parent_descriptor"] == descriptor
    assert pending.settings["parent_identity"] == identity
    assert pending.settings["expected_mount_id"] == mount_id
    assert attached[0].pending_directory_creation is pending
    assert attached[0].ancestors == []
    assert attached[0].final_parent_descriptor == descriptor
    assert not (tmp_path / "nested").exists()


def test_removed_ancestor_unknown_name_suppresses_missing_name_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    revalidations: list[bool] = []

    class RootAuthority:
        def revalidate(self, *, require_named: bool) -> None:
            revalidations.append(require_named)

    class LinkMetadata:
        def __init__(self, links: int) -> None:
            self.st_nlink = links

    missing_name = FileNotFoundError(
        errno.ENOENT,
        "incidental publication name lookup",
        "nested",
    )

    class HostOs:
        def __init__(self) -> None:
            self.links = iter((0, 0, 1))
            self.fstat_descriptors: list[int] = []
            self.fsync_descriptors: list[int] = []
            self.stat_calls: list[tuple[str, int, bool]] = []

        def fstat(self, descriptor: int) -> LinkMetadata:
            self.fstat_descriptors.append(descriptor)
            return LinkMetadata(next(self.links))

        def fsync(self, descriptor: int) -> None:
            self.fsync_descriptors.append(descriptor)

        def stat(
            self,
            name: str,
            *,
            dir_fd: int,
            follow_symlinks: bool,
        ) -> None:
            self.stat_calls.append((name, dir_fd, follow_symlinks))
            raise missing_name

    removal_state = host_mod._QuarantineRemovalState(
        binding=("nested", ".nested.crucible-quarantine")
    )
    ancestor = host_mod._PublicationAncestor(
        parent_descriptor=20,
        name="nested",
        descriptor=21,
        identity=(1,),
        mount_id=1,
        created=True,
        creation_unsynced=True,
        removal_state=removal_state,
    )
    transaction = host_mod._ParentPublicationTransaction(
        root=RootAuthority(),
        normalized="nested/target.txt",
        token="authority-token",
        ancestors=[ancestor],
        leaf="target.txt",
        final_parent_descriptor=20,
        stage_name=".target.txt.crucible-authority-token.tmp",
    )
    host_os = HostOs()
    quarantine_calls: list[tuple[int, str, int, bool, str, object]] = []

    def record_quarantine(
        parent_descriptor: int,
        name: str,
        descriptor: int,
        *,
        directory: bool,
        token: str,
        removal_state: object,
    ) -> None:
        quarantine_calls.append(
            (
                parent_descriptor,
                name,
                descriptor,
                directory,
                token,
                removal_state,
            )
        )

    monkeypatch.setattr(host_mod, "os", host_os)
    monkeypatch.setattr(
        host_mod,
        "_quarantine_remove_bound_entry",
        record_quarantine,
    )

    with pytest.raises(
        MutationHostLifecycleError,
        match="^removed publication ancestor retained an unknown name$",
    ) as caught:
        transaction._cleanup_created_ancestors()

    failure = caught.value
    assert failure.__cause__ is None
    assert failure.__context__ is missing_name
    assert failure.__suppress_context__ is True
    assert host_os.fstat_descriptors == [21, 21, 21]
    assert host_os.fsync_descriptors == [20]
    assert host_os.stat_calls == [("nested", 20, False)]
    assert revalidations == [True]
    assert quarantine_calls == [
        (20, "nested", 21, True, "authority-token", removal_state)
    ]
    assert ancestor.descriptor == 21
    assert ancestor.creation_unsynced is True
    assert ancestor.removal_unsynced is True
    assert ancestor.removed is False
    assert ancestor.removal_state is removal_state


def test_existing_ancestor_validation_failure_retains_one_descriptor_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    (tmp_path / "nested").mkdir()
    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    real_mount_id = host_mod._descriptor_mount_id
    real_close = host_mod.os.close
    injected = False
    retained_numeric: int | None = None
    sentinel_descriptor: int | None = None
    sentinel = tmp_path / "existing-ancestor-sentinel"
    sentinel.mkdir()

    def fail_retained_ancestor_mount(descriptor: int) -> int:
        nonlocal injected, retained_numeric
        transaction = host._active_transaction  # noqa: SLF001
        if (
            not injected
            and transaction is not None
            and any(
                ancestor.name == "nested"
                and not ancestor.created
                and ancestor.descriptor == descriptor
                for ancestor in transaction.ancestors
            )
        ):
            injected = True
            retained_numeric = descriptor
            raise OSError(errno.EIO, "injected at existing ancestor validation")
        return real_mount_id(descriptor)

    def close_then_reuse(descriptor: int) -> None:
        nonlocal sentinel_descriptor
        real_close(descriptor)
        if descriptor == retained_numeric and sentinel_descriptor is None:
            sentinel_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            sentinel.rmdir()
            assert sentinel_descriptor == retained_numeric

    monkeypatch.setattr(host_mod, "_descriptor_mount_id", fail_retained_ancestor_mount)
    monkeypatch.setattr(host_mod.os, "close", close_then_reuse)
    with pytest.raises(OSError, match="existing ancestor validation"):
        host.dispatch(
            "write_source",
            {"file_path": "nested/target.txt", "content": "new\n"},
        )

    assert injected is True
    assert host.is_reaped is True
    assert not (tmp_path / "nested/target.txt").exists()
    assert sentinel_descriptor == retained_numeric
    assert sentinel_descriptor is not None
    os.fstat(sentinel_descriptor)
    real_close(sentinel_descriptor)


def test_file_quarantine_no_effect_unlink_restores_exact_retry_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    def abort_before_publication() -> None:
        raise RuntimeError("abort before publication")

    host = _host(tmp_path, before_publish_ack=abort_before_publication)
    real_unlinkat = host_mod._unlinkat_direct
    injected = False

    def no_effect_once(parent: int, name: str, *, directory: bool) -> None:
        nonlocal injected
        if not injected and not directory:
            injected = True
            raise OSError(errno.EIO, "injected before quarantine unlink")
        real_unlinkat(parent, name, directory=directory)

    monkeypatch.setattr(host_mod, "_unlinkat_direct", no_effect_once)
    with pytest.raises(MutationHostLifecycleError, match="cleanup was incomplete"):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )

    stages = tuple(tmp_path.glob(".*.crucible-*.tmp"))
    assert injected is True
    assert len(stages) == 1
    assert not tuple(tmp_path.glob(".crucible-remove-*"))
    assert host.is_reaped is False
    host.stop()
    assert not stages[0].exists()
    assert host.is_reaped is True


def test_file_quarantine_no_effect_restore_retains_retry_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    def abort_before_publication() -> None:
        raise RuntimeError("abort before publication")

    host = _host(tmp_path, before_publish_ack=abort_before_publication)
    real_unlinkat = host_mod._unlinkat_direct
    real_rename = host_mod._rename_noreplace
    unlink_injected = False
    restore_injected = False

    def no_effect_unlink_once(parent: int, name: str, *, directory: bool) -> None:
        nonlocal unlink_injected
        if not unlink_injected and not directory:
            unlink_injected = True
            raise OSError(errno.EIO, "injected before quarantine unlink")
        real_unlinkat(parent, name, directory=directory)

    def no_effect_restore_once(parent: int, source: str, destination: str) -> None:
        nonlocal restore_injected
        if (
            not restore_injected
            and source.startswith(".crucible-remove-")
            and destination.endswith(".tmp")
        ):
            restore_injected = True
            raise OSError(errno.EIO, "injected before quarantine restore")
        real_rename(parent, source, destination)

    monkeypatch.setattr(host_mod, "_unlinkat_direct", no_effect_unlink_once)
    monkeypatch.setattr(host_mod, "_rename_noreplace", no_effect_restore_once)
    with pytest.raises(MutationHostLifecycleError, match="cleanup was incomplete"):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )

    assert unlink_injected is True
    assert restore_injected is True
    assert not tuple(tmp_path.glob(".*.crucible-*.tmp"))
    assert len(tuple(tmp_path.glob(".crucible-remove-*"))) == 1
    assert host.is_reaped is False
    host.stop()
    assert not tuple(tmp_path.glob(".crucible-remove-*"))
    assert host.is_reaped is True


def test_target_quarantine_no_effect_restore_retains_retry_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    host = _host(tmp_path)
    real_link = host_mod._link_descriptor_to_name
    real_unlinkat = host_mod._unlinkat_direct
    real_rename = host_mod._rename_noreplace
    killed = False
    unlink_injected = False
    restore_injected = False

    def link_then_kill(source: int, parent: int, name: str) -> None:
        nonlocal killed
        real_link(source, parent, name)
        if not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    def no_effect_unlink_once(parent: int, name: str, *, directory: bool) -> None:
        nonlocal unlink_injected
        if not unlink_injected and not directory:
            unlink_injected = True
            raise OSError(errno.EIO, "injected before target quarantine unlink")
        real_unlinkat(parent, name, directory=directory)

    def no_effect_restore_once(parent: int, source: str, destination: str) -> None:
        nonlocal restore_injected
        if (
            not restore_injected
            and source.startswith(".crucible-remove-")
            and destination == "target.txt"
        ):
            restore_injected = True
            raise OSError(errno.EIO, "injected before target quarantine restore")
        real_rename(parent, source, destination)

    monkeypatch.setattr(host_mod, "_link_descriptor_to_name", link_then_kill)
    monkeypatch.setattr(host_mod, "_unlinkat_direct", no_effect_unlink_once)
    monkeypatch.setattr(host_mod, "_rename_noreplace", no_effect_restore_once)
    with pytest.raises(MutationHostLifecycleError):
        host.dispatch(
            "write_source",
            {"file_path": "target.txt", "content": "new\n"},
        )

    assert unlink_injected is True
    assert restore_injected is True
    assert not (tmp_path / "target.txt").exists()
    assert len(tuple(tmp_path.glob(".crucible-remove-*"))) == 1
    assert host.is_reaped is False
    host.stop()
    assert not tuple(tmp_path.glob(".crucible-remove-*"))
    assert host.is_reaped is True


@pytest.mark.parametrize("restore_takes_effect", [False, True])
def test_directory_quarantine_no_effect_unlink_restores_exact_retry_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    restore_takes_effect: bool,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    parent_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    state = host_mod._BoundDirectoryCreation(
        parent_descriptor=parent_descriptor,
        parent_identity=host_mod._stable_directory_identity(
            os.fstat(parent_descriptor)
        ),
        expected_mount_id=host_mod._descriptor_mount_id(parent_descriptor),
        final_name="owned",
        private_name=".owned-private",
        mode=0o755,
        token="b" * 32,
    )
    state.create()
    state.publish_and_sync()
    real_unlinkat = host_mod._unlinkat_direct
    real_rename = host_mod._rename_noreplace
    unlink_injected = False
    restore_injected = False

    def no_effect_once(parent: int, name: str, *, directory: bool) -> None:
        nonlocal unlink_injected
        if not unlink_injected and directory:
            unlink_injected = True
            raise OSError(errno.EIO, "injected before quarantine rmdir")
        real_unlinkat(parent, name, directory=directory)

    def restore_then_error(parent: int, source: str, destination: str) -> None:
        nonlocal restore_injected
        if (
            not restore_injected
            and source.startswith(".crucible-remove-")
            and destination == "owned"
        ):
            restore_injected = True
            if restore_takes_effect:
                real_rename(parent, source, destination)
            raise OSError(errno.EIO, "injected after quarantine restore")
        real_rename(parent, source, destination)

    monkeypatch.setattr(host_mod, "_unlinkat_direct", no_effect_once)
    monkeypatch.setattr(host_mod, "_rename_noreplace", restore_then_error)
    with pytest.raises(MutationHostLifecycleError):
        state.rollback()

    assert unlink_injected is True
    assert restore_injected is True
    if restore_takes_effect:
        assert (tmp_path / "owned").is_dir()
        assert not tuple(tmp_path.glob(".crucible-remove-*"))
    else:
        assert not (tmp_path / "owned").exists()
        assert len(tuple(tmp_path.glob(".crucible-remove-*"))) == 1
    state.rollback()
    assert not (tmp_path / "owned").exists()
    assert not tuple(tmp_path.glob(".crucible-remove-*"))
    os.close(parent_descriptor)


def test_created_ancestor_quarantine_retains_no_effect_restore_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import _isolated_mutation_host as host_mod

    toolset = _text_authoring_toolset(tmp_path, scope=("nested",))
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=frozenset({"write_source", "apply_patch"}),
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=time.monotonic() + 10.0,
    )
    real_link = host_mod._link_descriptor_to_name
    real_unlinkat = host_mod._unlinkat_direct
    real_rename = host_mod._rename_noreplace
    killed = False
    unlink_injected = False
    restore_injected = False

    def link_then_kill(source: int, parent: int, name: str) -> None:
        nonlocal killed
        real_link(source, parent, name)
        if not killed:
            killed = True
            assert host.pid is not None
            os.killpg(host.pid, signal.SIGKILL)

    def no_effect_rmdir_once(parent: int, name: str, *, directory: bool) -> None:
        nonlocal unlink_injected
        if not unlink_injected and directory:
            unlink_injected = True
            raise OSError(errno.EIO, "injected before ancestor quarantine rmdir")
        real_unlinkat(parent, name, directory=directory)

    def no_effect_restore_once(parent: int, source: str, destination: str) -> None:
        nonlocal restore_injected
        if (
            not restore_injected
            and source.startswith(".crucible-remove-")
            and destination == "nested"
        ):
            restore_injected = True
            raise OSError(errno.EIO, "injected before ancestor quarantine restore")
        real_rename(parent, source, destination)

    monkeypatch.setattr(host_mod, "_link_descriptor_to_name", link_then_kill)
    monkeypatch.setattr(host_mod, "_unlinkat_direct", no_effect_rmdir_once)
    monkeypatch.setattr(host_mod, "_rename_noreplace", no_effect_restore_once)
    with pytest.raises(MutationHostLifecycleError, match="cleanup was incomplete"):
        host.dispatch(
            "write_source",
            {"file_path": "nested/target.txt", "content": "new\n"},
        )

    assert unlink_injected is True
    assert restore_injected is True
    assert not (tmp_path / "nested").exists()
    assert len(tuple(tmp_path.glob(".crucible-remove-*"))) == 1
    assert host.is_reaped is False
    host.stop()
    assert not tuple(tmp_path.glob(".crucible-remove-*"))
    assert host.is_reaped is True
