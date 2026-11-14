# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hostile Git-admin rollback authority regressions."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.workflows import _mutation_journal as journals


_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent/git-admin-adversarial-test",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
_GIT_CONFIG = (
    "-c",
    "maintenance.auto=false",
    "-c",
    "gc.auto=0",
)


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", *_GIT_CONFIG, *args],
        cwd=cwd,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def test_fixture_git_commands_cannot_detach_automatic_maintenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=b"stable\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _git(tmp_path, "status", "--porcelain") == b"stable\n"
    assert observed["argv"] == [
        "/usr/bin/git",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
        "status",
        "--porcelain",
    ]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["env"] == _GIT_ENV
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def _init_repository(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "git-admin@test.invalid")
    _git(repo, "config", "user.name", "Git Admin Test")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "baseline")


def _begin(repo: Path, tmp_path: Path) -> journals.WorkspaceMutationJournal:
    return journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "hostile" / "mutations",
        slice_id="git-admin-hostile",
        scope=("tracked.txt",),
        sequence=1,
    )


def _control(
    journal: journals.WorkspaceMutationJournal,
    name: str,
) -> dict[str, object]:
    return next(
        record
        for record in journal._manifest["control_files"]
        if record["name"] == name
    )


@pytest.mark.parametrize("name", ["config", "index.lock"])
def test_git_admin_file_restore_rejects_parked_directory(
    tmp_path: Path,
    name: str,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = _begin(repo, tmp_path)
    if name == "index.lock":
        record = _control(journal, name)
        assert record["exists"] is False
        payload = None
    else:
        record = _control(journal, name)
        assert record["exists"] is True
        payload = journal._read_snapshot_blob_record(
            record,
            label=f"test control {name}",
        )

    target = Path(str(record["path"]))
    parked = tmp_path / f"parked-{name.replace('.', '-')}"
    with journals.workspace_mutation_lock(repo):
        if target.exists():
            target.rename(parked)
        target.mkdir()
        sentinel = target / "outside.txt"
        sentinel.write_text("outside authority\n", encoding="utf-8")
        with pytest.raises(
            journals.MutationJournalError,
            match=(
                "real regular file|recursively .*unauthenticated "
                "Git-admin directory"
            ),
        ):
            journal._restore_bound_git_admin_file(
                target,
                exists=bool(record["exists"]),
                record=record,
                payload=payload,
                label=f"test control {name}",
            )
        assert sentinel.read_text(encoding="utf-8") == "outside authority\n"
    shutil.rmtree(target)
    if parked.exists():
        parked.rename(target)
    assert journal.restore()


def test_git_admin_topology_rejects_directory_addition_before_removal(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = _begin(repo, tmp_path)
    admin = Path(str(journal._manifest["git_common_dir"]))
    addition = admin / "externally-parked"
    addition.mkdir()
    sentinel = addition / "outside.txt"
    sentinel.write_text("outside authority\n", encoding="utf-8")

    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="topology addition",
        ):
            journal._restore_git_admin_topology()

    assert sentinel.read_text(encoding="utf-8") == "outside authority\n"
    shutil.rmtree(addition)
    assert journal.restore()


def test_absent_git_logs_parent_remains_absent_on_restore(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    _git(repo, "config", "core.logAllRefUpdates", "false")
    logs = repo / ".git" / "logs"
    shutil.rmtree(logs)
    journal = _begin(repo, tmp_path)

    assert journal._manifest["log_refs_exists"] is False
    assert not logs.exists()

    assert journal.restore()
    assert not logs.exists()


def test_absent_git_logs_parent_recreation_is_rejected_without_removal(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    _git(repo, "config", "core.logAllRefUpdates", "false")
    logs = repo / ".git" / "logs"
    shutil.rmtree(logs)
    journal = _begin(repo, tmp_path)

    logs.mkdir()
    sentinel = logs / "outside.txt"
    sentinel.write_text("outside authority\n", encoding="utf-8")

    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="lacks captured authority",
        ):
            journal._restore_log_refs_tree(repo)
    assert not journal.restore()
    assert sentinel.read_text(encoding="utf-8") == "outside authority\n"
    assert journal.journal_dir.exists()


def test_absent_git_logs_parent_appearance_between_probes_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    _git(repo, "config", "core.logAllRefUpdates", "false")
    logs = repo / ".git" / "logs"
    shutil.rmtree(logs)
    journal = _begin(repo, tmp_path)
    original_open = journals._open_settlement_parent_if_present
    target_calls = 0
    sentinel = logs / "outside.txt"

    @contextmanager
    def create_parent_on_second_probe(
        root_fd: int,
        relative: str,
        *,
        allow_terminal_vcs_metadata: bool = False,
    ):
        nonlocal target_calls
        if relative == "logs/refs":
            target_calls += 1
            if target_calls == 2:
                logs.mkdir()
                sentinel.write_text("outside authority\n", encoding="utf-8")
        with original_open(
            root_fd,
            relative,
            allow_terminal_vcs_metadata=allow_terminal_vcs_metadata,
        ) as binding:
            yield binding

    monkeypatch.setattr(
        journals,
        "_open_settlement_parent_if_present",
        create_parent_on_second_probe,
    )

    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="Git-admin tree absence changed",
        ):
            journal._restore_log_refs_tree(repo)
    assert target_calls == 2
    assert sentinel.read_text(encoding="utf-8") == "outside authority\n"
    assert journal.journal_dir.exists()


def test_guarded_blob_is_authenticated_before_topology_removal(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    custom = repo / ".git" / "custom-guarded"
    custom.mkdir()
    baseline_payload = b"guarded baseline\n"
    (custom / "nested.dat").write_bytes(baseline_payload)
    journal = _begin(repo, tmp_path)

    root_record = next(
        record
        for record in journal._manifest["git_admin_topology"]
        if Path(str(record["root"])) == repo / ".git"
    )
    guarded = root_record["entries"]["custom-guarded"]
    nested = guarded["entries"]["nested.dat"]
    blob = journal.journal_dir / "blobs" / str(nested["sha256"])
    blob.write_bytes(b"poisoned guarded bytes\n")
    removable = repo / ".git" / "provider-addition"
    removable.write_bytes(b"remove only after complete preflight\n")

    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="snapshot blob",
        ):
            journal._restore_git_admin_topology()

    assert removable.read_bytes() == b"remove only after complete preflight\n"
    blob.write_bytes(baseline_payload)
    removable.unlink()
    assert journal.restore()


def test_shared_common_dir_authority_requires_captured_inode(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    lane = tmp_path / "lane"
    _init_repository(repo)
    _git(repo, "worktree", "add", "-q", "-b", "lane", str(lane))
    journal = _begin(lane, tmp_path)

    common = Path(str(journal._manifest["git_common_dir"]))
    parked = tmp_path / "parked-common"
    common.rename(parked)
    shutil.copytree(parked, common, symlinks=True)
    replacement = common.lstat()
    captured = next(
        record
        for record in journal._manifest["git_admin_topology"]
        if Path(str(record["root"])) == common
    )
    assert (replacement.st_dev, replacement.st_ino) != (
        captured["device"],
        captured["inode"],
    )

    try:
        with journals.workspace_mutation_lock(repo):
            with pytest.raises(
                journals.MutationJournalError,
                match="outside held authority",
            ):
                with journal._bound_git_admin_path(common / "config"):
                    pytest.fail("replacement common-dir descriptor was admitted")
    finally:
        shutil.rmtree(common)
        parked.rename(common)

    assert journal.restore()


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_git_entry_xattrs_restore_exactly(
    tmp_path: Path,
    linked_worktree: bool,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    worktree = repo
    if linked_worktree:
        worktree = tmp_path / "lane"
        _git(repo, "worktree", "add", "-q", "-b", "lane", str(worktree))
    git_entry = worktree / ".git"
    name = "user.crucible-git-entry"
    try:
        os.setxattr(git_entry, name, b"baseline")
    except (AttributeError, OSError) as exc:
        pytest.skip(f"filesystem does not support user xattrs: {exc}")
    journal = _begin(worktree, tmp_path)

    os.setxattr(git_entry, name, b"mutated")

    assert journal.restore()
    assert os.getxattr(git_entry, name) == b"baseline"


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_terminal_git_authority_rejects_git_entry_xattr_drift(
    tmp_path: Path,
    linked_worktree: bool,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    worktree = repo
    if linked_worktree:
        worktree = tmp_path / "lane"
        _git(repo, "worktree", "add", "-q", "-b", "lane", str(worktree))
    git_entry = worktree / ".git"
    name = "user.crucible-terminal-git-entry"
    try:
        os.setxattr(git_entry, name, b"baseline")
    except (AttributeError, OSError) as exc:
        pytest.skip(f"filesystem does not support user xattrs: {exc}")
    journal = _begin(worktree, tmp_path)
    expected = journal._manifest["git_entry"]
    proof_digests = {
        "git_entry": (
            expected.get("sha256") if expected["kind"] == "file" else None
        ),
    }

    os.setxattr(git_entry, name, b"mutated")

    with pytest.raises(
        journals.MutationJournalError,
        match="Git (administration|indirection) entry changed",
    ):
        journals._terminal_assert_git_authority(
            journal,
            SimpleNamespace(proof_digests=proof_digests),
        )

    os.setxattr(git_entry, name, b"baseline")
    assert journal.restore()


@pytest.mark.parametrize(
    "fault",
    ["device", "uid", "gid", "xattr", "semantic-race"],
)
def test_terminal_git_authority_rejects_index_metadata_drift(
    tmp_path: Path,
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = _begin(repo, tmp_path)
    index = Path(str(journal._manifest["index_path"]))
    proof_digests = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in zip(
            ("index_stage", "index_flags_v", "index_flags_f"),
            journals._index_semantic_views(repo, index),
            strict=True,
        )
    }
    git_entry = journal._manifest["git_entry"]
    proof_digests["git_entry"] = (
        git_entry.get("sha256") if git_entry["kind"] == "file" else None
    )
    field = f"index_{fault}"
    baseline: object | None = None
    xattr_name = "user.crucible-terminal-index"
    if fault == "semantic-race":
        original_views = journals._index_semantic_views

        def racing_views(cwd: Path, current_index: Path):
            views = original_views(cwd, current_index)
            replacement = index.with_name("terminal-index-race")
            shutil.copy2(index, replacement)
            os.replace(replacement, index)
            return views

        monkeypatch.setattr(journals, "_index_semantic_views", racing_views)
    elif fault == "xattr":
        try:
            os.setxattr(index, xattr_name, b"mutated")
        except (AttributeError, OSError) as exc:
            pytest.skip(f"filesystem does not support user xattrs: {exc}")
    else:
        baseline = journal._manifest[field]
        journal._manifest[field] = int(baseline) + 1

    with pytest.raises(
        journals.MutationJournalError,
        match="terminal cleanup Git index changed",
    ):
        journals._terminal_assert_git_authority(
            journal,
            SimpleNamespace(proof_digests=proof_digests),
        )

    if fault == "semantic-race":
        monkeypatch.undo()
    elif fault == "xattr":
        os.removexattr(index, xattr_name)
    elif fault != "semantic-race":
        journal._manifest[field] = baseline
    assert journal.restore()


@pytest.mark.parametrize("replacement", [False, True])
def test_cached_module_opaque_root_requires_captured_inode(
    tmp_path: Path,
    replacement: bool,
) -> None:
    module_source = tmp_path / "module-source"
    _init_repository(module_source)
    repo = tmp_path / "repo"
    _init_repository(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(module_source),
        "module",
    )
    _git(repo, "commit", "-qam", "add module")
    _git(repo, "submodule", "deinit", "-q", "-f", "module")
    journal = _begin(repo, tmp_path)
    cached_root = repo / ".git" / "modules" / "module"
    parked = tmp_path / "parked-module-cache"
    cached_root.rename(parked)
    sentinel: Path | None = None
    if replacement:
        shutil.copytree(parked, cached_root, symlinks=True)
        sentinel = cached_root / "outside-sentinel"
        sentinel.write_text("outside authority\n", encoding="utf-8")

    assert journal.restore() is False
    if sentinel is not None:
        assert sentinel.read_text(encoding="utf-8") == "outside authority\n"

    if cached_root.exists():
        shutil.rmtree(cached_root)
    parked.rename(cached_root)
    assert journal.restore()


def test_worktree_namespace_cleanup_never_uses_path_recursive_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    lane = tmp_path / "lane"
    _init_repository(repo)
    _git(repo, "worktree", "add", "-q", "-b", "lane", str(lane))
    journal = _begin(repo, tmp_path)
    worktrees_root = repo / ".git" / "worktrees"
    addition = worktrees_root / "evil"
    addition.mkdir()
    (addition / "provider.txt").write_text("provider\n", encoding="utf-8")
    external_parent = tmp_path / "external-parent"
    external_victim = external_parent / "evil"
    external_victim.mkdir(parents=True)
    sentinel = external_victim / "sentinel"
    sentinel.write_text("outside authority\n", encoding="utf-8")
    parked = tmp_path / "parked-worktrees"
    original_rmtree = shutil.rmtree
    injected = False

    def redirecting_rmtree(path, *args, **kwargs):
        nonlocal injected
        if Path(path) == addition and not injected:
            injected = True
            worktrees_root.rename(parked)
            worktrees_root.symlink_to(external_parent, target_is_directory=True)
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", redirecting_rmtree)
    with journals.workspace_mutation_lock(repo):
        journal._restore_worktree_namespace(repo)

    assert injected is False
    assert sentinel.read_text(encoding="utf-8") == "outside authority\n"
    assert not addition.exists()
    assert journal.restore()


def test_attribute_policy_restore_never_follows_replaced_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    policy_parent = repo / "sub"
    policy_parent.mkdir()
    policy = policy_parent / ".gitattributes"
    baseline = b"*.bin binary\n"
    policy.write_bytes(baseline)
    _git(repo, "add", "sub/.gitattributes")
    _git(repo, "commit", "-qm", "attribute policy")
    journal = _begin(repo, tmp_path)
    policy.write_bytes(b"*.bin -binary\n")
    external_parent = tmp_path / "external-policy"
    external_parent.mkdir()
    external_policy = external_parent / ".gitattributes"
    external_payload = b"outside authority\n"
    external_policy.write_bytes(external_payload)
    parked = tmp_path / "parked-policy-parent"
    original_unlink = Path.unlink
    injected = False

    def redirecting_unlink(path: Path, *args, **kwargs):
        nonlocal injected
        if path == policy and not injected:
            injected = True
            policy_parent.rename(parked)
            policy_parent.symlink_to(external_parent, target_is_directory=True)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", redirecting_unlink)
    with journals.workspace_mutation_lock(repo):
        changed = journal._restore_attribute_policy(repo)

    assert changed == ("sub/.gitattributes",)
    assert injected is False
    assert external_policy.read_bytes() == external_payload
    assert policy.read_bytes() == baseline
    assert journal.restore()
