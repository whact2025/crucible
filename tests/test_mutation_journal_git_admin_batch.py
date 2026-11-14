# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded Git-discovery proofs for administrative-tree validation."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from crucible.workflows import _mutation_journal as journals


_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent/git-admin-batch-test",
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
    _git(repo, "config", "user.email", "git-admin-batch@test.invalid")
    _git(repo, "config", "user.name", "Git Admin Batch Test")
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "baseline")


def _begin(
    repo: Path,
    tmp_path: Path,
) -> journals.WorkspaceMutationJournal:
    return journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / "corpus" / "runs" / "batch" / "mutations",
        slice_id="git-admin-batch",
        scope=("tracked.txt",),
        sequence=1,
    )


def _refs_authority(
    journal: journals.WorkspaceMutationJournal,
) -> tuple[Path, dict[str, object], dict[str, dict[str, object]]]:
    manifest = journal._manifest
    refs = Path(str(manifest["refs_path"]))
    record: dict[str, object] = {
        "mode": manifest["refs_root_mode"],
        "device": manifest["refs_root_device"],
        "inode": manifest["refs_root_inode"],
        "xattrs": manifest.get("refs_root_xattrs", {}),
    }
    return refs, record, manifest["refs_tree"]


@pytest.mark.parametrize("linked_worktree", [False, True])
def test_admin_tree_validation_git_discovery_is_constant_per_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_worktree: bool,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    for index in range(64):
        _git(repo, "branch", f"proof-{index:02d}")
    worktree = repo
    if linked_worktree:
        worktree = tmp_path / "lane"
        _git(repo, "worktree", "add", "-q", "-b", "lane", str(worktree))
    journal = _begin(worktree, tmp_path)
    refs, record, entries = _refs_authority(journal)
    assert len(entries) >= 65

    calls: list[tuple[str, ...]] = []
    original_git = journals._git

    def counting_git(cwd: Path, *args: str, **kwargs) -> bytes:
        calls.append(args)
        return original_git(cwd, *args, **kwargs)

    monkeypatch.setattr(journals, "_git", counting_git)
    with journals.workspace_mutation_lock(worktree):
        calls.clear()
        assert journal._bound_git_admin_tree_matches(
            refs,
            record=record,
            entries=entries,
            label="large refs proof",
        )

    # One pre-proof and one successful terminal proof.  Each performs the
    # checkout/superproject and common-dir discovery commands, independently
    # of the number of entries validated between them.
    assert len(calls) == 4
    assert all(call[0] == "rev-parse" for call in calls)


def test_admin_tree_batch_rejects_forged_cross_context_capabilities(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = _begin(repo, tmp_path)
    config = next(
        Path(str(record["path"]))
        for record in journal._manifest["control_files"]
        if record["name"] == "config"
    )
    clone = journals.WorkspaceMutationJournal(
        journal.journal_dir,
        dict(journal._manifest),
    )

    with journals.workspace_mutation_lock(repo):
        with journal._git_admin_authority_batch() as batch:
            journal._inspect_bound_git_admin_entry(
                config,
                label="valid batch entry",
                authority_batch=batch,
            )
            forged = (
                replace(batch, journal_token=object()),
                replace(batch, transaction_id="0" * 32),
                replace(batch, owner_pid=os.getpid() + 1),
                replace(batch, owner_thread_id=threading.get_ident() + 1),
            )
            for candidate in forged:
                with pytest.raises(
                    journals.MutationJournalError,
                    match="stale or belongs to another journal, process, or thread",
                ):
                    journal._inspect_bound_git_admin_entry(
                        config,
                        label="forged batch entry",
                        authority_batch=candidate,
                    )
            with pytest.raises(
                journals.MutationJournalError,
                match="stale or belongs to another journal, process, or thread",
            ):
                clone._inspect_bound_git_admin_entry(
                    config,
                    label="cross-journal batch entry",
                    authority_batch=batch,
                )

            failures: list[BaseException] = []

            def cross_thread_use() -> None:
                try:
                    journal._inspect_bound_git_admin_entry(
                        config,
                        label="cross-thread batch entry",
                        authority_batch=batch,
                    )
                except BaseException as exc:  # test captures thread failure
                    failures.append(exc)

            thread = threading.Thread(target=cross_thread_use)
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
            assert len(failures) == 1
            assert isinstance(failures[0], journals.MutationJournalError)
            assert "another journal, process, or thread" in str(failures[0])

            read_fd, write_fd = os.pipe()
            child_pid = os.fork()
            if child_pid == 0:  # pragma: no branch - child has its own exit
                os.close(read_fd)
                try:
                    journal._inspect_bound_git_admin_entry(
                        config,
                        label="cross-process batch entry",
                        authority_batch=batch,
                    )
                except journals.MutationJournalError as exc:
                    os.write(write_fd, f"rejected:{exc}".encode())
                    os._exit(0)
                os.write(write_fd, b"accepted")
                os._exit(1)
            os.close(write_fd)
            child_result = os.read(read_fd, 4096)
            os.close(read_fd)
            waited_pid, child_status = os.waitpid(child_pid, 0)
            assert waited_pid == child_pid
            assert os.WIFEXITED(child_status)
            assert os.WEXITSTATUS(child_status) == 0
            assert child_result.startswith(b"rejected:")
            assert b"another journal, process, or thread" in child_result

        with pytest.raises(
            journals.MutationJournalError,
            match="stale or belongs to another journal, process, or thread",
        ):
            journal._inspect_bound_git_admin_entry(
                config,
                label="expired batch entry",
                authority_batch=batch,
            )

    assert journal.restore()


def test_admin_tree_batch_terminal_proof_rejects_git_dir_retarget(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repository(repo)
    journal = _begin(repo, tmp_path)
    config = repo / ".git" / "config"
    parked = tmp_path / "parked-git-dir"
    inspection_completed = False

    with journals.workspace_mutation_lock(repo):
        try:
            with pytest.raises(
                journals.MutationJournalError,
                match="workspace (Git dir|authority common dir) changed",
            ):
                with journal._git_admin_authority_batch() as batch:
                    (repo / ".git").rename(parked)
                    shutil.copytree(parked, repo / ".git", symlinks=True)
                    # The held descriptor remains safely confined to the old
                    # inode; the terminal full proof must still reject the
                    # pathname retarget before accepting a clean tree result.
                    journal._inspect_bound_git_admin_entry(
                        config,
                        label="retargeted Git dir",
                        authority_batch=batch,
                    )
                    inspection_completed = True
        finally:
            shutil.rmtree(repo / ".git")
            parked.rename(repo / ".git")

    assert inspection_completed
    assert journal.restore()
