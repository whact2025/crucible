# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cross-worktree serialization for workspace mutation transactions."""

from __future__ import annotations

import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

from crucible.workflows import _mutation_journal as mutation_journal
from crucible.workflows._mutation_journal import (
    WorkspaceMutationJournal,
    workspace_mutation_lock,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "mutation-lock@test.invalid")
    _git(path, "config", "user.name", "mutation lock test")
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "base")


def _begin_journal(tmp_path: Path, name: str) -> WorkspaceMutationJournal:
    repo = tmp_path / f"repo-{name}"
    _init_repo(repo)
    return WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=tmp_path / f"journals-{name}",
        slice_id=name,
        scope=("tracked.txt",),
        sequence=1,
    )


def _record_fsck_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    original_git_fsck = mutation_journal._git_fsck
    calls: list[tuple[str, ...]] = []

    def recording_git_fsck(
        cwd: Path,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return original_git_fsck(cwd, *args, check=check)

    monkeypatch.setattr(mutation_journal, "_git_fsck", recording_git_fsck)
    return calls


def test_clean_rollback_skips_full_fsck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = _begin_journal(tmp_path, "clean-fsck-skip")
    calls = _record_fsck_calls(monkeypatch)

    assert journal.restore()
    assert calls == []


@pytest.mark.parametrize("reason", ["controls", "objects", "refs", "reset"])
def test_integrity_drift_class_requests_full_fsck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    journal = _begin_journal(tmp_path, reason)
    calls = _record_fsck_calls(monkeypatch)
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "_initial_fsck_reasons",
        lambda self, cwd: {reason},
    )

    assert journal.restore()
    assert calls == [("--full", "--no-dangling", "HEAD")]


def test_linked_worktrees_share_one_cross_process_mutation_lock(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    _init_repo(repo)
    _git(repo, "worktree", "add", "-q", "-b", "linked-test", str(linked), "HEAD")

    assert _git(repo, "rev-parse", "--git-common-dir") != _git(
        linked, "rev-parse", "--git-dir",
    )
    assert Path(
        _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve() == Path(
        _git(linked, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve()

    child = """
import sys
from crucible.workflows._mutation_journal import workspace_mutation_lock

print("attempting", flush=True)
with workspace_mutation_lock(sys.argv[1]):
    print("entered", flush=True)
"""
    process: subprocess.Popen[str] | None = None
    try:
        with workspace_mutation_lock(repo):
            process = subprocess.Popen(
                [sys.executable, "-c", child, str(linked)],
                cwd=repo,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "attempting"
            readable, _, _ = select.select([process.stdout], [], [], 0.3)
            assert readable == [], (
                "linked worktree acquired its mutation lock while the "
                "common Git directory was already locked"
            )

        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "entered"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_initialized_submodule_and_superproject_share_mutation_lock(
    tmp_path: Path,
) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    repo = tmp_path / "superproject "
    _init_repo(repo)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(origin),
        "vendor/nested",
    )
    _git(repo, "commit", "-qam", "add nested")
    nested = repo / "vendor" / "nested"

    child = """
import sys
from crucible.workflows._mutation_journal import workspace_mutation_lock

print("attempting", flush=True)
with workspace_mutation_lock(sys.argv[1]):
    print("entered", flush=True)
"""

    def assert_blocks(held: Path, attempted: Path) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            with workspace_mutation_lock(held):
                process = subprocess.Popen(
                    [sys.executable, "-c", child, str(attempted)],
                    cwd=repo,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert process.stdout is not None
                assert process.stdout.readline().strip() == "attempting"
                readable, _, _ = select.select([process.stdout], [], [], 0.3)
                assert readable == []
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            assert stdout.strip() == "entered"
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    assert_blocks(repo, nested)
    assert_blocks(nested, repo)
    with workspace_mutation_lock(repo):
        with workspace_mutation_lock(nested):
            pass


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_drops_inherited_lock_and_live_transaction_state(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "fork-repo"
    _init_repo(repo)
    script = r'''
import os
import sys
from crucible.workflows._mutation_journal import (
    _is_live,
    _register_live,
    workspace_mutation_lock,
)

with workspace_mutation_lock(sys.argv[1]):
    _register_live("inherited-parent-transaction")
    pid = os.fork()
    if pid == 0:
        if _is_live("inherited-parent-transaction"):
            print("inherited-live-state", flush=True)
            os._exit(7)
        with workspace_mutation_lock(sys.argv[1]):
            print("child-entered", flush=True)
        os._exit(0)
    os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(repo)],
        cwd=repo,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "child-entered"
