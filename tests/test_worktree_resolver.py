# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._worktree.WorktreeResolver``.

Each test fixtures a real tmpdir git repo so the resolver's
``git worktree add`` / ``git worktree remove`` subprocess calls
exercise real git. No mocks at the git layer — the value of this
helper is its behavior against actual git, not a contract.

Coverage (10 cases):
- Lazy creation: ``resolve_read`` before any write returns paths
  from the source repo, NO worktree is created.
- Eager creation: ``ensure_worktree`` or ``resolve_write`` triggers
  ``git worktree add``; subsequent reads route through the worktree.
- Write isolation: writes through the resolver land in the worktree,
  NOT the source repo.
- Path-traversal rejection: ``..`` escapes return None.
- ``resolve_worktree_root`` returns the worktree path (or None when
  no worktree exists yet).
- ``close()`` removes the worktree dir.
- ``close()`` is idempotent.
- ``worktree_session`` context manager creates eagerly and cleans up
  on exit.
- Exception inside ``worktree_session`` still cleans up.
- ``git worktree add`` failure surfaces a RuntimeError.
"""

from __future__ import annotations

import subprocess
import signal
import sys
import json
from pathlib import Path

import pytest


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)
    (root / "anchor.txt").write_text("original")
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=str(root), check=True)


def test_lazy_creation_no_worktree_on_cold_read(tmp_path: Path) -> None:
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    # Before any write, worktree shouldn't exist.
    assert resolver.worktree_path is None
    cand = resolver.resolve_read("anchor.txt")
    assert cand is not None
    assert cand.read_text() == "original"
    # Still no worktree.
    assert resolver.worktree_path is None


def test_eager_creation_on_first_write(tmp_path: Path) -> None:
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    try:
        target = resolver.resolve_write("anchor.txt")
        assert target is not None
        # Worktree should exist now.
        assert resolver.worktree_path is not None
        assert resolver.worktree_path.exists()
        # Target is inside the worktree, not the source repo.
        assert resolver.worktree_path in target.parents or target == resolver.worktree_path / "anchor.txt"
    finally:
        resolver.close()


def test_write_isolation(tmp_path: Path) -> None:
    """Writing through the resolver does NOT modify the source repo."""
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    try:
        target = resolver.resolve_write("anchor.txt")
        target.write_text("modified in worktree")
        # Source repo unchanged.
        assert (tmp_path / "anchor.txt").read_text() == "original"
        # Read through resolver sees the worktree's write.
        cold_read = resolver.resolve_read("anchor.txt")
        assert cold_read.read_text() == "modified in worktree"
    finally:
        resolver.close()


def test_path_traversal_rejected(tmp_path: Path) -> None:
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    try:
        resolver.ensure_worktree()
        assert resolver.resolve_write("../escape.txt") is None
        assert resolver.resolve_read("../escape.txt") is None
    finally:
        resolver.close()


def test_resolve_worktree_root_returns_path(tmp_path: Path) -> None:
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    # Pre-creation: returns None.
    assert resolver.resolve_worktree_root() is None
    try:
        resolver.ensure_worktree()
        wt = resolver.resolve_worktree_root()
        assert wt is not None
        assert wt == resolver.worktree_path
        # Project-scoped resolve: with no subdir, falls back to root.
        assert resolver.resolve_worktree_root("nonexistent") == wt
    finally:
        resolver.close()


def test_close_removes_worktree(tmp_path: Path) -> None:
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    resolver.ensure_worktree()
    wt_path = resolver.worktree_path
    assert wt_path.exists()
    resolver.close()
    assert not wt_path.exists()
    assert resolver.worktree_path is None


def test_close_is_idempotent(tmp_path: Path) -> None:
    from crucible.workflows._worktree import WorktreeResolver
    _init_git_repo(tmp_path)

    resolver = WorktreeResolver(tmp_path)
    resolver.ensure_worktree()
    resolver.close()
    # Calling close again does not raise.
    resolver.close()


def test_worktree_session_creates_and_cleans_up(tmp_path: Path) -> None:
    from crucible.workflows._worktree import worktree_session
    _init_git_repo(tmp_path)

    wt_path = None
    with worktree_session(tmp_path) as resolver:
        wt_path = resolver.worktree_path
        assert wt_path is not None
        assert wt_path.exists()
        # Mutate something to prove the worktree is writable.
        (wt_path / "anchor.txt").write_text("session write")
    # After exit, the worktree dir is gone.
    assert wt_path is not None and not wt_path.exists()
    # Source repo unchanged.
    assert (tmp_path / "anchor.txt").read_text() == "original"


def test_worktree_session_cleans_up_on_exception(tmp_path: Path) -> None:
    from crucible.workflows._worktree import worktree_session
    _init_git_repo(tmp_path)

    captured_wt = None
    with pytest.raises(RuntimeError, match="caller failure"):
        with worktree_session(tmp_path) as resolver:
            captured_wt = resolver.worktree_path
            raise RuntimeError("caller failure")
    assert captured_wt is not None
    assert not captured_wt.exists()


def test_worktree_session_unwraps_settled_setup_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _worktree as worktree_module

    _init_git_repo(tmp_path)
    owners = []

    class _SetupFailure(RuntimeError):
        pass

    def fail_after_publishing_owner(self):
        intent = tmp_path / "synthetic-worktree-intent"
        intent.mkdir()
        self._intent_path = intent
        owners.append(self)
        raise worktree_module.WorktreeCleanupError(
            owner=self,
            primary_error=_SetupFailure("setup failed"),
            cleanup_errors=(RuntimeError("first cleanup failed"),),
        )

    monkeypatch.setattr(
        worktree_module.WorktreeResolver,
        "_ensure_worktree_locked",
        fail_after_publishing_owner,
    )

    with pytest.raises(_SetupFailure, match="setup failed"):
        with worktree_module.worktree_session(tmp_path):
            pytest.fail("session unexpectedly yielded")

    assert len(owners) == 1
    assert not owners[0].owned


def test_worktree_session_retains_live_owner_when_final_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _worktree as worktree_module

    _init_git_repo(tmp_path)
    real_close_locked = worktree_module.WorktreeResolver._close_locked
    close_calls = 0

    class _SetupFailure(RuntimeError):
        pass

    def fail_after_publishing_owner(self):
        intent = tmp_path / "persistent-worktree-intent"
        intent.mkdir()
        self._intent_path = intent
        raise worktree_module.WorktreeCleanupError(
            owner=self,
            primary_error=_SetupFailure("setup failed"),
            cleanup_errors=(RuntimeError("first cleanup failed"),),
        )

    def fail_session_cleanup_once(self):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise worktree_module.WorktreeCleanupError(
                owner=self,
                primary_error=RuntimeError("worktree close"),
                cleanup_errors=(RuntimeError("second cleanup failed"),),
            )
        return real_close_locked(self)

    monkeypatch.setattr(
        worktree_module.WorktreeResolver,
        "_ensure_worktree_locked",
        fail_after_publishing_owner,
    )
    monkeypatch.setattr(
        worktree_module.WorktreeResolver,
        "_close_locked",
        fail_session_cleanup_once,
    )

    with pytest.raises(worktree_module.WorktreeCleanupError) as caught:
        with worktree_module.worktree_session(tmp_path):
            pytest.fail("session unexpectedly yielded")

    cleanup_error = caught.value
    assert isinstance(cleanup_error.primary_error, _SetupFailure)
    assert cleanup_error.owner.owned
    assert len(cleanup_error.cleanup_errors) == 2

    cleanup_error.retry_cleanup()
    assert close_calls == 2
    assert not cleanup_error.owner.owned


def test_worktree_session_wraps_fatal_exit_cleanup_with_live_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _worktree as worktree_module

    _init_git_repo(tmp_path)
    real_close_locked = worktree_module.WorktreeResolver._close_locked
    close_calls = 0

    class _FatalCleanup(BaseException):
        pass

    def fail_exit_cleanup_once(self):
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise _FatalCleanup("fatal worktree exit cleanup")
        return real_close_locked(self)

    monkeypatch.setattr(
        worktree_module.WorktreeResolver,
        "_close_locked",
        fail_exit_cleanup_once,
    )

    with pytest.raises(worktree_module.WorktreeCleanupError) as caught:
        with worktree_module.worktree_session(tmp_path):
            pass

    cleanup_error = caught.value
    assert isinstance(cleanup_error.cleanup_errors[0], _FatalCleanup)
    assert cleanup_error.owner.owned
    cleanup_error.retry_cleanup()
    assert close_calls == 2
    assert not cleanup_error.owner.owned


def test_git_worktree_add_failure_raises(tmp_path: Path) -> None:
    """A non-git directory can't host a worktree; ensure_worktree
    surfaces the failure rather than silently producing a phantom path."""
    from crucible.workflows._worktree import WorktreeResolver

    # tmp_path is NOT a git repo here.
    resolver = WorktreeResolver(tmp_path)
    with pytest.raises(RuntimeError, match="git worktree add failed"):
        resolver.ensure_worktree()
    # No phantom worktree path retained.
    assert resolver.worktree_path is None


def test_sigkill_orphan_is_reaped_from_durable_lease(tmp_path: Path) -> None:
    """A real process death leaves a lease the next owner can reconcile."""

    from crucible.workflows._worktree import reap_orphaned_worktree_leases

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    marker = tmp_path / "worktree-path.txt"
    script = "\n".join((
        "import os, signal, sys",
        "from pathlib import Path",
        "from crucible.workflows._worktree import WorktreeResolver",
        "resolver = WorktreeResolver(Path(sys.argv[1]))",
        "worktree = resolver.ensure_worktree()",
        "Path(sys.argv[2]).write_text(str(worktree), encoding='utf-8')",
        "os.kill(os.getpid(), signal.SIGKILL)",
    ))
    child = subprocess.run(
        [sys.executable, "-c", script, str(repo), str(marker)],
        check=False,
        timeout=30,
    )
    assert child.returncode == -signal.SIGKILL
    orphan = Path(marker.read_text(encoding="utf-8"))
    assert orphan.is_dir()

    assert reap_orphaned_worktree_leases(repo) == (orphan,)
    assert not orphan.exists()


def test_close_preserves_worktree_when_git_remove_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _worktree

    _init_git_repo(tmp_path)
    resolver = _worktree.WorktreeResolver(tmp_path)
    worktree = resolver.ensure_worktree()
    lease = resolver._lease_path
    real_run = _worktree.subprocess.run

    def fail_remove(command, *args, **kwargs):
        if "worktree" in command and "remove" in command:
            return subprocess.CompletedProcess(command, 1, b"", b"busy")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(_worktree.subprocess, "run", fail_remove)
    with pytest.raises(RuntimeError, match="could not remove"):
        resolver.close()
    assert worktree.is_dir()
    assert lease is not None and lease.is_file()
    assert resolver.worktree_path == worktree


def test_reaper_never_deletes_unvalidated_external_lease_target(
    tmp_path: Path,
) -> None:
    from crucible.workflows._worktree import (
        _git_common_dir,
        reap_orphaned_worktree_leases,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "evidence.txt").write_text("preserve\n")
    lease_root = _git_common_dir(repo) / "crucible-worktree-leases"
    lease_root.mkdir()
    lease = lease_root / "forged.json"
    lease.write_text(json.dumps({
        "schema": "crucible.worktree-lease.v2",
        "state": "created",
        "repo_root": str(repo),
        "worktree": str(victim),
        "pid": 999_999_999,
        "process_start": "forged",
    }))

    assert reap_orphaned_worktree_leases(repo) == ()
    assert (victim / "evidence.txt").read_text() == "preserve\n"
    assert lease.is_file()
