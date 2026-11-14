# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.swebench.repo_workspace``.

Focus: the per-repo flock that serializes the bare-clone step in
``_cached_worktree``. Without it, N parallel workers targeting the
same repo on a cold cache all pass the ``HEAD-exists`` guard before
any of them creates HEAD, then race on ``git clone --bare`` —
only one wins, the others error with ``fatal: destination path
'<repo_cache>' already exists and is not an empty directory``.

Observed in the 73-instance stratified sweep on 2026-05-24: all 4
parallel workers hit django on a cold cache simultaneously; 3 of
the 4 lost the race and shipped empty model_patch predictions.
Same shape repeated for matplotlib at the start of that repo's
batch.

These tests pin the lock contract WITHOUT actually performing a
``git clone`` — the bare-clone step is mocked to record entry
order, and a second concurrent worker is forced to wait via the
flock until the first releases.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.products.swebench import repo_workspace as _rw


@pytest.fixture
def _fake_instance():
    return SimpleNamespace(
        repo="django/django",
        repo_org="django",
        repo_name="django",
        base_commit="0" * 40,
        instance_id="django__django-1",
    )


def test_cached_worktree_flock_serializes_concurrent_clones(
    tmp_path, _fake_instance, monkeypatch,
):
    """Two threads hit ``_cached_worktree`` against the same cold
    repo cache. The flock must serialize them: the second thread
    blocks at ``_run(['git', 'clone', '--bare', ...])`` until the
    first releases, then SEES HEAD already-established under the
    lock and skips the clone (takes the prune branch instead).
    """
    clone_calls: list[str] = []
    prune_calls: list[str] = []
    in_clone = threading.Event()
    release_clone = threading.Event()

    def fake_run(cmd, *, cwd=None, timeout=600, **kw):
        """Mock ``_run`` so we don't actually invoke git."""
        if cmd[:3] == ["git", "clone", "--bare"]:
            # Simulate the clone taking time + actually creating HEAD,
            # like a real clone would. The first worker into this
            # branch holds while the test asserts the second worker
            # is blocked.
            repo_cache = Path(cmd[-1])
            in_clone.set()
            release_clone.wait(timeout=5)
            # Materialize HEAD so the lock's re-check on the second
            # worker sees an established cache.
            repo_cache.mkdir(parents=True, exist_ok=True)
            (repo_cache / "HEAD").write_text("ref: refs/heads/main\n")
            clone_calls.append(str(repo_cache))
        elif cmd[:3] == ["git", "cat-file", "-e"]:
            # Pretend the base_commit is already local so no fetch.
            pass
        elif cmd[:3] == ["git", "worktree", "add"]:
            # Materialize an empty worktree dir so the caller sees
            # ``worktree.exists()``.
            Path(cmd[-2]).mkdir(parents=True, exist_ok=True)
        # Mimic CompletedProcess for callers that inspect.
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def fake_prune(repo_cache):
        prune_calls.append(str(repo_cache))

    monkeypatch.setattr(_rw, "_run", fake_run)
    monkeypatch.setattr(_rw, "_prune_stale_repo_state", fake_prune)
    monkeypatch.setattr(
        _rw, "_materialize_isolated_checkout", lambda **_: None,
    )

    cache_dir = tmp_path / "repo_cache"

    results: list[Path | Exception] = []

    def worker():
        try:
            wt = _rw._cached_worktree(_fake_instance, cache_dir, timeout=60)
            results.append(wt)
        except Exception as e:  # noqa: BLE001
            results.append(e)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    # Wait for the first worker to enter the clone branch (under the
    # lock).
    assert in_clone.wait(timeout=5), (
        "first worker never entered the clone branch within 5s"
    )
    # Start the second worker while the first holds the lock.
    t2.start()
    # The second worker MUST be blocked at the flock acquire — give
    # it 200ms of grace, then assert no second clone has happened.
    time.sleep(0.2)
    assert len(clone_calls) == 0, (
        "second worker should be blocked at the flock — but it "
        "managed to call ``git clone --bare`` while the first "
        "worker still held the lock"
    )
    # Release the first worker; it creates HEAD and exits the lock.
    release_clone.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    # Exactly ONE clone happened. The second worker took the prune
    # branch under the re-checked lock.
    assert len(clone_calls) == 1, (
        f"expected exactly one ``git clone --bare`` call, got {len(clone_calls)}"
    )
    assert len(prune_calls) == 1, (
        f"second worker should have hit the prune branch, but didn't; "
        f"prune_calls={prune_calls}"
    )
    # Both workers succeeded.
    assert len(results) == 2
    assert all(isinstance(r, Path) for r in results), (
        f"both workers should succeed; got {results!r}"
    )


def test_recover_root_owned_leftover_invokes_chown_container(
    tmp_path, monkeypatch,
):
    """When ``shutil.rmtree(ignore_errors=True)`` silently leaves a
    worktree dir intact (because of root-owned files from a prior
    SIGKILL'd docker teardown), ``_recover_root_owned_leftover`` MUST
    spawn a throwaway docker container to chown the dir back to the
    host user, then re-attempt rmtree. Without this recovery the
    next ``git worktree add`` errors with ``already exists`` and the
    instance ships an empty prediction."""
    worktree = tmp_path / "_worktrees" / "psf__requests__psf__requests-2317"
    worktree.mkdir(parents=True)
    (worktree / "leftover.txt").write_text("orphan")

    captured_docker_cmd: list[list[str]] = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        captured_docker_cmd.append(list(cmd))
        # Simulate the chown succeeding by making the leftover
        # rmtree-able (no-op for non-root-owned test files, but the
        # contract is the chown subprocess returns 0).
        class _CP:
            returncode = 0
            stdout = ""
            stderr = ""
        return _CP()

    monkeypatch.setattr(
        "crucible.products.swebench.repo_workspace.subprocess.run",
        fake_subprocess_run,
    )

    from crucible.products.swebench.repo_workspace import (
        _recover_root_owned_leftover,
    )
    _recover_root_owned_leftover(worktree)

    # Exactly one ``docker run`` call to chown the parent path.
    assert len(captured_docker_cmd) == 1, (
        f"expected one docker call; got {captured_docker_cmd!r}"
    )
    cmd = captured_docker_cmd[0]
    assert cmd[0] == "docker" and cmd[1] == "run"
    assert "--rm" in cmd
    # Mount the worktree's PARENT, not the worktree itself — docker
    # bind-mounts a dir; chown then targets the child by name.
    bind_arg = [c for c in cmd if c.startswith("/") and ":" in c]
    assert any(str(worktree.parent) in b for b in bind_arg), (
        f"expected bind-mount of worktree's parent {worktree.parent!r}; "
        f"got {bind_arg!r}"
    )
    # chown command targets the worktree by name under /work.
    assert "chown" in cmd
    assert any(c.startswith("/work/") for c in cmd)
    # After chown, the rmtree should have cleared the worktree.
    assert not worktree.exists(), (
        f"worktree dir should be gone after recovery; still at {worktree}"
    )


def test_recover_root_owned_leftover_swallows_docker_failure(
    tmp_path, monkeypatch,
):
    """If docker isn't available or the chown subprocess fails, the
    recovery helper logs a warning and returns — it does NOT raise,
    because the subsequent ``git worktree add`` will surface a
    clear ``already exists`` error which is the right operator-
    visible diagnosis."""
    worktree = tmp_path / "_worktrees" / "psf__requests__psf__requests-2317"
    worktree.mkdir(parents=True)
    (worktree / "leftover.txt").write_text("orphan")

    def fake_subprocess_run(cmd, *args, **kwargs):
        class _CP:
            returncode = 1
            stdout = ""
            stderr = "docker: command not found"
        return _CP()

    monkeypatch.setattr(
        "crucible.products.swebench.repo_workspace.subprocess.run",
        fake_subprocess_run,
    )

    from crucible.products.swebench.repo_workspace import (
        _recover_root_owned_leftover,
    )
    # Must not raise.
    _recover_root_owned_leftover(worktree)
    # Worktree still exists since chown failed.
    assert worktree.exists()


def test_cached_worktree_lock_file_lives_outside_cache_dir(
    tmp_path, _fake_instance, monkeypatch,
):
    """The lockfile lives ALONGSIDE the cache dir, not inside it.
    A corrupt cache that gets nuked / re-created shouldn't take the
    lockfile with it — the lock invariant must survive ``rm -rf
    <repo_cache>`` + retry."""
    def fake_run(cmd, *, cwd=None, timeout=600, **kw):
        if cmd[:3] == ["git", "clone", "--bare"]:
            repo_cache = Path(cmd[-1])
            repo_cache.mkdir(parents=True, exist_ok=True)
            (repo_cache / "HEAD").write_text("ref: refs/heads/main\n")
        elif cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[-2]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(_rw, "_run", fake_run)
    monkeypatch.setattr(_rw, "_prune_stale_repo_state", lambda _: None)
    monkeypatch.setattr(
        _rw, "_materialize_isolated_checkout", lambda **_: None,
    )

    cache_dir = tmp_path / "repo_cache"
    _rw._cached_worktree(_fake_instance, cache_dir, timeout=60)

    # Lock file lives at <cache_dir>/<repo_org>/<repo_name>.clone.lock
    # NOT inside <repo_cache>/.
    lock_path = cache_dir / "django" / "django.clone.lock"
    assert lock_path.is_file(), (
        f"lock file must live at {lock_path} (alongside cache, not inside)"
    )
    # And the bare cache is at <cache_dir>/<repo_org>/<repo_name>/.
    repo_cache = cache_dir / "django" / "django"
    assert (repo_cache / "HEAD").is_file()


def test_cached_worktree_hydrates_promisor_snapshot_before_local_fetch(
    tmp_path,
):
    """A blobless bare cache must be made upload-pack complete first.

    ``git upload-pack`` deliberately disables lazy fetching in a promisor
    repository.  Reproducing that topology locally pins the live failure:
    the base commit is present, but serving a depth-one checkout fails until
    Crucible refetches that exact snapshot without the partial-clone filter.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=upstream, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Crucible Test"],
        cwd=upstream, check=True,
    )
    tracked = upstream / "module.py"
    tracked.write_text("value = 'base'\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=upstream, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"], cwd=upstream, check=True,
    )
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True,
    ).strip()
    blob = subprocess.check_output(
        ["git", "rev-parse", "HEAD:module.py"], cwd=upstream, text=True,
    ).strip()
    # The file:// upload-pack must explicitly advertise filter support or
    # Git ignores ``--filter=blob:none`` for a local test repository.
    subprocess.run(
        ["git", "config", "uploadpack.allowFilter", "true"],
        cwd=upstream, check=True,
    )

    cache_dir = tmp_path / "repo_cache"
    repo_cache = cache_dir / "django" / "django"
    repo_cache.parent.mkdir(parents=True)
    subprocess.run(
        [
            "git", "clone", "-q", "--bare", "--filter=blob:none",
            upstream.resolve().as_uri(), str(repo_cache),
        ],
        check=True,
    )
    no_lazy_env = os.environ.copy()
    no_lazy_env["GIT_NO_LAZY_FETCH"] = "1"
    assert subprocess.run(
        ["git", "cat-file", "-e", blob], cwd=repo_cache,
        capture_output=True, env=no_lazy_env,
    ).returncode != 0
    assert _rw._promisor_snapshot_incomplete(repo_cache, base_commit)

    instance = SimpleNamespace(
        repo="django/django",
        repo_org="django",
        repo_name="django",
        base_commit=base_commit,
        instance_id="django__django-promisor-regression",
    )
    with _rw.repo_workspace(
        instance, cache_dir=cache_dir, timeout=60,
    ) as checkout:
        assert (checkout / "module.py").read_text(
            encoding="utf-8",
        ) == "value = 'base'\n"
        assert subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=checkout, text=True,
        ).strip() == "1"
        assert subprocess.check_output(
            ["git", "remote"], cwd=checkout, text=True,
        ).strip() == ""

    # The repair is persistent and non-lazy: reruns can use the local cache
    # without contacting its upstream promisor remote again.
    assert subprocess.run(
        ["git", "cat-file", "-e", blob], cwd=repo_cache,
        capture_output=True, env=no_lazy_env,
    ).returncode == 0
    assert not _rw._promisor_snapshot_incomplete(repo_cache, base_commit)


def test_isolated_checkout_cannot_read_descendant_commit(tmp_path):
    """The authoring repo contains base_commit and no gold descendant."""
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=source, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Crucible Test"],
        cwd=source, check=True,
    )
    tracked = source / "module.py"
    tracked.write_text("value = 'base'\n")
    subprocess.run(["git", "add", "module.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
    ).strip()
    tracked.write_text("value = 'gold fix'\n")
    subprocess.run(["git", "commit", "-qam", "gold"], cwd=source, check=True)
    gold = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True,
    ).strip()

    checkout = tmp_path / "authoring"
    _rw._materialize_isolated_checkout(
        source=source.resolve().as_uri(),
        base_commit=base,
        dest=checkout,
        timeout=60,
    )

    assert (checkout / ".git").is_dir()
    assert subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=checkout, text=True,
    ).strip() == "1"
    assert subprocess.run(
        ["git", "cat-file", "-e", gold], cwd=checkout,
        capture_output=True,
    ).returncode != 0
    assert subprocess.check_output(
        ["git", "remote"], cwd=checkout, text=True,
    ).strip() == ""
