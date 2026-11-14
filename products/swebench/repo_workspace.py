# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-instance git working tree at the SWE-bench instance's base_commit.

The adapter needs a clean working tree at each instance's pinned
commit so ``assay.implement`` can run against the repo state the
agent is supposed to see. This module provides ``repo_workspace``,
a context manager that clones the instance's repo at base_commit
into a temp dir (or reuses a per-repo cache) and yields the path.

Two acquisition modes, with the same authoring boundary:

- **Ephemeral**: fetch exactly ``base_commit`` at depth one into a
  fresh repository.
- **Cached**: keep a full acquisition cache outside the authoring
  sandbox, then fetch exactly ``base_commit`` at depth one into a
  fresh repository.

The yielded checkout is always a standalone, single-commit shallow
repository with no remotes, alternates, worktree pointer, or shared
object database. This is a benchmark-integrity boundary: an agent
must not recover a post-base fix from descendant commits or a shared
cache. The acquisition cache is never mounted into the authoring
container.

GitHub URL is derived from the instance's ``repo`` field
(``<org>/<name>`` → ``https://github.com/<org>/<name>.git``).

Pure subprocess; no GitHub API, no auth. Public repos only.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .dataset import SwebenchInstance

logger = logging.getLogger(__name__)


# Transient network errors that warrant a retry. Conservative list —
# don't retry on auth failures, repo-not-found, or invalid revisions
# (those are spec contract violations, not transients).
_TRANSIENT_STDERR_SIGNALS: tuple[bytes, ...] = (
    b"Could not resolve host",
    b"unable to access",
    b"Connection reset by peer",
    b"Connection timed out",
    b"early EOF",
    b"the remote end hung up",
    b"server certificate verification failed",
    b"HTTP 5",  # 5xx responses
)


def _is_transient_error(stderr: bytes) -> bool:
    return any(sig in stderr for sig in _TRANSIENT_STDERR_SIGNALS)


class RepoWorkspaceError(RuntimeError):
    """Raised when acquisition, fetch, or isolated checkout fails."""


def _github_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 600,
    max_attempts: int = 3,
    retry_on_transient: bool = True,
) -> subprocess.CompletedProcess:
    """Subprocess wrapper that raises RepoWorkspaceError on non-zero exit.

    Retries up to ``max_attempts`` times on transient network errors
    (detected via ``_TRANSIENT_STDERR_SIGNALS``) with exponential
    backoff (1s, 2s, 4s, ...). 500 GitHub clones × 1% transient rate
    ≈ 5 instances lost without retry; with retry, near-zero.
    """
    if cmd and cmd[0] == "git":
        cmd = ["git", "-c", "core.hooksPath=/dev/null", *cmd[1:]]
    attempts = max(1, max_attempts) if retry_on_transient else 1
    last_stderr = b""
    last_rc = 0
    for attempt in range(attempts):
        result = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            capture_output=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result
        last_stderr = result.stderr or b""
        last_rc = result.returncode
        if not retry_on_transient or not _is_transient_error(last_stderr):
            break
        if attempt + 1 < attempts:
            backoff = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(
                "swebench: transient git failure on attempt %d/%d "
                "(rc=%d); retrying in %ds",
                attempt + 1, attempts, last_rc, backoff,
            )
            time.sleep(backoff)
    stderr_tail = last_stderr[-2048:].decode("utf-8", errors="replace")
    raise RepoWorkspaceError(
        f"command {cmd!r} exited {last_rc} after {attempts} attempt(s): "
        f"{stderr_tail}"
    )


def _ephemeral_clone(
    instance: SwebenchInstance, dest: Path, timeout: int,
) -> None:
    """Fetch one commit into an isolated authoring repository."""
    _materialize_isolated_checkout(
        source=_github_url(instance.repo),
        base_commit=instance.base_commit,
        dest=dest,
        timeout=timeout,
    )


def _materialize_isolated_checkout(
    *, source: str, base_commit: str, dest: Path, timeout: int,
) -> None:
    """Create a remote-free, single-commit checkout at ``dest``.

    Fetching an exact object with ``--depth=1 --no-tags`` preserves the
    dataset's base commit hash while making its parents and every
    descendant commit unavailable. Fetching by URL rather than adding a
    remote also avoids leaving a network endpoint in the sandbox.
    """
    dest.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "init", "--quiet", str(dest)],
        timeout=timeout,
        retry_on_transient=False,
    )
    _run(
        [
            "git", "fetch", "--depth=1", "--no-tags", source,
            base_commit,
        ],
        cwd=dest,
        timeout=timeout,
    )
    _run(
        ["git", "checkout", "--detach", "FETCH_HEAD"],
        cwd=dest,
        timeout=timeout,
        retry_on_transient=False,
    )
    try:
        (dest / ".git" / "FETCH_HEAD").unlink()
    except FileNotFoundError:
        pass
    _assert_isolated_checkout(dest, base_commit)


def _assert_isolated_checkout(dest: Path, base_commit: str) -> None:
    """Fail closed unless ``dest`` satisfies the authoring boundary."""
    git_dir = dest / ".git"
    if not git_dir.is_dir():
        raise RepoWorkspaceError(
            f"isolated checkout {dest} has no standalone .git directory"
        )
    if (git_dir / "objects" / "info" / "alternates").exists():
        raise RepoWorkspaceError(
            f"isolated checkout {dest} unexpectedly uses object alternates"
        )
    head = _run(
        ["git", "rev-parse", "HEAD"], cwd=dest, timeout=30,
        retry_on_transient=False,
    ).stdout.decode("ascii", errors="replace").strip()
    if head != base_commit:
        raise RepoWorkspaceError(
            f"isolated checkout HEAD {head!r} != base_commit {base_commit!r}"
        )
    count_text = _run(
        ["git", "rev-list", "--count", "HEAD"], cwd=dest, timeout=30,
        retry_on_transient=False,
    ).stdout.decode("ascii", errors="replace").strip()
    if count_text != "1":
        raise RepoWorkspaceError(
            f"isolated checkout {dest} exposes {count_text!r} commits; "
            "expected exactly one"
        )
    remotes = _run(
        ["git", "remote"], cwd=dest, timeout=30,
        retry_on_transient=False,
    ).stdout.strip()
    if remotes:
        raise RepoWorkspaceError(
            f"isolated checkout {dest} unexpectedly exposes remotes"
        )


def _prune_stale_repo_state(repo_cache: Path) -> None:
    """Clean up the bare cache's stale state from a prior crashed
    worker before adding a fresh worktree.

    SIGKILL'd subprocesses (per-instance timeout, OOM) can leave:

    - ``.git/index.lock`` / ``HEAD.lock`` / similar — git refuses
      new operations while these exist.
    - half-built packfiles under ``objects/pack/tmp_*`` — wasted
      disk, can confuse repacks.
    - phantom worktree entries under ``worktrees/`` — git
      enumerates them and complains if their working trees are
      gone.

    Best-effort cleanup: remove obvious .lock files at the top
    level and let ``git worktree prune`` reconcile the
    ``worktrees/`` registry against on-disk state. Failures are
    logged-and-ignored; the next operation will surface a clearer
    error if the cache is genuinely corrupted.
    """
    for lock_name in ("HEAD.lock", "index.lock", "packed-refs.lock"):
        lock_path = repo_cache / lock_name
        if lock_path.exists():
            try:
                lock_path.unlink()
                logger.info(
                    "swebench: removed stale lock %s", lock_path,
                )
            except OSError as e:
                logger.warning(
                    "swebench: could not remove stale lock %s: %s",
                    lock_path, e,
                )
    if (repo_cache / "worktrees").exists():
        try:
            _run(
                ["git", "worktree", "prune"],
                cwd=repo_cache, timeout=60, retry_on_transient=False,
            )
        except RepoWorkspaceError as e:
            logger.warning(
                "swebench: ``git worktree prune`` on %s failed: %s",
                repo_cache, e,
            )


def _recover_root_owned_leftover(worktree: Path) -> None:
    """Last-resort cleanup for a worktree dir that ``shutil.rmtree``
    couldn't clear because a prior container teardown left root-owned
    files (typically ``__pycache__`` / ``.eggs`` from pytest /
    ``pip install -e`` running as root inside docker).

    Spawns a throwaway alpine-like container with the worktree's
    PARENT bind-mounted, chowns the target back to the host user,
    then re-attempts ``shutil.rmtree``. Best-effort — if docker
    isn't available or the chown fails, we log + leave the worktree
    in place (the subsequent ``git worktree add`` will then error
    loudly with "already exists", which is the right shape for
    operator-visible diagnosis).

    Why a throwaway container instead of ``sudo chown``: this
    workflow doesn't have root, and docker is already a hard
    dependency of the SWE-bench cascade. Reusing the daemon's
    privileged file access is the minimum-trust escalation path.
    """
    import shutil as _shutil  # noqa: PLC0415 — keep import-local
    try:
        host_uid = os.getuid()
        host_gid = os.getgid()
    except AttributeError:
        # Non-POSIX — no recovery possible.
        return
    parent = worktree.parent
    target_name = worktree.name
    try:
        # Use a tiny image that's almost certainly cached locally on
        # any host that's been running the swebench cascade. ``alpine``
        # is < 10MB and ships ``chown`` in busybox; if it's not on
        # disk we accept the one-time pull (~5s) as the cost of
        # recovery.
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{parent}:/work:rw",
                "alpine:latest",
                "chown", "-R", f"{host_uid}:{host_gid}", f"/work/{target_name}",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(
                "repo_workspace: privileged-chown recovery failed for "
                "%s (rc=%d, stderr=%s) — leaving worktree in place; "
                "subsequent ``git worktree add`` will surface the "
                "leftover via a clear ``already exists`` error",
                worktree, result.returncode,
                (result.stderr or "")[:300],
            )
            return
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "repo_workspace: privileged-chown recovery raised for %s: "
            "%s — leaving worktree in place",
            worktree, exc,
        )
        return
    # Chown succeeded; re-attempt rmtree. This time errors should NOT
    # be silenced — we want to know if SOMETHING ELSE is blocking.
    try:
        _shutil.rmtree(worktree)
        logger.info(
            "repo_workspace: privileged-chown recovery cleared %s",
            worktree,
        )
    except OSError as exc:
        logger.warning(
            "repo_workspace: post-chown rmtree still failed for %s: "
            "%s — leftover persists; the ``git worktree add`` below "
            "will fail with ``already exists``",
            worktree, exc,
        )


def _promisor_snapshot_incomplete(
    repo_cache: Path, base_commit: str,
) -> bool:
    """Return whether a partial cache lacks objects for one checkout.

    A ``--filter=blob:none`` bare clone contains commits and trees but
    deliberately omits most blobs.  Such a cache can answer
    ``cat-file -e <commit>`` while still being unable to act as the local
    upload-pack source used by :func:`_materialize_isolated_checkout`.
    Upload-pack disables lazy fetching from a promisor remote, so that
    failure otherwise surfaces later as ``lazy fetching disabled`` /
    ``could not fetch ... from promisor remote``.

    Probe with ``GIT_NO_LAZY_FETCH`` so diagnosis never silently repairs
    or otherwise mutates the shared cache.  A failed/timeout traversal is
    treated as incomplete and sent through the explicit hydration path;
    a non-promisor cache needs no such probe.
    """
    env = os.environ.copy()
    env["GIT_NO_LAZY_FETCH"] = "1"
    git_prefix = ["git", "-c", "core.hooksPath=/dev/null"]
    try:
        promisor = subprocess.run(
            [
                *git_prefix, "config", "--local", "--bool", "--get",
                "remote.origin.promisor",
            ],
            cwd=str(repo_cache), capture_output=True, timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "swebench: could not inspect promisor state for %s: %s",
            repo_cache, exc,
        )
        return False
    if promisor.returncode == 1:
        # No such config key: this is an ordinary complete cache.
        return False
    if promisor.returncode != 0:
        logger.warning(
            "swebench: git config promisor probe failed for %s (rc=%d): %s",
            repo_cache, promisor.returncode,
            (promisor.stderr or b"")[-512:].decode(
                "utf-8", errors="replace",
            ),
        )
        return False
    if promisor.stdout.strip().lower() != b"true":
        return False

    try:
        closure = subprocess.run(
            [
                *git_prefix, "rev-list", "--objects", "--missing=print",
                f"{base_commit}^{{tree}}",
            ],
            cwd=str(repo_cache), capture_output=True, timeout=120,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "swebench: promisor snapshot probe failed for %s @ %s: %s",
            repo_cache, base_commit[:12], exc,
        )
        return True
    if closure.returncode != 0:
        logger.warning(
            "swebench: promisor snapshot traversal failed for %s @ %s "
            "(rc=%d): %s",
            repo_cache, base_commit[:12], closure.returncode,
            (closure.stderr or b"")[-512:].decode(
                "utf-8", errors="replace",
            ),
        )
        return True
    return any(line.startswith(b"?") for line in closure.stdout.splitlines())


def _cached_worktree(
    instance: SwebenchInstance, cache_dir: Path, timeout: int,
) -> Path:
    """Acquire through a bare cache, then create an isolated checkout.

    The result is deliberately not a Git linked worktree. Sharing the
    cache's object database would expose descendant commits inside the
    benchmark sandbox.
    """
    repo_cache = cache_dir / instance.repo_org / instance.repo_name
    # Serialize the bare-clone with a per-repo file lock. Without this,
    # N parallel workers targeting the same repo on a cold cache all
    # pass the ``HEAD-exists`` check before any of them creates HEAD,
    # then race on ``git clone --bare`` — only one wins, the others
    # error with ``fatal: destination path '<repo_cache>' already
    # exists and is not an empty directory``. The lock funnels the
    # first hit through clone-once; subsequent workers see HEAD
    # already-established on entry and skip to the prune branch.
    #
    # ``fcntl.flock`` on the lock file is OS-managed and auto-released
    # when the process dies (or the fd is closed). The lock file path
    # lives ALONGSIDE the cache dir (not inside it) so a corrupt cache
    # can't take the lockfile with it.
    cache_parent = cache_dir / instance.repo_org
    cache_parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_parent / f"{instance.repo_name}.clone.lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        # Re-check under the lock — another worker may have completed
        # the clone while we waited.
        if not (repo_cache / "HEAD").exists():
            repo_cache.mkdir(parents=True, exist_ok=True)
            _run(
                ["git", "clone", "--bare", "--filter=blob:none",
                 _github_url(instance.repo), str(repo_cache)],
                timeout=timeout,
            )
        else:
            # Cache already exists — a prior worker may have left
            # stale lock files or phantom worktree entries (SIGKILL
            # on per-instance timeout). Reconcile before continuing.
            _prune_stale_repo_state(repo_cache)
        # Fetch the specific commit if it isn't already local. The repo
        # cache may have been created at the default-branch tip and an
        # instance's base_commit might predate that — fetch into
        # FETCH_HEAD by sha; failure here usually means the commit isn't
        # reachable from any remote ref (rare on GitHub but possible).
        try:
            # ``cat-file -e <sha>`` is a probe: non-zero exit means "not
            # local", which is informational here. Disable transient
            # retry so we don't spin on a sha that legitimately needs a
            # fetch.
            _run(
                ["git", "cat-file", "-e", instance.base_commit],
                cwd=repo_cache, timeout=30, retry_on_transient=False,
            )
        except RepoWorkspaceError:
            _run(
                ["git", "fetch", "--filter=blob:none",
                 "origin", instance.base_commit],
                cwd=repo_cache, timeout=timeout,
            )

        # A commit-presence probe is insufficient for a partial clone:
        # its snapshot blobs can still be promised but absent. Hydrate
        # exactly this depth-one snapshot before asking the cache to act
        # as a local upload-pack server. ``--refetch`` is essential when
        # the commit itself was already present; it bypasses negotiation
        # that would otherwise conclude there is nothing new to send.
        if _promisor_snapshot_incomplete(
            repo_cache, instance.base_commit,
        ):
            logger.info(
                "swebench: hydrating promisor cache snapshot %s @ %s",
                instance.repo, instance.base_commit[:12],
            )
            _run(
                [
                    "git", "fetch", "--refetch", "--no-filter",
                    "--depth=1", "--no-tags", "origin",
                    instance.base_commit,
                ],
                cwd=repo_cache, timeout=timeout,
            )
            if _promisor_snapshot_incomplete(
                repo_cache, instance.base_commit,
            ):
                raise RepoWorkspaceError(
                    "promisor cache hydration completed without a full "
                    f"snapshot for {instance.repo} @ "
                    f"{instance.base_commit}"
                )
        # fcntl.flock auto-releases when lock_fd is closed by the
        # ``with`` block. Hydration is serialized with clone/fetch so
        # same-repo workers cannot race over shared FETCH_HEAD/shallow
        # state. The read-only materialization below may run concurrently.
    worktree_root = cache_dir / "_worktrees"
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree = Path(tempfile.mkdtemp(
        prefix=(
            f"{instance.repo.replace('/', '__')}__"
            f"{instance.instance_id}__"
        ),
        dir=worktree_root,
    ))
    try:
        _materialize_isolated_checkout(
            source=repo_cache.resolve().as_uri(),
            base_commit=instance.base_commit,
            dest=worktree,
            timeout=timeout,
        )
    except BaseException:
        shutil.rmtree(worktree, ignore_errors=True)
        raise
    return worktree


@contextlib.contextmanager
def repo_workspace(
    instance: SwebenchInstance,
    *,
    cache_dir: Path | None = None,
    timeout: int = 600,
) -> Iterator[Path]:
    """Yield a path to a clean git working tree at ``base_commit``.

    Args:
      instance: the SWE-bench record to materialize.
      cache_dir: if set, use a per-repo acquisition cache and materialize a
        standalone per-instance checkout below this directory. If None,
        fetch directly into an ephemeral checkout.
      timeout: per-subprocess timeout in seconds (default 600 = 10 min;
        the initial bare clone of a large repo can be slow over
        common connections).

    Raises ``RepoWorkspaceError`` on clone / fetch / checkout failure.
    """
    if cache_dir is None:
        tempdir = Path(tempfile.mkdtemp(prefix="swebench_"))
        try:
            _ephemeral_clone(instance, tempdir, timeout)
            yield tempdir
        finally:
            shutil.rmtree(tempdir, ignore_errors=True)
        return

    cache_dir = cache_dir.resolve()
    worktree = _cached_worktree(instance, cache_dir, timeout)
    try:
        yield worktree
    finally:
        # The standalone checkout has no registration in the cache, so
        # teardown cannot mutate shared Git state.
        shutil.rmtree(worktree, ignore_errors=True)


__all__ = ["RepoWorkspaceError", "repo_workspace"]
