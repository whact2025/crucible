# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Opt-in worktree isolation for tier / architect runs.

Each ``crucible tier`` or ``architect`` run owns a detached worktree.
Implementation, build/verification, recovery, and rollback use that checkout,
isolating their mutations from other runs against the same repository.

When invoked through ``with worktree_session(repo_root) as wt:``:
1. ``git worktree add --detach <tmp> HEAD`` from ``repo_root``.
2. Yield a ``WorktreeResolver`` whose ``resolve_worktree_root(".")``
   returns the worktree path. The caller uses that as ``cwd`` for
   downstream subprocess invocations.
3. On exit, recover or refuse retained mutation authority before removing the
   worktree. Cleanup failures retain their owner and lease for an explicit retry.

Workflows pass the resolved checkout as ``cwd`` to backend invocations and
deterministic runners. Attached ToolSets share the resolver through
``ToolContext.path_resolver`` so their scoped writes use the same checkout.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ._run import write_atomic_json

logger = logging.getLogger(__name__)


def _process_start_token(pid: int) -> str | None:
    """Return the kernel process birth token used to defeat PID reuse."""

    try:
        # Linux /proc stat field 22 is the process start time in clock ticks.
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        return raw.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _lease_owner_alive(lease: dict) -> bool:
    try:
        pid = int(lease["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    expected = lease.get("process_start")
    actual = _process_start_token(pid)
    return expected is None or actual is None or expected == actual


def _git_common_dir(repo_root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    path = Path(result.stdout.strip())
    return (repo_root / path).resolve() if not path.is_absolute() else path.resolve()


def _recover_or_refuse_mutation_registry(repo_root: Path) -> None:
    """Recover central mutation journals before touching worktree topology."""

    from assay.corpus.store import Corpus  # noqa: PLC0415
    from ._mutation_journal import (  # noqa: PLC0415
        _workspace_authority_common_dir,
        recover_mutation_journals,
    )

    registry = (
        _workspace_authority_common_dir(repo_root)
        / "crucible-mutation-journals"
    )
    if not registry.is_dir():
        return
    recovery_targets: set[tuple[Path, Path]] = set()
    for marker in sorted(registry.glob("*.json")):
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            cwd = Path(record["cwd"]).resolve()
            spec_dir = Path(record["spec_dir"]).resolve()
        except FileNotFoundError:
            # An earlier family recovery can consume every sibling marker.
            continue
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid active mutation registry marker {marker}: {exc}"
            ) from exc
        if cwd.is_dir() and spec_dir.is_dir():
            recovery_targets.add((spec_dir, cwd))
    recovery_failures: list[str] = []
    for spec_dir, cwd in sorted(
        recovery_targets,
        key=lambda item: (str(item[0]), str(item[1])),
    ):
        report = recover_mutation_journals(Corpus(spec_dir), cwd)
        recovery_failures.extend(report.failures)
    blockers = sorted(registry.glob("*.json"))
    if blockers:
        raise RuntimeError(
            "active mutation journal prevents Git worktree topology changes: "
            + "; ".join((
                *(f"recovery failed: {failure}" for failure in recovery_failures),
                *(str(path) for path in blockers),
            ))
        )
    if recovery_failures:
        logger.debug(
            "mutation registry family recovered after provisional reports: %s",
            recovery_failures,
        )


def _registry_references_worktree(repo_root: Path, worktree: Path) -> bool:
    from ._mutation_journal import _workspace_authority_common_dir  # noqa: PLC0415

    registry = (
        _workspace_authority_common_dir(repo_root)
        / "crucible-mutation-journals"
    )
    if not registry.is_dir():
        return False
    for marker in registry.glob("*.json"):
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
            recorded = Path(record["cwd"]).resolve()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return True
        if recorded == worktree or worktree in recorded.parents:
            return True
    return False


def _reap_orphaned_worktree_leases_locked(
    repo_root: Path | str,
) -> tuple[Path, ...]:
    """Remove Crucible worktrees whose exact owning process has died."""

    root = Path(repo_root).resolve()
    _recover_or_refuse_mutation_registry(root)
    lease_root = _git_common_dir(root) / "crucible-worktree-leases"
    if not lease_root.is_dir():
        return ()
    reaped: list[Path] = []
    registered_raw = subprocess.run(
        [
            "git", "-c", "core.hooksPath=/dev/null", "worktree", "list",
            "--porcelain", "-z",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    registered = {
        Path(os.fsdecode(field[len(b"worktree "):])).resolve()
        for field in registered_raw.split(b"\0")
        if field.startswith(b"worktree ")
    }
    temp_root = Path(tempfile.gettempdir()).resolve()
    for lease_path in sorted(lease_root.glob("*.json")):
        try:
            lease = json.loads(lease_path.read_text(encoding="utf-8"))
            worktree = Path(lease["worktree"]).resolve()
            if Path(lease["repo_root"]).resolve() != root:
                raise ValueError("lease repository mismatch")
            if lease.get("schema") not in {
                "crucible.worktree-lease.v1",
                "crucible.worktree-lease.v2",
            }:
                raise ValueError("unsupported lease schema")
            if lease.get("state", "created") not in {"intent", "created"}:
                raise ValueError("unsupported lease state")
            if (
                worktree.parent != temp_root
                or not worktree.name.startswith("crucible_worktree_")
            ):
                raise ValueError("lease worktree is outside Crucible temp authority")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.error("invalid worktree lease %s: %s", lease_path, exc)
            continue
        if _lease_owner_alive(lease):
            continue
        if worktree not in registered:
            # A dead creation intent may precede `git worktree add`. Remove
            # only an empty directory; never recursively trust lease content.
            try:
                worktree.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                logger.error(
                    "unregistered orphan lease target is non-empty; preserving %s",
                    worktree,
                )
                continue
            lease_path.unlink(missing_ok=True)
            reaped.append(worktree)
            continue
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "worktree",
             "remove", "--force", str(worktree)],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0 or worktree.exists():
            logger.error("could not remove orphaned worktree %s", worktree)
            continue
        lease_path.unlink(missing_ok=True)
        reaped.append(worktree)
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "worktree", "prune"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return tuple(reaped)


def reap_orphaned_worktree_leases(repo_root: Path | str) -> tuple[Path, ...]:
    """Remove dead leases while holding the shared mutation authority lock."""

    from ._mutation_journal import workspace_mutation_lock  # noqa: PLC0415

    root = Path(repo_root).resolve()
    with workspace_mutation_lock(root):
        return _reap_orphaned_worktree_leases_locked(root)


class WorktreeCleanupError(RuntimeError):
    """A failed worktree operation retained a retryable owned checkout."""

    def __init__(
        self,
        *,
        owner: WorktreeResolver,
        primary_error: BaseException,
        cleanup_errors: tuple[BaseException, ...],
    ) -> None:
        if not cleanup_errors:
            raise ValueError(
                "WorktreeCleanupError requires a cleanup failure"
            )
        if not owner.owned:
            raise ValueError(
                "WorktreeCleanupError requires live cleanup authority"
            )
        self.owner = owner
        self.primary_error = primary_error
        self.cleanup_errors = cleanup_errors
        details = "; ".join(
            f"{type(error).__name__}: {error}"
            for error in cleanup_errors
        )
        super().__init__(
            f"worktree operation failed ({type(primary_error).__name__}: "
            f"{primary_error}); cleanup retained worktree="
            f"{owner.worktree_path} intent={owner.intent_path} "
            f"lease={owner.lease_path}: {details}"
        )

    def retry_cleanup(self) -> None:
        """Retry release through the still-owning resolver."""

        self.owner.close()


class WorktreeResolver:
    """``PathResolver``-protocol implementation backed by a ``git worktree``.

    The worktree is created lazily on the first ``resolve_write`` call
    (or eagerly via ``ensure_worktree()``). All paths resolve against
    the worktree root once it exists; before that, reads fall through
    to the source repo so cold-start lookups don't pay creation cost.

    Path-traversal: every resolve checks that the resolved path stays
    rooted under the worktree (or source repo for cold reads).
    Mirrors ``resolve_within_repo``'s ``relative_to`` discipline.
    """

    def __init__(
        self,
        repo_root: Path | str,
        *,
        base_ref: str = "HEAD",
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._base_ref = base_ref
        self._worktree_path: Path | None = None
        self._intent_path: Path | None = None
        self._lease_path: Path | None = None

    # --- PathResolver protocol ----------------------------------------

    def resolve_read(self, relative: str) -> Path | None:
        # When a worktree exists, prefer it (writes propagate). Before
        # that, read from the source repo so cold-start navigation
        # doesn't trip worktree creation.
        if self._worktree_path is not None:
            cand = self._safe_resolve(self._worktree_path, relative)
            if cand is not None and cand.exists():
                return cand
        return self._safe_resolve(self._repo_root, relative)

    def resolve_write(self, relative: str) -> Path | None:
        # Writer schedulers materialize the checkout under the coordinator's
        # repository lock before launching worker threads. Re-entering that
        # process-local RLock from a different worker thread deadlocks. Once
        # ownership is published, path resolution needs no topology mutation;
        # cold callers still take the lock through ``ensure_worktree``.
        wt = self._worktree_path
        if wt is None:
            wt = self.ensure_worktree()
        return self._safe_resolve(wt, relative)

    def resolve_worktree_root(self, project: str = ".") -> Path | None:
        if self._worktree_path is None:
            return None
        if project in (".", ""):
            return self._worktree_path
        candidate = self._worktree_path / project
        return candidate if candidate.is_dir() else self._worktree_path

    # --- worktree lifecycle -------------------------------------------

    def ensure_worktree(self) -> Path:
        """Create the worktree if it doesn't exist; return its path."""
        from ._mutation_journal import workspace_mutation_lock  # noqa: PLC0415

        try:
            with workspace_mutation_lock(self._repo_root):
                return self._ensure_worktree_locked()
        except WorktreeCleanupError:
            raise
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(
                "WorktreeResolver:"
            ):
                raise
            raise RuntimeError(
                f"WorktreeResolver: git worktree add failed in "
                f"{self._repo_root}: not a Git repository"
            ) from exc

    def _ensure_worktree_locked(self) -> Path:
        if self._worktree_path is not None:
            return self._worktree_path
        # A plain ``git worktree prune`` cannot remove a crash-left directory
        # because Git still considers it present.  Durable PID+birth-token
        # leases let the next invocation prove ownership is dead first.
        try:
            _reap_orphaned_worktree_leases_locked(self._repo_root)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"WorktreeResolver: git worktree add failed in "
                f"{self._repo_root}: not a Git repository"
            ) from exc
        tmp = Path(tempfile.mkdtemp(prefix="crucible_worktree_")).resolve()
        self._intent_path = tmp
        lease_root = _git_common_dir(self._repo_root) / "crucible-worktree-leases"
        lease_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._lease_path = lease_root / f"{secrets.token_hex(16)}.json"
        lease = {
            "schema": "crucible.worktree-lease.v2",
            "state": "intent",
            "repo_root": str(self._repo_root),
            "worktree": str(tmp),
            "base_ref": self._base_ref,
            "pid": os.getpid(),
            "process_start": _process_start_token(os.getpid()),
        }
        try:
            write_atomic_json(self._lease_path, lease)
        except BaseException as primary_error:
            try:
                self._close_locked()
            except WorktreeCleanupError as cleanup_error:
                raise WorktreeCleanupError(
                    owner=self,
                    primary_error=primary_error,
                    cleanup_errors=cleanup_error.cleanup_errors,
                ) from primary_error
            raise
        # ``--detach`` keeps us off the source repo's branches —
        # no chance of accidentally moving HEAD on the operator's
        # checkout when this worktree commits.
        try:
            subprocess.run(
                ["git", "-c", "core.hooksPath=/dev/null",
                 "worktree", "add", "--detach",
                 str(tmp), self._base_ref],
                cwd=str(self._repo_root),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            operation_error = RuntimeError(
                f"WorktreeResolver: git worktree add failed in "
                f"{self._repo_root}: "
                f"{error.stderr.decode('utf-8', errors='replace').strip()}"
            )
            try:
                self._close_locked()
            except WorktreeCleanupError as cleanup_error:
                raise WorktreeCleanupError(
                    owner=self,
                    primary_error=operation_error,
                    cleanup_errors=cleanup_error.cleanup_errors,
                ) from error
            raise operation_error from error
        self._worktree_path = tmp
        self._intent_path = None
        try:
            write_atomic_json(self._lease_path, {**lease, "state": "created"})
        except BaseException as primary_error:
            # A worktree without a durable owner lease cannot be reaped safely
            # after process death, so creation is atomic with lease publish.
            try:
                self.close()
            except WorktreeCleanupError as cleanup_error:
                raise WorktreeCleanupError(
                    owner=self,
                    primary_error=primary_error,
                    cleanup_errors=cleanup_error.cleanup_errors,
                ) from primary_error
            except BaseException as cleanup_error:
                raise WorktreeCleanupError(
                    owner=self,
                    primary_error=primary_error,
                    cleanup_errors=(cleanup_error,),
                ) from primary_error
            raise
        logger.info(
            "WorktreeResolver: created %s from %s @ %s",
            tmp, self._repo_root, self._base_ref,
        )
        return tmp

    def close(self) -> None:
        """Remove the worktree if extant. Idempotent."""
        from ._mutation_journal import workspace_mutation_lock  # noqa: PLC0415

        with workspace_mutation_lock(self._repo_root):
            self._close_locked()

    def _close_locked(self) -> None:
        failures: list[BaseException] = []
        wt = self._worktree_path
        if wt is not None:
            try:
                _recover_or_refuse_mutation_registry(self._repo_root)
                if _registry_references_worktree(self._repo_root, wt):
                    raise RuntimeError(
                        f"active mutation journal retains worktree {wt}; "
                        "refusing removal"
                    )
                result = subprocess.run(
                    ["git", "-c", "core.hooksPath=/dev/null",
                     "worktree", "remove", "--force", str(wt)],
                    cwd=str(self._repo_root),
                    check=False,
                    capture_output=True,
                )
                if result.returncode != 0 or wt.exists():
                    raise RuntimeError(
                        "WorktreeResolver: could not remove registered "
                        f"worktree {wt}"
                    )
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                failures.append(error)
            else:
                # Git ownership is settled even if the independent lease
                # unlink below fails. A retry must proceed directly to that
                # lease rather than attempting to remove an absent worktree.
                self._worktree_path = None
                logger.info("WorktreeResolver: removed %s", wt)

        intent = self._intent_path
        if intent is not None:
            try:
                shutil.rmtree(intent)
            except FileNotFoundError:
                self._intent_path = None
            except OSError as error:
                failures.append(error)
            else:
                self._intent_path = None

        # A live checkout or creation intent needs its durable reaping
        # authority. Only unlink the lease once both path owners are settled.
        if (
            self._worktree_path is None
            and self._intent_path is None
            and self._lease_path is not None
        ):
            lease = self._lease_path
            try:
                lease.unlink(missing_ok=True)
            except OSError as error:
                failures.append(error)
            else:
                self._lease_path = None

        if failures:
            raise WorktreeCleanupError(
                owner=self,
                primary_error=RuntimeError("worktree close"),
                cleanup_errors=tuple(failures),
            )

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def worktree_path(self) -> Path | None:
        return self._worktree_path

    @property
    def lease_path(self) -> Path | None:
        return self._lease_path

    @property
    def intent_path(self) -> Path | None:
        return self._intent_path

    @property
    def owned(self) -> bool:
        return any((
            self._worktree_path is not None,
            self._intent_path is not None,
            self._lease_path is not None,
        ))

    # --- internals ----------------------------------------------------

    @staticmethod
    def _safe_resolve(root: Path, relative: str) -> Path | None:
        try:
            cand = (root / relative).resolve()
        except (ValueError, OSError):
            return None
        try:
            cand.relative_to(root)
        except ValueError:
            # Path-traversal escape attempt; reject.
            return None
        return cand


@contextmanager
def worktree_session(
    repo_root: Path | str,
    *,
    base_ref: str = "HEAD",
) -> Iterator[WorktreeResolver]:
    """Context-managed WorktreeResolver — eagerly creates the worktree
    so the caller can rebind ``cwd`` immediately, and always cleans
    up on exit (success or exception)."""
    from ._mutation_journal import workspace_mutation_lock  # noqa: PLC0415

    resolver = WorktreeResolver(repo_root, base_ref=base_ref)
    with workspace_mutation_lock(resolver.repo_root):
        try:
            resolver._ensure_worktree_locked()
            yield resolver
        except BaseException as primary_error:
            try:
                resolver._close_locked()
            except WorktreeCleanupError as cleanup_error:
                cleanup_primary = primary_error
                cleanup_failures = cleanup_error.cleanup_errors
                if (
                    isinstance(primary_error, WorktreeCleanupError)
                    and primary_error.owner is resolver
                ):
                    cleanup_primary = primary_error.primary_error
                    cleanup_failures = (
                        *primary_error.cleanup_errors,
                        *cleanup_failures,
                    )
                raise WorktreeCleanupError(
                    owner=resolver,
                    primary_error=cleanup_primary,
                    cleanup_errors=cleanup_failures,
                ) from primary_error
            except BaseException as cleanup_error:
                if resolver.owned:
                    raise WorktreeCleanupError(
                        owner=resolver,
                        primary_error=primary_error,
                        cleanup_errors=(cleanup_error,),
                    ) from primary_error
            if (
                isinstance(primary_error, WorktreeCleanupError)
                and not primary_error.owner.owned
            ):
                # Session exit settled the exact owner retained by the setup
                # failure, so the retry-owner exception is no longer truthful.
                raise primary_error.primary_error from primary_error
            raise
        else:
            try:
                resolver._close_locked()
            except WorktreeCleanupError:
                raise
            except BaseException as cleanup_error:
                if resolver.owned:
                    raise WorktreeCleanupError(
                        owner=resolver,
                        primary_error=RuntimeError("worktree session exit"),
                        cleanup_errors=(cleanup_error,),
                    ) from cleanup_error
                raise


__all__ = [
    "WorktreeResolver",
    "WorktreeCleanupError",
    "reap_orphaned_worktree_leases",
    "worktree_session",
]
