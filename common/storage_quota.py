# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed growth quotas for bind-mounted benchmark workspaces.

Docker's writable-layer ``--storage-opt size=`` is driver-specific and does
not account for bind mounts.  Crucible's benchmark sandboxes intentionally use
bind-mounted workspaces so host-side source tools and the container see the
same files.  This monitor closes that gap: it records the pre-authoring
allocated size, polls the mounted directories while untrusted code runs, and
terminates the sandbox as soon as growth exceeds the configured allowance.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import socket
import subprocess
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from crucible.common.benchmark_runtime import canonical_fingerprint, run_process_tree


STORAGE_QUOTA_SETTLEMENT_RECEIPT_SCHEMA = (
    "crucible.storage-quota-settlement-receipt.v1"
)
_STORAGE_QUOTA_CLOSE_PHASE_SCHEMA = "crucible.storage-quota-close-phase.v1"
_KERNEL_QUOTA_BACKEND_CONFIGURATION_SCHEMA = (
    "crucible.kernel-quota-backend-configuration.v1"
)
_QUOTA_ENFORCEMENT_OUTCOMES = frozenset(
    {"within_limit", "exceeded", "scan_failed"}
)
_LEASE_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SHA256_PREFIX = "sha256:"
_SHA256_LENGTH = len(_SHA256_PREFIX) + 64


class StorageQuotaExceeded(RuntimeError):
    """A sandbox's bind-mounted writable state exceeded its growth cap."""


class StorageQuotaBackendUnavailable(StorageQuotaExceeded):
    """The configured kernel-backed quota controller is unavailable."""


class StorageQuotaCleanupError(StorageQuotaBackendUnavailable):
    """A kernel-backed quota lease could not be settled cleanly."""


@dataclass(frozen=True)
class KernelQuotaLease:
    token: str

    def __post_init__(self) -> None:
        if type(self) is not KernelQuotaLease:
            raise ValueError("kernel quota lease subclasses are not accepted")
        if type(self.token) is not str or _LEASE_TOKEN_RE.fullmatch(self.token) is None:
            raise ValueError("kernel quota lease token does not match the exact shape")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and value.startswith(_SHA256_PREFIX)
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _exact_int(value: object, *, field: str, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"storage quota {field} must be an exact {qualifier} integer")
    return value


@dataclass(frozen=True)
class _StorageQuotaClosePhase:
    """Private pre-retirement phase retained for final receipt composition."""

    paths: tuple[str, ...]
    limit_bytes: int
    baseline_bytes: int
    peak_growth_bytes: int
    kernel_hard_limit_bytes: int
    enforcement_outcome: str
    require_kernel: bool
    kernel_backend_argv_fingerprint: str | None
    kernel_lease_fingerprint: str | None
    kernel_release_proven: bool

    def __post_init__(self) -> None:
        if type(self) is not _StorageQuotaClosePhase:
            raise ValueError("storage quota close phase subclasses are not accepted")
        if type(self.paths) is not tuple or not self.paths or any(
            type(path) is not str
            or not path
            or not os.path.isabs(path)
            or os.path.normpath(path) != path
            for path in self.paths
        ):
            raise ValueError("storage quota receipt paths must be canonical absolute paths")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("storage quota receipt paths must be distinct")
        _exact_int(self.limit_bytes, field="limit_bytes", positive=True)
        _exact_int(self.baseline_bytes, field="baseline_bytes")
        _exact_int(self.peak_growth_bytes, field="peak_growth_bytes")
        _exact_int(
            self.kernel_hard_limit_bytes,
            field="kernel_hard_limit_bytes",
            positive=True,
        )
        if self.kernel_hard_limit_bytes != self.baseline_bytes + self.limit_bytes:
            raise ValueError("storage quota hard limit conflicts with baseline")
        if (
            type(self.enforcement_outcome) is not str
            or self.enforcement_outcome not in _QUOTA_ENFORCEMENT_OUTCOMES
        ):
            raise ValueError("storage quota enforcement outcome is invalid")
        if self.enforcement_outcome == "within_limit" and (
            self.peak_growth_bytes > self.limit_bytes
        ):
            raise ValueError("storage quota enforcement outcome conflicts with peak")
        if self.enforcement_outcome == "exceeded" and (
            self.peak_growth_bytes <= self.limit_bytes
        ):
            raise ValueError("storage quota enforcement outcome conflicts with peak")
        if type(self.require_kernel) is not bool:
            raise ValueError("storage quota require_kernel must be an exact bool")
        if type(self.kernel_release_proven) is not bool:
            raise ValueError("storage quota kernel_release_proven must be an exact bool")
        fingerprints = (
            self.kernel_backend_argv_fingerprint,
            self.kernel_lease_fingerprint,
        )
        if any(value is not None and not _is_sha256(value) for value in fingerprints):
            raise ValueError("storage quota kernel fingerprints must be exact sha256")
        kernel_bound = all(value is not None for value in fingerprints)
        if any(value is not None for value in fingerprints) != kernel_bound:
            raise ValueError("storage quota kernel fingerprints must be present together")
        if self.kernel_release_proven != kernel_bound:
            raise ValueError("storage quota release proof must match kernel authority")
        if self.require_kernel and not kernel_bound:
            raise ValueError("required kernel quota is missing release proof")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": _STORAGE_QUOTA_CLOSE_PHASE_SCHEMA,
            "paths": list(self.paths),
            "limit_bytes": self.limit_bytes,
            "baseline_bytes": self.baseline_bytes,
            "peak_growth_bytes": self.peak_growth_bytes,
            "kernel_hard_limit_bytes": self.kernel_hard_limit_bytes,
            "enforcement_outcome": self.enforcement_outcome,
            "monitor_stopped": True,
            "final_sample_attempted": True,
            "require_kernel": self.require_kernel,
            "kernel_backend_argv_fingerprint": self.kernel_backend_argv_fingerprint,
            "kernel_lease_fingerprint": self.kernel_lease_fingerprint,
            "kernel_release_proven": self.kernel_release_proven,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_payload())

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "_StorageQuotaClosePhase":
        expected = {
            "schema_version",
            "paths",
            "limit_bytes",
            "baseline_bytes",
            "peak_growth_bytes",
            "kernel_hard_limit_bytes",
            "enforcement_outcome",
            "monitor_stopped",
            "final_sample_attempted",
            "require_kernel",
            "kernel_backend_argv_fingerprint",
            "kernel_lease_fingerprint",
            "kernel_release_proven",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("storage quota close phase has unexpected fields")
        if raw["schema_version"] != _STORAGE_QUOTA_CLOSE_PHASE_SCHEMA:
            raise ValueError("storage quota close phase schema is unsupported")
        if (
            raw["monitor_stopped"] is not True
            or raw["final_sample_attempted"] is not True
        ):
            raise ValueError("storage quota close phase lacks sampling shutdown proof")
        paths = raw["paths"]
        if type(paths) is not list or not all(
            type(path) is str for path in paths
        ):
            raise ValueError("storage quota close phase paths are invalid")
        return cls(
            paths=tuple(paths),
            limit_bytes=_exact_int(raw["limit_bytes"], field="limit_bytes", positive=True),
            baseline_bytes=_exact_int(raw["baseline_bytes"], field="baseline_bytes"),
            peak_growth_bytes=_exact_int(
                raw["peak_growth_bytes"], field="peak_growth_bytes"
            ),
            kernel_hard_limit_bytes=_exact_int(
                raw["kernel_hard_limit_bytes"],
                field="kernel_hard_limit_bytes",
                positive=True,
            ),
            enforcement_outcome=raw["enforcement_outcome"],
            require_kernel=raw["require_kernel"],
            kernel_backend_argv_fingerprint=raw[
                "kernel_backend_argv_fingerprint"
            ],
            kernel_lease_fingerprint=raw["kernel_lease_fingerprint"],
            kernel_release_proven=raw["kernel_release_proven"],
        )


@dataclass(frozen=True)
class StorageQuotaSettlementReceipt:
    """Canonical cleanup evidence composed only after exact retirement."""

    close: _StorageQuotaClosePhase
    kernel_backend_argv_fingerprint: str
    kernel_lease_fingerprint: str

    def __post_init__(self) -> None:
        if type(self) is not StorageQuotaSettlementReceipt:
            raise ValueError("storage quota settlement subclasses are not accepted")
        if type(self.close) is not _StorageQuotaClosePhase:
            raise ValueError("storage quota settlement close phase is invalid")
        if not all(
            _is_sha256(value)
            for value in (
                self.kernel_backend_argv_fingerprint,
                self.kernel_lease_fingerprint,
            )
        ):
            raise ValueError("storage quota settlement bindings must be exact sha256")
        if not self.close.require_kernel or not self.close.kernel_release_proven:
            raise ValueError("storage quota settlement lacks required kernel release")
        if (
            self.close.kernel_backend_argv_fingerprint
            != self.kernel_backend_argv_fingerprint
            or self.close.kernel_lease_fingerprint
            != self.kernel_lease_fingerprint
        ):
            raise ValueError("storage quota retirement bindings changed after close")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": STORAGE_QUOTA_SETTLEMENT_RECEIPT_SCHEMA,
            "close": self.close.to_payload(),
            "retirement": {
                "operation": "retire",
                "close_phase_fingerprint": self.close.fingerprint,
                "kernel_backend_argv_fingerprint": (
                    self.kernel_backend_argv_fingerprint
                ),
                "kernel_lease_fingerprint": self.kernel_lease_fingerprint,
                "returncode": 0,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "kernel_retirement_proven": True,
            },
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_payload())

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "StorageQuotaSettlementReceipt":
        if type(raw) is not dict or set(raw) != {
            "schema_version",
            "close",
            "retirement",
        }:
            raise ValueError("storage quota settlement receipt has unexpected fields")
        if raw["schema_version"] != STORAGE_QUOTA_SETTLEMENT_RECEIPT_SCHEMA:
            raise ValueError("storage quota settlement receipt schema is unsupported")
        close_raw = raw["close"]
        retirement = raw["retirement"]
        if type(close_raw) is not dict or type(retirement) is not dict:
            raise ValueError("storage quota settlement receipt phases are invalid")
        close = _StorageQuotaClosePhase.from_payload(close_raw)
        expected_retirement = {
            "operation",
            "close_phase_fingerprint",
            "kernel_backend_argv_fingerprint",
            "kernel_lease_fingerprint",
            "returncode",
            "stdout_bytes",
            "stderr_bytes",
            "kernel_retirement_proven",
        }
        if set(retirement) != expected_retirement:
            raise ValueError("storage quota retirement phase has unexpected fields")
        if (
            retirement["operation"] != "retire"
            or retirement["close_phase_fingerprint"] != close.fingerprint
            or type(retirement["returncode"]) is not int
            or retirement["returncode"] != 0
            or type(retirement["stdout_bytes"]) is not int
            or retirement["stdout_bytes"] != 0
            or type(retirement["stderr_bytes"]) is not int
            or retirement["stderr_bytes"] != 0
            or retirement["kernel_retirement_proven"] is not True
        ):
            raise ValueError("storage quota retirement phase is not exact terminal proof")
        return cls(
            close=close,
            kernel_backend_argv_fingerprint=retirement[
                "kernel_backend_argv_fingerprint"
            ],
            kernel_lease_fingerprint=retirement["kernel_lease_fingerprint"],
        )


class KernelQuotaBackend:
    """Adapter for an operator-installed project-quota helper.

    The helper must persist its own lease ledger and implement four commands:

    * ``reconcile --owner-host HOST`` reclaims same-host leases whose exact
      owner PID/start token is dead;
    * ``reserve --owner-host HOST --owner-pid PID --owner-start-token TOKEN
      --limit-bytes N --path P ...`` prints an opaque lease token; and
    * ``release --token TOKEN`` idempotently begins cleanup; and
    * ``retire --token TOKEN`` proves that the exact known tombstone reached
      the helper's durable terminal state.

    This helper-owned ledger is what makes kernel reservations recoverable
    after SIGKILL, when Python cannot execute ``__exit__``. It can provision
    XFS/ext4 project quotas or bounded volumes with the host's required
    privileges. Crucible never invokes it through a shell.
    """

    def __init__(self, command: tuple[str, ...]) -> None:
        if type(command) is not tuple or not command or not all(
            type(item) is str and item for item in command
        ):
            raise ValueError("kernel quota command must be non-empty strings")
        self.command = command

    @classmethod
    def from_environment(cls) -> "KernelQuotaBackend | None":
        raw = os.environ.get("CRUCIBLE_QUOTA_HELPER", "").strip()
        return None if not raw else cls(tuple(shlex.split(raw)))

    def reserve(
        self, paths: tuple[Path, ...], *, limit_bytes: int,
    ) -> KernelQuotaLease:
        owner_start = _process_start_token(os.getpid())
        if owner_start is None:
            raise StorageQuotaBackendUnavailable(
                "cannot establish exact process identity for kernel quota lease"
            )
        command = [
            *self.command,
            "reserve",
            "--owner-host", socket.gethostname(),
            "--owner-pid", str(os.getpid()),
            "--owner-start-token", owner_start,
            "--limit-bytes", str(limit_bytes),
        ]
        for path in paths:
            command.extend(["--path", str(path)])
        try:
            result = run_process_tree(command, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise StorageQuotaBackendUnavailable(
                f"kernel quota helper failed to start: {exc}"
            ) from exc
        stdout = result.stdout or b""
        stderr = result.stderr or b""
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(
                "utf-8", errors="replace",
            )[-1000:]
            raise StorageQuotaBackendUnavailable(
                f"kernel quota reserve failed: {stderr}"
            )
        if stderr != b"":
            raise StorageQuotaBackendUnavailable(
                "kernel quota reserve produced unexpected stderr"
            )
        try:
            token = stdout[:-1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise StorageQuotaBackendUnavailable(
                "kernel quota reserve token is not exact ASCII"
            ) from exc
        if (
            not stdout.endswith(b"\n")
            or stdout.count(b"\n") != 1
            or _LEASE_TOKEN_RE.fullmatch(token) is None
        ):
            raise StorageQuotaBackendUnavailable(
                "kernel quota reserve token does not match the exact protocol"
            )
        return KernelQuotaLease(token=token)

    def reconcile(self) -> None:
        """Ask the helper to reclaim dead exact-owner leases on this host."""

        try:
            result = run_process_tree(
                [
                    *self.command,
                    "reconcile",
                    "--owner-host", socket.gethostname(),
                ],
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise StorageQuotaBackendUnavailable(
                f"kernel quota helper reconciliation failed to start: {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(
                "utf-8", errors="replace",
            )[-1000:]
            raise StorageQuotaBackendUnavailable(
                f"kernel quota reconciliation failed: {stderr}"
            )
        if (result.stdout or b"") != b"" or (result.stderr or b"") != b"":
            raise StorageQuotaBackendUnavailable(
                "kernel quota reconciliation produced unexpected output"
            )

    def release(self, lease: KernelQuotaLease) -> None:
        try:
            result = run_process_tree(
                # Keep the opaque token in the same argv element as its
                # option. URL-safe tokens may legitimately begin with ``-``;
                # passing such a token as the next element makes argparse
                # interpret it as a new option and the fail-closed release
                # never reaches the helper.
                [*self.command, "release", f"--token={lease.token}"],
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise StorageQuotaCleanupError(
                f"kernel quota release failed to start: {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(
                "utf-8", errors="replace",
            )[-1000:]
            raise StorageQuotaCleanupError(
                f"kernel quota release failed: {stderr}"
            )
        if (result.stdout or b"") != b"" or (result.stderr or b"") != b"":
            raise StorageQuotaCleanupError(
                "kernel quota release produced unexpected output"
            )

    def retire(self, lease: KernelQuotaLease) -> None:
        """Require durable retirement proof for one exact issued lease."""

        try:
            result = run_process_tree(
                [*self.command, "retire", f"--token={lease.token}"],
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise StorageQuotaCleanupError(
                f"kernel quota retirement proof failed to start: {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(
                "utf-8", errors="replace",
            )[-1000:]
            raise StorageQuotaCleanupError(
                f"kernel quota retirement proof failed: {stderr}"
            )
        if (result.stdout or b"") != b"" or (result.stderr or b"") != b"":
            raise StorageQuotaCleanupError(
                "kernel quota retirement proof produced unexpected output"
            )


def _kernel_backend_argv_fingerprint(backend: object) -> str:
    command = getattr(backend, "command", None)
    if type(command) is not tuple or not command or not all(
        type(item) is str and item for item in command
    ):
        raise StorageQuotaCleanupError(
            "kernel quota backend lacks exact configured argv evidence"
        )
    return canonical_fingerprint(
        {
            "schema_version": _KERNEL_QUOTA_BACKEND_CONFIGURATION_SCHEMA,
            # This binds configured argv, not the helper binary/runtime
            # identity. That stronger closure remains a caller obligation.
            "argv": list(command),
        }
    )


def _kernel_lease_fingerprint(lease: KernelQuotaLease) -> str:
    return "sha256:" + hashlib.sha256(lease.token.encode("utf-8")).hexdigest()


def _process_start_token(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        command_end = raw.rfind(")")
        if command_end < 0:
            return None
        fields_after_command = raw[command_end + 1 :].split()
        token = fields_after_command[19]
        int(token)
        return token
    except (OSError, IndexError, ValueError):
        return None


def _allocated_bytes(paths: tuple[Path, ...]) -> int:
    """Return allocated bytes without following symlinks or double-counting.

    ``st_blocks`` reflects actual filesystem allocation (unlike apparent file
    size) and therefore handles sparse build artifacts correctly.  Inodes are
    de-duplicated so hard-linked package caches cannot inflate the reading.
    """

    seen: set[tuple[int, int]] = set()
    total = 0
    stack = list(paths)
    while stack:
        path = stack.pop()
        try:
            stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        total += stat.st_blocks * 512
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            with os.scandir(path) as entries:
                stack.extend(Path(entry.path) for entry in entries)
        except FileNotFoundError:
            # A concurrent build may remove a directory between stat/scandir.
            continue
    return total


def _distinct_roots(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw in paths:
        candidate = Path(raw).resolve()
        if any(candidate == root or candidate.is_relative_to(root) for root in roots):
            continue
        roots = [root for root in roots if not root.is_relative_to(candidate)]
        roots.append(candidate)
    return tuple(roots)


class DirectoryGrowthQuota:
    """Context manager enforcing a byte-growth cap on writable directories.

    The optional ``on_exceeded`` callback should terminate the corresponding
    container.  It runs at most once from the monitor thread.  On context exit
    a final synchronous measurement closes the normal-completion race, and a
    quota error replaces/chains any lower-level "container stopped" error.
    """

    def __init__(
        self,
        paths: Iterable[Path | str],
        *,
        limit_bytes: int,
        on_exceeded: Callable[[], None] | None = None,
        poll_seconds: float = 0.25,
        kernel_backend: KernelQuotaBackend | None = None,
        require_kernel: bool | None = None,
    ) -> None:
        if limit_bytes <= 0:
            raise ValueError("storage quota limit_bytes must be positive")
        if poll_seconds <= 0:
            raise ValueError("storage quota poll_seconds must be positive")
        self.paths = _distinct_roots(paths)
        if not self.paths:
            raise ValueError("storage quota requires at least one path")
        self.limit_bytes = int(limit_bytes)
        self.on_exceeded = on_exceeded
        self.poll_seconds = float(poll_seconds)
        self.kernel_backend = (
            kernel_backend
            if kernel_backend is not None
            else KernelQuotaBackend.from_environment()
        )
        if require_kernel is None:
            require_kernel = os.environ.get(
                "CRUCIBLE_REQUIRE_KERNEL_QUOTA", "",
            ).strip().lower() in {"1", "true", "yes", "on"}
        self.require_kernel = bool(require_kernel)
        if self.require_kernel and self.kernel_backend is None:
            raise StorageQuotaBackendUnavailable(
                "kernel-backed storage quota is required but "
                "CRUCIBLE_QUOTA_HELPER is not configured"
            )
        self.kernel_lease: KernelQuotaLease | None = None
        self.baseline_bytes = 0
        self.peak_growth_bytes = 0
        self._stop = threading.Event()
        self._tripped = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._monitor_entered = threading.Event()
        self._monitor_finished = threading.Event()
        self._monitor_failure: BaseException | None = None
        self._scan_error: OSError | None = None
        self._close_phase: _StorageQuotaClosePhase | None = None

    @property
    def exceeded(self) -> bool:
        return self._tripped.is_set()

    def _sample(self) -> None:
        invoke_callback = False
        with self._sample_lock:
            current = _allocated_bytes(self.paths)
            growth = max(0, current - self.baseline_bytes)
            self.peak_growth_bytes = max(self.peak_growth_bytes, growth)
            if growth > self.limit_bytes and not self._tripped.is_set():
                self._tripped.set()
                invoke_callback = self.on_exceeded is not None
        if invoke_callback and self.on_exceeded is not None:
            with self._callback_lock:
                try:
                    self.on_exceeded()
                except Exception:
                    # The quota violation is authoritative. Cleanup failures
                    # are reported by the owning container lifecycle.
                    pass

    def _run(self) -> None:
        self._monitor_entered.set()
        try:
            while not self._stop.wait(self.poll_seconds):
                try:
                    self._sample()
                except OSError as exc:
                    # Losing visibility into untrusted writable state defeats the
                    # quota.  Stop the sandbox immediately and surface the scan
                    # error at context exit.
                    self._scan_error = exc
                    self._tripped.set()
                    if self.on_exceeded is not None:
                        with self._callback_lock:
                            try:
                                self.on_exceeded()
                            except Exception:
                                pass
                    return
        except BaseException as exc:
            # Thread exceptions otherwise disappear after ``join``.  A dead
            # monitor is not positive sampling authority, even when the final
            # synchronous sample happens to be below the limit.
            self._monitor_failure = exc
            self._tripped.set()
        finally:
            self._monitor_finished.set()

    def __enter__(self) -> "DirectoryGrowthQuota":
        self.baseline_bytes = _allocated_bytes(self.paths)
        if self.kernel_backend is not None:
            self.kernel_backend.reconcile()
            self.kernel_lease = self.kernel_backend.reserve(
                self.paths,
                limit_bytes=self.baseline_bytes + self.limit_bytes,
            )
        self._thread = threading.Thread(
            target=self._run,
            name="crucible-storage-quota",
            daemon=True,
        )
        self._thread.start()
        return self

    def retire_kernel_lease(self) -> StorageQuotaSettlementReceipt | None:
        """Retry retirement after the guarded paths have been removed.

        The helper's ``release`` is deliberately two-phase: while a
        guarded path still exists it durably tombstones the lease and retains
        the hard quota.  Workspace owners call this method after their path
        teardown so ``retire`` can prove that the same opaque token reached
        the helper's durable terminal state instead of merely accepting a
        repeated idempotent release request.
        """

        if self.kernel_backend is None or self.kernel_lease is None:
            return None
        close_phase = self._close_phase
        if close_phase is None or not close_phase.kernel_release_proven:
            raise StorageQuotaCleanupError(
                "kernel quota retirement cannot attest an unproven initial release"
            )
        backend_fingerprint = _kernel_backend_argv_fingerprint(
            self.kernel_backend
        )
        lease_fingerprint = _kernel_lease_fingerprint(self.kernel_lease)
        if (
            close_phase.kernel_backend_argv_fingerprint != backend_fingerprint
            or close_phase.kernel_lease_fingerprint != lease_fingerprint
        ):
            raise StorageQuotaCleanupError(
                "kernel quota retirement bindings changed after close"
            )
        try:
            self.kernel_backend.retire(self.kernel_lease)
        except Exception as exc:
            if isinstance(exc, StorageQuotaCleanupError):
                raise
            raise StorageQuotaCleanupError(str(exc)) from exc
        post_backend_fingerprint = _kernel_backend_argv_fingerprint(
            self.kernel_backend
        )
        post_lease_fingerprint = _kernel_lease_fingerprint(self.kernel_lease)
        if (
            post_backend_fingerprint != backend_fingerprint
            or post_lease_fingerprint != lease_fingerprint
        ):
            raise StorageQuotaCleanupError(
                "kernel quota retirement bindings changed during proof"
            )

        return StorageQuotaSettlementReceipt(
            close=close_phase,
            kernel_backend_argv_fingerprint=post_backend_fingerprint,
            kernel_lease_fingerprint=post_lease_fingerprint,
        )

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        release_error: BaseException | None = None
        monitor_error: BaseException | None = None
        try:
            try:
                self._sample()
            except OSError as scan_exc:
                self._scan_error = scan_exc
                self._tripped.set()
        finally:
            self._stop.set()
            try:
                if self._thread is not None:
                    try:
                        self._thread.join(
                            timeout=max(2.0, self.poll_seconds * 2)
                        )
                        if self._thread.is_alive():
                            monitor_error = StorageQuotaCleanupError(
                                "storage quota monitor did not stop after join"
                            )
                        elif (
                            not self._monitor_entered.is_set()
                            or not self._monitor_finished.is_set()
                        ):
                            monitor_error = StorageQuotaCleanupError(
                                "storage quota monitor lacks exact terminal proof"
                            )
                        elif self._monitor_failure is not None:
                            monitor_error = StorageQuotaCleanupError(
                                "storage quota monitor failed before settlement: "
                                f"{type(self._monitor_failure).__name__}: "
                                f"{self._monitor_failure}"
                            )
                    except BaseException as join_exc:
                        monitor_error = join_exc
                else:
                    monitor_error = StorageQuotaCleanupError(
                        "storage quota monitor was never started"
                    )
            finally:
                if self.kernel_backend is not None and self.kernel_lease is not None:
                    try:
                        self.kernel_backend.release(self.kernel_lease)
                    except BaseException as backend_exc:
                        release_error = backend_exc
        if monitor_error is not None:
            error = (
                monitor_error
                if isinstance(monitor_error, StorageQuotaCleanupError)
                else StorageQuotaCleanupError(str(monitor_error))
            )
            if release_error is not None:
                error.add_note(
                    "kernel quota release also failed: "
                    f"{type(release_error).__name__}: {release_error}"
                )
            if exc is not None:
                raise error from exc
            if release_error is not None:
                raise error from release_error
            if error is monitor_error:
                raise error
            raise error from monitor_error
        if release_error is not None:
            error = (
                release_error
                if isinstance(release_error, StorageQuotaCleanupError)
                else StorageQuotaCleanupError(str(release_error))
            )
            if exc is not None:
                raise error from exc
            raise error
        backend_fingerprint: str | None = None
        lease_fingerprint: str | None = None
        release_proven = False
        if self.kernel_backend is not None and self.kernel_lease is not None:
            backend_fingerprint = _kernel_backend_argv_fingerprint(
                self.kernel_backend
            )
            lease_fingerprint = _kernel_lease_fingerprint(self.kernel_lease)
            release_proven = True
        enforcement_outcome = (
            "scan_failed"
            if self._scan_error is not None
            else "exceeded" if self.exceeded else "within_limit"
        )
        self._close_phase = _StorageQuotaClosePhase(
            paths=tuple(str(path) for path in self.paths),
            limit_bytes=self.limit_bytes,
            baseline_bytes=self.baseline_bytes,
            peak_growth_bytes=self.peak_growth_bytes,
            kernel_hard_limit_bytes=self.baseline_bytes + self.limit_bytes,
            enforcement_outcome=enforcement_outcome,
            require_kernel=self.require_kernel,
            kernel_backend_argv_fingerprint=backend_fingerprint,
            kernel_lease_fingerprint=lease_fingerprint,
            kernel_release_proven=release_proven,
        )
        if self._scan_error is not None:
            enforcement_error = StorageQuotaExceeded(
                f"sandbox storage quota scan failed closed: {self._scan_error}"
            )
            if exc is not None:
                raise enforcement_error from exc
            raise enforcement_error from self._scan_error
        if self.exceeded:
            enforcement_error = StorageQuotaExceeded(
                "sandbox writable storage growth exceeded cap: "
                f"peak={self.peak_growth_bytes} bytes, "
                f"limit={self.limit_bytes} bytes, "
                f"paths={[str(path) for path in self.paths]}"
            )
            if exc is not None:
                raise enforcement_error from exc
            raise enforcement_error
        return False


__all__ = [
    "DirectoryGrowthQuota",
    "KernelQuotaBackend",
    "KernelQuotaLease",
    "STORAGE_QUOTA_SETTLEMENT_RECEIPT_SCHEMA",
    "StorageQuotaBackendUnavailable",
    "StorageQuotaCleanupError",
    "StorageQuotaExceeded",
    "StorageQuotaSettlementReceipt",
]
