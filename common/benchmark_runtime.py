# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crash-safe primitives shared by benchmark workflow runtimes.

The benchmark adapters execute expensive, untrusted tasks.  This module keeps
the small pieces that must have identical semantics across SWE-bench and
DeepSWE in one place: canonical resume identities, cross-process task leases,
atomic budget reservations, process-tree termination, immutable Docker image
resolution, and orphan-container reconciliation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


class TaskLeaseBusy(RuntimeError):
    """Another process owns the task's execution/publication lease."""


class BudgetReservationError(RuntimeError):
    """A reservation cannot be made or settled without violating the cap."""


class ImmutableImageError(RuntimeError):
    """A mutable image reference could not be resolved and verified."""


_IMAGE_DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


def _canonicalize(value: Any) -> Any:
    """Return a JSON-safe, deterministic representation of ``value``."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ),
        )
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint identity cannot contain NaN or infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(
        f"unsupported checkpoint identity value: {type(value).__name__}"
    )


def canonical_fingerprint(value: Any) -> str:
    """SHA-256 of a canonical JSON projection of all behavior inputs."""

    payload = json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def file_digest(path: Path | str | None) -> dict[str, Any] | None:
    """Return a content identity for an optional configuration/code file."""

    if path is None:
        return None
    candidate = Path(path)
    payload = candidate.read_bytes()
    return {
        "name": candidate.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def file_set_fingerprint(root: Path | str, pattern: str) -> str:
    """Fingerprint a deterministic set of policy/recipe files."""

    base = Path(root)
    return canonical_fingerprint({
        str(path.relative_to(base)): file_digest(path)
        for path in sorted(base.glob(pattern))
        if path.is_file()
    })


@contextlib.contextmanager
def task_execution_lease(
    root: Path | str,
    *,
    namespace: str,
    task_id: str,
    blocking: bool = False,
) -> Iterator[Path]:
    """Hold a process-death-safe task lease until durable publication.

    The filename contains only hashes, so dataset-controlled task ids cannot
    escape the private lease directory.  ``flock`` is released by the kernel
    on process death, including SIGKILL/OOM.
    """

    lease_dir = Path(root) / ".benchmark-leases" / namespace
    lease_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    lease_path = lease_dir / f"{digest}.lock"
    fd = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, operation)
            acquired = True
        except BlockingIOError as exc:
            raise TaskLeaseBusy(
                f"{namespace} task {task_id!r} is already executing"
            ) from exc
        yield lease_path
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


@dataclasses.dataclass(frozen=True)
class Reservation:
    token: int
    amount: float


class BudgetReservations:
    """Thread-safe reserve/settle ledger for bounded parallel fan-out."""

    def __init__(self, cap: float | None, *, spent: float = 0.0) -> None:
        if cap is not None and (cap < 0.0 or not math.isfinite(cap)):
            raise ValueError("budget cap must be a finite non-negative number")
        if spent < 0.0 or not math.isfinite(spent):
            raise ValueError("spent must be a finite non-negative number")
        if cap is not None and spent > cap:
            raise BudgetReservationError("restored spend exceeds budget cap")
        self.cap = cap
        self._spent = float(spent)
        self._reserved: dict[int, float] = {}
        self._next_token = 1
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def reserved(self) -> float:
        with self._lock:
            return sum(self._reserved.values())

    @property
    def available(self) -> float | None:
        with self._lock:
            if self.cap is None:
                return None
            return max(0.0, self.cap - self._spent - sum(self._reserved.values()))

    def reserve(self, requested: float | None) -> Reservation:
        """Atomically reserve a launch ceiling.

        With an aggregate cap, ``requested=None`` means all currently
        available budget.  Without a cap, ``None`` is represented by an
        infinite reservation and settlement accepts any finite spend.
        """

        if requested is not None and (
            requested <= 0.0 or not math.isfinite(requested)
        ):
            raise BudgetReservationError("reservation must be positive")
        with self._lock:
            available = (
                math.inf
                if self.cap is None
                else self.cap - self._spent - sum(self._reserved.values())
            )
            amount = available if requested is None else min(requested, available)
            if amount <= 0.0:
                raise BudgetReservationError("aggregate budget is exhausted")
            token = self._next_token
            self._next_token += 1
            self._reserved[token] = amount
            return Reservation(token=token, amount=amount)

    def settle(self, reservation: Reservation, actual: float) -> None:
        """Release a reservation and atomically account its actual spend."""

        if actual < 0.0 or not math.isfinite(actual):
            raise BudgetReservationError("actual spend must be finite and non-negative")
        with self._lock:
            reserved = self._reserved.pop(reservation.token, None)
            if reserved is None or reserved != reservation.amount:
                raise BudgetReservationError("unknown or already-settled reservation")
            # Backends are expected to honor the child ceiling.  Do not hide a
            # violation by quietly charging beyond the amount granted.
            if actual > reserved and math.isfinite(reserved):
                self._spent += actual
                raise BudgetReservationError(
                    f"worker spent {actual:.6f} beyond reservation {reserved:.6f}"
                )
            if self.cap is not None and self._spent + actual > self.cap:
                self._spent += actual
                raise BudgetReservationError("settlement would exceed aggregate cap")
            self._spent += actual

    def cancel(self, reservation: Reservation) -> None:
        with self._lock:
            self._reserved.pop(reservation.token, None)


def run_process_tree(
    args: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command and kill its complete process group on timeout."""

    process = subprocess.Popen(
        list(args),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            exc.cmd, exc.timeout, output=stdout, stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(list(args), process.returncode, stdout, stderr)


DockerRunner = Callable[..., subprocess.CompletedProcess]


def resolve_immutable_image(image: str, *, run_docker: DockerRunner) -> str:
    """Resolve a pulled Docker tag to a repository digest and verify it.

    All subsequent ``create``/``run`` calls must use the returned ``@sha256``
    reference.  A local image id is deliberately insufficient because it does
    not bind the registry repository identity recorded in checkpoints.
    """

    if "@sha256:" in image:
        immutable = image
    else:
        result = run_docker(
            ["image", "inspect", "--format", "{{json .RepoDigests}}", image],
            step_name="resolve image digest",
            timeout=60,
        )
        raw = (result.stdout or "").strip()
        try:
            digests = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ImmutableImageError(
                f"Docker returned invalid RepoDigests for {image!r}"
            ) from exc
        if not isinstance(digests, list):
            raise ImmutableImageError(f"Docker returned no RepoDigests for {image!r}")
        repository = image.split("@", 1)[0]
        # A colon denotes a tag only when it occurs after the final slash;
        # registry ports (``localhost:5000/repo``) are part of the repository.
        last_colon = repository.rfind(":")
        if last_colon > repository.rfind("/"):
            repository = repository[:last_colon]
        matching = [
            item for item in digests
            if isinstance(item, str)
            and item.startswith(f"{repository}@sha256:")
        ]
        if not matching:
            raise ImmutableImageError(
                f"image {image!r} has no immutable digest for requested "
                f"repository {repository!r}; refusing a digest from another "
                "repository trust domain"
            )
        immutable = sorted(matching)[0]
    verify_immutable_image(immutable)
    return immutable


def image_verification_policy_identity() -> str:
    """Fingerprint the configured verifier command without executing a shell."""

    configured = os.environ.get("CRUCIBLE_IMAGE_VERIFY_COMMAND", "").strip()
    if not configured:
        return "disabled"
    try:
        command = shlex.split(configured)
    except ValueError as exc:
        raise ImmutableImageError(
            "CRUCIBLE_IMAGE_VERIFY_COMMAND is not valid shell-style argv"
        ) from exc
    if not command:
        return "disabled"
    return canonical_fingerprint({"command": command})


def verify_immutable_image(image_digest: str) -> None:
    """Validate an immutable reference and apply the current trust policy."""

    if not _IMAGE_DIGEST_RE.fullmatch(image_digest):
        raise ImmutableImageError(
            f"invalid immutable image digest {image_digest!r}"
        )
    verify_image_signature(image_digest)


def verify_image_signature(image_digest: str) -> None:
    """Run an optional operator-provided signature verifier hook.

    ``CRUCIBLE_IMAGE_VERIFY_COMMAND`` is parsed without a shell and receives
    the digest as its final argument.  Examples include ``cosign verify`` or
    a policy-engine wrapper.  A configured hook is fail-closed.
    """

    configured = os.environ.get("CRUCIBLE_IMAGE_VERIFY_COMMAND", "").strip()
    if not configured:
        return
    command = [*shlex.split(configured), image_digest]
    result = run_process_tree(command, timeout=120)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")[-2000:]
        raise ImmutableImageError(
            f"image signature verification failed for {image_digest}: {stderr}"
        )


def managed_container_labels(
    *, scope: str, task_id: str, phase: str,
) -> dict[str, str]:
    """Labels used for deterministic startup orphan reconciliation."""

    return {
        "io.crucible.managed": "true",
        "io.crucible.scope": canonical_fingerprint(scope).split(":", 1)[1],
        "io.crucible.task": canonical_fingerprint(task_id).split(":", 1)[1],
        "io.crucible.phase": phase,
        "io.crucible.owner-host": socket.gethostname(),
        "io.crucible.owner-pid": str(os.getpid()),
        "io.crucible.owner-start": _pid_start_token(os.getpid()) or "unavailable",
        "io.crucible.started-ns": str(time.time_ns()),
    }


def managed_container_name(
    *, prefix: str, scope: str, task_id: str, phase: str | None = None,
) -> str:
    """Return a Docker-global name scoped to one logical benchmark owner.

    Dataset-controlled identifiers never enter the name directly.  Scope and
    task hashes prevent independent spec roots from colliding in a shared
    Docker daemon; an optional phase distinguishes authoring and verification.
    """

    normalized_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "-", prefix).strip("-.")
    if not normalized_prefix:
        raise ValueError("managed container prefix must contain a safe character")
    scope_hash = canonical_fingerprint(scope).split(":", 1)[1][:16]
    task_hash = canonical_fingerprint(task_id).split(":", 1)[1][:20]
    normalized_phase = (
        re.sub(r"[^a-zA-Z0-9_.-]+", "-", phase).strip("-.")
        if phase else ""
    )
    suffix = f"-{normalized_phase}" if normalized_phase else ""
    return f"{normalized_prefix}-{scope_hash}-{task_hash}{suffix}"[:200]


def label_args(labels: Mapping[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in sorted(labels.items()):
        args.extend(["--label", f"{key}={value}"])
    return args


def _pid_start_token(pid: int) -> str | None:
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


def _pid_is_owner(pid: int, start_token: str | None) -> bool:
    if pid <= 0 or not start_token or start_token == "unavailable":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return _pid_start_token(pid) == start_token


def reconcile_orphaned_containers(
    *, scope: str, run_docker: DockerRunner,
) -> tuple[str, ...]:
    """Remove managed containers whose same-host owner process is dead."""

    scope_hash = canonical_fingerprint(scope).split(":", 1)[1]
    listed = run_docker(
        [
            "ps", "-aq",
            "--filter", "label=io.crucible.managed=true",
            "--filter", f"label=io.crucible.scope={scope_hash}",
        ],
        step_name="list managed containers",
        timeout=30,
    )
    removed: list[str] = []
    for container_id in (listed.stdout or "").splitlines():
        container_id = container_id.strip()
        if not container_id:
            continue
        inspected = run_docker(
            [
                "inspect", "--format",
                "{{json .Config.Labels}}", container_id,
            ],
            step_name="inspect managed container",
            timeout=30,
        )
        try:
            labels = json.loads((inspected.stdout or "{}").strip())
            owner_host = labels.get("io.crucible.owner-host")
            owner_pid = int(labels.get("io.crucible.owner-pid", "0"))
            owner_start = labels.get("io.crucible.owner-start")
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            # A managed container with malformed ownership metadata cannot be
            # proven live.  Fail closed and reclaim it.
            owner_host, owner_pid, owner_start = socket.gethostname(), 0, None
        if owner_host != socket.gethostname() or _pid_is_owner(
            owner_pid,
            str(owner_start) if owner_start is not None else None,
        ):
            continue
        run_docker(
            ["rm", "-f", container_id],
            step_name="reconcile orphan container",
            timeout=30,
        )
        removed.append(container_id)
    return tuple(removed)


__all__ = [
    "BudgetReservationError",
    "BudgetReservations",
    "ImmutableImageError",
    "Reservation",
    "TaskLeaseBusy",
    "canonical_fingerprint",
    "file_digest",
    "file_set_fingerprint",
    "image_verification_policy_identity",
    "label_args",
    "managed_container_name",
    "managed_container_labels",
    "reconcile_orphaned_containers",
    "resolve_immutable_image",
    "run_process_tree",
    "task_execution_lease",
    "verify_immutable_image",
    "verify_image_signature",
]
