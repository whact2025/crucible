# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded backend-neutral read-only child scheduler.

This module owns concurrency.  Provider adapters remain single-invocation
workers and never see or create sibling children.  Every requested child gets
fresh mutable state:

* one :class:`~crucible.tools.ToolContext`;
* one capability-classified read-only ToolSet;
* one backend instance (and therefore one adapter session);
* one explicit working-directory binding; and
* one private debug directory.

The caller supplies factories, but the scheduler validates object identity and
tool classification before transport.  It closes active backends to request
cancellation, drains all futures, joins the executor, and reports children in
request order on success and failure alike.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    Invocation,
    InvocationResult,
    Usage,
    UsagePricingEvidence,
)

from crucible.backends.capabilities import validate_toolset_capability
from crucible.tools import ToolContext

from ._backends import BackendCleanupError
from ._run_policy import OrchestrationPolicy


_CHILD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_POLL_SECONDS = 0.025
ChildStatus = Literal[
    "succeeded",
    "failed",
    "cancelled",
    "incomplete",
    "cleanup_failed",
]

ToolSetFactory = Callable[["ReadOnlyChildRequest", ToolContext], Any]
BackendFactory = Callable[["ReadOnlyChildRequest", Any, Path], Any]


class ReadOnlySchedulerError(RuntimeError):
    """Base class for deterministic scheduler admission failures."""


class ReadOnlyIsolationError(ReadOnlySchedulerError):
    """A factory attempted to share mutable state between children."""


@dataclass(frozen=True, slots=True)
class ReadOnlyChildRequest:
    """One logical read-only child.

    ``role`` is the logical attribution name.  ``route_role`` selects a
    qualified existing backend route and must be one of the policy's eligible
    parent roles.  A child cannot create new routing authority by choosing its
    own logical name.
    """

    child_id: str
    role: str
    route_role: str
    cwd: Path
    invocation: Invocation

    def __post_init__(self) -> None:
        if (
            not isinstance(self.child_id, str)
            or _CHILD_ID_RE.fullmatch(self.child_id) is None
        ):
            raise ValueError(
                "ReadOnlyChildRequest.child_id must be a portable 1-64 "
                "character identifier"
            )
        for field_name in ("role", "route_role"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ReadOnlyChildRequest.{field_name} must be non-empty")
        if not isinstance(self.cwd, Path):
            raise TypeError("ReadOnlyChildRequest.cwd must be pathlib.Path")
        if not isinstance(self.invocation, Invocation):
            raise TypeError("ReadOnlyChildRequest.invocation must be assay.Invocation")
        # The scheduler-attached ToolSet is the only child tool authority.
        # Invocation.tools lacks Crucible's read/write classification, so
        # accepting a second descriptor surface would make the no-write claim
        # unprovable.
        if self.invocation.tools is not None:
            raise ValueError(
                "read-only scheduler children must use only the attached "
                "classified ToolSet; Invocation.tools must be None"
            )


@dataclass(frozen=True, slots=True)
class ReadOnlyChildOutcome:
    child_id: str
    role: str
    route_role: str
    status: ChildStatus
    result: InvocationResult | None
    backend_name: str | None
    model: str | None
    cwd: str
    debug_dir: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    token_usage_complete: bool
    cost_complete: bool
    terminal_complete: bool
    trace_complete: bool
    cleanup_complete: bool
    error_kind: str | None = None
    error_message: str | None = None
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    pricing_evidence: UsagePricingEvidence | None = None

    @property
    def evidence_complete(self) -> bool:
        return (
            self.token_usage_complete
            and self.cost_complete
            and self.terminal_complete
            and self.trace_complete
            and self.cleanup_complete
        )

    @property
    def successful(self) -> bool:
        return self.status == "succeeded" and self.evidence_complete

    @property
    def has_extended_pricing_evidence(self) -> bool:
        return (
            self.cache_creation_tokens != 0
            or self.cache_read_tokens != 0
            or self.pricing_evidence is not None
        )

    def to_summary(
        self,
        *,
        include_extended_pricing: bool | None = None,
    ) -> dict[str, Any]:
        final_message = self.result.final_message if self.result is not None else ""
        message_bytes = final_message.encode("utf-8")
        summary: dict[str, Any] = {
            "child_id": self.child_id,
            "role": self.role,
            "route_role": self.route_role,
            "status": self.status,
            "backend": self.backend_name,
            "model": self.model,
            "cwd": self.cwd,
            "debug_dir": self.debug_dir,
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_usage_complete": self.token_usage_complete,
            "cost_complete": self.cost_complete,
            "terminal_complete": self.terminal_complete,
            "trace_complete": self.trace_complete,
            "cleanup_complete": self.cleanup_complete,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "final_message_bytes": len(message_bytes),
            "final_message_digest": (
                "sha256:" + hashlib.sha256(message_bytes).hexdigest()
            ),
        }
        if include_extended_pricing is None:
            include_extended_pricing = self.has_extended_pricing_evidence
        if include_extended_pricing:
            summary["cache_creation_tokens"] = self.cache_creation_tokens
            summary["cache_read_tokens"] = self.cache_read_tokens
            summary["pricing_evidence"] = _pricing_evidence_summary(
                self.pricing_evidence
            )
        return summary


@dataclass(frozen=True, slots=True)
class ReadOnlyFanoutResult:
    """Joined scheduler result in deterministic request order."""

    children: tuple[ReadOnlyChildOutcome, ...]
    usage: Usage
    trace: BackendExecutionTrace
    joined_text: str
    status: Literal[
        "succeeded",
        "failed",
        "cancelled",
        "join_limit_exceeded",
    ]
    all_joined: bool
    cancellation_drained: bool
    join_error: str | None = None

    @property
    def evidence_complete(self) -> bool:
        return (
            self.all_joined
            and self.cancellation_drained
            and all(child.evidence_complete for child in self.children)
        )

    @property
    def successful(self) -> bool:
        return self.status == "succeeded" and self.evidence_complete

    def to_summary(self) -> dict[str, Any]:
        joined_bytes = self.joined_text.encode("utf-8")
        has_extended_pricing = any(
            child.has_extended_pricing_evidence for child in self.children
        )
        summary = {
            "schema_version": (
                "crucible.read-only-fanout-result.v2"
                if has_extended_pricing
                else "crucible.read-only-fanout-result.v1"
            ),
            "status": self.status,
            "child_count": len(self.children),
            "all_joined": self.all_joined,
            "cancellation_drained": self.cancellation_drained,
            "evidence_complete": self.evidence_complete,
            "cost_usd": round(self.usage.cost_usd, 6),
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "joined_output_bytes": len(joined_bytes),
            "joined_output_digest": (
                "sha256:" + hashlib.sha256(joined_bytes).hexdigest()
            ),
            "join_error": self.join_error,
            "events": [
                {
                    "sequence": event.sequence,
                    "kind": event.kind,
                    "name": event.name,
                    "parent_id": event.parent_id,
                    "child_id": event.child_id,
                }
                for event in self.trace.events
            ],
            "children": [
                child.to_summary(
                    include_extended_pricing=has_extended_pricing,
                )
                for child in self.children
            ],
        }
        if has_extended_pricing:
            summary["cache_creation_tokens"] = self.usage.cache_creation_tokens
            summary["cache_read_tokens"] = self.usage.cache_read_tokens
        return summary


def _pricing_evidence_summary(
    pricing_evidence: UsagePricingEvidence | None,
) -> dict[str, Any] | None:
    """Project typed pricing evidence without accepting arbitrary keys."""

    if pricing_evidence is None:
        return None
    return {
        "segments": [
            {
                "input_tokens": segment.input_tokens,
                "output_tokens": segment.output_tokens,
                "cache_creation_tokens": segment.cache_creation_tokens,
                "cache_read_tokens": segment.cache_read_tokens,
                "cost_usd": segment.cost_usd,
            }
            for segment in pricing_evidence.segments
        ],
        "complete": pricing_evidence.complete,
    }


class _ManagedBackend:
    """Idempotent close gate shared by coordinator and worker."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._lock = threading.Lock()
        self._closed = False
        self._close_errors: list[BaseException] = []

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def close_error(self) -> BaseException | None:
        with self._lock:
            errors = tuple(self._close_errors)
        if not errors:
            return None
        if len(errors) == 1:
            return errors[0]
        return BackendCleanupError(
            tuple(
                (f"backend close attempt {index}", error)
                for index, error in enumerate(errors, start=1)
            ),
            retry_cleanup=self.retry_cleanup,
        )

    def retry_cleanup(self) -> None:
        """Retry close and raise an actionable aggregate if still owned."""

        self.close()
        if self.closed:
            return
        error = self.close_error
        if isinstance(error, BackendCleanupError):
            raise error
        raise BackendCleanupError(
            (("backend close retry", error or RuntimeError("close failed")),),
            retry_cleanup=self.retry_cleanup,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            close = getattr(self.backend, "close", None)
            if close is None:
                self._closed = True
                return
            try:
                close()
            except BaseException as exc:  # noqa: BLE001 - child evidence
                self._close_errors.append(exc)
            else:
                self._closed = True


class ReadOnlyCleanupError(ReadOnlySchedulerError):
    """A completed fan-out still owns one or more exclusive backends.

    ``partial_result`` retains all normalized child usage and trace evidence.
    The caller owns this exception until :meth:`retry_cleanup` succeeds (or
    raises a replacement ``ReadOnlyCleanupError`` with the same evidence).
    """

    def __init__(
        self,
        *,
        partial_result: ReadOnlyFanoutResult,
        retained_backends: tuple[tuple[str, _ManagedBackend], ...],
        primary_error: BaseException | None = None,
    ) -> None:
        retained_backends = tuple(
            (child_id, managed)
            for child_id, managed in retained_backends
            if not managed.closed
        )
        if not retained_backends:
            raise ValueError(
                "ReadOnlyCleanupError requires live backend cleanup authority"
            )
        self.partial_result = partial_result
        self._retained_backends = retained_backends
        self.primary_error = primary_error
        details = "; ".join(
            f"{child_id}: {managed.close_error}"
            for child_id, managed in retained_backends
        )
        primary = (
            f" after {type(primary_error).__name__}: {primary_error}"
            if primary_error is not None
            else ""
        )
        super().__init__(
            "read-only fan-out cleanup retained exclusive backend ownership"
            f"{primary}: "
            f"{details}"
        )

    @property
    def retained_child_ids(self) -> tuple[str, ...]:
        """Stable child IDs whose exclusive backend remains owned."""

        return tuple(
            child_id
            for child_id, managed in self._retained_backends
            if not managed.closed
        )

    def retry_cleanup(self) -> ReadOnlyFanoutResult:
        """Retry every retained backend and return the partial result on drain."""

        retained: list[tuple[str, _ManagedBackend]] = []
        for child_id, managed in self._retained_backends:
            if managed.closed:
                continue
            managed.close()
            if not managed.closed:
                retained.append((child_id, managed))
        if retained:
            raise ReadOnlyCleanupError(
                partial_result=self.partial_result,
                retained_backends=tuple(retained),
                primary_error=self.primary_error,
            ) from self
        return replace(
            self.partial_result,
            cancellation_drained=True,
        )


def _validated_workspace(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ReadOnlySchedulerError(
            f"child working directory is unavailable: {path}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise ReadOnlySchedulerError(
            f"child working directory is not a directory: {resolved}"
        )
    return resolved


def _validate_read_only_toolset(toolset: Any) -> None:
    validate_toolset_capability("read_only", toolset)
    read_names = getattr(toolset, "read_only_names", None)
    write_names = getattr(toolset, "write_names", None)
    if read_names is None or write_names is None:
        raise ReadOnlySchedulerError(
            "read-only child ToolSet must expose complete read_only_names "
            "and write_names metadata"
        )
    descriptors = toolset.to_anthropic_tools()
    if not isinstance(descriptors, list):
        raise ReadOnlySchedulerError(
            "read-only child ToolSet descriptors must be a list"
        )
    try:
        descriptor_names = {entry["name"] for entry in descriptors}
    except (KeyError, TypeError) as exc:
        raise ReadOnlySchedulerError(
            "read-only child ToolSet descriptors require names"
        ) from exc
    if any(not isinstance(name, str) or not name for name in descriptor_names):
        raise ReadOnlySchedulerError(
            "read-only child ToolSet descriptor names must be non-empty strings"
        )
    classified = {str(name) for name in read_names} | {
        str(name) for name in write_names
    }
    if descriptor_names != classified:
        raise ReadOnlySchedulerError(
            "read-only child ToolSet classification does not exactly match "
            "its descriptors"
        )
    if write_names:
        # ``validate_toolset_capability`` already rejects this.  Keep the
        # explicit branch as a defense if a custom iterable changes between
        # property reads.
        raise ReadOnlySchedulerError("read-only child ToolSet exposes mutating tools")


def _zero_outcome(
    request: ReadOnlyChildRequest,
    *,
    debug_dir: Path,
    status: ChildStatus,
    error_kind: str,
    error_message: str,
    cleanup_complete: bool = True,
) -> ReadOnlyChildOutcome:
    # No backend was invoked, so zero usage is complete and attributable.
    # There is no child adapter trace to complete; the scheduler lifecycle
    # trace remains complete, but adapter trace completeness is deliberately
    # false for a rejected/failed child.
    cancelled_before_transport = status == "cancelled"
    return ReadOnlyChildOutcome(
        child_id=request.child_id,
        role=request.role,
        route_role=request.route_role,
        status=status,
        result=None,
        backend_name=None,
        model=None,
        cwd=str(request.cwd),
        debug_dir=str(debug_dir),
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        token_usage_complete=True,
        cost_complete=True,
        terminal_complete=True,
        trace_complete=cancelled_before_transport,
        cleanup_complete=cleanup_complete,
        error_kind=error_kind,
        error_message=error_message[:2000],
    )


def _usage_values(result: InvocationResult) -> tuple[float, int, int, int, int]:
    usage = result.usage
    cost = usage.cost_usd
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cache_creation_tokens = usage.cache_creation_tokens
    cache_read_tokens = usage.cache_read_tokens
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0.0
    ):
        raise ValueError("child usage.cost_usd is not finite and non-negative")
    for name, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cache_creation_tokens", cache_creation_tokens),
        ("cache_read_tokens", cache_read_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"child usage.{name} is not an exact non-negative integer")
    if cache_creation_tokens + cache_read_tokens > input_tokens:
        raise ValueError(
            "child usage cache-token subsets exceed input_tokens"
        )
    return (
        float(cost),
        input_tokens,
        output_tokens,
        cache_creation_tokens,
        cache_read_tokens,
    )


def _result_outcome(
    request: ReadOnlyChildRequest,
    *,
    result: InvocationResult,
    backend: Any,
    debug_dir: Path,
    max_output_bytes: int,
    cleanup_complete: bool,
    cleanup_error: BaseException | None,
) -> ReadOnlyChildOutcome:
    (
        cost,
        input_tokens,
        output_tokens,
        cache_creation_tokens,
        cache_read_tokens,
    ) = _usage_values(result)
    completeness = result.completeness
    error = result.error
    error_kind = getattr(error, "kind", None)
    error_message = getattr(error, "message", None)
    if cleanup_error is not None:
        status: ChildStatus = "cleanup_failed"
        error_kind = type(cleanup_error).__name__
        error_message = str(cleanup_error)
    elif error_kind == "cancellation":
        status = "cancelled"
    elif error is not None:
        status = "failed"
    elif len(result.final_message.encode("utf-8")) > max_output_bytes:
        status = "failed"
        error_kind = "ChildOutputLimitExceeded"
        error_message = (
            f"child final message exceeds max_child_output_bytes={max_output_bytes}"
        )
    elif not completeness.complete:
        status = "incomplete"
        error_kind = "IncompleteChildEvidence"
        error_message = "child result lacks complete token/cost/terminal/trace evidence"
    else:
        status = "succeeded"
    return ReadOnlyChildOutcome(
        child_id=request.child_id,
        role=request.role,
        route_role=request.route_role,
        status=status,
        result=result,
        backend_name=str(getattr(backend, "name", "unknown")),
        model=(
            str(backend.model)
            if getattr(backend, "model", None) is not None
            else None
        ),
        cwd=str(request.cwd),
        debug_dir=str(debug_dir),
        cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_usage_complete=completeness.token_usage_complete,
        cost_complete=completeness.cost_complete,
        terminal_complete=completeness.terminal_complete,
        trace_complete=completeness.trace_complete,
        cleanup_complete=cleanup_complete,
        error_kind=error_kind,
        error_message=(
            str(error_message)[:2000] if error_message is not None else None
        ),
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        pricing_evidence=result.pricing_evidence,
    )


def _reserved_invocation(
    request: ReadOnlyChildRequest,
    *,
    policy: OrchestrationPolicy,
    parent_budget_usd: float | None,
    child_count: int,
) -> Invocation:
    invocation = request.invocation
    child_budget = invocation.budget_usd
    if parent_budget_usd is not None:
        total_slot = parent_budget_usd * policy.max_total_cost_share / child_count
        individual_slot = parent_budget_usd * policy.max_child_cost_share
        reserved = min(total_slot, individual_slot)
        child_budget = reserved if child_budget is None else min(child_budget, reserved)
    return replace(
        invocation,
        max_turns=min(invocation.max_turns, policy.max_child_turns),
        budget_usd=child_budget,
    )


def _joined_text(
    outcomes: tuple[ReadOnlyChildOutcome, ...],
) -> str:
    parts = [
        "<read_only_child_analysis "
        f"child_id={outcome.child_id!r} role={outcome.role!r}>\n"
        f"{outcome.result.final_message}\n"
        "</read_only_child_analysis>"
        for outcome in outcomes
        if outcome.result is not None and outcome.status == "succeeded"
    ]
    return "\n\n".join(parts)


def run_read_only_fanout(
    requests: tuple[ReadOnlyChildRequest, ...],
    *,
    policy: OrchestrationPolicy,
    backend_factory: BackendFactory,
    toolset_factory: ToolSetFactory,
    debug_root: Path,
    parent_budget_usd: float | None = None,
    cancel_event: threading.Event | None = None,
) -> ReadOnlyFanoutResult:
    """Execute and join a bounded set of read-only backend children.

    Child transport/tool/backend failures become deterministic child outcomes
    so cost and cleanup remain inspectable. Admission/configuration errors
    that occur before an executor exists raise
    :class:`ReadOnlySchedulerError`. If an exclusive backend remains owned
    after the executor joins and internal cleanup retries finish, the function
    raises :class:`ReadOnlyCleanupError`; its ``partial_result`` preserves
    normalized cost/trace evidence and ``retry_cleanup()`` retains authority.
    """

    if policy.mode != "read_only":
        raise ReadOnlySchedulerError(
            "read-only scheduler requires OrchestrationPolicy.mode='read_only'"
        )
    if not requests:
        raise ReadOnlySchedulerError("read-only scheduler requires at least one child")
    if len(requests) > policy.max_children:
        raise ReadOnlySchedulerError(
            f"requested {len(requests)} children exceeds max_children="
            f"{policy.max_children}"
        )
    ids = tuple(request.child_id for request in requests)
    if len(ids) != len(set(ids)):
        raise ReadOnlySchedulerError("read-only child IDs must be unique")
    for request in requests:
        if request.role not in policy.child_roles:
            raise ReadOnlySchedulerError(
                f"child role {request.role!r} is not eligible under policy"
            )
        if request.route_role not in policy.parent_roles:
            raise ReadOnlySchedulerError(
                f"child route role {request.route_role!r} is not eligible under policy"
            )
    if parent_budget_usd is not None and (
        isinstance(parent_budget_usd, bool)
        or not isinstance(parent_budget_usd, (int, float))
        or not math.isfinite(parent_budget_usd)
        or parent_budget_usd < 0.0
    ):
        raise ReadOnlySchedulerError(
            "parent_budget_usd must be finite and non-negative"
        )
    if parent_budget_usd is None and any(
        request.invocation.budget_usd is None for request in requests
    ):
        raise ReadOnlySchedulerError(
            "read-only children require a finite parent budget or an "
            "explicit budget on every child invocation"
        )

    try:
        resolved_debug_root = debug_root.expanduser().resolve()
        resolved_debug_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ReadOnlySchedulerError(
            f"cannot create read-only scheduler debug root: {exc}"
        ) from exc
    workspace_by_id = {
        request.child_id: _validated_workspace(request.cwd) for request in requests
    }
    debug_by_id = {
        request.child_id: resolved_debug_root / request.child_id for request in requests
    }
    for child_id, debug_dir in debug_by_id.items():
        try:
            debug_dir.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise ReadOnlySchedulerError(
                f"child debug state already exists for {child_id!r}: {debug_dir}"
            ) from exc
        except OSError as exc:
            raise ReadOnlySchedulerError(
                f"cannot create child debug state for {child_id!r}: {exc}"
            ) from exc

    internal_cancel = threading.Event()
    state_lock = threading.Lock()
    claimed_backends: list[Any] = []
    claimed_toolsets: list[Any] = []
    claimed_contexts: list[ToolContext] = []
    active: dict[str, _ManagedBackend] = {}

    def cancellation_requested() -> bool:
        return internal_cancel.is_set() or (
            cancel_event is not None and cancel_event.is_set()
        )

    def claim_unique(bucket: list[Any], value: Any, *, kind: str) -> None:
        with state_lock:
            if any(prior is value for prior in bucket):
                raise ReadOnlyIsolationError(
                    f"{kind} object was shared across read-only children"
                )
            bucket.append(value)

    def run_child(request: ReadOnlyChildRequest) -> ReadOnlyChildOutcome:
        debug_dir = debug_by_id[request.child_id]
        workspace = workspace_by_id[request.child_id]
        if cancellation_requested():
            return _zero_outcome(
                request,
                debug_dir=debug_dir,
                status="cancelled",
                error_kind="SchedulerCancellation",
                error_message="cancelled before backend construction",
            )

        managed: _ManagedBackend | None = None
        backend: Any | None = None
        result: InvocationResult | None = None
        try:
            context = ToolContext(
                workspace_root=workspace,
                repo_root=workspace,
                context_budget=policy.max_child_context_bytes,
            )
            claim_unique(claimed_contexts, context, kind="ToolContext")
            toolset = toolset_factory(request, context)
            _validate_read_only_toolset(toolset)
            claim_unique(claimed_toolsets, toolset, kind="ToolSet")
            backend = backend_factory(request, toolset, debug_dir)
            claim_unique(claimed_backends, backend, kind="backend")
            managed = _ManagedBackend(backend)
            with state_lock:
                active[request.child_id] = managed
            if cancellation_requested():
                managed.close()
                return _zero_outcome(
                    request,
                    debug_dir=debug_dir,
                    status="cancelled",
                    error_kind="SchedulerCancellation",
                    error_message="cancelled before backend invocation",
                    cleanup_complete=managed.close_error is None,
                )
            invocation = _reserved_invocation(
                request,
                policy=policy,
                parent_budget_usd=parent_budget_usd,
                child_count=len(requests),
            )
            candidate = backend.invoke(invocation)
            if not isinstance(candidate, InvocationResult):
                raise TypeError(
                    "read-only child backend returned a non-InvocationResult"
                )
            result = candidate
        except Exception as exc:  # noqa: BLE001 - normalized child terminal
            if managed is not None:
                managed.close()
            cleanup_error = managed.close_error if managed is not None else None
            if backend is None and cleanup_error is None:
                return _zero_outcome(
                    request,
                    debug_dir=debug_dir,
                    status=("cancelled" if cancellation_requested() else "failed"),
                    error_kind=type(exc).__name__,
                    error_message=str(exc),
                )
            return ReadOnlyChildOutcome(
                child_id=request.child_id,
                role=request.role,
                route_role=request.route_role,
                status=(
                    "cleanup_failed"
                    if cleanup_error is not None
                    else "cancelled"
                    if cancellation_requested()
                    else "failed"
                ),
                result=result,
                backend_name=(
                    str(getattr(backend, "name", "unknown"))
                    if backend is not None
                    else None
                ),
                model=(
                    str(backend.model)
                    if backend is not None
                    and getattr(backend, "model", None) is not None
                    else None
                ),
                cwd=str(workspace),
                debug_dir=str(debug_dir),
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=True,
                trace_complete=False,
                cleanup_complete=cleanup_error is None,
                error_kind=(
                    type(cleanup_error).__name__
                    if cleanup_error is not None
                    else type(exc).__name__
                ),
                error_message=(
                    str(cleanup_error if cleanup_error is not None else exc)[:2000]
                ),
            )
        finally:
            if managed is not None:
                managed.close()
                if not managed.closed:
                    managed.close()
                if managed.closed:
                    with state_lock:
                        active.pop(request.child_id, None)

        assert result is not None
        assert backend is not None
        assert managed is not None
        try:
            return _result_outcome(
                request,
                result=result,
                backend=backend,
                debug_dir=debug_dir,
                max_output_bytes=policy.max_child_output_bytes,
                cleanup_complete=managed.close_error is None,
                cleanup_error=managed.close_error,
            )
        except Exception as exc:  # noqa: BLE001 - malformed typed evidence
            return ReadOnlyChildOutcome(
                child_id=request.child_id,
                role=request.role,
                route_role=request.route_role,
                status="failed",
                result=result,
                backend_name=str(getattr(backend, "name", "unknown")),
                model=(
                    str(backend.model)
                    if getattr(backend, "model", None) is not None
                    else None
                ),
                cwd=str(workspace),
                debug_dir=str(debug_dir),
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=True,
                trace_complete=False,
                cleanup_complete=managed.close_error is None,
                error_kind=type(exc).__name__,
                error_message=str(exc)[:2000],
            )

    def cancel_and_drain(futures: tuple[Future, ...]) -> None:
        internal_cancel.set()
        for future in futures:
            future.cancel()
        with state_lock:
            active_snapshot = tuple(active.values())
        for managed in active_snapshot:
            managed.close()

    def retry_active_cleanup() -> None:
        with state_lock:
            active_snapshot = tuple(active.items())
        for child_id, managed in active_snapshot:
            managed.close()
            if managed.closed:
                with state_lock:
                    if active.get(child_id) is managed:
                        active.pop(child_id, None)

    futures_by_id: dict[str, Future[ReadOnlyChildOutcome]] = {}
    outcomes_by_id: dict[str, ReadOnlyChildOutcome] = {}
    cancellation_started = False
    executor = ThreadPoolExecutor(
        max_workers=min(policy.max_concurrency, len(requests)),
        thread_name_prefix="crucible-readonly",
    )
    all_futures: tuple[Future[ReadOnlyChildOutcome], ...] = ()
    fatal_error: BaseException | None = None
    try:
        for request in requests:
            futures_by_id[request.child_id] = executor.submit(
                run_child,
                request,
            )
        all_futures = tuple(futures_by_id.values())
        pending: set[Future[ReadOnlyChildOutcome]] = set(all_futures)
        while pending:
            if (
                not cancellation_started
                and cancel_event is not None
                and cancel_event.is_set()
            ):
                cancellation_started = True
                cancel_and_drain(all_futures)
            done, still_pending = wait(
                pending,
                timeout=_POLL_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            pending = set(still_pending)
            for future in done:
                child_id = next(
                    key
                    for key, candidate in futures_by_id.items()
                    if candidate is future
                )
                request = next(item for item in requests if item.child_id == child_id)
                if future.cancelled():
                    outcome = _zero_outcome(
                        request,
                        debug_dir=debug_by_id[child_id],
                        status="cancelled",
                        error_kind="SchedulerCancellation",
                        error_message="cancelled before worker start",
                    )
                else:
                    outcome = future.result()
                outcomes_by_id[child_id] = outcome
                if (
                    policy.fail_fast
                    and not outcome.successful
                    and not cancellation_started
                ):
                    cancellation_started = True
                    cancel_and_drain(all_futures)
        # ``wait`` + executor.shutdown(wait=True) below is the explicit drain
        # and join.  No child thread survives a normal return.
    except BaseException as error:
        cancellation_started = True
        cancel_and_drain(all_futures)
        wait(all_futures)
        fatal_error = error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        retry_active_cleanup()

    if fatal_error is not None:
        with state_lock:
            retained_backends = tuple(active.items())
            retained_by_id = dict(retained_backends)
        if not retained_backends:
            raise fatal_error

        # Preserve every result that did reach a typed child boundary. A
        # BaseException that escaped a provider invocation has unknown
        # usage/terminal evidence, so its synthetic child is explicitly
        # incomplete rather than claiming an attributable zero-cost call.
        partial_outcomes: list[ReadOnlyChildOutcome] = []
        for request in requests:
            partial_outcome = outcomes_by_id.get(request.child_id)
            partial_future = futures_by_id.get(request.child_id)
            if (
                partial_outcome is None
                and partial_future is not None
                and partial_future.cancelled()
            ):
                partial_outcome = _zero_outcome(
                    request,
                    debug_dir=debug_by_id[request.child_id],
                    status="cancelled",
                    error_kind="SchedulerCancellation",
                    error_message="cancelled while draining a fatal child",
                )
            elif partial_outcome is None and partial_future is not None:
                try:
                    partial_outcome = partial_future.result()
                except BaseException as child_error:
                    managed = retained_by_id.get(request.child_id)
                    backend = managed.backend if managed is not None else None
                    partial_outcome = ReadOnlyChildOutcome(
                        child_id=request.child_id,
                        role=request.role,
                        route_role=request.route_role,
                        status=(
                            "cleanup_failed"
                            if managed is not None
                            else "failed"
                        ),
                        result=None,
                        backend_name=(
                            str(getattr(backend, "name", "unknown"))
                            if backend is not None
                            else None
                        ),
                        model=(
                            str(backend.model)
                            if backend is not None
                            and getattr(backend, "model", None) is not None
                            else None
                        ),
                        cwd=str(workspace_by_id[request.child_id]),
                        debug_dir=str(debug_by_id[request.child_id]),
                        cost_usd=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        token_usage_complete=False,
                        cost_complete=False,
                        terminal_complete=False,
                        trace_complete=False,
                        cleanup_complete=managed is None,
                        error_kind=type(child_error).__name__,
                        error_message=str(child_error)[:2000],
                    )
            elif partial_outcome is None:
                partial_outcome = _zero_outcome(
                    request,
                    debug_dir=debug_by_id[request.child_id],
                    status="cancelled",
                    error_kind="SchedulerCancellation",
                    error_message="fatal unwind occurred before worker submission",
                )
            if (
                request.child_id in retained_by_id
                and partial_outcome.cleanup_complete
            ):
                partial_outcome = replace(
                    partial_outcome,
                    status="cleanup_failed",
                    cleanup_complete=False,
                    error_kind="BackendCleanupError",
                    error_message=str(
                        retained_by_id[request.child_id].close_error
                    )[:2000],
                )
            partial_outcomes.append(partial_outcome)
        children = tuple(partial_outcomes)
        partial_events: list[BackendExecutionEvent] = []
        for child_outcome in children:
            partial_events.append(
                BackendExecutionEvent(
                    sequence=len(partial_events),
                    kind="child_start",
                    name=child_outcome.role,
                    parent_id=child_outcome.route_role,
                    child_id=child_outcome.child_id,
                )
            )
            partial_events.append(
                BackendExecutionEvent(
                    sequence=len(partial_events),
                    kind="child_terminal",
                    name=child_outcome.status,
                    parent_id=child_outcome.route_role,
                    child_id=child_outcome.child_id,
                )
            )
        partial_events.append(
            BackendExecutionEvent(
                sequence=len(partial_events),
                kind="terminal",
                name="failed",
            )
        )
        partial_result = ReadOnlyFanoutResult(
            children=children,
            usage=Usage(
                input_tokens=sum(child.input_tokens for child in children),
                output_tokens=sum(child.output_tokens for child in children),
                cost_usd=sum(child.cost_usd for child in children),
                cache_creation_tokens=sum(
                    child.cache_creation_tokens for child in children
                ),
                cache_read_tokens=sum(
                    child.cache_read_tokens for child in children
                ),
            ),
            trace=BackendExecutionTrace(
                backend="crucible_scheduler",
                requested_model="backend-neutral",
                events=tuple(partial_events),
            ),
            joined_text="",
            status="failed",
            all_joined=True,
            cancellation_drained=False,
            join_error=(
                "fatal child exception retained backend cleanup ownership"
            ),
        )
        raise ReadOnlyCleanupError(
            partial_result=partial_result,
            retained_backends=retained_backends,
            primary_error=fatal_error,
        ) from fatal_error

    # Futures canceled while another failure was being processed may not have
    # appeared in a FIRST_COMPLETED batch before the pending set emptied.
    for request in requests:
        if request.child_id in outcomes_by_id:
            continue
        future = futures_by_id[request.child_id]
        if future.cancelled():
            outcomes_by_id[request.child_id] = _zero_outcome(
                request,
                debug_dir=debug_by_id[request.child_id],
                status="cancelled",
                error_kind="SchedulerCancellation",
                error_message="cancelled before worker start",
            )
        else:
            outcomes_by_id[request.child_id] = future.result()

    outcomes = tuple(outcomes_by_id[request.child_id] for request in requests)
    if parent_budget_usd is not None:
        budget_checked: list[ReadOnlyChildOutcome] = []
        for request, outcome in zip(requests, outcomes, strict=True):
            reserved_budget = _reserved_invocation(
                request,
                policy=policy,
                parent_budget_usd=parent_budget_usd,
                child_count=len(requests),
            ).budget_usd
            if (
                reserved_budget is not None
                and outcome.cost_usd > reserved_budget + 1e-12
            ):
                outcome = replace(
                    outcome,
                    status="failed",
                    error_kind="ChildBudgetShareExceeded",
                    error_message=(
                        f"child cost ${outcome.cost_usd:.6f} exceeded its "
                        f"reserved ${reserved_budget:.6f} scheduler share"
                    ),
                )
            budget_checked.append(outcome)
        outcomes = tuple(budget_checked)
    joined = _joined_text(outcomes)
    join_error = None
    if len(joined.encode("utf-8")) > policy.max_joined_output_bytes:
        join_error = (
            "joined child output exceeds max_joined_output_bytes="
            f"{policy.max_joined_output_bytes}"
        )
        joined = ""

    if join_error is not None:
        aggregate_status: Literal[
            "succeeded",
            "failed",
            "cancelled",
            "join_limit_exceeded",
        ] = "join_limit_exceeded"
    elif all(outcome.successful for outcome in outcomes):
        aggregate_status = "succeeded"
    elif any(outcome.status == "cancelled" for outcome in outcomes):
        aggregate_status = "cancelled"
    else:
        aggregate_status = "failed"

    events: list[BackendExecutionEvent] = []
    for outcome in outcomes:
        events.append(
            BackendExecutionEvent(
                sequence=len(events),
                kind="child_start",
                name=outcome.role,
                parent_id=outcome.route_role,
                child_id=outcome.child_id,
            )
        )
        events.append(
            BackendExecutionEvent(
                sequence=len(events),
                kind="child_terminal",
                name=outcome.status,
                parent_id=outcome.route_role,
                child_id=outcome.child_id,
            )
        )
    events.append(
        BackendExecutionEvent(
            sequence=len(events),
            kind="terminal",
            name=aggregate_status,
        )
    )
    usage = Usage(
        input_tokens=sum(outcome.input_tokens for outcome in outcomes),
        output_tokens=sum(outcome.output_tokens for outcome in outcomes),
        cost_usd=sum(outcome.cost_usd for outcome in outcomes),
        cache_creation_tokens=sum(
            outcome.cache_creation_tokens for outcome in outcomes
        ),
        cache_read_tokens=sum(
            outcome.cache_read_tokens for outcome in outcomes
        ),
    )
    fanout_result = ReadOnlyFanoutResult(
        children=outcomes,
        usage=usage,
        trace=BackendExecutionTrace(
            backend="crucible_scheduler",
            requested_model="backend-neutral",
            events=tuple(events),
        ),
        joined_text=joined,
        status=aggregate_status,
        all_joined=True,
        cancellation_drained=not active,
        join_error=join_error,
    )
    with state_lock:
        retained_backends = tuple(active.items())
    if retained_backends:
        raise ReadOnlyCleanupError(
            partial_result=fanout_result,
            retained_backends=retained_backends,
        )
    return fanout_result


__all__ = [
    "BackendFactory",
    "ChildStatus",
    "ReadOnlyChildOutcome",
    "ReadOnlyChildRequest",
    "ReadOnlyCleanupError",
    "ReadOnlyFanoutResult",
    "ReadOnlyIsolationError",
    "ReadOnlySchedulerError",
    "ToolSetFactory",
    "run_read_only_fanout",
]
