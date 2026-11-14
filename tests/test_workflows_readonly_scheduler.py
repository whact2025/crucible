# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Synthetic contract tests for ORC-01's read-only scheduler."""

from __future__ import annotations

import threading
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    ExternalCancellation,
    Invocation,
    InvocationResult,
    PlanOutcome,
    ProjectContext,
    Spec,
    Tier,
    TierDAG,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice

from crucible.backends.config import BackendRouting, RoleConfig
from crucible.backends.features import FeatureAdmissionError
from crucible.tools._dispatch_budget import (
    BoundedReadOnlyToolSet,
    ToolDispatchBudgetExceeded,
)
from crucible.workflows._readonly_scheduler import (
    ReadOnlyChildRequest,
    ReadOnlyCleanupError,
    ReadOnlySchedulerError,
    run_read_only_fanout,
)
from crucible.workflows._run_policy import OrchestrationPolicy
from crucible.workflows import _backends as backend_pool_module
from crucible.workflows import architect as architect_workflow
from crucible.workflows._backends import _BackendPool
from crucible.workflows.architect import (
    ArchitectReadOnlyTask,
    _JoinedReadOnlyAnalysisBackend,
)


class _ReadToolSet:
    def __init__(self, context, *, writes: bool = False) -> None:
        self.context = context
        self.write_names = {"write_source"} if writes else set()
        self.read_only_names = {"read_source"}

    def to_anthropic_tools(self) -> list[dict]:
        descriptors = [
            {
                "name": "read_source",
                "description": "read",
                "input_schema": {"type": "object"},
            }
        ]
        if self.write_names:
            descriptors.append(
                {
                    "name": "write_source",
                    "description": "write",
                    "input_schema": {"type": "object"},
                }
            )
        return descriptors

    def to_openai_tools(self) -> list[dict]:
        return []

    def dispatch(self, name: str, arguments: dict) -> str:
        return f"{name}:{arguments}"

    def mcp_config(self):
        return None


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("child_capability", "workspace_write"),
        ("project_instruction_policy", "inherit"),
        ("cancellation_policy", "abandon"),
        ("approval_policy", "allow"),
        ("delivery_mode", "child_commit"),
    ],
)
def test_orchestration_policy_closed_literals_fail_closed(
    field_name,
    value,
) -> None:
    kwargs = {
        "mode": "read_only",
        "child_roles": ("repository_mapper",),
        field_name: value,
    }
    with pytest.raises(ValueError, match=field_name):
        OrchestrationPolicy(**kwargs)


@dataclass
class _ConcurrencyTracker:
    active: int = 0
    max_active: int = 0
    started: int = 0
    closed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    two_started: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        with self.lock:
            self.active += 1
            self.started += 1
            self.max_active = max(self.max_active, self.active)
            if self.started >= 2:
                self.two_started.set()

    def finish(self) -> None:
        with self.lock:
            self.active -= 1

    def note_close(self) -> None:
        with self.lock:
            self.closed += 1


def _complete_result(
    child_id: str,
    *,
    cost: float = 0.25,
    cancellation: bool = False,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    pricing_evidence: UsagePricingEvidence | None = None,
) -> InvocationResult:
    events = (
        BackendExecutionEvent(sequence=0, kind="request", name="invoke"),
        BackendExecutionEvent(sequence=1, kind="usage", name="usage"),
        BackendExecutionEvent(
            sequence=2,
            kind="terminal",
            name="cancelled" if cancellation else "complete",
        ),
    )
    return InvocationResult(
        final_message=f"analysis:{child_id}",
        transcript=(),
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cost_usd=cost,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        ),
        stop_reason="cancelled" if cancellation else "end_turn",
        error=(
            ExternalCancellation(
                cause="scheduler",
                message="cancelled by parent",
            )
            if cancellation
            else None
        ),
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="synthetic",
            requested_model="synthetic-model",
            events=events,
        ),
        pricing_evidence=pricing_evidence,
    )


class _Backend:
    def __init__(
        self,
        child_id: str,
        tracker: _ConcurrencyTracker,
        *,
        delay: float = 0.0,
        behavior: str = "success",
        name: str = "synthetic",
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        pricing_evidence: UsagePricingEvidence | None = None,
    ) -> None:
        self.child_id = child_id
        self.tracker = tracker
        self.delay = delay
        self.behavior = behavior
        self.name = name
        self.model = f"{name}-model"
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens
        self.pricing_evidence = pricing_evidence
        self.closed = threading.Event()
        self._close_lock = threading.Lock()
        self._close_counted = False
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        self.tracker.start()
        try:
            if self.behavior == "block":
                assert self.closed.wait(5), "scheduler did not cancel backend"
                return _complete_result(self.child_id, cancellation=True)
            if self.behavior == "error_after_two":
                assert self.tracker.two_started.wait(5)
                raise RuntimeError(f"boom:{self.child_id}")
            if self.delay:
                time.sleep(self.delay)
            return _complete_result(
                self.child_id,
                cache_creation_tokens=self.cache_creation_tokens,
                cache_read_tokens=self.cache_read_tokens,
                pricing_evidence=self.pricing_evidence,
            )
        finally:
            self.tracker.finish()

    def close(self) -> None:
        with self._close_lock:
            if not self._close_counted:
                self._close_counted = True
                self.tracker.note_close()
        self.closed.set()


def _policy(
    child_roles: tuple[str, ...],
    *,
    max_concurrency: int = 2,
    max_children: int | None = None,
    fail_fast: bool = True,
) -> OrchestrationPolicy:
    return OrchestrationPolicy(
        mode="read_only",
        child_roles=child_roles,
        max_concurrency=max_concurrency,
        max_children=max_children or len(child_roles),
        max_child_turns=3,
        max_total_cost_share=0.5,
        max_child_cost_share=0.25,
        fail_fast=fail_fast,
    )


def _requests(tmp_path: Path, count: int) -> tuple[ReadOnlyChildRequest, ...]:
    requests = []
    for index in range(count):
        cwd = tmp_path / f"workspace-{index}"
        cwd.mkdir()
        child_id = f"child-{index}"
        requests.append(
            ReadOnlyChildRequest(
                child_id=child_id,
                role=child_id,
                route_role="architect_planner",
                cwd=cwd,
                invocation=Invocation(
                    system="read only",
                    user=f"inspect partition {index}",
                    tools=None,
                    max_turns=9,
                    budget_usd=None,
                ),
            )
        )
    return tuple(requests)


@pytest.mark.parametrize(
    "backend_name",
    ["codex", "claude", "kimi", "openai_compat"],
)
def test_scheduler_is_backend_neutral_bounded_and_fully_attributed(
    tmp_path: Path,
    backend_name: str,
) -> None:
    requests = _requests(tmp_path, 4)
    tracker = _ConcurrencyTracker()
    backends: list[_Backend] = []
    toolsets: list[_ReadToolSet] = []
    contexts = []

    def build_toolset(request, context):
        contexts.append(context)
        toolset = _ReadToolSet(context)
        toolsets.append(toolset)
        return toolset

    def build_backend(request, toolset, debug_dir):
        assert toolset.context.workspace_root == request.cwd.resolve()
        assert debug_dir.parent == (tmp_path / "debug").resolve()
        backend = _Backend(
            request.child_id,
            tracker,
            delay=0.04 if request.child_id.endswith(("0", "2")) else 0.01,
            name=backend_name,
        )
        backends.append(backend)
        return backend

    result = run_read_only_fanout(
        requests,
        policy=_policy(
            tuple(request.role for request in requests),
            max_concurrency=2,
        ),
        backend_factory=build_backend,
        toolset_factory=build_toolset,
        debug_root=tmp_path / "debug",
        parent_budget_usd=8.0,
    )

    assert result.successful
    assert tracker.max_active == 2
    assert tracker.active == 0
    assert tracker.closed == 4
    assert [child.child_id for child in result.children] == [
        request.child_id for request in requests
    ]
    assert len({id(backend) for backend in backends}) == 4
    assert len({id(toolset) for toolset in toolsets}) == 4
    assert len({id(context) for context in contexts}) == 4
    assert all(context.context_budget == 262_144 for context in contexts)
    assert len({child.cwd for child in result.children}) == 4
    assert len({child.debug_dir for child in result.children}) == 4
    assert all(not toolset.write_names for toolset in toolsets)
    assert result.usage == Usage(
        input_tokens=40,
        output_tokens=20,
        cost_usd=1.0,
    )
    legacy_summary = result.to_summary()
    assert legacy_summary["schema_version"] == (
        "crucible.read-only-fanout-result.v1"
    )
    assert "cache_creation_tokens" not in legacy_summary
    assert "pricing_evidence" not in legacy_summary["children"][0]
    assert [event.sequence for event in result.trace.events] == list(range(9))
    assert [event.kind for event in result.trace.events] == [
        "child_start",
        "child_terminal",
        "child_start",
        "child_terminal",
        "child_start",
        "child_terminal",
        "child_start",
        "child_terminal",
        "terminal",
    ]
    assert [event.child_id for event in result.trace.events[:-1:2]] == [
        request.child_id for request in requests
    ]
    assert all(backend.invocations[0].max_turns == 3 for backend in backends)
    # 50% total budget / four children = $1 each, tighter than the
    # individual 25% ($2) slot.
    assert all(backend.invocations[0].budget_usd == 1.0 for backend in backends)


def test_scheduler_preserves_cache_usage_and_per_child_pricing_evidence(
    tmp_path: Path,
) -> None:
    requests = _requests(tmp_path, 2)
    tracker = _ConcurrencyTracker()
    evidence = UsagePricingEvidence(
        segments=(
            UsagePricingSegment(
                input_tokens=10,
                output_tokens=5,
                cache_creation_tokens=3,
                cache_read_tokens=4,
                cost_usd=0.25,
            ),
        ),
        complete=True,
    )

    def build_backend(request, _toolset, _debug_dir):
        priced = request.child_id == "child-0"
        return _Backend(
            request.child_id,
            tracker,
            cache_creation_tokens=3 if priced else 0,
            cache_read_tokens=4 if priced else 0,
            pricing_evidence=evidence if priced else None,
        )

    result = run_read_only_fanout(
        requests,
        policy=_policy(tuple(request.role for request in requests)),
        backend_factory=build_backend,
        toolset_factory=lambda _request, context: _ReadToolSet(context),
        debug_root=tmp_path / "debug",
        parent_budget_usd=4.0,
    )

    assert result.successful
    assert result.usage == Usage(
        input_tokens=20,
        output_tokens=10,
        cost_usd=0.5,
        cache_creation_tokens=3,
        cache_read_tokens=4,
    )
    assert result.children[0].pricing_evidence is evidence
    assert result.children[1].pricing_evidence is None
    summary = result.to_summary()
    assert summary["schema_version"] == "crucible.read-only-fanout-result.v2"
    assert summary["cache_creation_tokens"] == 3
    assert summary["cache_read_tokens"] == 4
    assert summary["children"][0]["pricing_evidence"] == {
        "segments": [
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_tokens": 3,
                "cache_read_tokens": 4,
                "cost_usd": 0.25,
            },
        ],
        "complete": True,
    }
    # Missing pricing evidence is observable but does not become a new
    # scheduler success gate.
    assert summary["children"][1]["pricing_evidence"] is None
    assert summary["children"][1]["cache_creation_tokens"] == 0
    assert summary["children"][1]["cache_read_tokens"] == 0


def test_scheduler_rejects_a_shared_backend_instance(tmp_path: Path) -> None:
    requests = _requests(tmp_path, 2)
    tracker = _ConcurrencyTracker()
    singleton = _Backend("singleton", tracker, delay=0.02)

    result = run_read_only_fanout(
        requests,
        policy=_policy(tuple(request.role for request in requests)),
        backend_factory=lambda request, toolset, debug_dir: singleton,
        toolset_factory=lambda request, context: _ReadToolSet(context),
        debug_root=tmp_path / "debug",
        parent_budget_usd=8.0,
    )

    assert not result.successful
    assert any(
        child.error_kind == "ReadOnlyIsolationError" for child in result.children
    )
    assert tracker.closed == 1
    assert tracker.active == 0


def test_scheduler_rejects_write_tools_before_backend_construction(
    tmp_path: Path,
) -> None:
    requests = _requests(tmp_path, 1)
    backend_calls = 0

    def build_backend(request, toolset, debug_dir):
        nonlocal backend_calls
        backend_calls += 1
        raise AssertionError("must not construct a backend")

    result = run_read_only_fanout(
        requests,
        policy=_policy((requests[0].role,), max_concurrency=1),
        backend_factory=build_backend,
        toolset_factory=lambda request, context: _ReadToolSet(
            context,
            writes=True,
        ),
        debug_root=tmp_path / "debug",
        parent_budget_usd=8.0,
    )

    assert not result.successful
    assert result.children[0].status == "failed"
    assert "mutating tools" in (result.children[0].error_message or "")
    assert backend_calls == 0
    assert result.usage.cost_usd == 0.0


def test_parent_cancellation_closes_drains_and_joins_every_child(
    tmp_path: Path,
) -> None:
    requests = _requests(tmp_path, 3)
    tracker = _ConcurrencyTracker()
    cancel = threading.Event()
    backends: list[_Backend] = []

    def build_backend(request, toolset, debug_dir):
        backend = _Backend(request.child_id, tracker, behavior="block")
        backends.append(backend)
        return backend

    def cancel_after_start() -> None:
        assert tracker.two_started.wait(5)
        cancel.set()

    trigger = threading.Thread(target=cancel_after_start)
    trigger.start()
    result = run_read_only_fanout(
        requests,
        policy=_policy(tuple(request.role for request in requests)),
        backend_factory=build_backend,
        toolset_factory=lambda request, context: _ReadToolSet(context),
        debug_root=tmp_path / "debug",
        parent_budget_usd=8.0,
        cancel_event=cancel,
    )
    trigger.join(timeout=5)

    assert not trigger.is_alive()
    assert result.status == "cancelled"
    assert result.all_joined
    assert result.cancellation_drained
    assert len(result.children) == 3
    assert all(child.status == "cancelled" for child in result.children)
    assert tracker.active == 0
    assert tracker.closed == len(backends)
    assert all(backend.closed.is_set() for backend in backends)


def test_child_exception_cleans_up_and_drains_running_siblings(
    tmp_path: Path,
) -> None:
    requests = _requests(tmp_path, 2)
    tracker = _ConcurrencyTracker()
    backends: list[_Backend] = []

    def build_backend(request, toolset, debug_dir):
        behavior = "error_after_two" if request.child_id == "child-0" else "block"
        backend = _Backend(request.child_id, tracker, behavior=behavior)
        backends.append(backend)
        return backend

    result = run_read_only_fanout(
        requests,
        policy=_policy(tuple(request.role for request in requests)),
        backend_factory=build_backend,
        toolset_factory=lambda request, context: _ReadToolSet(context),
        debug_root=tmp_path / "debug",
        parent_budget_usd=8.0,
    )

    assert result.status in {"failed", "cancelled"}
    assert result.all_joined
    assert result.cancellation_drained
    assert result.children[0].error_kind == "RuntimeError"
    assert result.children[1].status == "cancelled"
    assert tracker.active == 0
    assert tracker.closed == 2
    assert all(backend.closed.is_set() for backend in backends)


def test_transient_close_failure_is_retried_but_remains_fail_closed(
    tmp_path: Path,
) -> None:
    tracker = _ConcurrencyTracker()

    class _FailOnceBackend(_Backend):
        def __init__(self) -> None:
            super().__init__("child-0", tracker)
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("transient close failure")
            super().close()

    backend = _FailOnceBackend()
    result = run_read_only_fanout(
        _requests(tmp_path, 1),
        policy=_policy(("child-0",), max_concurrency=1),
        backend_factory=lambda _request, _toolset, _debug: backend,
        toolset_factory=lambda _request, context: _ReadToolSet(context),
        debug_root=tmp_path / "debug",
        parent_budget_usd=8.0,
    )

    assert backend.close_calls == 2
    assert backend.closed.is_set()
    assert result.cancellation_drained
    assert result.status == "failed"
    assert result.children[0].status == "cleanup_failed"
    assert not result.children[0].cleanup_complete
    assert result.children[0].error_kind == "RuntimeError"


def test_persistent_close_failure_returns_caller_owned_retry_handle(
    tmp_path: Path,
) -> None:
    tracker = _ConcurrencyTracker()

    class _NeedsCallerRetryBackend(_Backend):
        close_calls = 0

        def __init__(self) -> None:
            super().__init__("child-0", tracker)

        def close(self) -> None:
            type(self).close_calls += 1
            if type(self).close_calls <= 4:
                raise RuntimeError("persistent close failure")
            super().close()

    with pytest.raises(ReadOnlyCleanupError) as caught:
        run_read_only_fanout(
            _requests(tmp_path, 1),
            policy=_policy(("child-0",), max_concurrency=1),
            backend_factory=(
                lambda _request, _toolset, _debug: (
                    _NeedsCallerRetryBackend()
                )
            ),
            toolset_factory=lambda _request, context: _ReadToolSet(context),
            debug_root=tmp_path / "debug",
            parent_budget_usd=8.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert _NeedsCallerRetryBackend.close_calls == 4
    assert cleanup_error.retained_child_ids == ("child-0",)
    assert not partial.cancellation_drained
    assert partial.status == "failed"
    assert partial.usage == Usage(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.25,
    )
    assert partial.children[0].status == "cleanup_failed"
    assert partial.children[0].error_kind == "BackendCleanupError"

    settled = cleanup_error.retry_cleanup()
    assert _NeedsCallerRetryBackend.close_calls == 5
    assert cleanup_error.retained_child_ids == ()
    assert settled.cancellation_drained
    assert not settled.successful
    assert settled.usage == partial.usage
    with pytest.raises(
        ValueError,
        match="live backend cleanup authority",
    ):
        ReadOnlyCleanupError(
            partial_result=partial,
            retained_backends=cleanup_error._retained_backends,
        )


def test_fatal_child_unwind_retains_backend_for_caller_retry(
    tmp_path: Path,
) -> None:
    tracker = _ConcurrencyTracker()

    class _FatalChildEscape(BaseException):
        pass

    class _NeedsFatalRetryBackend(_Backend):
        close_calls = 0

        def __init__(self) -> None:
            super().__init__("child-0", tracker)

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            raise _FatalChildEscape("fatal child escape")

        def close(self) -> None:
            type(self).close_calls += 1
            if type(self).close_calls <= 4:
                raise RuntimeError("persistent fatal-unwind close failure")
            super().close()

    with pytest.raises(ReadOnlyCleanupError) as caught:
        run_read_only_fanout(
            _requests(tmp_path, 1),
            policy=_policy(("child-0",), max_concurrency=1),
            backend_factory=(
                lambda _request, _toolset, _debug: (
                    _NeedsFatalRetryBackend()
                )
            ),
            toolset_factory=lambda _request, context: _ReadToolSet(context),
            debug_root=tmp_path / "debug",
            parent_budget_usd=8.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert isinstance(cleanup_error.primary_error, _FatalChildEscape)
    assert cleanup_error.retained_child_ids == ("child-0",)
    assert _NeedsFatalRetryBackend.close_calls == 4
    assert partial.usage == Usage()
    assert not partial.cancellation_drained
    assert partial.children[0].status == "cleanup_failed"
    assert not partial.children[0].token_usage_complete
    assert not partial.children[0].cost_complete
    assert not partial.children[0].terminal_complete
    assert not partial.children[0].trace_complete

    settled = cleanup_error.retry_cleanup()
    assert _NeedsFatalRetryBackend.close_calls == 5
    assert cleanup_error.retained_child_ids == ()
    assert settled.cancellation_drained
    assert settled.usage == partial.usage
    assert tracker.closed == 1


def test_admission_failure_does_not_create_an_executor(tmp_path: Path) -> None:
    request = _requests(tmp_path, 1)[0]
    with pytest.raises(ReadOnlySchedulerError, match="not eligible"):
        run_read_only_fanout(
            (request,),
            policy=_policy(("different-role",), max_concurrency=1),
            backend_factory=lambda request, toolset, debug_dir: object(),
            toolset_factory=lambda request, context: _ReadToolSet(context),
            debug_root=tmp_path / "debug",
            parent_budget_usd=8.0,
        )


def test_backend_pool_exclusive_path_never_reuses_role_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = []

    class Backend:
        name = "synthetic"

        def __init__(self) -> None:
            self.attachments = []
            self.closed = False

        def attach_toolset(self, toolset, *, context_budget=None) -> None:
            self.attachments.append((toolset, context_budget))

        def close(self) -> None:
            self.closed = True

    def resolve_for_role(role, **kwargs):
        backend = Backend()
        constructed.append((role, kwargs, backend))
        return backend

    monkeypatch.setattr(
        backend_pool_module,
        "resolve_for_role",
        resolve_for_role,
    )
    pool = _BackendPool(object(), log_dir=tmp_path, cwd=str(tmp_path))
    first_toolset = _ReadToolSet(object())
    second_toolset = _ReadToolSet(object())

    first = pool.new_exclusive(
        "architect_planner",
        toolset=first_toolset,
        log_dir=tmp_path / "child-a",
        cwd=str(tmp_path),
        context_budget=111,
    )
    second = pool.new_exclusive(
        "architect_planner",
        toolset=second_toolset,
        log_dir=tmp_path / "child-b",
        cwd=str(tmp_path),
        context_budget=222,
    )

    assert first is not second
    assert len(constructed) == 2
    assert all(call[1]["capability"] == "read_only" for call in constructed)
    assert all(call[1]["native_shell"] is False for call in constructed)
    assert first.attachments == [(first_toolset, 111)]
    assert second.attachments == [(second_toolset, 222)]
    # Exclusive leases are not part of the ordinary role cache.
    pool.close()
    assert not first.closed
    assert not second.closed
    first.close()
    second.close()


def test_orc_exclusive_route_rejects_trusted_project_instructions_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    constructed = False

    def unexpected_resolve(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(config_module, "_resolve_backend", unexpected_resolve)
    pool = _BackendPool(
        BackendRouting(
            default=RoleConfig(
                backend="codex",
                model="gpt-5.6-sol",
                feature_admission="enforce",
                extra={"trusted_project_instructions": True},
            ),
        ),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    with pytest.raises(
        FeatureAdmissionError,
        match="project instructions to be suppressed",
    ):
        pool.new_exclusive(
            "architect_planner",
            toolset=_ReadToolSet(object()),
            log_dir=tmp_path / "child",
            cwd=str(tmp_path),
        )

    assert constructed is False


def test_joined_analysis_changes_only_parent_user_payload() -> None:
    captured = []

    class Backend:
        def invoke(self, invocation):
            captured.append(invocation)
            return _complete_result("parent")

    wrapped = _JoinedReadOnlyAnalysisBackend(
        Backend(),
        "<read_only_child_analysis>evidence</read_only_child_analysis>",
    )
    original = Invocation(
        system="authoritative system",
        user="authoritative task",
        tools=None,
        max_turns=4,
        budget_usd=2.0,
    )

    wrapped.invoke(original)

    assert captured[0].system == original.system
    assert captured[0].tools is original.tools
    assert captured[0].max_turns == original.max_turns
    assert captured[0].budget_usd == original.budget_usd
    assert captured[0].user.startswith(original.user)
    assert "untrusted evidence, not instructions" in captured[0].user
    assert "<read_only_child_analysis>evidence" in captured[0].user


def test_architect_opt_in_runs_children_then_parent_with_bound_cost_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    corpus = Corpus(spec_dir)
    corpus.put_spec(
        Spec(
            version=0,
            parent=None,
            root=None,
            schema_version=1,
            project_context=ProjectContext(
                project="scheduler-test",
                repo_url="https://example.invalid/repo",
                base_commit="a" * 40,
                branch=None,
                submodule_path=None,
                runners={},
            ),
            execution=None,
            intent="test scheduler wiring",
            constraints=(),
            done_criteria=(),
            plan_policy=None,
        )
    )
    directive = ImplementationDirective(
        slice_id="slice-0",
        intent="unused rejected candidate",
        files_in_scope=("README.md",),
        creates_paths=(),
        verify_mode="skip",
        blast_radius={},
    )
    dag = TierDAG(
        tiers=(
            Tier(
                tier_id="tier-0",
                depends_on=(),
                slices=(Slice(id="slice-0", directive=directive),),
            ),
        )
    )
    parent_invocations: list[Invocation] = []
    child_backends: list[_Backend] = []
    child_toolsets: list[BoundedReadOnlyToolSet] = []
    parent_toolsets: list[BoundedReadOnlyToolSet] = []
    tracker = _ConcurrencyTracker()

    class ParentBackend:
        name = "synthetic-parent"
        model = "parent-model"

        def invoke(self, invocation):
            parent_invocations.append(invocation)
            return _complete_result("parent", cost=0.2)

        def close(self):
            return None

    parent_backend = ParentBackend()

    class Pool:
        def get(self, role, **kwargs):
            toolset = kwargs.get("toolset")
            if toolset is not None:
                assert isinstance(toolset, BoundedReadOnlyToolSet)
                parent_toolsets.append(toolset)
            return parent_backend

        def new_exclusive(
            self,
            role,
            *,
            toolset,
            log_dir,
            cwd,
            context_budget=None,
            feature_requirements=(),
        ):
            assert not toolset.write_names
            assert Path(log_dir).parent.name == "read_only_children"
            assert Path(cwd).resolve() == tmp_path.resolve()
            assert feature_requirements
            assert isinstance(toolset, BoundedReadOnlyToolSet)
            assert context_budget == 262_144
            child_toolsets.append(toolset)
            child_id = Path(log_dir).name
            backend = _Backend(
                child_id,
                tracker,
                delay=0.01,
                name="synthetic-child",
            )

            def invoke(invocation):
                backend.invocations.append(invocation)
                tracker.start()
                try:
                    toolset.dispatch(
                        "read_source",
                        {"path": f"{child_id}-missing.txt"},
                    )
                    time.sleep(0.01)
                    return _complete_result(child_id, cost=0.1)
                finally:
                    tracker.finish()

            backend.invoke = invoke
            child_backends.append(backend)
            return backend

    @contextmanager
    def pool(*args, **kwargs):
        yield Pool()

    def run_plan(spec, **kwargs):
        kwargs["planner_backend"].invoke(
            Invocation(
                system="parent system",
                user="parent task",
                tools=None,
                max_turns=3,
                budget_usd=9.0,
            )
        )
        return PlanOutcome(
            spec=spec.with_appended_refs(),
            candidate_dag=dag,
            candidate_ref=None,
            decision="rejected",
            status="rejected",
            rounds_used=1,
            final_critic_decision="reject",
            final_judge_decision="reject",
            reason="synthetic terminal after scheduler wiring",
            cost_usd=0.2,
        )

    monkeypatch.setattr(architect_workflow, "backend_pool", pool)
    monkeypatch.setattr(
        architect_workflow,
        "load_rubric",
        lambda name: name,
    )
    monkeypatch.setattr(architect_workflow, "run_plan", run_plan)
    monkeypatch.setattr(
        architect_workflow,
        "plan_contract_fingerprint",
        lambda *args, **kwargs: "sha256:" + "a" * 64,
    )

    policy = OrchestrationPolicy(
        mode="read_only",
        child_roles=("repository_mapper", "risk_reviewer"),
        max_concurrency=2,
        max_children=2,
        max_total_cost_share=0.5,
        max_child_cost_share=0.25,
    )
    rc = architect_workflow.run_architect(
        planner_count=1,
        spec=spec_dir,
        cwd=str(tmp_path),
        budget_usd=10.0,
        enable_source_tools=True,
        planner_tool_call_limit=2,
        planner_tool_result_bytes=4096,
        replan=True,
        skip_validate=True,
        orchestration_policy=policy,
        read_only_tasks=(
            ArchitectReadOnlyTask(
                child_id="map",
                role="repository_mapper",
                instruction="Map the public API surface.",
            ),
            ArchitectReadOnlyTask(
                child_id="risk",
                role="risk_reviewer",
                instruction="Identify high-risk implementation boundaries.",
            ),
        ),
    )

    assert rc == 2
    assert len(child_backends) == 2
    assert len({id(backend) for backend in child_backends}) == 2
    assert tracker.max_active == 2
    assert tracker.active == 0
    assert tracker.closed == 2
    assert [backend.invocations[0].budget_usd for backend in child_backends] == [
        2.5,
        2.5,
    ]
    assert len(parent_invocations) == 1
    assert len(parent_toolsets) == 1
    assert len(child_toolsets) == 2
    assert parent_toolsets[0].snapshot().calls_used == 2
    assert all(
        toolset.snapshot() == parent_toolsets[0].snapshot()
        for toolset in child_toolsets
    )
    with pytest.raises(
        ToolDispatchBudgetExceeded,
        match="underlying dispatch was not started",
    ):
        parent_toolsets[0].dispatch(
            "read_source",
            {"path": "parent-must-not-run.txt"},
        )
    assert "analysis:map" in parent_invocations[0].user
    assert "analysis:risk" in parent_invocations[0].user
    assert parent_invocations[0].user.index("analysis:map") < (
        parent_invocations[0].user.index("analysis:risk")
    )

    run_dirs = list((spec_dir / "runs").glob("*_architect_*"))
    assert len(run_dirs) == 1
    summary = json.loads((run_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dirs[0] / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert summary["orchestration"]["status"] == "succeeded"
    assert summary["orchestration"]["child_count"] == 2
    assert summary["orchestration"]["cost_usd"] == pytest.approx(0.2)
    assert summary["authoritative_spent_usd"] == pytest.approx(0.4)
    planning_by_role = summary["costs"]["planning_usage"]["by_role"]
    assert set(planning_by_role) == {"architect_planner"}
    assert planning_by_role["architect_planner"]["calls"] == 3
    assert planning_by_role["architect_planner"]["cost_usd"] == pytest.approx(0.4)
    assert [
        (child["role"], child["route_role"])
        for child in summary["orchestration"]["children"]
    ] == [
        ("repository_mapper", "architect_planner"),
        ("risk_reviewer", "architect_planner"),
    ]
    assert manifest["provenance"]["inputs"]["execution_policy"] == (policy.digest)
