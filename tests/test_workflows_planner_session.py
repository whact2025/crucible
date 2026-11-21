# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendFeatureSet,
    ConversationTransportOutcome,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    StaleConversation,
    Usage,
)

from crucible.backends.features import feature_profile_digest
from crucible.tools._context import ToolContext
from crucible.tools.source import build_source_tools
from crucible.workflows import plan as plan_workflow
from crucible.workflows import _control
from crucible.workflows import _conversation
from crucible.workflows import _planner_session as planner_session


_PROFILE = BackendFeatureSet(session_continuity="workflow_bound")
_COMPLETE = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _invocation(user: str = "plan this") -> Invocation:
    return Invocation(
        system="stable planner protocol",
        user=user,
        tools=None,
        max_turns=5,
        budget_usd=1.0,
        response_schema={
            "type": "object",
            "properties": {"tiers": {"type": "array"}},
        },
    )


class _Backend:
    name = "codex"
    model = "gpt-5.6-sol"
    transport = "legacy_app_server"
    capability = "read_only"

    def __init__(
        self,
        *outcomes: str,
        profile: BackendFeatureSet = _PROFILE,
        trace_tool_authority_digest: str | None = None,
        trace_overrides: dict[str, str | None] | None = None,
        trace_complete: bool = True,
        omit_trace: bool = False,
    ) -> None:
        self.features = profile
        self.outcomes = list(outcomes)
        self.invocations: list[Invocation] = []
        self.trace_tool_authority_digest = trace_tool_authority_digest
        self.trace_overrides = trace_overrides or {}
        self.trace_complete = trace_complete
        self.omit_trace = omit_trace

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        outcome_kind = self.outcomes.pop(0) if self.outcomes else "active"
        if outcome_kind == "raise":
            raise OSError("connection outcome unknown")
        request = invocation.conversation
        assert request is not None
        trace = BackendExecutionTrace(
            backend=str(self.trace_overrides.get("backend", self.name)),
            requested_model=str(
                self.trace_overrides.get("model", self.model)
            ),
            feature_profile_digest=self.trace_overrides.get(
                "profile",
                feature_profile_digest(self.features),
            ),
            tool_authority_digest=self.trace_overrides.get(
                "tool",
                self.trace_tool_authority_digest,
            ),
            events=(
                BackendExecutionEvent(sequence=0, kind="request"),
                BackendExecutionEvent(sequence=1, kind="usage"),
                BackendExecutionEvent(sequence=2, kind="terminal"),
            ),
        )
        if outcome_kind in {"stale", "stale_incomplete"}:
            accounting_complete = outcome_kind == "stale"
            return InvocationResult(
                final_message="",
                transcript=(),
                usage=(
                    Usage()
                    if accounting_complete
                    else Usage(
                        input_tokens=23,
                        output_tokens=0,
                        cost_usd=0.07,
                    )
                ),
                stop_reason="stale_conversation",
                error=StaleConversation(message="provider rejected thread"),
                completeness=ExecutionCompleteness(
                    token_usage_complete=accounting_complete,
                    cost_complete=accounting_complete,
                    terminal_complete=True,
                    trace_complete=self.trace_complete,
                ),
                execution_trace=None if self.omit_trace else trace,
                conversation=ConversationTransportOutcome(
                    state="stale",
                    opaque_id=None,
                    identity_fingerprint=request.identity_fingerprint,
                    operation_digest=request.operation_digest,
                ),
            )
        if outcome_kind.startswith("turn_limit:"):
            # A complete turn-limit terminal; the provider's own turn count
            # follows the colon.
            return InvocationResult(
                final_message="",
                transcript=(),
                usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.2),
                stop_reason="max_turns",
                error=MaxTurnsExceeded(
                    turns_used=int(outcome_kind.split(":", 1)[1]),
                    turns_limit=invocation.max_turns,
                    message=f"hit max_turns={invocation.max_turns}",
                ),
                completeness=_COMPLETE,
                execution_trace=trace,
                conversation=ConversationTransportOutcome(
                    state="active",
                    opaque_id=f"thread-{len(self.invocations)}",
                    identity_fingerprint=request.identity_fingerprint,
                    operation_digest=request.operation_digest,
                ),
            )
        assistant_messages = (
            int(outcome_kind.split(":", 1)[1])
            if outcome_kind.startswith("messages:")
            else 1
        )
        completeness = (
            ExecutionCompleteness()
            if outcome_kind == "incomplete"
            else _COMPLETE
        )
        return InvocationResult(
            final_message='{"tiers": []}',
            transcript=tuple(
                Message(role="assistant", content='{"tiers": []}')
                for _ in range(assistant_messages)
            ),
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.2),
            stop_reason="end_turn",
            completeness=completeness,
            execution_trace=trace,
            conversation=ConversationTransportOutcome(
                state="active",
                opaque_id=f"thread-{len(self.invocations)}",
                identity_fingerprint=request.identity_fingerprint,
                operation_digest=request.operation_digest,
            ),
        )


class _PlainBackend:
    name = "plain"
    model = "plain-model"

    def __init__(self) -> None:
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        return InvocationResult(
            final_message="ok",
            transcript=(Message(role="assistant", content="ok"),),
            usage=Usage(),
            stop_reason="end_turn",
        )


def _config(
    tmp_path: Path,
    *,
    limits: _conversation.ConversationLimits = (
        _conversation.ConversationLimits(max_turns=5, max_cost_usd=2.0)
    ),
    route: str = "route-a",
    tool_digest: str = _digest("2"),
    trace_tool_digest: str | None = None,
    workspace: str = "workspace-a",
    recovery: planner_session.PlannerRecoveryInstruction | None = None,
) -> planner_session.PlannerConversationConfig:
    return planner_session.PlannerConversationConfig(
        state_root=tmp_path / "planner-sessions",
        run_identity_digest=_digest("1"),
        route_identity={"route": route},
        tool_authority_digest=tool_digest,
        trace_tool_authority_digest=trace_tool_digest,
        runtime_identity={"adapter": "test-adapter-v1"},
        trusted_context_identity={
            "spec": _digest("3"),
            "workspace": workspace,
        },
        limits=limits,
        recovery=recovery,
        now=lambda: "2026-07-28T12:00:00Z",
    )


def test_stale_reuse_gets_one_fresh_attempt_without_resetting_ledger(
    tmp_path: Path,
) -> None:
    backend = _Backend("active", "stale", "active")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )

    first = wrapper.invoke(_invocation("initial"))
    second = wrapper.invoke(_invocation("corrected"))

    assert first.conversation is not None
    assert second.conversation is not None
    assert [
        invocation.conversation.action
        for invocation in backend.invocations
        if invocation.conversation is not None
    ] == ["fresh", "reuse", "fresh"]
    controller = wrapper._controller(_invocation("corrected"))  # noqa: SLF001
    handle = controller.load()
    assert handle is not None
    assert handle.turns_used == 2
    assert handle.usage.cost_usd == pytest.approx(0.4)
    assert len(handle.operation_digests) == 2
    assert len(handle.recovery_resolutions) == 1
    resolution = handle.recovery_resolutions[0]
    assert resolution.disposition == "not_started"
    assert resolution.actor == "provider:codex:typed-stale"


def test_incomplete_stale_reuse_returns_debt_without_fresh_retry(
    tmp_path: Path,
) -> None:
    invocation = _invocation()
    backend = _Backend("stale_incomplete", "active")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    controller = wrapper._controller(invocation)  # noqa: SLF001
    controller.resume(scope="workflow_correction")
    controller.begin_operation(
        operation_digest=_digest("e"),
        started_at="2026-07-28T11:00:00Z",
    )
    controller.commit_operation(
        operation_digest=_digest("e"),
        opaque_id="existing-thread",
    )

    result = wrapper.invoke(invocation)

    assert isinstance(result.error, StaleConversation)
    assert result.usage == Usage(
        input_tokens=23,
        output_tokens=0,
        cost_usd=0.07,
    )
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert len(backend.invocations) == 1
    assert backend.invocations[0].conversation is not None
    assert backend.invocations[0].conversation.action == "reuse"

    handle = controller.load()
    assert handle is not None
    assert handle.state == "stale"
    assert handle.pending_operation is not None
    assert handle.pending_operation.state == "indeterminate"
    assert handle.recovery_resolutions == ()


def test_uncertain_crash_fails_closed_until_exact_audited_resolution(
    tmp_path: Path,
) -> None:
    invocation = _invocation()
    crashed = planner_session.PlannerBoundaryBackend(
        _Backend("raise"),
        conversation=_config(tmp_path),
    )
    with pytest.raises(planner_session.PlannerSessionIndeterminate):
        crashed.invoke(invocation)

    controller = crashed._controller(invocation)  # noqa: SLF001
    handle = controller.load()
    assert handle is not None
    assert handle.pending_operation is not None
    assert handle.pending_operation.state == "indeterminate"
    pending_digest = handle.pending_operation.operation_digest

    restarted = planner_session.PlannerBoundaryBackend(
        _Backend("active"),
        conversation=_config(tmp_path),
    )
    with pytest.raises(_conversation.ConversationOperationRecoveryRequired):
        restarted.invoke(invocation)

    recovered = planner_session.PlannerBoundaryBackend(
        _Backend("active"),
        conversation=_config(
            tmp_path,
            recovery=planner_session.PlannerRecoveryInstruction(
                operation_digest=pending_digest,
                evidence_digest=_digest("9"),
                actor="operator:alice",
            ),
        ),
    )
    result = recovered.invoke(invocation)

    assert result.error is None
    recovered_handle = recovered._controller(invocation).load()  # noqa: SLF001
    assert recovered_handle is not None
    assert recovered_handle.pending_operation is None
    assert recovered_handle.recovery_resolutions[0].evidence_digest == (
        _digest("9")
    )
    assert recovered_handle.recovery_resolutions[0].actor == "operator:alice"


def test_committed_call_restart_fails_closed_without_replaying_transport(
    tmp_path: Path,
) -> None:
    invocation = _invocation()
    initial = planner_session.PlannerBoundaryBackend(
        _Backend("active"),
        conversation=_config(tmp_path),
    )
    initial.invoke(invocation)

    backend = _Backend("active")
    restarted = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    with pytest.raises(
        planner_session.PlannerSessionReplayBlocked,
        match="already committed",
    ):
        restarted.invoke(invocation)
    assert backend.invocations == []


def test_concurrent_same_session_claim_allows_only_one_transport(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    backend = _Backend("active")

    class _BarrierBoundary(planner_session.PlannerBoundaryBackend):
        def _next_operation(self, *args, **kwargs):
            operation = super()._next_operation(*args, **kwargs)
            barrier.wait(timeout=5)
            return operation

    wrappers = [
        _BarrierBoundary(
            backend,
            conversation=_config(tmp_path),
        )
        for _ in range(2)
    ]
    results: list[InvocationResult] = []
    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(wrapper.invoke, _invocation())
            for wrapper in wrappers
        ]
        for future in futures:
            try:
                results.append(future.result(timeout=10))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], _conversation.ConversationStateError)
    assert len(backend.invocations) == 1


def test_incomplete_result_is_marked_indeterminate(
    tmp_path: Path,
) -> None:
    backend = _Backend("incomplete")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    invocation = _invocation()

    with pytest.raises(
        planner_session.PlannerSessionIndeterminate,
        match="complete token/cost/terminal/trace",
    ):
        wrapper.invoke(invocation)

    handle = wrapper._controller(invocation).load()  # noqa: SLF001
    assert handle is not None
    assert handle.pending_operation is not None
    assert handle.pending_operation.state == "indeterminate"


@pytest.mark.parametrize(
    ("field_name", "trace_overrides"),
    [
        ("backend", {"backend": "claude"}),
        ("model", {"model": "different-model"}),
        ("profile", {"profile": _digest("8")}),
        ("tool", {"tool": _digest("9")}),
    ],
)
def test_trace_identity_mismatch_is_marked_indeterminate(
    tmp_path: Path,
    field_name: str,
    trace_overrides: dict[str, str | None],
) -> None:
    backend = _Backend("active", trace_overrides=trace_overrides)
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    invocation = _invocation()

    with pytest.raises(
        planner_session.PlannerSessionIndeterminate,
        match="trace does not match",
    ):
        wrapper.invoke(invocation)

    handle = wrapper._controller(invocation).load()  # noqa: SLF001
    assert handle is not None
    assert handle.pending_operation is not None, field_name
    assert handle.pending_operation.state == "indeterminate"


@pytest.mark.parametrize(
    ("field_name", "trace_overrides", "trace_complete", "omit_trace"),
    [
        ("missing", {}, False, True),
        ("incomplete", {}, False, False),
        ("backend", {"backend": "claude"}, True, False),
        ("model", {"model": "different-model"}, True, False),
        ("profile", {"profile": _digest("8")}, True, False),
        ("tool", {"tool": _digest("9")}, True, False),
    ],
)
def test_stale_trace_mismatch_is_indeterminate_without_fresh_retry(
    tmp_path: Path,
    field_name: str,
    trace_overrides: dict[str, str | None],
    trace_complete: bool,
    omit_trace: bool,
) -> None:
    invocation = _invocation()
    backend = _Backend(
        "stale",
        trace_overrides=trace_overrides,
        trace_complete=trace_complete,
        omit_trace=omit_trace,
    )
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    controller = wrapper._controller(invocation)  # noqa: SLF001
    controller.resume(scope="workflow_correction")
    controller.begin_operation(
        operation_digest=_digest("e"),
        started_at="2026-07-28T11:00:00Z",
    )
    controller.commit_operation(
        operation_digest=_digest("e"),
        opaque_id="existing-thread",
    )

    with pytest.raises(
        planner_session.PlannerSessionIndeterminate,
        match="stale planner result trace does not match",
    ):
        wrapper.invoke(invocation)

    handle = controller.load()
    assert handle is not None
    assert handle.pending_operation is not None, field_name
    assert handle.pending_operation.state == "indeterminate"
    assert handle.recovery_resolutions == ()
    assert len(backend.invocations) == 1


def test_source_tool_authority_matches_complete_result_trace(
    tmp_path: Path,
) -> None:
    toolset = build_source_tools(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
        )
    )
    authority = planner_session.planner_tool_authority_digest(
        toolset,
        capability="read_only",
    )
    backend = _Backend(
        "active",
        trace_tool_authority_digest=authority,
    )
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(
            tmp_path,
            tool_digest=authority,
            trace_tool_digest=authority,
        ),
    )

    result = wrapper.invoke(_invocation())

    assert result.error is None
    assert result.execution_trace is not None
    assert result.execution_trace.tool_authority_digest == authority


@pytest.mark.parametrize(
    "changed",
    ["route", "tool", "profile", "workspace"],
)
def test_restart_rejects_identity_mismatch_before_transport(
    tmp_path: Path,
    changed: str,
) -> None:
    invocation = _invocation()
    initial = planner_session.PlannerBoundaryBackend(
        _Backend("active"),
        conversation=_config(tmp_path),
    )
    initial.invoke(invocation)

    config = _config(
        tmp_path,
        route="route-b" if changed == "route" else "route-a",
        tool_digest=_digest("8") if changed == "tool" else _digest("2"),
        workspace=(
            "workspace-b" if changed == "workspace" else "workspace-a"
        ),
    )
    profile = (
        replace(_PROFILE, structured_output="native")
        if changed == "profile"
        else _PROFILE
    )
    backend = _Backend("active", profile=profile)
    restarted = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=config,
    )

    with pytest.raises(_conversation.ConversationReuseRejected):
        restarted.invoke(invocation)
    assert backend.invocations == []


def test_cumulative_remaining_turn_and_usd_are_forwarded(
    tmp_path: Path,
) -> None:
    backend = _Backend("active", "active")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(
            tmp_path,
            limits=_conversation.ConversationLimits(
                max_turns=2,
                max_cost_usd=0.5,
            ),
        ),
    )

    wrapper.invoke(_invocation("first"))
    wrapper.invoke(_invocation("second"))

    assert [item.max_turns for item in backend.invocations] == [2, 1]
    assert [item.budget_usd for item in backend.invocations] == pytest.approx(
        [0.5, 0.3]
    )
    with pytest.raises(_conversation.ConversationBudgetExceeded):
        wrapper.invoke(_invocation("third"))
    assert len(backend.invocations) == 2


def test_turn_limit_terminal_charges_the_admitted_turns(
    tmp_path: Path,
) -> None:
    """A provider count of limit + 1 at the turn limit is not an overrun.

    The Claude CLI reports ``num_turns == limit + 1`` after ``limit`` requests.
    The typed turn-limit result must reach the caller; the session ledger
    ends at the ceiling this call was admitted with.
    """

    backend = _Backend("turn_limit:6")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    invocation = _invocation()

    result = wrapper.invoke(invocation)

    assert [item.max_turns for item in backend.invocations] == [5]
    assert isinstance(result.error, MaxTurnsExceeded)
    assert result.error.kind == "max_turns"
    assert result.error.turns_used == 6
    assert result.stop_reason == "max_turns"
    handle = wrapper._controller(invocation).load()  # noqa: SLF001
    assert handle is not None
    assert handle.pending_operation is None
    assert handle.turns_used == 5
    assert handle.usage.cost_usd == pytest.approx(0.2)
    assert len(handle.operation_digests) == 1


def test_turn_limit_terminal_after_earlier_turns_charges_the_remaining_ceiling(
    tmp_path: Path,
) -> None:
    """The ceiling is what this call was admitted with, not the session cap."""

    backend = _Backend("messages:2", "turn_limit:4")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )

    wrapper.invoke(_invocation("first"))
    result = wrapper.invoke(_invocation("second"))

    assert [item.max_turns for item in backend.invocations] == [5, 3]
    assert isinstance(result.error, MaxTurnsExceeded)
    handle = wrapper._controller(_invocation("second")).load()  # noqa: SLF001
    assert handle is not None
    assert handle.turns_used == 5


def test_turn_limit_terminal_below_the_session_ceiling_admits_a_further_call(
    tmp_path: Path,
) -> None:
    """An invocation limit below the session's remainder is what is charged.

    Session limit 10, invocation limit 5: the provider reports 6 turns at the
    turn limit, 5 are charged, and the next call is admitted with the 5
    turns that remain.
    """

    backend = _Backend("turn_limit:6", "active")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(
            tmp_path,
            limits=_conversation.ConversationLimits(
                max_turns=10,
                max_cost_usd=2.0,
            ),
        ),
    )

    first = wrapper.invoke(_invocation("first"))

    assert isinstance(first.error, MaxTurnsExceeded)
    assert first.error.turns_used == 6
    handle = wrapper._controller(_invocation("first")).load()  # noqa: SLF001
    assert handle is not None
    assert handle.turns_used == 5

    second = wrapper.invoke(_invocation("second"))

    assert second.error is None
    assert [item.max_turns for item in backend.invocations] == [5, 5]
    handle = wrapper._controller(_invocation("second")).load()  # noqa: SLF001
    assert handle is not None
    assert handle.turns_used == 6


def test_turn_limit_terminal_on_the_fresh_attempt_after_a_stale_reuse(
    tmp_path: Path,
) -> None:
    """The fresh attempt after a stale reuse is charged at its own ceiling.

    One turn is used, the reuse is rejected as stale, and the fresh attempt
    is admitted with the 4 turns that remain. Its turn-limit terminal reports
    5; the ledger ends at 1 + 4.
    """

    backend = _Backend("active", "stale", "turn_limit:5")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )

    wrapper.invoke(_invocation("first"))
    result = wrapper.invoke(_invocation("second"))

    assert isinstance(result.error, MaxTurnsExceeded)
    assert result.error.turns_used == 5
    assert [
        invocation.conversation.action
        for invocation in backend.invocations
        if invocation.conversation is not None
    ] == ["fresh", "reuse", "fresh"]
    assert [item.max_turns for item in backend.invocations] == [5, 4, 4]
    handle = wrapper._controller(_invocation("second")).load()  # noqa: SLF001
    assert handle is not None
    assert handle.pending_operation is None
    assert handle.turns_used == 5


def test_turn_limit_terminal_leaves_no_turn_for_a_further_call(
    tmp_path: Path,
) -> None:
    backend = _Backend("turn_limit:6", "active")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )

    wrapper.invoke(_invocation("first"))
    with pytest.raises(_conversation.ConversationBudgetExceeded):
        wrapper.invoke(_invocation("second"))

    assert len(backend.invocations) == 1


def test_turn_limit_terminal_more_than_one_over_the_ceiling_still_blocks(
    tmp_path: Path,
) -> None:
    """Only the exact refused-turn unit is discounted; an overrun is charged."""

    backend = _Backend("turn_limit:7")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    invocation = _invocation()

    with pytest.raises(_conversation.ConversationBudgetExceeded):
        wrapper.invoke(invocation)

    handle = wrapper._controller(invocation).load()  # noqa: SLF001
    assert handle is not None
    assert handle.turns_used == 7


def test_result_with_more_turns_than_admitted_still_exceeds_the_ceiling(
    tmp_path: Path,
) -> None:
    """A result that is not a turn-limit terminal is charged by its transcript."""

    backend = _Backend("messages:6")
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )
    invocation = _invocation()

    with pytest.raises(_conversation.ConversationBudgetExceeded):
        wrapper.invoke(invocation)

    handle = wrapper._controller(invocation).load()  # noqa: SLF001
    assert handle is not None
    assert handle.turns_used == 6


def test_unsupported_route_fails_before_backend_transport(
    tmp_path: Path,
) -> None:
    backend = _Backend(
        "active",
        profile=BackendFeatureSet(session_continuity="none"),
    )
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        conversation=_config(tmp_path),
    )

    with pytest.raises(_conversation.ConversationContinuityUnsupported):
        wrapper.invoke(_invocation())
    assert backend.invocations == []


def test_control_boundary_pauses_applies_steering_and_cancels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    binding = _control.bind_workflow_control(
        run_dir,
        identity=_control.ControlIdentity(
            run_fingerprint=_digest("1"),
            route_fingerprint=_digest("2"),
            authority_fingerprint=_digest("3"),
        ),
        owner="plan:planner",
        heartbeat_at="2026-07-28T11:59:00Z",
    )
    backend = _PlainBackend()
    wrapper = planner_session.PlannerBoundaryBackend(
        backend,
        control=binding,
    )
    state = binding.controller.load()
    assert state is not None
    paused = binding.controller.pause(
        expected_control_revision=state.control_revision,
    )

    with pytest.raises(planner_session.PlannerWorkflowPaused):
        wrapper.invoke(_invocation())
    assert backend.invocations == []

    resumed = binding.controller.resume(
        expected_control_revision=paused.control_revision,
    )
    steered = binding.controller.steer(
        instruction="Prefer a single independently verifiable slice",
        actor="operator",
        created_at="2026-07-28T12:01:00Z",
        expected_steering_revision=0,
    )
    assert steered.control_revision > resumed.control_revision
    wrapper.invoke(_invocation())
    assert len(backend.invocations) == 1
    assert "OPERATOR WORKFLOW STEERING" in backend.invocations[0].user
    assert "single independently verifiable slice" in (
        backend.invocations[0].user
    )

    binding.controller.request_cancel(
        expected_control_revision=steered.control_revision,
    )
    with pytest.raises(planner_session.PlannerWorkflowCancelled):
        wrapper.invoke(_invocation("another round"))
    assert len(backend.invocations) == 1


def test_cli_recovery_requires_exact_operation_evidence_and_actor() -> None:
    args = plan_workflow._parse_args([  # noqa: SLF001
        "--planner-count", "1",
        "--spec",
        "spec",
        "--planner-session-id",
        "session-1",
        "--planner-recovery-operation",
        _digest("1"),
        "--planner-recovery-evidence",
        _digest("2"),
        "--planner-recovery-actor",
        "operator:alice",
    ])
    recovery = plan_workflow._planner_recovery_from_args(args)  # noqa: SLF001
    assert recovery == planner_session.PlannerRecoveryInstruction(
        operation_digest=_digest("1"),
        evidence_digest=_digest("2"),
        actor="operator:alice",
    )

    args.planner_recovery_evidence = None
    with pytest.raises(ValueError, match="together"):
        plan_workflow._planner_recovery_from_args(args)  # noqa: SLF001
