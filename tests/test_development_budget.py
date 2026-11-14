# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import threading

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    Message,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
import pytest

from crucible.workflows._development_budget import (
    DevelopmentBudgetLedger,
    DevelopmentBudgetPolicy,
    RoleAllocation,
)


def _policy(*, calls: int = 1) -> DevelopmentBudgetPolicy:
    return DevelopmentBudgetPolicy(
        task_budget_usd=10.0,
        planning_review_cap_usd=4.0,
        implementation_verifier_reserve_usd=5.0,
        contingency_usd=1.0,
        roles=(
            RoleAllocation(
                role="planner",
                group="planning_review",
                max_calls=calls,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=4.0,
            ),
            RoleAllocation(
                role="implementer",
                group="implementation_verifier",
                max_calls=2,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=6.0,
            ),
        ),
    )


def _tail_policy(*, calls: int = 3) -> DevelopmentBudgetPolicy:
    base = _policy(calls=calls)
    return DevelopmentBudgetPolicy(
        task_budget_usd=base.task_budget_usd,
        planning_review_cap_usd=base.planning_review_cap_usd,
        implementation_verifier_reserve_usd=(
            base.implementation_verifier_reserve_usd
        ),
        contingency_usd=base.contingency_usd,
        roles=base.roles + (
            RoleAllocation(
                role="diff_scope",
                group="implementation_verifier",
                max_calls=2,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.5,
                protected_tail_usd=0.5,
            ),
            RoleAllocation(
                role="tier_completeness",
                group="implementation_verifier",
                max_calls=2,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.5,
                protected_tail_usd=0.5,
            ),
        ),
    )


def _release_policy(*, calls: int = 3) -> DevelopmentBudgetPolicy:
    base = _policy(calls=calls)
    return DevelopmentBudgetPolicy(
        task_budget_usd=base.task_budget_usd,
        planning_review_cap_usd=base.planning_review_cap_usd,
        implementation_verifier_reserve_usd=(
            base.implementation_verifier_reserve_usd
        ),
        contingency_usd=base.contingency_usd,
        roles=base.roles,
        release_unused_planning_authority=True,
    )


def _invocation(*, budget_usd: float = 10.0) -> Invocation:
    return Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=1,
        budget_usd=budget_usd,
    )


def _result(
    *,
    cost: float = 1.0,
    complete: bool = True,
    output_tokens: int = 3,
) -> InvocationResult:
    events = (
        BackendExecutionEvent(sequence=0, kind="request"),
        BackendExecutionEvent(sequence=1, kind="usage"),
        BackendExecutionEvent(sequence=2, kind="terminal", name="end_turn"),
    )
    return InvocationResult(
        final_message="done",
        transcript=(Message(role="assistant", content="done"),),
        usage=Usage(
            input_tokens=2,
            output_tokens=output_tokens,
            cost_usd=cost,
        ),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=complete,
            cost_complete=complete,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="stub",
            requested_model="stub-model",
            served_model="stub-model",
            events=events,
        ),
    )


class _Backend:
    name = "stub"
    model = "stub-model"

    def __init__(self, results: list[InvocationResult]) -> None:
        self.results = list(results)
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        return self.results.pop(0)


def test_policy_enforces_40_50_10_boundaries():
    assert _policy().schema_version == "crucible.development-budget.v3"
    assert _policy().digest.startswith("sha256:")
    with pytest.raises(ValueError, match="40%"):
        DevelopmentBudgetPolicy(
            task_budget_usd=10.0,
            planning_review_cap_usd=4.01,
            implementation_verifier_reserve_usd=5.0,
            contingency_usd=1.0,
            roles=_policy().roles,
        )
    with pytest.raises(ValueError, match="50%"):
        DevelopmentBudgetPolicy(
            task_budget_usd=10.0,
            planning_review_cap_usd=4.0,
            implementation_verifier_reserve_usd=4.99,
            contingency_usd=1.01,
            roles=_policy().roles,
        )


def test_policy_rejects_invalid_protected_tail_authority():
    planning_owner = RoleAllocation(
        role="planner",
        group="planning_review",
        max_calls=1,
        max_output_tokens=100,
        max_wall_seconds=60.0,
        max_cost_usd=1.0,
        protected_tail_usd=0.5,
    )
    with pytest.raises(ValueError, match="implementation/verifier"):
        DevelopmentBudgetPolicy(
            task_budget_usd=10.0,
            planning_review_cap_usd=4.0,
            implementation_verifier_reserve_usd=5.0,
            contingency_usd=1.0,
            roles=(planning_owner,),
        )

    base = _policy()
    oversized_tail = RoleAllocation(
        role="review",
        group="implementation_verifier",
        max_calls=1,
        max_output_tokens=100,
        max_wall_seconds=60.0,
        max_cost_usd=1.1,
        protected_tail_usd=1.1,
    )
    with pytest.raises(ValueError, match="exceeds contingency"):
        DevelopmentBudgetPolicy(
            task_budget_usd=10.0,
            planning_review_cap_usd=4.0,
            implementation_verifier_reserve_usd=5.0,
            contingency_usd=1.0,
            roles=base.roles + (oversized_tail,),
        )


def test_policy_digest_binds_protected_tail_allocations():
    protected = _tail_policy()
    unprotected = DevelopmentBudgetPolicy(
        task_budget_usd=protected.task_budget_usd,
        planning_review_cap_usd=protected.planning_review_cap_usd,
        implementation_verifier_reserve_usd=(
            protected.implementation_verifier_reserve_usd
        ),
        contingency_usd=protected.contingency_usd,
        roles=tuple(
            RoleAllocation(
                role=allocation.role,
                group=allocation.group,
                max_calls=allocation.max_calls,
                max_output_tokens=allocation.max_output_tokens,
                max_wall_seconds=allocation.max_wall_seconds,
                max_cost_usd=allocation.max_cost_usd,
            )
            for allocation in protected.roles
        ),
    )

    assert protected.digest != unprotected.digest


def test_policy_digest_binds_unused_planning_release_authority():
    assert _policy().digest != _release_policy().digest


def test_policy_rejects_non_boolean_unused_planning_release():
    base = _policy()
    with pytest.raises(ValueError, match="must be a boolean"):
        DevelopmentBudgetPolicy(
            task_budget_usd=base.task_budget_usd,
            planning_review_cap_usd=base.planning_review_cap_usd,
            implementation_verifier_reserve_usd=(
                base.implementation_verifier_reserve_usd
            ),
            contingency_usd=base.contingency_usd,
            roles=base.roles,
            release_unused_planning_authority=1,  # type: ignore[arg-type]
        )


def test_policy_only_allows_potential_planning_authority_when_released():
    base = _policy()
    expanded_implementer = RoleAllocation(
        role="implementer",
        group="implementation_verifier",
        max_calls=2,
        max_output_tokens=100,
        max_wall_seconds=60.0,
        max_cost_usd=10.0,
    )
    roles = (base.roles[0], expanded_implementer)

    with pytest.raises(ValueError, match="exceeds its group USD cap"):
        DevelopmentBudgetPolicy(
            task_budget_usd=base.task_budget_usd,
            planning_review_cap_usd=base.planning_review_cap_usd,
            implementation_verifier_reserve_usd=(
                base.implementation_verifier_reserve_usd
            ),
            contingency_usd=base.contingency_usd,
            roles=roles,
            release_unused_planning_authority=False,
        )

    released = DevelopmentBudgetPolicy(
        task_budget_usd=base.task_budget_usd,
        planning_review_cap_usd=base.planning_review_cap_usd,
        implementation_verifier_reserve_usd=(
            base.implementation_verifier_reserve_usd
        ),
        contingency_usd=base.contingency_usd,
        roles=roles,
        release_unused_planning_authority=True,
    )
    assert released.roles[1].max_cost_usd == 10.0

    above_potential_cap = RoleAllocation(
        role="implementer",
        group="implementation_verifier",
        max_calls=2,
        max_output_tokens=100,
        max_wall_seconds=60.0,
        max_cost_usd=10.01,
    )
    with pytest.raises(ValueError, match="exceeds its group USD cap"):
        DevelopmentBudgetPolicy(
            task_budget_usd=base.task_budget_usd,
            planning_review_cap_usd=base.planning_review_cap_usd,
            implementation_verifier_reserve_usd=(
                base.implementation_verifier_reserve_usd
            ),
            contingency_usd=base.contingency_usd,
            roles=(base.roles[0], above_potential_cap),
            release_unused_planning_authority=True,
        )


def test_policy_rejects_a_stale_or_forged_schema_identity():
    base = _policy()
    with pytest.raises(ValueError, match="schema_version must be"):
        DevelopmentBudgetPolicy(
            task_budget_usd=base.task_budget_usd,
            planning_review_cap_usd=base.planning_review_cap_usd,
            implementation_verifier_reserve_usd=(
                base.implementation_verifier_reserve_usd
            ),
            contingency_usd=base.contingency_usd,
            roles=base.roles,
            schema_version="crucible.development-budget.v2",
        )


def test_policy_preserves_the_existing_positional_schema_slot():
    base = _policy()
    positional = DevelopmentBudgetPolicy(
        base.task_budget_usd,
        base.planning_review_cap_usd,
        base.implementation_verifier_reserve_usd,
        base.contingency_usd,
        base.roles,
        "crucible.development-budget.v3",
    )

    assert positional.release_unused_planning_authority is False


def test_missing_role_rejects_before_backend_use():
    ledger = DevelopmentBudgetLedger(_policy())
    with pytest.raises(ValueError, match="no allocation"):
        ledger.wrap("judge", _Backend([]))


def test_call_allocation_stops_second_transport():
    ledger = DevelopmentBudgetLedger(_policy(calls=1))
    raw = _Backend([_result()])
    backend = ledger.wrap("planner", raw)

    first = backend.invoke(_invocation())
    second = backend.invoke(_invocation())

    assert first.final_message == "done"
    assert second.error is not None
    assert second.error.kind == "budget_exhausted"
    assert type(second.pricing_evidence) is UsagePricingEvidence
    assert second.pricing_evidence.segments == ()
    assert second.pricing_evidence.complete is True
    assert len(raw.invocations) == 1
    summary = ledger.to_summary()
    assert summary["roles"]["planner"]["usage"]["calls"] == 1
    assert summary["roles"]["planner"]["exhausted_by"] == "call_count"
    assert summary["group_remaining_usd"]["implementation_verifier"] == 6.0


def test_incomplete_accounting_revokes_all_later_paid_authority():
    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    planner_raw = _Backend([_result(complete=False)])
    implementer_raw = _Backend([_result()])
    planner = ledger.wrap("planner", planner_raw)
    implementer = ledger.wrap("implementer", implementer_raw)

    planner.invoke(_invocation())
    blocked = implementer.invoke(_invocation())

    assert blocked.error is not None
    assert blocked.error.kind == "budget_exhausted"
    assert implementer_raw.invocations == []
    summary = ledger.to_summary()
    assert summary["authority_revoked"] is True
    assert summary["roles"]["planner"]["accounting_debt"] is True


def test_terminal_incomplete_error_with_exact_accounting_keeps_authority():
    q8_cost = 0.059551
    segment = UsagePricingSegment(
        input_tokens=10_738,
        output_tokens=4,
        cache_creation_tokens=9_404,
        cache_read_tokens=1_332,
        cost_usd=q8_cost,
    )
    accounted_error = InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(
            input_tokens=10_738,
            output_tokens=4,
            cost_usd=q8_cost,
            cache_creation_tokens=9_404,
            cache_read_tokens=1_332,
        ),
        stop_reason="error",
        error=TransportError(
            http_status=None,
            message="terminal API error status unavailable",
            retriable=False,
        ),
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=False,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="claude",
            requested_model="claude-opus-5",
            served_model=None,
            events=(
                BackendExecutionEvent(sequence=0, kind="request"),
                BackendExecutionEvent(sequence=1, kind="usage"),
                BackendExecutionEvent(
                    sequence=2,
                    kind="terminal",
                    name="error",
                ),
            ),
        ),
        pricing_evidence=UsagePricingEvidence(
            segments=(segment,),
            complete=True,
        ),
    )
    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    planner = ledger.wrap("planner", _Backend([accounted_error]))
    implementer_raw = _Backend([_result()])
    implementer = ledger.wrap("implementer", implementer_raw)

    observed = planner.invoke(_invocation())
    continued = implementer.invoke(_invocation())

    assert observed is accounted_error
    assert continued.error is None
    assert len(implementer_raw.invocations) == 1
    summary = ledger.to_summary()
    assert summary["authority_revoked"] is False
    assert summary["roles"]["planner"]["accounting_debt"] is False
    assert summary["roles"]["planner"]["usage"]["cost_usd"] == pytest.approx(
        q8_cost
    )


@pytest.mark.parametrize(
    "malformed_cost",
    [math.nan, 10**10_000],
    ids=["nan", "overflowing_integer"],
)
def test_malformed_incomplete_usage_revokes_authority_without_raising(
    malformed_cost,
):
    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    malformed = InvocationResult(
        final_message="partial",
        transcript=(Message(role="assistant", content="partial"),),
        usage=Usage(
            output_tokens=math.nan,  # type: ignore[arg-type]
            cost_usd=malformed_cost,
        ),
        stop_reason="error",
        completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=False,
            trace_complete=False,
        ),
    )
    planner = ledger.wrap("planner", _Backend([malformed]))
    implementer_raw = _Backend([_result()])
    implementer = ledger.wrap("implementer", implementer_raw)

    observed = planner.invoke(_invocation())
    blocked = implementer.invoke(_invocation())

    assert observed is malformed
    assert blocked.error is not None
    assert blocked.error.kind == "budget_exhausted"
    assert implementer_raw.invocations == []
    summary = ledger.to_summary()
    assert summary["authority_revoked"] is True
    assert summary["roles"]["planner"]["accounting_debt"] is True
    assert summary["roles"]["planner"]["usage"]["cost_usd"] == 0.0
    assert summary["roles"]["planner"]["usage"]["output_tokens"] == 0


def test_in_flight_reservation_prevents_concurrent_group_oversubscription():
    entered = threading.Event()
    release = threading.Event()

    class _BlockingBackend(_Backend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            self.invocations.append(invocation)
            entered.set()
            assert release.wait(timeout=5.0)
            return self.results.pop(0)

    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    raw = _BlockingBackend([_result()])
    backend = ledger.wrap("planner", raw)
    first: list[InvocationResult] = []
    thread = threading.Thread(
        target=lambda: first.append(backend.invoke(_invocation())),
    )
    thread.start()
    assert entered.wait(timeout=5.0)

    blocked = backend.invoke(_invocation())
    summary_while_in_flight = ledger.to_summary()
    release.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert first[0].final_message == "done"
    assert blocked.error is not None
    assert blocked.error.kind == "budget_exhausted"
    assert len(raw.invocations) == 1
    assert summary_while_in_flight["group_reserved_usd"][
        "planning_review"
    ] == 4.0
    assert summary_while_in_flight["group_remaining_usd"][
        "planning_review"
    ] == 0.0


def test_phase_admission_during_planning_reservation_releases_nothing():
    entered = threading.Event()
    release = threading.Event()

    class _BlockingBackend(_Backend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            self.invocations.append(invocation)
            entered.set()
            assert release.wait(timeout=5.0)
            return self.results.pop(0)

    ledger = DevelopmentBudgetLedger(_release_policy(calls=2))
    planner = ledger.wrap("planner", _BlockingBackend([_result(cost=1.0)]))
    thread = threading.Thread(
        target=lambda: planner.invoke(_invocation(budget_usd=4.0)),
    )
    thread.start()
    assert entered.wait(timeout=5.0)

    assert ledger.admit_implementation_phase() == (
        False,
        "planning_review_accounting_unsettled",
    )
    blocked = ledger.wrap("implementer", _Backend([_result()])).invoke(
        _invocation()
    )
    release.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert blocked.error is not None
    summary = ledger.to_summary()
    assert summary["authority_revoked"] is True
    assert summary["phase_authority"][
        "implementation_phase_admitted"
    ] is False
    assert summary["phase_authority"][
        "planning_authority_released_usd"
    ] == 0.0


def test_implementation_phase_admission_revokes_after_planning_soft_overshoot():
    within_ledger = DevelopmentBudgetLedger(_policy())
    within_ledger.wrap(
        "planner",
        _Backend([_result(cost=4.0)]),
    ).invoke(_invocation(budget_usd=4.0))

    assert within_ledger.admit_implementation_phase() == (True, None)

    overshoot_ledger = DevelopmentBudgetLedger(_policy())
    overshoot_ledger.wrap(
        "planner",
        _Backend([_result(cost=6.0)]),
    ).invoke(_invocation(budget_usd=4.0))

    admitted, reason = overshoot_ledger.admit_implementation_phase()
    implementer_raw = _Backend([_result()])
    blocked = overshoot_ledger.wrap(
        "implementer",
        implementer_raw,
    ).invoke(_invocation())

    assert admitted is False
    assert reason == "planning_review_cap_exceeded"
    assert blocked.error is not None
    assert blocked.error.kind == "budget_exhausted"
    assert implementer_raw.invocations == []
    summary = overshoot_ledger.to_summary()
    assert summary["authority_revoked"] is True
    assert summary["group_spend_usd"]["planning_review"] == 6.0
    assert summary["group_remaining_usd"]["implementation_verifier"] == 6.0


def test_unused_settled_planning_authority_releases_once_and_closes_phase():
    ledger = DevelopmentBudgetLedger(_release_policy(calls=2))
    premature_raw = _Backend([_result()])
    premature = ledger.wrap("implementer", premature_raw).invoke(
        _invocation()
    )
    assert premature.error is not None
    assert premature_raw.invocations == []

    planner_raw = _Backend([_result(cost=1.0), _result(cost=0.1)])
    planner = ledger.wrap("planner", planner_raw)
    planner.invoke(_invocation(budget_usd=4.0))

    assert ledger.admit_implementation_phase() == (True, None)
    assert ledger.admit_implementation_phase() == (True, None)
    blocked_planner = planner.invoke(_invocation(budget_usd=1.0))
    assert blocked_planner.error is not None
    assert len(planner_raw.invocations) == 1

    implementer_raw = _Backend([_result(cost=5.5)])
    result = ledger.wrap("implementer", implementer_raw).invoke(
        _invocation(budget_usd=9.0)
    )
    assert result.error is None
    assert implementer_raw.invocations[0].budget_usd == 6.0

    summary = ledger.to_summary()
    assert summary["phase_authority"] == {
        "implementation_phase_admitted": True,
        "unused_planning_authority_release_enabled": True,
        "planning_authority_released_usd": 3.0,
        "settled_planning_spend_at_close_usd": 1.0,
    }
    assert summary["group_cap_usd"] == {
        "planning_review": 4.0,
        "implementation_verifier": 6.0,
    }
    assert summary["active_group_cap_usd"] == {
        "planning_review": 0.0,
        "implementation_verifier": 9.0,
    }
    assert summary["group_remaining_usd"] == {
        "planning_review": 0.0,
        "implementation_verifier": 3.5,
    }
    authority = summary["implementation_verifier_authority"]
    assert authority["released_planning_cap_usd"] == 3.0
    assert authority["released_planning_spend_usd"] == 0.5
    assert authority["released_planning_remaining_usd"] == 2.5
    assert authority["contingency_remaining_usd"] == 1.0
    assert summary["roles"]["implementer"]["exhausted_by"] is None
    assert summary["task_authority"] == {
        "phase_committed_cap_usd": 10.0,
        "settled_spend_usd": 6.5,
        "reserved_usd": 0.0,
        "remaining_usd": 3.5,
        "post_hoc_overshoot_usd": 0.0,
        "phase_commitment_within_task_cap": True,
    }
    assert summary["group_spend_usd"]["planning_review"] + summary[
        "group_spend_usd"
    ]["implementation_verifier"] == 6.5


def test_deepswe_v16_costs_fit_reclaimed_authority_and_breaker_tail():
    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(_deepswe_development_budget_policy(12.0))
    planning_costs = {
        "architect_planner": 0.4748235,
        "architect_critic": 0.64054,
        "architect_judge": 0.586784,
    }
    for role, cost in planning_costs.items():
        observed = ledger.wrap(role, _Backend([_result(cost=cost)])).invoke(
            _invocation(budget_usd=12.0)
        )
        assert observed.error is None
    assert ledger.admit_implementation_phase() == (True, None)

    continuation_costs = {
        "implementer": 5.87098025,
        "impl_breaker": 1.25020125,
        "breach_judge": 0.683548,
        "diff_scope": 0.10700375,
        "tier_completeness": 0.139505,
    }
    bounded: dict[str, float | None] = {}
    for role, cost in continuation_costs.items():
        raw = _Backend([_result(cost=cost)])
        observed = ledger.wrap(role, raw).invoke(
            _invocation(budget_usd=12.0)
        )
        assert observed.error is None
        bounded[role] = raw.invocations[0].budget_usd

    assert bounded["implementer"] == pytest.approx(9.0978525)
    assert bounded["impl_breaker"] == pytest.approx(3.82687225)
    calls_by_role = ledger.to_summary()["autonomous_development_metrics"][
        "calls_by_role"
    ]
    for role in ledger.protected_tail_roles:
        if calls_by_role[role] > 0:
            ledger.complete_protected_tail_role(role)

    summary = ledger.to_summary()
    assert summary["phase_authority"][
        "planning_authority_released_usd"
    ] == pytest.approx(3.0978525)
    assert summary["active_group_cap_usd"][
        "implementation_verifier"
    ] == pytest.approx(10.2978525)
    assert summary["group_spend_usd"][
        "implementation_verifier"
    ] == pytest.approx(8.05123825)
    assert sum(summary["group_spend_usd"].values()) == pytest.approx(
        9.75338575
    )
    assert summary["active_group_cap_usd"][
        "implementation_verifier"
    ] + summary["group_spend_usd"]["planning_review"] == pytest.approx(
        12.0
    )
    authority = summary["implementation_verifier_authority"]
    assert authority["released_planning_spend_usd"] == pytest.approx(
        2.051238
    )
    assert authority["contingency_spend_usd"] == 0.0
    assert authority["contingency_remaining_usd"] == 1.2
    assert authority["post_hoc_overshoot_usd"] == 0.0
    assert summary["active_group_cap_usd"] == {
        "planning_review": 0.0,
        "implementation_verifier": pytest.approx(10.2978525),
    }
    assert summary["task_authority"][
        "phase_committed_cap_usd"
    ] == pytest.approx(12.0)
    assert summary["task_authority"]["post_hoc_overshoot_usd"] == 0.0
    assert summary["task_authority"][
        "phase_commitment_within_task_cap"
    ] is True
    owners = authority["protected_tail"]["owners"]
    assert set(owners) == {
        "impl_breaker",
        "breach_defender",
        "diff_scope",
        "tier_completeness",
    }
    assert owners["breach_defender"]["terminal_completed"] is False
    assert owners["breach_defender"]["remaining_usd"] == pytest.approx(0.36)
    assert all(
        record["terminal_completed"]
        for role, record in owners.items()
        if role != "breach_defender"
    )


def test_deepswe_review_token_softening_preserves_twelve_dollar_authority():
    """Planning convergence caps preserve the unchanged $12 authority."""

    from crucible.products.deepswe.spec_synthesis import (
        resolve_deepswe_plan_policy,
    )
    from crucible.workflows import architect, deepswe

    policy = deepswe._deepswe_development_budget_policy(12.0, planner_count=1)
    allocations = {allocation.role: allocation for allocation in policy.roles}
    plan_policy = resolve_deepswe_plan_policy(12.0)

    assert policy.task_budget_usd == 12.0
    assert policy.planning_review_cap_usd == pytest.approx(4.8)
    assert policy.implementation_verifier_reserve_usd == 6.0
    assert policy.contingency_usd == pytest.approx(1.2)
    assert policy.implementation_verifier_cap_usd == pytest.approx(7.2)
    planner = allocations["architect_planner"]
    assert (
        planner.max_calls,
        planner.max_output_tokens,
        planner.max_wall_seconds,
        planner.max_cost_usd,
    ) == (9, 147_456, 3_600.0, pytest.approx(4.8))
    assert deepswe._PLANNER_SEMANTIC_CANDIDATE_LIMIT == (
        plan_policy.max_critique_rounds + 1
    )
    assert {
        resolve_deepswe_plan_policy(envelope).max_critique_rounds + 1
        for envelope in (None, 12.0, 12.01)
    } == {deepswe._PLANNER_SEMANTIC_CANDIDATE_LIMIT}
    assert deepswe._PLANNER_CONDITIONAL_REPAIR_LIMIT == (
        plan_policy.inflation_compaction_retries
        + architect._CANDIDATE_VALIDATION_REPAIR_LIMIT
    )
    for role in ("architect_critic", "architect_judge"):
        reviewer = allocations[role]
        assert reviewer.group == "planning_review"
        assert reviewer.max_calls == 21
        assert reviewer.max_output_tokens == 688_128
        assert reviewer.max_wall_seconds == 1_800.0
        assert reviewer.max_cost_usd == pytest.approx(4.8)
        assert reviewer.protected_tail_usd == 0.0


@pytest.mark.parametrize("role", ["architect_critic", "architect_judge"])
def test_deepswe_review_repair_headroom_covers_all_candidates(role):
    """Physical repair calls cover every semantic candidate."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(
        _deepswe_development_budget_policy(12.0, planner_count=1)
    )
    raw = _Backend([
        _result(cost=0.01, output_tokens=32_767) for _ in range(21)
    ])
    wrapped = ledger.wrap(role, raw)

    for _ in range(21):
        observed = wrapped.invoke(_invocation(budget_usd=12.0))
        assert observed.error is None

    denied = wrapped.invoke(_invocation(budget_usd=12.0))
    assert denied.error is not None
    assert denied.error.kind == "budget_exhausted"
    assert denied.completeness.token_usage_complete is True
    assert denied.completeness.cost_complete is True
    assert len(raw.invocations) == 21
    summary = ledger.to_summary()["roles"][role]
    assert summary["limits"]["calls"] == 21
    assert summary["limits"]["output_tokens"] == 688_128
    assert summary["usage"]["calls"] == 21
    assert summary["usage"]["output_tokens"] == 688_107
    assert summary["exhausted_by"] == "call_count"


@pytest.mark.parametrize("count,critic_calls,judge_calls", [(3, 21, 28), (8, 45, 58)])
def test_deepswe_competition_accounts_for_all_reviews_within_shared_authority(
    count, critic_calls, judge_calls,
):
    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    policy = _deepswe_development_budget_policy(12.0, planner_count=count)
    allocations = {item.role: item for item in policy.roles}
    assert allocations["architect_planner"].max_calls == count
    assert allocations["architect_critic"].max_calls == critic_calls
    assert allocations["architect_judge"].max_calls == judge_calls
    assert allocations["architect_planner"].max_output_tokens >= 147_456
    assert allocations["architect_critic"].max_output_tokens >= 688_128
    assert allocations["architect_judge"].max_output_tokens >= 688_128
    assert policy.planning_review_cap_usd == pytest.approx(4.8)
    assert policy.implementation_verifier_reserve_usd == 6.0
    assert policy.contingency_usd == pytest.approx(1.2)

    ledger = DevelopmentBudgetLedger(policy)
    # All planners and reviews settle against one planning group. Exhausting
    # it cannot dispatch another review or spend the implementation reserve.
    half_planning_cap = policy.planning_review_cap_usd / 2
    planner = ledger.wrap(
        "architect_planner", _Backend([_result(cost=half_planning_cap)]),
    )
    critic_raw = _Backend([
        _result(cost=half_planning_cap), _result(cost=0.01),
    ])
    critic = ledger.wrap("architect_critic", critic_raw)
    assert planner.invoke(_invocation(budget_usd=12.0)).error is None
    assert critic.invoke(_invocation(budget_usd=12.0)).error is None
    denied = critic.invoke(_invocation(budget_usd=12.0))
    assert denied.error is not None
    assert denied.error.kind == "budget_exhausted"
    assert len(critic_raw.invocations) == 1


def test_deepswe_planner_output_boundary_settles_after_final_call():
    """Seven candidates retain both conditional Planner repair slots."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(
        _deepswe_development_budget_policy(12.0, planner_count=1)
    )
    raw = _Backend([
        _result(cost=0.01, output_tokens=16_384) for _ in range(9)
    ])
    wrapped = ledger.wrap("architect_planner", raw)

    for _ in range(9):
        observed = wrapped.invoke(_invocation(budget_usd=12.0))
        assert observed.error is None

    denied = wrapped.invoke(_invocation(budget_usd=12.0))
    assert denied.error is not None
    assert denied.error.kind == "budget_exhausted"
    assert len(raw.invocations) == 9
    summary = ledger.to_summary()["roles"]["architect_planner"]
    assert summary["limits"]["calls"] == 9
    assert summary["limits"]["output_tokens"] == 147_456
    assert summary["usage"]["calls"] == 9
    assert summary["usage"]["output_tokens"] == 147_456
    assert summary["exhausted_by"] == "output_tokens"


def test_deepswe_r13_critic_shape_can_dispatch_next_bounded_repair():
    """R13's settled Critic output cannot deny the next authorized repair."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    observed_outputs = [13_793, 9_912, 7_422, 7_325, 13_625, 13_643]
    ledger = DevelopmentBudgetLedger(_deepswe_development_budget_policy(12.0))
    raw = _Backend([
        *(
            _result(cost=0.01, output_tokens=output_tokens)
            for output_tokens in observed_outputs
        ),
        _result(cost=0.01, output_tokens=32_768),
    ])
    wrapped = ledger.wrap("architect_critic", raw)

    for _ in range(7):
        observed = wrapped.invoke(_invocation(budget_usd=12.0))
        assert observed.error is None

    assert len(raw.invocations) == 7
    summary = ledger.to_summary()
    critic = summary["roles"]["architect_critic"]
    assert sum(observed_outputs) == 65_720
    assert critic["usage"]["calls"] == 7
    assert critic["usage"]["output_tokens"] == 98_488
    assert critic["exhausted_by"] is None
    assert summary["group_remaining_usd"]["planning_review"] == pytest.approx(
        4.73
    )


@pytest.mark.parametrize("role", ["architect_critic", "architect_judge"])
def test_deepswe_review_output_boundary_settles_after_final_call(role):
    """The exact output boundary cannot deny any authorized physical call."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(
        _deepswe_development_budget_policy(12.0, planner_count=1)
    )
    raw = _Backend([
        _result(cost=0.01, output_tokens=32_768) for _ in range(21)
    ])
    wrapped = ledger.wrap(role, raw)

    for _ in range(21):
        observed = wrapped.invoke(_invocation(budget_usd=12.0))
        assert observed.error is None

    denied = wrapped.invoke(_invocation(budget_usd=12.0))
    assert denied.error is not None
    assert denied.error.kind == "budget_exhausted"
    assert len(raw.invocations) == 21
    summary = ledger.to_summary()["roles"][role]
    assert summary["limits"]["calls"] == 21
    assert summary["limits"]["output_tokens"] == 688_128
    assert summary["usage"]["calls"] == 21
    assert summary["usage"]["output_tokens"] == 688_128
    # Both ceilings are now exactly spent. The ledger reports output first,
    # but only after every authorized call completed and settled.
    assert summary["exhausted_by"] == "output_tokens"


def test_deepswe_tail_reviewer_can_borrow_unprotected_r3_authority():
    """Regression for E6-R3's fixed reviewer micro-cap starvation."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(_deepswe_development_budget_policy(12.0))
    planning_costs = {
        "architect_planner": [0.32578425],
        "architect_critic": [0.2132585],
        "architect_judge": [0.283847, 0.268237],
    }
    for role, costs in planning_costs.items():
        backend = _Backend([_result(cost=cost) for cost in costs])
        wrapped = ledger.wrap(role, backend)
        for _ in costs:
            observed = wrapped.invoke(_invocation(budget_usd=12.0))
            assert observed.error is None
    assert ledger.admit_implementation_phase() == (True, None)

    continuation_costs = {
        "implementer": [5.85333875, 0.18030725],
        "impl_breaker": [1.27440125],
        "breach_judge": [0.331031, 0.36399275, 0.096069],
        "diff_scope": [0.214723],
    }
    for role, costs in continuation_costs.items():
        backend = _Backend([_result(cost=cost) for cost in costs])
        wrapped = ledger.wrap(role, backend)
        for _ in costs:
            observed = wrapped.invoke(_invocation(budget_usd=12.0))
            assert observed.error is None

    completeness = _Backend([_result(cost=0.32262525)])
    observed = ledger.wrap("tier_completeness", completeness).invoke(
        _invocation(budget_usd=12.0)
    )

    assert observed.error is None
    assert completeness.invocations[0].budget_usd == pytest.approx(2.23501025)
    assert completeness.invocations[0].budget_usd > 0.32262525
    summary = ledger.to_summary()
    assert summary["roles"]["tier_completeness"]["limits"][
        "protected_tail_usd"
    ] == pytest.approx(0.12)
    assert summary["roles"]["tier_completeness"]["usage"][
        "request_bound_overshoot_usd"
    ] == 0.0
    tail = summary["implementation_verifier_authority"]["protected_tail"]
    assert tail["owners"]["diff_scope"]["remaining_usd"] == 0.0
    assert tail["owners"]["breach_defender"]["remaining_usd"] == (
        pytest.approx(0.36)
    )
    assert tail["request_bound_overshoot_usd"] == 0.0
    assert summary["task_authority"]["post_hoc_overshoot_usd"] == 0.0
    assert summary["task_authority"][
        "phase_commitment_within_task_cap"
    ] is True


def test_deepswe_q6_shape_retains_a_production_defender_dispatch() -> None:
    """The shared mutator gets repair authority before nested budget close."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(_deepswe_development_budget_policy(12.0))
    planning_costs = {
        "architect_planner": 0.27306275,
        "architect_critic": 0.29223175,
        "architect_judge": 0.2085975,
    }
    for role, cost in planning_costs.items():
        observed = ledger.wrap(role, _Backend([_result(cost=cost)])).invoke(
            _invocation(budget_usd=12.0)
        )
        assert observed.error is None
    assert ledger.admit_implementation_phase() == (True, None)

    continuation_costs = {
        "implementer": [6.5882275, 0.797063],
        "impl_breaker": [1.6967575],
        "breach_judge": [0.302385, 0.535976],
    }
    for role, costs in continuation_costs.items():
        backend = _Backend([_result(cost=cost) for cost in costs])
        wrapped = ledger.wrap(role, backend)
        for _ in costs:
            observed = wrapped.invoke(_invocation(budget_usd=12.0))
            assert observed.error is None

    shared_mutator = _Backend([
        _result(cost=0.482651),
        _result(cost=0.25720625),
    ])
    wrapped_mutator = ledger.wrap("breach_defender", shared_mutator)
    test_author = wrapped_mutator.invoke(_invocation(budget_usd=12.0))
    defender_capacity = ledger.role_dispatch_capacity(
        "breach_defender",
        requested_cost_usd=12.0,
    )
    defender = wrapped_mutator.invoke(_invocation(budget_usd=12.0))

    assert test_author.error is None
    assert defender.error is None
    assert defender_capacity == pytest.approx(0.58304825)
    assert shared_mutator.invocations[1].budget_usd == pytest.approx(
        defender_capacity
    )
    assert defender_capacity > 0.25720625
    final_judge_capacity = ledger.role_dispatch_capacity(
        "breach_judge",
        requested_cost_usd=12.0,
    )
    assert final_judge_capacity == pytest.approx(0.32584175)
    assert final_judge_capacity < ledger.role_protected_tail_usd(
        "impl_breaker"
    )


def test_deepswe_larger_budget_retains_terminal_reviewer_tails() -> None:
    """The compact repair experiment does not weaken larger-task reviews."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    policy = _deepswe_development_budget_policy(12.01)
    allocations = {allocation.role: allocation for allocation in policy.roles}

    assert allocations["impl_breaker"].protected_tail_usd == pytest.approx(
        12.01 * 0.05
    )
    assert allocations["breach_defender"].protected_tail_usd == 0.0
    for role in ("diff_scope", "tier_completeness"):
        assert allocations[role].protected_tail_usd == pytest.approx(
            12.01 * 0.025
        )


def test_deepswe_implementer_repair_can_use_reclaimed_authority_without_eroding_tails():
    """Regression for E7-R10's stranded released planning authority."""

    from crucible.workflows.deepswe import _deepswe_development_budget_policy

    ledger = DevelopmentBudgetLedger(_deepswe_development_budget_policy(12.0))
    for role, cost in {
        "architect_planner": 0.465925,
        "architect_critic": 0.710727,
        "architect_judge": 0.576892,
    }.items():
        observed = ledger.wrap(role, _Backend([_result(cost=cost)])).invoke(
            _invocation(budget_usd=12.0)
        )
        assert observed.error is None
    assert ledger.admit_implementation_phase() == (True, None)

    assert ledger.role_dispatch_capacity("implementer") == pytest.approx(
        9.046456
    )
    implementer_raw = _Backend([
        _result(cost=5.68617475),
        _result(cost=1.1788455),
        _result(cost=0.34045775),
    ])
    implementer = ledger.wrap("implementer", implementer_raw)
    first = implementer.invoke(_invocation(budget_usd=12.0))
    assert ledger.role_dispatch_capacity("implementer") == pytest.approx(
        3.36028125
    )
    repair = implementer.invoke(_invocation(budget_usd=12.0))
    third_call_capacity = ledger.role_dispatch_capacity("implementer")
    terminal = implementer.invoke(_invocation(budget_usd=12.0))

    assert first.error is None
    assert repair.error is None
    assert terminal.error is None
    assert third_call_capacity == pytest.approx(2.18143575)
    assert [item.budget_usd for item in implementer_raw.invocations] == [
        pytest.approx(9.046456),
        pytest.approx(3.36028125),
        pytest.approx(2.18143575),
    ]
    summary = ledger.to_summary()
    assert summary["roles"]["implementer"]["usage"]["cost_usd"] == (
        pytest.approx(7.205478)
    )
    assert summary["roles"]["implementer"]["exhausted_by"] is None
    authority = summary["implementation_verifier_authority"]
    assert authority["effective_cap_usd"] == pytest.approx(10.246456)
    assert authority["effective_remaining_usd"] == pytest.approx(3.040978)
    assert authority["released_planning_cap_usd"] == pytest.approx(3.046456)
    assert authority["released_planning_spend_usd"] == pytest.approx(1.205478)
    assert authority["released_planning_remaining_usd"] == pytest.approx(
        1.840978
    )
    assert authority["contingency_remaining_usd"] == pytest.approx(1.2)
    assert authority["protected_tail"]["outstanding_usd"] == pytest.approx(1.2)
    assert authority["protected_tail"]["physically_available_usd"] == (
        pytest.approx(1.2)
    )
    assert authority["protected_tail"]["eroded_usd"] == 0.0
    assert summary["task_authority"]["remaining_usd"] == pytest.approx(
        3.040978
    )
    assert summary["task_authority"]["phase_commitment_within_task_cap"] is True


def test_reclaimed_authority_concurrently_preserves_all_tail_reservations():
    base = _tail_policy(calls=2)
    policy = DevelopmentBudgetPolicy(
        task_budget_usd=base.task_budget_usd,
        planning_review_cap_usd=base.planning_review_cap_usd,
        implementation_verifier_reserve_usd=(
            base.implementation_verifier_reserve_usd
        ),
        contingency_usd=base.contingency_usd,
        roles=base.roles
        + (
            RoleAllocation(
                role="reviewer",
                group="implementation_verifier",
                max_calls=1,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=6.0,
            ),
        ),
        release_unused_planning_authority=True,
    )
    ledger = DevelopmentBudgetLedger(policy)
    ledger.wrap("planner", _Backend([_result(cost=1.0)])).invoke(
        _invocation(budget_usd=4.0)
    )
    assert ledger.admit_implementation_phase() == (True, None)

    entered = threading.Event()
    release = threading.Event()

    class _BlockingBackend(_Backend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            self.invocations.append(invocation)
            entered.set()
            assert release.wait(timeout=5.0)
            return self.results.pop(0)

    implementer_raw = _BlockingBackend([_result(cost=6.0)])
    thread = threading.Thread(
        target=lambda: ledger.wrap("implementer", implementer_raw).invoke(
            _invocation(budget_usd=10.0)
        ),
    )
    thread.start()
    assert entered.wait(timeout=5.0)

    reviewer_raw = _Backend([_result(cost=2.0)])
    reviewer = ledger.wrap("reviewer", reviewer_raw).invoke(
        _invocation(budget_usd=10.0)
    )
    in_flight = ledger.to_summary()
    release.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert reviewer.error is None
    assert implementer_raw.invocations[0].budget_usd == 6.0
    assert reviewer_raw.invocations[0].budget_usd == 2.0
    assert in_flight["group_reserved_usd"]["implementation_verifier"] == 6.0
    assert in_flight["group_spend_usd"]["implementation_verifier"] == 2.0
    assert in_flight["group_remaining_usd"]["implementation_verifier"] == 1.0
    tail = in_flight["implementation_verifier_authority"]["protected_tail"]
    assert tail["outstanding_usd"] == 1.0
    assert tail["physically_available_usd"] == 1.0


def test_task_summary_surfaces_post_hoc_overshoot_after_reclamation():
    ledger = DevelopmentBudgetLedger(_release_policy(calls=2))
    ledger.wrap("planner", _Backend([_result(cost=1.0)])).invoke(
        _invocation(budget_usd=4.0)
    )
    assert ledger.admit_implementation_phase() == (True, None)

    raw = _Backend([_result(cost=10.0)])
    ledger.wrap("implementer", raw).invoke(_invocation(budget_usd=10.0))

    assert raw.invocations[0].budget_usd == 6.0
    summary = ledger.to_summary()
    assert summary["task_authority"]["settled_spend_usd"] == 11.0
    assert summary["task_authority"]["post_hoc_overshoot_usd"] == 1.0
    assert summary["implementation_verifier_authority"][
        "post_hoc_overshoot_usd"
    ] == 1.0


def test_autonomous_development_metrics_are_present_before_any_tier():
    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    ledger.note_validation_repairs(1)
    ledger.note_source_dispatch(dispatched_calls=7, denied_calls=2)

    metrics = ledger.to_summary()["autonomous_development_metrics"]

    assert metrics["time_to_first_mutation_seconds"] is None
    assert metrics["cost_to_first_mutation_usd"] is None
    assert metrics["planning_to_implementation_spend_ratio"] is None
    assert metrics["calls_by_role"] == {
        "planner": 0,
        "implementer": 0,
    }
    assert metrics["validation_repair_count"] == 1
    assert metrics["source_calls_dispatched"] == 7
    assert metrics["source_calls_denied"] == 2
    assert metrics["implementation_reserve_remaining_usd"] == 5.0
    assert metrics["implementation_authority_remaining_usd"] == 6.0


def test_unowned_contingency_is_available_only_to_implementation_group():
    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    implementer_raw = _Backend([_result(cost=5.5)])
    implementer = ledger.wrap("implementer", implementer_raw)

    implementer.invoke(_invocation(budget_usd=6.0))

    assert implementer_raw.invocations[0].budget_usd == 6.0
    summary = ledger.to_summary()
    assert summary["group_cap_usd"] == {
        "planning_review": 4.0,
        "implementation_verifier": 6.0,
    }
    authority = summary["implementation_verifier_authority"]
    assert authority["base_spend_usd"] == 5.0
    assert authority["contingency_spend_usd"] == 0.5
    assert authority["contingency_remaining_usd"] == 0.5
    assert authority["protected_tail"]["allocated_usd"] == 0.0


@pytest.mark.parametrize(
    ("requested_cost_usd", "expected_capacity", "expected_admitted"),
    (
        (None, 5.0, True),
        (1.25, 1.25, True),
        (99.0, 5.0, True),
        (0.0, 0.0, False),
    ),
)
def test_role_dispatch_capacity_exactly_previews_begin_reservation(
    requested_cost_usd,
    expected_capacity,
    expected_admitted,
):
    preview_ledger = DevelopmentBudgetLedger(_tail_policy())
    before_preview = preview_ledger.to_summary()

    capacity = preview_ledger.role_dispatch_capacity(
        "implementer",
        requested_cost_usd=requested_cost_usd,
    )

    assert capacity == pytest.approx(expected_capacity)
    assert preview_ledger.to_summary() == before_preview
    assert before_preview["group_remaining_usd"][
        "implementation_verifier"
    ] == 6.0
    assert before_preview["implementation_verifier_authority"][
        "protected_tail"
    ]["outstanding_usd"] == 1.0
    assert before_preview["roles"]["implementer"]["exhausted_by"] is None

    begin_ledger = DevelopmentBudgetLedger(_tail_policy())
    admitted, reason, reservation = begin_ledger._begin(
        "implementer",
        requested_cost_usd=requested_cost_usd,
    )
    assert admitted is expected_admitted
    assert reservation == pytest.approx(capacity)
    assert reason == (None if expected_admitted else "cost_usd")


def test_role_dispatch_capacity_does_not_latch_exhausted_or_closed_roles():
    exhausted = DevelopmentBudgetLedger(_policy(calls=1))
    exhausted.wrap("planner", _Backend([_result(cost=1.0)])).invoke(
        _invocation(budget_usd=4.0)
    )
    exhausted_before = exhausted.to_summary()
    assert exhausted_before["roles"]["planner"]["exhausted_by"] is None

    assert exhausted.role_dispatch_capacity("planner") == 0.0
    assert exhausted.to_summary() == exhausted_before

    closed = DevelopmentBudgetLedger(_release_policy(calls=2))
    closed.wrap("planner", _Backend([_result(cost=1.0)])).invoke(
        _invocation(budget_usd=4.0)
    )
    assert closed.admit_implementation_phase() == (True, None)
    closed_before = closed.to_summary()
    assert closed_before["roles"]["planner"]["exhausted_by"] is None

    assert closed.role_dispatch_capacity("planner") == 0.0
    assert closed.to_summary() == closed_before


def test_role_dispatch_capacity_is_zero_after_authority_revocation():
    ledger = DevelopmentBudgetLedger(_policy(calls=2))
    ledger.wrap(
        "planner",
        _Backend([_result(cost=1.0, complete=False)]),
    ).invoke(_invocation(budget_usd=4.0))
    before_preview = ledger.to_summary()
    assert before_preview["authority_revoked"] is True
    assert before_preview["roles"]["implementer"]["exhausted_by"] is None

    assert ledger.role_dispatch_capacity("implementer") == 0.0
    assert ledger.to_summary() == before_preview


def test_role_protected_tail_is_configured_after_tail_role_closes():
    ledger = DevelopmentBudgetLedger(_tail_policy())

    assert ledger.role_protected_tail_usd("implementer") == 0.0
    assert ledger.role_protected_tail_usd("diff_scope") == 0.5
    assert ledger.role_protected_tail_usd("tier_completeness") == 0.5
    with pytest.raises(ValueError, match="no allocation"):
        ledger.role_protected_tail_usd("unknown")

    ledger.wrap("diff_scope", _Backend([_result(cost=0.1)])).invoke(
        _invocation(budget_usd=0.5)
    )
    ledger.complete_protected_tail_role("diff_scope")
    before_preview = ledger.to_summary()
    assert before_preview["roles"]["diff_scope"]["exhausted_by"] is None

    assert ledger.role_protected_tail_usd("diff_scope") == 0.5
    assert ledger.role_dispatch_capacity("diff_scope") == 0.0
    assert ledger.to_summary() == before_preview


def test_non_tail_role_cannot_reserve_mandatory_tail_before_completion():
    ledger = DevelopmentBudgetLedger(_tail_policy())
    raw = _Backend([_result(cost=5.0), _result(cost=0.1)])
    implementer = ledger.wrap("implementer", raw)

    implementer.invoke(_invocation(budget_usd=6.0))
    blocked = implementer.invoke(_invocation(budget_usd=1.0))

    assert raw.invocations[0].budget_usd == 5.0
    assert len(raw.invocations) == 1
    assert blocked.error is not None
    assert blocked.error.kind == "budget_exhausted"
    summary = ledger.to_summary()
    assert summary["group_remaining_usd"]["implementation_verifier"] == 1.0
    tail = summary["implementation_verifier_authority"]["protected_tail"]
    assert tail["outstanding_usd"] == 1.0
    assert tail["physically_available_usd"] == 1.0
    assert summary["roles"]["implementer"]["exhausted_by"] == (
        "protected_tail_usd"
    )


def test_tail_owner_completion_releases_only_its_unused_share():
    ledger = DevelopmentBudgetLedger(_tail_policy())
    ledger.wrap(
        "implementer", _Backend([_result(cost=5.0)])
    ).invoke(_invocation(budget_usd=6.0))
    diff_raw = _Backend([_result(cost=0.1)])
    ledger.wrap("diff_scope", diff_raw).invoke(_invocation(budget_usd=1.0))

    before = ledger.to_summary()["implementation_verifier_authority"]
    assert diff_raw.invocations[0].budget_usd == 0.5
    assert before["protected_tail"]["outstanding_usd"] == 0.9
    with pytest.raises(ValueError, match="before its first call"):
        ledger.complete_protected_tail_role("tier_completeness")

    ledger.complete_protected_tail_role("diff_scope")
    ledger.complete_protected_tail_role("diff_scope")
    completed_raw = _Backend([_result(cost=0.1)])
    completed_call = ledger.wrap("diff_scope", completed_raw).invoke(
        _invocation(budget_usd=0.5)
    )
    assert completed_call.error is not None
    assert completed_call.error.kind == "budget_exhausted"
    assert completed_raw.invocations == []
    released = ledger.to_summary()["implementation_verifier_authority"]
    assert released["protected_tail"]["outstanding_usd"] == 0.5
    assert released["protected_tail"]["owners"]["diff_scope"] == {
        "allocated_usd": 0.5,
        "role_spend_usd": 0.1,
        "role_reserved_usd": 0.0,
        "consumed_usd": 0.1,
        "remaining_usd": 0.0,
        "released_unused_usd": 0.4,
        "role_cap_overshoot_usd": 0.0,
        "terminal_completed": True,
    }
    assert ledger.to_summary()["roles"]["diff_scope"]["exhausted_by"] == (
        "tail_role_completed"
    )

    implementer_raw = _Backend([_result(cost=0.4)])
    ledger.wrap("implementer", implementer_raw).invoke(
        _invocation(budget_usd=1.0)
    )
    assert implementer_raw.invocations[0].budget_usd == pytest.approx(0.4)
    assert ledger.to_summary()["implementation_verifier_authority"][
        "protected_tail"
    ]["outstanding_usd"] == 0.5


def test_concurrent_tail_owners_cannot_reserve_sibling_shares():
    entered = {role: threading.Event() for role in (
        "diff_scope", "tier_completeness"
    )}
    release = threading.Event()

    class _BlockingTailBackend(_Backend):
        def __init__(self, role: str) -> None:
            super().__init__([_result(cost=0.1)])
            self.role = role

        def invoke(self, invocation: Invocation) -> InvocationResult:
            self.invocations.append(invocation)
            entered[self.role].set()
            assert release.wait(timeout=5.0)
            return self.results.pop(0)

    ledger = DevelopmentBudgetLedger(_tail_policy())
    ledger.wrap(
        "implementer", _Backend([_result(cost=5.0)])
    ).invoke(_invocation(budget_usd=6.0))
    raw_by_role = {
        role: _BlockingTailBackend(role)
        for role in ("diff_scope", "tier_completeness")
    }
    threads = [
        threading.Thread(
            target=lambda role=role: ledger.wrap(
                role, raw_by_role[role]
            ).invoke(_invocation(budget_usd=1.0))
        )
        for role in raw_by_role
    ]
    for thread in threads:
        thread.start()
    for event in entered.values():
        assert event.wait(timeout=5.0)

    in_flight = ledger.to_summary()
    assert {
        role: raw.invocations[0].budget_usd
        for role, raw in raw_by_role.items()
    } == {"diff_scope": 0.5, "tier_completeness": 0.5}
    assert in_flight["group_reserved_usd"][
        "implementation_verifier"
    ] == 1.0
    assert in_flight["implementation_verifier_authority"][
        "protected_tail"
    ]["outstanding_usd"] == 0.0

    release.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()


def test_non_tail_soft_overshoot_reports_protected_tail_erosion():
    ledger = DevelopmentBudgetLedger(_tail_policy())
    raw = _Backend([_result(cost=5.2)])

    ledger.wrap("implementer", raw).invoke(_invocation(budget_usd=6.0))

    assert raw.invocations[0].budget_usd == 5.0
    authority = ledger.to_summary()["implementation_verifier_authority"]
    assert authority["contingency_spend_usd"] == pytest.approx(0.2)
    assert authority["contingency_remaining_usd"] == pytest.approx(0.8)
    assert authority["protected_tail"]["outstanding_usd"] == 1.0
    assert authority["protected_tail"]["physically_available_usd"] == (
        pytest.approx(0.8)
    )
    assert authority["protected_tail"]["eroded_usd"] == pytest.approx(0.2)
    assert authority["protected_tail"][
        "request_bound_overshoot_usd"
    ] == pytest.approx(0.2)
    assert ledger.to_summary()["roles"]["implementer"]["usage"][
        "request_bound_overshoot_usd"
    ] == pytest.approx(0.2)


def test_first_mutation_metrics_are_write_once():
    ledger = DevelopmentBudgetLedger(_policy())

    ledger.note_first_mutation(elapsed_seconds=3.5, cost_usd=1.25)
    ledger.note_first_mutation(elapsed_seconds=9.0, cost_usd=4.0)

    metrics = ledger.to_summary()["autonomous_development_metrics"]
    assert metrics["time_to_first_mutation_seconds"] == 3.5
    assert metrics["cost_to_first_mutation_usd"] == 1.25
