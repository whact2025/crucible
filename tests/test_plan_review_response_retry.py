# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Plan review-response failures stop before entering semantic debate."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from assay import (
    BudgetExhausted,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    LifecycleEvent,
    Message,
    PlanPolicy,
    ProjectContext,
    ReviewCoverage,
    Spec,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
    Verdict,
    hash_record,
)
from assay.corpus.store import Corpus
from assay.corpus.runs import RunHandle

from crucible.rubrics import load
from crucible.workflows import _review as review_workflow
from crucible.workflows import plan as plan_workflow
from crucible.workflows._review import (
    _finalize_retry_provenance,
    _retry_cache_token_accounting,
)
from crucible.workflows._development_budget import (
    DevelopmentBudgetLedger,
    DevelopmentBudgetPolicy,
    RoleAllocation,
)
from crucible.workflows.plan import run_plan


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=False,
)


_DAG = {
    "tiers": [{
        "tier_id": "t0",
        "depends_on": [],
        "slices": [{
            "id": "s0",
            "directive": {
                "slice_id": "s0",
                "intent": "implement the thing",
                "requirement_refs": ["implement the thing"],
                "files_in_scope": ["src/thing.rs"],
                "creates_paths": [],
                "verify_mode": "tests",
                "blast_radius": {"max_files": 1},
            },
        }],
    }],
}


def _spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="demo",
            repo_url="https://example.invalid/demo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="implement the thing",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _pass_response() -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "The candidate is complete.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [],
    })


def _schema_failure_response(key: str) -> str:
    payload = json.loads(_pass_response())
    payload[key] = "reviewer formatting defect"
    return json.dumps(payload)


def _revise_response() -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "The candidate intent remains ambiguous.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [{
            "id": "ambiguous_intent",
            "criterion_id": "intent_clarity",
            "severity": "High",
            "confidence": 1.0,
            "disposition": "confirmed",
            "category": "ambiguity",
            "targets": [],
            "evidence": [{
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "artifact",
                "excerpt": "implement the thing",
            }],
            "proposal": {
                "action": "make the intended behavior explicit",
                "replacement": None,
            },
            "attack_spec": None,
        }],
    })


def _contract_failure_response(
    *,
    sources_reviewed: list[str] | None = None,
) -> str:
    criterion_results = []
    for index, criterion_id in enumerate(load("plan_critic").required_criteria):
        source_ids = ["artifact"]
        if criterion_id == "intent_coverage":
            source_ids.append("context:spec_intent")
        elif criterion_id == "done_criteria_coverage":
            source_ids.append("context:spec_done_criteria")
        criterion_results.append({
            "criterion_id": criterion_id,
            "outcome": (
                "violated" if criterion_id == "intent_coverage" else "satisfied"
            ),
            "reasoning": "Evidence-backed criterion result.",
            "evidence": [
                {
                    "source_id": source_id,
                    "kind": "source_quote",
                    "location": source_id,
                    "excerpt": (
                        "tiers"
                        if source_id == "artifact"
                        else "implement the thing"
                        if source_id == "context:spec_intent"
                        else "[]"
                    ),
                }
                for source_id in source_ids
            ],
            "headline": index == 0,
        })
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "uses vocabulary outside the loaded rubric",
        "coverage": {
            "complete": True,
            "sources_reviewed": sources_reviewed or ["artifact"],
            "omissions": [],
        },
        "findings": [{
            "id": "outside-rubric-vocabulary",
            "criterion_id": "scope.in_bounds",
            "severity": "High",
            "confidence": 1.0,
            "disposition": "confirmed",
            "category": "scope",
            "targets": [],
            "evidence": [{
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "artifact",
                "excerpt": "tiers",
            }],
            "proposal": {
                "action": "use only declared rubric vocabulary",
                "replacement": None,
            },
            "attack_spec": None,
        }],
        "criterion_results": criterion_results,
    })


class _Backend:
    name = "plan-response-repair-stub"
    model = "plan-response-repair-model"
    capability = "text"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        if not self.responses:
            raise AssertionError("unexpected extra planning call")
        final = self.responses.pop(0)
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            error=None,
            completeness=_COMPLETE_EXECUTION,
        )


class _PricingBackend(_Backend):
    """Return distinct exact pricing evidence for each response attempt."""

    def __init__(
        self,
        *responses: str,
        incomplete_calls: frozenset[int] = frozenset(),
        missing_calls: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__(*responses)
        self.incomplete_calls = incomplete_calls
        self.missing_calls = missing_calls

    def invoke(self, invocation: Invocation) -> InvocationResult:
        result = super().invoke(invocation)
        sequence = len(self.invocations)
        usage = Usage(
            input_tokens=10 * sequence,
            output_tokens=sequence + 1,
            cost_usd=0.01 * sequence,
            cache_creation_tokens=sequence,
            cache_read_tokens=2 * sequence,
        )
        pricing_evidence = (
            None
            if sequence in self.missing_calls
            else UsagePricingEvidence(
                segments=(
                    UsagePricingSegment(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_tokens=usage.cache_creation_tokens,
                        cache_read_tokens=usage.cache_read_tokens,
                        cost_usd=usage.cost_usd,
                    ),
                ),
                complete=sequence not in self.incomplete_calls,
            )
        )
        return replace(
            result,
            usage=usage,
            pricing_evidence=pricing_evidence,
        )


class _KimiBackend(_Backend):
    name = "kimi"
    model = "kimi-k3.1-pro"


class _BudgetRefusalBackend:
    name = "plan-budget-refusal-stub"
    model = "plan-budget-refusal-model"
    capability = "text"

    def __init__(self) -> None:
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        return InvocationResult(
            final_message="",
            transcript=(),
            usage=Usage(),
            stop_reason="budget_exhausted",
            error=BudgetExhausted(
                spent_usd=0.0,
                limit_usd=invocation.budget_usd or 0.0,
                message="reservation refused before dispatch",
            ),
            completeness=_COMPLETE_EXECUTION,
            pricing_evidence=UsagePricingEvidence(segments=(), complete=True),
        )


class _SecondCallBudgetRefusalBackend(_Backend):
    """Return one malformed response, then refuse the bounded replacement."""

    def invoke(self, invocation: Invocation) -> InvocationResult:
        if not self.invocations:
            return super().invoke(invocation)
        self.invocations.append(invocation)
        return InvocationResult(
            final_message="",
            transcript=(),
            usage=Usage(),
            stop_reason="budget_exhausted",
            error=BudgetExhausted(
                spent_usd=0.0,
                limit_usd=invocation.budget_usd or 0.0,
                message="replacement reservation refused before dispatch",
            ),
            completeness=_COMPLETE_EXECUTION,
            pricing_evidence=UsagePricingEvidence(segments=(), complete=True),
        )


class _TransportFailureBackend:
    name = "plan-transport-failure-stub"
    model = "plan-transport-failure-model"
    capability = "text"

    def __init__(self) -> None:
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        return InvocationResult(
            final_message="",
            transcript=(),
            usage=Usage(input_tokens=1, cost_usd=0.01),
            stop_reason="error",
            error=TransportError(
                http_status=500,
                message="provider returned HTTP 500",
            ),
        )


def _run(
    tmp_path,
    *,
    planner,
    critic,
    judge,
    spec: Spec | None = None,
    budget_usd: float | None = None,
):
    corpus = Corpus(tmp_path / "corpus")
    selected_spec = spec or _spec()
    corpus.put_spec(selected_spec)
    rubric = load("spec_clarity")
    outcome = run_plan(
        selected_spec,
        planner_backend=planner,
        critic_backend=critic,
        judge_backend=judge,
        corpus=corpus,
        critic_rubric=rubric,
        judge_rubric=rubric,
        budget_usd=budget_usd,
    )
    return corpus, outcome


def _planning_budget_ledger(*, critic_calls: int) -> DevelopmentBudgetLedger:
    return DevelopmentBudgetLedger(DevelopmentBudgetPolicy(
        task_budget_usd=10.0,
        planning_review_cap_usd=4.0,
        implementation_verifier_reserve_usd=5.0,
        contingency_usd=1.0,
        roles=tuple(
            RoleAllocation(
                role=role,
                group="planning_review",
                max_calls=max_calls,
                max_output_tokens=100_000,
                max_wall_seconds=1_800.0,
                max_cost_usd=4.0,
            )
            for role, max_calls in (
                ("architect_planner", 2),
                ("architect_critic", critic_calls),
                ("architect_judge", 2),
            )
        ),
    ))


def test_plan_accepts_repaired_critic_without_poisoning_history(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _Backend(
        _schema_failure_response("unexpected"),
        _pass_response(),
    )
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
    )

    assert outcome.accepted is True
    assert outcome.cost_usd == pytest.approx(0.04)
    assert len(planner.invocations) == 1
    assert len(critic.invocations) == 2
    assert len(judge.invocations) == 1
    verdicts = [
        corpus.get_record(ref, as_type=Verdict)
        for ref in outcome.spec.verdict_refs
    ]
    assert [verdict.decision for verdict in verdicts] == ["pass", "pass"]
    assert "review_retry_manifest" in verdicts[0].artifacts
    assert all(
        finding.id not in {"schema_failure", "rubric_contract_failure"}
        for verdict in verdicts
        for finding in verdict.findings
    )


def test_repaired_review_aggregates_ordered_physical_pricing(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _PricingBackend(
        _schema_failure_response("unexpected"),
        _pass_response(),
    )
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
    )

    assert outcome.accepted is True
    assert outcome.cost_usd == pytest.approx(0.05)
    critic_verdict = corpus.get_record(
        outcome.spec.verdict_refs[0],
        as_type=Verdict,
    )
    assert critic_verdict.input_tokens == 30
    assert critic_verdict.output_tokens == 5
    assert critic_verdict.cache_creation_tokens == 3
    assert critic_verdict.cache_read_tokens == 6
    assert critic_verdict.cost_usd == pytest.approx(0.03)
    assert critic_verdict.cache_creation_tokens == 3
    assert critic_verdict.cache_read_tokens == 6
    assert critic_verdict.accounting_debt == ()
    assert critic_verdict.pricing_evidence is not None
    assert critic_verdict.pricing_evidence.complete is True
    assert [
        segment.input_tokens
        for segment in critic_verdict.pricing_evidence.segments
    ] == [10, 20]
    assert critic_verdict.pricing_evidence.reconciles(
        Usage(
            input_tokens=30,
            output_tokens=5,
            cost_usd=0.03,
            cache_creation_tokens=3,
            cache_read_tokens=6,
        )
    )
    for artifact_name in (
        "review_call_telemetry",
        "review_retry_manifest",
    ):
        accounting = json.loads(corpus.get_artifact(
            critic_verdict.artifacts[artifact_name]
        ))
        assert accounting["total_cache_creation_tokens"] == 3
        assert accounting["total_cache_read_tokens"] == 6
        assert [
            attempt["cache_creation_tokens"]
            for attempt in accounting["attempts"]
        ] == [1, 2]
        assert [
            attempt["cache_read_tokens"]
            for attempt in accounting["attempts"]
        ] == [2, 4]


def test_retry_cache_accounting_keeps_exact_zero_and_unknown_null():
    paid = SimpleNamespace(
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.1,
        cache_creation_tokens=3,
        cache_read_tokens=4,
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=None,
    )
    exact_zero = SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        execution_completeness=None,
        pricing_evidence=None,
    )
    legacy_invoked_unknown = SimpleNamespace(
        input_tokens=5,
        output_tokens=1,
        cost_usd=0.05,
        cache_creation_tokens=None,
        cache_read_tokens=None,
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=None,
    )
    legacy_priced = SimpleNamespace(
        input_tokens=7,
        output_tokens=1,
        cost_usd=0.07,
        cache_creation_tokens=None,
        cache_read_tokens=None,
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=UsagePricingEvidence(
            segments=(UsagePricingSegment(
                input_tokens=7,
                output_tokens=1,
                cache_creation_tokens=2,
                cache_read_tokens=5,
                cost_usd=0.07,
            ),),
            complete=True,
        ),
    )

    zero_usage_legacy = SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        cache_creation_tokens=None,
        cache_read_tokens=None,
        execution_completeness=None,
        pricing_evidence=None,
    )

    per_attempt, aggregate = _retry_cache_token_accounting((paid, exact_zero))
    assert per_attempt == ((3, 4), (0, 0))
    assert aggregate == (3, 4)

    per_attempt, aggregate = _retry_cache_token_accounting(
        (paid, legacy_invoked_unknown)
    )
    assert per_attempt == ((3, 4), None)
    assert aggregate is None

    per_attempt, aggregate = _retry_cache_token_accounting(
        (paid, zero_usage_legacy)
    )
    assert per_attempt == ((3, 4), None)
    assert aggregate is None

    per_attempt, aggregate = _retry_cache_token_accounting(
        (paid, legacy_priced)
    )
    assert per_attempt == ((3, 4), (2, 5))
    assert aggregate == (5, 9)


def test_zero_usage_legacy_retry_stays_incomplete_in_all_accounting(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    artifact_ref = corpus.put_artifact(
        b"candidate",
        kind="review_inputs",
        name_hint="candidate.txt",
    )
    coverage = ReviewCoverage(
        complete=True,
        sources_reviewed=("artifact",),
        omissions=(),
        artifact_chars_total=9,
        artifact_chars_reviewed=9,
    )
    paid_pricing = UsagePricingEvidence(
        segments=(UsagePricingSegment(
            input_tokens=10,
            output_tokens=2,
            cache_creation_tokens=3,
            cache_read_tokens=4,
            cost_usd=0.1,
        ),),
        complete=True,
    )
    paid = Verdict(
        rubric="spec_clarity",
        decision="pass",
        findings=(),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="paid attempt",
        coverage=coverage,
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.1,
        execution=None,
        input_tokens=10,
        output_tokens=2,
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=paid_pricing,
        cache_creation_tokens=3,
        cache_read_tokens=4,
    )
    zero_usage_legacy = replace(
        paid,
        reasoning="ambiguous legacy attempt",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        execution_completeness=None,
        pricing_evidence=None,
        cache_creation_tokens=None,
        cache_read_tokens=None,
    )
    attempts = [
        {
            "attempt": 0,
            "verdict": paid,
            "validation_error": "first response invalid",
            "source_payload_bytes": 9,
            "source_payload_count": 1,
        },
        {
            "attempt": 1,
            "verdict": zero_usage_legacy,
            "validation_error": None,
            "source_payload_bytes": 9,
            "source_payload_count": 1,
        },
    ]

    aggregate = _finalize_retry_provenance(
        corpus=corpus,
        verdict=zero_usage_legacy,
        attempts=attempts,
        terminal_reason="repaired",
    )

    assert aggregate.cache_creation_tokens is None
    assert aggregate.cache_read_tokens is None
    assert aggregate.execution_completeness is not None
    assert aggregate.execution_completeness.token_usage_complete is False
    assert "token_usage_complete" in aggregate.accounting_debt
    assert aggregate.pricing_evidence is not None
    assert aggregate.pricing_evidence.complete is False
    for artifact_name in (
        "review_call_telemetry",
        "review_retry_manifest",
    ):
        accounting = json.loads(corpus.get_artifact(
            aggregate.artifacts[artifact_name]
        ))
        assert accounting["total_cache_creation_tokens"] is None
        assert accounting["total_cache_read_tokens"] is None
        assert [
            attempt["cache_creation_tokens"]
            for attempt in accounting["attempts"]
        ] == [3, None]
        assert [
            attempt["cache_read_tokens"]
            for attempt in accounting["attempts"]
        ] == [4, None]
        assert accounting["execution_completeness"][
            "token_usage_complete"
        ] is False
        assert "token_usage_complete" in accounting["accounting_debt"]


def test_retry_provenance_rejects_empty_attempts():
    with pytest.raises(
        ValueError,
        match="requires at least one completed attempt",
    ):
        review_workflow._finalize_retry_provenance(
            corpus=None,
            verdict=None,
            attempts=[],
            terminal_reason="unreachable_empty_attempt_probe",
        )


def test_first_attempt_review_preserves_exact_pricing_evidence(tmp_path):
    critic = _PricingBackend(_pass_response())
    corpus, outcome = _run(
        tmp_path,
        planner=_Backend(json.dumps(_DAG)),
        critic=critic,
        judge=_Backend(_pass_response()),
    )

    assert outcome.accepted is True
    critic_verdict = corpus.get_record(
        outcome.spec.verdict_refs[0],
        as_type=Verdict,
    )
    assert critic_verdict.input_tokens == 10
    assert critic_verdict.output_tokens == 2
    assert critic_verdict.cost_usd == pytest.approx(0.01)
    assert critic_verdict.pricing_evidence is not None
    assert critic_verdict.pricing_evidence.complete is True
    assert len(critic_verdict.pricing_evidence.segments) == 1
    assert critic_verdict.pricing_evidence.segments[0].cache_read_tokens == 2


@pytest.mark.parametrize("gap", ["incomplete", "missing"])
def test_repaired_review_does_not_overstate_partial_pricing(
    tmp_path,
    gap: str,
):
    backend_kwargs = (
        {"incomplete_calls": frozenset({1})}
        if gap == "incomplete"
        else {"missing_calls": frozenset({1})}
    )
    critic = _PricingBackend(
        _schema_failure_response("unexpected"),
        _pass_response(),
        **backend_kwargs,
    )
    corpus, outcome = _run(
        tmp_path,
        planner=_Backend(json.dumps(_DAG)),
        critic=critic,
        judge=_Backend(_pass_response()),
    )

    assert outcome.accepted is True
    critic_verdict = corpus.get_record(
        outcome.spec.verdict_refs[0],
        as_type=Verdict,
    )
    assert critic_verdict.pricing_evidence is not None
    assert critic_verdict.pricing_evidence.complete is False
    assert critic_verdict.cache_creation_tokens == 3
    assert critic_verdict.cache_read_tokens == 6
    assert critic_verdict.execution_completeness is not None
    assert critic_verdict.execution_completeness.token_usage_complete is True
    assert critic_verdict.accounting_debt == ()
    expected_inputs = [10, 20] if gap == "incomplete" else [20]
    assert [
        segment.input_tokens
        for segment in critic_verdict.pricing_evidence.segments
    ] == expected_inputs


def test_repaired_review_materializes_legacy_cache_totals_from_complete_pricing(
    tmp_path,
    monkeypatch,
):
    critic = _PricingBackend(
        _schema_failure_response("unexpected"),
        _pass_response(),
    )
    assay_review = review_workflow._assay_review

    def _legacy_assay_review(*args, **kwargs):
        verdict = assay_review(*args, **kwargs)
        if kwargs["backend"] is critic:
            return replace(
                verdict,
                cache_creation_tokens=None,
                cache_read_tokens=None,
            )
        return verdict

    monkeypatch.setattr(review_workflow, "_assay_review", _legacy_assay_review)
    corpus, outcome = _run(
        tmp_path,
        planner=_Backend(json.dumps(_DAG)),
        critic=critic,
        judge=_Backend(_pass_response()),
    )

    assert outcome.accepted is True
    critic_verdict = corpus.get_record(
        outcome.spec.verdict_refs[0],
        as_type=Verdict,
    )
    assert critic_verdict.cache_creation_tokens == 3
    assert critic_verdict.cache_read_tokens == 6
    assert critic_verdict.execution_completeness is not None
    assert critic_verdict.execution_completeness.token_usage_complete is True
    assert critic_verdict.pricing_evidence is not None
    assert critic_verdict.pricing_evidence.complete is True
    assert critic_verdict.accounting_debt == ()


@pytest.mark.parametrize("gap", ["incomplete", "missing"])
def test_repaired_review_fails_closed_on_unknown_legacy_cache_totals(
    tmp_path,
    monkeypatch,
    gap: str,
):
    backend_kwargs = (
        {"incomplete_calls": frozenset({2})}
        if gap == "incomplete"
        else {"missing_calls": frozenset({2})}
    )
    critic = _PricingBackend(
        _schema_failure_response("unexpected"),
        _pass_response(),
        **backend_kwargs,
    )
    assay_review = review_workflow._assay_review

    def _legacy_second_assay_review(*args, **kwargs):
        verdict = assay_review(*args, **kwargs)
        if kwargs["backend"] is critic and verdict.input_tokens == 20:
            return replace(
                verdict,
                cache_creation_tokens=None,
                cache_read_tokens=None,
            )
        return verdict

    monkeypatch.setattr(
        review_workflow,
        "_assay_review",
        _legacy_second_assay_review,
    )
    corpus, outcome = _run(
        tmp_path,
        planner=_Backend(json.dumps(_DAG)),
        critic=critic,
        judge=_Backend(_pass_response()),
    )

    assert outcome.status == "incomplete_accounting"
    assert outcome.execution_completeness is not None
    assert outcome.execution_completeness.token_usage_complete is False
    assert outcome.pricing_evidence is not None
    assert outcome.pricing_evidence.complete is False
    critic_verdict = corpus.get_record(
        outcome.spec.verdict_refs[0],
        as_type=Verdict,
    )
    assert critic_verdict.cache_creation_tokens is None
    assert critic_verdict.cache_read_tokens is None
    assert critic_verdict.execution_completeness is not None
    assert critic_verdict.execution_completeness.token_usage_complete is False
    assert critic_verdict.execution_completeness.cost_complete is True
    assert critic_verdict.pricing_evidence is not None
    assert critic_verdict.pricing_evidence.complete is False
    assert critic_verdict.accounting_debt == ("token_usage_complete",)


def test_kimi_plan_review_uses_two_anchor_only_response_repairs(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _KimiBackend(
        _schema_failure_response("bad_zero"),
        _schema_failure_response("bad_one"),
        _pass_response(),
    )
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
    )

    assert outcome.accepted is True
    assert len(critic.invocations) == 3
    assert len(judge.invocations) == 1
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in invocation.system
        for invocation in critic.invocations
    )
    critic_verdict = corpus.get_record(
        outcome.spec.verdict_refs[0],
        as_type=Verdict,
    )
    manifest = json.loads(corpus.get_artifact(
        critic_verdict.artifacts["review_retry_manifest"]
    ))
    assert [attempt["attempt"] for attempt in manifest["attempts"]] == [
        0,
        1,
        2,
    ]
    assert manifest["terminal_reason"] == "repaired"
    assert all(
        "Do not emit a location that begins with ``json:``"
        in " ".join(invocation.system.split())
        for invocation in critic.invocations
    )


def test_kimi_plan_review_accepts_valid_anchor_only_first_attempt(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _KimiBackend(_pass_response())
    judge = _KimiBackend(_pass_response())

    _, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
    )

    assert outcome.accepted is True
    assert len(critic.invocations) == 1
    assert len(judge.invocations) == 1
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in invocation.system
        and "Do not emit a location that begins with ``json:``"
        in " ".join(invocation.system.split())
        for invocation in (*critic.invocations, *judge.invocations)
    )


def test_later_critic_uses_second_repair_without_poisoning_candidate_history(
    tmp_path,
):
    """A second-round protocol repair stays inside the same review slot."""

    spec = replace(
        _spec(),
        plan_policy=PlanPolicy(
            max_critique_rounds=1,
            max_critic_failures=1,
            max_judge_failures=1,
        ),
    )
    planner = _PricingBackend(json.dumps(_DAG), json.dumps(_DAG))
    critic = _PricingBackend(
        _revise_response(),
        _schema_failure_response("decision_finding_inconsistent"),
        _schema_failure_response("missing_required_evidence"),
        _pass_response(),
    )
    judge = _PricingBackend(_pass_response(), _pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        spec=spec,
    )

    assert outcome.accepted is True
    assert len(planner.invocations) == 2
    assert len(critic.invocations) == 4
    assert len(judge.invocations) == 2
    assert outcome.cost_usd == pytest.approx(0.16)
    assert outcome.execution_completeness is not None
    assert outcome.execution_completeness.token_usage_complete is True
    assert outcome.execution_completeness.cost_complete is True
    assert outcome.pricing_evidence is not None
    assert outcome.pricing_evidence.complete is True
    assert outcome.pricing_evidence.reconciles(outcome.usage)

    verdicts = [
        corpus.get_record(ref, as_type=Verdict)
        for ref in outcome.spec.verdict_refs
    ]
    assert [verdict.decision for verdict in verdicts] == [
        "revise",
        "pass",
        "pass",
        "pass",
    ]
    repaired_critic = verdicts[2]
    manifest = json.loads(corpus.get_artifact(
        repaired_critic.artifacts["review_retry_manifest"]
    ))
    assert [attempt["attempt"] for attempt in manifest["attempts"]] == [
        0,
        1,
        2,
    ]
    assert manifest["terminal_reason"] == "repaired"
    assert all(
        finding.id not in {"schema_failure", "rubric_contract_failure"}
        for verdict in verdicts
        for finding in verdict.findings
    )


def test_exhausted_critic_protocol_repair_stops_before_judge_or_replan(
    tmp_path,
):
    spec = replace(
        _spec(),
        plan_policy=PlanPolicy(
            max_critique_rounds=2,
            max_critic_failures=3,
            max_judge_failures=3,
        ),
    )
    planner = _Backend(json.dumps(_DAG), json.dumps(_DAG))
    critic = _Backend(
        _schema_failure_response("bad_zero"),
        _schema_failure_response("bad_one"),
        _schema_failure_response("bad_two"),
        _pass_response(),
    )
    judge = _Backend(_pass_response(), _pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        spec=spec,
    )

    assert outcome.decision == "inconclusive"
    assert outcome.status == "critic_protocol_failure"
    assert outcome.rounds_used == 1
    assert outcome.cost_usd == pytest.approx(0.04)
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag is not None
    assert outcome.candidate_ref is not None
    assert len(planner.invocations) == 1
    assert len(critic.invocations) == 3
    assert len(judge.invocations) == 0
    verdict = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    assert [finding.id for finding in verdict.findings] == ["schema_failure"]


def test_later_critic_protocol_failure_does_not_report_prior_round_judge(
    tmp_path,
):
    spec = replace(
        _spec(),
        plan_policy=PlanPolicy(
            max_critique_rounds=2,
            max_critic_failures=3,
            max_judge_failures=3,
        ),
    )
    planner = _Backend(json.dumps(_DAG), json.dumps(_DAG))
    critic = _Backend(
        _revise_response(),
        _schema_failure_response("bad_zero"),
        _schema_failure_response("bad_one"),
        _schema_failure_response("bad_two"),
    )
    judge = _Backend(_pass_response())

    _, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        spec=spec,
    )

    assert outcome.status == "critic_protocol_failure"
    assert outcome.rounds_used == 2
    assert outcome.final_critic_decision == "inconclusive"
    assert outcome.final_judge_decision is None
    assert outcome.cost_usd == pytest.approx(0.07)
    assert len(planner.invocations) == 2
    assert len(critic.invocations) == 4
    assert len(judge.invocations) == 1


def test_intercepted_review_terminal_retains_ordered_aggregate_pricing(
    tmp_path,
):
    """A typed control terminal must not become false accounting debt."""

    spec = replace(
        _spec(),
        plan_policy=PlanPolicy(
            max_critique_rounds=2,
            max_critic_failures=3,
            max_judge_failures=3,
        ),
    )
    planner = _PricingBackend(json.dumps(_DAG), json.dumps(_DAG))
    critic = _PricingBackend(
        _revise_response(),
        _schema_failure_response("bad_zero"),
        _schema_failure_response("bad_one"),
        _schema_failure_response("bad_two"),
    )
    judge = _PricingBackend(_pass_response())

    _, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        spec=spec,
    )

    assert outcome.status == "critic_protocol_failure"
    assert outcome.cost_usd == pytest.approx(0.14)
    assert outcome.execution_completeness is not None
    assert outcome.execution_completeness.token_usage_complete is True
    assert outcome.execution_completeness.cost_complete is True
    assert outcome.pricing_evidence is not None
    assert outcome.pricing_evidence.complete is True
    assert outcome.pricing_evidence.reconciles(outcome.usage)
    assert [
        segment.cost_usd for segment in outcome.pricing_evidence.segments
    ] == pytest.approx([0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04])


def test_exhausted_judge_protocol_repair_stops_before_replan(tmp_path):
    spec = replace(
        _spec(),
        plan_policy=PlanPolicy(
            max_critique_rounds=2,
            max_critic_failures=3,
            max_judge_failures=3,
        ),
    )
    planner = _Backend(json.dumps(_DAG), json.dumps(_DAG))
    critic = _Backend(_pass_response(), _pass_response())
    judge = _Backend(
        _schema_failure_response("bad_zero"),
        _schema_failure_response("bad_one"),
        _schema_failure_response("bad_two"),
        _pass_response(),
    )

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        spec=spec,
    )

    assert outcome.status == "judge_protocol_failure"
    assert "response-protocol" in outcome.reason
    assert "infrastructure" not in outcome.reason
    assert outcome.final_critic_decision == "pass"
    assert outcome.final_judge_decision == "inconclusive"
    assert outcome.cost_usd == pytest.approx(0.05)
    assert outcome.candidate_dag is not None
    assert len(planner.invocations) == 1
    assert len(critic.invocations) == 1
    assert len(judge.invocations) == 3
    verdicts = [
        corpus.get_record(ref, as_type=Verdict)
        for ref in outcome.spec.verdict_refs
    ]
    assert [verdict.decision for verdict in verdicts] == [
        "pass",
        "inconclusive",
    ]


def test_judge_backend_failure_with_incomplete_usage_revokes_authority(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _Backend(_pass_response())
    judge = _TransportFailureBackend()

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
    )

    assert outcome.status == "incomplete_accounting"
    assert "judge returned incomplete aggregate accounting" in outcome.reason
    assert outcome.final_critic_decision == "pass"
    assert outcome.final_judge_decision == "inconclusive"
    assert outcome.cost_usd == pytest.approx(0.03)
    assert len(judge.invocations) == 1
    verdict = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    assert [finding.id for finding in verdict.findings] == [
        "backend_transport",
    ]


@pytest.mark.parametrize("failure_mode", ["protocol", "accounting"])
def test_fixed_candidate_review_failure_retains_read_only_reference_identity(
    tmp_path,
    failure_mode,
):
    source = tmp_path / "tests" / "reference.py"
    source.parent.mkdir()
    source.write_bytes(b"EXISTING_BEHAVIOR = 1\n")
    reference = {
        "reference_id": "existing-behavior",
        "path": "tests/reference.py",
        "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_line": 1,
        "end_line": 1,
    }
    payload = json.loads(json.dumps(_DAG))
    directive = payload["tiers"][0]["slices"][0]["directive"]
    directive["blast_radius"]["languages"] = ["rust"]
    directive["read_only_references"] = [reference]
    candidate = plan_workflow._strict_review_candidate_dag(payload)
    critic = (
        _Backend(*(_schema_failure_response("bad") for _ in range(3)))
        if failure_mode == "protocol" else _Backend(_pass_response())
    )
    judge = (
        _Backend() if failure_mode == "protocol" else _TransportFailureBackend()
    )
    corpus = Corpus(tmp_path / "corpus")
    spec = _spec()
    corpus.put_spec(spec)
    rubric = load("spec_clarity")

    outcome = run_plan(
        spec,
        candidate_dag=candidate,
        critic_backend=critic,
        judge_backend=judge,
        corpus=corpus,
        critic_rubric=rubric,
        judge_rubric=rubric,
        cwd=str(tmp_path),
    )

    assert outcome.status == (
        "critic_protocol_failure" if failure_mode == "protocol"
        else "incomplete_accounting"
    )
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_dag == candidate
    assert hash_record(outcome.candidate_dag) == hash_record(candidate)
    assert outcome.candidate_ref is not None
    retained = json.loads(corpus.get_artifact(outcome.candidate_ref))
    assert retained["tiers"][0]["slices"][0]["directive"][
        "read_only_references"
    ] == [reference]
    retained_directive = outcome.candidate_dag.tiers[0].slices[0].directive
    assert retained_directive.files_in_scope == ("src/thing.rs",)
    assert retained_directive.creates_paths == ()
    assert source.read_bytes() == b"EXISTING_BEHAVIOR = 1\n"
    assert len(critic.invocations) == (3 if failure_mode == "protocol" else 1)
    assert len(judge.invocations) == (0 if failure_mode == "protocol" else 1)


def test_review_repair_budget_terminal_takes_precedence(tmp_path):
    planner = _Backend(json.dumps(_DAG), json.dumps(_DAG))
    critic = _Backend(
        _schema_failure_response("budget_limited_bad_response"),
        _pass_response(),
    )
    judge = _Backend(_pass_response())

    _, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        budget_usd=0.02,
    )

    assert outcome.status == "budget_exhausted"
    assert "during bounded response repair" in outcome.reason
    assert outcome.cost_usd == pytest.approx(0.02)
    assert len(critic.invocations) == 1
    assert len(judge.invocations) == 0


def test_deterministic_terminal_budget_manifest_maps_to_repair_stage(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    manifest_ref = corpus.put_artifact(
        json.dumps(
            {
                "schema_version": "crucible.review-response-retry.v1",
                "attempts": [{"attempt": 0}, {"attempt": 1}],
                "terminal_reason": (
                    "budget_exhausted_after_deterministic_terminal"
                ),
            }
        ).encode(),
        kind="review_manifests",
        name_hint="deterministic-terminal-budget.json",
    )
    verdict = SimpleNamespace(
        artifacts={"review_retry_manifest": manifest_ref},
        findings=(),
    )

    assert (
        plan_workflow._review_budget_terminal_stage(verdict, corpus)
        == "response_repair"
    )


def test_invalid_first_repair_at_plan_cap_stops_before_second_repair(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _Backend(
        _schema_failure_response("bad_zero"),
        _schema_failure_response("bad_one"),
    )
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        budget_usd=0.03,
    )

    assert outcome.status == "budget_exhausted"
    assert outcome.cost_usd == pytest.approx(0.03)
    assert len(critic.invocations) == 2
    assert len(judge.invocations) == 0
    verdict = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert manifest["terminal_reason"] == "budget_exhausted_before_retry"


def test_successful_repair_over_budget_is_terminal_and_scrubbed(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _Backend(
        _schema_failure_response("first_response_invalid"),
        _pass_response(),
    )
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        budget_usd=0.025,
    )

    assert outcome.status == "budget_exhausted"
    assert outcome.cost_usd == pytest.approx(0.03)
    assert len(critic.invocations) == 2
    assert len(judge.invocations) == 0
    verdict = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    assert [finding.id for finding in verdict.findings] == [
        "review_response_retry_budget_exhausted",
    ]


def test_reviewer_backend_budget_refusal_maps_to_budget_status(tmp_path):
    planner = _PricingBackend(json.dumps(_DAG))
    critic = _BudgetRefusalBackend()
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        budget_usd=1.0,
    )

    assert outcome.status == "budget_exhausted"
    assert "during review execution" in outcome.reason
    assert "during bounded response repair" not in outcome.reason
    assert outcome.cost_usd == pytest.approx(0.01)
    assert outcome.execution_completeness is not None
    assert outcome.execution_completeness.token_usage_complete is True
    assert outcome.execution_completeness.cost_complete is True
    assert outcome.pricing_evidence is not None
    assert outcome.pricing_evidence.complete is True
    assert outcome.pricing_evidence.reconciles(outcome.usage)
    assert len(outcome.pricing_evidence.segments) == 1
    assert len(critic.invocations) == 1
    assert len(judge.invocations) == 0
    verdict = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    assert [finding.id for finding in verdict.findings] == [
        "backend_budget_exhausted",
    ]


def test_reviewer_repair_budget_refusal_reports_response_repair(tmp_path):
    planner = _Backend(json.dumps(_DAG))
    critic = _SecondCallBudgetRefusalBackend(
        _schema_failure_response("first_response_invalid"),
    )
    judge = _Backend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=planner,
        critic=critic,
        judge=judge,
        budget_usd=1.0,
    )

    assert outcome.status == "budget_exhausted"
    assert "during bounded response repair" in outcome.reason
    assert "during review execution" not in outcome.reason
    assert len(critic.invocations) == 2
    assert len(judge.invocations) == 0
    verdict = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert manifest["terminal_reason"] == (
        "budget_exhausted_after_deterministic_terminal"
    )
    assert [attempt["attempt"] for attempt in manifest["attempts"]] == [0, 1]
    assert verdict.cache_creation_tokens == 0
    assert verdict.cache_read_tokens == 0
    for artifact_name in (
        "review_call_telemetry",
        "review_retry_manifest",
    ):
        accounting = json.loads(corpus.get_artifact(
            verdict.artifacts[artifact_name]
        ))
        assert accounting["total_cache_creation_tokens"] == 0
        assert accounting["total_cache_read_tokens"] == 0
        assert [
            attempt["cache_creation_tokens"]
            for attempt in accounting["attempts"]
        ] == [0, 0]
        assert [
            attempt["cache_read_tokens"]
            for attempt in accounting["attempts"]
        ] == [0, 0]
    assert [finding.id for finding in verdict.findings] == [
        "backend_budget_exhausted",
    ]


@pytest.mark.parametrize("denial_stage", ["initial_review", "response_repair"])
def test_real_development_ledger_denial_retains_complete_plan_accounting(
    tmp_path,
    denial_stage: str,
):
    """Reproduce R11's exact typed refusal through the real role ledger."""

    spec = replace(
        _spec(),
        plan_policy=PlanPolicy(
            max_critique_rounds=1,
            max_critic_failures=2,
            max_judge_failures=2,
        ),
    )
    ledger = _planning_budget_ledger(critic_calls=1)
    if denial_stage == "initial_review":
        planner_raw = _PricingBackend(json.dumps(_DAG), json.dumps(_DAG))
        critic_raw = _PricingBackend(_revise_response())
        judge_raw = _PricingBackend(_pass_response())
    else:
        planner_raw = _PricingBackend(json.dumps(_DAG))
        critic_raw = _PricingBackend(
            _schema_failure_response("first_response_invalid")
        )
        judge_raw = _PricingBackend(_pass_response())

    corpus, outcome = _run(
        tmp_path,
        planner=ledger.wrap("architect_planner", planner_raw),
        critic=ledger.wrap("architect_critic", critic_raw),
        judge=ledger.wrap("architect_judge", judge_raw),
        spec=spec,
        budget_usd=4.0,
    )

    assert outcome.status == "budget_exhausted"
    assert outcome.decision == "inconclusive"
    assert outcome.execution_completeness is not None
    assert outcome.execution_completeness.token_usage_complete is True
    assert outcome.execution_completeness.cost_complete is True
    assert outcome.pricing_evidence is not None
    assert outcome.pricing_evidence.complete is True
    assert outcome.pricing_evidence.reconciles(outcome.usage)
    expected_reason = (
        "during review execution"
        if denial_stage == "initial_review"
        else "during bounded response repair"
    )
    assert expected_reason in outcome.reason

    summary = ledger.to_summary()
    assert summary["roles"]["architect_critic"]["usage"]["calls"] == 1
    assert summary["roles"]["architect_critic"]["exhausted_by"] == (
        "call_count"
    )
    assert summary["roles"]["architect_critic"]["accounting_debt"] is False
    assert summary["task_authority"]["post_hoc_overshoot_usd"] == 0.0
    assert len(critic_raw.invocations) == 1
    assert len(outcome.pricing_evidence.segments) == (
        4 if denial_stage == "initial_review" else 2
    )
    verdict = corpus.get_record(outcome.spec.verdict_refs[-1], as_type=Verdict)
    assert verdict.accounting_debt == ()


def test_plan_cli_contract_exhaustion_persists_inconclusive_provenance(
    tmp_path,
    monkeypatch,
):
    spec_dir = tmp_path / "corpus"
    corpus = Corpus(spec_dir)
    corpus.put_spec(_spec())
    planner = _Backend(json.dumps(_DAG), json.dumps(_DAG))
    invalid = _contract_failure_response(sources_reviewed=[
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
    ])
    critic = _Backend(invalid, invalid, invalid, _pass_response())
    judge = _Backend(_pass_response(), _pass_response())

    class _Pool:
        def get(self, role, **_kwargs):
            return {
                "plan_planner": planner,
                "plan_critic": critic,
                "plan_judge": judge,
            }[role]

    @contextmanager
    def _pool(*_args, **_kwargs):
        yield _Pool()

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    monkeypatch.setattr(
        RunHandle,
        "elapsed_s",
        property(lambda _handle: 73.5),
    )

    rc = plan_workflow.main([
        "--planner-count", "1",
        "--spec",
        str(spec_dir),
        "--no-validate-gate",
    ])

    assert rc == 3
    assert len(planner.invocations) == 1
    assert len(critic.invocations) == 3
    assert len(judge.invocations) == 0
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    typed_outcome = json.loads((run_dir / "outcome.json").read_text())
    assert summary["status"] == "critic_protocol_failure"
    assert summary["decision"] == "inconclusive"
    assert summary["candidate_ref"].startswith("sha256:")
    assert summary["verdict_refs_count"] == 1
    assert result["status"] == "critic_protocol_failure"
    assert summary["duration_s"] == result["duration_s"] == 73.5
    assert result["cost_usd"] == pytest.approx(0.04)
    assert typed_outcome["terminal_class"] == "inconclusive"

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.tier_dag is None
    assert len(latest.verdict_refs) == 1
    verdict = corpus.get_record(latest.verdict_refs[0], as_type=Verdict)
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert len(latest.lifecycle_event_refs) == 1
    event = corpus.get_record(
        latest.lifecycle_event_refs[0],
        as_type=LifecycleEvent,
    )
    assert event.kind == "plan_inconclusive"
    assert event.affected_versions == (latest.version,)
    assert "response-protocol" in event.reason
    assert "infrastructure" not in event.reason
