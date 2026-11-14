# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Thin compatibility forwarding checks for the programmatic plan entry."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    ImplementationDirective,
    InvocationResult,
    PlanOutcome,
    PlanPolicy,
    ProjectContext,
    Rubric,
    Slice,
    Spec,
    Tier,
    TierDAG,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.findings import ReviewCoverage, Verdict

from crucible.constraints import LanguagesConstraint
from crucible.workflows import plan as plan_workflow


def _forwarding_outcome(spec: Spec) -> PlanOutcome:
    """Honor Assay's return contract without introducing source references."""
    candidate = _candidate_with_languages()
    return PlanOutcome(
        spec=spec.with_appended_refs(tier_dag=candidate),
        candidate_dag=candidate,
        candidate_ref=None,
        decision="accepted",
        status="accepted",
        rounds_used=1,
        final_critic_decision="pass",
        final_judge_decision=None,
        reason="Typed fixture for forwarding-only assertions",
    )


def _diagnostic_verdict(
    kind: str,
    decision: str,
    *findings: SimpleNamespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        decision=decision,
        findings=findings,
    )


def _diagnostic_finding(
    *,
    severity: str = "High",
    disposition: str = "confirmed",
    criterion_id: str = "intent_coverage",
) -> SimpleNamespace:
    return SimpleNamespace(
        severity=severity,
        disposition=disposition,
        criterion_id=criterion_id,
    )


def test_plan_review_diagnostics_measure_judge_blockers_after_critic_pass():
    diagnostics = plan_workflow._plan_review_diagnostics((
        _diagnostic_verdict("critic", "pass"),
        _diagnostic_verdict(
            "judge",
            "revise",
            _diagnostic_finding(),
            _diagnostic_finding(criterion_id="contradiction"),
            _diagnostic_finding(disposition="addressed"),
            _diagnostic_finding(severity="Low"),
            _diagnostic_finding(criterion_id="executability"),
        ),
        _diagnostic_verdict("critic", "pass"),
        _diagnostic_verdict("judge", "pass"),
    ))

    assert diagnostics == {
        "schema_version": "crucible.plan-review-diagnostics.v1",
        "complete": True,
        "critic_passed_judged_rounds": 2,
        "judge_blocked_after_critic_pass_rounds": 1,
        "novel_high_critical_judge_blockers": 2,
        "judge_block_after_critic_pass_rate": 0.5,
        "unreadable_verdict_refs": 0,
    }


def test_plan_review_diagnostics_ignore_nonpassing_critic_and_unpaired_judge():
    diagnostics = plan_workflow._plan_review_diagnostics((
        _diagnostic_verdict("judge", "revise", _diagnostic_finding()),
        _diagnostic_verdict("critic", "revise", _diagnostic_finding()),
        _diagnostic_verdict("inflation", "revise", _diagnostic_finding()),
        _diagnostic_verdict("judge", "revise", _diagnostic_finding()),
    ))

    assert diagnostics["critic_passed_judged_rounds"] == 0
    assert diagnostics["judge_blocked_after_critic_pass_rounds"] == 0
    assert diagnostics["novel_high_critical_judge_blockers"] == 0
    assert diagnostics["judge_block_after_critic_pass_rate"] is None


def test_run_plan_composes_candidate_transform_before_product_policy(monkeypatch):
    captured = {}
    spec = _language_spec("python")
    expected_outcome = _forwarding_outcome(spec)

    def fake_plan(received_spec, **kwargs):
        captured.update(kwargs)
        assert received_spec is spec
        return expected_outcome

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    transform = lambda _dag: _candidate_with_languages(["cpp"])  # noqa: E731

    outcome = plan_workflow.run_plan(
        spec,
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
        candidate_transform=transform,
    )

    assert outcome is expected_outcome
    assert outcome.accepted
    effective = captured["candidate_transform"]
    assert callable(effective)
    candidate = _candidate_with_languages()
    transformed = effective(candidate)
    assert transformed.tiers[0].slices[0].directive.blast_radius == {
        "max_files": 1,
        "languages": ("cpp",),
    }


def _language_spec(*languages: str) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="polyglot",
            repo_url="https://example.invalid/polyglot.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="plan a polyglot change",
        constraints=(LanguagesConstraint(languages=languages),),
        done_criteria=(),
        plan_policy=None,
    )


_LANGUAGES_ABSENT = object()


def _candidate_with_languages(languages: object = _LANGUAGES_ABSENT) -> TierDAG:
    blast_radius: dict[str, object] = {"max_files": 1}
    if languages is not _LANGUAGES_ABSENT:
        blast_radius["languages"] = languages
    directive = ImplementationDirective(
        slice_id="s0",
        intent="implement",
        files_in_scope=("src/x.cpp",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius=blast_radius,
    )
    return TierDAG(
        tiers=(
            Tier(
                tier_id="t0",
                depends_on=(),
                slices=(Slice(id="s0", directive=directive),),
            ),
        )
    )


def test_run_plan_projects_missing_spec_languages_before_assay_gates(monkeypatch):
    captured = {}
    spec = _language_spec("python", "cpp")
    expected_outcome = _forwarding_outcome(spec)

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return expected_outcome

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    candidate = _candidate_with_languages()

    outcome = plan_workflow.run_plan(
        spec,
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    assert outcome is expected_outcome
    assert outcome.accepted
    transform = captured["candidate_transform"]
    assert callable(transform)
    transformed = transform(candidate)
    assert transformed is not candidate
    assert "languages" not in candidate.tiers[0].slices[0].directive.blast_radius
    assert transformed.tiers[0].slices[0].directive.blast_radius == {
        "max_files": 1,
        "languages": ("python", "cpp"),
    }


@pytest.mark.parametrize("inheritance_marker", [[], ()])
def test_run_plan_projects_explicit_language_inheritance_before_assay_gates(
    monkeypatch,
    inheritance_marker: object,
):
    """Accept the empty marker required by Assay's strict Planner schema."""

    captured = {}

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    plan_workflow.run_plan(
        _language_spec("python", "cpp"),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    candidate = _candidate_with_languages(inheritance_marker)
    transformed = captured["candidate_transform"](candidate)
    assert transformed.tiers[0].slices[0].directive.blast_radius == {
        "max_files": 1,
        "languages": ("python", "cpp"),
    }


@pytest.mark.parametrize("inheritance_marker", [[], ()])
def test_run_plan_accepts_explicit_language_inheritance_without_spec_default(
    monkeypatch,
    inheritance_marker: object,
):
    captured = {}

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    plan_workflow.run_plan(
        _language_spec(),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    candidate = _candidate_with_languages(inheritance_marker)
    transformed = captured["candidate_transform"](candidate)
    assert transformed.tiers[0].slices[0].directive.blast_radius == {
        "max_files": 1,
        "languages": (),
    }


def test_run_plan_rejects_duplicate_spec_language_declarations(monkeypatch):
    spec = replace(
        _language_spec("python"),
        constraints=(
            LanguagesConstraint(languages=("python",)),
            LanguagesConstraint(languages=("cpp",)),
        ),
    )

    with pytest.raises(ValueError, match="at most one LanguagesConstraint"):
        plan_workflow.run_plan(
            spec,
            planner_backend=SimpleNamespace(),
            critic_backend=SimpleNamespace(),
            judge_backend=None,
            corpus=SimpleNamespace(),
            critic_rubric=SimpleNamespace(),
        )


def test_run_plan_preserves_valid_explicit_slice_language_override(monkeypatch):
    captured = {}

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    plan_workflow.run_plan(
        _language_spec("python", "cpp"),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    candidate = _candidate_with_languages(("cpp",))
    assert captured["candidate_transform"](candidate) is candidate


def test_run_plan_preserves_explicit_override_without_spec_language_default(monkeypatch):
    captured = {}

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    plan_workflow.run_plan(
        _language_spec(),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    assert callable(captured["candidate_transform"])
    candidate = _candidate_with_languages(["cpp"])
    transformed = captured["candidate_transform"](candidate)
    assert transformed.tiers[0].slices[0].directive.blast_radius == {
        "max_files": 1,
        "languages": ("cpp",),
    }


@pytest.mark.parametrize(
    "languages",
    [None, "cpp", ("cpp", "cpp"), ("cpp", "")],
)
def test_run_plan_rejects_malformed_slice_language_override(
    monkeypatch,
    languages: object,
):
    captured = {}

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    plan_workflow.run_plan(
        _language_spec("python", "cpp"),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="blast_radius.languages"):
        captured["candidate_transform"](_candidate_with_languages(languages))


def test_run_plan_rejects_malformed_override_without_spec_default(monkeypatch):
    captured = {}

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)
    plan_workflow.run_plan(
        _language_spec(),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="blast_radius.languages"):
        captured["candidate_transform"](_candidate_with_languages("cpp"))


def test_run_plan_forwards_prompt_toolset_as_metadata(monkeypatch):
    captured = {}
    toolset = object()

    def fake_plan(_spec, **kwargs):
        captured.update(kwargs)
        return _forwarding_outcome(_spec)

    monkeypatch.setattr(plan_workflow, "plan", fake_plan)

    plan_workflow.run_plan(
        _language_spec(),
        planner_backend=SimpleNamespace(),
        critic_backend=SimpleNamespace(),
        judge_backend=None,
        corpus=SimpleNamespace(),
        critic_rubric=SimpleNamespace(),
        prompt_toolset=toolset,
    )

    assert captured["prompt_toolset"] is toolset


def test_run_plan_stops_on_planner_accounting_debt_before_critic(
    tmp_path: Path,
):
    dag = {
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "s0",
                "directive": {
                    "slice_id": "s0",
                    "intent": "implement",
                    "requirement_refs": ["implement"],
                    "files_in_scope": ["src/x.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            }],
        }],
    }

    class _DebtPlanner:
        name = "planner"
        model = "planner-model"

        def __init__(self):
            self.calls = 0

        def invoke(self, _invocation):
            self.calls += 1
            return InvocationResult(
                final_message=json.dumps(dag),
                transcript=(),
                usage=Usage(
                    input_tokens=3,
                    output_tokens=4,
                    cost_usd=0.25,
                ),
                stop_reason="end_turn",
                completeness=ExecutionCompleteness(
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=True,
                    trace_complete=True,
                ),
                execution_trace=BackendExecutionTrace(
                    backend=self.name,
                    requested_model=self.model,
                    events=(
                        BackendExecutionEvent(
                            sequence=0,
                            kind="terminal",
                            name="end_turn",
                        ),
                    ),
                ),
            )

    class _NoCritic:
        name = "critic"
        model = "critic-model"

        def __init__(self):
            self.calls = 0

        def invoke(self, _invocation):
            self.calls += 1
            raise AssertionError(
                "critic must not run after planner accounting debt"
            )

    planner = _DebtPlanner()
    critic = _NoCritic()
    spec = Spec(
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
        intent="implement",
        constraints=(),
        done_criteria=(),
        plan_policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=False,
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )

    outcome = plan_workflow.run_plan(
        spec,
        planner_backend=planner,
        critic_backend=critic,
        judge_backend=None,
        corpus=Corpus(tmp_path / "corpus"),
        critic_rubric=Rubric(
            name="critic",
            system_prompt="review",
            user_template="{artifact}",
        ),
    )

    assert outcome.status == "incomplete_accounting"
    assert outcome.decision == "inconclusive"
    assert outcome.cost_usd == 0.25
    assert planner.calls == 1
    assert critic.calls == 0
    assert outcome.spec.tier_dag is None


@pytest.mark.parametrize("debt_role", ["critic", "judge"])
def test_run_plan_stops_on_review_accounting_debt(
    tmp_path: Path,
    monkeypatch,
    debt_role: str,
):
    corpus = Corpus(tmp_path / "corpus")
    spec = Spec(
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
        intent="implement",
        constraints=(),
        done_criteria=(),
        plan_policy=PlanPolicy(
            max_critique_rounds=1,
            judge_required=debt_role == "judge",
            breaker_required=False,
            persistence_veto_threshold=10,
        ),
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )
    artifact = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "s0",
                "directive": {
                    "slice_id": "s0",
                    "intent": "implement",
                    "requirement_refs": ["implement"],
                    "files_in_scope": ["src/x.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            }],
        }],
    }).encode()
    artifact_ref = corpus.put_artifact(
        artifact,
        kind="reviewed_artifacts",
        name_hint="candidate.json",
    )
    transcript_ref = corpus.put_artifact(
        b"[]",
        kind="transcripts",
        name_hint="review.json",
    )

    def _verdict(role: str) -> Verdict:
        debt = role == debt_role
        return Verdict(
            rubric=role,
            decision="pass",
            findings=(),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning=f"{role} pass diagnostic",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(artifact),
                artifact_chars_reviewed=len(artifact),
            ),
            backend_name="stub",
            transcript_ref=transcript_ref,
            cost_usd=0.2 if role == "judge" else 0.1,
            execution=None,
            kind=role,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=not debt,
                cost_complete=not debt,
                terminal_complete=True,
                trace_complete=True,
            ),
            cache_creation_tokens=0,
            cache_read_tokens=0,
        )

    review_calls: list[str] = []

    def _strict_review(_artifact, *args, **kwargs):
        role = kwargs["kind"]
        review_calls.append(role)
        return _verdict(role)

    def _fake_plan(_spec, **kwargs):
        review_fn = kwargs["review_fn"]
        review_fn(
            artifact,
            backend=kwargs["critic_backend"],
            corpus=corpus,
            rubric=kwargs["critic_rubric"],
            kind="critic",
        )
        if debt_role == "critic":
            raise AssertionError("critic debt should have terminated")
        review_fn(
            artifact,
            backend=kwargs["judge_backend"],
            corpus=corpus,
            rubric=kwargs["judge_rubric"],
            kind="judge",
        )
        raise AssertionError("judge debt should have terminated")

    monkeypatch.setattr(plan_workflow, "strict_review", _strict_review)
    monkeypatch.setattr(plan_workflow, "plan", _fake_plan)

    planner_backend = SimpleNamespace()
    critic_backend = SimpleNamespace()
    judge_backend = (
        SimpleNamespace() if debt_role == "judge" else None
    )
    outcome = plan_workflow.run_plan(
        spec,
        planner_backend=planner_backend,
        critic_backend=critic_backend,
        judge_backend=judge_backend,
        corpus=corpus,
        critic_rubric=Rubric(
            name="critic",
            system_prompt="review",
            user_template="{artifact}",
        ),
        judge_rubric=(
            Rubric(
                name="judge",
                system_prompt="judge",
                user_template="{artifact}",
            )
            if debt_role == "judge"
            else None
        ),
    )

    assert outcome.status == "incomplete_accounting"
    assert outcome.decision == "inconclusive"
    assert outcome.cost_usd == pytest.approx(
        0.1 if debt_role == "critic" else 0.3
    )
    assert review_calls == (
        ["critic"] if debt_role == "critic" else ["critic", "judge"]
    )
    assert outcome.spec.tier_dag is None
    assert len(outcome.spec.verdict_refs) == (
        1 if debt_role == "critic" else 2
    )
    persisted = corpus.get_record(
        outcome.spec.verdict_refs[-1],
        as_type=Verdict,
    )
    assert persisted.kind == debt_role
    assert persisted.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert f"{debt_role} returned incomplete aggregate accounting" in (
        outcome.reason
    )
