# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the architect validate-gate repair loop.

When the planned tier_dag trips a blocking (Critical/High) validation
finding, the gate feeds the structured findings back into the planner's
intent and re-plans (bounded by ``max_repair_attempts``) rather than
aborting straight to an empty diff. This is the SWE-bench-critical
behavior: an over-decomposed or scope-mislabeled plan that the
deterministic validator can localize ("add X to creates_paths",
"parallel_tier_conflict") gets one repair round instead of forfeiting
the instance.

These tests drive the loop by stubbing the two collaborators the gate
calls — ``_assay_validate`` (deterministic ruleset) and ``run_plan``
(the LLM re-plan) — so we exercise the control flow without a backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from assay import PlanOutcome
from assay.types.findings import Finding, ReviewCoverage, Verdict
from assay.types.policy import PlanPolicy
from crucible.workflows import architect as arch


def _verdict(findings):
    return Verdict(
        rubric="architect_gate",
        decision="pass" if not findings else "reject",
        findings=tuple(findings),
        artifact_ref="a" * 64,
        artifact_commit=None,
        reasoning="",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("spec",),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        kind="validate",
    )


def _blocker(fid="parallel_tier_conflict"):
    return Finding(
        id=fid,
        severity="High",
        category="scope",
        evidence=(),
        suggested_action="add foo.py to creates_paths",
        source="deterministic",
    )


def _advisory(fid="phantom_anchor"):
    return Finding(
        id=fid,
        severity="Low",
        category="phantom_anchor",
        evidence=(),
        suggested_action="confirm the planned parent creation",
        source="deterministic",
    )


@dataclass(frozen=True)
class _SpecStub:
    # A frozen dataclass (not SimpleNamespace) so dataclasses.replace
    # works inside the gate. The gate only touches .intent / .tier_dag.
    intent: str = "do the thing"
    constraints: tuple = ()
    done_criteria: tuple = ()
    plan_policy: PlanPolicy | None = None
    tier_dag: object = "DAG"
    project_context: object = field(default_factory=lambda: SimpleNamespace(
        submodule_path=None,
    ))

    def with_verdict(self, verdict):
        return self


def _spec(intent="do the thing", *, tier_dag="DAG"):
    return _SpecStub(intent=intent, tier_dag=tier_dag)


def _plan_outcome(spec, *, cost_usd=0.0, decision="accepted", status="accepted"):
    return PlanOutcome(
        spec=spec,
        candidate_dag=None,
        candidate_ref=None,
        decision=decision,
        status=status,
        rounds_used=1,
        final_critic_decision="pass",
        final_judge_decision="pass",
        reason="stub",
        cost_usd=cost_usd,
    )


def _patch_common(monkeypatch, validate_seq, replan_outcomes):
    """Stub the gate's two collaborators.

    ``validate_seq`` — list of findings-lists, consumed one per
    validate pass. ``replan_outcomes`` — typed outcomes returned by
    successive ``run_plan`` calls. ``corpus`` records deterministic
    validation verdicts.
    """
    v_iter = iter(validate_seq)
    monkeypatch.setattr(
        arch, "_assay_validate",
        lambda spec, ruleset=None, ctx=None: _verdict(next(v_iter)),
    )
    r_iter = iter(replan_outcomes)
    replan_calls = []

    def _fake_run_plan(spec, **kw):
        replan_calls.append((spec, kw))
        return next(r_iter)

    monkeypatch.setattr(arch, "run_plan", _fake_run_plan)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    return replan_calls


def _call_gate(
    tmp_path,
    *,
    planned,
    spec,
    allow_repair,
    max_repair_attempts=2,
    budget_usd=None,
):
    put_records = []
    corpus = SimpleNamespace(
        put_record=lambda record, kind: put_records.append((record, kind)),
    )
    result_spec, status, repair_cost = arch._validation_gate(
        spec=spec,
        planned=planned,
        run_dir=tmp_path,
        corpus=corpus,
        planner_backend=object(),
        critic_backend=object(),
        judge_backend=object(),
        critic_rubric_name="swebench_plan_critic",
        judge_rubric_name=None,
        max_turns=50,
        budget_usd=budget_usd,
        cwd=str(tmp_path),
        allowed_tools=("Read", "Grep"),
        allow_repair=allow_repair,
        max_repair_attempts=max_repair_attempts,
    )
    return result_spec, status, repair_cost, put_records


def test_gate_passes_clean_plan_without_replan(tmp_path, monkeypatch):
    """A plan with no blocking findings passes on the first validate
    pass — no re-plan, no corpus write."""
    replan_calls = _patch_common(
        monkeypatch, validate_seq=[[]], replan_outcomes=[],
    )
    planned = _spec()
    result, status, repair_cost, put_records = _call_gate(
        tmp_path, planned=planned, spec=planned, allow_repair=True,
    )
    assert status == "passed"
    assert result is planned
    assert replan_calls == []
    assert repair_cost == 0.0
    assert len(put_records) == 1


def test_gate_treats_cwd_as_target_repo_root_with_submodule_spec(
    tmp_path,
    monkeypatch,
):
    """Architect must not append the plan-time submodule path twice."""

    captured_contexts = []

    def _capture_validate(spec, ruleset=None, ctx=None):
        captured_contexts.append(ctx)
        return _verdict([])

    monkeypatch.setattr(arch, "_assay_validate", _capture_validate)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    planned = _SpecStub(project_context=SimpleNamespace(
        submodule_path="fabric-usd",
    ))

    result, status, repair_cost, put_records = _call_gate(
        tmp_path,
        planned=planned,
        spec=planned,
        allow_repair=False,
    )

    assert status == "passed"
    assert result is planned
    assert repair_cost == 0.0
    assert len(put_records) == 1
    assert len(captured_contexts) == 1
    assert captured_contexts[0].workspace_root == tmp_path.resolve()
    assert captured_contexts[0].submodule_path is None


def test_gate_repairs_then_passes(tmp_path, monkeypatch):
    """First pass blocks; the gate re-plans with findings fed back and
    the repaired plan passes the second pass."""
    repaired = _spec("repaired")
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker()], []],   # fail, then pass
        replan_outcomes=[_plan_outcome(repaired, cost_usd=1.5)],
    )
    original_policy = PlanPolicy(planner_guidance="Keep existing policy.")
    original = _SpecStub(
        intent="original intent",
        constraints=("original constraint",),
        done_criteria=("original criterion",),
        plan_policy=original_policy,
    )
    result, status, repair_cost, put_records = _call_gate(
        tmp_path, planned=_spec("planned"), spec=original, allow_repair=True,
    )
    assert status == "passed"
    assert result.tier_dag == repaired.tier_dag
    assert result.intent == original.intent
    assert result.constraints == original.constraints
    assert result.done_criteria == original.done_criteria
    assert result.plan_policy == original_policy
    assert repair_cost == pytest.approx(1.5)
    assert len(put_records) == 2
    # Exactly one repair round.
    assert len(replan_calls) == 1
    # Findings are policy guidance, never generated root intent/requirements.
    repair_spec, kw = replan_calls[0]
    assert repair_spec.intent == original.intent
    assert repair_spec.constraints == original.constraints
    assert repair_spec.done_criteria == original.done_criteria
    guidance = repair_spec.plan_policy.planner_guidance
    assert guidance.startswith("Keep existing policy.")
    assert "VALIDATION FAILURES" in guidance
    assert "add foo.py to creates_paths" in guidance
    assert "MUST NOT be copied into requirement_refs" in guidance
    # Planner setup is forwarded so source-grounding survives the round.
    assert kw["cwd"] == str(tmp_path)
    assert kw["allowed_tools"] == ("Read", "Grep")


def test_gate_critique_uses_temporary_guidance_each_round(tmp_path, monkeypatch):
    """Every repair keeps root authority clean and does not stack feedback."""
    r1 = _spec("r1")
    r2 = _spec("r2")
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker("cycle")], [_blocker("blast_radius")], []],
        replan_outcomes=[_plan_outcome(r1), _plan_outcome(r2)],
    )
    original = _spec("BASE")
    result, status, _, _ = _call_gate(
        tmp_path, planned=_spec("planned"), spec=original,
        allow_repair=True, max_repair_attempts=2,
    )
    assert status == "passed"
    assert result.tier_dag == r2.tier_dag
    assert result.intent == original.intent
    assert result.plan_policy == original.plan_policy
    assert len(replan_calls) == 2
    # Both repair specs retain BASE exactly; feedback is one temporary policy
    # block, never a generated requirement source.
    for repair_spec, _kw in replan_calls:
        assert repair_spec.intent == "BASE"
        assert repair_spec.plan_policy.planner_guidance.count(
            "VALIDATION FAILURES"
        ) == 1


def test_gate_exhausts_repair_budget(tmp_path, monkeypatch):
    """If every repair round still blocks, the gate gives up with
    ``replan_validate_failed`` after max_repair_attempts re-plans."""
    r1 = _spec("r1")
    r2 = _spec("r2")
    replan_calls = _patch_common(
        monkeypatch,
        # 3 passes (initial + 2 repairs), all blocking.
        validate_seq=[[_blocker()], [_blocker()], [_blocker()]],
        replan_outcomes=[_plan_outcome(r1), _plan_outcome(r2)],
    )
    result, status, _, _ = _call_gate(
        tmp_path, planned=_spec("planned"), spec=_spec("base"),
        allow_repair=True, max_repair_attempts=2,
    )
    assert status == "replan_validate_failed"
    assert len(replan_calls) == 2


def test_gate_no_repair_aborts_on_block(tmp_path, monkeypatch):
    """With allow_repair=False (reused tier_dag, no opt-in) a blocking
    finding aborts to ``validate_failed`` without re-planning — we don't
    silently rewrite an operator's hand-tuned plan."""
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker(), _advisory()]],
        replan_outcomes=[],
    )
    planned = _spec("planned")
    result, status, repair_cost, put_records = _call_gate(
        tmp_path, planned=planned, spec=planned, allow_repair=False,
    )
    assert status == "validate_failed"
    assert result is planned
    assert replan_calls == []
    assert repair_cost == 0.0
    assert len(put_records) == 1
    # Findings still written for human review.
    findings_path = tmp_path / "validate_findings.json"
    assert findings_path.is_file()
    payload = json.loads(findings_path.read_text())
    assert payload["blocking_count"] == 1
    assert payload["advisory_count"] == 1
    assert payload["total_findings"] == 2
    assert payload["finding_counts"] == {
        "total": 2,
        "blocking": 1,
        "advisory": 1,
    }


def test_gate_replan_no_tier_dag_fails(tmp_path, monkeypatch):
    """If the repair re-plan itself produces no tier_dag, the gate
    fails rather than descending into implement against a None DAG."""
    dead = _spec("dead", tier_dag=None)
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker()]],
        replan_outcomes=[_plan_outcome(
            dead,
            decision="rejected",
            status="critic_rejected",
        )],
    )
    result, status, _, _ = _call_gate(
        tmp_path, planned=_spec("planned"), spec=_spec("base"),
        allow_repair=True,
    )
    assert status == "replan_validate_failed"
    assert len(replan_calls) == 1


@pytest.mark.parametrize(
    ("decision", "replan_status"),
    [
        ("inconclusive", "planner_infrastructure_failure"),
        ("inconclusive", "critic_infrastructure_failure"),
        ("inconclusive", "judge_protocol_failure"),
        ("inconclusive", "cancelled"),
        # An open accounting terminal keeps its own name as well, as on the
        # planning path; a parent reads that name as accounting debt.
        ("inconclusive", "incomplete_accounting"),
    ],
)
def test_gate_replan_without_a_ruling_keeps_its_status(
    tmp_path, monkeypatch, decision, replan_status,
):
    """A repair re-plan nobody ruled on is not a failed validation repair.

    The gate still blocks: the plan is not accepted and no further repair
    round runs. Only the reported status keeps the re-plan's own typed
    terminal instead of ``replan_validate_failed``.
    """
    lost = _spec("lost", tier_dag=None)
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker()]],
        replan_outcomes=[_plan_outcome(
            lost,
            decision=decision,
            status=replan_status,
            cost_usd=0.25,
        )],
    )
    result, status, repair_cost, _ = _call_gate(
        tmp_path, planned=_spec("planned"), spec=_spec("base"),
        allow_repair=True, max_repair_attempts=2,
    )
    assert status == replan_status
    assert result.tier_dag is None
    assert repair_cost == pytest.approx(0.25)
    assert len(replan_calls) == 1


@pytest.mark.parametrize(
    ("decision", "replan_status"),
    [
        ("inconclusive", "rounds_exhausted"),
        ("inconclusive", "no_candidate"),
        ("rejected", "judge_rejected"),
        # In no table: an unknown status is never read as "no ruling".
        ("inconclusive", "accounting_incomplete"),
    ],
)
def test_gate_replan_ruled_against_stays_replan_validate_failed(
    tmp_path, monkeypatch, decision, replan_status,
):
    """The pass-through keys on the typed class, not on the plan decision."""
    dead = _spec("dead", tier_dag=None)
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker()]],
        replan_outcomes=[_plan_outcome(
            dead,
            decision=decision,
            status=replan_status,
        )],
    )
    _result, status, _, _ = _call_gate(
        tmp_path, planned=_spec("planned"), spec=_spec("base"),
        allow_repair=True,
    )
    assert status == "replan_validate_failed"
    assert len(replan_calls) == 1


def test_gate_replan_budget_stop_keeps_budget_exhausted(tmp_path, monkeypatch):
    """The budget terminal is tested first and keeps its exit-1 status."""
    dead = _spec("dead", tier_dag=None)
    _patch_common(
        monkeypatch,
        validate_seq=[[_blocker()]],
        replan_outcomes=[_plan_outcome(
            dead,
            decision="inconclusive",
            status="budget_exhausted",
        )],
    )
    _result, status, _, _ = _call_gate(
        tmp_path, planned=_spec("planned"), spec=_spec("base"),
        allow_repair=True,
    )
    assert status == "budget_exhausted"


def test_gate_passes_only_remaining_budget_to_each_repair(tmp_path, monkeypatch):
    r1 = _spec("r1")
    r2 = _spec("r2")
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker("one")], [_blocker("two")], []],
        replan_outcomes=[
            _plan_outcome(r1, cost_usd=3.0),
            _plan_outcome(r2, cost_usd=4.0),
        ],
    )

    result, status, repair_cost, _ = _call_gate(
        tmp_path,
        planned=_spec("planned"),
        spec=_spec("base"),
        allow_repair=True,
        budget_usd=10.0,
    )

    assert result.tier_dag == r2.tier_dag
    assert result.intent == "base"
    assert status == "passed"
    assert repair_cost == pytest.approx(7.0)
    assert [call[1]["budget_usd"] for call in replan_calls] == [10.0, 7.0]


def test_gate_halts_when_repairs_consume_remaining_budget(tmp_path, monkeypatch):
    r1 = _spec("r1")
    replan_calls = _patch_common(
        monkeypatch,
        validate_seq=[[_blocker("one")], [_blocker("two")]],
        replan_outcomes=[_plan_outcome(r1, cost_usd=3.0)],
    )

    _result, status, repair_cost, _ = _call_gate(
        tmp_path,
        planned=_spec("planned"),
        spec=_spec("base"),
        allow_repair=True,
        budget_usd=3.0,
    )

    assert status == "budget_exhausted"
    assert repair_cost == pytest.approx(3.0)
    assert len(replan_calls) == 1
