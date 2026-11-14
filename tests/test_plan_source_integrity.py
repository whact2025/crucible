# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Source identity cannot be waived by optional deterministic-gate overrides."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json

import pytest
from assay import ExecutionCompleteness, LifecycleEvent, ReviewCoverage, Verdict
from assay.corpus.store import Corpus

from crucible.rubrics import load
from crucible.workflows import plan as workflow
from tests.test_plan_review_response_retry import _Backend
from tests.test_workflows_plan_candidate_import import _candidate_payload, _spec


@pytest.fixture
def source_plan(tmp_path, monkeypatch):
    source = tmp_path / "evidence.py"
    source.write_text("original source evidence\n")
    payload = _candidate_payload()
    payload["tiers"][0]["slices"][0]["directive"]["read_only_references"] = [{
        "reference_id": "existing",
        "path": source.name,
        "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_line": 1,
        "end_line": 1,
    }]
    corpus = Corpus(tmp_path / "corpus")
    contexts = []
    state = {"mutate": lambda: None}

    def review(artifact, rubric, **kwargs):
        contexts.append(kwargs["context"]["read_only_sources"])
        if kwargs["kind"] == "critic":
            state["mutate"]()
        return Verdict(
            rubric=rubric.name, decision="pass", findings=(),
            artifact_ref=corpus.put_artifact(artifact, kind="reviewed_artifacts"),
            artifact_commit=None, reasoning="fixture review",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact", "context:read_only_sources"),
                omissions=(), artifact_chars_total=len(artifact),
                artifact_chars_reviewed=len(artifact),
            ),
            backend_name="fixture", transcript_ref=None, cost_usd=0,
            execution=None, kind=kwargs["kind"],
            cache_creation_tokens=0, cache_read_tokens=0,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True, cost_complete=True,
                terminal_complete=True, trace_complete=True,
            ),
        )

    monkeypatch.setattr(workflow, "strict_review", review)
    return source, payload, corpus, contexts, state


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("mutation", ["unchanged", "changed", "removed", "symlink"])
def test_programmatic_source_recheck_preserves_usage_and_prior_authority(
    source_plan, fixed, mutation,
):
    source, payload, corpus, contexts, state = source_plan
    prior = workflow._strict_review_candidate_dag(_candidate_payload())
    base = replace(_spec(), tier_dag=prior)
    historical_verdict = corpus.put_record(workflow.validate(
        base, ruleset=workflow.ValidationRuleset(name="historic", rule_ids=()),
    ), kind="verdicts")
    historical_event = corpus.put_record(LifecycleEvent(
        kind="planned", reason="Historical accepted authority",
        affected_versions=(base.version,), execution=workflow.exec_ctx_now("plan"),
    ), kind="lifecycle_events")
    base = replace(
        base, verdict_refs=(historical_verdict,), lifecycle_event_refs=(historical_event,),
    )
    corpus.put_spec(base)
    original_bytes = source.read_bytes()

    def mutate():
        if mutation == "changed":
            source.write_text("changed during review\n")
        elif mutation == "removed":
            source.unlink()
        elif mutation == "symlink":
            target = source.with_name("alias-target.py")
            target.write_bytes(original_bytes)
            source.unlink()
            source.symlink_to(target)

    state["mutate"] = mutate
    planner = _Backend(json.dumps(payload))
    candidate = workflow._strict_review_candidate_dag(payload)
    outcome = workflow.run_plan(
        base, corpus=corpus, cwd=str(source.parent),
        planner_backend=None if fixed else planner,
        candidate_dag=candidate if fixed else None,
        critic_backend=_Backend(), judge_backend=_Backend(),
        critic_rubric=load("spec_clarity"), judge_rubric=load("spec_clarity"),
        # No optional candidate validator, as with Architect --skip-validate.
    )
    assert outcome.accepted is (mutation == "unchanged")
    assert outcome.spec.tier_dag == (candidate if outcome.accepted else prior)
    assert outcome.candidate_dag == candidate
    assert corpus.get_artifact(outcome.candidate_ref)
    assert corpus.latest_spec() == base  # The programmatic entry only stages.
    assert len(contexts) == 2 and contexts[0] == contexts[1]
    assert len(planner.invocations) == (0 if fixed else 1)
    assert outcome.cost_usd == outcome.usage.cost_usd == (0 if fixed else 0.01)
    assert outcome.input_tokens == outcome.output_tokens == (0 if fixed else 1)
    assert outcome.final_critic_decision == outcome.final_judge_decision == "pass"
    assert outcome.spec.lifecycle_event_refs[:1] == base.lifecycle_event_refs
    assert outcome.spec.verdict_refs[:1] == base.verdict_refs
    events = [corpus.get_record(ref, as_type=LifecycleEvent)
              for ref in outcome.spec.lifecycle_event_refs[1:]]
    assert [event.kind for event in events] == [
        "planned" if outcome.accepted else "plan_rejected",
    ]
    verdicts = [corpus.get_record(ref, as_type=Verdict)
                for ref in outcome.spec.verdict_refs[1:]]
    assert [verdict.kind for verdict in verdicts] == (
        ["critic", "judge"] if outcome.accepted else ["critic", "judge", "validate"]
    )
    if not outcome.accepted:
        assert outcome.status == "validate_blocked"
        assert verdicts[-1].findings[0].id == "read_only_source_invalid"


@pytest.mark.parametrize("stage", ["review", "after_return", "stable", "no_references"])
def test_cli_skip_gate_still_rechecks_sources_at_publication(
    source_plan, monkeypatch, stage,
):
    source, payload, corpus, contexts, state = source_plan
    base = _spec()
    corpus.put_spec(base)
    if stage == "no_references":
        payload["tiers"][0]["slices"][0]["directive"].pop("read_only_references")

        def forbidden_validation(*_args, **_kwargs):
            pytest.fail("no-reference --no-validate-gate must remain a true skip")

        monkeypatch.setattr(workflow, "validate", forbidden_validation)
    if stage == "review":
        state["mutate"] = lambda: source.write_text("changed during review\n")
    original_run_plan = workflow.run_plan

    def run_plan(*args, **kwargs):
        outcome = original_run_plan(*args, **kwargs)
        if stage == "after_return":
            assert outcome.accepted
            source.write_text("changed between acceptance and publication\n")
        return outcome

    planner = _Backend(json.dumps(payload))

    class Pool:
        def get(self, role, **_kwargs):
            return planner if role == "plan_planner" else _Backend()

    @contextmanager
    def pool(*_args, **_kwargs):
        yield Pool()

    monkeypatch.setattr(workflow, "run_plan", run_plan)
    monkeypatch.setattr(workflow, "backend_pool", pool)
    monkeypatch.setattr(workflow, "plan_contract_fingerprint", lambda *_a, **_k: "sha256:" + "a" * 64)
    monkeypatch.setattr(workflow, "bind_plan_contract", lambda _corpus, staged, **_kwargs: staged)
    rejected = stage in {"review", "after_return"}
    assert workflow.main([
        "--planner-count", "1",
        "--spec", str(corpus.spec_dir), "--cwd", str(source.parent),
        "--no-validate-gate",
    ]) == (1 if rejected else 0)
    latest = corpus.latest_spec()
    assert latest.version == base.version + 1
    assert (latest.tier_dag is None) is rejected
    assert corpus.get_spec(base.version) == base
    assert len(planner.invocations) == 1
    assert len(contexts) == 2 and contexts[0] == contexts[1]
    events = [corpus.get_record(ref, as_type=LifecycleEvent)
              for ref in latest.lifecycle_event_refs]
    assert [event.kind for event in events] == ["plan_rejected" if rejected else "planned"]
    (run,) = (corpus.spec_dir / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["costs"]["total_usd"] == 0.01
    if rejected:
        assert summary["status"] == "validate_blocked"
    if stage == "after_return":
        assert summary["validate_attempts"][0]["findings"][0]["id"] == "read_only_source_invalid"
    elif stage == "no_references":
        assert summary["validate_attempts"] == []


@pytest.mark.parametrize("stage", ["fresh_review", "fresh_after_return", "reuse"])
def test_architect_skip_validation_cannot_execute_stale_source_plan(
    source_plan, monkeypatch, stage,
):
    from crucible.workflows import architect
    from crucible.workflows._plan_provenance import bind_plan_contract
    from tests.test_workflows_architect_reaccept import _fixed_fingerprint

    source, payload, corpus, contexts, state = source_plan
    base = _spec()
    corpus.put_spec(base)
    fingerprint = _fixed_fingerprint()
    reuse = stage == "reuse"
    if reuse:
        event_ref = corpus.put_record(LifecycleEvent(
            kind="planned", reason="Retained reviewed plan",
            affected_versions=(base.version + 1,), execution=workflow.exec_ctx_now("plan"),
        ), kind="lifecycle_events")
        base = bind_plan_contract(corpus, base.with_appended_refs(
            tier_dag=workflow._strict_review_candidate_dag(payload),
            lifecycle_event_refs=(event_ref,),
        ), fingerprint=fingerprint)
        corpus.put_spec(base)
        source.write_text("changed since the original accepted plan\n")
    elif stage == "fresh_review":
        state["mutate"] = lambda: source.write_text("changed during fresh review\n")
    else:
        original_run_plan = architect.run_plan

        def run_plan(*args, **kwargs):
            outcome = original_run_plan(*args, **kwargs)
            assert outcome.accepted
            source.write_text("changed after review before Architect publication\n")
            return outcome

        monkeypatch.setattr(architect, "run_plan", run_plan)

    planner = _Backend(json.dumps(payload))

    class Pool:
        def get(self, role, **_kwargs):
            return planner if role == "architect_planner" else _Backend()

    @contextmanager
    def pool(*_args, **_kwargs):
        yield Pool()

    monkeypatch.setattr(architect, "backend_pool", pool)
    monkeypatch.setattr(architect, "plan_contract_fingerprint", lambda *_a, **_k: fingerprint)
    monkeypatch.setattr(architect, "run_tier", lambda **_k: pytest.fail("stale evidence must stop before tier execution"))
    assert architect.run_architect(
        planner_count=1,
        spec=corpus.spec_dir, cwd=str(source.parent), skip_validate=True,
        replan_on_validate_fail=True,
        critic_rubric="spec_clarity", judge_rubric="spec_clarity",
    ) == 2
    latest = corpus.latest_spec()
    assert latest.tier_dag == base.tier_dag
    assert corpus.get_spec(base.version) == base
    assert latest.lifecycle_event_refs[:len(base.lifecycle_event_refs)] == base.lifecycle_event_refs
    event = corpus.get_record(latest.lifecycle_event_refs[-1], as_type=LifecycleEvent)
    assert event.kind == "plan_rejected"
    assert latest.lifecycle_event_refs[:-1] == base.lifecycle_event_refs
    assert len(planner.invocations) == (0 if reuse else 1)
    assert len(contexts) == (0 if reuse else 2)
    (run,) = (corpus.spec_dir / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == (
        "validate_blocked" if stage == "fresh_review" else "validate_failed"
    )
    assert summary["costs"]["total_usd"] == (0 if reuse else 0.01)
    assert summary["tiers_executed"] == 0


def test_architect_source_only_gate_does_not_enable_generic_rules(
    source_plan, monkeypatch,
):
    from crucible.workflows import architect

    source, payload, corpus, _contexts, _state = source_plan
    base = replace(_spec(), tier_dag=workflow._strict_review_candidate_dag(payload))
    corpus.put_spec(base)
    monkeypatch.setattr(architect, "_available_rules", lambda: pytest.fail(
        "source-only validation must not select the generic ruleset",
    ))
    staged, status, cost = architect._validation_gate(
        spec=base, planned=base, run_dir=source.parent, corpus=corpus,
        planner_backend=None, critic_backend=None, judge_backend=None,
        critic_rubric_name="spec_clarity", judge_rubric_name="spec_clarity",
        max_turns=1, budget_usd=0, cwd=str(source.parent), allowed_tools=(),
        allow_repair=False, source_identity_only=True,
    )
    assert status == "passed" and cost == 0
    assert staged.tier_dag == base.tier_dag
    assert corpus.latest_spec() == base
    verdict = corpus.get_record(staged.verdict_refs[-1], as_type=Verdict)
    assert verdict.passed
