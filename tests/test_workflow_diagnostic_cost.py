# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Explicit diagnostic budget continuation never creates promotion evidence."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from assay import Implementation, LifecycleEvent, Message
from assay.corpus.store import Corpus
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY
from crucible.common.cost import CostAccumulator
from crucible.workflows._development_budget import DevelopmentBudgetLedger
from crucible.workflows._diagnostic_cost import (
    accounting_blocks_paid_work,
    diagnostic_call_covered,
    diagnostic_summary_bounds,
    validate_diagnostic_cost_policy,
)
from crucible.workflows.architect import (
    _PlanningAccountingIncomplete,
    _PlanningUsageBackend,
)
from crucible.workflows.plan import _CostTrackingBackend, _PlanAccountingIncomplete
from tests.test_backend_limits_cost_bounds import _bound_result
from tests.test_development_budget import _Backend, _invocation, _policy
from tests.test_workflows_tier_empty_diff_retry import (
    _contract_aware_build_verify_stub,
    _run,
    _successful_gate_outcome,
)


@pytest.mark.parametrize("mode,budget,policy", [
    ("strict", 1, DIAGNOSTIC_COST_POLICY),
    (None, 1, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", None, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", 0, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", float("inf"), DIAGNOSTIC_COST_POLICY),
    ("diagnostic", True, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", 1, "other"),
])
def test_diagnostic_policy_requires_explicit_finite_authority(mode, budget, policy):
    with pytest.raises(ValueError):
        validate_diagnostic_cost_policy(policy, limit_mode=mode, budget_usd=budget)


def test_legacy_mode_is_unmodified():
    validate_diagnostic_cost_policy(None, limit_mode=None, budget_usd=None)
    assert not diagnostic_call_covered(_bound_result(), policy=None)


def _costs():
    costs = CostAccumulator(max_cost_usd=1, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    costs.record(role="planner", usage=_bound_result().usage, backend_name="kimi", model="kimi-k3.1-pro")
    return costs


def test_summary_recomputes_bound_and_preserves_debt():
    costs = _costs()
    summary = costs.to_dict()
    assert summary["accounting_debt"] is True
    assert not accounting_blocks_paid_work(costs)
    bounds = diagnostic_summary_bounds(summary, policy=DIAGNOSTIC_COST_POLICY)
    assert bounds == _bound_result().cost_bounds
    assert bounds.segments[0].cache_read_tokens is None
    assert costs.remaining() == pytest.approx(1 - bounds.upper_bound_usd)


@pytest.mark.parametrize("mutation", [
    "count", "debit", "coverage", "total", "tokens", "unloaded", "unloaded_bool",
    "missing_call", "missing_bound", "unknown_policy", "formula", "unknown_counter",
])
def test_bound_summary_tampering_is_not_spend_authority(mutation):
    summary = copy.deepcopy(_costs().to_dict())
    call = summary["cost_bounds_by_call"][0]
    if mutation == "count":
        summary["by_role"]["planner"]["calls"] += 1
    elif mutation == "debit":
        call["budget_debit_usd"] = 0
    elif mutation == "coverage":
        call["coverage_complete"] = False
    elif mutation == "total":
        summary["total_usd"] = 0
    elif mutation == "tokens":
        summary["total_input_tokens"] += 1
    elif mutation == "unloaded":
        summary["unloaded_refs"] = 1
    elif mutation == "unloaded_bool":
        summary["unloaded_refs"] = False
    elif mutation == "missing_call":
        summary["cost_bounds_by_call"] = []
    elif mutation == "missing_bound":
        call["cost_bounds"] = None
    elif mutation == "unknown_policy":
        summary["diagnostic_cost_policy"] = "other"
    elif mutation == "formula":
        call["cost_bounds"]["segments"][0]["upper_bound_usd"] = 10
    elif mutation == "unknown_counter":
        call["cost_bounds"]["segments"][0]["cache_creation_tokens"] = None
    assert diagnostic_summary_bounds(summary, policy=DIAGNOSTIC_COST_POLICY) is None


def test_anonymous_debt_is_never_recovered_by_later_bound():
    costs = _costs()
    costs.note_accounting_debt(role="unreturned", missing=("cost_complete",))
    costs.record(role="critic", usage=_bound_result().usage, backend_name="kimi")
    assert accounting_blocks_paid_work(costs)
    assert costs.remaining() == 0


@pytest.mark.parametrize("policy", [None, DIAGNOSTIC_COST_POLICY])
def test_planner_wrappers_allow_only_explicit_covered_debt(policy):
    result = _bound_result()
    costs = CostAccumulator(max_cost_usd=1, diagnostic_cost_policy=policy)
    raw = _Backend([result, result])
    architect = _PlanningUsageBackend(raw, role="planner", usage=costs)
    planner = _CostTrackingBackend(architect, diagnostic_cost_policy=policy)
    if policy is None:
        with pytest.raises(_PlanningAccountingIncomplete):
            planner.invoke(_invocation())
        assert len(raw.invocations) == 1
        return
    assert planner.invoke(_invocation()) == result
    assert planner.invoke(_invocation()) == result
    assert planner.accounting_debt == ("token_usage_complete", "cost_complete")
    assert costs.to_dict()["accounting_debt"] is True
    assert costs.budget_cost_usd == pytest.approx(2 * result.cost_bounds.upper_bound_usd)


@pytest.mark.parametrize("field", ["terminal_complete", "trace_complete"])
def test_incomplete_terminal_or_trace_remains_a_planning_hold(field):
    result = _bound_result()
    result = replace(result, completeness=replace(result.completeness, **{field: False}))
    planner = _CostTrackingBackend(_Backend([result]), diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    with pytest.raises(_PlanAccountingIncomplete):
        planner.invoke(_invocation())


def test_development_budget_debits_upper_without_erasing_unknown_actuals():
    result = _bound_result()
    ledger = DevelopmentBudgetLedger(_policy(calls=2), diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    raw = _Backend([result, result])
    backend = ledger.wrap("planner", raw)
    assert backend.invoke(_invocation()).error is None
    assert backend.invoke(_invocation()).error is None
    summary = ledger.to_summary()
    assert summary["authority_revoked"] is False
    assert summary["roles"]["planner"]["accounting_debt"] is True
    assert summary["roles"]["planner"]["usage"]["cost_usd"] == pytest.approx(2 * result.cost_bounds.upper_bound_usd)
    assert summary["cost_values_are_budget_upper_bounds"] is True
    assert summary["exact_cost_complete"] is False


class _DiagnosticWriter:
    name = "kimi"
    model = "kimi-k3.1-pro"

    def __init__(self, *, known_read=False, incomplete_terminal=False):
        self.repo = None
        self.invocations = []
        self.known_read = known_read
        self.incomplete_terminal = incomplete_terminal

    def attach_toolset(self, toolset, *, context_budget=None):
        pass

    def close(self):
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        self.repo.joinpath("target.py").write_text("value = diagnostic\n", encoding="utf-8")
        result = _bound_result(known_read=self.known_read)
        result = replace(result, transcript=(Message(role="assistant", content="DONE"),))
        if self.incomplete_terminal:
            result = replace(result, completeness=replace(result.completeness, terminal_complete=False))
        return result


@pytest.mark.parametrize("known_read", [False, True])
def test_fresh_diagnostic_tier_is_checked_nonlanded_and_not_resumable(tmp_path, monkeypatch, known_read):
    from crucible.workflows import _partition, tier

    monkeypatch.setattr(tier._build_verify, "build_and_verify", _contract_aware_build_verify_stub(lambda *args, **kwargs: _successful_gate_outcome()))
    backend = _DiagnosticWriter(known_read=known_read)
    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend, skip_build_verify=False,
        diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
        backend_limit_mode="diagnostic", backend_limit_deadline_seconds=1200,
        budget_usd=12,
    )
    assert rc == 1
    assert summary["status"] == "diagnostic_completed"
    assert summary["rankable"] is summary["promotable"] is summary["landed"] is False
    assert len(backend.invocations) == 1
    assert repo.joinpath("target.py").read_text() == "value = diagnostic\n"
    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    events = [corpus.get_record(ref, as_type=LifecycleEvent) for ref in spec.lifecycle_event_refs]
    assert events[-1].kind == "diagnostic_completed"
    assert not any(event.kind == "landed" for event in events)
    implementations = [corpus.get_record(ref, as_type=Implementation) for ref in spec.implementation_refs]
    assert implementations[-1].cost_bounds is not None
    assert _partition.partition_for_resume(corpus, spec.tier_dag.tiers[0], spec=spec)[0] == {}
    with pytest.raises(ValueError, match="fresh"):
        tier.run_tier(
            spec=tmp_path / "corpus", tier_id="t0", cwd=str(repo),
            diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
            backend_limit_mode="diagnostic", backend_limit_deadline_seconds=1200,
            budget_usd=12,
        )


def test_diagnostic_resume_predicate_rejects_even_exact_cost():
    from crucible.workflows._partition import _is_skip_worthy

    implementation = SimpleNamespace(cost_bounds=_bound_result(known_read=True).cost_bounds)
    assert not _is_skip_worthy("success", SimpleNamespace(), implementation)


def test_review_retry_preserves_bounds_and_debits_upper(tmp_path):
    from crucible.rubrics import load
    from crucible.workflows._review import review
    from tests.test_review_response_retry import (
        _ARTIFACT, _Reply, _SequenceBackend, _pass_response,
        _retry_manifest, _schema_failure_response,
    )

    class BoundReviewBackend(_SequenceBackend):
        def invoke(self, invocation):
            result = super().invoke(invocation)
            bounded = _bound_result()
            return replace(result, usage=bounded.usage, completeness=bounded.completeness)

    backend = BoundReviewBackend(
        _Reply(_schema_failure_response("bad_extra_field", "bad"), 0),
        _Reply(json.dumps(_pass_response()), 0),
    )
    corpus = Corpus(tmp_path / "review")
    verdict = review(
        _ARTIFACT, load("spec_clarity"), backend=backend, corpus=corpus,
        budget_usd=1, response_retry_budget=1,
        diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
    )
    assert verdict.decision == "pass"
    assert verdict.accounting_debt
    assert verdict.cost_bounds is not None
    assert len(verdict.cost_bounds.segments) == 2
    assert len(backend.invocations) == 2
    assert backend.invocations[1].budget_usd == pytest.approx(1 - _bound_result().cost_bounds.upper_bound_usd)
    manifest = _retry_manifest(corpus, verdict)
    assert manifest["accounting_debt"]
    assert manifest["cost_bounds"] == verdict.cost_bounds.to_record()
    assert all(attempt["cost_bounds"]["segments"][0]["cache_read_tokens"] is None for attempt in manifest["attempts"])


def test_terminal_incomplete_diagnostic_writer_is_restored_and_not_retried(tmp_path, monkeypatch):
    backend = _DiagnosticWriter(incomplete_terminal=True)
    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend, inherited_dirty=True,
        diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
        backend_limit_mode="diagnostic", backend_limit_deadline_seconds=1200,
        budget_usd=12,
    )
    assert rc == 1
    assert summary["status"] == "slice_failed"
    assert len(backend.invocations) == 1
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"


def test_diagnostic_plan_contract_is_not_ordinary_plan_authority(tmp_path):
    from crucible.rubrics import load
    from crucible.workflows._plan_provenance import (
        accepted_plan_contract_state, bind_plan_contract, plan_contract_fingerprint,
    )
    from tests.test_plan_provenance import _accepted_staged, _fingerprint, _spec

    spec = _spec()
    corpus = Corpus(tmp_path / "plan")
    corpus.put_spec(spec)
    diagnostic = plan_contract_fingerprint(
        spec, critic_rubric=load("deepswe_plan_critic"),
        judge_rubric=load("plan_judge"), candidate_transform=None,
        diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
    )
    strict = _fingerprint(spec)
    assert diagnostic != strict
    bound = bind_plan_contract(corpus, _accepted_staged(corpus, spec), fingerprint=diagnostic)
    assert accepted_plan_contract_state(corpus, bound, expected_fingerprint=strict).status == "stale"


def test_deepswe_explicit_policy_reaches_architect_and_rerun_is_denied(tmp_path):
    from crucible.workflows import deepswe
    from tests.test_workflows_deepswe import _build_corpus

    manifest = _build_corpus(tmp_path)
    argv = [
        "--spec", str(tmp_path / "pilot"), "--dataset", str(manifest),
        "--implement", "--planner-count", "1", "--budget-usd", "12", "--backend-limit-mode", "diagnostic",
        "--backend-limit-deadline-seconds", "1200", "--diagnostic-cost-policy", DIAGNOSTIC_COST_POLICY,
    ]
    args = deepswe._parse_args(argv)
    inst = deepswe._load_and_filter(args)[0]
    kwargs = deepswe._build_run_architect_kwargs(
        args=args, inst=inst, instance_spec_dir=tmp_path / "instance",
        container_id="unused", instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )
    assert kwargs["diagnostic_cost_policy"] == DIAGNOSTIC_COST_POLICY
    assert kwargs["backend_limit_mode"] == "diagnostic"
    assert kwargs["planner_count"] == 1
    with pytest.raises(SystemExit):
        deepswe._parse_args([*argv, "--planner-count", "3"])
    with pytest.raises(SystemExit):
        deepswe._parse_args([*argv, "--rerun"])


# This integration case proves both Tier and Breach journal settlement. With
# coverage its real metadata checks exceed the ordinary 120-second allowance;
# retain those checks and bound only this test, not production cleanup.
@pytest.mark.timeout(300)
def test_diagnostic_breach_child_bound_receipt_reaches_tier_without_landing(tmp_path, monkeypatch):
    from crucible.common.policies import BreachPolicy
    from crucible.workflows import tier
    from tests.test_workflows_breach import _breaker_clean

    class Breaker:
        name = "kimi"
        model = "kimi-k3.1-pro"
        calls = 0

        def invoke(self, invocation):
            self.calls += 1
            result = _bound_result()
            return replace(result, final_message=_breaker_clean(), transcript=(Message(role="assistant", content=_breaker_clean()),))

        def close(self):
            pass

    writer = _DiagnosticWriter()
    breaker = Breaker()
    monkeypatch.setattr(tier._build_verify, "build_and_verify", _contract_aware_build_verify_stub(lambda *args, **kwargs: _successful_gate_outcome()))
    monkeypatch.setattr(tier.breach_workflow, "validate_breach_observation_runner", lambda *args, **kwargs: None)
    rc, summary, repo = _run(
        tmp_path, monkeypatch, writer, skip_build_verify=False,
        backend_resolver=lambda role, **kwargs: breaker if "breaker" in role else writer,
        breach_policy=BreachPolicy(enabled=True, max_rounds=1),
        diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
        backend_limit_mode="diagnostic", backend_limit_deadline_seconds=1200,
        budget_usd=12,
    )
    assert rc == 1
    assert summary["status"] == "diagnostic_completed"
    assert breaker.calls == 1
    assert len(writer.invocations) == 1
    assert repo.joinpath("target.py").read_text() == "value = diagnostic\n"
    receipt = summary["phase_3_breach"]["cost_reconciliation"]
    assert receipt["diagnostic_upper_bound_covered"] is True
    assert receipt["cost_complete"] is False
    assert summary["costs"]["budget_cost_usd"] == pytest.approx(2 * _bound_result().cost_bounds.upper_bound_usd)


def test_diagnostic_reward_one_is_durable_truth_but_never_landed_or_reusable(tmp_path, monkeypatch):
    from crucible.workflows import deepswe as wf
    from tests.test_workflows_deepswe import (
        _build_corpus, _patch_container_lifecycle, _patch_partial_tier_settlement,
        _prepare_parent_run,
    )

    manifest = _build_corpus(tmp_path)
    args = wf._parse_args([
        "--spec", str(tmp_path / "s"), "--dataset", str(manifest), "--implement",
        "--planner-count", "1",
        "--budget-usd", "12", "--backend-limit-mode", "diagnostic",
        "--backend-limit-deadline-seconds", "1200", "--diagnostic-cost-policy", DIAGNOSTIC_COST_POLICY,
    ])
    inst = wf._load_and_filter(args)[0]
    (manifest.parent / inst.task_id / "tests").mkdir(parents=True, exist_ok=True)
    candidate = wf.candidate_from_image_patch(
        b"diagnostic candidate\n", files_modified=("adaptix/_internal.py",),
        raw_diff_bytes=21, max_bytes=args.max_patch_bytes,
    )
    calls = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls, candidate=candidate)
    _patch_partial_tier_settlement(monkeypatch)

    def architect(**kwargs):
        assert kwargs["diagnostic_cost_policy"] == DIAGNOSTIC_COST_POLICY
        return 1

    monkeypatch.setattr("crucible.workflows.architect.run_architect", architect)
    monkeypatch.setattr(wf, "_run_dataset_verify", lambda *args, **kwargs: {
        "rc": 0, "reward": 1.0, "resolved": 1.0, "verify_returncode": 0,
    })
    summary = wf._run_task_cascade(
        inst, args=args, run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "diagnostic"),
        cost=CostAccumulator(),
    )
    assert summary["status"] == "resolved"
    assert summary["reward"] == 1.0
    assert summary["rankable"] is summary["promotable"] is False
    corpus = Corpus(tmp_path / "s" / "instances" / inst.task_id / "corpus")
    spec = corpus.latest_spec()
    event = corpus.get_record(summary["hidden_verifier_lifecycle_ref"], as_type=LifecycleEvent)
    assert event.kind == "diagnostic_completed"
    measurement = corpus.get_measurement(summary["hidden_verifier_measurement_ref"])
    assert measurement.values["reward"] == 1.0
    assert measurement.values["rankable"] == 0.0
    assert measurement.values["promotable"] == 0.0
    raw = json.loads(corpus.get_artifact(measurement.raw_output_ref))
    assert raw["checkpoint"]["diagnostic_cost_policy"] == DIAGNOSTIC_COST_POLICY
    assert raw["checkpoint"]["diagnostic_only"] is True
    assert raw["checkpoint"]["status_policy"] == "diagnostic_only"
    assert not any(corpus.get_record(ref, as_type=LifecycleEvent).kind == "landed" for ref in spec.lifecycle_event_refs)
    record = wf._validate_lifecycle_record(wf._lifecycle_record(summary), line_number=1)
    assert record["reward"] == 1.0
    assert not wf._checkpoint_is_reusable(record, args=args)
    strict_args = copy.copy(args)
    strict_args.diagnostic_cost_policy = None
    assert not wf._checkpoint_is_reusable(record, args=strict_args)
    checkpoint = wf._load_hidden_oracle_checkpoint(args.spec, inst)
    assert checkpoint["diagnostic_only"] is True
    with pytest.raises(wf._LifecycleCheckpointError, match="prior oracle"):
        wf._execute_task_with_checkpoint(
            inst, args=strict_args, run_dir=tmp_path / "never", cost=CostAccumulator(),
            child_budget_usd=12, handle=None, index=0,
        )
    contradictory = dict(record, rankable=True)
    with pytest.raises(ValueError, match="diagnostic lifecycle"):
        wf._validate_lifecycle_record(contradictory, line_number=1)
