# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Standalone Plan persists actual accounting on normal and exceptional exits."""

from contextlib import contextmanager
from dataclasses import replace
import json

import pytest

from assay import (
    ExecutionCompleteness, ImplementationDirective, InvocationResult,
    PlanOutcome, Slice, Spec, Tier, TierDAG, Usage,
    UsagePricingEvidence, UsagePricingSegment,
)
from assay.corpus import Corpus

from crucible.common.cost import CostAccumulator
from crucible.workflows import plan


def _result(*, complete=True):
    return InvocationResult(
        final_message="fixture", transcript=(), stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=2, cost_usd=0.1),
        completeness=ExecutionCompleteness(
            token_usage_complete=complete, cost_complete=complete,
            terminal_complete=True, trace_complete=False,
        ),
        pricing_evidence=UsagePricingEvidence(
            segments=(UsagePricingSegment(
                input_tokens=10, output_tokens=2, cache_creation_tokens=0,
                cache_read_tokens=0, cost_usd=0.1,
            ),), complete=complete,
        ),
    )


def _install(tmp_path, monkeypatch, *, status="accepted", incomplete=False, raises=False):
    spec_dir = tmp_path / "spec"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    corpus = Corpus(spec_dir)
    corpus.put_spec(Spec.from_yaml("""
schema_version: 1
project_context:
  project: accounting-test
  repo_url: https://example.invalid/fixture
  base_commit: '0000000000000000000000000000000000000000'
  branch: null
  submodule_path: null
  runners: {}
intent: Persist actual Plan accounting.
constraints: []
done_criteria: []
"""))

    class Backend:
        name = "codex"
        model = "fixture-priced-model"

        def invoke(self, _invocation):
            return _result(complete=not incomplete)

    class Pool:
        def get(self, *_args, **_kwargs):
            return Backend()

    @contextmanager
    def pool(*_args, **_kwargs):
        yield Pool()

    def run_plan(spec, **kwargs):
        results = [kwargs[role].invoke(object()) for role in (
            "planner_backend", "critic_backend", "judge_backend",
        )]
        exact_fixture_cost = sum(result.usage.cost_usd for result in results)
        if raises:
            raise RuntimeError("post-provider publication failed")
        dag = TierDAG(tiers=(Tier(tier_id="t0", depends_on=(), slices=(
            Slice(id="s0", directive=ImplementationDirective(
                slice_id="s0", intent="Create the implementation.",
                creates_paths=("created.txt",), files_in_scope=(),
                requirement_refs=("Persist actual Plan accounting.",),
                verify_mode="tests", blast_radius={"max_files": 1},
            )),
        )),))
        return PlanOutcome(
            spec=replace(spec, tier_dag=dag if status == "accepted" else None),
            candidate_dag=dag, candidate_ref=None,
            decision=status, status=status, rounds_used=1,
            final_critic_decision="pass", final_judge_decision="pass",
            reason="fixture", cost_usd=exact_fixture_cost,
            input_tokens=30, output_tokens=6,
            cache_creation_tokens=0, cache_read_tokens=0,
            execution_completeness=results[0].completeness,
            pricing_evidence=UsagePricingEvidence(
                segments=tuple(segment for result in results for segment in result.pricing_evidence.segments),
                complete=not incomplete,
            ),
        )

    monkeypatch.setattr(plan, "backend_pool", pool)
    monkeypatch.setattr(plan, "run_plan", run_plan)
    monkeypatch.setattr(plan, "bind_plan_contract", lambda _corpus, staged, **_kwargs: staged)
    monkeypatch.setattr(plan, "plan_contract_fingerprint", lambda *_args, **_kwargs: "sha256:" + "a" * 64)
    return spec_dir, ["--spec", str(spec_dir), "--cwd", str(workspace), "--no-validate-gate"]


@pytest.mark.parametrize("status, code", [("accepted", 0), ("rejected", 2), ("inconclusive", 3)])
def test_terminal_plan_persists_actual_role_ledger(tmp_path, monkeypatch, status, code):
    spec_dir, args = _install(tmp_path, monkeypatch, status=status)
    assert plan.main(["--planner-count", "1", *args]) == code
    (run,) = (spec_dir / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text())
    result = json.loads((run / "result.json").read_text())
    assert summary["status"] == status
    assert summary["costs"]["total_usd"] == pytest.approx(result["cost_usd"])
    assert summary["costs"]["total_usd"] == pytest.approx(0.3)
    assert summary["costs"]["accounting_debt"] is False
    assert summary["costs"]["rollup_incomplete"] is False
    assert set(summary["costs"]["by_role"]) == {"plan_planner", "plan_critic", "plan_judge"}
    assert len(summary["costs"]["pricing_evidence_by_call"]) == 3
    assert summary["plan_outcome_accounting"]["execution_completeness"]["trace_complete"] is False
    assert summary["plan_outcome_accounting"]["pricing_evidence_complete"] is True


def test_missing_returned_usage_flags_remain_diagnostic(tmp_path, monkeypatch):
    spec_dir, args = _install(tmp_path, monkeypatch, incomplete=True)
    # The stub's semantic disposition is intentionally unchanged; the actual
    # wrapper ledger independently prevents a false complete-accounting claim.
    assert plan.main(["--planner-count", "1", *args]) == 0
    (run,) = (spec_dir / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["rollup_incomplete"] is True
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["plan_outcome_accounting"]["execution_completeness"]["cost_complete"] is False


def test_exception_preserves_complete_actual_prior_call_ledger(tmp_path, monkeypatch):
    spec_dir, args = _install(tmp_path, monkeypatch, raises=True)
    with pytest.raises(RuntimeError, match="publication failed"):
        plan.main(["--planner-count", "1", *args])
    (run,) = (spec_dir / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["costs"]["total_usd"] == pytest.approx(0.3)
    assert summary["costs"]["accounting_debt"] is False
    assert summary["costs"]["pricing_evidence"]["complete"] is True


def test_raised_provider_call_retains_lower_bound_with_debt():
    class Backend:
        name = "fixture"
        model = "fixture"

        def invoke(self, _invocation):
            raise RuntimeError("provider did not return usage")

    costs = CostAccumulator()
    backend = plan._CompletedPlanCallBackend(Backend(), role="plan_planner", costs=costs)
    with pytest.raises(RuntimeError, match="did not return usage"):
        backend.invoke(object())
    failure = plan._plan_failure_accounting(costs, 0.0)
    assert failure.status == "failed"
    assert failure.summary_fields["costs"]["accounting_debt"] is True
    assert failure.summary_fields["cost_usd_is_lower_bound"] is True


def test_scalar_cost_mismatch_cannot_claim_complete_ledger():
    summary = plan._plan_accounting_summary(
        {"status": "accepted"}, costs=CostAccumulator(), expected_cost_usd=2,
    )
    assert summary["status"] == "accepted"
    assert summary["costs"]["total_usd"] == 0
    assert summary["costs"]["accounting_debt"] is True


def test_sub_microdollar_scalar_cost_mismatch_still_creates_debt():
    costs = CostAccumulator()
    summary = plan._plan_accounting_summary(
        {"status": "accepted"},
        costs=costs,
        expected_cost_usd=0.0000005,
    )
    assert summary["costs"]["total_usd"] == 0
    assert summary["costs"]["accounting_debt"] is True
