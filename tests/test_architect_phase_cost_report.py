# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Planning receipts survive early termination without reopening authority."""

from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest

from assay import ExecutionCompleteness, ReviewCoverage, Usage, UsageCostBounds, UsageCostBoundSegment, Verdict
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY
from crucible.backends.pricing import DEFAULT_PRICING
from crucible.common.cost import CostAccumulator
from crucible.workflows._diagnostic_cost import diagnostic_summary_bounds
from crucible.workflows.architect import _authoritative_cost_summary


def _usage(counters):
    segments = []
    for prompt, output, write, read in counters:
        pricing = DEFAULT_PRICING["kimi-k3"]
        segments.append(UsageCostBoundSegment(
            input_tokens=prompt, output_tokens=output,
            cache_creation_tokens=write, cache_read_tokens=read,
            lower_bound_usd=pricing.compute_cost(
                input_tokens=prompt, output_tokens=output,
                cache_creation_tokens=write,
                cache_read_tokens=prompt - write if read is None else read,
            ),
            upper_bound_usd=pricing.compute_cost(
                input_tokens=prompt, output_tokens=output,
                cache_creation_tokens=write, cache_read_tokens=read or 0,
            ),
            policy=DIAGNOSTIC_COST_POLICY,
        ))
    bounds = UsageCostBounds(tuple(segments), coverage_complete=True)
    return Usage(
        input_tokens=sum(s.input_tokens for s in segments),
        output_tokens=sum(s.output_tokens for s in segments),
        cache_creation_tokens=sum(s.cache_creation_tokens for s in segments),
        cache_read_tokens=sum(s.cache_read_tokens or 0 for s in segments),
        cost_usd=bounds.lower_bound_usd, cost_bounds=bounds,
    )


def _ledgers():
    planning = CostAccumulator(max_cost_usd=4.8, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    # Numeric-only replay of the two rejected pilot planner calls (5 + 1
    # generation segments), with no retained response or task material.
    for counters in (
        ((4488, 216, 4352, None), (5187, 177, 768, 4352),
         (7936, 259, 2816, 5120), (10052, 369, 2048, 7936),
         (11190, 2387, 1024, 9984)),
        ((6186, 2843, 2816, 3328),),
    ):
        planning.record(role="architect_planner", usage=_usage(counters),
                        backend_name="kimi", model="kimi-k3")
    records = CostAccumulator(diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    for _ in range(2):
        records.record(
            role="candidate_validation", backend_name="local",
            usage=Verdict(
                rubric="candidate_validation", kind="validate", decision="reject",
                findings=(), artifact_ref="sha256:" + "0" * 64, artifact_commit=None,
                reasoning="", coverage=ReviewCoverage(
                    complete=True, sources_reviewed=("artifact",), omissions=(),
                    artifact_chars_total=1, artifact_chars_reviewed=1,
                ),
                backend_name="local", transcript_ref=None, cost_usd=0.0, execution=None,
                execution_completeness=ExecutionCompleteness(
                    terminal_complete=True, trace_complete=True,
                    cost_complete=True, token_usage_complete=True,
                ),
            ),
        )
    return records, planning


def _summary(records, planning):
    return _authoritative_cost_summary(
        records, budget_usd=12, spent_usd=planning.total_cost_usd,
        planning_usage=planning,
    )


def test_early_rejected_plan_retains_validated_phase_bounds_only():
    records, planning = _ledgers()
    original_records = copy.deepcopy(records.to_dict())
    original_planning = copy.deepcopy(planning.to_dict())
    summary = json.loads(json.dumps(_summary(records, planning)))
    evidence = summary["planning_usage"]["diagnostic_cost_evidence"]
    bounds = UsageCostBounds.from_record(evidence["cost_bounds"])
    assert bounds == diagnostic_summary_bounds(evidence, policy=DIAGNOSTIC_COST_POLICY)
    assert [len(call["cost_bounds"]["segments"]) for call in evidence["cost_bounds_by_call"]] == [5, 1]
    assert bounds.segments[0].cache_read_tokens is None
    assert evidence["cost_lower_bound_usd"] == pytest.approx(.1455708)
    assert evidence["cost_upper_bound_usd"] == pytest.approx(.145938)
    assert evidence["budget_cost_usd"] == pytest.approx(.145938)
    assert evidence["cost_bounds_by_call"][0]["budget_debit_usd"] == pytest.approx(.0937206)
    assert evidence["actual_cost_usd"] is None
    assert evidence["accounting_debt"] is True
    assert evidence["by_role"]["architect_planner"]["calls"] == 2
    assert "candidate_validation" not in evidence["by_role"]
    assert evidence["future_spend_authority"] is False
    assert evidence["remaining_usd"] == 0
    assert evidence["promotion_evidence"] is False
    # Existing parent aggregate remains closed and byte-shape compatible.
    assert summary["diagnostic_budget_covered"] is False
    assert summary["budget_cost_usd"] == 0
    assert summary["cost_upper_bound_usd"] is None
    assert summary["remaining_usd"] == 0
    assert summary["promotion_evidence"] is False
    assert diagnostic_summary_bounds(summary, policy=DIAGNOSTIC_COST_POLICY) is None
    # Reporting does not mutate even the active phase accumulator.
    assert planning.remaining() == pytest.approx(4.8 - .145938)
    assert records.to_dict() == original_records
    assert planning.to_dict() == original_planning


@pytest.mark.parametrize("damage", ["debit", "missing_call", "anonymous_debt"])
def test_unvalidated_phase_report_never_claims_upper_coverage(damage):
    records, planning = _ledgers()
    if damage == "anonymous_debt":
        planning.note_accounting_debt(role="uncovered", missing=("cost_complete",))
    raw = copy.deepcopy(planning.to_dict())
    if damage == "debit":
        raw["cost_bounds_by_call"][0]["budget_debit_usd"] = 0
    elif damage == "missing_call":
        raw["cost_bounds_by_call"].pop()
    with patch.object(planning, "to_dict", return_value=raw):
        evidence = _summary(records, planning)["planning_usage"]["diagnostic_cost_evidence"]
    for name in ("cost_bounds", "budget_cost_usd", "cost_lower_bound_usd",
                 "cost_upper_bound_usd", "actual_cost_usd"):
        assert evidence[name] is None
    assert evidence["diagnostic_budget_covered"] is False
    assert evidence["future_spend_authority"] is False
    assert evidence["remaining_usd"] == 0


def test_ordinary_summary_shape_stays_unchanged():
    planning = CostAccumulator()
    planning.record(role="architect_planner", usage=Usage(cost_usd=.1), backend_name="codex")
    summary = _summary(CostAccumulator(), planning)
    assert "diagnostic_cost_evidence" not in summary["planning_usage"]
    assert "diagnostic_cost_policy" not in summary
    assert summary["remaining_usd"] == pytest.approx(11.9)
