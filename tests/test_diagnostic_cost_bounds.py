# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""An explicit cost interval grants budget authority, never exact accounting."""

from dataclasses import replace

import pytest

from assay import Usage, UsageCostBounds, UsageCostBoundSegment, UsagePricingEvidence, UsagePricingSegment
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY, valid_diagnostic_cost_bounds
from crucible.common.cost import CostAccumulator, CostLimitExceeded


def bounded_usage():
    bounds = UsageCostBounds((UsageCostBoundSegment(
        input_tokens=126, output_tokens=74, cache_creation_tokens=0,
        cache_read_tokens=None, lower_bound_usd=0.0007652, upper_bound_usd=0.000992,
        policy=DIAGNOSTIC_COST_POLICY,
    ),), coverage_complete=True)
    return Usage(126, 74, bounds.lower_bound_usd, cost_bounds=bounds)


def record(costs, value=None):
    costs.record(role="planner", backend_name="kimi", model="kimi-k3.1-pro", usage=value or bounded_usage())


def test_explicit_policy_debits_upper_bound_and_retains_lower_bound_and_debt():
    costs = CostAccumulator(max_cost_usd=1.0, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    record(costs)
    assert costs.total_cost_usd == pytest.approx(0.0007652)
    assert costs.budget_cost_usd == pytest.approx(0.000992)
    assert costs.remaining() == pytest.approx(1.0 - 0.000992)
    assert costs.diagnostic_budget_covered
    summary = costs.to_dict()
    assert summary["accounting_debt"] is True
    assert summary["rollup_incomplete"] is True
    assert summary["promotion_evidence"] is False
    assert summary["cost_upper_bound_usd"] == pytest.approx(0.000992)
    assert summary["total_usd"] == costs.total_cost_usd
    assert summary["remaining_usd"] == costs.remaining()
    assert costs.cost_bounds == bounded_usage().cost_bounds


def test_default_policy_does_not_gain_spend_authority_from_bound():
    costs = CostAccumulator(max_cost_usd=1.0)
    record(costs)
    assert costs.remaining() == 0.0
    assert not costs.diagnostic_budget_covered
    assert "diagnostic_cost_policy" not in costs.to_dict()


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("gap", ["debt", "unloaded", "unknown-call"])
def test_later_bounds_cannot_clear_uncovered_history(before, gap):
    costs = CostAccumulator(max_cost_usd=1.0, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    if before:
        record(costs)
    if gap == "debt":
        costs.note_accounting_debt(role="earlier", missing=("cost_complete",))
    elif gap == "unloaded":
        costs.note_unloaded_ref()
    else:
        record(costs, Usage())
    if not before:
        record(costs)
    assert not costs.diagnostic_budget_covered
    assert costs.remaining() == 0.0
    assert costs.to_dict()["cost_upper_bound_usd"] is None


def test_upper_bound_overshoot_halts_even_when_lower_bound_fits():
    costs = CostAccumulator(max_cost_usd=0.0008, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    with pytest.raises(CostLimitExceeded):
        record(costs)
    assert costs.remaining() == 0.0
    assert costs.total_cost_usd < 0.0008 < costs.budget_cost_usd


@pytest.mark.parametrize("change", [
    {"policy": "unapproved"}, {"lower_bound_usd": 0.0}, {"upper_bound_usd": 0.0009},
])
def test_formula_or_policy_drift_is_not_coverage(change):
    value = bounded_usage()
    segment = replace(value.cost_bounds.segments[0], **change)
    bounds = UsageCostBounds((segment,), True)
    value = replace(value, cost_usd=bounds.lower_bound_usd, cost_bounds=bounds)
    assert not valid_diagnostic_cost_bounds(value)
    costs = CostAccumulator(max_cost_usd=1.0, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    record(costs, value)
    assert costs.remaining() == 0.0


def test_policy_without_cap_never_returns_uncapped_authority():
    costs = CostAccumulator(diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    record(costs)
    assert costs.remaining() == 0.0


@pytest.mark.parametrize("field,value", [
    ("input_tokens", 126.0), ("output_tokens", "74"),
    ("cache_creation_tokens", False), ("cost_usd", "0.0007652"),
])
def test_legacy_numeric_coercion_cannot_grant_diagnostic_budget_authority(field, value):
    malformed = replace(bounded_usage(), **{field: value})
    assert not valid_diagnostic_cost_bounds(malformed)
    costs = CostAccumulator(max_cost_usd=1.0, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    record(costs, malformed)
    assert not costs.diagnostic_budget_covered
    assert costs.remaining() == 0.0


def test_exact_pricing_and_interval_history_reconcile_the_same_budget_total():
    exact = UsagePricingEvidence((UsagePricingSegment(5, 6, 0, 0, 0.1),), True)
    costs = CostAccumulator(max_cost_usd=1.0, diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY)
    costs.record(role="exact", backend_name="test", usage=exact.usage, pricing_evidence=exact)
    record(costs)
    assert costs.diagnostic_budget_covered
    assert costs.cost_bounds.coverage_complete
    assert costs.cost_bounds.upper_bound_usd == costs.budget_cost_usd
    assert costs.to_dict()["cost_bounds_by_call"][0]["cost_bounds"] is not None
