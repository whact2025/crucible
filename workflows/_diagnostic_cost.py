# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Explicit diagnostic budget continuation, never exact-cost admission.

Accounting debt remains on every source record. These predicates authorize
only bounded diagnostic computation when independently validated upper bounds
cover that debt. They must not replace a promotion or rankability check.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from assay import Usage, UsageCostBounds

from crucible.backends.cost_bounds import (
    DIAGNOSTIC_COST_POLICY,
    diagnostic_accounting_covered,
    valid_diagnostic_cost_bounds,
)


def validate_diagnostic_cost_policy(
    policy: str | None,
    *,
    limit_mode: str | None,
    budget_usd: float | None,
) -> None:
    if policy is None:
        return
    if policy != DIAGNOSTIC_COST_POLICY:
        raise ValueError("unknown diagnostic cost policy")
    if limit_mode != "diagnostic":
        raise ValueError("diagnostic cost policy requires diagnostic backend limits")
    if (
        type(budget_usd) not in (int, float)
        or not math.isfinite(budget_usd)
        or budget_usd <= 0
    ):
        raise ValueError("diagnostic cost policy requires a finite positive USD budget")


def diagnostic_call_covered(value: Any, *, policy: str | None) -> bool:
    """A typed returned call may continue only under the exact opt-in."""

    return policy == DIAGNOSTIC_COST_POLICY and diagnostic_accounting_covered(value)


def accounting_blocks_paid_work(costs: Any) -> bool:
    """Keep raw debt visible while distinguishing covered diagnostic debt."""

    summary = costs.to_dict()
    incomplete = bool(summary.get("accounting_debt") or summary.get("unloaded_refs"))
    return incomplete and not (
        getattr(costs, "diagnostic_cost_policy", None) == DIAGNOSTIC_COST_POLICY
        and costs.diagnostic_budget_covered
    )


def diagnostic_budget_debit(value: Any, *, policy: str | None) -> float:
    """Return a spending debit, never a replacement for observed exact cost."""

    usage = getattr(value, "usage", value)
    bounds = getattr(value, "cost_bounds", getattr(usage, "cost_bounds", None))
    if policy == DIAGNOSTIC_COST_POLICY and valid_diagnostic_cost_bounds(usage, bounds):
        return float(bounds.upper_bound_usd)
    return usage.cost_usd


def diagnostic_summary_bounds(
    summary: Mapping[str, Any], *, policy: str | None,
) -> UsageCostBounds | None:
    """Recompute a closed child's bound ledger; report labels grant nothing.

    The caller still owns child identity, terminal, cleanup and writer proofs.
    This narrow successor requires a bound on every logical call and deliberately
    does not recover missing call records from aggregate totals.
    """

    if policy != DIAGNOSTIC_COST_POLICY or summary.get("diagnostic_cost_policy") != policy:
        return None
    if summary.get("diagnostic_budget_covered") is not True:
        return None
    if type(summary.get("unloaded_refs")) is not int or summary["unloaded_refs"] != 0:
        return None
    calls = summary.get("cost_bounds_by_call")
    roles = summary.get("by_role")
    if type(calls) is not list or not calls or type(roles) is not dict:
        return None
    expected_calls = 0
    for slot in roles.values():
        if type(slot) is not dict or type(slot.get("calls")) is not int or slot["calls"] < 0:
            return None
        expected_calls += slot["calls"]
    if len(calls) != expected_calls:
        return None
    segments = []
    for call in calls:
        if type(call) is not dict or call.get("coverage_complete") is not True:
            return None
        try:
            bounds = UsageCostBounds.from_record(call.get("cost_bounds"))
        except (TypeError, ValueError, AttributeError):
            return None
        usage = Usage(
            input_tokens=sum(segment.input_tokens for segment in bounds.segments),
            output_tokens=sum(segment.output_tokens for segment in bounds.segments),
            cache_creation_tokens=sum(segment.cache_creation_tokens for segment in bounds.segments),
            cache_read_tokens=sum(segment.cache_read_tokens or 0 for segment in bounds.segments),
            cost_usd=bounds.lower_bound_usd,
            cost_bounds=bounds,
        )
        debit = call.get("budget_debit_usd")
        if (
            not valid_diagnostic_cost_bounds(usage, bounds)
            or type(debit) not in (int, float)
            or not math.isfinite(debit)
            or not math.isclose(debit, bounds.upper_bound_usd, rel_tol=1e-12, abs_tol=1e-12)
        ):
            return None
        segments.extend(bounds.segments)
    aggregate = UsageCostBounds(segments=tuple(segments), coverage_complete=True)
    usage = Usage(
        input_tokens=summary.get("total_input_tokens"),
        output_tokens=summary.get("total_output_tokens"),
        cache_creation_tokens=summary.get("total_cache_creation_tokens", 0),
        cache_read_tokens=summary.get("total_cache_read_tokens", 0),
        cost_usd=summary.get("total_usd"),
        cost_bounds=aggregate,
    )
    return aggregate if valid_diagnostic_cost_bounds(usage, aggregate) else None
