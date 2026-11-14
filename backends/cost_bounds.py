# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Explicit K3 diagnostic budget policy; never exact billing authority."""

from __future__ import annotations

import math
from typing import Any

from assay import UsageCostBounds

from .pricing import DEFAULT_PRICING


DIAGNOSTIC_COST_POLICY = "kimi-k3-cache-read-upper-bound-v1"
EXACT_COST_BOUND_POLICY = "observed-exact-cost-v1"


def valid_diagnostic_cost_bounds(usage: Any, bounds: UsageCostBounds | None = None) -> bool:
    """Validate typed full coverage and independently replay the bound formula."""

    if bounds is None:
        bounds = getattr(usage, "cost_bounds", None)
    if type(bounds) is not UsageCostBounds or not bounds.segments or not bounds.covers(usage):
        return False
    saw_k3 = False
    for segment in bounds.segments:
        if segment.policy == EXACT_COST_BOUND_POLICY:
            if segment.cache_read_tokens is None or segment.lower_bound_usd != segment.upper_bound_usd:
                return False
            continue
        if segment.policy != DIAGNOSTIC_COST_POLICY:
            return False
        saw_k3 = True
        known_read = segment.cache_read_tokens
        maximum_read = (
            segment.input_tokens - segment.cache_creation_tokens
            if known_read is None else known_read
        )
        pricing = DEFAULT_PRICING["kimi-k3"]
        lower = pricing.compute_cost(
            input_tokens=segment.input_tokens, output_tokens=segment.output_tokens,
            cache_creation_tokens=segment.cache_creation_tokens,
            cache_read_tokens=maximum_read,
        )
        upper = pricing.compute_cost(
            input_tokens=segment.input_tokens, output_tokens=segment.output_tokens,
            cache_creation_tokens=segment.cache_creation_tokens,
            cache_read_tokens=0 if known_read is None else known_read,
        )
        if not (
            math.isclose(segment.lower_bound_usd, lower, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(segment.upper_bound_usd, upper, rel_tol=1e-12, abs_tol=1e-12)
        ):
            return False
    return saw_k3


def diagnostic_accounting_covered(value: Any) -> bool:
    """Evidence predicate only: callers must still bind the explicit policy."""

    usage = getattr(value, "usage", value)
    bounds = getattr(value, "cost_bounds", None) or getattr(usage, "cost_bounds", None)
    completeness = getattr(value, "completeness", None)
    if completeness is None:
        completeness = getattr(value, "execution_completeness", None)
    return (
        valid_diagnostic_cost_bounds(usage, bounds)
        and getattr(completeness, "terminal_complete", False) is True
        and getattr(completeness, "trace_complete", False) is True
    )
