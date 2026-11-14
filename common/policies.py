# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Typed policies consumed by Crucible workflow orchestration.

Workflows compose breach and review policies from CLI flags at ``main()``
entry and pass them through to ``run_tier`` / ``run_architect``. A spec's
nested policy blocks may override those defaults.

Policies live in ``crucible/`` (not ``assay/``) because they describe
workflow-layer orchestration, not primitive-layer behavior — they
reference crucible-specific concepts (breach loop, review rubrics)
that assay knows nothing about.

Loading order (highest precedence first):
  1. Explicit kwarg to ``run_tier`` / ``run_architect`` (per-call override)
  2. ``spec.yaml::{breach,review}_policy:`` block
  3. Dataclass default (this module)

Scope discipline
----------------
Every field in this module has a workflow consumer. Add a policy field only
alongside its workflow and loader wiring so configuration cannot be accepted
without affecting execution.

Currently-consumed surface
--------------------------
- ``BreachPolicy``: ``enabled``, ``max_rounds``,
  ``persistence_veto_threshold``, ``breaker_rubric``,
  ``judge_rubric``, ``require_attack_authority`` — read by
  ``workflows/tier.py``'s breach pass.
- ``ReviewPolicy``: ``rubrics``, ``advisory_rubrics``, and ``retry_budget`` —
  read by ``workflows/tier.py``'s review pass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BreachPolicy:
    """Breaker → Defender → Judge loop knobs.

    Composed from CLI flags at ``main()`` entry: ``--include-breach``
    (architect-level) + ``--skip-breach`` (tier-level) translate into
    ``BreachPolicy.enabled``; ``--max-breach-rounds`` /
    ``--persistence-veto-threshold`` map field-for-field. Per-call
    data (``--breach-tests``, ``--breach-issue-text``) stays as
    direct kwargs on ``run_tier`` / ``run_architect`` — it's
    instance-specific, not policy.
    """

    # Generic workflows opt in with --include-breach. Benchmark suite
    # policies enable adversarial hardening by default.
    enabled: bool = False

    # Cap on Breaker → Defender → Judge rounds before the breach
    # loop terminates with whatever findings have accumulated.
    max_rounds: int = 5

    # Mirror of PlanPolicy.persistence_veto_threshold for the breach
    # debate. A finding surviving this many Breaker rounds vetos.
    persistence_veto_threshold: int = 3

    # Per-spec rubric overrides — None means use crucible's bundled
    # rubrics (impl_breaker / breach_judge). SWE-bench supplies
    # alternatives via ``SWEBENCH_BREACH_POLICY``.
    breaker_rubric: str | None = None
    judge_rubric: str | None = None

    # Require a hash-bound, independent admission decision for every
    # Breaker-generated attack before the breach workflow may write an attack
    # test or mutate production code. This is a generic mutation-safety
    # invariant; products should not need to opt into it by convention.
    require_attack_authority: bool = True


@dataclass(frozen=True)
class ReviewPolicy:
    """Per-tier rubric review pass knobs.

    ``--skip-review`` (tier-level) gates the whole pass; ``rubrics``
    selects which of the supported set actually runs. Consumed by
    ``workflows/tier.py::run_tier(review_policy=...)`` and forwarded
    from ``workflows/architect.py::run_architect(review_policy=...)``.
    """

    # The set of rubric names that the tier review pass runs. Must be a
    # subset of ``workflows/tier.py::_SUPPORTED_REVIEW_RUBRICS``;
    # unknown names raise at run_tier entry.
    #
    # Product-specific trims live next to the product's spec
    # synthesizer (e.g., ``products/swebench/spec_synthesis.py``
    # exports ``SWEBENCH_REVIEW_POLICY`` which drops ``spec_clarity``
    # — SWE-bench specs are dataset-authored, ``spec_clarity`` would
    # produce guaranteed false negatives on the verbatim GitHub
    # ``problem_statement`` intent text).
    rubrics: tuple[str, ...] = (
        "diff_scope", "spec_clarity", "tier_completeness",
    )

    # Rubrics whose findings are RECORDED but do NOT fail the tier — no
    # ``review_failed`` / ``abandoned`` lifecycle / ``rc=3``. ``diff_scope``
    # and ``tier_completeness`` review the tier's *work product* (the union
    # diff) and stay gating; ``spec_clarity`` reviews the spec's *authoring*
    # quality, a design-stage concern that shouldn't block landing work the
    # diff/completeness gates already cleared. Advisory findings are still
    # persisted as verdicts for the author. A landed tier (rc=0) thus
    # reflects sound work, so a driver can auto-advance on rc=0 / lifecycle
    # ``landed`` without an advisory prose nit stalling the pipeline. Must
    # be a subset of ``rubrics``.
    advisory_rubrics: tuple[str, ...] = ("spec_clarity",)

    # Review-feedback retry budget. When a GATING rubric (a non-advisory
    # ``rubrics`` entry, e.g. ``tier_completeness`` / ``diff_scope``)
    # fails, re-implement the tier's slice(s) with the reviewer's
    # findings as feedback and re-review — up to this many times before
    # the tier abandons (``rc=3``). Mirrors ``run_tier(build_retries=...)``
    # for the build/verify gate. Default 0 = OFF (review failure is
    # immediately terminal, preserving prior behavior for every product
    # that doesn't opt in). DeepSWE/SWE-bench-style multi-tier plans
    # set this > 0 so one incomplete slice doesn't abandon the whole
    # task when the reviewer handed back an actionable fix.
    retry_budget: int = 0


@dataclass(frozen=True)
class CruciblePolicies:
    """Aggregate of policies currently wired through the workflow layer.

    Only consumed policy groups belong here; add a group together with its
    workflow and YAML-loader integration.

    Idiomatic-Python note: the field defaults are class instances
    (``BreachPolicy()``) rather than ``field(default_factory=...)``.
    Safe here because each inner policy is itself ``frozen=True`` —
    every ``CruciblePolicies()`` shares the same default-instance
    refs but they're immutable, so cross-instance state-bleed
    isn't possible. The factory pattern would be needed if the
    inner policies were mutable; here it'd be cargo-cult.
    """

    breach: BreachPolicy = BreachPolicy()
    review: ReviewPolicy = ReviewPolicy()


# Public surface
__all__ = [
    "BreachPolicy",
    "ReviewPolicy",
    "CruciblePolicies",
]
