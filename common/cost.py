# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cumulative cost accounting with per-role / per-backend / per-model breakdown.

Architectural contract: backends own ``Usage.cost_usd`` and vendor-specific
usage reconciliation. Claude consumes usage from its selected CLI or direct
API; Codex and the OpenAI-compatible adapter reconcile provider usage with
the explicit pricing table. Every accepted model has a pricing entry,
including deliberately zero-cost models; an unknown model is a configuration
error. This module aggregates the reported usage, costs, pricing evidence,
and accounting gaps. Vendor-specific cost repair belongs in the backend.

What the accumulator gives you that summing ``verdict.cost_usd`` doesn't:

- Per-role attribution. ``goal_red`` cost vs ``goal_judge`` cost shows
  where the dollars went; raw totals hide that.
- Per-backend split. A mixed ``planner → claude`` + ``goal_judge → codex``
  run reports each separately so operators can attribute spend to the
  right vendor invoice.
- Per-model breakdown when the same role hits different models across
  rounds.
- Optional cost cap that raises ``CostLimitExceeded`` on the call that
  trips it, halting mid-pipeline before the next backend invocation.

Workflow integration shape::

    accum = CostAccumulator(max_cost_usd=args.budget_usd)
    ...
    verdict = review(artifact, rubric, backend=red_backend, corpus=corpus)
    accum.record(
        role="goal_red", usage=verdict.usage,
        backend_name=red_backend.name, model=red_backend.model,
    )
    ...
    summary["costs"] = accum.to_dict()
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

from assay import Usage, UsageCostBoundSegment, UsageCostBounds, UsagePricingEvidence, UsagePricingSegment, combine_cost_bounds

from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY, EXACT_COST_BOUND_POLICY, valid_diagnostic_cost_bounds


PRICING_EVIDENCE_SCHEMA = "crucible.usage-pricing-ledger.v1"


def parse_token_count(value: Any) -> int | None:
    """Parse an untrusted child-summary token count without truncation.

    JSON child envelopes should contain non-negative integers.  Accept a
    decimal integer string for forwarder compatibility, but reject booleans,
    negative values, fractional floats, NaN/Inf, and container/object values.
    ``None`` means the field is absent or untrustworthy; zero remains a valid
    authoritative count.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        rendered = value.strip()
        if not rendered:
            return None
        try:
            parsed = int(rendered, 10)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed >= 0 else None


class CostLimitExceeded(RuntimeError):
    """Raised by ``CostAccumulator.record`` when ``max_cost_usd`` trips.

    Callers that want soft-budget semantics (record-and-warn, no halt)
    should catch this and decide whether to keep going. The default
    pipeline pattern is to let it propagate so the workflow exits
    cleanly on budget exhaustion.
    """


@dataclass
class _RoleSlot:
    """Per-role rollup tracked inside the accumulator.

    Mutable on purpose — ``record()`` increments in place. The
    ``backends`` and ``models`` sets surface mixed-routing runs
    (same role hitting multiple backends or models across calls).
    """

    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    backends: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)

    def to_dict(self, *, exact_cost: bool = False) -> dict[str, Any]:
        result = {
            "cost_usd": self.cost_usd if exact_cost else round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "backends": sorted(self.backends),
            "models": sorted(self.models),
        }
        # Omit absent cache telemetry. Present counters remain inclusive
        # subsets of ``input_tokens`` under the Assay Usage contract.
        if self.cache_creation_tokens or self.cache_read_tokens:
            result["cache_creation_tokens"] = self.cache_creation_tokens
            result["cache_read_tokens"] = self.cache_read_tokens
        return result


@dataclass(frozen=True)
class _PricingCall:
    """One logical call's ordered pricing-accounting record."""

    role: str
    backend_name: str
    model: str | None
    usage: Usage
    pricing_evidence: UsagePricingEvidence | None


@dataclass
class CostAccumulator:
    """Aggregate cost + tokens across a workflow run, broken out per role.

    Thread-safe (every accessor + ``record`` is guarded by the
    internal lock) — workflows that fan out parallel calls (architect
    tiers, swebench instances) can share a single accumulator across
    threads.
    """

    max_cost_usd: float | None = None
    _total_cost_usd: float = 0.0
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0
    _total_cache_creation_tokens: int = 0
    _total_cache_read_tokens: int = 0
    _by_role: dict[str, _RoleSlot] = field(default_factory=dict)
    _by_backend: dict[str, float] = field(default_factory=dict)
    _by_model: dict[str, float] = field(default_factory=dict)
    _unloaded_refs: int = 0
    _accounting_debt_by_role: dict[str, set[str]] = field(
        default_factory=dict,
    )
    _pricing_segments: list[UsagePricingSegment] = field(default_factory=list)
    _pricing_calls: list[_PricingCall] = field(default_factory=list)
    _pricing_evidence_seen: bool = False
    _pricing_evidence_complete: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    diagnostic_cost_policy: str | None = None
    _budget_debit_usd: float = 0.0
    _diagnostic_bounds_seen: bool = False
    _diagnostic_coverage_complete: bool = True
    _unbounded_accounting_debt: bool = False
    _cost_bound_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.diagnostic_cost_policy not in {None, DIAGNOSTIC_COST_POLICY}:
            raise ValueError("unsupported diagnostic cost policy")
        if self.max_cost_usd is None:
            return
        cap = float(self.max_cost_usd)
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError(
                "max_cost_usd must be a finite, non-negative number"
            )
        self.max_cost_usd = cap

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        role: str,
        usage: Any,
        backend_name: str,
        model: str | None = None,
        pricing_evidence: UsagePricingEvidence | None = None,
        cost_bounds: UsageCostBounds | None = None,
    ) -> None:
        """Add ``usage`` from a backend call to the running totals.

        ``role`` is the per-call attribution string — typically the
        rubric name (``goal_red`` / ``breach_judge``) or workflow
        slot name (``implementer`` / ``planner``).
        ``backend_name`` comes from the backend instance's ``name``
        attribute (``claude`` / ``codex`` / ``openai_compat``). ``model`` is
        optional; when populated it surfaces in ``to_dict()``. Typed pricing
        evidence can be supplied explicitly or discovered on a workflow
        artifact. Once present, summaries include both a derived aggregate
        and an ordered logical-call ledger; missing evidence on any other call
        remains an explicit incomplete gap.

        Raises ``CostLimitExceeded`` AFTER recording the call when
        the cumulative total exceeds ``max_cost_usd``. The recording
        happens first so the workflow's summary.json accurately
        reflects spend up to and including the over-budget call.

        Backends populate ``Usage.cost_usd`` and reconcile their provider
        evidence before returning it. This layer records the resulting
        figures and completeness; it performs no vendor-specific cost repair.
        """
        if not isinstance(role, str) or not role:
            raise ValueError("cost role must be a non-empty string")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError("cost backend_name must be a non-empty string")
        if model is not None and (
            not isinstance(model, str) or not model
        ):
            raise ValueError("cost model must be a non-empty string or None")

        # ``usage`` is duck-typed; accept anything with ``cost_usd`` +
        # ``input_tokens`` + ``output_tokens`` attributes. ``Usage``,
        # ``InvocationResult.usage``, and adapter objects all match.
        discovered_evidence = getattr(usage, "pricing_evidence", None)
        if pricing_evidence is None:
            pricing_evidence = discovered_evidence
        if pricing_evidence is not None and (
            type(pricing_evidence) is not UsagePricingEvidence
        ):
            raise TypeError(
                "pricing_evidence must be a UsagePricingEvidence or None"
            )

        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        # Implementation and Verdict records may retain exact cache counters
        # only in pricing evidence rather than duplicating them as top-level
        # fields.  Derive an absent counter only from a complete ledger that
        # reconciles all aggregate dimensions.  An incomplete ledger can
        # contain partial or repeated snapshots whose cache subsets use a
        # different denominator from the durable aggregate usage; those
        # segments remain useful pricing evidence but cannot authoritatively
        # fill a top-level cache counter.
        evidence_usage = (
            pricing_evidence.usage if pricing_evidence is not None else None
        )
        raw_cache_creation = getattr(usage, "cache_creation_tokens", None)
        raw_cache_read = getattr(usage, "cache_read_tokens", None)
        # ``None`` means the typed artifact's field was absent on the wire,
        # not that the
        # backend authoritatively reported zero cache tokens.
        has_cache_creation = raw_cache_creation is not None
        has_cache_read = raw_cache_read is not None
        direct_cache_creation = int(raw_cache_creation or 0)
        direct_cache_read = int(raw_cache_read or 0)
        evidence_cache_creation = (
            evidence_usage.cache_creation_tokens
            if evidence_usage is not None
            else 0
        )
        evidence_cache_read = (
            evidence_usage.cache_read_tokens
            if evidence_usage is not None
            else 0
        )
        candidate_cache_creation = (
            direct_cache_creation
            if has_cache_creation
            else evidence_cache_creation
        )
        candidate_cache_read = (
            direct_cache_read
            if has_cache_read
            else evidence_cache_read
        )
        evidence_can_supply_cache = bool(
            pricing_evidence is not None
            and pricing_evidence.complete
            and pricing_evidence.reconciles(Usage(
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                cache_creation_tokens=candidate_cache_creation,
                cache_read_tokens=candidate_cache_read,
            ))
        )
        cache_creation_tokens = (
            direct_cache_creation
            if has_cache_creation
            else (
                evidence_cache_creation if evidence_can_supply_cache else 0
            )
        )
        cache_read_tokens = (
            direct_cache_read
            if has_cache_read
            else evidence_cache_read if evidence_can_supply_cache else 0
        )
        accounting_debt = tuple(
            getattr(usage, "accounting_debt", ()) or ()
        )
        if cost_bounds is None:
            cost_bounds = getattr(usage, "cost_bounds", None)
        if cost_bounds is not None and type(cost_bounds) is not UsageCostBounds:
            raise TypeError("cost_bounds must be UsageCostBounds or None")
        # A numeric Usage projection must not hide the absent cache subset.
        if cost_bounds is not None and any(
            segment.cache_read_tokens is None for segment in cost_bounds.segments
        ):
            accounting_debt = tuple(sorted(set(accounting_debt) | {
                "token_usage_complete", "cost_complete",
            }))
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(
                "backend usage cost_usd must be finite and non-negative"
            )
        if (
            in_tokens < 0
            or out_tokens < 0
            or cache_creation_tokens < 0
            or cache_read_tokens < 0
        ):
            raise ValueError("backend token usage must be non-negative")
        if cache_creation_tokens + cache_read_tokens > in_tokens:
            raise ValueError(
                "backend cache-token subsets must not exceed input tokens"
            )
        allowed_debt = {"token_usage_complete", "cost_complete"}
        if accounting_debt and (
            not isinstance(role, str) or not role.strip()
        ):
            raise ValueError(
                "backend accounting-debt role must be a non-empty string"
            )
        if any(field_name not in allowed_debt for field_name in accounting_debt):
            raise ValueError(
                "backend accounting debt must name token_usage_complete "
                "and/or cost_complete"
            )

        normalized_usage = Usage(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_bounds=cost_bounds,
        )
        evidence_complete = (
            pricing_evidence is not None
            and pricing_evidence.complete
            and not accounting_debt
            and pricing_evidence.reconciles(normalized_usage)
        )
        effective_evidence = (
            None
            if pricing_evidence is None
            else UsagePricingEvidence(
                segments=pricing_evidence.segments,
                complete=evidence_complete,
            )
        )

        bounded = (
            self.diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
            and valid_diagnostic_cost_bounds(usage, cost_bounds)
            and valid_diagnostic_cost_bounds(normalized_usage, cost_bounds)
        )
        completeness = getattr(usage, "execution_completeness", None)
        exact = not accounting_debt and (
            evidence_complete or bool(getattr(completeness, "complete", False))
        )
        if self.diagnostic_cost_policy is not None and exact and cost_bounds is None:
            cost_bounds = UsageCostBounds((UsageCostBoundSegment(
                input_tokens=in_tokens, output_tokens=out_tokens,
                cache_creation_tokens=cache_creation_tokens, cache_read_tokens=cache_read_tokens,
                lower_bound_usd=cost, upper_bound_usd=cost, policy=EXACT_COST_BOUND_POLICY,
            ),), coverage_complete=True)
            normalized_usage = Usage(
                input_tokens=in_tokens, output_tokens=out_tokens, cost_usd=cost,
                cache_creation_tokens=cache_creation_tokens, cache_read_tokens=cache_read_tokens,
                cost_bounds=cost_bounds,
            )
        budget_debit = cost_bounds.upper_bound_usd if bounded else cost

        with self._lock:
            slot = self._by_role.setdefault(role, _RoleSlot())
            slot.cost_usd += cost
            slot.input_tokens += in_tokens
            slot.output_tokens += out_tokens
            slot.cache_creation_tokens += cache_creation_tokens
            slot.cache_read_tokens += cache_read_tokens
            slot.calls += 1
            slot.backends.add(backend_name)
            if model:
                slot.models.add(model)

            self._by_backend[backend_name] = (
                self._by_backend.get(backend_name, 0.0) + cost
            )
            if model:
                self._by_model[model] = (
                    self._by_model.get(model, 0.0) + cost
                )

            self._total_cost_usd += cost
            self._budget_debit_usd += budget_debit
            self._diagnostic_bounds_seen |= bounded
            self._diagnostic_coverage_complete &= bounded or exact
            self._unbounded_accounting_debt |= bool(accounting_debt) and not bounded
            if self.diagnostic_cost_policy is not None:
                self._cost_bound_calls.append({
                    "role": role, "backend": backend_name, "model": model,
                    "cost_bounds": cost_bounds.to_record() if cost_bounds is not None else None,
                    "budget_debit_usd": budget_debit,
                    "coverage_complete": bounded or exact,
                })
            self._total_input_tokens += in_tokens
            self._total_output_tokens += out_tokens
            self._total_cache_creation_tokens += cache_creation_tokens
            self._total_cache_read_tokens += cache_read_tokens
            self._pricing_calls.append(
                _PricingCall(
                    role=role,
                    backend_name=backend_name,
                    model=model,
                    usage=normalized_usage,
                    pricing_evidence=effective_evidence,
                )
            )
            if pricing_evidence is None:
                # Remember missing evidence even before the first evidence-
                # bearing call, so a later segment cannot make an incomplete
                # aggregate look exact.
                self._pricing_evidence_complete = False
            else:
                self._pricing_evidence_seen = True
                self._pricing_segments.extend(pricing_evidence.segments)
                self._pricing_evidence_complete = (
                    self._pricing_evidence_complete and evidence_complete
                )
            if accounting_debt:
                self._accounting_debt_by_role.setdefault(
                    role,
                    set(),
                ).update(accounting_debt)

            total = self._budget_debit_usd if self.diagnostic_cost_policy else self._total_cost_usd
            cap = self.max_cost_usd

        if cap is not None and total > cap:
            raise CostLimitExceeded(
                f"cumulative cost ${total:.4f} exceeds cap ${cap:.4f} "
                f"(last call: role={role!r}, backend={backend_name!r}, "
                f"cost=${cost:.4f})"
            )

    def note_unloaded_ref(self, count: int = 1) -> None:
        """Record that ``count`` cost-bearing corpus records could not be
        loaded during a post-run rollup, so the accumulated total is a
        known *under-count*.

        Cost rollups that walk a Spec's ``implementation_refs`` /
        ``verdict_refs`` (architect, swebench) skip any ref that fails to
        load — corpus drift, a deleted blob — and continue. Without this
        signal the resulting total looks complete. Surfaced as
        ``unloaded_refs`` + ``rollup_incomplete`` in ``to_dict()`` so an
        unattended run's reported spend (and any post-hoc budget
        reconciliation) is never silently trusted as complete
        (CRU-05a / ARC-2). Lives here, not at each call site, so every
        ``to_dict()`` consumer sees the same flag uniformly.
        """
        with self._lock:
            self._unloaded_refs += count

    def note_accounting_debt(
        self,
        *,
        role: str,
        missing: tuple[str, ...],
    ) -> None:
        """Mark a role's recorded totals as a known accounting under-count.

        The numeric lower bound remains useful, but it cannot authorize more
        paid work or promotion. Callers persist this alongside the aggregate
        execution-completeness fields on the source verdict.
        """

        if not isinstance(role, str) or not role.strip():
            raise ValueError("accounting-debt role must be a non-empty string")
        allowed = {"token_usage_complete", "cost_complete"}
        if (
            not missing
            or any(field_name not in allowed for field_name in missing)
        ):
            raise ValueError(
                "accounting debt must name token_usage_complete and/or "
                "cost_complete"
            )
        with self._lock:
            self._accounting_debt_by_role.setdefault(role, set()).update(
                missing
            )
            # Out-of-band debt has no bounded receipt. Never erase it merely
            # because a later response supplies a valid upper bound.
            self._unbounded_accounting_debt = True

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return self._total_cost_usd

    @property
    def budget_cost_usd(self) -> float:
        with self._lock:
            return self._budget_debit_usd if self.diagnostic_cost_policy else self._total_cost_usd

    def _diagnostic_budget_covered_locked(self) -> bool:
        return (
            self.diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
            and self._diagnostic_bounds_seen
            and self._diagnostic_coverage_complete
            and not self._unbounded_accounting_debt
            and not self._unloaded_refs
        )

    @property
    def diagnostic_budget_covered(self) -> bool:
        with self._lock:
            return self._diagnostic_budget_covered_locked()

    @property
    def cost_bounds(self) -> UsageCostBounds | None:
        with self._lock:
            bounds = combine_cost_bounds([call.usage for call in self._pricing_calls])
            if bounds is None:
                return None
            return UsageCostBounds(
                segments=bounds.segments,
                coverage_complete=bounds.coverage_complete and self._diagnostic_budget_covered_locked(),
            )

    def _remaining_locked(self) -> float | None:
        if self.diagnostic_cost_policy is not None and self._pricing_calls and not self._diagnostic_budget_covered_locked():
            return 0.0
        if self._unloaded_refs or (
            self._accounting_debt_by_role and not self._diagnostic_budget_covered_locked()
        ):
            return 0.0
        if self.max_cost_usd is None:
            return 0.0 if self.diagnostic_cost_policy else None
        debit = self._budget_debit_usd if self.diagnostic_cost_policy else self._total_cost_usd
        return max(0.0, self.max_cost_usd - debit)

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return self._total_output_tokens

    @property
    def total_cache_creation_tokens(self) -> int:
        with self._lock:
            return self._total_cache_creation_tokens

    @property
    def total_cache_read_tokens(self) -> int:
        with self._lock:
            return self._total_cache_read_tokens

    def _pricing_evidence_locked(self) -> UsagePricingEvidence | None:
        """Build aggregate evidence while the caller holds ``_lock``."""

        if not self._pricing_evidence_seen:
            return None
        segments = tuple(self._pricing_segments)
        candidate = UsagePricingEvidence(segments=segments, complete=False)
        aggregate_usage = Usage(
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
            cost_usd=self._total_cost_usd,
            cache_creation_tokens=self._total_cache_creation_tokens,
            cache_read_tokens=self._total_cache_read_tokens,
        )
        complete = (
            self._pricing_evidence_complete
            and self._unloaded_refs == 0
            and not self._accounting_debt_by_role
            and candidate.reconciles(aggregate_usage)
        )
        return UsagePricingEvidence(segments=segments, complete=complete)

    @property
    def pricing_evidence(self) -> UsagePricingEvidence | None:
        """Ordered exact segments, with fail-closed aggregate completeness."""

        with self._lock:
            return self._pricing_evidence_locked()

    @staticmethod
    def _pricing_evidence_dict(
        pricing_evidence: UsagePricingEvidence,
    ) -> dict[str, Any]:
        """Project typed evidence without accepting arbitrary payload keys."""

        return {
            "segments": [
                {
                    "input_tokens": segment.input_tokens,
                    "output_tokens": segment.output_tokens,
                    "cache_creation_tokens": segment.cache_creation_tokens,
                    "cache_read_tokens": segment.cache_read_tokens,
                    "cost_usd": segment.cost_usd,
                }
                for segment in pricing_evidence.segments
            ],
            "complete": pricing_evidence.complete,
        }

    def cost_by_role(self) -> dict[str, float]:
        with self._lock:
            return {r: round(s.cost_usd, 6) for r, s in self._by_role.items()}

    def calls_by_role(self) -> dict[str, int]:
        with self._lock:
            return {r: s.calls for r, s in self._by_role.items()}

    def cost_by_backend(self) -> dict[str, float]:
        with self._lock:
            return {b: round(c, 6) for b, c in self._by_backend.items()}

    def cost_by_model(self) -> dict[str, float]:
        with self._lock:
            return {m: round(c, 6) for m, c in self._by_model.items()}

    def remaining(self) -> float | None:
        """Remaining spend authority, or None only when uncapped and complete.

        Once aggregate accounting is incomplete, the observed numeric total is
        only a lower bound. Return the fail-closed zero even for an otherwise
        uncapped accumulator so a generic ``remaining()`` check cannot launch
        another paid call under unknown spend.
        """
        with self._lock:
            return self._remaining_locked()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Stable JSON shape for inclusion in workflow ``summary.json``.

        Shape::

            {
              "total_usd":       4.231,
              "total_input_tokens":   12345,
              "total_output_tokens":  4567,
              "max_cost_usd":    50.0,        // or null
              "remaining_usd":   45.769,      // or null
              "by_role": {
                  "goal_red":   {cost_usd, input_tokens, output_tokens,
                                 calls, backends: [...], models: [...]},
                  ...
              },
              "by_backend": {"claude": 4.231, "codex": 0.0, "local": 0.0},
              "by_model":   {"aws/anthropic/bedrock-claude-opus-4-8": 3.1, "aws/anthropic/bedrock-claude-sonnet-4-6": 1.13},
              "unloaded_refs":     0,           // refs a rollup couldn't load
              "accounting_debt": false,         // incomplete provider usage
              "accounting_debt_by_role": {},
              "rollup_incomplete": false,       // any known under-count
              // Optional when any call carries typed pricing evidence:
              "pricing_evidence": {segments: [...], complete: true},
              "pricing_evidence_by_call": [
                  {role, backend, model, usage, pricing_evidence}, ...
              ],
            }
        """
        with self._lock:
            pricing_evidence = self._pricing_evidence_locked()
            extended_pricing = bool(
                self._total_cache_creation_tokens
                or self._total_cache_read_tokens
                or pricing_evidence is not None
                or self.diagnostic_cost_policy is not None
            )
            result = {
                "total_usd": (
                    self._total_cost_usd
                    if extended_pricing
                    else round(self._total_cost_usd, 6)
                ),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "max_cost_usd": self.max_cost_usd,
                "remaining_usd": (
                    None if self._remaining_locked() is None
                    else self._remaining_locked() if self.diagnostic_cost_policy is not None
                    else round(self._remaining_locked(), 6)
                ),
                "by_role": {
                    role: slot.to_dict(exact_cost=extended_pricing)
                    for role, slot in sorted(self._by_role.items())
                },
                "by_backend": {
                    b: c if extended_pricing else round(c, 6)
                    for b, c in sorted(self._by_backend.items())
                },
                "by_model": {
                    m: c if extended_pricing else round(c, 6)
                    for m, c in sorted(self._by_model.items())
                },
                "unloaded_refs": self._unloaded_refs,
                "accounting_debt": bool(self._accounting_debt_by_role),
                "accounting_debt_by_role": {
                    role: sorted(missing)
                    for role, missing in sorted(
                        self._accounting_debt_by_role.items()
                    )
                },
                "rollup_incomplete": (
                    self._unloaded_refs > 0
                    or bool(self._accounting_debt_by_role)
                ),
            }
            if self.diagnostic_cost_policy is not None:
                covered = self._diagnostic_budget_covered_locked()
                result["diagnostic_cost_policy"] = self.diagnostic_cost_policy
                result["diagnostic_budget_covered"] = covered
                result["budget_cost_usd"] = self._budget_debit_usd
                result["cost_upper_bound_usd"] = self._budget_debit_usd if covered else None
                result["cost_bounds_by_call"] = list(self._cost_bound_calls)
                result["promotion_evidence"] = False
            if extended_pricing:
                result["pricing_evidence_schema"] = PRICING_EVIDENCE_SCHEMA
                result["total_cache_creation_tokens"] = (
                    self._total_cache_creation_tokens
                )
                result["total_cache_read_tokens"] = (
                    self._total_cache_read_tokens
                )
            if pricing_evidence is not None:
                result["pricing_evidence"] = self._pricing_evidence_dict(
                    pricing_evidence
                )
                result["pricing_evidence_by_call"] = [
                    {
                        "role": call.role,
                        "backend": call.backend_name,
                        "model": call.model,
                        "usage": {
                            "input_tokens": call.usage.input_tokens,
                            "output_tokens": call.usage.output_tokens,
                            "cache_creation_tokens": (
                                call.usage.cache_creation_tokens
                            ),
                            "cache_read_tokens": call.usage.cache_read_tokens,
                            "cost_usd": call.usage.cost_usd,
                        },
                        "pricing_evidence": self._pricing_evidence_dict(
                            call.pricing_evidence
                            if call.pricing_evidence is not None
                            else UsagePricingEvidence(
                                segments=(),
                                complete=False,
                            )
                        ),
                    }
                    for call in self._pricing_calls
                ]
            return result


__all__ = [
    "CostAccumulator",
    "CostLimitExceeded",
    "PRICING_EVIDENCE_SCHEMA",
    "parse_token_count",
]
