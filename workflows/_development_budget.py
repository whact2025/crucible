# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Protected per-role authority for autonomous development workflows."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Literal

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BudgetExhausted,
    ExecutionCompleteness,
    InvocationResult,
    Usage,
    UsagePricingEvidence,
)
from ._diagnostic_cost import (
    DIAGNOSTIC_COST_POLICY,
    diagnostic_budget_debit,
    diagnostic_call_covered,
)


BudgetGroup = Literal["planning_review", "implementation_verifier"]
_GROUPS: tuple[BudgetGroup, ...] = (
    "planning_review",
    "implementation_verifier",
)


@dataclass(frozen=True, slots=True)
class RoleAllocation:
    """Frozen authority for one logical model role."""

    role: str
    group: BudgetGroup
    max_calls: int
    max_output_tokens: int
    max_wall_seconds: float
    max_cost_usd: float
    protected_tail_usd: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("RoleAllocation.role must be non-empty")
        if self.group not in _GROUPS:
            raise ValueError(f"unknown development budget group {self.group!r}")
        for name in ("max_calls", "max_output_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"RoleAllocation.{name} must be an integer >= 1")
        for name in ("max_wall_seconds", "max_cost_usd"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    f"RoleAllocation.{name} must be finite and positive"
                )
        tail = self.protected_tail_usd
        if (
            isinstance(tail, bool)
            or not isinstance(tail, (int, float))
            or not math.isfinite(float(tail))
            or float(tail) < 0
        ):
            raise ValueError(
                "RoleAllocation.protected_tail_usd must be finite and "
                "non-negative"
            )
        if tail > self.max_cost_usd:
            raise ValueError(
                "RoleAllocation.protected_tail_usd must not exceed "
                "max_cost_usd"
            )


@dataclass(frozen=True, slots=True)
class DevelopmentBudgetPolicy:
    """Task-wide 40/50/10 authority split plus explicit role ceilings.

    A policy may close planning after admission and transfer only its settled,
    unused authority to implementation.  In that mode, an implementation role
    ceiling may include the potential one-way transfer, but the live group cap
    grows only after admission and settlement.  The task-wide ceiling remains
    fixed.
    """

    task_budget_usd: float
    planning_review_cap_usd: float
    implementation_verifier_reserve_usd: float
    contingency_usd: float
    roles: tuple[RoleAllocation, ...]
    schema_version: str = "crucible.development-budget.v3"
    release_unused_planning_authority: bool = False

    def __post_init__(self) -> None:
        task = self.task_budget_usd
        if (
            isinstance(task, bool)
            or not isinstance(task, (int, float))
            or not math.isfinite(float(task))
            or float(task) <= 0
        ):
            raise ValueError("task_budget_usd must be finite and positive")
        for name in (
            "planning_review_cap_usd",
            "implementation_verifier_reserve_usd",
            "contingency_usd",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        tolerance = 1e-9
        if self.planning_review_cap_usd > task * 0.40 + tolerance:
            raise ValueError("planning/review authority must not exceed 40%")
        if (
            self.implementation_verifier_reserve_usd
            < task * 0.50 - tolerance
        ):
            raise ValueError(
                "implementation/verifier reserve must be at least 50%"
            )
        if self.contingency_usd < task * 0.10 - tolerance:
            raise ValueError("contingency must be at least 10%")
        if (
            self.planning_review_cap_usd
            + self.implementation_verifier_reserve_usd
            + self.contingency_usd
            > task + tolerance
        ):
            raise ValueError("development budget groups exceed task authority")
        if not isinstance(self.roles, tuple) or not self.roles:
            raise ValueError("development budget requires role allocations")
        if self.schema_version != "crucible.development-budget.v3":
            raise ValueError(
                "schema_version must be crucible.development-budget.v3"
            )
        if not isinstance(self.release_unused_planning_authority, bool):
            raise ValueError(
                "release_unused_planning_authority must be a boolean"
            )
        names = [allocation.role for allocation in self.roles]
        if len(names) != len(set(names)):
            raise ValueError("development budget role allocations must be unique")
        implementation_role_cap_usd = (
            self.implementation_verifier_cap_usd
            + (
                self.planning_review_cap_usd
                if self.release_unused_planning_authority
                else 0.0
            )
        )
        group_caps = {
            "planning_review": self.planning_review_cap_usd,
            # This is only the maximum policy-time role ceiling.  The ledger's
            # active group cap still gains solely the settled, unused planning
            # authority at implementation-phase admission, and dispatch still
            # protects every outstanding mandatory tail.
            "implementation_verifier": implementation_role_cap_usd,
        }
        protected_tail_usd = 0.0
        for allocation in self.roles:
            if allocation.max_cost_usd > group_caps[allocation.group] + tolerance:
                raise ValueError(
                    f"role {allocation.role!r} exceeds its group USD cap"
                )
            if allocation.protected_tail_usd > 0:
                if allocation.group != "implementation_verifier":
                    raise ValueError(
                        "protected tail authority belongs only to the "
                        "implementation/verifier group"
                    )
                protected_tail_usd += allocation.protected_tail_usd
        if protected_tail_usd > self.contingency_usd + tolerance:
            raise ValueError(
                "protected tail authority exceeds contingency"
            )

    @property
    def implementation_verifier_cap_usd(self) -> float:
        """Return the one-way reserve plus contingency authority."""

        return (
            self.implementation_verifier_reserve_usd
            + self.contingency_usd
        )

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "task_budget_usd": self.task_budget_usd,
            "planning_review_cap_usd": self.planning_review_cap_usd,
            "implementation_verifier_reserve_usd": (
                self.implementation_verifier_reserve_usd
            ),
            "contingency_usd": self.contingency_usd,
            "release_unused_planning_authority": (
                self.release_unused_planning_authority
            ),
            "roles": [
                {
                    "role": item.role,
                    "group": item.group,
                    "max_calls": item.max_calls,
                    "max_output_tokens": item.max_output_tokens,
                    "max_wall_seconds": item.max_wall_seconds,
                    "max_cost_usd": item.max_cost_usd,
                    "protected_tail_usd": item.protected_tail_usd,
                }
                for item in self.roles
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class _RoleUsage:
    calls: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0
    reserved_cost_usd: float = 0.0
    in_flight: int = 0
    exhausted_by: str | None = None
    accounting_debt: bool = False
    request_bound_overshoot_usd: float = 0.0


class DevelopmentBudgetLedger:
    """Thread-safe task ledger shared by every backend pool in the cascade."""

    def __init__(
        self, policy: DevelopmentBudgetPolicy, *,
        diagnostic_cost_policy: str | None = None,
    ) -> None:
        if not isinstance(policy, DevelopmentBudgetPolicy):
            raise TypeError("policy must be DevelopmentBudgetPolicy")
        self.policy = policy
        if diagnostic_cost_policy not in (None, DIAGNOSTIC_COST_POLICY):
            raise ValueError("unknown diagnostic cost policy")
        self.diagnostic_cost_policy = diagnostic_cost_policy
        self._allocations = {item.role: item for item in policy.roles}
        self._usage = {role: _RoleUsage() for role in self._allocations}
        self._group_spend = {group: 0.0 for group in _GROUPS}
        self._group_reserved = {group: 0.0 for group in _GROUPS}
        self._completed_tail_roles: set[str] = set()
        self._lock = threading.RLock()
        self._authority_revoked = False
        self._revocation_reason: str | None = None
        self._validation_repair_count = 0
        self._source_calls_dispatched = 0
        self._source_calls_denied = 0
        self._first_mutation_seconds: float | None = None
        self._cost_to_first_mutation_usd: float | None = None
        self._implementation_phase_admitted = False
        self._released_planning_authority_usd = 0.0
        self._planning_spend_at_close_usd: float | None = None

    def require_role(self, role: str) -> None:
        if role not in self._allocations:
            raise ValueError(
                f"development budget has no allocation for active role {role!r}"
            )

    def _group_cap(self, group: BudgetGroup) -> float:
        if group == "planning_review":
            return self.policy.planning_review_cap_usd
        return (
            self.policy.implementation_verifier_cap_usd
            + self._released_planning_authority_usd
        )

    def _group_remaining(self, group: BudgetGroup) -> float:
        if group == "planning_review" and self._implementation_phase_admitted:
            return 0.0
        return max(
            0.0,
            self._group_cap(group)
            - self._group_spend[group]
            - self._group_reserved[group],
        )

    def _protected_tail_remaining(self, role: str) -> float:
        """Return one owner's still-protected, unsettled tail authority."""

        allocation = self._allocations[role]
        if (
            allocation.protected_tail_usd <= 0.0
            or role in self._completed_tail_roles
        ):
            return 0.0
        usage = self._usage[role]
        return max(
            0.0,
            allocation.protected_tail_usd
            - usage.cost_usd
            - usage.reserved_cost_usd,
        )

    def _tail_protected_from(self, role: str) -> float:
        """Return tail authority the caller is forbidden to reserve."""

        allocation = self._allocations[role]
        if allocation.group != "implementation_verifier":
            return 0.0
        return sum(
            self._protected_tail_remaining(owner)
            for owner, owner_allocation in self._allocations.items()
            if (
                owner_allocation.protected_tail_usd > 0.0
                and owner != role
            )
        ) + (
            self._protected_tail_remaining(role)
            if allocation.protected_tail_usd <= 0.0
            else 0.0
        )

    def _role_dispatch_capacity_unlocked(
        self,
        role: str,
        *,
        requested_cost_usd: float | None,
    ) -> tuple[float, str | None, bool]:
        """Preview one role's exact reservation without mutating the ledger.

        The boolean says whether a denied dispatch should latch its reason in
        ``usage.exhausted_by``.  Phase/authority denials are not allocation
        exhaustion, while terminal role limits and protected-tail boundaries
        are.  Keeping this calculation shared with :meth:`_begin` prevents a
        workflow preflight from estimating against the broader group balance
        and accidentally spending a sibling's protected tail.
        """

        self.require_role(role)
        allocation = self._allocations[role]
        usage = self._usage[role]
        if self._authority_revoked:
            return (
                0.0,
                self._revocation_reason or "authority_revoked",
                False,
            )
        if (
            self.policy.release_unused_planning_authority
            and allocation.group == "implementation_verifier"
            and not self._implementation_phase_admitted
        ):
            return 0.0, "implementation_phase_not_admitted", False
        if (
            allocation.group == "planning_review"
            and self._implementation_phase_admitted
        ):
            return 0.0, "planning_phase_closed", False
        if (
            allocation.protected_tail_usd > 0.0
            and role in self._completed_tail_roles
        ):
            return 0.0, "tail_role_completed", True
        checks = (
            ("call_count", usage.calls >= allocation.max_calls),
            (
                "output_tokens",
                usage.output_tokens >= allocation.max_output_tokens,
            ),
            (
                "wall_seconds",
                usage.wall_seconds >= allocation.max_wall_seconds,
            ),
            ("cost_usd", usage.cost_usd >= allocation.max_cost_usd),
            (
                "group_cost_usd",
                self._group_spend[allocation.group]
                + self._group_reserved[allocation.group]
                >= self._group_cap(allocation.group),
            ),
        )
        exhausted = next(
            (name for name, tripped in checks if tripped),
            None,
        )
        if exhausted is not None:
            return 0.0, exhausted, True
        available = min(
            allocation.max_cost_usd
            - usage.cost_usd
            - usage.reserved_cost_usd,
            self._group_cap(allocation.group)
            - self._group_spend[allocation.group]
            - self._group_reserved[allocation.group]
            - self._tail_protected_from(role),
        )
        if available <= 0.0:
            group_remaining = (
                self._group_cap(allocation.group)
                - self._group_spend[allocation.group]
                - self._group_reserved[allocation.group]
            )
            reason = (
                "protected_tail_usd"
                if group_remaining > 0.0
                and self._tail_protected_from(role) > 0.0
                else "cost_usd"
            )
            return 0.0, reason, True
        requested = (
            available
            if requested_cost_usd is None
            else max(0.0, float(requested_cost_usd))
        )
        reservation = min(available, requested)
        if reservation <= 0.0:
            return 0.0, "cost_usd", True
        return reservation, None, False

    def group_dispatch_capacity(self, group: BudgetGroup) -> float:
        """Preview unreserved group dollars; dispatch still reserves atomically."""
        if group not in _GROUPS:
            raise ValueError(f"unknown development budget group {group!r}")
        with self._lock:
            if self._authority_revoked:
                return 0.0
            return self._group_remaining(group)

    def role_dispatch_capacity(
        self,
        role: str,
        *,
        requested_cost_usd: float | None = None,
    ) -> float:
        """Return the exact reservation a role could obtain right now.

        This is a read-only admission preview for serial workflow boundaries;
        it does not consume call authority.  Concurrent callers must use the
        atomic :meth:`_begin` path instead of treating a preview as a token.
        """

        with self._lock:
            capacity, _reason, _latch = (
                self._role_dispatch_capacity_unlocked(
                    role,
                    requested_cost_usd=requested_cost_usd,
                )
            )
            return capacity

    def role_protected_tail_usd(self, role: str) -> float:
        """Return one role's configured (not remaining) tail tranche."""

        with self._lock:
            self.require_role(role)
            return float(self._allocations[role].protected_tail_usd)

    def _begin(
        self,
        role: str,
        *,
        requested_cost_usd: float | None,
    ) -> tuple[bool, str | None, float]:
        with self._lock:
            reservation, reason, latch_exhaustion = (
                self._role_dispatch_capacity_unlocked(
                    role,
                    requested_cost_usd=requested_cost_usd,
                )
            )
            usage = self._usage[role]
            if reason is not None:
                if latch_exhaustion:
                    usage.exhausted_by = usage.exhausted_by or reason
                return False, reason, 0.0
            allocation = self._allocations[role]
            if reservation <= 0.0:  # pragma: no cover - helper invariant
                usage.exhausted_by = usage.exhausted_by or "cost_usd"
                return False, "cost_usd", 0.0
            usage.calls += 1
            usage.in_flight += 1
            usage.reserved_cost_usd += reservation
            self._group_reserved[allocation.group] += reservation
            return True, None, reservation

    def _settle(
        self,
        role: str,
        *,
        result: InvocationResult | None,
        elapsed: float,
        raised: BaseException | None,
        reserved_cost_usd: float,
    ) -> None:
        with self._lock:
            allocation = self._allocations[role]
            usage = self._usage[role]
            usage.in_flight = max(0, usage.in_flight - 1)
            released = min(
                max(0.0, reserved_cost_usd),
                usage.reserved_cost_usd,
            )
            usage.reserved_cost_usd = max(
                0.0,
                usage.reserved_cost_usd - released,
            )
            self._group_reserved[allocation.group] = max(
                0.0,
                self._group_reserved[allocation.group] - released,
            )
            usage.wall_seconds += max(0.0, elapsed)
            if result is not None:
                completeness = result.completeness
                if (
                    not completeness.token_usage_complete
                    or not completeness.cost_complete
                ):
                    usage.accounting_debt = True
                    if not diagnostic_call_covered(
                        result, policy=self.diagnostic_cost_policy,
                    ):
                        self._authority_revoked = True
                        self._revocation_reason = (
                            f"{role} returned incomplete accounting"
                        )
                raw_output_tokens = result.usage.output_tokens
                if (
                    type(raw_output_tokens) is int
                    and raw_output_tokens >= 0
                ):
                    usage.output_tokens += raw_output_tokens
                else:
                    usage.accounting_debt = True
                    self._authority_revoked = True
                    self._revocation_reason = (
                        f"{role} returned malformed output-token accounting"
                    )
                raw_cost = diagnostic_budget_debit(
                    result, policy=self.diagnostic_cost_policy,
                )
                try:
                    normalized_cost = (
                        float(raw_cost)
                        if type(raw_cost) in (int, float)
                        else None
                    )
                except (OverflowError, TypeError, ValueError):
                    normalized_cost = None
                if (
                    normalized_cost is not None
                    and math.isfinite(normalized_cost)
                    and normalized_cost >= 0.0
                ):
                    usage.request_bound_overshoot_usd += max(
                        0.0,
                        normalized_cost - max(0.0, reserved_cost_usd),
                    )
                    usage.cost_usd += normalized_cost
                    self._group_spend[
                        allocation.group
                    ] += normalized_cost
                else:
                    usage.accounting_debt = True
                    self._authority_revoked = True
                    self._revocation_reason = (
                        f"{role} returned malformed cost accounting"
                    )
            if raised is not None:
                usage.accounting_debt = True
                self._authority_revoked = True
                self._revocation_reason = (
                    f"{role} raised before complete usage settlement"
                )
            for name, tripped in (
                ("output_tokens", usage.output_tokens >= allocation.max_output_tokens),
                ("wall_seconds", usage.wall_seconds >= allocation.max_wall_seconds),
                ("cost_usd", usage.cost_usd >= allocation.max_cost_usd),
                (
                    "group_cost_usd",
                    self._group_spend[allocation.group]
                    >= self._group_cap(allocation.group),
                ),
            ):
                if tripped:
                    usage.exhausted_by = usage.exhausted_by or name

    def wrap(self, role: str, backend: Any) -> "BudgetedBackend":
        self.require_role(role)
        return BudgetedBackend(backend, role=role, ledger=self)

    def note_validation_repairs(self, count: int) -> None:
        """Retain the greatest observed deterministic repair count."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("validation repair count must be an integer >= 0")
        with self._lock:
            self._validation_repair_count = max(
                self._validation_repair_count,
                count,
            )

    def note_source_dispatch(
        self,
        *,
        dispatched_calls: int,
        denied_calls: int,
    ) -> None:
        """Retain terminal task-wide source-dispatch counters."""

        for name, value in (
            ("dispatched_calls", dispatched_calls),
            ("denied_calls", denied_calls),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        with self._lock:
            self._source_calls_dispatched = max(
                self._source_calls_dispatched,
                dispatched_calls,
            )
            self._source_calls_denied = max(
                self._source_calls_denied,
                denied_calls,
            )

    def note_first_mutation(
        self,
        *,
        elapsed_seconds: float,
        cost_usd: float,
    ) -> None:
        """Record the first persisted implementation-mutation boundary."""

        elapsed = max(0.0, float(elapsed_seconds))
        cost = max(0.0, float(cost_usd))
        if not math.isfinite(elapsed) or not math.isfinite(cost):
            raise ValueError("first mutation metrics must be finite")
        with self._lock:
            if self._first_mutation_seconds is None:
                self._first_mutation_seconds = elapsed
                self._cost_to_first_mutation_usd = cost

    def group_remaining(self, group: BudgetGroup) -> float:
        with self._lock:
            return self._group_remaining(group)

    @property
    def protected_tail_roles(self) -> tuple[str, ...]:
        """Return the frozen mandatory tail owners in stable role order."""

        return tuple(
            role
            for role, allocation in self._allocations.items()
            if allocation.protected_tail_usd > 0.0
        )

    def complete_protected_tail_role(self, role: str) -> None:
        """Release one mandatory role's unused tail after typed completion.

        A provider settlement is not completion: review workflows may retry a
        settled call.  The workflow must explicitly close the role after its
        final typed result, and cannot do so before at least one fully settled
        call.  Completion is monotonic and idempotent.
        """

        with self._lock:
            self.require_role(role)
            allocation = self._allocations[role]
            if allocation.protected_tail_usd <= 0.0:
                raise ValueError(
                    f"role {role!r} has no protected tail authority"
                )
            if role in self._completed_tail_roles:
                return
            usage = self._usage[role]
            if usage.calls < 1:
                raise ValueError(
                    f"role {role!r} cannot complete before its first call"
                )
            if usage.in_flight or usage.reserved_cost_usd > 1e-9:
                raise ValueError(
                    f"role {role!r} cannot complete with unsettled calls"
                )
            self._completed_tail_roles.add(role)

    def admit_implementation_phase(self) -> tuple[bool, str | None]:
        """Atomically prove that the protected implementation reserve remains.

        Provider USD enforcement is post-hoc, so a final planning/review call
        can settle above the amount reserved for it.  This boundary runs only
        after every planning call has settled and before an executable plan is
        authorized.  A failed proof permanently revokes later paid authority.
        """

        with self._lock:
            if self._authority_revoked:
                return (
                    False,
                    self._revocation_reason or "authority_revoked",
                )
            if self._implementation_phase_admitted:
                return True, None

            tolerance = 1e-9
            planning_spend = self._group_spend["planning_review"]
            planning_reserved = self._group_reserved["planning_review"]
            implementation_remaining = max(
                0.0,
                self.policy.implementation_verifier_reserve_usd
                - self._group_spend["implementation_verifier"]
                - self._group_reserved["implementation_verifier"],
            )
            task_remaining = max(
                0.0,
                self.policy.task_budget_usd
                - sum(self._group_spend.values())
                - sum(self._group_reserved.values()),
            )

            reason: str | None = None
            if planning_reserved > tolerance:
                reason = "planning_review_accounting_unsettled"
            elif (
                planning_spend
                > self.policy.planning_review_cap_usd + tolerance
            ):
                reason = "planning_review_cap_exceeded"
            elif task_remaining + tolerance < implementation_remaining:
                reason = "implementation_verifier_reserve_eroded"

            if reason is not None:
                self._authority_revoked = True
                self._revocation_reason = (
                    "implementation phase admission failed: " + reason
                )
                return False, reason
            self._implementation_phase_admitted = True
            self._planning_spend_at_close_usd = planning_spend
            if self.policy.release_unused_planning_authority:
                self._released_planning_authority_usd = max(
                    0.0,
                    self.policy.planning_review_cap_usd
                    - planning_spend,
                )
            return True, None

    def to_summary(self) -> dict[str, Any]:
        with self._lock:
            planning_spend = self._group_spend["planning_review"]
            implementation_spend = self._group_spend[
                "implementation_verifier"
            ]
            implementation_reserved = self._group_reserved[
                "implementation_verifier"
            ]
            effective_remaining = max(
                0.0,
                self._group_cap("implementation_verifier")
                - implementation_spend
                - implementation_reserved,
            )
            base_cap = self.policy.implementation_verifier_reserve_usd
            contingency_cap = self.policy.contingency_usd
            released_cap = self._released_planning_authority_usd
            base_spend = min(base_cap, implementation_spend)
            released_spend = min(
                released_cap,
                max(0.0, implementation_spend - base_cap),
            )
            contingency_spend = min(
                contingency_cap,
                max(
                    0.0,
                    implementation_spend - base_cap - released_cap,
                ),
            )
            effective_overshoot = max(
                0.0,
                implementation_spend
                - base_cap
                - released_cap
                - contingency_cap,
            )
            base_remaining_before_reservations = max(
                0.0,
                base_cap - implementation_spend,
            )
            base_reserved = min(
                implementation_reserved,
                base_remaining_before_reservations,
            )
            released_reserved = min(
                max(0.0, released_cap - released_spend),
                max(0.0, implementation_reserved - base_reserved),
            )
            contingency_reserved = min(
                max(0.0, contingency_cap - contingency_spend),
                max(
                    0.0,
                    implementation_reserved
                    - base_reserved
                    - released_reserved,
                ),
            )
            base_remaining = max(
                0.0,
                base_cap - base_spend - base_reserved,
            )
            contingency_remaining = max(
                0.0,
                contingency_cap
                - contingency_spend
                - contingency_reserved,
            )
            released_remaining = max(
                0.0,
                released_cap - released_spend - released_reserved,
            )
            tail_owners = {}
            for role in self.protected_tail_roles:
                allocation = self._allocations[role]
                usage = self._usage[role]
                completed = role in self._completed_tail_roles
                remaining = self._protected_tail_remaining(role)
                consumed = min(
                    allocation.protected_tail_usd,
                    usage.cost_usd,
                )
                released = (
                    max(0.0, allocation.protected_tail_usd - consumed)
                    if completed else 0.0
                )
                tail_owners[role] = {
                    "allocated_usd": round(
                        allocation.protected_tail_usd, 6
                    ),
                    "role_spend_usd": round(usage.cost_usd, 6),
                    "role_reserved_usd": round(
                        usage.reserved_cost_usd, 6
                    ),
                    "consumed_usd": round(consumed, 6),
                    "remaining_usd": round(remaining, 6),
                    "released_unused_usd": round(released, 6),
                    "role_cap_overshoot_usd": round(
                        max(
                            0.0,
                            usage.cost_usd
                            - allocation.protected_tail_usd,
                        ),
                        6,
                    ),
                    "terminal_completed": completed,
                }
            tail_outstanding = sum(
                self._protected_tail_remaining(role)
                for role in self.protected_tail_roles
            )
            active_group_cap = {
                "planning_review": (
                    0.0
                    if self._implementation_phase_admitted
                    else self.policy.planning_review_cap_usd
                ),
                "implementation_verifier": (
                    self._group_cap("implementation_verifier")
                    if (
                        self._implementation_phase_admitted
                        or not self.policy.release_unused_planning_authority
                    )
                    else 0.0
                ),
            }
            phase_committed_cap = (
                (
                    self._planning_spend_at_close_usd
                    if self._planning_spend_at_close_usd is not None
                    else planning_spend
                )
                + active_group_cap["implementation_verifier"]
                if self._implementation_phase_admitted
                else sum(active_group_cap.values())
            )
            task_spend = sum(self._group_spend.values())
            task_reserved = sum(self._group_reserved.values())
            return {
                "schema_version": self.policy.schema_version,
                **({
                    "diagnostic_cost_policy": self.diagnostic_cost_policy,
                    "cost_values_are_budget_upper_bounds": True,
                    "exact_cost_complete": False,
                    "rankable": False,
                } if self.diagnostic_cost_policy is not None else {}),
                "policy_digest": self.policy.digest,
                "task_budget_usd": self.policy.task_budget_usd,
                "planning_review_cap_usd": (
                    self.policy.planning_review_cap_usd
                ),
                "implementation_verifier_reserve_usd": (
                    self.policy.implementation_verifier_reserve_usd
                ),
                "contingency_usd": self.policy.contingency_usd,
                "phase_authority": {
                    "implementation_phase_admitted": (
                        self._implementation_phase_admitted
                    ),
                    "unused_planning_authority_release_enabled": (
                        self.policy.release_unused_planning_authority
                    ),
                    "planning_authority_released_usd": released_cap,
                    "settled_planning_spend_at_close_usd": (
                        self._planning_spend_at_close_usd
                    ),
                },
                "group_cap_usd": {
                    "planning_review": (
                        self.policy.planning_review_cap_usd
                    ),
                    "implementation_verifier": (
                        self.policy.implementation_verifier_cap_usd
                    ),
                },
                "active_group_cap_usd": active_group_cap,
                "group_spend_usd": dict(self._group_spend),
                "group_reserved_usd": dict(self._group_reserved),
                "group_remaining_usd": {
                    group: self._group_remaining(group)
                    for group in _GROUPS
                },
                "task_authority": {
                    "phase_committed_cap_usd": phase_committed_cap,
                    "settled_spend_usd": task_spend,
                    "reserved_usd": task_reserved,
                    "remaining_usd": max(
                        0.0,
                        self.policy.task_budget_usd
                        - task_spend
                        - task_reserved,
                    ),
                    "post_hoc_overshoot_usd": max(
                        0.0,
                        task_spend - self.policy.task_budget_usd,
                    ),
                    "phase_commitment_within_task_cap": (
                        phase_committed_cap
                        <= self.policy.task_budget_usd + 1e-9
                    ),
                },
                "implementation_verifier_authority": {
                    "base_cap_usd": base_cap,
                    "effective_cap_usd": self._group_cap(
                        "implementation_verifier"
                    ),
                    "base_spend_usd": round(base_spend, 6),
                    "base_reserved_usd": round(base_reserved, 6),
                    "base_remaining_usd": round(base_remaining, 6),
                    "released_planning_cap_usd": round(
                        released_cap,
                        6,
                    ),
                    "released_planning_spend_usd": round(
                        released_spend,
                        6,
                    ),
                    "released_planning_reserved_usd": round(
                        released_reserved,
                        6,
                    ),
                    "released_planning_remaining_usd": round(
                        released_remaining,
                        6,
                    ),
                    "contingency_cap_usd": contingency_cap,
                    "contingency_spend_usd": round(
                        contingency_spend, 6
                    ),
                    "contingency_reserved_usd": round(
                        contingency_reserved, 6
                    ),
                    "contingency_remaining_usd": round(
                        contingency_remaining, 6
                    ),
                    "post_hoc_overshoot_usd": round(
                        effective_overshoot, 6
                    ),
                    "effective_remaining_usd": round(
                        effective_remaining, 6
                    ),
                    "protected_tail": {
                        "allocated_usd": round(
                            sum(
                                allocation.protected_tail_usd
                                for allocation in self._allocations.values()
                            ),
                            6,
                        ),
                        "outstanding_usd": round(tail_outstanding, 6),
                        "physically_available_usd": round(
                            min(tail_outstanding, effective_remaining), 6
                        ),
                        "request_bound_overshoot_usd": round(
                            sum(
                                usage.request_bound_overshoot_usd
                                for role, usage in self._usage.items()
                                if self._allocations[role].group
                                == "implementation_verifier"
                            ),
                            6,
                        ),
                        "eroded_usd": round(
                            max(0.0, tail_outstanding - effective_remaining),
                            6,
                        ),
                        "owners": tail_owners,
                    },
                },
                "authority_revoked": self._authority_revoked,
                "revocation_reason": self._revocation_reason,
                "enforcement": {
                    "calls": "hard_pretransport",
                    "output_tokens": "post_hoc_lockout",
                    "wall_seconds": "post_hoc_lockout",
                    "usd": "request_bound_post_hoc_lockout",
                },
                "autonomous_development_metrics": {
                    "time_to_first_mutation_seconds": (
                        round(self._first_mutation_seconds, 6)
                        if self._first_mutation_seconds is not None
                        else None
                    ),
                    "cost_to_first_mutation_usd": (
                        round(self._cost_to_first_mutation_usd, 6)
                        if self._cost_to_first_mutation_usd is not None
                        else None
                    ),
                    "planning_to_implementation_spend_ratio": (
                        round(planning_spend / implementation_spend, 6)
                        if implementation_spend > 0.0
                        else None
                    ),
                    "calls_by_role": {
                        role: usage.calls
                        for role, usage in self._usage.items()
                    },
                    "validation_repair_count": (
                        self._validation_repair_count
                    ),
                    "source_calls_dispatched": (
                        self._source_calls_dispatched
                    ),
                    "source_calls_denied": self._source_calls_denied,
                    "implementation_reserve_remaining_usd": round(
                        base_remaining,
                        6,
                    ),
                    "implementation_authority_remaining_usd": round(
                        effective_remaining,
                        6,
                    ),
                },
                "roles": {
                    role: {
                        "group": self._allocations[role].group,
                        "limits": {
                            "calls": self._allocations[role].max_calls,
                            "output_tokens": (
                                self._allocations[role].max_output_tokens
                            ),
                            "wall_seconds": (
                                self._allocations[role].max_wall_seconds
                            ),
                            "cost_usd": self._allocations[role].max_cost_usd,
                            "protected_tail_usd": (
                                self._allocations[role].protected_tail_usd
                            ),
                        },
                        "usage": {
                            "calls": usage.calls,
                            "output_tokens": usage.output_tokens,
                            "wall_seconds": round(usage.wall_seconds, 6),
                            "cost_usd": round(usage.cost_usd, 6),
                            "reserved_cost_usd": round(
                                usage.reserved_cost_usd,
                                6,
                            ),
                            "in_flight": usage.in_flight,
                            "request_bound_overshoot_usd": round(
                                usage.request_bound_overshoot_usd,
                                6,
                            ),
                        },
                        "exhausted_by": usage.exhausted_by,
                        "accounting_debt": usage.accounting_debt,
                    }
                    for role, usage in self._usage.items()
                },
            }


def _allocation_exhausted_result(
    backend: Any,
    *,
    reason: str,
) -> InvocationResult:
    name = str(getattr(backend, "name", "unknown"))
    model = str(getattr(backend, "model", "unknown"))
    events = (
        BackendExecutionEvent(sequence=0, kind="limit", name=reason),
        BackendExecutionEvent(
            sequence=1,
            kind="terminal",
            name="budget_exhausted",
        ),
    )
    return InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(),
        stop_reason="budget_exhausted",
        error=BudgetExhausted(
            spent_usd=0.0,
            limit_usd=0.0,
            message=f"role allocation exhausted before transport: {reason}",
        ),
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend=name,
            requested_model=model,
            served_model=None,
            events=events,
        ),
        # This is one outer logical denial, not a provider transport. Its
        # exact zero usage therefore has complete pricing evidence with no
        # physical response segments.
        pricing_evidence=UsagePricingEvidence(segments=(), complete=True),
    )


class BudgetedBackend:
    """Backend proxy enforcing one role allocation at logical-call boundaries."""

    def __init__(
        self,
        backend: Any,
        *,
        role: str,
        ledger: DevelopmentBudgetLedger,
    ) -> None:
        self._backend = backend
        self._role = role
        self._ledger = ledger

    def invoke(self, invocation: Any) -> InvocationResult:
        admitted, reason, remaining_usd = self._ledger._begin(
            self._role,
            requested_cost_usd=invocation.budget_usd,
        )
        if not admitted:
            return _allocation_exhausted_result(
                self._backend,
                reason=reason or "unknown",
            )
        bounded_budget = (
            remaining_usd
            if invocation.budget_usd is None
            else min(float(invocation.budget_usd), remaining_usd)
        )
        bounded = replace(invocation, budget_usd=bounded_budget)
        started = time.monotonic()
        result: InvocationResult | None = None
        raised: BaseException | None = None
        try:
            result = self._backend.invoke(bounded)
            return result
        except BaseException as exc:
            raised = exc
            raise
        finally:
            self._ledger._settle(
                self._role,
                result=result,
                elapsed=time.monotonic() - started,
                raised=raised,
                reserved_cost_usd=remaining_usd,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


__all__ = [
    "BudgetedBackend",
    "DevelopmentBudgetLedger",
    "DevelopmentBudgetPolicy",
    "RoleAllocation",
]
