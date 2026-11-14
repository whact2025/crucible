# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-invocation backend limit negotiation and terminal evidence.

Feature admission answers whether a route exposes a workflow feature.  It
cannot make a requested turn or dollar ceiling enforceable.  This module owns
that separate UIR-03 boundary: callers present the exact invocation envelope
*before* adapter construction, the route's observed feature profile resolves
the envelope, and an invocation guard preserves the resolution in terminal
trace evidence. Canonical workflows additionally use a per-run write-ahead
ledger: the exact dispatch identity is fsync'd before transport and atomically
settled with terminal usage afterward.

``strict`` accepts only limits whose exact route advertises hard enforcement.
``diagnostic`` retains weaker routes as explicit evidence, disables transport
retry, and turns a post-call wall/USD overshoot into a non-retriable terminal.
The diagnostic path is useful for qualification; its downgrade record is not
an authorization to describe post-hoc accounting as a hard provider limit.
Incomplete or unpersisted diagnostic evidence revokes later call authority.
One explicit, fresh-run K3 policy may instead settle a separately typed cost
upper bound when only cache-read accounting is unknown. It never completes
actual accounting or grants ranking authority; failed or unpersisted bounds
still revoke continuation, including after reopening the durable ledger.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import secrets
import stat
import threading
import time
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendFeatureSet,
    BackendPolicy,
    BudgetExhausted,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    TransportError,
    Usage,
    UsageCostBounds,
)

from .cost_bounds import (
    DIAGNOSTIC_COST_POLICY,
    valid_diagnostic_cost_bounds,
)
from .features import feature_profile_digest


INVOCATION_LIMIT_SCHEMA = "crucible.backend-invocation-limits.v1"
INVOCATION_LIMIT_LEDGER_SCHEMA = (
    "crucible.backend-invocation-limit-ledger.v1"
)
INVOCATION_LIMIT_LEDGER_LEAF = "invocation-limits.json"
InvocationLimitMode = Literal["strict", "diagnostic"]
INVOCATION_LIMIT_MODES: frozenset[str] = frozenset(
    {"strict", "diagnostic"}
)
_TRACE_EVENT_PREFIX = "invocation_limits:"
_BOUND_ATTRIBUTE = "_crucible_invocation_limits_guarded"
_CHAT_BOUND_ATTRIBUTE = "_crucible_invocation_limits_chat_guarded"
_RESOLUTION_ATTRIBUTE = "_crucible_invocation_limit_resolution"
_TERMINAL_ATTRIBUTE = "_crucible_invocation_limit_terminal"
_POLICY_ATTRIBUTE = "_crucible_invocation_limit_policy"
_BINDING_ATTRIBUTE = "_crucible_invocation_limit_binding"
_EVIDENCE_FAILURE_ATTRIBUTE = "_crucible_invocation_limit_evidence_failed"


class InvocationLimitAdmissionError(ValueError):
    """The exact route cannot satisfy a requested invocation envelope."""


class InvocationLimitContractError(RuntimeError):
    """A dispatched invocation does not match its preconstructed envelope."""


def _validate_diagnostic_cost_policy(
    policy: str | None, *, mode: InvocationLimitMode, budgeted: bool,
) -> None:
    if policy is None:
        return
    if type(policy) is not str or policy != DIAGNOSTIC_COST_POLICY:
        raise ValueError("unknown diagnostic cost policy")
    if mode != "diagnostic" or not budgeted:
        raise ValueError(
            "diagnostic cost policy requires diagnostic mode and a positive "
            "finite USD budget"
        )


def _valid_invocation_cost_bounds(
    usage: Usage, bounds: UsageCostBounds | None = None,
) -> bool:
    bounds = bounds if bounds is not None else usage.cost_bounds
    return (
        type(bounds) is UsageCostBounds
        and valid_diagnostic_cost_bounds(usage, bounds)
        and all(segment.policy == DIAGNOSTIC_COST_POLICY for segment in bounds.segments)
    )


def _optional_budget(value: float | None, *, field_name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be None or a finite non-negative number"
        )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_strict_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one benchmark-authority ledger or fail before returning."""

    destination = Path(path)
    leaf = destination.name
    if leaf in {"", ".", ".."} or "/" in leaf or "\x00" in leaf:
        raise ValueError("invocation-limit ledger leaf is invalid")
    encoded = _canonical_json_bytes(payload)
    directory_fd = os.open(
        destination.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = f".tmp_{leaf}_{os.getpid()}_{secrets.token_hex(8)}"
    replaced = False
    try:
        file_fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(file_fd, view)
                if written <= 0:
                    raise OSError(
                        "invocation-limit ledger write made no progress"
                    )
                view = view[written:]
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.replace(
            temporary,
            leaf,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        os.fsync(directory_fd)
    finally:
        if not replaced:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


@dataclass(frozen=True)
class InvocationLimitRequest:
    """Exact hard-limit request known before backend construction."""

    mode: InvocationLimitMode
    max_turns: int
    budget_usd: float | None
    deadline_seconds: float
    diagnostic_cost_policy: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in INVOCATION_LIMIT_MODES:
            raise ValueError(
                f"unknown invocation limit mode {self.mode!r}; expected one "
                f"of {sorted(INVOCATION_LIMIT_MODES)}"
            )
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns < 1
        ):
            raise ValueError(
                "InvocationLimitRequest.max_turns must be an integer >= 1"
            )
        _optional_budget(
            self.budget_usd,
            field_name="InvocationLimitRequest.budget_usd",
        )
        _validate_diagnostic_cost_policy(
            self.diagnostic_cost_policy,
            mode=self.mode,
            budgeted=self.budget_usd is not None and self.budget_usd > 0,
        )
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0
        ):
            raise ValueError(
                "InvocationLimitRequest.deadline_seconds must be a finite "
                "number > 0"
            )


@dataclass(frozen=True)
class InvocationLimitPolicy:
    """Reusable workflow policy that resolves each actual invocation.

    Workflow pools reuse one adapter across calls whose remaining USD budget
    can change.  The policy therefore binds the invariant admission mode and
    deadline before construction, then derives the exact request from each
    :class:`assay.Invocation` immediately before transport.

    ``require_budget_usd`` lets a budgeted workflow fail closed during route
    admission instead of discovering only on its first paid call that the
    route cannot enforce a hard USD ceiling.
    """

    mode: InvocationLimitMode
    deadline_seconds: float
    require_budget_usd: bool = False
    diagnostic_cost_policy: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in INVOCATION_LIMIT_MODES:
            raise ValueError(
                f"unknown invocation limit mode {self.mode!r}; expected one "
                f"of {sorted(INVOCATION_LIMIT_MODES)}"
            )
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0
        ):
            raise ValueError(
                "InvocationLimitPolicy.deadline_seconds must be a finite "
                "number > 0"
            )
        if type(self.require_budget_usd) is not bool:
            raise TypeError(
                "InvocationLimitPolicy.require_budget_usd must be bool"
            )
        _validate_diagnostic_cost_policy(
            self.diagnostic_cost_policy,
            mode=self.mode,
            budgeted=self.require_budget_usd,
        )


class InvocationLimitState:
    """Workflow-local latch that forbids calls after uncertain diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revocation: InvocationLimitTerminal | None = None
        self._binding: tuple[str, InvocationLimitPolicy] | None = None

    def bind(self, *, run_key: str, policy: InvocationLimitPolicy) -> None:
        """Bind this revocation latch to one exact run and policy."""

        if not isinstance(run_key, str) or not run_key.strip():
            raise ValueError("InvocationLimitState.run_key must be non-empty")
        if type(policy) is not InvocationLimitPolicy:
            raise TypeError("policy must be InvocationLimitPolicy")
        binding = (run_key, policy)
        with self._lock:
            if self._binding is None:
                self._binding = binding
            elif self._binding != binding:
                raise InvocationLimitContractError(
                    "invocation-limit state is already bound to a different "
                    "run or workflow policy"
                )

    def require_call_authority(self) -> None:
        with self._lock:
            if self._revocation is not None:
                reason = (
                    "overshot its limit"
                    if self._revocation.overshoots
                    else "lost complete invocation-limit authority"
                )
                raise InvocationLimitContractError(
                    f"a prior diagnostic invocation {reason}; no recovery "
                    "provider call is authorized in this workflow"
                )

    def record(self, terminal: InvocationLimitTerminal) -> None:
        if terminal.revokes_diagnostic_authority:
            with self._lock:
                if self._revocation is None:
                    self._revocation = terminal

    def revoke(self, terminal: InvocationLimitTerminal) -> None:
        """Unconditionally revoke after durable evidence persistence fails."""

        with self._lock:
            if self._revocation is None:
                self._revocation = terminal

    @property
    def overshoot(self) -> InvocationLimitTerminal | None:
        with self._lock:
            value = self._revocation
            return value if value is not None and value.overshoots else None

    @property
    def revocation(self) -> InvocationLimitTerminal | None:
        with self._lock:
            return self._revocation


def _enforcement_location(value: str) -> str:
    return {
        "hard": "provider_hard",
        "provider_stop": "provider_stop",
        "round_boundary": "adapter_round_boundary",
        "post_hoc": "terminal_accounting",
        "hard_invocation": "adapter_invocation",
        "request_only": "request_process",
        "unsupported": "none",
    }.get(value, "none")


@dataclass(frozen=True)
class ResolvedInvocationLimits:
    """Pretransport resolution of one exact request against one route."""

    route: str
    mode: InvocationLimitMode
    requested_max_turns: int
    requested_budget_usd: float | None
    deadline_seconds: float
    admitted_max_turns: int | None
    admitted_budget_usd: float | None
    downgraded: tuple[str, ...]
    turn_enforcement: str
    turn_enforcement_location: str
    usd_enforcement: str
    usd_enforcement_location: str
    deadline_enforcement: str
    deadline_enforcement_location: str
    retry_budget: int | None
    feature_profile_digest: str
    schema_version: str = INVOCATION_LIMIT_SCHEMA
    diagnostic_cost_policy: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != INVOCATION_LIMIT_SCHEMA:
            raise ValueError(
                "ResolvedInvocationLimits.schema_version must be "
                f"{INVOCATION_LIMIT_SCHEMA!r}"
            )
        if self.mode not in INVOCATION_LIMIT_MODES:
            raise ValueError("ResolvedInvocationLimits.mode is invalid")
        if not isinstance(self.route, str) or not self.route.strip():
            raise ValueError("ResolvedInvocationLimits.route must be non-empty")
        if (
            type(self.requested_max_turns) is not int
            or self.requested_max_turns < 1
        ):
            raise ValueError(
                "ResolvedInvocationLimits.requested_max_turns must be an "
                "integer >= 1"
            )
        _optional_budget(
            self.requested_budget_usd,
            field_name="ResolvedInvocationLimits.requested_budget_usd",
        )
        _validate_diagnostic_cost_policy(
            self.diagnostic_cost_policy,
            mode=self.mode,
            budgeted=(
                self.requested_budget_usd is not None
                and self.requested_budget_usd > 0
            ),
        )
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0
        ):
            raise ValueError(
                "ResolvedInvocationLimits.deadline_seconds must be finite "
                "and positive"
            )
        if self.admitted_max_turns is not None and (
            type(self.admitted_max_turns) is not int
            or self.admitted_max_turns != self.requested_max_turns
        ):
            raise ValueError(
                "ResolvedInvocationLimits.admitted_max_turns must be None or "
                "the exact requested value"
            )
        _optional_budget(
            self.admitted_budget_usd,
            field_name="ResolvedInvocationLimits.admitted_budget_usd",
        )
        if (
            self.admitted_budget_usd is not None
            and self.admitted_budget_usd != self.requested_budget_usd
        ):
            raise ValueError(
                "ResolvedInvocationLimits.admitted_budget_usd must be None "
                "or the exact requested value"
            )
        if type(self.downgraded) is not tuple or self.downgraded not in {
            (),
            ("turn_limit",),
            ("usd_limit",),
            ("turn_limit", "usd_limit"),
        }:
            raise ValueError("ResolvedInvocationLimits.downgraded is invalid")
        if self.turn_enforcement not in BackendFeatureSet.allowed_values(
            "turn_limit"
        ):
            raise ValueError(
                "ResolvedInvocationLimits.turn_enforcement is invalid"
            )
        if self.usd_enforcement not in BackendFeatureSet.allowed_values(
            "usd_limit"
        ):
            raise ValueError(
                "ResolvedInvocationLimits.usd_enforcement is invalid"
            )
        if self.deadline_enforcement not in BackendFeatureSet.allowed_values(
            "deadline"
        ):
            raise ValueError(
                "ResolvedInvocationLimits.deadline_enforcement is invalid"
            )
        if self.turn_enforcement_location != _enforcement_location(
            self.turn_enforcement
        ):
            raise ValueError(
                "ResolvedInvocationLimits.turn_enforcement_location does not "
                "match turn_enforcement"
            )
        if self.usd_enforcement_location != _enforcement_location(
            self.usd_enforcement
        ):
            raise ValueError(
                "ResolvedInvocationLimits.usd_enforcement_location does not "
                "match usd_enforcement"
            )
        if self.deadline_enforcement_location != _enforcement_location(
            self.deadline_enforcement
        ):
            raise ValueError(
                "ResolvedInvocationLimits.deadline_enforcement_location does "
                "not match deadline_enforcement"
            )
        if self.retry_budget != 1:
            raise ValueError(
                "ResolvedInvocationLimits.retry_budget must be exactly one"
            )
        if (
            not isinstance(self.feature_profile_digest, str)
            or not self.feature_profile_digest.startswith("sha256:")
            or len(self.feature_profile_digest) != 71
            or any(
                character not in "0123456789abcdef"
                for character in self.feature_profile_digest[7:]
            )
        ):
            raise ValueError(
                "ResolvedInvocationLimits.feature_profile_digest is invalid"
            )
        expected_downgraded = (
            (() if self.turn_enforcement == "hard" else ("turn_limit",))
            + (
                ()
                if (
                    self.requested_budget_usd is None
                    or self.usd_enforcement == "hard"
                )
                else ("usd_limit",)
            )
        )
        if self.downgraded != expected_downgraded:
            raise ValueError(
                "ResolvedInvocationLimits.downgraded does not match the "
                "enforcement profile"
            )
        if self.mode == "strict" and (
            self.downgraded
            or self.deadline_enforcement != "hard_invocation"
        ):
            raise ValueError(
                "strict ResolvedInvocationLimits contains a downgraded "
                "limit or non-hard deadline"
            )
        if self.mode == "diagnostic" and self.deadline_enforcement not in {
            "hard_invocation",
            "request_only",
        }:
            raise ValueError(
                "diagnostic ResolvedInvocationLimits lacks an enforceable "
                "outer deadline"
            )
        if self.admitted_max_turns != (
            self.requested_max_turns
            if self.turn_enforcement == "hard"
            else None
        ):
            raise ValueError(
                "ResolvedInvocationLimits.admitted_max_turns does not match "
                "turn enforcement"
            )
        if self.admitted_budget_usd != (
            self.requested_budget_usd
            if self.usd_enforcement == "hard"
            else None
        ):
            raise ValueError(
                "ResolvedInvocationLimits.admitted_budget_usd does not match "
                "USD enforcement"
            )

    def to_record(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["downgraded"] = list(self.downgraded)
        if self.diagnostic_cost_policy is None:
            del record["diagnostic_cost_policy"]
        return record


@dataclass(frozen=True)
class InvocationLimitObservedUsage:
    """Exact result usage/completeness used to reconcile the durable ledger."""

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float
    token_usage_complete: bool
    cost_complete: bool
    terminal_complete: bool
    trace_complete: bool

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"InvocationLimitObservedUsage.{field_name} must be an "
                    "exact integer >= 0"
                )
        if self.cache_creation_tokens + self.cache_read_tokens > self.input_tokens:
            raise ValueError(
                "InvocationLimitObservedUsage cache-token subsets must not "
                "exceed input tokens"
            )
        if (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError(
                "InvocationLimitObservedUsage.cost_usd must be finite and "
                "non-negative"
            )
        for field_name in (
            "token_usage_complete",
            "cost_complete",
            "terminal_complete",
            "trace_complete",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(
                    f"InvocationLimitObservedUsage.{field_name} must be bool"
                )

    @classmethod
    def from_result(
        cls,
        result: InvocationResult,
    ) -> "InvocationLimitObservedUsage":
        usage = result.usage
        completeness = result.completeness
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=float(usage.cost_usd),
            token_usage_complete=completeness.token_usage_complete,
            cost_complete=completeness.cost_complete,
            terminal_complete=completeness.terminal_complete,
            trace_complete=completeness.trace_complete,
        )

    def to_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class InvocationLimitIdentity:
    """Secret-free workflow route identity bound before provider transport."""

    role: str
    backend: str
    model: str

    def __post_init__(self) -> None:
        for field_name in ("role", "backend", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"InvocationLimitIdentity.{field_name} must be non-empty"
                )

    def to_record(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class InvocationLimitTerminal:
    """Terminal per-call provenance retained in the normalized trace."""

    resolution: ResolvedInvocationLimits
    physical_attempts: int | None
    elapsed_seconds: float
    terminal_reason: str
    overshoots: tuple[str, ...]
    rankable: bool
    observed_usage: InvocationLimitObservedUsage | None
    cost_bounds: UsageCostBounds | None = None

    def __post_init__(self) -> None:
        if type(self.resolution) is not ResolvedInvocationLimits:
            raise TypeError(
                "InvocationLimitTerminal.resolution must be "
                "ResolvedInvocationLimits"
            )
        if self.physical_attempts is not None and (
            type(self.physical_attempts) is not int
            or self.physical_attempts < 0
        ):
            raise ValueError(
                "InvocationLimitTerminal.physical_attempts must be None or "
                "an exact integer >= 0"
            )
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError(
                "InvocationLimitTerminal.elapsed_seconds must be finite and "
                "non-negative"
            )
        if (
            not isinstance(self.terminal_reason, str)
            or not self.terminal_reason.strip()
        ):
            raise ValueError(
                "InvocationLimitTerminal.terminal_reason must be non-empty"
            )
        if type(self.overshoots) is not tuple or any(
            item not in {"deadline", "usd"}
            for item in self.overshoots
        ) or len(set(self.overshoots)) != len(self.overshoots):
            raise ValueError("InvocationLimitTerminal.overshoots is invalid")
        if type(self.rankable) is not bool:
            raise TypeError("InvocationLimitTerminal.rankable must be bool")
        if (
            self.observed_usage is not None
            and type(self.observed_usage) is not InvocationLimitObservedUsage
        ):
            raise TypeError(
                "InvocationLimitTerminal.observed_usage must be "
                "InvocationLimitObservedUsage or None"
            )
        if self.cost_bounds is not None and (
            self.resolution.diagnostic_cost_policy != DIAGNOSTIC_COST_POLICY
            or self.observed_usage is None
            or not _valid_invocation_cost_bounds(
                self._observed_assay_usage(), self.cost_bounds,
            )
        ):
            raise ValueError("invocation-limit cost bounds lack exact coverage")
        if self.cost_bounds is not None and self.observed_usage is not None and any(
            segment.cache_read_tokens is None for segment in self.cost_bounds.segments
        ) and (
            self.observed_usage.cost_complete
            or self.observed_usage.token_usage_complete
        ):
            raise ValueError("unknown cache reads cannot be complete actual accounting")
        if self.rankable and (
            self.resolution.diagnostic_cost_policy is not None
            or self.cost_bounds is not None
            or self.overshoots
            or self.observed_usage is None
            or not self.observed_usage.cost_complete
            or not self.observed_usage.token_usage_complete
            or not self.observed_usage.terminal_complete
            or not self.observed_usage.trace_complete
            or (
                self.resolution.mode == "diagnostic"
                and self.physical_attempts != 1
            )
        ):
            raise ValueError(
                "rankable invocation-limit terminal has incomplete or "
                "contradictory evidence"
            )

    def _observed_assay_usage(self) -> Usage:
        usage = self.observed_usage
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cost_usd=usage.cost_usd,
        )

    @property
    def diagnostic_cost_authority(self) -> bool:
        """A settled bound authorizes diagnostics, never complete accounting."""

        usage = self.observed_usage
        budget = self.resolution.requested_budget_usd
        return (
            self.resolution.diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
            and self.cost_bounds is not None
            and usage is not None
            and usage.terminal_complete
            and usage.trace_complete
            and self.physical_attempts == 1
            and self.terminal_reason == "diagnostic_cost_upper_bound"
            and not self.overshoots
            and self.elapsed_seconds <= self.resolution.deadline_seconds
            and budget is not None
            and self.cost_bounds.upper_bound_usd <= budget
        )

    @property
    def revokes_diagnostic_authority(self) -> bool:
        """Whether another provider call would make the evidence ambiguous."""

        if self.resolution.diagnostic_cost_policy is not None:
            return not self.diagnostic_cost_authority
        usage = self.observed_usage
        incomplete = (
            usage is None
            or not usage.cost_complete
            or not usage.token_usage_complete
            or not usage.terminal_complete
            or not usage.trace_complete
        )
        return (
            incomplete
            or self.terminal_reason == "limit_evidence_incomplete"
            or bool(self.overshoots)
        ) or (
            self.resolution.mode == "diagnostic"
            and self.physical_attempts != 1
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": INVOCATION_LIMIT_SCHEMA,
            "request": self.resolution.to_record(),
            "terminal": {
                "physical_attempts": self.physical_attempts,
                "elapsed_seconds": round(self.elapsed_seconds, 9),
                "terminal_reason": self.terminal_reason,
                "overshoots": list(self.overshoots),
                "rankable": self.rankable,
                "observed_usage": (
                    self.observed_usage.to_record()
                    if self.observed_usage is not None
                    else None
                ),
            },
        }
        if self.cost_bounds is not None:
            record["terminal"]["cost_bounds"] = self.cost_bounds.to_record()
        return record


@dataclass(frozen=True)
class InvocationLimitReservation:
    """Opaque write-ahead reservation for one potentially paid dispatch."""

    sequence: int
    invocation_id: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError(
                "InvocationLimitReservation.sequence must be an integer >= 1"
            )
        if (
            not isinstance(self.invocation_id, str)
            or len(self.invocation_id) != 32
            or any(
                character not in "0123456789abcdef"
                for character in self.invocation_id
            )
        ):
            raise ValueError(
                "InvocationLimitReservation.invocation_id must be 32 "
                "lowercase hexadecimal characters"
            )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _resolution_from_record(payload: Any) -> ResolvedInvocationLimits:
    expected = {field.name for field in dataclasses.fields(
        ResolvedInvocationLimits
    )}
    if not isinstance(payload, dict) or set(payload) not in (
        expected, expected - {"diagnostic_cost_policy"},
    ):
        raise ValueError("invocation-limit request record is not closed")
    downgraded = payload.get("downgraded")
    if not isinstance(downgraded, list):
        raise ValueError("invocation-limit downgraded record must be a list")
    values = dict(payload)
    values["downgraded"] = tuple(downgraded)
    resolution = ResolvedInvocationLimits(**values)
    if resolution.to_record() != payload:
        raise ValueError("invocation-limit request record is not canonical")
    return resolution


def _terminal_from_record(payload: Any) -> InvocationLimitTerminal:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "request", "terminal"}
        or payload.get("schema_version") != INVOCATION_LIMIT_SCHEMA
    ):
        raise ValueError("invocation-limit terminal record is not closed")
    resolution = _resolution_from_record(payload["request"])
    terminal_payload = payload["terminal"]
    expected_terminal = {
        "physical_attempts",
        "elapsed_seconds",
        "terminal_reason",
        "overshoots",
        "rankable",
        "observed_usage",
    }
    if not isinstance(terminal_payload, dict) or set(terminal_payload) not in (
        expected_terminal, expected_terminal | {"cost_bounds"},
    ):
        raise ValueError("invocation-limit terminal payload is not closed")
    overshoots = terminal_payload.get("overshoots")
    if not isinstance(overshoots, list):
        raise ValueError("invocation-limit overshoots must be a list")
    observed_payload = terminal_payload.get("observed_usage")
    observed: InvocationLimitObservedUsage | None
    if observed_payload is None:
        observed = None
    else:
        expected_usage = {
            field.name
            for field in dataclasses.fields(InvocationLimitObservedUsage)
        }
        if (
            not isinstance(observed_payload, dict)
            or set(observed_payload) != expected_usage
        ):
            raise ValueError(
                "invocation-limit observed usage record is not closed"
            )
        observed = InvocationLimitObservedUsage(**observed_payload)
    terminal = InvocationLimitTerminal(
        resolution=resolution,
        physical_attempts=terminal_payload["physical_attempts"],
        elapsed_seconds=terminal_payload["elapsed_seconds"],
        terminal_reason=terminal_payload["terminal_reason"],
        overshoots=tuple(overshoots),
        rankable=terminal_payload["rankable"],
        observed_usage=observed,
        cost_bounds=(
            UsageCostBounds.from_record(terminal_payload["cost_bounds"])
            if "cost_bounds" in terminal_payload else None
        ),
    )
    if terminal.to_record() != payload:
        raise ValueError("invocation-limit terminal record is not canonical")
    return terminal


class InvocationLimitLedger:
    """Crash-durable write-ahead ledger shared by one exact workflow run.

    A ``dispatched`` record is fsync'd before entering provider transport and
    is replaced in place by its typed terminal afterward. A process kill can
    therefore leave an explicit open intent, never an apparently complete
    ledger with a missing paid call.
    """

    def __init__(self, path: Path | str) -> None:
        candidate = Path(path)
        if candidate.name != INVOCATION_LIMIT_LEDGER_LEAF:
            raise ValueError(
                "invocation-limit ledger must use the canonical leaf name"
            )
        try:
            parent = candidate.parent.resolve(strict=True)
            if not parent.is_dir():
                raise ValueError(
                    "invocation-limit ledger parent must already exist"
                )
            self._path = parent / candidate.name
            try:
                existing = os.lstat(self._path)
            except FileNotFoundError:
                existing = None
            if existing is not None and (
                not stat.S_ISREG(existing.st_mode)
                or existing.st_nlink != 1
                or existing.st_uid != os.getuid()
            ):
                raise ValueError(
                    "existing invocation-limit ledger must be a regular, "
                    "single-link file owned by the current user"
                )
        except OSError as exc:
            raise ValueError(
                "invocation-limit ledger path is unavailable"
            ) from exc
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = []
        self._persisted_bytes: bytes | None = None
        self._poisoned_reason: str | None = None
        self._policy: InvocationLimitPolicy | None = None
        if self._path.exists():
            self._load()
        else:
            self._persist()

    @property
    def path(self) -> Path:
        return self._path

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": INVOCATION_LIMIT_LEDGER_SCHEMA,
            "records": list(self._records),
        }

    def _payload(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "ledger_fingerprint": _canonical_fingerprint(body)}

    @staticmethod
    def _decode(raw: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            payload = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_strict_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value!r}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "invocation-limit ledger is unreadable or malformed"
            ) from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {
                "schema_version",
                "records",
                "ledger_fingerprint",
            }
            or payload.get("schema_version") != INVOCATION_LIMIT_LEDGER_SCHEMA
            or not isinstance(payload.get("records"), list)
        ):
            raise ValueError(
                "invocation-limit ledger has an unknown shape"
            )
        body = {
            "schema_version": payload["schema_version"],
            "records": payload["records"],
        }
        if payload.get("ledger_fingerprint") != _canonical_fingerprint(body):
            raise ValueError(
                "invocation-limit ledger fingerprint does not match"
            )
        if raw != _canonical_json_bytes(payload):
            raise ValueError("invocation-limit ledger is not canonical JSON")
        for index, record in enumerate(payload["records"], start=1):
            if (
                not isinstance(record, dict)
                or set(record)
                != {"sequence", "state", "intent", "terminal"}
                or record.get("sequence") != index
            ):
                raise ValueError(
                    "invocation-limit ledger records are not contiguous and "
                    "closed"
                )
            state = record.get("state")
            intent = record.get("intent")
            if (
                state not in {"dispatched", "terminal"}
                or not isinstance(intent, dict)
                or set(intent)
                != {"invocation_id", "role", "backend", "model", "request"}
            ):
                raise ValueError(
                    "invocation-limit ledger state or intent is not closed"
                )
            reservation = InvocationLimitReservation(
                sequence=index,
                invocation_id=cast(str, intent.get("invocation_id")),
            )
            identity = InvocationLimitIdentity(
                role=cast(str, intent.get("role")),
                backend=cast(str, intent.get("backend")),
                model=cast(str, intent.get("model")),
            )
            resolution = _resolution_from_record(intent.get("request"))
            if resolution.diagnostic_cost_policy is not None and (
                identity.backend, identity.model
            ) != ("kimi", "kimi-k3"):
                raise ValueError("diagnostic cost policy has a different route identity")
            if intent != {
                "invocation_id": reservation.invocation_id,
                **identity.to_record(),
                "request": resolution.to_record(),
            }:
                raise ValueError(
                    "invocation-limit ledger intent is not canonical"
                )
            terminal_payload = record.get("terminal")
            if state == "dispatched":
                if terminal_payload is not None:
                    raise ValueError(
                        "dispatched invocation-limit intent has a terminal"
                    )
            else:
                terminal = _terminal_from_record(terminal_payload)
                if terminal.resolution != resolution:
                    raise ValueError(
                        "invocation-limit intent and terminal requests differ"
                    )
        return payload, list(payload["records"])

    def _load(self) -> None:
        try:
            raw = self._path.read_bytes()
            _payload, records = self._decode(raw)
        except Exception as exc:
            raise InvocationLimitContractError(
                "invocation-limit ledger is unreadable or malformed"
            ) from exc
        self._records = records
        self._persisted_bytes = raw
        if any(record["state"] == "dispatched" for record in records):
            self._poisoned_reason = (
                "the invocation-limit ledger contains a prior unsettled "
                "provider dispatch"
            )
        elif any(
            _terminal_from_record(record["terminal"]).revokes_diagnostic_authority
            for record in records
        ):
            self._poisoned_reason = (
                "the invocation-limit ledger retains revoked diagnostic authority"
            )

    def _persist(self) -> None:
        if self._persisted_bytes is not None:
            try:
                observed = self._path.read_bytes()
            except OSError as exc:
                raise InvocationLimitContractError(
                    "invocation-limit ledger disappeared before mutation"
                ) from exc
            if observed != self._persisted_bytes:
                raise InvocationLimitContractError(
                    "invocation-limit ledger changed through another owner; "
                    "refusing to overwrite concurrent evidence"
                )
        payload = self._payload()
        _write_strict_atomic_json(self._path, payload)
        self._persisted_bytes = _canonical_json_bytes(payload)

    def bind_policy(self, policy: InvocationLimitPolicy) -> None:
        """Bind one immutable workflow policy to this exact-run ledger."""

        if type(policy) is not InvocationLimitPolicy:
            raise TypeError("policy must be InvocationLimitPolicy")
        with self._lock:
            if self._policy is not None:
                if self._policy != policy:
                    raise InvocationLimitContractError(
                        "invocation-limit ledger is already bound to a "
                        "different workflow policy"
                    )
                return
            for record in self._records:
                request = _resolution_from_record(record["intent"]["request"])
                if (
                    request.mode != policy.mode
                    or request.diagnostic_cost_policy != policy.diagnostic_cost_policy
                    or request.deadline_seconds != policy.deadline_seconds
                    or (
                        policy.require_budget_usd
                        and request.requested_budget_usd is None
                    )
                ):
                    raise InvocationLimitContractError(
                        "invocation-limit ledger evidence does not match the "
                        "workflow policy"
                    )
            self._policy = policy

    def arm(
        self,
        identity: InvocationLimitIdentity,
        resolution: ResolvedInvocationLimits,
    ) -> InvocationLimitReservation:
        """Persist one open dispatch intent before provider transport."""

        if type(identity) is not InvocationLimitIdentity:
            raise TypeError("identity must be InvocationLimitIdentity")
        if type(resolution) is not ResolvedInvocationLimits:
            raise TypeError("resolution must be ResolvedInvocationLimits")
        with self._lock:
            if self._poisoned_reason is not None:
                raise InvocationLimitContractError(
                    f"{self._poisoned_reason}; no provider call is authorized"
                )
            if self._policy is not None and (
                resolution.mode != self._policy.mode
                or resolution.diagnostic_cost_policy
                != self._policy.diagnostic_cost_policy
                or resolution.deadline_seconds
                != self._policy.deadline_seconds
                or (
                    self._policy.require_budget_usd
                    and resolution.requested_budget_usd is None
                )
            ):
                raise InvocationLimitContractError(
                    "invocation-limit request does not match the exact-run "
                    "ledger policy"
                )
            if any(
                _resolution_from_record(record["intent"]["request"])
                .diagnostic_cost_policy != resolution.diagnostic_cost_policy
                for record in self._records
            ):
                raise InvocationLimitContractError(
                    "diagnostic cost policy cannot reinterpret prior ledger evidence"
                )
            if resolution.diagnostic_cost_policy is not None:
                if (identity.backend, identity.model) != ("kimi", "kimi-k3"):
                    raise InvocationLimitContractError(
                        "diagnostic cost policy requires exact Kimi K3 identity"
                    )
                if any(record["state"] == "dispatched" for record in self._records):
                    raise InvocationLimitContractError(
                        "diagnostic cost policy requires durable prior settlement"
                    )
            invocation_id = secrets.token_hex(16)
            existing_ids = {
                record["intent"]["invocation_id"] for record in self._records
            }
            while invocation_id in existing_ids:
                invocation_id = secrets.token_hex(16)
            reservation = InvocationLimitReservation(
                sequence=len(self._records) + 1,
                invocation_id=invocation_id,
            )
            self._records.append(
                {
                    "sequence": reservation.sequence,
                    "state": "dispatched",
                    "intent": {
                        "invocation_id": reservation.invocation_id,
                        **identity.to_record(),
                        "request": resolution.to_record(),
                    },
                    "terminal": None,
                }
            )
            try:
                self._persist()
            except BaseException:
                self._records.pop()
                self._poisoned_reason = (
                    "the invocation-limit dispatch intent could not be "
                    "durably persisted"
                )
                raise
            return reservation

    def settle(
        self,
        reservation: InvocationLimitReservation,
        terminal: InvocationLimitTerminal,
    ) -> None:
        """Atomically replace one open intent with its typed terminal."""

        if type(reservation) is not InvocationLimitReservation:
            raise TypeError("reservation must be InvocationLimitReservation")
        if type(terminal) is not InvocationLimitTerminal:
            raise TypeError("terminal must be InvocationLimitTerminal")
        with self._lock:
            index = reservation.sequence - 1
            if index < 0 or index >= len(self._records):
                raise InvocationLimitContractError(
                    "invocation-limit reservation sequence is unknown"
                )
            prior = self._records[index]
            if (
                prior["state"] != "dispatched"
                or prior["terminal"] is not None
                or prior["intent"]["invocation_id"]
                != reservation.invocation_id
            ):
                raise InvocationLimitContractError(
                    "invocation-limit reservation is not the exact open intent"
                )
            if prior["intent"]["request"] != terminal.resolution.to_record():
                raise InvocationLimitContractError(
                    "invocation-limit terminal does not match its armed request"
                )
            settled = {
                **prior,
                "state": "terminal",
                "terminal": terminal.to_record(),
            }
            self._records[index] = settled
            try:
                self._persist()
            except BaseException:
                self._records[index] = prior
                self._poisoned_reason = (
                    "the invocation-limit terminal could not be durably "
                    "persisted"
                )
                raise
            if terminal.revokes_diagnostic_authority:
                self._poisoned_reason = (
                    "the invocation-limit ledger retains revoked diagnostic authority"
                )

    def snapshot(self) -> Mapping[str, Any]:
        with self._lock:
            return copy.deepcopy(self._payload())


def resolve_invocation_limits(
    profile: BackendFeatureSet,
    request: InvocationLimitRequest,
    *,
    route: str,
) -> ResolvedInvocationLimits:
    """Resolve ``request`` without constructing or contacting an adapter."""

    if type(profile) is not BackendFeatureSet:
        raise TypeError("profile must be an assay BackendFeatureSet")
    if type(request) is not InvocationLimitRequest:
        raise TypeError("request must be an InvocationLimitRequest")
    if not isinstance(route, str) or not route.strip():
        raise ValueError("route must be a non-empty string")

    turn_hard = profile.turn_limit == "hard"
    usd_hard = request.budget_usd is None or profile.usd_limit == "hard"
    downgraded: list[str] = []
    if not turn_hard:
        downgraded.append("turn_limit")
    if request.budget_usd is not None and not usd_hard:
        downgraded.append("usd_limit")

    if request.mode == "strict" and downgraded:
        details = []
        if "turn_limit" in downgraded:
            details.append(
                "requested hard turn ceiling cannot be enforced "
                f"(turn_limit={profile.turn_limit!r})"
            )
        if "usd_limit" in downgraded:
            details.append(
                "requested hard USD ceiling cannot be enforced "
                f"(usd_limit={profile.usd_limit!r}; post-call accounting "
                "is not hard enforcement)"
            )
        raise InvocationLimitAdmissionError(
            f"backend route {route} rejected strict invocation limits before "
            f"adapter construction or transport: {'; '.join(details)}"
        )

    if request.mode == "strict" and profile.deadline != "hard_invocation":
        raise InvocationLimitAdmissionError(
            f"backend route {route} rejected strict invocation limits before "
            "adapter construction or transport: requested hard invocation "
            "deadline cannot be enforced "
            f"(deadline={profile.deadline!r})"
        )

    if request.mode == "diagnostic" and profile.deadline not in {
        "hard_invocation",
        "request_only",
    }:
        raise InvocationLimitAdmissionError(
            f"backend route {route} cannot bind the diagnostic invocation to "
            "one monotonic wall/process deadline before transport "
            f"(deadline={profile.deadline!r})"
        )

    return ResolvedInvocationLimits(
        route=route,
        mode=request.mode,
        requested_max_turns=request.max_turns,
        requested_budget_usd=request.budget_usd,
        deadline_seconds=float(request.deadline_seconds),
        admitted_max_turns=(request.max_turns if turn_hard else None),
        admitted_budget_usd=(request.budget_usd if usd_hard else None),
        downgraded=tuple(downgraded),
        turn_enforcement=profile.turn_limit,
        turn_enforcement_location=_enforcement_location(profile.turn_limit),
        usd_enforcement=profile.usd_limit,
        usd_enforcement_location=_enforcement_location(profile.usd_limit),
        deadline_enforcement=profile.deadline,
        deadline_enforcement_location=_enforcement_location(profile.deadline),
        retry_budget=1,
        feature_profile_digest=feature_profile_digest(profile),
        diagnostic_cost_policy=request.diagnostic_cost_policy,
    )


def preflight_invocation_limit_policy(
    profile: BackendFeatureSet,
    policy: InvocationLimitPolicy,
    *,
    route: str,
) -> None:
    """Admit a reusable policy before adapter construction.

    Every invocation has a turn ceiling, so a strict workflow can reject a
    route lacking hard turn enforcement immediately.  A workflow that has a
    cumulative/per-task USD authority sets ``require_budget_usd`` and obtains
    the same preconstruction rejection for a non-hard USD profile.  The
    synthetic values are used only to exercise feature admission; terminal
    evidence always records the real invocation values.
    """

    if type(policy) is not InvocationLimitPolicy:
        raise TypeError("policy must be InvocationLimitPolicy")
    resolve_invocation_limits(
        profile,
        InvocationLimitRequest(
            mode=policy.mode,
            max_turns=1,
            budget_usd=(
                1.0 if policy.diagnostic_cost_policy is not None
                else 0.0 if policy.require_budget_usd else None
            ),
            deadline_seconds=policy.deadline_seconds,
            diagnostic_cost_policy=policy.diagnostic_cost_policy,
        ),
        route=route,
    )


def _bounded_policy(
    invocation: Invocation,
    resolution: ResolvedInvocationLimits,
) -> BackendPolicy:
    policy = invocation.policy
    timeout = resolution.deadline_seconds
    if policy is not None and policy.timeout_seconds is not None:
        timeout = min(timeout, float(policy.timeout_seconds))
    if policy is None:
        return BackendPolicy(
            retry_budget=1,
            timeout_seconds=timeout,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        )
    return replace(
        policy,
        retry_budget=1,
        timeout_seconds=timeout,
    )


def _validate_invocation_envelope(
    invocation: Any,
    resolution: ResolvedInvocationLimits,
) -> Invocation:
    if type(invocation) is not Invocation:
        raise InvocationLimitContractError(
            "invocation-limit guard requires an exact assay.Invocation"
        )
    mismatches: list[str] = []
    if invocation.max_turns != resolution.requested_max_turns:
        mismatches.append(
            f"max_turns={invocation.max_turns!r} (bound "
            f"{resolution.requested_max_turns!r})"
        )
    if invocation.budget_usd != resolution.requested_budget_usd:
        mismatches.append(
            f"budget_usd={invocation.budget_usd!r} (bound "
            f"{resolution.requested_budget_usd!r})"
        )
    if mismatches:
        raise InvocationLimitContractError(
            "invocation differs from the preconstruction limit envelope; "
            "refusing transport: " + "; ".join(mismatches)
        )
    return invocation


def _trace_with_terminal(
    result: InvocationResult,
    terminal: InvocationLimitTerminal,
    *,
    backend_name: str,
    model: str,
) -> InvocationResult:
    record = json.dumps(
        terminal.to_record(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    limit_event = BackendExecutionEvent(
        sequence=0,
        kind="limit",
        name=_TRACE_EVENT_PREFIX + record,
    )
    trace = result.execution_trace
    if trace is None:
        trace = BackendExecutionTrace(
            backend=backend_name,
            requested_model=model,
            feature_profile_digest=terminal.resolution.feature_profile_digest,
        )
    events = list(trace.events)
    prior_terminal = next(
        (event for event in reversed(events) if event.kind == "terminal"),
        None,
    )
    events = [event for event in events if event.kind != "terminal"]
    events.append(limit_event)
    events.append(
        replace(prior_terminal, name=result.stop_reason)
        if prior_terminal is not None
        else BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name=result.stop_reason,
        )
    )
    normalized = tuple(
        replace(event, sequence=index)
        for index, event in enumerate(events)
    )
    return replace(result, execution_trace=replace(trace, events=normalized))


def _overshoot_result(
    result: InvocationResult,
    terminal: InvocationLimitTerminal,
) -> InvocationResult:
    resolution = terminal.resolution
    if "usd" in terminal.overshoots:
        limit = cast(float, resolution.requested_budget_usd)
        spent = (
            terminal.cost_bounds.upper_bound_usd
            if terminal.cost_bounds is not None else result.usage.cost_usd
        )
        subject = (
            "diagnostic cost upper bound"
            if terminal.cost_bounds is not None else "provider call"
        )
        return replace(
            result,
            stop_reason="budget_exhausted",
            error=BudgetExhausted(
                spent_usd=float(spent),
                limit_usd=limit,
                message=(
                    f"{subject} exceeded its indivisible USD "
                    f"ceiling ({spent:.9f} > {limit:.9f}); "
                    "the result is terminal and non-rankable"
                ),
            ),
        )
    return replace(
        result,
        stop_reason="timeout",
        error=TransportError(
            message=(
                "provider call returned after its one outer "
                f"monotonic deadline ({terminal.elapsed_seconds:.9f}s > "
                f"{resolution.deadline_seconds:.9f}s); the result is "
                "terminal and non-rankable"
            ),
            retriable=False,
        ),
    )


def _physical_attempt_count(result: InvocationResult) -> int | None:
    trace = result.execution_trace
    if trace is None:
        return None
    requests = sum(event.kind == "request" for event in trace.events)
    if requests or result.completeness.trace_complete:
        return requests
    return None


def _attempt_evidence_result(
    result: InvocationResult,
    *,
    physical_attempts: int | None,
) -> InvocationResult:
    if physical_attempts is None:
        return replace(
            result,
            stop_reason="limit_evidence_incomplete",
            error=IncompleteEvidence(
                missing=("invocation_limits.physical_attempts",),
                message=(
                    "diagnostic invocation cannot prove that exactly one "
                    "physical provider attempt occurred; the result is "
                    "terminal and non-rankable"
                ),
            ),
        )
    return replace(
        result,
        stop_reason="limit_contract_violated",
        error=TransportError(
            message=(
                "diagnostic invocation recorded "
                f"{physical_attempts} physical provider attempts after "
                "retry_budget=1; the result is terminal and non-rankable"
            ),
            retriable=False,
        ),
    )


def _incomplete_limit_evidence_result(
    result: InvocationResult,
    *,
    physical_attempts: int | None,
) -> InvocationResult:
    """Convert nominal success with incomplete limit evidence to a failure."""

    missing: list[str] = []
    if physical_attempts is None:
        missing.append("invocation_limits.physical_attempts")
    completeness = result.completeness
    for field_name in (
        "token_usage_complete",
        "cost_complete",
        "terminal_complete",
        "trace_complete",
    ):
        if not getattr(completeness, field_name):
            missing.append(f"invocation_limits.{field_name}")
    if not missing:
        missing.append("invocation_limits.diagnostic_cost_authority")
    return replace(
        result,
        stop_reason="limit_evidence_incomplete",
        error=IncompleteEvidence(
            missing=tuple(missing),
            message=(
                "diagnostic invocation returned without complete terminal, "
                "trace, token, cost, or physical-attempt evidence; the "
                "result is terminal and non-rankable"
            ),
        ),
    )


def _retain_terminal(
    backend: Any,
    terminal: InvocationLimitTerminal,
    *,
    state_lock: Any,
    shared_state: InvocationLimitState | None,
    evidence_ledger: InvocationLimitLedger | None,
    reservation: InvocationLimitReservation | None,
) -> None:
    with state_lock:
        setattr(backend, _RESOLUTION_ATTRIBUTE, terminal.resolution)
        setattr(backend, _TERMINAL_ATTRIBUTE, terminal)
    if shared_state is not None:
        shared_state.record(terminal)
    if evidence_ledger is None:
        if reservation is not None:
            raise InvocationLimitContractError(
                "invocation-limit reservation has no evidence ledger"
            )
        return
    if reservation is None:
        raise InvocationLimitContractError(
            "invocation-limit evidence ledger has no dispatch reservation"
        )
    try:
        evidence_ledger.settle(reservation, terminal)
    except BaseException as exc:
        with state_lock:
            setattr(backend, _EVIDENCE_FAILURE_ATTRIBUTE, True)
        if shared_state is not None:
            shared_state.revoke(terminal)
        raise InvocationLimitContractError(
            "invocation-limit terminal could not be durably persisted; no "
            "further provider call is authorized"
        ) from exc


def _failure_terminal(
    resolution: ResolvedInvocationLimits,
    *,
    elapsed_seconds: float,
    terminal_reason: str,
) -> InvocationLimitTerminal:
    """Construct a conservative terminal when result normalization fails."""

    return InvocationLimitTerminal(
        resolution=resolution,
        physical_attempts=None,
        elapsed_seconds=elapsed_seconds,
        terminal_reason=terminal_reason,
        overshoots=(
            ("deadline",)
            if elapsed_seconds > resolution.deadline_seconds
            else ()
        ),
        rankable=False,
        observed_usage=None,
    )


def _normalize_returned_result(
    result: InvocationResult,
    resolution: ResolvedInvocationLimits,
    *,
    elapsed: float,
    backend_name: str,
    model: str,
) -> tuple[InvocationResult, InvocationLimitTerminal]:
    """Build the typed terminal and normalized result before settlement."""

    physical_attempts = _physical_attempt_count(result)
    bound_policy = resolution.diagnostic_cost_policy is not None
    cost_bounds = (
        result.usage.cost_bounds
        if bound_policy and _valid_invocation_cost_bounds(result.usage)
        else None
    )
    trace = result.execution_trace
    bound_covered = (
        cost_bounds is not None
        and result.error is None
        and result.stop_reason == "end_turn"
        and result.completeness.terminal_complete
        and result.completeness.trace_complete
        and physical_attempts == 1
        and trace is not None
        and (trace.backend, trace.requested_model, trace.served_model)
        == ("kimi", "kimi-k3", "kimi-k3")
        and trace.requested_effort == "max"
        and trace.endpoint == "https://api.moonshot.ai/v1"
        and trace.feature_profile_digest == resolution.feature_profile_digest
        and trace.terminal_event_count == 1
        and trace.events[-1].kind == "terminal"
        and trace.events[-1].name == "end_turn"
    )
    overshoots: list[str] = []
    if (
        elapsed > resolution.deadline_seconds
    ):
        overshoots.append("deadline")
    requested_budget = resolution.requested_budget_usd
    if (
        requested_budget is not None
        and (
            (cost_bounds is not None
             and cost_bounds.upper_bound_usd > requested_budget)
            or (result.completeness.cost_complete
                and result.usage.cost_usd > requested_budget)
        )
    ):
        overshoots.append("usd")
    incomplete_evidence = (
        (not result.completeness.complete and not bound_covered)
        or (bound_policy and not bound_covered)
        or (not bound_policy and result.usage.cost_bounds is not None)
    )
    terminal_reason = result.stop_reason
    if bound_covered:
        terminal_reason = "diagnostic_cost_upper_bound"
    if "usd" in overshoots:
        terminal_reason = "budget_overshoot"
    elif "deadline" in overshoots:
        terminal_reason = "deadline_overshoot"
    elif (
        resolution.mode == "diagnostic"
        and physical_attempts is not None
        and physical_attempts != 1
    ):
        terminal_reason = "physical_attempt_contract_violated"
    elif resolution.mode == "diagnostic" and (
        physical_attempts is None or incomplete_evidence
    ):
        terminal_reason = "limit_evidence_incomplete"
    accounting_proved = (
        requested_budget is None or result.completeness.cost_complete
    )
    terminal = InvocationLimitTerminal(
        resolution=resolution,
        physical_attempts=physical_attempts,
        elapsed_seconds=elapsed,
        terminal_reason=terminal_reason,
        overshoots=tuple(overshoots),
        rankable=(
            not overshoots
            and not bound_policy
            and result.usage.cost_bounds is None
            and (
                resolution.mode != "diagnostic"
                or physical_attempts == 1
            )
            and accounting_proved
            and result.error is None
            and result.completeness.complete
        ),
        observed_usage=InvocationLimitObservedUsage.from_result(result),
        cost_bounds=cost_bounds,
    )
    if overshoots:
        result = _overshoot_result(result, terminal)
    elif (
        resolution.mode == "diagnostic"
        and physical_attempts is not None
        and physical_attempts != 1
    ):
        result = _attempt_evidence_result(
            result,
            physical_attempts=physical_attempts,
        )
    elif (
        resolution.mode == "diagnostic"
        and result.error is None
        and (physical_attempts is None or incomplete_evidence)
    ):
        result = _incomplete_limit_evidence_result(
            result,
            physical_attempts=physical_attempts,
        )
    return (
        _trace_with_terminal(
            result,
            terminal,
            backend_name=backend_name,
            model=model,
        ),
        terminal,
    )


def _validate_diagnostic_backend_binding(
    backend: Any,
    diagnostic_cost_policy: str | None,
    evidence_ledger: InvocationLimitLedger | None,
    identity: InvocationLimitIdentity | None,
) -> None:
    enabled = diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
    adapter_flag = getattr(backend, "diagnostic_cost_upper_bound", False)
    if type(adapter_flag) is not bool or adapter_flag is not enabled:
        raise InvocationLimitContractError(
            "diagnostic cost policy differs from the adapter's explicit opt-in"
        )
    if not enabled:
        return
    if evidence_ledger is None or identity is None:
        raise InvocationLimitContractError(
            "diagnostic cost policy requires durable invocation-limit evidence"
        )
    if (
        getattr(backend, "name", None),
        getattr(backend, "model", None),
        getattr(backend, "effort", None),
        identity.backend,
        identity.model,
    ) != ("kimi", "kimi-k3", "max", "kimi", "kimi-k3"):
        raise InvocationLimitContractError(
            "diagnostic cost policy requires the exact Kimi K3/max route"
        )


def _invoke_with_resolution(
    backend: Any,
    original_invoke: Any,
    invocation: Invocation,
    resolution: ResolvedInvocationLimits,
    *,
    state_lock: Any,
    shared_state: InvocationLimitState | None,
    backend_name: str,
    model: str,
    evidence_ledger: InvocationLimitLedger | None,
    identity: InvocationLimitIdentity | None,
) -> InvocationResult:
    """Execute one resolved envelope and retain its terminal evidence."""

    _validate_diagnostic_backend_binding(
        backend, resolution.diagnostic_cost_policy, evidence_ledger, identity,
    )
    dispatched = replace(
        invocation,
        policy=_bounded_policy(invocation, resolution),
    )
    if (evidence_ledger is None) != (identity is None):
        raise InvocationLimitContractError(
            "invocation-limit evidence ledger and identity must be bound "
            "together"
        )
    reservation = (
        evidence_ledger.arm(cast(InvocationLimitIdentity, identity), resolution)
        if evidence_ledger is not None
        else None
    )
    try:
        started = time.monotonic()
    except BaseException:
        failed = _failure_terminal(
            resolution,
            elapsed_seconds=0.0,
            terminal_reason="limit_clock_failure",
        )
        _retain_terminal(
            backend,
            failed,
            state_lock=state_lock,
            shared_state=shared_state,
            evidence_ledger=evidence_ledger,
            reservation=reservation,
        )
        raise
    try:
        result = original_invoke(dispatched)
    except BaseException:
        try:
            elapsed = max(0.0, time.monotonic() - started)
        except BaseException:
            elapsed = 0.0
        failed = _failure_terminal(
            resolution,
            elapsed_seconds=elapsed,
            terminal_reason="backend_exception",
        )
        _retain_terminal(
            backend,
            failed,
            state_lock=state_lock,
            shared_state=shared_state,
            evidence_ledger=evidence_ledger,
            reservation=reservation,
        )
        raise
    if type(result) is not InvocationResult:
        try:
            elapsed = max(0.0, time.monotonic() - started)
        except BaseException:
            elapsed = 0.0
        failed = _failure_terminal(
            resolution,
            elapsed_seconds=elapsed,
            terminal_reason="backend_contract_violation",
        )
        _retain_terminal(
            backend,
            failed,
            state_lock=state_lock,
            shared_state=shared_state,
            evidence_ledger=evidence_ledger,
            reservation=reservation,
        )
        raise InvocationLimitContractError(
            "backend invoke returned a non-InvocationResult value"
        )
    elapsed = 0.0
    try:
        elapsed = max(0.0, time.monotonic() - started)
        result, terminal = _normalize_returned_result(
            result,
            resolution,
            elapsed=elapsed,
            backend_name=backend_name,
            model=model,
        )
    except BaseException:
        failed = _failure_terminal(
            resolution,
            elapsed_seconds=elapsed,
            terminal_reason="limit_normalization_failure",
        )
        _retain_terminal(
            backend,
            failed,
            state_lock=state_lock,
            shared_state=shared_state,
            evidence_ledger=evidence_ledger,
            reservation=reservation,
        )
        raise
    _retain_terminal(
        backend,
        terminal,
        state_lock=state_lock,
        shared_state=shared_state,
        evidence_ledger=evidence_ledger,
        reservation=reservation,
    )
    return result


def _raise_if_terminal_latched(backend: Any) -> None:
    if getattr(backend, _EVIDENCE_FAILURE_ATTRIBUTE, False) is True:
        raise InvocationLimitContractError(
            "a prior invocation-limit terminal could not be durably "
            "persisted; no recovery provider call is authorized on this "
            "backend"
        )
    prior = getattr(backend, _TERMINAL_ATTRIBUTE, None)
    if type(prior) is not InvocationLimitTerminal:
        return
    if prior.overshoots:
        raise InvocationLimitContractError(
            "a prior diagnostic invocation overshot its limit; no recovery "
            "provider call is authorized on this backend"
        )
    if prior.revokes_diagnostic_authority:
        raise InvocationLimitContractError(
            "a prior invocation lost complete invocation-limit "
            "authority; no recovery provider call is authorized on this "
            "backend"
        )


def _block_alternate_chat_boundary(backend: Any) -> None:
    original_chat = getattr(backend, "chat_with_tools", None)
    if callable(original_chat) and not getattr(
        backend,
        _CHAT_BOUND_ATTRIBUTE,
        False,
    ):
        @wraps(original_chat)
        def blocked_chat(*_args: Any, **_kwargs: Any) -> InvocationResult:
            raise InvocationLimitContractError(
                "an invocation-limit contract must use the vendor-neutral "
                "invoke boundary; chat_with_tools would bypass its exact "
                "request envelope"
            )

        backend.chat_with_tools = blocked_chat
        setattr(backend, _CHAT_BOUND_ATTRIBUTE, True)


def bind_invocation_limit_contract(
    backend: Any,
    resolution: ResolvedInvocationLimits,
    *,
    evidence_ledger: InvocationLimitLedger | None = None,
    identity: InvocationLimitIdentity | None = None,
) -> None:
    """Bind one preconstructed resolution to an adapter's invoke boundary."""

    if type(resolution) is not ResolvedInvocationLimits:
        raise TypeError("resolution must be ResolvedInvocationLimits")
    if evidence_ledger is not None and type(evidence_ledger) is not InvocationLimitLedger:
        raise TypeError("evidence_ledger must be InvocationLimitLedger or None")
    if identity is not None and type(identity) is not InvocationLimitIdentity:
        raise TypeError("identity must be InvocationLimitIdentity or None")
    if (evidence_ledger is None) != (identity is None):
        raise ValueError("evidence_ledger and identity must be provided together")
    _validate_diagnostic_backend_binding(
        backend, resolution.diagnostic_cost_policy, evidence_ledger, identity,
    )
    binding = ("resolution", resolution, evidence_ledger, identity)
    if getattr(backend, _BOUND_ATTRIBUTE, False):
        if getattr(backend, _BINDING_ATTRIBUTE, None) != binding:
            raise InvocationLimitContractError(
                "backend already carries a different invocation limit "
                "contract"
            )
        return
    original_invoke = getattr(backend, "invoke", None)
    if not callable(original_invoke):
        raise InvocationLimitContractError(
            "backend has no callable invoke boundary for limit enforcement"
        )
    state_lock = threading.RLock()
    backend_name = str(getattr(backend, "name", "unknown")) or "unknown"
    model = str(getattr(backend, "model", "unknown")) or "unknown"

    @wraps(original_invoke)
    def guarded_invoke(invocation: Any) -> InvocationResult:
        with state_lock:
            _raise_if_terminal_latched(backend)
            exact = _validate_invocation_envelope(invocation, resolution)
            return _invoke_with_resolution(
                backend,
                original_invoke,
                exact,
                resolution,
                state_lock=state_lock,
                shared_state=None,
                backend_name=backend_name,
                model=model,
                evidence_ledger=evidence_ledger,
                identity=identity,
            )

    setattr(backend, _RESOLUTION_ATTRIBUTE, resolution)
    setattr(backend, _TERMINAL_ATTRIBUTE, None)
    setattr(backend, _EVIDENCE_FAILURE_ATTRIBUTE, False)
    setattr(backend, _BINDING_ATTRIBUTE, binding)
    backend.invoke = guarded_invoke
    setattr(backend, _BOUND_ATTRIBUTE, True)
    _block_alternate_chat_boundary(backend)


def bind_invocation_limit_policy(
    backend: Any,
    profile: BackendFeatureSet,
    policy: InvocationLimitPolicy,
    *,
    route: str,
    shared_state: InvocationLimitState | None = None,
    evidence_ledger: InvocationLimitLedger | None = None,
    identity: InvocationLimitIdentity | None = None,
) -> None:
    """Bind a reusable workflow policy to the adapter invoke boundary."""

    if type(profile) is not BackendFeatureSet:
        raise TypeError("profile must be an assay BackendFeatureSet")
    if type(policy) is not InvocationLimitPolicy:
        raise TypeError("policy must be InvocationLimitPolicy")
    if not isinstance(route, str) or not route.strip():
        raise ValueError("route must be a non-empty string")
    if shared_state is not None and type(shared_state) is not InvocationLimitState:
        raise TypeError("shared_state must be InvocationLimitState or None")
    if evidence_ledger is not None and type(evidence_ledger) is not InvocationLimitLedger:
        raise TypeError("evidence_ledger must be InvocationLimitLedger or None")
    if identity is not None and type(identity) is not InvocationLimitIdentity:
        raise TypeError("identity must be InvocationLimitIdentity or None")
    if (evidence_ledger is None) != (identity is None):
        raise ValueError("evidence_ledger and identity must be provided together")
    _validate_diagnostic_backend_binding(
        backend, policy.diagnostic_cost_policy, evidence_ledger, identity,
    )
    binding = (
        "policy",
        profile,
        policy,
        route,
        shared_state,
        evidence_ledger,
        identity,
    )
    if getattr(backend, _BOUND_ATTRIBUTE, False):
        if getattr(backend, _BINDING_ATTRIBUTE, None) != binding:
            raise InvocationLimitContractError(
                "backend already carries a different invocation limit "
                "contract"
            )
        return
    original_invoke = getattr(backend, "invoke", None)
    if not callable(original_invoke):
        raise InvocationLimitContractError(
            "backend has no callable invoke boundary for limit enforcement"
        )
    state_lock = threading.RLock()
    backend_name = str(getattr(backend, "name", "unknown")) or "unknown"
    model = str(getattr(backend, "model", "unknown")) or "unknown"

    @wraps(original_invoke)
    def guarded_invoke(invocation: Any) -> InvocationResult:
        with state_lock:
            _raise_if_terminal_latched(backend)
            if shared_state is not None:
                shared_state.require_call_authority()
            if type(invocation) is not Invocation:
                raise InvocationLimitContractError(
                    "invocation-limit guard requires an exact assay.Invocation"
                )
            if policy.require_budget_usd and invocation.budget_usd is None:
                raise InvocationLimitContractError(
                    "budgeted workflow invocation omitted budget_usd; refusing "
                    "provider transport"
                )
            request = InvocationLimitRequest(
                mode=policy.mode,
                max_turns=invocation.max_turns,
                budget_usd=invocation.budget_usd,
                deadline_seconds=policy.deadline_seconds,
                diagnostic_cost_policy=policy.diagnostic_cost_policy,
            )
            resolution = resolve_invocation_limits(
                profile,
                request,
                route=route,
            )
            return _invoke_with_resolution(
                backend,
                original_invoke,
                invocation,
                resolution,
                state_lock=state_lock,
                shared_state=shared_state,
                backend_name=backend_name,
                model=model,
                evidence_ledger=evidence_ledger,
                identity=identity,
            )

    setattr(backend, _POLICY_ATTRIBUTE, policy)
    setattr(backend, _RESOLUTION_ATTRIBUTE, None)
    setattr(backend, _TERMINAL_ATTRIBUTE, None)
    setattr(backend, _EVIDENCE_FAILURE_ATTRIBUTE, False)
    setattr(backend, _BINDING_ATTRIBUTE, binding)
    backend.invoke = guarded_invoke
    setattr(backend, _BOUND_ATTRIBUTE, True)
    _block_alternate_chat_boundary(backend)


def invocation_limit_resolution(
    backend: Any,
) -> ResolvedInvocationLimits | None:
    value = getattr(backend, _RESOLUTION_ATTRIBUTE, None)
    return value if type(value) is ResolvedInvocationLimits else None


def invocation_limit_terminal(
    backend: Any,
) -> InvocationLimitTerminal | None:
    value = getattr(backend, _TERMINAL_ATTRIBUTE, None)
    return value if type(value) is InvocationLimitTerminal else None


def invocation_limit_record_from_result(
    result: InvocationResult,
) -> Mapping[str, Any] | None:
    """Return the persisted UIR-03 trace record, if one is present."""

    trace = result.execution_trace
    if trace is None:
        return None
    matches = [
        event.name[len(_TRACE_EVENT_PREFIX) :]
        for event in trace.events
        if event.kind == "limit"
        and event.name is not None
        and event.name.startswith(_TRACE_EVENT_PREFIX)
    ]
    if len(matches) != 1:
        return None
    try:
        decoded = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(decoded, dict)
        or decoded.get("schema_version") != INVOCATION_LIMIT_SCHEMA
    ):
        return None
    return cast(Mapping[str, Any], decoded)


__all__ = [
    "INVOCATION_LIMIT_LEDGER_LEAF",
    "INVOCATION_LIMIT_LEDGER_SCHEMA",
    "INVOCATION_LIMIT_MODES",
    "INVOCATION_LIMIT_SCHEMA",
    "InvocationLimitAdmissionError",
    "InvocationLimitContractError",
    "InvocationLimitIdentity",
    "InvocationLimitLedger",
    "InvocationLimitMode",
    "InvocationLimitPolicy",
    "InvocationLimitRequest",
    "InvocationLimitReservation",
    "InvocationLimitState",
    "InvocationLimitTerminal",
    "InvocationLimitObservedUsage",
    "ResolvedInvocationLimits",
    "bind_invocation_limit_contract",
    "bind_invocation_limit_policy",
    "invocation_limit_record_from_result",
    "invocation_limit_resolution",
    "invocation_limit_terminal",
    "preflight_invocation_limit_policy",
    "resolve_invocation_limits",
]
