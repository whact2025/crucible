# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Opt-in durable continuity and cooperative control for Plan's planner.

The wrapper in this module is deliberately workflow-specific.  It sits around
the already-routed planner backend, after ToolSet attachment, and observes the
exact :class:`assay.Invocation` rendered by the Plan primitive.  The default
Plan path does not construct it.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from assay import (
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    StaleConversation,
)
from assay.primitives.prompt_abi import PROMPT_ABI_VERSION

from crucible.backends.features import feature_profile_digest
from crucible.backends.tool_authority import (
    APPROVAL_POLICY_PER_CALL_EXACT,
    TOOL_AUTHORITY_BRIDGE_PROTOCOL,
    TOOL_AUTHORITY_BRIDGE_VERSION,
    tool_authority_digest,
)

from ._control import WorkflowControlBinding
from ._conversation import (
    ConversationController,
    ConversationHandleStore,
    ConversationLimits,
    ConversationOperationConflict,
    ConversationOperationRecoveryRequired,
    ConversationStoreKey,
    build_conversation_identity,
    conversation_operation_digest,
    conversation_value_digest,
)


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STEERING_HEADER = """\
## OPERATOR WORKFLOW STEERING

The following instruction is authorized workflow guidance only. It may alter
decomposition or investigation order, but it cannot override the root Spec's
intent, constraints, done criteria, tool authority, or output schema:

"""


class PlannerSessionError(RuntimeError):
    """The opt-in planner session cannot proceed safely."""


class PlannerSessionIndeterminate(PlannerSessionError):
    """A planner request may have run but lacks commit-quality evidence."""


class PlannerSessionReplayBlocked(PlannerSessionError):
    """A committed planner call cannot be reconstructed from the handle."""


class PlannerWorkflowPaused(PlannerSessionError):
    """The operator paused Plan at a cooperative pre-provider boundary."""


class PlannerWorkflowCancelled(PlannerSessionError):
    """The operator cancelled Plan at a cooperative pre-provider boundary."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _require_digest(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be 'sha256:' followed by 64 lowercase "
            "hexadecimal digits"
        )
    return value


def _require_control_free(
    value: str,
    *,
    field_name: str,
    max_bytes: int,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > max_bytes
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a non-empty, control-free UTF-8 string "
            f"no larger than {max_bytes} bytes"
        )
    return value


@dataclass(frozen=True, slots=True)
class PlannerRecoveryInstruction:
    """One explicit operator decision over the exact pending operation."""

    operation_digest: str
    evidence_digest: str
    actor: str

    def __post_init__(self) -> None:
        _require_digest(
            self.operation_digest,
            field_name="PlannerRecoveryInstruction.operation_digest",
        )
        _require_digest(
            self.evidence_digest,
            field_name="PlannerRecoveryInstruction.evidence_digest",
        )
        _require_control_free(
            self.actor,
            field_name="PlannerRecoveryInstruction.actor",
            max_bytes=1024,
        )


@dataclass(frozen=True, slots=True)
class PlannerConversationConfig:
    """Static identity and limits for one explicitly named Plan session."""

    state_root: Path
    run_identity_digest: str
    route_identity: Any
    tool_authority_digest: str
    runtime_identity: Any
    trusted_context_identity: Any
    limits: ConversationLimits
    trace_tool_authority_digest: str | None = None
    recovery: PlannerRecoveryInstruction | None = None
    allow_stale_fallback: bool = True
    now: Callable[[], str] = field(
        default=_utc_now,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.state_root, Path):
            object.__setattr__(self, "state_root", Path(self.state_root))
        _require_digest(
            self.run_identity_digest,
            field_name="PlannerConversationConfig.run_identity_digest",
        )
        _require_digest(
            self.tool_authority_digest,
            field_name="PlannerConversationConfig.tool_authority_digest",
        )
        if self.trace_tool_authority_digest is not None:
            _require_digest(
                self.trace_tool_authority_digest,
                field_name=(
                    "PlannerConversationConfig.trace_tool_authority_digest"
                ),
            )
        if not isinstance(self.limits, ConversationLimits):
            raise TypeError(
                "PlannerConversationConfig.limits must be ConversationLimits"
            )
        if self.recovery is not None and not isinstance(
            self.recovery,
            PlannerRecoveryInstruction,
        ):
            raise TypeError(
                "PlannerConversationConfig.recovery must be a "
                "PlannerRecoveryInstruction or None"
            )
        if type(self.allow_stale_fallback) is not bool:
            raise TypeError(
                "PlannerConversationConfig.allow_stale_fallback must be bool"
            )
        if not callable(self.now):
            raise TypeError("PlannerConversationConfig.now must be callable")


def planner_tool_authority_digest(
    toolset: Any | None,
    *,
    capability: str,
    context_budget: int | None = None,
) -> str:
    """Return the exact attached authority, including the no-tools case."""

    if toolset is None:
        return conversation_value_digest(
            {
                "toolset": None,
                "capability": capability,
                "context_budget": context_budget,
            },
            domain="plan.planner_tool_authority.none",
        )
    return tool_authority_digest(
        toolset,
        capability=capability,
        approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
        bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
        bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION,
        context_budget=context_budget,
    )


def _policy_identity(invocation: Invocation) -> Mapping[str, Any]:
    return {
        "backend_policy": invocation.policy,
        "invocation_tools": invocation.tools,
    }


def _operation_content(invocation: Invocation) -> Mapping[str, Any]:
    return {
        "system": invocation.system,
        "user": invocation.user,
        "tools": invocation.tools,
        "max_turns": invocation.max_turns,
        "budget_usd": invocation.budget_usd,
        "policy": invocation.policy,
        "response_schema": invocation.response_schema,
    }


def _request_context_bytes(invocation: Invocation) -> int:
    return len(invocation.system.encode("utf-8")) + len(
        invocation.user.encode("utf-8")
    )


def _turns_used(result: InvocationResult, *, admitted_turns: int) -> int:
    error = result.error
    if isinstance(error, MaxTurnsExceeded):
        # A turn-limit terminal may count the turn the provider refused to
        # start (Claude CLI: num_turns == limit + 1 after ``limit`` requests).
        # That one unit was neither admitted nor run. Any larger count is a
        # real overrun and is charged in full.
        if error.turns_used == admitted_turns + 1:
            return admitted_turns
        return error.turns_used
    assistant_messages = sum(
        message.role == "assistant" for message in result.transcript
    )
    # A complete active provider outcome with a final response represents at
    # least one logical planner turn even if an adapter elides transcript
    # fragments from its public result.
    return max(1, assistant_messages)


def _proves_zero_accounting(result: InvocationResult) -> bool:
    """Return whether a typed result proves that no paid work occurred."""

    completeness = result.completeness
    usage = result.usage
    return (
        completeness.token_usage_complete
        and completeness.cost_complete
        and type(usage.input_tokens) is int
        and usage.input_tokens == 0
        and type(usage.output_tokens) is int
        and usage.output_tokens == 0
        and type(usage.cost_usd) in (int, float)
        and usage.cost_usd == 0
    )


class PlannerBoundaryBackend:
    """Forward one routed backend through optional SES/CTL boundaries."""

    def __init__(
        self,
        backend: Any,
        *,
        conversation: PlannerConversationConfig | None = None,
        control: WorkflowControlBinding | None = None,
    ) -> None:
        if conversation is None and control is None:
            raise ValueError(
                "PlannerBoundaryBackend requires conversation or control"
            )
        self._backend = backend
        self._conversation = conversation
        self._control = control
        self._observed_steering_revision = 0
        self._call_index = 0
        self._recovery_consumed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def _now(self) -> str:
        if self._conversation is not None:
            value = self._conversation.now()
        else:
            value = _utc_now()
        return _require_control_free(
            value,
            field_name="planner boundary time",
            max_bytes=128,
        )

    def _apply_control(self, invocation: Invocation) -> Invocation:
        binding = self._control
        if binding is None:
            return invocation
        now = self._now()
        binding.heartbeat(at=now)
        checkpoint = binding.checkpoint(
            observed_steering_revision=self._observed_steering_revision,
            at=now,
        )
        if checkpoint.should_cancel:
            raise PlannerWorkflowCancelled(
                "Plan was cancelled before the next planner provider call"
            )
        if checkpoint.should_pause:
            raise PlannerWorkflowPaused(
                "Plan is paused before the next planner provider call"
            )
        steering = checkpoint.steering
        self._observed_steering_revision = checkpoint.steering_revision
        if steering is None:
            return invocation
        return replace(
            invocation,
            user=(
                invocation.user.rstrip()
                + "\n\n"
                + _STEERING_HEADER
                + steering.instruction.strip()
            ),
        )

    def _identity(self, invocation: Invocation):
        config = self._conversation
        assert config is not None
        backend_name = str(getattr(self._backend, "name", "")).strip()
        model = str(getattr(self._backend, "model", "")).strip()
        if not backend_name or not model:
            raise PlannerSessionError(
                "routed planner backend must expose non-empty name and model"
            )
        profile = getattr(self._backend, "features", None)
        if profile is None:
            raise PlannerSessionError(
                "routed planner backend does not expose a feature profile"
            )
        return build_conversation_identity(
            role="plan_planner",
            backend=backend_name,
            model=model,
            prompt_abi=PROMPT_ABI_VERSION,
            conversation_mode="workflow_bound",
            route_identity=config.route_identity,
            prompt_content=invocation.system,
            response_schema=invocation.response_schema,
            tool_authority_digest=config.tool_authority_digest,
            feature_profile=profile,
            policy_identity={
                "policy": _policy_identity(invocation),
                "cumulative_limits": dataclasses.asdict(config.limits),
            },
            runtime_identity=config.runtime_identity,
            trusted_context_identity=config.trusted_context_identity,
        )

    def _controller(self, invocation: Invocation) -> ConversationController:
        config = self._conversation
        assert config is not None
        identity = self._identity(invocation)
        return ConversationController(
            store=ConversationHandleStore(config.state_root),
            key=ConversationStoreKey(
                run_identity_digest=config.run_identity_digest,
                workflow="plan",
                role="plan_planner",
                cycle="planner-refinement",
            ),
            identity=identity,
            mode="workflow_bound",
            feature_profile=self._backend.features,
            limits=config.limits,
        )

    def _operation_digest(
        self,
        controller: ConversationController,
        invocation: Invocation,
        *,
        attempt: int,
    ) -> str:
        return conversation_operation_digest(
            controller.identity,
            operation_id=(
                f"planner-call-{self._call_index}/physical-attempt-{attempt}"
            ),
            content_identity=_operation_content(invocation),
        )

    def _resolve_operator_recovery(
        self,
        controller: ConversationController,
        invocation: Invocation,
    ) -> int:
        """Reconcile only the exact current call and return its next attempt."""

        config = self._conversation
        assert config is not None
        handle = controller.load()
        recovery = config.recovery
        pending = None if handle is None else handle.pending_operation
        if pending is None:
            if recovery is not None and not self._recovery_consumed:
                raise ConversationOperationConflict(
                    "operator recovery was supplied but this planner session "
                    "has no pending operation"
                )
            return 0
        if recovery is None or self._recovery_consumed:
            raise ConversationOperationRecoveryRequired(handle, pending)
        assert handle is not None
        if recovery.operation_digest != pending.operation_digest:
            raise ConversationOperationConflict(
                "operator recovery operation does not match the exact "
                "pending planner operation"
            )

        resolutions = {
            resolution.operation_digest
            for resolution in handle.recovery_resolutions
        }
        matching_attempt: int | None = None
        for attempt in range(len(resolutions) + 2):
            candidate = self._operation_digest(
                controller,
                invocation,
                attempt=attempt,
            )
            if candidate == pending.operation_digest:
                matching_attempt = attempt
                break
            if candidate not in resolutions:
                break
        if matching_attempt is None:
            raise ConversationOperationConflict(
                "pending planner operation does not belong to the current "
                "rendered invocation; Plan cannot replay an earlier debate "
                "round without its semantic checkpoint"
            )
        controller.resolve_pending_not_started(
            operation_digest=pending.operation_digest,
            resolution_evidence_digest=recovery.evidence_digest,
            resolved_at=self._now(),
            actor=recovery.actor,
        )
        self._recovery_consumed = True
        return matching_attempt + 1

    def _next_operation(
        self,
        controller: ConversationController,
        invocation: Invocation,
        *,
        minimum_attempt: int,
    ) -> tuple[int, str]:
        handle = controller.load()
        assert handle is not None
        resolved = {
            resolution.operation_digest
            for resolution in handle.recovery_resolutions
        }
        for attempt in range(
            minimum_attempt,
            minimum_attempt + len(resolved) + 2,
        ):
            operation = self._operation_digest(
                controller,
                invocation,
                attempt=attempt,
            )
            if operation in handle.operation_digests:
                raise PlannerSessionReplayBlocked(
                    "planner operation was already committed, but Plan has no "
                    "durable mid-debate result checkpoint; start a new named "
                    "session instead of replaying paid work"
                )
            if operation in resolved:
                continue
            return attempt, operation
        raise PlannerSessionError(
            "planner operation attempt space is inconsistent with its "
            "durable recovery ledger"
        )

    def _tighten(
        self,
        invocation: Invocation,
        controller: ConversationController,
        *,
        operation_digest: str,
    ) -> Invocation:
        remaining = controller.remaining_budget()
        max_turns = invocation.max_turns
        if remaining.turns is not None:
            if remaining.turns < 1:
                # Normally rejected by controller.resume(). Keep this guard
                # adjacent to request construction so zero never reaches an
                # Invocation constructor through a concurrent update.
                raise PlannerSessionError(
                    "planner conversation has no remaining turns"
                )
            max_turns = min(max_turns, remaining.turns)
        budget_usd = invocation.budget_usd
        if remaining.cost_usd is not None:
            budget_usd = (
                remaining.cost_usd
                if budget_usd is None
                else min(budget_usd, remaining.cost_usd)
            )
        return replace(
            invocation,
            max_turns=max_turns,
            budget_usd=budget_usd,
            conversation=controller.transport_request(
                operation_digest=operation_digest,
            ),
        )

    def _mark_indeterminate(
        self,
        controller: ConversationController,
        operation_digest: str,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        controller.mark_pending_indeterminate(
            operation_digest=operation_digest,
        )
        error = PlannerSessionIndeterminate(message)
        if cause is None:
            raise error
        raise error from cause

    def _validate_complete_result(
        self,
        controller: ConversationController,
        operation_digest: str,
        result: InvocationResult,
    ) -> None:
        outcome = result.conversation
        if (
            outcome is None
            or outcome.state != "active"
            or outcome.identity_fingerprint
            != controller.identity.fingerprint
            or outcome.operation_digest != operation_digest
        ):
            self._mark_indeterminate(
                controller,
                operation_digest,
                "planner result lacks the exact active conversation outcome",
            )
        if not result.completeness.complete:
            self._mark_indeterminate(
                controller,
                operation_digest,
                "planner result lacks complete token/cost/terminal/trace "
                "evidence",
            )
        if not self._trace_matches(result):
            self._mark_indeterminate(
                controller,
                operation_digest,
                "planner result trace does not match the exact routed "
                "backend/model/profile/tool authority",
            )

    def _trace_matches(self, result: InvocationResult) -> bool:
        trace = result.execution_trace
        config = self._conversation
        assert config is not None
        expected_profile = feature_profile_digest(self._backend.features)
        return not (
            trace is None
            or trace.backend != self._backend.name
            or trace.requested_model != self._backend.model
            or trace.feature_profile_digest != expected_profile
            or trace.tool_authority_digest
            != config.trace_tool_authority_digest
        )

    def _invoke_attempt(
        self,
        controller: ConversationController,
        invocation: Invocation,
        *,
        operation_digest: str,
    ) -> tuple[InvocationResult, int]:
        """Return the result and the turn ceiling forwarded to the adapter."""

        controller.claim_operation(
            operation_digest=operation_digest,
            started_at=self._now(),
        )
        try:
            forwarded = self._tighten(
                invocation,
                controller,
                operation_digest=operation_digest,
            )
        except BaseException:
            # No adapter call has happened. Persist the local proof rather
            # than leaving a false crash-ambiguity marker behind.
            controller.resolve_pending_not_started(
                operation_digest=operation_digest,
                resolution_evidence_digest=conversation_value_digest(
                    {
                        "stage": "pre_transport_request_construction",
                        "identity_fingerprint": (
                            controller.identity.fingerprint
                        ),
                        "operation_digest": operation_digest,
                    },
                    domain="plan.local_not_started_resolution",
                ),
                resolved_at=self._now(),
                actor="workflow:plan:pre-transport",
            )
            raise
        try:
            result = self._backend.invoke(forwarded)
        except BaseException as exc:
            self._mark_indeterminate(
                controller,
                operation_digest,
                "planner provider outcome is uncertain after invocation "
                "raised",
                cause=exc,
            )
        if not isinstance(result, InvocationResult):
            self._mark_indeterminate(
                controller,
                operation_digest,
                "planner backend returned a non-InvocationResult outcome",
            )
        return result, forwarded.max_turns

    def _record_typed_stale(
        self,
        controller: ConversationController,
        operation_digest: str,
        result: InvocationResult,
    ) -> bool:
        """Record stale evidence and return whether a fresh retry is safe."""

        outcome = result.conversation
        if (
            not isinstance(result.error, StaleConversation)
            or outcome is None
            or outcome.state != "stale"
            or outcome.opaque_id is not None
            or outcome.identity_fingerprint
            != controller.identity.fingerprint
            or outcome.operation_digest != operation_digest
            or result.stop_reason != "stale_conversation"
        ):
            self._mark_indeterminate(
                controller,
                operation_digest,
                "stale planner result lacks exact typed provider rejection",
            )
        if (
            not result.completeness.trace_complete
            or not self._trace_matches(result)
        ):
            self._mark_indeterminate(
                controller,
                operation_digest,
                "stale planner result trace does not match the exact routed "
                "backend/model/profile/tool authority",
            )
        if not _proves_zero_accounting(result):
            # The provider rejection is typed, but without complete, exact-zero
            # accounting it is not proof that the physical attempt never ran.
            # Keep the operation unresolved so it cannot disappear from the
            # continuity ledger, and return the original result to the caller
            # so its usage lower bound and accounting debt remain observable.
            controller.mark_pending_indeterminate(
                operation_digest=operation_digest,
            )
            controller.mark_stale()
            return False
        trace = result.execution_trace
        assert trace is not None
        evidence_digest = conversation_value_digest(
            {
                "error_kind": result.error.kind,
                "stop_reason": result.stop_reason,
                "conversation_state": outcome.state,
                "identity_fingerprint": outcome.identity_fingerprint,
                "operation_digest": outcome.operation_digest,
                "trace": trace,
                "completeness": result.completeness,
                "usage": result.usage,
            },
            domain="plan.provider_stale_resolution",
        )
        controller.resolve_pending_not_started(
            operation_digest=operation_digest,
            resolution_evidence_digest=evidence_digest,
            resolved_at=self._now(),
            actor=f"provider:{self._backend.name}:typed-stale",
        )
        controller.mark_stale()
        return True

    def invoke(self, invocation: Invocation) -> InvocationResult:
        invocation = self._apply_control(invocation)
        if self._conversation is None:
            return self._backend.invoke(invocation)

        controller = self._controller(invocation)
        try:
            return self._invoke_with_conversation(controller, invocation)
        finally:
            controller.close_store()

    def _invoke_with_conversation(
        self,
        controller: ConversationController,
        invocation: Invocation,
    ) -> InvocationResult:
        """Invoke with one deterministically scoped conversation store."""

        conversation_config = self._conversation
        assert conversation_config is not None
        minimum_attempt = self._resolve_operator_recovery(
            controller,
            invocation,
        )
        handle = controller.load()
        controller.resume(
            scope="workflow_correction",
            provider_reports_stale=(
                handle is not None and handle.state == "stale"
            ),
            allow_stale_fallback=conversation_config.allow_stale_fallback,
        )
        attempt, operation = self._next_operation(
            controller,
            invocation,
            minimum_attempt=minimum_attempt,
        )
        result, admitted_turns = self._invoke_attempt(
            controller,
            invocation,
            operation_digest=operation,
        )
        request = controller.transport_request(operation_digest=operation)
        if isinstance(result.error, StaleConversation):
            safe_fresh_fallback = self._record_typed_stale(
                controller,
                operation,
                result,
            )
            if (
                request.action != "reuse"
                or not conversation_config.allow_stale_fallback
                or not safe_fresh_fallback
            ):
                return result
            controller.resume(
                scope="workflow_correction",
                provider_reports_stale=True,
                allow_stale_fallback=True,
            )
            attempt, operation = self._next_operation(
                controller,
                invocation,
                minimum_attempt=attempt + 1,
            )
            result, admitted_turns = self._invoke_attempt(
                controller,
                invocation,
                operation_digest=operation,
            )
            if isinstance(result.error, StaleConversation):
                self._record_typed_stale(controller, operation, result)
                return result

        self._validate_complete_result(
            controller,
            operation,
            result,
        )
        outcome = result.conversation
        assert outcome is not None
        controller.commit_operation(
            operation_digest=operation,
            opaque_id=outcome.opaque_id,
            turns=_turns_used(result, admitted_turns=admitted_turns),
            context_bytes=_request_context_bytes(invocation),
            usage=result.usage,
        )
        self._call_index += 1
        return result


__all__ = [
    "PlannerBoundaryBackend",
    "PlannerConversationConfig",
    "PlannerRecoveryInstruction",
    "PlannerSessionError",
    "PlannerSessionIndeterminate",
    "PlannerSessionReplayBlocked",
    "PlannerWorkflowCancelled",
    "PlannerWorkflowPaused",
    "planner_tool_authority_digest",
]
