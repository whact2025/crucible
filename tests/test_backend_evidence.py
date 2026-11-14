# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    TransportError,
    Usage,
)

from crucible.backends.evidence import (
    admit_configured_result,
    bind_result_evidence_admission,
)
from crucible.backends.config import BackendRouting, RoleConfig, resolve_for_role


def _invocation() -> Invocation:
    return Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=1,
        budget_usd=None,
    )


def _complete_result() -> InvocationResult:
    return InvocationResult(
        final_message="ok",
        transcript=(),
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="fake",
            requested_model="fake-model",
            events=(
                BackendExecutionEvent(
                    sequence=0,
                    kind="terminal",
                    name="end_turn",
                ),
            ),
        ),
    )


def _semantic_result_with_accounting_debt() -> InvocationResult:
    result = _complete_result()
    return InvocationResult(
        final_message=result.final_message,
        transcript=result.transcript,
        usage=result.usage,
        stop_reason=result.stop_reason,
        completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=result.execution_trace,
    )


@dataclass
class _Backend:
    result: InvocationResult
    calls: int = 0
    name: str = "fake"
    model: str = "fake-model"

    def invoke(self, invocation: Invocation) -> InvocationResult:
        del invocation
        self.calls += 1
        return self.result


@dataclass
class _ChatBackend(_Backend):
    chat_calls: int = 0

    def chat_with_tools(self, invocation, toolset, **kwargs):
        del invocation, toolset, kwargs
        self.chat_calls += 1
        return self.result


def test_direct_adapter_result_is_diagnostic_until_routing_binds_policy():
    backend = _Backend(
        InvocationResult(
            final_message="diagnostic",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )
    )

    result = admit_configured_result(backend, backend.invoke(_invocation()))

    assert result.error is None


def test_enforcing_route_converts_incomplete_success_to_typed_failure():
    backend = _Backend(
        InvocationResult(
            final_message="nominal",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )
    )
    bind_result_evidence_admission(backend, mode="enforce")

    result = backend.invoke(_invocation())

    assert isinstance(result.error, IncompleteEvidence)
    assert result.final_message == "nominal"
    assert backend.calls == 1


def test_diagnostic_route_retains_incomplete_success():
    backend = _Backend(
        InvocationResult(
            final_message="diagnostic",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )
    )
    bind_result_evidence_admission(backend, mode="diagnostic")

    assert backend.invoke(_invocation()).error is None


def test_semantic_route_preserves_accounting_debt_but_not_control_debt():
    backend = _Backend(_semantic_result_with_accounting_debt())
    bind_result_evidence_admission(backend, mode="semantic")

    result = backend.invoke(_invocation())

    assert result is backend.result
    assert result.error is None
    assert result.completeness.terminal_complete is True
    assert result.completeness.trace_complete is True
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False

    backend.result = InvocationResult(
        final_message="unterminated",
        transcript=(),
        usage=Usage(),
        stop_reason="unknown",
    )
    assert isinstance(backend.invoke(_invocation()).error, IncompleteEvidence)


def test_enforcing_route_accepts_complete_success_and_preserves_failures():
    backend = _Backend(_complete_result())
    bind_result_evidence_admission(backend, mode="enforce")
    assert backend.invoke(_invocation()) is backend.result

    failure = InvocationResult(
        final_message="",
        transcript=(),
        usage=Usage(),
        stop_reason="error",
        error=TransportError(message="provider failed"),
    )
    backend.result = failure
    assert backend.invoke(_invocation()) is failure


def test_rebinding_policy_does_not_stack_invoke_wrappers():
    backend = _Backend(
        InvocationResult(
            final_message="diagnostic",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )
    )
    bind_result_evidence_admission(backend, mode="diagnostic")
    bind_result_evidence_admission(backend, mode="enforce")

    assert isinstance(backend.invoke(_invocation()).error, IncompleteEvidence)
    assert backend.calls == 1


def test_enforcing_route_guards_peer_chat_with_tools_result_boundary():
    backend = _ChatBackend(
        InvocationResult(
            final_message="nominal tool result",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )
    )
    bind_result_evidence_admission(backend, mode="diagnostic")
    bind_result_evidence_admission(backend, mode="enforce")

    result = backend.chat_with_tools(_invocation(), object())

    assert isinstance(result.error, IncompleteEvidence)
    assert result.final_message == "nominal tool result"
    assert backend.chat_calls == 1


def test_canonical_route_binds_evidence_policy_before_return(monkeypatch):
    backend = _Backend(
        InvocationResult(
            final_message="nominal",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )
    )
    monkeypatch.setattr(
        "crucible.backends.config._resolve_backend",
        lambda *_args, **_kwargs: backend,
    )
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission="enforce",
        )
    )

    routed = resolve_for_role("plan_critic", routing=routing)

    assert routed is backend
    assert isinstance(routed.invoke(_invocation()).error, IncompleteEvidence)
