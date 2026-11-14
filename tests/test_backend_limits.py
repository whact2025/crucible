# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Offline UIR-03 invocation-limit admission and terminal evidence tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    Usage,
)
from crucible.backends.config import (
    BackendRouting,
    RoleConfig,
    resolve_for_role,
)
from crucible.backends.features import feature_profile_for_route
from crucible.backends.limits import (
    INVOCATION_LIMIT_LEDGER_LEAF,
    INVOCATION_LIMIT_LEDGER_SCHEMA,
    INVOCATION_LIMIT_SCHEMA,
    InvocationLimitAdmissionError,
    InvocationLimitContractError,
    InvocationLimitIdentity,
    InvocationLimitLedger,
    InvocationLimitPolicy,
    InvocationLimitRequest,
    InvocationLimitState,
    bind_invocation_limit_contract,
    bind_invocation_limit_policy,
    invocation_limit_record_from_result,
    invocation_limit_terminal,
    resolve_invocation_limits,
)


_COMPLETE = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)


def _result(*, cost: float = 0.0) -> InvocationResult:
    return InvocationResult(
        final_message="done",
        transcript=(),
        usage=Usage(input_tokens=10, output_tokens=2, cost_usd=cost),
        stop_reason="end_turn",
        completeness=_COMPLETE,
        execution_trace=BackendExecutionTrace(
            backend="fake",
            requested_model="fake-model",
            served_model="fake-model",
            events=(
                BackendExecutionEvent(sequence=0, kind="request"),
                BackendExecutionEvent(
                    sequence=1,
                    kind="terminal",
                    name="end_turn",
                ),
            ),
        ),
    )


def _invocation(
    *,
    max_turns: int = 3,
    budget_usd: float | None = 1.0,
    policy: BackendPolicy | None = None,
) -> Invocation:
    return Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=max_turns,
        budget_usd=budget_usd,
        policy=policy,
    )


@dataclass
class _FakeBackend:
    result: InvocationResult
    name: str = "fake"
    model: str = "fake-model"

    def __post_init__(self) -> None:
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        return self.result


@pytest.mark.parametrize(
    ("backend", "model", "base_url", "turn", "usd", "deadline"),
    [
        (
            "codex",
            "gpt-5.6-sol",
            None,
            "unsupported",
            "post_hoc",
            "request_only",
        ),
        (
            "claude",
            "claude-opus-5",
            None,
            "provider_stop",
            "post_hoc",
            "request_only",
        ),
        (
            "openai_compat",
            "frontier",
            "https://example.invalid/v1",
            "round_boundary",
            "post_hoc",
            "request_only",
        ),
        (
            "kimi",
            "kimi-k2.7-code-highspeed",
            "https://api.moonshot.ai/v1",
            "hard",
            "post_hoc",
            "hard_invocation",
        ),
    ],
)
def test_frontier_limit_profiles_are_not_normalized_upward(
    backend: str,
    model: str,
    base_url: str | None,
    turn: str,
    usd: str,
    deadline: str,
) -> None:
    profile = feature_profile_for_route(
        backend,
        model=model,
        base_url=base_url,
    )

    assert (profile.turn_limit, profile.usd_limit, profile.deadline) == (
        turn,
        usd,
        deadline,
    )


def test_limit_request_mode_is_closed() -> None:
    with pytest.raises(ValueError, match="unknown invocation limit mode"):
        InvocationLimitRequest(
            mode="best_effort",  # type: ignore[arg-type]
            max_turns=1,
            budget_usd=None,
            deadline_seconds=30,
        )


def test_strict_codex_turn_limit_rejects_before_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    constructed = False

    def _unexpected(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal constructed
        constructed = True
        raise AssertionError("strict rejection must precede construction")

    monkeypatch.setattr(config_module, "_resolve_backend", _unexpected)

    with pytest.raises(
        InvocationLimitAdmissionError,
        match="strict invocation limits before adapter construction or transport",
    ):
        resolve_for_role(
            "implementer",
            routing=BackendRouting(
                default=RoleConfig(
                    backend="codex",
                    model="gpt-5.6-sol",
                ),
            ),
            limit_request=InvocationLimitRequest(
                mode="strict",
                max_turns=1,
                budget_usd=None,
                deadline_seconds=30,
            ),
        )

    assert constructed is False


def test_strict_kimi_usd_limit_rejects_post_hoc_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    constructed = False

    def _construct(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal constructed
        constructed = True
        return _FakeBackend(_result())

    monkeypatch.setattr(config_module, "_resolve_backend", _construct)
    with pytest.raises(
        InvocationLimitAdmissionError,
        match="post-call accounting is not hard enforcement",
    ):
        resolve_for_role(
            "implementer",
            routing=BackendRouting(
                default=RoleConfig(
                    backend="kimi",
                    model="kimi-k2.7-code-highspeed",
                    base_url="https://api.moonshot.ai/v1",
                ),
            ),
            limit_request=InvocationLimitRequest(
                mode="strict",
                max_turns=4,
                budget_usd=2.0,
                deadline_seconds=30,
            ),
        )

    assert constructed is False


def test_reusable_strict_policy_rejects_before_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    constructed = False

    def _unexpected(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal constructed
        constructed = True
        raise AssertionError("strict policy must reject before construction")

    monkeypatch.setattr(config_module, "_resolve_backend", _unexpected)

    with pytest.raises(
        InvocationLimitAdmissionError,
        match="strict invocation limits before adapter construction or transport",
    ):
        resolve_for_role(
            "implementer",
            routing=BackendRouting(
                default=RoleConfig(
                    backend="codex",
                    model="gpt-5.6-sol",
                ),
            ),
            limit_policy=InvocationLimitPolicy(
                mode="strict",
                deadline_seconds=30,
                require_budget_usd=True,
            ),
        )

    assert constructed is False


def test_reusable_strict_kimi_budget_policy_rejects_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    constructed = False

    def _unexpected(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal constructed
        constructed = True
        raise AssertionError("strict policy must reject before construction")

    monkeypatch.setattr(config_module, "_resolve_backend", _unexpected)

    with pytest.raises(
        InvocationLimitAdmissionError,
        match="post-call accounting is not hard enforcement",
    ):
        resolve_for_role(
            "implementer",
            routing=BackendRouting(
                default=RoleConfig(
                    backend="kimi",
                    model="kimi-k2.7-code-highspeed",
                    base_url="https://api.moonshot.ai/v1",
                ),
            ),
            limit_policy=InvocationLimitPolicy(
                mode="strict",
                deadline_seconds=30,
                require_budget_usd=True,
            ),
        )

    assert constructed is False


def test_reusable_diagnostic_policy_resolves_each_actual_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    fake = _FakeBackend(_result(cost=0.25))
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: fake,
    )
    backend = resolve_for_role(
        "implementer",
        routing=BackendRouting(
            default=RoleConfig(
                backend="codex",
                model="gpt-5.6-sol",
            ),
        ),
        limit_policy=InvocationLimitPolicy(
            mode="diagnostic",
            deadline_seconds=12,
            require_budget_usd=True,
        ),
    )

    first = backend.invoke(_invocation(max_turns=3, budget_usd=1.0))
    second = backend.invoke(_invocation(max_turns=5, budget_usd=0.5))
    first_record = invocation_limit_record_from_result(first)
    second_record = invocation_limit_record_from_result(second)

    assert len(fake.invocations) == 2
    assert [call.max_turns for call in fake.invocations] == [3, 5]
    assert [call.budget_usd for call in fake.invocations] == [1.0, 0.5]
    assert all(
        call.policy is not None
        and call.policy.retry_budget == 1
        and call.policy.timeout_seconds == 12
        for call in fake.invocations
    )
    assert first_record is not None and second_record is not None
    assert first_record["request"]["requested_budget_usd"] == 1.0
    assert second_record["request"]["requested_budget_usd"] == 0.5
    assert first_record["terminal"]["physical_attempts"] == 1
    assert second_record["terminal"]["physical_attempts"] == 1


def test_budgeted_reusable_policy_rejects_missing_call_budget_without_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    fake = _FakeBackend(_result())
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: fake,
    )
    backend = resolve_for_role(
        "implementer",
        routing=BackendRouting(
            default=RoleConfig(
                backend="codex",
                model="gpt-5.6-sol",
            ),
        ),
        limit_policy=InvocationLimitPolicy(
            mode="diagnostic",
            deadline_seconds=12,
            require_budget_usd=True,
        ),
    )

    with pytest.raises(InvocationLimitContractError, match="omitted budget_usd"):
        backend.invoke(_invocation(budget_usd=None))

    assert fake.invocations == []


def test_shared_workflow_state_blocks_recovery_on_a_fresh_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    first = _FakeBackend(_result(cost=1.25))
    second = _FakeBackend(_result(cost=0.1))
    constructed = iter((first, second))
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: next(constructed),
    )
    routing = BackendRouting(
        default=RoleConfig(backend="codex", model="gpt-5.6-sol"),
    )
    policy = InvocationLimitPolicy(
        mode="diagnostic",
        deadline_seconds=12,
        require_budget_usd=True,
    )
    state = InvocationLimitState()
    first_backend = resolve_for_role(
        "implementer",
        routing=routing,
        limit_policy=policy,
        limit_state=state,
    )
    second_backend = resolve_for_role(
        "implementer",
        routing=routing,
        limit_policy=policy,
        limit_state=state,
    )

    result = first_backend.invoke(_invocation(budget_usd=1.0))
    assert result.stop_reason == "budget_exhausted"
    with pytest.raises(
        InvocationLimitContractError,
        match="no recovery provider call is authorized in this workflow",
    ):
        second_backend.invoke(_invocation(budget_usd=1.0))
    assert len(first.invocations) == 1
    assert second.invocations == []


def test_strict_kimi_turn_only_contract_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    fake = _FakeBackend(_result())
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: fake,
    )
    backend = resolve_for_role(
        "implementer",
        routing=BackendRouting(
            default=RoleConfig(
                backend="kimi",
                model="kimi-k2.7-code-highspeed",
                base_url="https://api.moonshot.ai/v1",
            ),
        ),
        limit_request=InvocationLimitRequest(
            mode="strict",
            max_turns=4,
            budget_usd=None,
            deadline_seconds=30,
        ),
    )

    result = backend.invoke(_invocation(max_turns=4, budget_usd=None))
    record = invocation_limit_record_from_result(result)

    assert len(fake.invocations) == 1
    assert fake.invocations[0].policy == BackendPolicy(
        retry_budget=1,
        timeout_seconds=30,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )
    assert record is not None
    assert record["request"]["admitted_max_turns"] == 4
    assert record["request"]["downgraded"] == []
    assert record["terminal"]["rankable"] is True


def test_diagnostic_forces_one_retry_attempt_and_one_deadline() -> None:
    profile = feature_profile_for_route("codex", model="gpt-5.6-sol")
    resolution = resolve_invocation_limits(
        profile,
        InvocationLimitRequest(
            mode="diagnostic",
            max_turns=3,
            budget_usd=1.0,
            deadline_seconds=10,
        ),
        route="codex/gpt-5.6-sol",
    )
    fake = _FakeBackend(_result(cost=0.5))
    bind_invocation_limit_contract(fake, resolution)

    result = fake.invoke(_invocation(policy=BackendPolicy(
        retry_budget=5,
        timeout_seconds=50,
        on_cancellation="raise",
        on_max_turns="continue_with_warning",
    )))
    record = invocation_limit_record_from_result(result)

    assert len(fake.invocations) == 1
    assert fake.invocations[0].policy == BackendPolicy(
        retry_budget=1,
        timeout_seconds=10,
        on_cancellation="raise",
        on_max_turns="continue_with_warning",
    )
    assert record is not None
    assert record["schema_version"] == INVOCATION_LIMIT_SCHEMA
    assert record["request"]["retry_budget"] == 1
    assert record["request"]["downgraded"] == [
        "turn_limit",
        "usd_limit",
    ]
    assert record["request"]["turn_enforcement_location"] == "none"
    assert record["request"]["usd_enforcement_location"] == (
        "terminal_accounting"
    )
    assert record["terminal"]["physical_attempts"] == 1


def test_diagnostic_retry_budget_suppresses_physical_transport_retry() -> None:
    class _RetryingFake(_FakeBackend):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.transport_attempts = 0

        def invoke(self, invocation: Invocation) -> InvocationResult:
            self.invocations.append(invocation)
            assert invocation.policy is not None
            for _ in range(invocation.policy.retry_budget):
                self.transport_attempts += 1
            return self.result

    resolution = resolve_invocation_limits(
        feature_profile_for_route("codex"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="codex",
    )
    fake = _RetryingFake(_result(cost=0.1))
    bind_invocation_limit_contract(fake, resolution)

    fake.invoke(_invocation(policy=BackendPolicy(
        retry_budget=9,
        timeout_seconds=20,
        on_cancellation="raise",
        on_max_turns="raise",
    )))

    assert fake.transport_attempts == 1


def test_envelope_mismatch_rejects_without_transport() -> None:
    resolution = resolve_invocation_limits(
        feature_profile_for_route("codex"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="codex",
    )
    fake = _FakeBackend(_result())
    bind_invocation_limit_contract(fake, resolution)

    with pytest.raises(InvocationLimitContractError, match="refusing transport"):
        fake.invoke(_invocation(max_turns=4))

    assert fake.invocations == []


def test_limit_contract_blocks_alternate_chat_transport_boundary() -> None:
    class _ChatBackend(_FakeBackend):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.chat_calls = 0

        def chat_with_tools(self, *_args: Any, **_kwargs: Any) -> InvocationResult:
            self.chat_calls += 1
            return self.result

    resolution = resolve_invocation_limits(
        feature_profile_for_route("codex"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="codex",
    )
    fake = _ChatBackend(_result())
    bind_invocation_limit_contract(fake, resolution)

    with pytest.raises(
        InvocationLimitContractError,
        match="chat_with_tools would bypass",
    ):
        fake.chat_with_tools(_invocation(), object())

    assert fake.chat_calls == 0


def test_bound_contract_rejects_different_durable_ledger(
    tmp_path: Path,
) -> None:
    resolution = _diagnostic_resolution()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = InvocationLimitLedger(first_dir / INVOCATION_LIMIT_LEDGER_LEAF)
    second = InvocationLimitLedger(second_dir / INVOCATION_LIMIT_LEDGER_LEAF)
    fake = _FakeBackend(_result())
    bind_invocation_limit_contract(
        fake,
        resolution,
        evidence_ledger=first,
        identity=_identity(),
    )
    bind_invocation_limit_contract(
        fake,
        resolution,
        evidence_ledger=first,
        identity=_identity(),
    )

    with pytest.raises(
        InvocationLimitContractError,
        match="different invocation limit contract",
    ):
        bind_invocation_limit_contract(
            fake,
            resolution,
            evidence_ledger=second,
            identity=_identity(),
        )


def test_diagnostic_rejects_route_without_a_process_deadline() -> None:
    with pytest.raises(
        InvocationLimitAdmissionError,
        match="one monotonic wall/process deadline before transport",
    ):
        resolve_invocation_limits(
            feature_profile_for_route("unqualified"),
            InvocationLimitRequest("diagnostic", 1, None, 10),
            route="unqualified/model",
        )


def test_more_than_one_physical_request_is_terminal_contract_violation() -> None:
    repeated = _result(cost=0.1)
    assert repeated.execution_trace is not None
    repeated = InvocationResult(
        final_message=repeated.final_message,
        transcript=repeated.transcript,
        usage=repeated.usage,
        stop_reason=repeated.stop_reason,
        completeness=repeated.completeness,
        execution_trace=BackendExecutionTrace(
            backend="fake",
            requested_model="fake-model",
            served_model="fake-model",
            events=(
                BackendExecutionEvent(sequence=0, kind="request"),
                BackendExecutionEvent(sequence=1, kind="request"),
                BackendExecutionEvent(
                    sequence=2,
                    kind="terminal",
                    name="end_turn",
                ),
            ),
        ),
    )
    resolution = resolve_invocation_limits(
        feature_profile_for_route("codex"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="codex",
    )
    fake = _FakeBackend(repeated)
    bind_invocation_limit_contract(fake, resolution)

    result = fake.invoke(_invocation())
    record = invocation_limit_record_from_result(result)

    assert result.stop_reason == "limit_contract_violated"
    assert result.error is not None and result.error.retriable is False
    assert record is not None
    assert record["terminal"]["physical_attempts"] == 2
    assert record["terminal"]["rankable"] is False
    assert result.execution_trace is not None
    assert result.execution_trace.events[-1].name == (
        "limit_contract_violated"
    )


def test_configured_limit_trace_never_persists_url_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    fake = _FakeBackend(_result(cost=0.1))
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: fake,
    )
    backend = resolve_for_role(
        "implementer",
        routing=BackendRouting(
            default=RoleConfig(
                backend="openai_compat",
                model="frontier",
                base_url="https://user:secret@example.invalid/v1?token=hidden",
                feature_admission="diagnostic",
            ),
        ),
        limit_request=InvocationLimitRequest("diagnostic", 3, 1.0, 10),
    )

    result = backend.invoke(_invocation())
    record = invocation_limit_record_from_result(result)

    assert record is not None
    route = record["request"]["route"]
    assert route == "openai_compat/frontier for role 'implementer'"
    assert "secret" not in str(record)
    assert "hidden" not in str(record)


def test_configured_ledger_binds_explicit_role_backend_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.config as config_module

    fake = _FakeBackend(_result(cost=0.1))
    monkeypatch.setattr(
        config_module,
        "_resolve_backend",
        lambda *_args, **_kwargs: fake,
    )
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = resolve_for_role(
        "implementer",
        routing=BackendRouting(
            default=RoleConfig(
                backend="codex",
                model="gpt-5.6-sol",
            ),
        ),
        limit_policy=InvocationLimitPolicy(
            mode="diagnostic",
            deadline_seconds=10,
            require_budget_usd=True,
        ),
        limit_ledger=ledger,
    )

    backend.invoke(_invocation())
    intent = ledger.snapshot()["records"][0]["intent"]
    assert {
        field_name: intent[field_name]
        for field_name in ("role", "backend", "model")
    } == {
        "role": "implementer",
        "backend": "codex",
        "model": "gpt-5.6-sol",
    }


def test_backend_pool_cache_is_bound_to_exact_limit_request(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.workflows._backends as pool_module

    constructed: list[tuple[InvocationLimitRequest, _FakeBackend]] = []

    class _ClosableFake(_FakeBackend):
        def close(self) -> None:
            return None

    def _resolve(_role: str, **kwargs: Any) -> _FakeBackend:
        request = kwargs["limit_request"]
        backend = _ClosableFake(_result())
        constructed.append((request, backend))
        return backend

    monkeypatch.setattr(pool_module, "resolve_for_role", _resolve)
    pool = pool_module._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )
    first_request = InvocationLimitRequest("diagnostic", 3, 1.0, 10)
    second_request = InvocationLimitRequest("diagnostic", 3, 0.5, 10)

    first = pool.get("implementer", limit_request=first_request)
    repeated = pool.get("implementer", limit_request=first_request)
    second = pool.get("implementer", limit_request=second_request)

    assert first is repeated
    assert second is not first
    assert [item[0] for item in constructed] == [
        first_request,
        second_request,
    ]
    pool.close()


def test_workflow_backend_pool_consumes_explicit_diagnostic_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.workflows._backends as pool_module

    observed: list[
        tuple[
            InvocationLimitPolicy,
            InvocationLimitState,
            InvocationLimitLedger,
        ]
    ] = []
    provenance: list[dict[str, Any]] = []

    class _ClosableFake(_FakeBackend):
        def close(self) -> None:
            return None

    def _resolve(_role: str, **kwargs: Any) -> _FakeBackend:
        observed.append(
            (
                kwargs["limit_policy"],
                kwargs["limit_state"],
                kwargs["limit_ledger"],
            )
        )
        return _ClosableFake(_result())

    monkeypatch.setattr(pool_module, "load_routing", lambda **_kwargs: object())
    monkeypatch.setattr(pool_module, "resolve_for_role", _resolve)
    monkeypatch.setattr(
        pool_module,
        "routing_fingerprint",
        lambda _routing: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        pool_module,
        "routing_feature_fingerprint",
        lambda _routing: "sha256:" + "2" * 64,
    )
    monkeypatch.setattr(
        pool_module,
        "routing_source_fingerprint",
        lambda _routing: "sha256:" + "3" * 64,
    )
    monkeypatch.setattr(
        pool_module._outcome,
        "record_provenance_inputs",
        lambda *_args, **kwargs: provenance.append(kwargs),
    )
    args = argparse.Namespace(
        backend_config=None,
        spec=tmp_path,
        budget_usd=2.0,
        backend_limit_mode="diagnostic",
        backend_limit_deadline_seconds=1200.0,
    )

    with pool_module.backend_pool(args, tmp_path, cwd=str(tmp_path)) as pool:
        pool.get("implementer")
    with pool_module.backend_pool(args, tmp_path, cwd=str(tmp_path)) as pool:
        pool.get("reviewer")
    other_run = tmp_path / "other-run"
    other_run.mkdir()
    with pool_module.backend_pool(args, other_run, cwd=str(tmp_path)) as pool:
        pool.get("reviewer")

    expected_policy = InvocationLimitPolicy(
            mode="diagnostic",
            deadline_seconds=1200.0,
            require_budget_usd=True,
    )
    assert [item[0] for item in observed] == [
        expected_policy,
        expected_policy,
        expected_policy,
    ]
    assert observed[0][1] is observed[1][1]
    assert observed[0][2] is observed[1][2]
    assert observed[0][2].path == tmp_path / INVOCATION_LIMIT_LEDGER_LEAF
    assert observed[2][1] is not observed[0][1]
    assert observed[2][2] is not observed[0][2]
    assert len(provenance) == 3
    assert all(
        item["backend_routing_source_sha256"] == "sha256:" + "3" * 64
        for item in provenance
    )
    assert observed[2][2].path == other_run / INVOCATION_LIMIT_LEDGER_LEAF


def test_indivisible_call_usd_overshoot_is_terminal_and_non_rankable() -> None:
    resolution = resolve_invocation_limits(
        feature_profile_for_route("claude"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="claude",
    )
    fake = _FakeBackend(_result(cost=1.25))
    bind_invocation_limit_contract(fake, resolution)

    result = fake.invoke(_invocation())
    terminal = invocation_limit_terminal(fake)
    record = invocation_limit_record_from_result(result)

    assert len(fake.invocations) == 1
    assert result.stop_reason == "budget_exhausted"
    assert result.error is not None
    assert result.error.kind == "budget_exhausted"
    assert result.error.retriable is False
    assert terminal is not None and terminal.rankable is False
    assert terminal.overshoots == ("usd",)
    assert record is not None
    assert record["terminal"]["rankable"] is False
    assert record["terminal"]["overshoots"] == ["usd"]
    assert result.execution_trace is not None
    assert result.execution_trace.terminal_event_count == 1
    assert result.execution_trace.events[-1].kind == "terminal"

    with pytest.raises(
        InvocationLimitContractError,
        match="no recovery provider call is authorized",
    ):
        fake.invoke(_invocation())
    assert len(fake.invocations) == 1


def test_monotonic_deadline_overshoot_is_terminal_and_non_rankable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.limits as limits_module

    clock = iter((100.0, 101.5))
    monkeypatch.setattr(limits_module.time, "monotonic", lambda: next(clock))
    resolution = resolve_invocation_limits(
        feature_profile_for_route("openai_compat"),
        InvocationLimitRequest("diagnostic", 3, None, 1.0),
        route="openai_compat",
    )
    fake = _FakeBackend(_result())
    bind_invocation_limit_contract(fake, resolution)

    result = fake.invoke(_invocation(budget_usd=None))
    record = invocation_limit_record_from_result(result)

    assert len(fake.invocations) == 1
    assert result.stop_reason == "timeout"
    assert result.error is not None
    assert result.error.retriable is False
    assert record is not None
    assert record["terminal"]["terminal_reason"] == "deadline_overshoot"
    assert record["terminal"]["overshoots"] == ["deadline"]
    assert record["terminal"]["rankable"] is False


def test_unproved_diagnostic_cost_is_non_rankable_without_invented_overshoot(
) -> None:
    incomplete = InvocationResult(
        final_message="diagnostic",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )
    resolution = resolve_invocation_limits(
        feature_profile_for_route("codex"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="codex",
    )
    fake = _FakeBackend(incomplete)
    bind_invocation_limit_contract(fake, resolution)

    result = fake.invoke(_invocation())
    record = invocation_limit_record_from_result(result)

    assert result.error is not None
    assert result.error.kind == "incomplete_evidence"
    assert result.stop_reason == "limit_evidence_incomplete"
    assert record is not None
    assert record["terminal"]["overshoots"] == []
    assert record["terminal"]["rankable"] is False
    assert record["terminal"]["physical_attempts"] is None


def _identity() -> InvocationLimitIdentity:
    return InvocationLimitIdentity(
        role="implementer",
        backend="codex",
        model="gpt-5.6-sol",
    )


def _diagnostic_resolution():
    return resolve_invocation_limits(
        feature_profile_for_route("codex", model="gpt-5.6-sol"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 10),
        route="codex/gpt-5.6-sol for role 'implementer'",
    )


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_canonical(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_durable_ledger_arms_before_transport_and_settles_exact_terminal(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / INVOCATION_LIMIT_LEDGER_LEAF
    ledger = InvocationLimitLedger(ledger_path)

    class _InspectingBackend(_FakeBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            assert payload["schema_version"] == INVOCATION_LIMIT_LEDGER_SCHEMA
            assert len(payload["records"]) == 1
            armed = payload["records"][0]
            assert armed["state"] == "dispatched"
            assert armed["terminal"] is None
            assert armed["intent"]["role"] == "implementer"
            return super().invoke(invocation)

    fake = _InspectingBackend(_result(cost=0.25))
    bind_invocation_limit_contract(
        fake,
        _diagnostic_resolution(),
        evidence_ledger=ledger,
        identity=_identity(),
    )

    fake.invoke(_invocation())
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "records",
        "ledger_fingerprint",
    }
    assert payload["ledger_fingerprint"] == _canonical_fingerprint(
        {
            "schema_version": payload["schema_version"],
            "records": payload["records"],
        }
    )
    record = payload["records"][0]
    assert set(record) == {"sequence", "state", "intent", "terminal"}
    assert record["state"] == "terminal"
    assert record["intent"]["request"] == record["terminal"]["request"]
    usage = record["terminal"]["terminal"]["observed_usage"]
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.25,
        "token_usage_complete": True,
        "cost_complete": True,
        "terminal_complete": True,
        "trace_complete": True,
    }
    assert InvocationLimitLedger(ledger_path).snapshot() == payload


def test_reopened_ledger_refuses_prior_unsettled_dispatch(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / INVOCATION_LIMIT_LEDGER_LEAF
    ledger = InvocationLimitLedger(ledger_path)
    ledger.arm(_identity(), _diagnostic_resolution())
    reopened = InvocationLimitLedger(ledger_path)

    with pytest.raises(
        InvocationLimitContractError,
        match="prior unsettled provider dispatch",
    ):
        reopened.arm(_identity(), _diagnostic_resolution())


def test_ledger_rejects_existing_symlink_leaf(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    ledger_path = tmp_path / INVOCATION_LIMIT_LEDGER_LEAF
    ledger_path.symlink_to(outside)

    with pytest.raises(ValueError, match="regular, single-link file"):
        InvocationLimitLedger(ledger_path)


def test_separate_ledger_snapshot_cannot_overwrite_newer_evidence(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / INVOCATION_LIMIT_LEDGER_LEAF
    first = InvocationLimitLedger(ledger_path)
    stale = InvocationLimitLedger(ledger_path)
    first.arm(_identity(), _diagnostic_resolution())

    with pytest.raises(
        InvocationLimitContractError,
        match="changed through another owner",
    ):
        stale.arm(_identity(), _diagnostic_resolution())


def test_exact_run_ledger_rejects_policy_rebinding_and_mismatched_request(
    tmp_path: Path,
) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    ledger.bind_policy(
        InvocationLimitPolicy(
            mode="diagnostic",
            deadline_seconds=10,
            require_budget_usd=True,
        )
    )

    with pytest.raises(
        InvocationLimitContractError,
        match="different workflow policy",
    ):
        ledger.bind_policy(
            InvocationLimitPolicy(
                mode="diagnostic",
                deadline_seconds=20,
                require_budget_usd=True,
            )
        )
    mismatched = resolve_invocation_limits(
        feature_profile_for_route("codex", model="gpt-5.6-sol"),
        InvocationLimitRequest("diagnostic", 3, 1.0, 20),
        route="codex/gpt-5.6-sol for role 'implementer'",
    )
    with pytest.raises(
        InvocationLimitContractError,
        match="does not match the exact-run ledger policy",
    ):
        ledger.arm(_identity(), mismatched)


def test_limit_state_rejects_cross_run_or_policy_rebinding() -> None:
    state = InvocationLimitState()
    policy = InvocationLimitPolicy(
        mode="diagnostic",
        deadline_seconds=10,
        require_budget_usd=True,
    )
    state.bind(run_key="/runs/a/invocation-limits.json", policy=policy)
    state.bind(run_key="/runs/a/invocation-limits.json", policy=policy)

    with pytest.raises(
        InvocationLimitContractError,
        match="different run or workflow policy",
    ):
        state.bind(run_key="/runs/b/invocation-limits.json", policy=policy)


def test_nested_ledger_tamper_is_rejected_even_with_recomputed_fingerprint(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / INVOCATION_LIMIT_LEDGER_LEAF
    ledger = InvocationLimitLedger(ledger_path)
    fake = _FakeBackend(_result())
    bind_invocation_limit_contract(
        fake,
        _diagnostic_resolution(),
        evidence_ledger=ledger,
        identity=_identity(),
    )
    fake.invoke(_invocation())
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["records"][0]["terminal"]["terminal"]["unexpected"] = True
    body = {
        "schema_version": payload["schema_version"],
        "records": payload["records"],
    }
    payload["ledger_fingerprint"] = _canonical_fingerprint(body)
    _write_canonical(ledger_path, payload)

    with pytest.raises(
        InvocationLimitContractError,
        match="unreadable or malformed",
    ):
        InvocationLimitLedger(ledger_path)


def test_terminal_persistence_failure_blocks_same_backend_after_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.limits as limits_module

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)

    class _BreakSettlementBackend(_FakeBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            outcome = super().invoke(invocation)

            def _fail_fsync(_fd: int) -> None:
                raise OSError("directory durability denied")

            monkeypatch.setattr(limits_module.os, "fsync", _fail_fsync)
            return outcome

    fake = _BreakSettlementBackend(_result())
    bind_invocation_limit_contract(
        fake,
        _diagnostic_resolution(),
        evidence_ledger=ledger,
        identity=_identity(),
    )

    with pytest.raises(
        InvocationLimitContractError,
        match="terminal could not be durably persisted",
    ):
        fake.invoke(_invocation())
    with pytest.raises(
        InvocationLimitContractError,
        match="no recovery provider call is authorized",
    ):
        fake.invoke(_invocation())
    assert len(fake.invocations) == 1
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert payload["records"][0]["state"] == "dispatched"


def test_dispatch_arm_persistence_failure_precedes_provider_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.limits as limits_module

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    fake = _FakeBackend(_result())
    bind_invocation_limit_contract(
        fake,
        _diagnostic_resolution(),
        evidence_ledger=ledger,
        identity=_identity(),
    )
    monkeypatch.setattr(
        limits_module,
        "_write_strict_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("arm durability denied")
        ),
    )

    with pytest.raises(OSError, match="arm durability denied"):
        fake.invoke(_invocation())
    with pytest.raises(
        InvocationLimitContractError,
        match="dispatch intent could not be durably persisted",
    ):
        fake.invoke(_invocation())
    assert fake.invocations == []
    assert ledger.snapshot()["records"] == []


def test_backend_exception_is_settled_and_revokes_fixed_contract(
    tmp_path: Path,
) -> None:
    class _ExplodingBackend:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            self.calls += 1
            raise RuntimeError("provider exploded")

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    fake = _ExplodingBackend()
    bind_invocation_limit_contract(
        fake,
        _diagnostic_resolution(),
        evidence_ledger=ledger,
        identity=_identity(),
    )

    with pytest.raises(RuntimeError, match="provider exploded"):
        fake.invoke(_invocation())
    with pytest.raises(
        InvocationLimitContractError,
        match="lost complete invocation-limit authority",
    ):
        fake.invoke(_invocation())
    assert fake.calls == 1
    record = ledger.snapshot()["records"][0]
    assert record["state"] == "terminal"
    assert record["terminal"]["terminal"]["terminal_reason"] == (
        "backend_exception"
    )
    assert record["terminal"]["terminal"]["observed_usage"] is None


def test_non_result_backend_contract_violation_is_settled_and_revoked(
    tmp_path: Path,
) -> None:
    class _InvalidBackend:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _invocation: Invocation) -> Any:
            self.calls += 1
            return object()

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    fake = _InvalidBackend()
    bind_invocation_limit_contract(
        fake,
        _diagnostic_resolution(),
        evidence_ledger=ledger,
        identity=_identity(),
    )

    with pytest.raises(
        InvocationLimitContractError,
        match="non-InvocationResult",
    ):
        fake.invoke(_invocation())
    with pytest.raises(
        InvocationLimitContractError,
        match="lost complete invocation-limit authority",
    ):
        fake.invoke(_invocation())
    assert fake.calls == 1
    record = ledger.snapshot()["records"][0]
    assert record["state"] == "terminal"
    assert record["terminal"]["terminal"]["terminal_reason"] == (
        "backend_contract_violation"
    )


def test_post_return_normalization_failure_settles_and_revokes_shared_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crucible.backends.limits as limits_module

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    policy = InvocationLimitPolicy(
        mode="diagnostic",
        deadline_seconds=10,
        require_budget_usd=True,
    )
    ledger.bind_policy(policy)
    state = InvocationLimitState()
    state.bind(run_key=str(ledger.path), policy=policy)
    profile = feature_profile_for_route("codex", model="gpt-5.6-sol")
    first = _FakeBackend(_result())
    second = _FakeBackend(_result())
    for backend in (first, second):
        bind_invocation_limit_policy(
            backend,
            profile,
            policy,
            route="codex/gpt-5.6-sol for role 'implementer'",
            shared_state=state,
            evidence_ledger=ledger,
            identity=_identity(),
        )
    monkeypatch.setattr(
        limits_module,
        "_trace_with_terminal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("trace normalization failed")
        ),
    )

    with pytest.raises(RuntimeError, match="trace normalization failed"):
        first.invoke(_invocation())
    with pytest.raises(
        InvocationLimitContractError,
        match="lost complete invocation-limit authority",
    ):
        second.invoke(_invocation())
    assert len(first.invocations) == 1
    assert second.invocations == []
    record = ledger.snapshot()["records"][0]
    assert record["state"] == "terminal"
    assert record["terminal"]["terminal"]["terminal_reason"] == (
        "limit_normalization_failure"
    )
    assert state.revocation is not None


def test_incomplete_cost_with_proved_attempt_is_typed_and_revokes_backend(
) -> None:
    incomplete = _result(cost=0.0)
    incomplete = InvocationResult(
        final_message=incomplete.final_message,
        transcript=incomplete.transcript,
        usage=incomplete.usage,
        stop_reason=incomplete.stop_reason,
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=incomplete.execution_trace,
    )
    fake = _FakeBackend(incomplete)
    bind_invocation_limit_contract(fake, _diagnostic_resolution())

    result = fake.invoke(_invocation())
    record = invocation_limit_record_from_result(result)

    assert result.stop_reason == "limit_evidence_incomplete"
    assert result.error is not None
    assert result.error.kind == "incomplete_evidence"
    assert "invocation_limits.cost_complete" in result.error.missing
    assert record is not None
    observed = record["terminal"]["observed_usage"]
    assert observed["cost_complete"] is False
    assert record["terminal"]["rankable"] is False
    with pytest.raises(
        InvocationLimitContractError,
        match="lost complete invocation-limit authority",
    ):
        fake.invoke(_invocation())
    assert len(fake.invocations) == 1
