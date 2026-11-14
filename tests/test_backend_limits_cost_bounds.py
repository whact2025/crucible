# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Offline diagnostic K3 bounds never become complete billing or old authority."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    InvocationResult,
    TransportError,
    Usage,
    UsageCostBounds,
    UsageCostBoundSegment,
)
from crucible.backends.cost_bounds import (
    DIAGNOSTIC_COST_POLICY,
    EXACT_COST_BOUND_POLICY,
)
from crucible.backends.features import feature_profile_for_route
from crucible.backends.limits import (
    INVOCATION_LIMIT_LEDGER_LEAF,
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

from .test_backend_limits import (
    _FakeBackend,
    _canonical_fingerprint,
    _diagnostic_resolution,
    _identity,
    _invocation,
    _result,
    _write_canonical,
)


def _profile():
    return feature_profile_for_route(
        "kimi", model="kimi-k3", base_url="https://api.moonshot.ai/v1",
    )


def _resolution(*, budget: float = 1.0):
    return resolve_invocation_limits(
        _profile(),
        InvocationLimitRequest(
            "diagnostic", 3, budget, 10,
            diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
        ),
        route="kimi/kimi-k3 for role 'implementer'",
    )


def _bound_result(*, known_read: bool = False) -> InvocationResult:
    read = 20 if known_read else None
    lower = (106 * 3 + 20 * .3 + 74 * 15) / 1e6 if known_read else .0011478
    upper = lower if known_read else .001488
    bounds = UsageCostBounds(
        segments=(UsageCostBoundSegment(
            input_tokens=126, output_tokens=74, cache_creation_tokens=0,
            cache_read_tokens=read, lower_bound_usd=lower,
            upper_bound_usd=upper, policy=DIAGNOSTIC_COST_POLICY,
        ),),
        coverage_complete=True,
    )
    return InvocationResult(
        final_message="done", transcript=(), stop_reason="end_turn",
        usage=Usage(
            input_tokens=126, output_tokens=74, cache_read_tokens=read or 0,
            cost_usd=lower, cost_bounds=bounds,
        ),
        completeness=ExecutionCompleteness(
            token_usage_complete=known_read, cost_complete=known_read,
            terminal_complete=True, trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="kimi", requested_model="kimi-k3", served_model="kimi-k3",
            requested_effort="max", endpoint="https://api.moonshot.ai/v1",
            feature_profile_digest=_resolution().feature_profile_digest,
            events=(
                BackendExecutionEvent(sequence=0, kind="request"),
                BackendExecutionEvent(sequence=1, kind="terminal", name="end_turn"),
            ),
        ),
    )


def _backend(result: InvocationResult | None = None):
    backend = _FakeBackend(result or _bound_result(), name="kimi", model="kimi-k3")
    backend.effort = "max"
    backend.diagnostic_cost_upper_bound = True
    return backend


def _kimi_identity():
    return InvocationLimitIdentity("implementer", "kimi", "kimi-k3")


def _bind(backend, ledger, *, budget: float = 1.0):
    bind_invocation_limit_contract(
        backend, _resolution(budget=budget),
        evidence_ledger=ledger, identity=_kimi_identity(),
    )


@pytest.mark.parametrize("known_read", [False, True])
def test_bound_settlement_separates_actuals_and_allows_durable_next_call(
    tmp_path: Path, known_read: bool,
) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend(_bound_result(known_read=known_read))
    _bind(backend, ledger)
    result = backend.invoke(_invocation())
    assert result.error is None
    assert result.completeness.cost_complete is known_read
    assert result.completeness.token_usage_complete is known_read
    terminal = invocation_limit_terminal(backend)
    assert terminal is not None and terminal.rankable is False
    assert terminal.diagnostic_cost_authority
    assert not terminal.revokes_diagnostic_authority
    record = invocation_limit_record_from_result(result)
    assert record == ledger.snapshot()["records"][0]["terminal"]
    observed = record["terminal"]["observed_usage"]
    assert observed["cost_usd"] == result.usage.cost_usd
    assert observed["cost_complete"] is known_read
    bound_record = record["terminal"]["cost_bounds"]
    assert bound_record["segments"][0]["cache_read_tokens"] == (20 if known_read else None)
    assert record["request"]["diagnostic_cost_policy"] == DIAGNOSTIC_COST_POLICY
    # A fresh object may continue only because the previous typed bound settled.
    reopened = InvocationLimitLedger(ledger.path)
    second = _backend()
    _bind(second, reopened)
    assert second.invoke(_invocation()).error is None
    assert len(reopened.snapshot()["records"]) == 2


@pytest.mark.parametrize("mode,budget,policy", [
    ("strict", 1.0, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", None, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", 0.0, DIAGNOSTIC_COST_POLICY),
    ("diagnostic", float("inf"), DIAGNOSTIC_COST_POLICY),
    ("diagnostic", 1.0, "other"),
    ("diagnostic", 1.0, True),
])
def test_bound_request_rejects_unauthorized_policy(mode, budget, policy) -> None:
    with pytest.raises(ValueError):
        InvocationLimitRequest(mode, 3, budget, 10, diagnostic_cost_policy=policy)


@pytest.mark.parametrize("mode,budgeted", [("strict", True), ("diagnostic", False)])
def test_bound_workflow_policy_requires_diagnostic_budget(mode, budgeted) -> None:
    with pytest.raises(ValueError):
        InvocationLimitPolicy(mode, 10, budgeted, DIAGNOSTIC_COST_POLICY)


def test_bound_mode_requires_durable_ledger_before_transport() -> None:
    backend = _backend()
    with pytest.raises(InvocationLimitContractError, match="durable"):
        bind_invocation_limit_contract(backend, _resolution())
    assert backend.invocations == []


@pytest.mark.parametrize("field,value", [
    ("diagnostic_cost_upper_bound", False),
    ("diagnostic_cost_upper_bound", 1),
    ("model", "kimi-k2.7-code-highspeed"),
    ("effort", "high"),
    ("name", "openai_compat"),
])
def test_bound_route_drift_rejected_before_transport(tmp_path, field, value) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend()
    _bind(backend, ledger)
    setattr(backend, field, value)
    with pytest.raises(InvocationLimitContractError):
        backend.invoke(_invocation())
    assert backend.invocations == []
    assert ledger.snapshot()["records"] == []


def test_adapter_optin_without_limit_policy_is_rejected() -> None:
    backend = _backend()
    with pytest.raises(InvocationLimitContractError, match="explicit opt-in"):
        bind_invocation_limit_contract(backend, _diagnostic_resolution())
    assert backend.invocations == []


@pytest.mark.parametrize("failure", [
    "missing_bounds", "incomplete_coverage", "wrong_formula", "wrong_policy",
    "missing_terminal", "missing_trace", "missing_attempt", "two_attempts",
    "served_model", "requested_model", "requested_effort", "profile_digest",
    "endpoint_drift", "endpoint_missing",
    "terminal_event", "error", "stop_reason", "mismatched_counters",
])
def test_other_incomplete_or_failed_evidence_revokes_bound_authority(
    tmp_path: Path, failure: str,
) -> None:
    result = _bound_result()
    trace = result.execution_trace
    bounds = result.usage.cost_bounds
    if failure == "missing_bounds":
        result = replace(result, usage=replace(result.usage, cost_bounds=None))
    elif failure in {"incomplete_coverage", "wrong_formula", "wrong_policy"}:
        if failure == "incomplete_coverage":
            bounds = replace(bounds, coverage_complete=False)
        else:
            change = {"upper_bound_usd": .002} if failure == "wrong_formula" else {"policy": "other"}
            bounds = replace(bounds, segments=(replace(bounds.segments[0], **change),))
        result = replace(result, usage=replace(result.usage, cost_bounds=bounds))
    elif failure in {"missing_terminal", "missing_trace"}:
        field = "terminal_complete" if failure == "missing_terminal" else "trace_complete"
        result = replace(result, completeness=replace(result.completeness, **{field: False}))
    elif failure in {"missing_attempt", "two_attempts", "terminal_event"}:
        events = trace.events
        if failure == "missing_attempt":
            events = (events[-1],)
        elif failure == "terminal_event":
            events = (events[0], replace(events[-1], name="other"))
        else:
            events = (events[0], BackendExecutionEvent(sequence=1, kind="request"),
                      replace(events[-1], sequence=2))
        events = tuple(replace(event, sequence=i) for i, event in enumerate(events))
        result = replace(result, execution_trace=replace(trace, events=events))
    elif failure in {"served_model", "requested_model", "requested_effort", "profile_digest"}:
        field = "feature_profile_digest" if failure == "profile_digest" else failure
        value = "sha256:" + "0" * 64 if failure == "profile_digest" else "other"
        result = replace(result, execution_trace=replace(trace, **{field: value}))
    elif failure in {"endpoint_drift", "endpoint_missing"}:
        endpoint = "https://example.invalid/v1" if failure == "endpoint_drift" else None
        result = replace(result, execution_trace=replace(trace, endpoint=endpoint))
    elif failure == "error":
        result = replace(result, error=TransportError(message="failed", retriable=False))
    elif failure == "stop_reason":
        result = replace(result, stop_reason="max_turns")
    else:
        result = replace(result, usage=replace(result.usage, input_tokens=127))
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend(result)
    _bind(backend, ledger)
    returned = backend.invoke(_invocation())
    assert returned.error is not None
    terminal = invocation_limit_terminal(backend)
    assert terminal.rankable is False and terminal.revokes_diagnostic_authority
    with pytest.raises(InvocationLimitContractError):
        backend.invoke(_invocation())
    reopened = InvocationLimitLedger(ledger.path)
    with pytest.raises(InvocationLimitContractError, match="revoked"):
        reopened.arm(_kimi_identity(), _resolution())
    assert len(backend.invocations) == 1


def test_upper_bound_not_actual_lower_bound_enforces_usd_cap(tmp_path) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend()
    _bind(backend, ledger, budget=.0012)
    result = backend.invoke(_invocation(budget_usd=.0012))
    assert result.error.kind == "budget_exhausted"
    assert result.error.spent_usd == .001488
    assert "diagnostic cost upper bound" in result.error.message
    assert result.usage.cost_usd == .0011478
    assert result.completeness.cost_complete is False
    terminal = invocation_limit_terminal(backend)
    assert terminal.overshoots == ("usd",)
    assert terminal.cost_bounds.upper_bound_usd == .001488
    assert terminal.revokes_diagnostic_authority
    with pytest.raises(InvocationLimitContractError):
        backend.invoke(_invocation(budget_usd=.0012))


def test_exact_upper_bound_at_cap_is_still_diagnostic(tmp_path) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend()
    _bind(backend, ledger, budget=.001488)
    assert backend.invoke(_invocation(budget_usd=.001488)).error is None
    terminal = invocation_limit_terminal(backend)
    assert terminal.diagnostic_cost_authority and not terminal.rankable


@pytest.mark.parametrize("field", ["cost_complete", "token_usage_complete"])
def test_unknown_read_cannot_claim_complete_actuals(tmp_path, field) -> None:
    result = _bound_result()
    # The Assay constructor already rejects this contradiction. Corrupt the
    # fixture deliberately to exercise the independent durable-ledger guard.
    object.__setattr__(result, "completeness", replace(result.completeness, **{field: True}))
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend(result)
    _bind(backend, ledger)
    with pytest.raises(ValueError, match="complete actual accounting"):
        backend.invoke(_invocation())
    assert invocation_limit_terminal(backend).revokes_diagnostic_authority
    assert ledger.snapshot()["records"][0]["state"] == "terminal"
    with pytest.raises(InvocationLimitContractError):
        backend.invoke(_invocation())


def test_aggregate_exact_cost_segment_is_not_per_invocation_k3_authority(tmp_path) -> None:
    result = _bound_result()
    bounds = result.usage.cost_bounds
    injected = UsageCostBoundSegment(
        input_tokens=1000, output_tokens=0, cache_creation_tokens=0,
        cache_read_tokens=0, lower_bound_usd=0., upper_bound_usd=0.,
        policy=EXACT_COST_BOUND_POLICY,
    )
    bounds = replace(bounds, segments=(*bounds.segments, injected))
    result = replace(result, usage=replace(result.usage, input_tokens=1126, cost_bounds=bounds))
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend(result)
    _bind(backend, ledger)
    assert backend.invoke(_invocation()).error is not None
    assert invocation_limit_terminal(backend).revokes_diagnostic_authority


def test_bound_deadline_overshoot_revokes_authority(tmp_path, monkeypatch) -> None:
    import crucible.backends.limits as module

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend()
    _bind(backend, ledger)
    ticks = iter((0., 11.))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    assert backend.invoke(_invocation()).stop_reason == "timeout"
    assert invocation_limit_terminal(backend).revokes_diagnostic_authority


def test_bound_policy_requires_prior_durable_settlement(tmp_path) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    ledger.arm(_kimi_identity(), _resolution())
    with pytest.raises(InvocationLimitContractError, match="prior settlement"):
        ledger.arm(_kimi_identity(), _resolution())


def test_bound_policy_cannot_retrofit_legacy_complete_run(tmp_path) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _FakeBackend(_result(cost=.25))
    bind_invocation_limit_contract(
        backend, _diagnostic_resolution(), evidence_ledger=ledger, identity=_identity(),
    )
    backend.invoke(_invocation())
    before = ledger.path.read_bytes()
    reopened = InvocationLimitLedger(ledger.path)
    assert reopened.path.read_bytes() == before
    with pytest.raises(InvocationLimitContractError, match="reinterpret"):
        reopened.arm(_kimi_identity(), _resolution())
    with pytest.raises(InvocationLimitContractError, match="workflow policy"):
        reopened.bind_policy(InvocationLimitPolicy(
            "diagnostic", 10, True, DIAGNOSTIC_COST_POLICY,
        ))


def test_legacy_revocation_survives_ledger_reload(tmp_path) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _FakeBackend(replace(_result(), completeness=ExecutionCompleteness()))
    bind_invocation_limit_contract(
        backend, _diagnostic_resolution(), evidence_ledger=ledger, identity=_identity(),
    )
    backend.invoke(_invocation())
    for owner in (ledger, InvocationLimitLedger(ledger.path)):
        with pytest.raises(InvocationLimitContractError, match="revoked"):
            owner.arm(_identity(), _diagnostic_resolution())


def test_prior_shared_revocation_is_never_cleared_by_valid_bound(tmp_path) -> None:
    state = InvocationLimitState()
    failed = _FakeBackend(replace(_result(), completeness=ExecutionCompleteness()))
    bind_invocation_limit_contract(failed, _diagnostic_resolution())
    failed.invoke(_invocation())
    prior = invocation_limit_terminal(failed)
    state.revoke(prior)
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    policy = InvocationLimitPolicy("diagnostic", 10, True, DIAGNOSTIC_COST_POLICY)
    backend = _backend()
    bind_invocation_limit_policy(
        backend, _profile(), policy, route=_resolution().route,
        shared_state=state, evidence_ledger=ledger, identity=_kimi_identity(),
    )
    with pytest.raises(InvocationLimitContractError):
        backend.invoke(_invocation())
    assert state.revocation is prior
    assert backend.invocations == []
    assert ledger.snapshot()["records"] == []


def test_bound_workflow_policy_preserves_shared_authority_only_after_fsync(
    tmp_path, monkeypatch,
) -> None:
    import crucible.backends.limits as module

    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    policy = InvocationLimitPolicy("diagnostic", 10, True, DIAGNOSTIC_COST_POLICY)
    ledger.bind_policy(policy)
    state = InvocationLimitState()
    state.bind(run_key=str(ledger.path), policy=policy)
    first, second = _backend(), _backend()
    for backend in (first, second):
        bind_invocation_limit_policy(
            backend, _profile(), policy, route=_resolution().route,
            shared_state=state, evidence_ledger=ledger, identity=_kimi_identity(),
        )
    assert first.invoke(_invocation()).error is None
    assert state.revocation is None
    original = module._write_strict_atomic_json

    def fail_terminal(path, payload):
        if payload["records"][-1]["state"] == "terminal":
            raise OSError("terminal fsync denied")
        original(path, payload)

    monkeypatch.setattr(module, "_write_strict_atomic_json", fail_terminal)
    with pytest.raises(InvocationLimitContractError, match="durably persisted"):
        second.invoke(_invocation(budget_usd=.5))
    assert state.revocation is not None
    with pytest.raises(InvocationLimitContractError):
        first.invoke(_invocation())
    assert len(first.invocations) == len(second.invocations) == 1


@pytest.mark.parametrize("tamper", ["extra", "null", "rankable", "formula", "policy", "identity"])
def test_closed_bound_ledger_rejects_tampering_even_with_outer_fingerprint(
    tmp_path, tamper,
) -> None:
    ledger = InvocationLimitLedger(tmp_path / INVOCATION_LIMIT_LEDGER_LEAF)
    backend = _backend()
    _bind(backend, ledger)
    backend.invoke(_invocation())
    payload = json.loads(ledger.path.read_text())
    terminal = payload["records"][0]["terminal"]["terminal"]
    if tamper == "extra":
        terminal["cost_bounds"]["unknown"] = True
    elif tamper == "null":
        terminal["cost_bounds"] = None
    elif tamper == "rankable":
        terminal["rankable"] = True
    elif tamper == "formula":
        terminal["cost_bounds"]["segments"][0]["upper_bound_usd"] = .002
    elif tamper == "identity":
        payload["records"][0]["intent"]["backend"] = "codex"
    else:
        payload["records"][0]["intent"]["request"].pop("diagnostic_cost_policy")
        payload["records"][0]["terminal"]["request"].pop("diagnostic_cost_policy")
    payload["ledger_fingerprint"] = _canonical_fingerprint({
        "schema_version": payload["schema_version"], "records": payload["records"],
    })
    _write_canonical(ledger.path, payload)
    with pytest.raises(InvocationLimitContractError, match="malformed"):
        InvocationLimitLedger(ledger.path)
