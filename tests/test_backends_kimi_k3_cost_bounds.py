# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Explicit, non-exact diagnostic accounting against the real locked worker."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from assay import BudgetExhausted, Usage, UsageCostBounds, UsageCostBoundSegment

from crucible.backends import cost_bounds
from crucible.backends import kimi as parent
from crucible.backends.pricing import DEFAULT_PRICING
from crucible.tests import test_backends_kimi_k3 as fixtures
from crucible.tests import test_backends_kimi_k3_stream_contract as stream_fixtures

worker = fixtures.worker
POLICY = "kimi-k3.1-pro-cache-read-upper-bound-v1"


def _missing_read(*, tool=False):
    chunks = fixtures._k3_chunks(tool=tool)
    usage = chunks[-1]["usage"]
    usage["prompt_tokens_details"].pop("cached_tokens")
    usage.pop("cached_tokens", None)
    return chunks


@pytest.mark.parametrize("module", [parent, worker])
def test_only_explicit_missing_both_read_counters_can_be_bounded(module):
    usage = _missing_read()[-1]["usage"]
    with pytest.raises(Exception, match="counter_missing"):
        module._validate_k3_raw_usage(usage)
    assert module._validate_k3_raw_usage(
        usage, diagnostic_cost_upper_bound=True,
    ) == (100, 20, 120, None, 30)
    for value in (None, False, 0.0, "0", -1):
        malformed = copy.deepcopy(usage)
        malformed["prompt_tokens_details"]["cached_tokens"] = value
        with pytest.raises(Exception):
            module._validate_k3_raw_usage(malformed, diagnostic_cost_upper_bound=True)
    malformed = copy.deepcopy(usage)
    malformed["cached_tokens"] = 0
    with pytest.raises(Exception, match="counter_missing"):
        module._validate_k3_raw_usage(malformed, diagnostic_cost_upper_bound=True)


@pytest.mark.parametrize("module", [parent, worker])
def test_bound_ledger_stops_at_cap_before_a_later_generation(module):
    emitted = []
    ledger = module._K3CostBoundLedger(True, 0.0004, emitted.append)
    ledger.settle((100, 20, 120, None, 30))
    assert ledger.upper == pytest.approx(0.0004)
    with pytest.raises(module._K3CostBoundBudget):
        ledger.admit_next()
    assert len(emitted) == 1
    assert emitted[0]["segment"]["cache_read_tokens"] is None


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("missing", [False, True])
def test_real_worker_bounded_response_retains_unknown_as_unknown(monkeypatch, missing):
    chunks = _missing_read() if missing else fixtures._k3_chunks()
    result, requests, _ = fixtures._invoke_real_worker(
        monkeypatch, chunks, diagnostic_cost_upper_bound=True, budget_usd=1.0,
    )
    assert len(requests) == 1
    assert result.error is None, result.error
    bounds = result.cost_bounds
    assert bounds is not None and bounds.covers(result.usage)
    assert bounds.coverage_complete
    assert len(bounds.segments) == 1
    assert bounds.segments[0].cache_read_tokens == (None if missing else 60)
    assert bounds.segments[0].policy == POLICY
    assert result.usage.input_tokens == 100
    assert result.usage.cache_creation_tokens == 30
    assert result.usage.cache_read_tokens == (0 if missing else 60)
    assert result.usage.cost_usd == pytest.approx(0.000274 if missing else 0.000292)
    assert bounds.upper_bound_usd == pytest.approx(0.0004 if missing else 0.000292)
    assert result.completeness.cost_complete is not missing
    assert result.completeness.token_usage_complete is not missing
    assert result.completeness.terminal_complete
    assert result.completeness.trace_complete


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_upper_bound_exhaustion_denies_tool_and_preserves_bound(monkeypatch):
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True,
        diagnostic_cost_upper_bound=True, budget_usd=0.0003,
    )
    assert len(requests) == 1
    assert toolset.calls == []
    assert isinstance(result.error, BudgetExhausted), result.error
    assert result.cost_bounds is not None
    assert result.cost_bounds.covers(result.usage)
    assert result.cost_bounds.upper_bound_usd == pytest.approx(0.0004)
    assert result.error.spent_usd == pytest.approx(0.0004)
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_multi_step_bounds_cover_each_response(monkeypatch):
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True,
        second_chunks=fixtures._k3_chunks(),
        diagnostic_cost_upper_bound=True, budget_usd=1.0,
    )
    assert len(requests) == 2
    assert len(toolset.calls) == 1
    assert result.error is None, result.error
    assert result.cost_bounds is not None and result.cost_bounds.covers(result.usage)
    assert [s.cache_read_tokens for s in result.cost_bounds.segments] == [None, 60]
    assert result.usage.cost_usd == pytest.approx(0.000274 + 0.000292)
    assert result.cost_bounds.upper_bound_usd == pytest.approx(0.0004 + 0.000292)
    assert not result.completeness.cost_complete
    assert result.completeness.terminal_complete and result.completeness.trace_complete


def test_parent_rejects_unbound_and_repeated_or_forged_checkpoints():
    latest = parent._WorkerOutcome(steps_used=1)
    record = {
        "type": "cost_bound", "ordinal": 0,
        "segment": {"input_tokens": 100, "output_tokens": 20,
                    "cache_creation_tokens": 30, "cache_read_tokens": None,
                    "policy": POLICY},
    }
    with pytest.raises(ValueError):
        parent._admit_cost_bound_checkpoint(
            latest, record, model=fixtures.PRO, diagnostic_cost_upper_bound=False,
        )
    admitted = parent._admit_cost_bound_checkpoint(
        latest, record, model=fixtures.PRO, diagnostic_cost_upper_bound=True,
    )
    for invalid_latest, invalid in (
        (admitted, record), (replace(latest, steps_used=2), record),
        (latest, {**record, "segment": {**record["segment"], "upper_bound_usd": 0}}),
        (latest, {**record, "segment": {**record["segment"], "input_tokens": 10 ** 1000}}),
    ):
        with pytest.raises(ValueError):
            parent._admit_cost_bound_checkpoint(
                invalid_latest, invalid, model=fixtures.PRO, diagnostic_cost_upper_bound=True,
            )
    _, fault = parent._consume_worker_line(
        json.dumps({"protocol": parent._WORKER_PROTOCOL, **record}).encode(),
        latest_usage=latest, max_steps=2, model=fixtures.PRO,
    )
    assert fault is not None and fault.error_kind == "qualification"


@pytest.mark.parametrize("override", [
    {"diagnostic_cost_upper_bound": 1}, {"diagnostic_budget_usd": None},
    {"diagnostic_budget_usd": float("inf")}, {"diagnostic_budget_usd": float("nan")},
    {"diagnostic_budget_usd": True}, {"diagnostic_budget_usd": -1},
    {"diagnostic_budget_usd": 0}, {"diagnostic_budget_usd": "1"},
    # Not an admitted model, so it cannot reach the bound policy either.
    {"model": fixtures.HIGHSPEED},
    {"reasoning_effort": "high"}, {"protocol": "crucible.kimi-worker.v6"},
    {"diagnostic_cost_upper_bound": False},
])
def test_worker_cost_policy_is_explicit_model_effort_and_budget_bound(override):
    payload = fixtures.fixtures._worker_request_payload(
        model=fixtures.PRO, reasoning_effort="max",
        diagnostic_cost_upper_bound=True, diagnostic_budget_usd=1.0,
    )
    accepted = worker._Request.from_wire(payload)
    assert accepted.diagnostic_cost_upper_bound and accepted.diagnostic_budget_usd == 1.0
    payload.update(override)
    with pytest.raises(worker.WorkerQualificationError):
        worker._Request.from_wire(payload)


@pytest.mark.parametrize("budget", [None, True, 0, -1, float("nan"), float("inf")])
def test_parent_missing_finite_budget_denies_before_worker(budget):
    called = []
    backend = parent.KimiBackend(
        model=fixtures.PRO, diagnostic_cost_upper_bound=True,
        _worker_runner=lambda *args, **kwargs: called.append(True),
    )
    try:
        with pytest.raises(ValueError, match="finite"):
            backend.invoke(replace(fixtures.fixtures._invocation(), budget_usd=budget))
    finally:
        backend.close()
    assert called == []


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("fault", [
    "null-read", "top-only-read", "missing-write", "malformed-write",
    "cache-overlap", "wrong-model", "no-done", "missing-final", "conflicting-final",
])
def test_real_worker_opt_in_does_not_admit_other_incomplete_or_unsafe_evidence(monkeypatch, fault):
    chunks = _missing_read(tool=True)
    usage = chunks[-1]["usage"]
    details = usage["prompt_tokens_details"]
    done = "[DONE]"
    if fault == "null-read":
        details["cached_tokens"] = None
    elif fault == "top-only-read":
        usage["cached_tokens"] = 0
    elif fault == "missing-write":
        del details["cache_write_tokens"]
    elif fault == "malformed-write":
        details["cache_write_tokens"] = "30"
    elif fault == "cache-overlap":
        details["cache_write_tokens"] = 101
    elif fault == "wrong-model":
        for chunk in chunks:
            chunk["model"] = fixtures.HIGHSPEED
    elif fault == "no-done":
        done = None
    elif fault == "missing-final":
        chunks.pop()
    elif fault == "conflicting-final":
        chunks[-2]["usage"] = copy.deepcopy(usage)
        usage["completion_tokens"] += 1
        usage["total_tokens"] += 1
    result, requests, toolset = stream_fixtures._run(
        monkeypatch, chunks, tool=True, done=done,
        diagnostic_cost_upper_bound=True, budget_usd=1.0,
    )
    assert len(requests) == 1
    assert toolset.calls == []
    assert result.error is not None
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete
    assert result.cost_bounds is not None and not result.cost_bounds.coverage_complete


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_budget_is_cumulative_and_denies_second_step_tools(monkeypatch):
    second = _missing_read(tool=True)
    second[0]["choices"][0]["delta"]["tool_calls"][0]["id"] = "second-tool"
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True, second_chunks=second,
        diagnostic_cost_upper_bound=True, budget_usd=0.0006,
    )
    assert len(requests) == 2
    assert len(toolset.calls) == 1
    assert isinstance(result.error, BudgetExhausted), result.error
    assert result.cost_bounds is not None and result.cost_bounds.covers(result.usage)
    assert len(result.cost_bounds.segments) == 2
    assert result.cost_bounds.upper_bound_usd == pytest.approx(0.0008)


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_exactly_exhausted_budget_stops_before_next_step(monkeypatch):
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True,
        second_chunks=fixtures._k3_chunks(),
        diagnostic_cost_upper_bound=True, budget_usd=0.0004,
    )
    assert len(requests) == 1
    assert len(toolset.calls) == 1
    assert isinstance(result.error, BudgetExhausted), result.error
    assert result.cost_bounds is not None and result.cost_bounds.covers(result.usage)
    assert result.cost_bounds.coverage_complete


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("budget,target", [
    (1.0, "output.txt"), (1.0, "outside.txt"), (0.0003, "output.txt"),
])
def test_real_worker_bound_policy_preserves_confined_write_authority(monkeypatch, tmp_path, budget, target):
    chunks = _missing_read(tool=True)
    chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"] = {
        "name": "write_source",
        "arguments": json.dumps({"file_path": target, "content": "bounded diagnostic write\n"}),
    }
    result, requests, _ = fixtures._invoke_real_worker(
        monkeypatch, chunks, tool=True, write_root=tmp_path,
        diagnostic_cost_upper_bound=True, budget_usd=budget,
    )
    assert len(requests) == 1
    assert not (tmp_path / "outside.txt").exists()
    if target == "output.txt" and budget == 1.0:
        assert (tmp_path / "output.txt").read_text() == "bounded diagnostic write\n"
    else:
        assert not (tmp_path / "output.txt").exists()
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete


@pytest.mark.parametrize("module", [parent, worker], ids=["parent", "worker"])
def test_bound_policy_identity_and_rates_agree_with_the_price_table(module):
    # The worker cannot import Crucible, so the policy id, the bound model and
    # the rate card exist three times. They must describe one policy.
    assert (
        cost_bounds.DIAGNOSTIC_COST_POLICY
        == parent._K3_BOUND_POLICY
        == worker._K3_BOUND_POLICY
        == POLICY
    )
    assert (
        cost_bounds.DIAGNOSTIC_COST_MODEL
        == parent._COST_BOUND_MODEL
        == worker._COST_BOUND_MODEL
        == "kimi-k3.1-pro"
    )
    pricing = DEFAULT_PRICING[cost_bounds.DIAGNOSTIC_COST_MODEL]
    for known_read in (60, None):
        record = {
            "input_tokens": 100, "output_tokens": 20,
            "cache_creation_tokens": 30, "cache_read_tokens": known_read,
            "policy": POLICY,
        }
        lower, upper = module._cost_bound_amounts(record)
        # Unknown reads: the lower bound prices every non-written input token
        # as a cache hit, the upper bound prices none of them as one.
        expected_lower = pricing.compute_cost(
            input_tokens=100, output_tokens=20, cache_creation_tokens=30,
            cache_read_tokens=70 if known_read is None else known_read,
        )
        expected_upper = pricing.compute_cost(
            input_tokens=100, output_tokens=20, cache_creation_tokens=30,
            cache_read_tokens=0 if known_read is None else known_read,
        )
        assert lower == pytest.approx(expected_lower, rel=1e-12)
        assert upper == pytest.approx(expected_upper, rel=1e-12)
        assert (lower, upper) == pytest.approx(
            (0.000292, 0.000292) if known_read is not None else (0.000274, 0.0004)
        )
        bounds = UsageCostBounds((UsageCostBoundSegment(
            input_tokens=100, output_tokens=20, cache_creation_tokens=30,
            cache_read_tokens=known_read, lower_bound_usd=lower,
            upper_bound_usd=upper, policy=POLICY,
        ),), coverage_complete=True)
        usage = Usage(
            100, 20, lower, cache_creation_tokens=30,
            cache_read_tokens=known_read or 0, cost_bounds=bounds,
        )
        assert cost_bounds.valid_diagnostic_cost_bounds(usage)


def test_bound_policy_does_not_reach_another_model_id():
    # One policy id, one exact route, one rate card. kimi-k3.1-pro-highspeed
    # is billed under another card: it is not admitted at all, and a bound
    # checkpoint attributed to it is never authorized.
    with pytest.raises(ValueError, match="supports only the qualified model"):
        parent.KimiBackend(model=fixtures.HIGHSPEED, diagnostic_cost_upper_bound=True)
    payload = fixtures.fixtures._worker_request_payload(
        model=fixtures.HIGHSPEED, reasoning_effort="max",
        diagnostic_cost_upper_bound=True, diagnostic_budget_usd=1.0,
    )
    with pytest.raises(worker.WorkerQualificationError, match="unqualified model/effort"):
        worker._Request.from_wire(payload)
    record = {
        "type": "cost_bound", "ordinal": 0,
        "segment": {"input_tokens": 100, "output_tokens": 20,
                    "cache_creation_tokens": 30, "cache_read_tokens": None,
                    "policy": POLICY},
    }
    latest = parent._WorkerOutcome(steps_used=1)
    with pytest.raises(ValueError, match="not authorized"):
        parent._admit_cost_bound_checkpoint(
            latest, record, model=fixtures.HIGHSPEED, diagnostic_cost_upper_bound=True,
        )
    assert parent._admit_cost_bound_checkpoint(
        latest, record, model=fixtures.PRO, diagnostic_cost_upper_bound=True,
    ).cost_bound_segments


def test_bound_policy_stays_with_its_model_if_another_id_were_admitted(monkeypatch):
    # The admitted set and the policy's model are separate tables. The policy
    # guards are kept so that admitting a second id cannot extend the bound
    # to a rate card the policy does not replay.
    other = "kimi-second-admitted-id"
    monkeypatch.setitem(parent._MODEL_CONTEXT_SIZES, other, 1_048_576)
    monkeypatch.setitem(worker._MODEL_CONTEXT_SIZES, other, 1_048_576)
    parent.KimiBackend(model=other).close()
    with pytest.raises(ValueError, match="exact kimi-k3.1-pro/max"):
        parent.KimiBackend(model=other, diagnostic_cost_upper_bound=True)
    payload = fixtures.fixtures._worker_request_payload(
        model=other, reasoning_effort="max",
    )
    assert worker._Request.from_wire(payload).model == other
    payload.update(diagnostic_cost_upper_bound=True, diagnostic_budget_usd=1.0)
    with pytest.raises(worker.WorkerQualificationError, match="diagnostic cost policy"):
        worker._Request.from_wire(payload)
