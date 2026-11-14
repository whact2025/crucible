# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Explicit, non-exact diagnostic accounting against the real locked worker."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from assay import BudgetExhausted

from crucible.backends import kimi as parent
from crucible.tests import test_backends_kimi_k3 as fixtures
from crucible.tests import test_backends_kimi_k3_stream_contract as stream_fixtures

worker = fixtures.worker
POLICY = "kimi-k3-cache-read-upper-bound-v1"


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
    ledger = module._K3CostBoundLedger(True, 0.0006, emitted.append)
    ledger.settle((100, 20, 120, None, 30))
    assert ledger.upper == pytest.approx(0.0006)
    with pytest.raises(module._K3CostBoundBudget):
        ledger.admit_next()
    assert len(emitted) == 1
    assert emitted[0]["segment"]["cache_read_tokens"] is None


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
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
    assert result.usage.cost_usd == pytest.approx(0.000411 if missing else 0.000438)
    assert bounds.upper_bound_usd == pytest.approx(0.0006 if missing else 0.000438)
    assert result.completeness.cost_complete is not missing
    assert result.completeness.token_usage_complete is not missing
    assert result.completeness.terminal_complete
    assert result.completeness.trace_complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
def test_real_worker_upper_bound_exhaustion_denies_tool_and_preserves_bound(monkeypatch):
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True,
        diagnostic_cost_upper_bound=True, budget_usd=0.0005,
    )
    assert len(requests) == 1
    assert toolset.calls == []
    assert isinstance(result.error, BudgetExhausted), result.error
    assert result.cost_bounds is not None
    assert result.cost_bounds.covers(result.usage)
    assert result.cost_bounds.upper_bound_usd == pytest.approx(0.0006)
    assert result.error.spent_usd == pytest.approx(0.0006)
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
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
    assert result.usage.cost_usd == pytest.approx(0.000411 + 0.000438)
    assert result.cost_bounds.upper_bound_usd == pytest.approx(0.0006 + 0.000438)
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
            latest, record, model="kimi-k3", diagnostic_cost_upper_bound=False,
        )
    admitted = parent._admit_cost_bound_checkpoint(
        latest, record, model="kimi-k3", diagnostic_cost_upper_bound=True,
    )
    for invalid_latest, invalid in (
        (admitted, record), (replace(latest, steps_used=2), record),
        (latest, {**record, "segment": {**record["segment"], "upper_bound_usd": 0}}),
        (latest, {**record, "segment": {**record["segment"], "input_tokens": 10 ** 1000}}),
    ):
        with pytest.raises(ValueError):
            parent._admit_cost_bound_checkpoint(
                invalid_latest, invalid, model="kimi-k3", diagnostic_cost_upper_bound=True,
            )
    _, fault = parent._consume_worker_line(
        json.dumps({"protocol": parent._WORKER_PROTOCOL, **record}).encode(),
        latest_usage=latest, max_steps=2, model="kimi-k3",
    )
    assert fault is not None and fault.error_kind == "qualification"


@pytest.mark.parametrize("override", [
    {"diagnostic_cost_upper_bound": 1}, {"diagnostic_budget_usd": None},
    {"diagnostic_budget_usd": float("inf")}, {"diagnostic_budget_usd": float("nan")},
    {"diagnostic_budget_usd": True}, {"diagnostic_budget_usd": -1},
    {"diagnostic_budget_usd": 0}, {"diagnostic_budget_usd": "1"},
    {"model": fixtures.K27, "reasoning_effort": None},
    {"reasoning_effort": "high"}, {"protocol": "crucible.kimi-worker.v5"},
    {"diagnostic_cost_upper_bound": False},
])
def test_worker_cost_policy_is_explicit_model_effort_and_budget_bound(override):
    payload = fixtures.fixtures._worker_request_payload(
        model=fixtures.K3, reasoning_effort="max",
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
        model=fixtures.K3, diagnostic_cost_upper_bound=True,
        _worker_runner=lambda *args, **kwargs: called.append(True),
    )
    try:
        with pytest.raises(ValueError, match="finite"):
            backend.invoke(replace(fixtures.fixtures._invocation(), budget_usd=budget))
    finally:
        backend.close()
    assert called == []


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
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
            chunk["model"] = fixtures.K27
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


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
def test_real_worker_budget_is_cumulative_and_denies_second_step_tools(monkeypatch):
    second = _missing_read(tool=True)
    second[0]["choices"][0]["delta"]["tool_calls"][0]["id"] = "second-tool"
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True, second_chunks=second,
        diagnostic_cost_upper_bound=True, budget_usd=0.001,
    )
    assert len(requests) == 2
    assert len(toolset.calls) == 1
    assert isinstance(result.error, BudgetExhausted), result.error
    assert result.cost_bounds is not None and result.cost_bounds.covers(result.usage)
    assert len(result.cost_bounds.segments) == 2
    assert result.cost_bounds.upper_bound_usd == pytest.approx(0.0012)


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
def test_real_worker_exactly_exhausted_budget_stops_before_next_step(monkeypatch):
    result, requests, toolset = fixtures._invoke_real_worker(
        monkeypatch, _missing_read(tool=True), tool=True,
        second_chunks=fixtures._k3_chunks(),
        diagnostic_cost_upper_bound=True, budget_usd=0.0006,
    )
    assert len(requests) == 1
    assert len(toolset.calls) == 1
    assert isinstance(result.error, BudgetExhausted), result.error
    assert result.cost_bounds is not None and result.cost_bounds.covers(result.usage)
    assert result.cost_bounds.coverage_complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("budget,target", [
    (1.0, "output.txt"), (1.0, "outside.txt"), (0.0005, "output.txt"),
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
