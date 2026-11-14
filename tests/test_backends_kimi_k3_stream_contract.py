# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Documented K3 progress/final stream shapes, without provider calls."""

from __future__ import annotations

import copy
import json

import pytest

from crucible.tests import test_backends_kimi_k3 as fixtures

parent = fixtures.parent
worker = fixtures.worker


def _progress():
    return {
        "prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": None, "cache_write_tokens": None},
    }


def _run(monkeypatch, chunks, *, tool=False, done="[DONE]", **kwargs):
    # Exercise the installed SDK worker and its real HTTP/SSE parser. This
    # transport differs from the shared fixture only in its final SSE marker.
    class Endpoint(fixtures.fixtures._NativeStreamEndpoint):
        def do_POST(self):  # noqa: N802 - HTTP handler API
            size = int(self.headers.get("content-length", "0"))
            self.server.requests.append(json.loads(self.rfile.read(size)))
            encoded = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n"
                for chunk in self.server.chunks
            )
            if done is not None:
                encoded += b"data: " + done.encode() + b"\n\n"
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(encoded)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
            self.close_connection = True

    monkeypatch.setattr(fixtures.fixtures, "_NativeStreamEndpoint", Endpoint)
    return fixtures._invoke_real_worker(monkeypatch, chunks, tool=tool, **kwargs)


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("shape", [
    "finish-top-level", "partial-progress", "partial-finish", "null-finish-details",
    "choice-local-progress", "matching-finish-and-stats",
])
def test_real_worker_admits_documented_final_usage_once(monkeypatch, shape):
    chunks = fixtures._k3_chunks()
    full_usage = copy.deepcopy(chunks[-1]["usage"])
    if shape == "finish-top-level":
        chunks[-2]["usage"] = full_usage
        chunks.pop()
    elif shape == "partial-progress":
        chunks[0]["usage"] = _progress()
    elif shape == "partial-finish":
        chunks[-2]["usage"] = _progress()
    elif shape == "null-finish-details":
        chunks[-2]["usage"] = {**_progress(), "prompt_tokens_details": None}
    elif shape == "choice-local-progress":
        chunks[-2]["choices"][0]["usage"] = {"prompt_tokens": 10}
    elif shape == "matching-finish-and-stats":
        chunks[-2]["usage"] = full_usage
    result, requests, _ = _run(monkeypatch, chunks)
    assert len(requests) == 1
    assert result.error is None
    assert result.final_message == "qualified K3 answer"
    assert result.completeness.cost_complete
    assert result.completeness.terminal_complete
    assert result.pricing_evidence.complete
    assert len(result.pricing_evidence.segments) == 1
    assert result.usage.input_tokens == 100
    assert result.usage.cache_creation_tokens == 30
    assert result.usage.cache_read_tokens == 60
    assert result.usage.output_tokens == 20
    assert result.usage.cost_usd == pytest.approx(0.000438)


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("shape,code", [
    ("missing-done", "missing_done"), ("invalid-done", "invalid_done"),
    ("partial-finish-without-stats", "missing_terminal_usage"),
    ("choice-local-without-stats", "missing_terminal_usage"),
    ("missing-final-write", "counter_missing"),
    ("malformed-final-write", "counter_type"),
    ("contradictory-finish-and-stats", "contradictory_terminal_usage"),
    ("malformed-complete-finish", "counter_type"),
    ("duplicate-final-stats", "data_after_final_stats"),
    ("choice-after-finish", "choice_after_finish"),
])
def test_real_worker_rejects_incomplete_or_ambiguous_final_before_tools(
    monkeypatch, shape, code,
):
    chunks = fixtures._k3_chunks(tool=True)
    done = "[DONE]"
    if shape == "missing-done":
        done = None
    elif shape == "invalid-done":
        done = "[DONE]unexpected"
    elif shape == "partial-finish-without-stats":
        chunks[-2]["usage"] = _progress()
        chunks.pop()
    elif shape == "choice-local-without-stats":
        chunks[-2]["choices"][0]["usage"] = chunks[-1]["usage"]
        chunks.pop()
    elif shape == "missing-final-write":
        del chunks[-1]["usage"]["prompt_tokens_details"]["cache_write_tokens"]
    elif shape == "malformed-final-write":
        chunks[-1]["usage"]["prompt_tokens_details"]["cache_write_tokens"] = "30"
    elif shape == "contradictory-finish-and-stats":
        chunks[-2]["usage"] = copy.deepcopy(chunks[-1]["usage"])
        chunks[-1]["usage"].update(prompt_tokens=101, total_tokens=121)
    elif shape == "malformed-complete-finish":
        chunks[-2]["usage"] = copy.deepcopy(chunks[-1]["usage"])
        chunks[-2]["usage"]["prompt_tokens"] = 100.0
    elif shape == "duplicate-final-stats":
        chunks.append(copy.deepcopy(chunks[-1]))
        chunks[-1]["usage"].update(prompt_tokens=101, total_tokens=121)
    elif shape == "choice-after-finish":
        chunks.insert(-1, fixtures.fixtures._native_chunk(
            model=fixtures.K3, delta={"content": "unexpected trailing content"},
        ))
    result, requests, toolset = _run(monkeypatch, chunks, tool=True, done=done)
    assert len(requests) == 1
    assert toolset.calls == []
    assert result.error is not None
    assert f"code={code}" in result.error.message
    assert result.final_message == ""
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete
    assert not result.pricing_evidence.complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("tool", [False, True], ids=["observed-stop", "tool-dispatch-denied"])
def test_real_worker_rejects_observed_missing_cache_read(monkeypatch, tool):
    # Numeric-only shape observed on 2026-09-24: both the finish choice and
    # final statistics omit cache reads. Explicit W=0 does not establish R=0.
    # No live output, provider identifiers, or credentials enter this fixture.
    usage = {
        "prompt_tokens": 126,
        "completion_tokens": 74,
        "total_tokens": 200,
        "prompt_tokens_details": {"cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 51},
    }
    chunks = fixtures._k3_chunks(tool=tool)
    chunks[-2]["choices"][0]["usage"] = copy.deepcopy(usage)
    chunks[-1]["usage"] = usage

    result, requests, toolset = _run(monkeypatch, chunks, tool=tool, done="[DONE]")

    assert len(requests) == 1
    assert requests[0]["model"] == fixtures.K3
    assert toolset.calls == []
    assert result.error is not None
    assert result.error.message == (
        "WorkerQualificationError: Kimi K3 usage has invalid raw billing counters; "
        "code=counter_missing; field=R; state=missing"
    )
    assert result.error.retriable is False
    assert result.final_message == ""
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete
    assert not result.pricing_evidence.complete


@pytest.mark.parametrize("module", [parent, worker], ids=["parent", "worker"])
@pytest.mark.parametrize("value,state", [
    (None, "null"), (True, "boolean"), (1.0, "float"),
    ("never-publish-provider-content-or-credentials", "string"),
    (["secret"], "array"), ({"secret": "value"}, "object"),
])
def test_billing_diagnostic_contains_only_constant_field_and_type_labels(module, value, state):
    usage = fixtures._k3_chunks()[-1]["usage"]
    usage["prompt_tokens"] = value
    with pytest.raises(RuntimeError) as error:
        module._validate_k3_raw_usage(usage)
    assert str(error.value) == (
        "Kimi K3 usage has invalid raw billing counters; "
        f"code=counter_type; field=P; state={state}"
    )


@pytest.mark.parametrize("module", [parent, worker], ids=["parent", "worker"])
@pytest.mark.parametrize("mutation,code", [
    ("total", "total_mismatch"), ("overlap", "cache_overlap"),
    ("alias", "alias_conflict"), ("alias-null", "alias_type"),
])
def test_billing_diagnostic_distinguishes_arithmetic_and_alias_failures(module, mutation, code):
    usage = fixtures._k3_chunks()[-1]["usage"]
    if mutation == "total":
        usage["total_tokens"] = 121
    elif mutation == "overlap":
        usage["prompt_tokens_details"]["cache_write_tokens"] = 41
    elif mutation == "alias":
        usage["cached_tokens"] = 61
    else:
        usage["cached_tokens"] = None
    with pytest.raises(RuntimeError, match=f"code={code}"):
        module._validate_k3_raw_usage(usage)
