# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Documented K3-series progress/final stream shapes, without provider calls."""

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


# Final statistics recorded for kimi-k3.1-pro on 2026-11-18 by a raw streamed
# request without the adapter: key names and counters only, no provider text,
# identifier or credential. Moonshot omitted ``cached_tokens`` when nothing
# was read from cache, with and without a cache write, and sent it under
# ``prompt_tokens_details`` and as an equal top-level alias on a cache hit.
# The finish choice repeated the same counters in each of the three streams.
_K31_PRO_RECORDED_USAGE = {
    "miss": {
        "prompt_tokens": 106, "completion_tokens": 52, "total_tokens": 158,
        "prompt_tokens_details": {"cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 36},
    },
    "miss-with-write": {
        "prompt_tokens": 4533, "completion_tokens": 68, "total_tokens": 4601,
        "prompt_tokens_details": {"cache_write_tokens": 4352},
        "completion_tokens_details": {"reasoning_tokens": 52},
    },
    "hit": {
        "prompt_tokens": 4533, "completion_tokens": 42, "total_tokens": 4575,
        "cached_tokens": 4352,
        "prompt_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 4352},
        "completion_tokens_details": {"reasoning_tokens": 26},
    },
}


def _recorded_k31_pro_chunks(shape):
    usage = copy.deepcopy(_K31_PRO_RECORDED_USAGE[shape])
    chunks = fixtures._k3_chunks()
    chunks[-2]["choices"][0]["usage"] = copy.deepcopy(usage)
    chunks[-1]["usage"] = usage
    return chunks


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


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
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
    assert result.usage.cost_usd == pytest.approx(0.000292)


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
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
            model=fixtures.PRO, delta={"content": "unexpected trailing content"},
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


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("tool", [False, True], ids=["observed-stop", "tool-dispatch-denied"])
def test_real_worker_rejects_observed_missing_cache_read(monkeypatch, tool):
    # Numeric-only shape observed with kimi-k3 on 2026-09-24 and again with
    # kimi-k3.1-pro on 2026-11-18: both the finish choice and final statistics
    # omit cache reads. Explicit W=0 does not establish R=0. The counters
    # below are the kimi-k3 recording, replayed against the current request
    # id; the kimi-k3.1-pro counters are _K31_PRO_RECORDED_USAGE above.
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
    assert requests[0]["model"] == fixtures.PRO
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
@pytest.mark.parametrize("shape,exact", [
    ("miss", None),
    ("miss-with-write", None),
    ("hit", (4533, 42, 4575, 4352, 0)),
], ids=["miss", "miss-with-write", "hit"])
def test_recorded_k31_pro_final_usage_is_exact_only_with_a_read_count(module, shape, exact):
    # Needs no worker environment: the validator and the stream state of the
    # parent and of the worker source are called directly.
    usage = _K31_PRO_RECORDED_USAGE[shape]
    prompt, output, total = (
        usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
    )
    write = usage["prompt_tokens_details"]["cache_write_tokens"]
    missing_read = (
        "Kimi K3 usage has invalid raw billing counters; "
        "code=counter_missing; field=R; state=missing"
    )
    if exact is None:
        # Strict accounting never reads the missing counter as zero.
        with pytest.raises(RuntimeError) as error:
            module._validate_k3_raw_usage(copy.deepcopy(usage))
        assert str(error.value) == missing_read
        bounded = (prompt, output, total, None, write)
    else:
        # The only recorded shape strict accounting accepts: the nested
        # counter and an equal top-level alias.
        assert module._validate_k3_raw_usage(copy.deepcopy(usage)) == exact
        bounded = exact
    assert module._validate_k3_raw_usage(
        copy.deepcopy(usage), diagnostic_cost_upper_bound=True,
    ) == bounded

    chunks = _recorded_k31_pro_chunks(shape)
    for bound in (False, True):
        state = module._K3StreamUsageState(diagnostic_cost_upper_bound=bound)
        state.observe(copy.deepcopy(chunks[0]))
        state.observe(copy.deepcopy(chunks[1]))
        # Choice-local usage on the finish chunk is never billing authority.
        assert state.finish_seen and state.counters is None
        if exact is None and not bound:
            with pytest.raises(RuntimeError) as error:
                state.observe(copy.deepcopy(chunks[2]))
            assert str(error.value) == missing_read
            assert state.counters is None and not state.final_stats_seen
            continue
        state.observe(copy.deepcopy(chunks[2]))
        assert state.counters == bounded
        state.done = True
        assert state.canonical_usage() == {
            "prompt_tokens": prompt, "completion_tokens": output,
            "total_tokens": total,
            "prompt_tokens_details": {
                "cached_tokens": bounded[3], "cache_write_tokens": write,
            },
        }


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_accepts_recorded_cache_hit_with_equal_alias(monkeypatch):
    # The hit shape through the pinned SDK and its SSE parser, under strict
    # accounting: nested read count, equal top-level alias, W=0.
    result, requests, _ = _run(monkeypatch, _recorded_k31_pro_chunks("hit"))

    assert len(requests) == 1
    assert requests[0]["model"] == fixtures.PRO
    assert result.error is None
    assert result.stop_reason == "end_turn"
    assert result.final_message == "qualified K3 answer"
    assert result.usage.input_tokens == 4533
    assert result.usage.cache_creation_tokens == 0
    assert result.usage.cache_read_tokens == 4352
    assert result.usage.output_tokens == 42
    # (181 x 2.00 + 4352 x 0.20 + 42 x 10.00) / 1e6
    assert result.usage.cost_usd == pytest.approx(0.0016524)
    assert result.usage.cost_bounds is None
    assert result.completeness.token_usage_complete
    assert result.completeness.cost_complete
    assert result.completeness.terminal_complete
    assert result.pricing_evidence.complete
    assert len(result.pricing_evidence.segments) == 1
    segment = result.pricing_evidence.segments[0]
    assert (
        segment.input_tokens, segment.output_tokens,
        segment.cache_creation_tokens, segment.cache_read_tokens,
    ) == (4533, 42, 0, 4352)
    assert segment.cost_usd == pytest.approx(0.0016524)
    assert result.execution_trace.requested_model == fixtures.PRO
    assert result.execution_trace.served_model == fixtures.PRO


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
