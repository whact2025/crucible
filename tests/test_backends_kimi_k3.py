# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free K3-series qualification, including the actual isolated SDK worker."""

from __future__ import annotations

import http.server
import copy
import json
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from assay import BackendPolicy

from crucible.backends import kimi as parent
from crucible.backends.kimi import KimiBackend
from crucible.tests import test_backends_kimi as fixtures

worker = fixtures.worker_mod
PRO = "kimi-k3.1-pro"
# A current Moonshot id that the dedicated route does not admit.
HIGHSPEED = "kimi-k3.1-pro-highspeed"


def test_explicit_model_effort_round_trips_v7():
    model, effort = PRO, "max"
    request = parent._WorkerRequest(
        system="system", user="user", model=model, reasoning_effort=effort,
        base_url=fixtures._BASE_URL, auth_token=fixtures._TOKEN,
    )
    payload = parent._subprocess_request_wire(request, Path("/tmp/private"))
    decoded = worker._Request.from_wire(payload)
    assert decoded.model == model
    assert decoded.reasoning_effort == effort
    assert payload["protocol"] == worker._PROTOCOL == "crucible.kimi-worker.v7"


@pytest.mark.parametrize("override", [
    {"protocol": "crucible.kimi-worker.v6"},
    # Retired route ids: no alias survives the move to K3.1 Pro.
    {"model": "kimi-k3"},
    {"model": "kimi-k2.7-code-highspeed", "reasoning_effort": None},
    # A current id outside the admitted set.
    {"model": HIGHSPEED},
    {"model": "kimi-k3.1-pro-preview"},
    {"reasoning_effort": None}, {"reasoning_effort": "high"},
    {"reasoning_effort": "low"}, {"model": None},
    {"prompt_cache_options": {"mode": "implicit", "ttl": "1h"}},
])
def test_worker_rejects_unqualified_model_effort_or_old_protocol(override):
    payload = fixtures._worker_request_payload(model=PRO, reasoning_effort="max")
    payload.update(override)
    with pytest.raises(worker.WorkerQualificationError):
        worker._Request.from_wire(payload)


@pytest.mark.parametrize("missing", ["model", "reasoning_effort"])
def test_worker_requires_explicit_v7_identity_fields(missing):
    payload = fixtures._worker_request_payload()
    del payload[missing]
    with pytest.raises(worker.WorkerQualificationError):
        worker._Request.from_wire(payload)


@pytest.mark.parametrize("module", [parent, worker], ids=["parent", "worker"])
def test_sdk_config_and_qualified_generation_are_model_bound(monkeypatch, module):
    model, effort, context = PRO, "max", 1_048_576
    session = fixtures._FakeSession([])
    installs = []
    monkeypatch.setattr(
        module, "_install_moonshot_stream_compat",
        lambda provider, **kwargs: installs.append(kwargs),
    )
    module._qualify_moonshot_provider(
        session, max_response_tokens=32_768, model=model, reasoning_effort=effort,
    )
    options = session.provider._generation_kwargs
    assert installs == [{"model": model, "cost_bounds": None}]
    # The K3-series request contract of the admitted id.
    assert options == {
        "prompt_cache_key": session.id,
        "max_completion_tokens": 32_768,
        "reasoning_effort": "max",
        "extra_body": {
            "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
        },
    }
    request_type = parent._WorkerRequest if module is parent else worker._Request
    args = {} if module is parent else {"root": Path("/tmp/private")}
    request = request_type(
        system="s", user="u", model=model, reasoning_effort=effort, **args,
    )
    bindings = SimpleNamespace(Config=SimpleNamespace(model_validate=lambda value: value))
    build = parent._build_sdk_config if module is parent else worker._build_config
    config = build(bindings, request)
    assert config["default_model"] == model
    assert config["models"] == {model: {
        "provider": module._PROVIDER_NAME, "model": model,
        "max_context_size": context, "capabilities": [],
    }}


@pytest.mark.parametrize("effort", ["low", "high", "", "MAX", True])
def test_parent_rejects_unqualified_k3_effort(effort):
    with pytest.raises(ValueError, match="model/effort"):
        KimiBackend(model=PRO, effort=effort)


def test_k3_context_bound_is_explicit_without_widening_default_response_cap():
    model = PRO
    backend = KimiBackend(model=model)
    try:
        assert backend._max_context_size == 1_048_576
        assert backend._max_response_tokens == 32_768
        assert backend.effort == "max"
    finally:
        backend.close()
    payload = fixtures._worker_request_payload(
        model=model, reasoning_effort="max", max_response_tokens=1_048_576,
    )
    assert worker._Request.from_wire(payload).max_response_tokens == 1_048_576
    with pytest.raises(ValueError, match="1048576"):
        KimiBackend(model=model, max_response_tokens=1_048_577)
    with pytest.raises(worker.WorkerQualificationError, match="response_tokens"):
        worker._Request.from_wire({**payload, "max_response_tokens": 1_048_577})


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("drift", ["model", "effort", "cache-ttl", "temperature"])
def test_real_native_generator_rejects_post_qualification_request_drift(drift):
    # The local, unbound port must never be contacted: rejection precedes HTTP.
    script = '''
import asyncio, sys
from kosong.chat_provider.kimi import Kimi
from crucible_kimi_worker import __main__ as worker
async def main():
    provider = Kimi(model="kimi-k3.1-pro", api_key="not-a-real-token", base_url="http://127.0.0.1:1/v1")
    provider._generation_kwargs = {
        "max_completion_tokens": 32768,
        "reasoning_effort": "max",
        "extra_body": {"prompt_cache_options": {"mode": "implicit", "ttl": "5m"}},
    }
    worker._install_moonshot_stream_compat(provider, model="kimi-k3.1-pro")
    if sys.argv[1] == "model":
        provider.model = "kimi-k3.1-pro-highspeed"
    elif sys.argv[1] == "effort":
        provider._generation_kwargs["reasoning_effort"] = "high"
    elif sys.argv[1] == "cache-ttl":
        provider._generation_kwargs["extra_body"]["prompt_cache_options"]["ttl"] = "1h"
    else:
        provider._generation_kwargs["temperature"] = 1
    try:
        await provider.generate("", [], [])
    except worker.WorkerQualificationError as exc:
        print(str(exc))
    else:
        raise AssertionError("request drift was admitted")
    finally:
        await provider.client.close()
asyncio.run(main())
'''
    completed = subprocess.run(
        [str(parent._DEFAULT_WORKER_PYTHON), "-c", script, drift],
        env=parent._worker_environment(), capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Kimi provider changed" in completed.stdout


@pytest.mark.parametrize("served", [HIGHSPEED, "kimi-k3"])
def test_parent_rejects_a_terminal_that_names_another_kimi_model(monkeypatch, served):
    requested = PRO
    monkeypatch.setenv("KIMI_BASE_URL", fixtures._BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", fixtures._TOKEN)
    captured = []

    def run(request, *args, **kwargs):
        captured.append(request)
        return fixtures._success_outcome(configured_model=requested, served_model=served)

    backend = KimiBackend(model=requested, _worker_runner=run)
    try:
        result = backend.invoke(fixtures._invocation())
    finally:
        backend.close()
    assert len(captured) == 1
    assert captured[0].model == requested
    assert captured[0].reasoning_effort == "max"
    assert result.final_message == ""
    assert result.error is not None
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete
    assert result.execution_trace.served_effort is None


def test_parent_prices_each_checkpoint_with_the_requested_model_rates():
    model, cost = PRO, 0.000292
    # 10 uncached + 30 written + 60 read input tokens and 20 output tokens.
    payload = {
        "protocol": parent._WORKER_PROTOCOL, "type": "usage",
        "input_tokens": 100, "output_tokens": 20,
        "cache_creation_tokens": 30, "cache_read_tokens": 60,
        "usage_valid": True,
    }
    outcome, terminal = parent._consume_worker_line(
        json.dumps(payload).encode(), latest_usage=parent._WorkerOutcome(steps_used=1),
        max_steps=1, model=model,
    )
    assert terminal is None
    assert outcome.input_tokens == 100
    assert outcome.pricing_segments[0].cost_usd == pytest.approx(cost)
    assert outcome.physical_usage_responses[0].reported_cache_write_tokens == 30


def _k3_chunks(*, tool=False, model=PRO):
    delta = {"content": "qualified K3 answer"}
    if tool:
        delta = {"tool_calls": [{
            "index": 0, "id": "call-k3", "type": "function",
            "function": {"name": "echo", "arguments": '{"value":"nonce"}'},
        }]}
    return [
        fixtures._native_chunk(model=model, delta=delta),
        fixtures._native_chunk(model=model, finish_reason="tool_calls" if tool else "stop"),
        {
            "id": "chatcmpl-native-qualification", "object": "chat.completion.chunk",
            "created": 1, "model": model, "choices": [],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 30},
            },
        },
    ]


def _coercible_usage(field, scalar_type):
    # Use coherent arithmetic for every field: e.g. bool prompt/total must
    # fail for its JSON type, not because the other counters happen to disagree.
    usage = copy.deepcopy(_k3_chunks()[-1]["usage"])
    details = usage["prompt_tokens_details"]
    if field == "prompt_tokens":
        usage.update(prompt_tokens=1, completion_tokens=0, total_tokens=1)
        details.update(cached_tokens=0, cache_write_tokens=0)
    elif field == "completion_tokens":
        usage.update(completion_tokens=1, total_tokens=101)
    elif field == "total_tokens":
        usage.update(prompt_tokens=1, completion_tokens=0, total_tokens=1)
        details.update(cached_tokens=0, cache_write_tokens=0)
    else:
        details[field] = 1
    owner = details if field in {"cached_tokens", "cache_write_tokens"} else usage
    owner[field] = scalar_type(owner[field])
    return usage


@pytest.mark.parametrize("module", [parent, worker], ids=["parent", "worker"])
@pytest.mark.parametrize("field", [
    "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens",
])
@pytest.mark.parametrize("scalar_type", [bool, float, str])
def test_raw_k3_billing_counters_reject_coercible_noninteger_json(module, field, scalar_type):
    error = parent.KimiSDKQualificationError if module is parent else worker.WorkerQualificationError
    with pytest.raises(error, match="raw billing counters"):
        module._validate_k3_raw_usage(_coercible_usage(field, scalar_type))


def _invoke_real_worker(
    monkeypatch, chunks, *, tool=False, write_root=None, second_chunks=None,
    diagnostic_cost_upper_bound=False, budget_usd=None, model=PRO,
):
    class Endpoint(fixtures._NativeStreamEndpoint):
        def do_POST(self):  # noqa: N802 - HTTP handler API
            if second_chunks is not None and self.server.requests:
                self.server.chunks = second_chunks
            super().do_POST()

    endpoint = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    endpoint.requests = []
    endpoint.chunks = chunks
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.delenv("CRUCIBLE_KIMI_WORKER_PYTHON", raising=False)
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", fixtures._TOKEN)
    backend = KimiBackend(
        model=model, effort="max", expected_base_url=base_url,
        capability=("workspace_write" if write_root is not None else "read_only")
        if tool else "text",
        worker_source="project_locked",
        diagnostic_cost_upper_bound=diagnostic_cost_upper_bound,
    )
    toolset = fixtures._ToolSet()
    if write_root is not None:
        from crucible.tools import ToolContext
        from crucible.tools.write import build_write_tools

        toolset = build_write_tools(ToolContext(
            workspace_root=write_root, repo_root=write_root,
            write_scope=("output.txt",),
        )).filter_to({"write_source"})
    if tool:
        backend.attach_toolset(toolset)
    try:
        result = backend.invoke(replace(
            fixtures._invocation(policy=BackendPolicy(
                retry_budget=1, timeout_seconds=20,
                on_cancellation="convert_to_backend_error", on_max_turns="raise",
            )), max_turns=2 if second_chunks is not None else 1,
            budget_usd=budget_usd,
        ))
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        thread.join(timeout=2)
    return result, endpoint.requests, toolset


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
def test_real_k3_multi_step_replay_preserves_reasoning_and_each_priced_step(monkeypatch):
    chunks = _k3_chunks(tool=True)
    chunks[0]["choices"][0]["delta"]["reasoning_content"] = "Use the confined tool."
    result, requests, toolset = _invoke_real_worker(
        monkeypatch, chunks, tool=True, second_chunks=_k3_chunks(),
    )
    assert result.error is None
    assert len(requests) == 2
    assert toolset.calls == [("echo", {"value": "nonce"})]
    for request in requests:
        assert request["model"] == PRO
        assert request["reasoning_effort"] == "max"
        assert request["prompt_cache_options"] == {"mode": "implicit", "ttl": "5m"}
        assert "max_tokens" not in request
    assistant = next(item for item in requests[1]["messages"] if item["role"] == "assistant")
    assert assistant["reasoning_content"] == "Use the confined tool."
    assert result.usage.input_tokens == 200
    assert result.usage.cache_creation_tokens == 60
    assert result.usage.cache_read_tokens == 120
    assert result.usage.output_tokens == 40
    assert result.usage.cost_usd == pytest.approx(0.000584)
    assert len(result.pricing_evidence.segments) == 2
    assert result.pricing_evidence.complete
    assert result.completeness.terminal_complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("field", [
    "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens",
])
@pytest.mark.parametrize("scalar_type", [bool, float, str])
def test_real_worker_rejects_all_k3_counter_types_before_sdk_coercion(
    monkeypatch, field, scalar_type,
):
    chunks = _k3_chunks()
    chunks[-1]["usage"] = _coercible_usage(field, scalar_type)
    result, requests, _ = _invoke_real_worker(monkeypatch, chunks)
    assert len(requests) == 1
    assert result.error is not None
    assert result.final_message == ""
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("target", ["output.txt", "outside.txt"])
def test_real_k3_coding_worker_preserves_scoped_mutation_authority(
    monkeypatch, tmp_path, target,
):
    chunks = _k3_chunks(tool=True)
    chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"] = {
        "name": "write_source",
        "arguments": json.dumps({"file_path": target, "content": "bounded K3 write\n"}),
    }
    result, requests, _ = _invoke_real_worker(
        monkeypatch, chunks, tool=True, write_root=tmp_path,
    )
    assert len(requests) == 1
    assert [item["function"]["name"] for item in requests[0]["tools"]] == ["write_source"]
    assert not (tmp_path / "outside.txt").exists()
    if target == "output.txt":
        assert (tmp_path / target).read_text() == "bounded K3 write\n"
    else:
        assert not (tmp_path / "output.txt").exists()
    assert result.error.kind == "max_turns"
    assert result.completeness.cost_complete
    assert result.completeness.trace_complete
    assert not result.completeness.terminal_complete


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("tool", [False, True], ids=["text-terminal", "settled-max-turns"])
def test_real_k3_worker_preserves_exact_request_usage_and_cost(monkeypatch, tool):
    model, cost = PRO, 0.000292
    result, requests, toolset = _invoke_real_worker(
        monkeypatch, _k3_chunks(tool=tool, model=model), tool=tool, model=model,
    )
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == model
    assert request["reasoning_effort"] == "max"
    assert request["max_completion_tokens"] == 32_768
    assert request["prompt_cache_options"] == {"mode": "implicit", "ttl": "5m"}
    assert request["stream_options"] == {"include_usage": True}
    assert isinstance(request["prompt_cache_key"], str) and request["prompt_cache_key"]
    # K3-series rules: the deprecated cap, ``thinking`` and every fixed
    # sampling parameter must be absent from what the pinned SDK sends.
    assert not {
        "max_tokens", "temperature", "top_p", "thinking", "n",
        "presence_penalty", "frequency_penalty",
    } & request.keys()
    assert result.usage.input_tokens == 100
    assert result.usage.cache_creation_tokens == 30
    assert result.usage.cache_read_tokens == 60
    assert result.usage.output_tokens == 20
    assert result.usage.cost_usd == pytest.approx(cost)
    assert result.completeness.token_usage_complete
    assert result.completeness.cost_complete
    assert result.pricing_evidence.complete
    assert result.pricing_evidence.segments[0].cost_usd == pytest.approx(cost)
    assert result.execution_trace.requested_model == result.execution_trace.served_model == model
    assert result.execution_trace.requested_effort == "max"
    assert result.execution_trace.served_effort is None
    assert result.completeness.terminal_complete is not tool
    if tool:
        assert result.error.kind == "max_turns"
        assert toolset.calls == [("echo", {"value": "nonce"})]
    else:
        assert result.error is None
        assert result.final_message == "qualified K3 answer"


@pytest.mark.skipif(not parent._DEFAULT_WORKER_PYTHON.is_file(), reason="locked SDK absent")
@pytest.mark.parametrize("malformation", [
    "missing-write", "missing-read", "missing-details", "cache-overlap", "boolean-write",
    "negative-write", "string-write", "wrong-total", "early-usage", "duplicate-usage",
    "missing-usage", "wrong-model", "conflicting-cache",
])
def test_real_k3_worker_rejects_unproven_final_usage(monkeypatch, malformation):
    chunks = _k3_chunks()
    usage = chunks[-1]["usage"]
    details = usage["prompt_tokens_details"]
    if malformation == "missing-write":
        del details["cache_write_tokens"]
    elif malformation == "missing-read":
        del details["cached_tokens"]
    elif malformation == "missing-details":
        del usage["prompt_tokens_details"]
    elif malformation == "cache-overlap":
        details["cache_write_tokens"] = 41
    elif malformation == "boolean-write":
        details["cache_write_tokens"] = True
    elif malformation == "negative-write":
        details["cache_write_tokens"] = -1
    elif malformation == "string-write":
        details["cache_write_tokens"] = "30"
    elif malformation == "wrong-total":
        usage["total_tokens"] = 121
    elif malformation == "early-usage":
        chunks.insert(0, chunks.pop())
    elif malformation == "duplicate-usage":
        chunks.append(chunks[-1])
    elif malformation == "missing-usage":
        chunks.pop()
    elif malformation == "wrong-model":
        for chunk in chunks:
            chunk["model"] = HIGHSPEED
    elif malformation == "conflicting-cache":
        usage["cached_tokens"] = 61
    result, requests, _ = _invoke_real_worker(monkeypatch, chunks)
    assert len(requests) == 1
    assert result.error is not None
    assert result.final_message == ""
    assert not result.completeness.cost_complete
    assert not result.completeness.terminal_complete
    assert not result.pricing_evidence.complete
