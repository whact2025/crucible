# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.backends.openai_compat.OpenAIBackend``.

OpenAIBackend wraps an OpenAI-compatible HTTP endpoint via the
``openai`` SDK — local inference servers (vLLM, llama.cpp, …) AND
hosted gateways like the NVIDIA Inference Hub. These tests stub the
openai client so no live HTTP calls go out; they exercise the
OpenAIBackend → InvocationResult translation path, the retry /
classification logic, the multi-turn tool-use loop, and ``close``.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from assay import (
    BackendPolicy,
    ConversationTransportRequest,
    Invocation,
    MaxTurnsExceeded,
)
from crucible.backends.openai_compat import OpenAIBackend
from crucible.backends.pricing import PricingConfigurationError
from crucible.tools import ToolContext, ToolMutationIndeterminate
from crucible.tools.profiles import build_exact_create_recovery_profile


@pytest.fixture(autouse=True)
def _isolate_inference_gateway_env(monkeypatch, tmp_path):
    """Host config must not change local-default tests.

    Test-only local model ids are explicitly zero-priced; production local
    routes have the same requirement.
    """
    monkeypatch.delenv("INFER_BASE_URL", raising=False)
    monkeypatch.delenv("INFER_AUTH_TOKEN", raising=False)
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "\n".join(
            f"{model}:\n  input_per_mtok: 0\n  output_per_mtok: 0"
            for model in (
                "x",
                "kwarg-model",
                "env-model",
                "qwen2.5-coder-32b",
                "nvidia/zai-org/glm-5.3",
                "nvidia/moonshotai/kimi-k3.1-pro",
            )
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_PRICING_CONFIG", str(pricing))


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_openai_compat_rejects_conversation_before_transport(monkeypatch):
    backend = OpenAIBackend(model="x")
    monkeypatch.setattr(
        backend,
        "_get_client",
        lambda: (_ for _ in ()).throw(AssertionError("transport started")),
    )
    invocation = Invocation(
        system="sys",
        user="correction",
        tools=None,
        max_turns=1,
        budget_usd=None,
        conversation=ConversationTransportRequest(
            mode="workflow_bound",
            action="reuse",
            opaque_id="provider-thread",
            identity_fingerprint="sha256:" + "a" * 64,
            operation_digest="sha256:" + "b" * 64,
        ),
    )

    with pytest.raises(NotImplementedError, match="does not support"):
        backend.invoke(invocation)


def test_openai_backend_requires_model_explicitly_or_via_env(monkeypatch):
    """No default model — caller must pass one or set CRUCIBLE_LOCAL_MODEL."""
    monkeypatch.delenv("CRUCIBLE_LOCAL_MODEL", raising=False)
    with pytest.raises(ValueError, match="requires a model id"):
        OpenAIBackend()


def test_openai_backend_picks_up_model_from_env(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_LOCAL_MODEL", "qwen2.5-coder-32b")
    b = OpenAIBackend()
    assert b.model == "qwen2.5-coder-32b"


def test_openai_backend_kwarg_beats_env(monkeypatch):
    monkeypatch.setenv("CRUCIBLE_LOCAL_MODEL", "env-model")
    b = OpenAIBackend(model="kwarg-model")
    assert b.model == "kwarg-model"


@pytest.mark.parametrize("value", ["", " x", "x ", True, 7])
def test_openai_backend_rejects_invalid_expected_served_model(value):
    with pytest.raises(ValueError, match="expected_served_model"):
        OpenAIBackend(model="x", expected_served_model=value)


def test_openai_backend_base_url_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_LOCAL_URL", raising=False)
    b = OpenAIBackend(model="x")
    assert b._base_url == "http://localhost:8000/v1"

    monkeypatch.setenv("CRUCIBLE_LOCAL_URL", "http://vllm-server:9000/v1")
    b2 = OpenAIBackend(model="x")
    assert b2._base_url == "http://vllm-server:9000/v1"

    b3 = OpenAIBackend(model="x", base_url="http://kwarg:1/v1")
    assert b3._base_url == "http://kwarg:1/v1"


def test_openai_backend_api_key_defaults_to_not_needed(monkeypatch):
    """Most local servers accept any non-empty string; we default to
    ``not-needed`` so the openai SDK's empty-key refusal doesn't fire."""
    monkeypatch.delenv("CRUCIBLE_LOCAL_API_KEY", raising=False)
    b = OpenAIBackend(model="x")
    assert b._api_key == "not-needed"


# ----------------------------------------------------------------------
# invoke → InvocationResult translation
# ----------------------------------------------------------------------


def test_invoke_rejects_unpriced_model_before_transport():
    backend = OpenAIBackend(model="unpriced-hosted-model")
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: pytest.fail("transport must not run"),
            ),
        ),
    )
    with pytest.raises(PricingConfigurationError, match="unpriced-hosted-model"):
        backend.invoke(Invocation(
            system="s", user="u", tools=None,
            max_turns=1, budget_usd=1.0,
        ))


def _stub_client(response):
    """Build an openai-compat client stub returning ``response`` on
    ``chat.completions.create``."""
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: response,
            ),
        ),
    )


def _stub_response(
    text: str | None,
    *,
    in_tokens: int = 10,
    out_tokens: int = 20,
    finish_reason: str = "stop",
    tool_calls=None,
    reasoning_content: str | None = None,
    provider_specific_fields=None,
    model: str | None = "x",
    request_id: str | None = "request-1",
):
    return SimpleNamespace(
        id=request_id,
        model=model,
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=text,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                provider_specific_fields=provider_specific_fields,
            ),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=in_tokens, completion_tokens=out_tokens,
        ),
    )


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _policy(retry_budget: int) -> BackendPolicy:
    return BackendPolicy(
        retry_budget=retry_budget,
        timeout_seconds=1.0,
        on_cancellation="raise",
        on_max_turns="raise",
    )


def test_invoke_translates_response_to_invocation_result(monkeypatch):
    b = OpenAIBackend(model="x")
    b._client = _stub_client(_stub_response("model-says-hi"))

    inv = Invocation(
        system="s", user="u", tools=None,
        max_turns=1, budget_usd=None,
    )
    result = b.invoke(inv)

    assert result.final_message == "model-says-hi"
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.usage.cost_usd == 0.0   # local servers don't bill per-token
    assert result.error is None


def test_exact_served_model_contract_accepts_matching_single_shot():
    backend = OpenAIBackend(model="x", expected_served_model="x")
    backend._client = _stub_client(_stub_response("model-says-hi", model="x"))

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None,
        max_turns=1, budget_usd=None,
    ))

    assert result.stop_reason == "end_turn"
    assert result.error is None
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == "x"


@pytest.mark.parametrize(
    ("served_model", "stop_reason"),
    [("served-x", "model_mismatch"), (None, "missing_model_identity")],
)
def test_exact_served_model_contract_rejects_unproved_single_shot(
    served_model,
    stop_reason,
):
    backend = OpenAIBackend(model="x", expected_served_model="x")
    backend._client = _stub_client(_stub_response(
        "must-not-be-admitted",
        model=served_model,
    ))

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None,
        max_turns=1, budget_usd=None,
    ))

    assert result.stop_reason == stop_reason
    assert result.final_message == ""
    assert result.error is not None
    assert result.error.kind == "incomplete_evidence"
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert all(
        message.content != "must-not-be-admitted"
        for message in result.transcript
    )
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == served_model


def test_late_single_shot_response_retains_identity_but_not_content(
    monkeypatch,
):
    clock = {"now": 100.0}
    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.monotonic",
        lambda: clock["now"],
    )
    backend = OpenAIBackend(model="x", expected_served_model="x")

    def _create(**_kwargs):
        clock["now"] = 102.0
        return _stub_response(
            "must-not-cross-deadline",
            model="substituted-model",
            request_id="late-request",
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    policy = BackendPolicy(
        retry_budget=1,
        timeout_seconds=1.0,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=policy,
    ))

    assert result.stop_reason == "timeout"
    assert result.final_message == ""
    assert [message.content for message in result.transcript] == ["u"]
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == "substituted-model"
    assert result.execution_trace.provider_request_id == "late-request"


def test_invoke_passes_vendor_neutral_schema_to_native_strict_json():
    captured = {}
    response = _stub_response('{"ok":true}')
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: captured.update(kwargs) or response,
            ),
        ),
    )
    backend = OpenAIBackend(model="x")
    backend._client = client
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        response_schema=schema,
    ))

    assert result.error is None
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "assay_structured_response",
            "strict": True,
            "schema": schema,
        },
    }


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("nvidia/zai-org/glm-5.3", "https://inference-api.nvidia.com"),
        ("nvidia/zai-org/glm-5.3", "https://inference-api-dev.nvidia.com/v1"),
        ("nvidia/zai-org/glm-5.3", "https://other.example/v1"),
        ("x", "https://inference-api.nvidia.com/v1"),
    ],
)
def test_unique_items_schema_reaches_every_route_unchanged(model, base_url):
    """No route, the hub's GLM route included, gets a weakened schema."""

    backend = OpenAIBackend(model=model, base_url=base_url)
    schema = {
        "type": "array",
        "items": {"type": "array", "uniqueItems": True},
        "uniqueItems": True,
    }
    original = copy.deepcopy(schema)

    kwargs = backend._structured_output_kwargs(schema)

    assert kwargs["response_format"]["json_schema"]["schema"] == original
    assert schema == original


@pytest.mark.parametrize(
    (
        "model",
        "base_url",
        "response_schema",
        "reasoning_content",
        "finish_reason",
    ),
    [
        # The route and the shape that used to be taken as a structured answer.
        pytest.param(
            "nvidia/zai-org/glm-5.3",
            "https://inference-api.nvidia.com",
            {"type": "object"},
            '{"private":"reasoning"}',
            "stop",
            id="hub-glm-schema-json-object",
        ),
        pytest.param(
            "nvidia/zai-org/glm-5.3",
            "https://inference-api.nvidia.com",
            {"type": "object"},
            "The caller wants one object. I will answer now.",
            "length",
            id="hub-glm-schema-prose-length",
        ),
        pytest.param(
            "nvidia/zai-org/glm-5.3",
            "https://inference-api.nvidia.com",
            None,
            '{"private":"reasoning"}',
            "stop",
            id="hub-glm-no-schema",
        ),
        pytest.param(
            "x",
            "http://localhost:8000/v1",
            {"type": "object"},
            '{"private":"reasoning"}',
            "stop",
            id="local-schema",
        ),
    ],
)
def test_reasoning_content_is_never_a_completion_answer(
    model, base_url, response_schema, reasoning_content, finish_reason,
):
    """A content-free reply stays empty whatever its reasoning field holds."""

    backend = OpenAIBackend(model=model, base_url=base_url)
    backend._client = _stub_client(_stub_response(
        None,
        reasoning_content=reasoning_content,
        finish_reason=finish_reason,
    ))

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        response_schema=response_schema,
        policy=_policy(1),
    ))

    assert result.final_message == ""
    assert result.error is not None
    if finish_reason == "length":
        assert result.stop_reason == "max_turns"
        assert isinstance(result.error, MaxTurnsExceeded)
    else:
        assert result.error.retriable is True
        assert "empty completion" in result.error.message
    assert [message.content for message in result.transcript] == ["u", ""]


def test_invoke_finish_reason_length_classifies_as_max_turns(monkeypatch):
    b = OpenAIBackend(model="x")
    b._client = _stub_client(_stub_response(
        "...truncated", finish_reason="length",
    ))

    inv = Invocation(
        system="s", user="u", tools=None,
        max_turns=1, budget_usd=None,
    )
    result = b.invoke(inv)

    assert result.stop_reason == "max_turns"
    assert result.error is not None
    assert result.error.kind == "max_turns"


def test_invoke_empty_content_becomes_retriable_transport_error():
    """A clean finish_reason=stop with empty visible content (a reasoning
    model can burn its whole budget on the hidden think block) is a
    content-free stall. Single-shot judge/review/plan callers would
    otherwise hit a parse_failure on the empty text and fail the gate;
    the guard surfaces a retriable TransportError so the retry budget
    gets a fresh attempt instead."""
    b = OpenAIBackend(model="x")
    b._client = _stub_client(_stub_response("", finish_reason="stop"))

    inv = Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=_policy(1),
    )
    result = b.invoke(inv)

    assert result.error is not None
    assert result.error.kind == "transport"
    assert result.error.retriable is True
    assert result.final_message == ""
    assert "empty completion" in result.error.message


def test_invoke_retries_retriable_result_error(monkeypatch):
    responses = iter([
        _stub_response("", finish_reason="stop"),
        _stub_response("recovered"),
    ])
    calls = []

    def _create(**_kwargs):
        calls.append(1)
        return next(responses)

    monkeypatch.setattr("crucible.backends.openai_compat.time.sleep", lambda _: None)
    backend = OpenAIBackend(model="x")
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=_policy(2),
    ))

    assert calls == [1, 1]
    assert result.error is None
    assert result.final_message == "recovered"
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.completeness.trace_complete is True
    assert result.execution_trace is not None
    assert "retry" in {
        event.kind for event in result.execution_trace.events
    }


# ----------------------------------------------------------------------
# Error classification
# ----------------------------------------------------------------------


def test_invoke_classifies_connection_error_as_retriable_transport(
    monkeypatch,
):
    """An unreachable server → TransportError with retriable=True."""
    b = OpenAIBackend(model="x")

    def _raise_conn(**_):
        raise ConnectionError("connection refused")

    b._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_raise_conn),
        ),
    )

    inv = Invocation(
        system="s", user="u", tools=None,
        max_turns=1, budget_usd=None,
    )
    # Policy retry_budget defaults to 3; ConnectionError trips retry
    # then bails. We only care that the eventual result classifies it.
    result = b.invoke(inv)
    assert result.error is not None
    assert result.error.kind == "transport"
    assert "ConnectionError" in result.error.message


def test_classify_internal_server_error_subclass_by_status_code():
    class APIStatusError(Exception):
        pass

    class InternalServerError(APIStatusError):
        status_code = 500

    error = OpenAIBackend(model="x")._classify(
        InternalServerError("provider failed"),
    )

    assert error is not None
    assert error.kind == "transport"
    assert error.http_status == 500
    assert error.retriable is True


def test_classify_future_status_exception_by_response_status():
    class FutureGatewayError(Exception):
        response = SimpleNamespace(status_code=400)

    error = OpenAIBackend(model="x")._classify(
        FutureGatewayError("invalid request"),
    )

    assert error is not None
    assert error.kind == "transport"
    assert error.http_status == 400
    assert error.retriable is False


def test_openai_sdk_retries_are_disabled_and_crucible_owns_attempt_count(
    monkeypatch,
):
    class APIStatusError(Exception):
        pass

    class InternalServerError(APIStatusError):
        status_code = 500

    constructed_retries = []
    physical_attempts = []

    class _FakeCompletions:
        def __init__(self, max_retries):
            self.max_retries = max_retries

        def create(self, **_kwargs):
            # Model the SDK's own retry multiplier.  With max_retries=0 each
            # Crucible attempt results in exactly one physical HTTP request.
            for _ in range(self.max_retries + 1):
                physical_attempts.append(1)
            raise InternalServerError("upstream 500")

    class _FakeOpenAI:
        def __init__(self, *, base_url, api_key, max_retries):
            del base_url, api_key
            constructed_retries.append(max_retries)
            self.chat = SimpleNamespace(
                completions=_FakeCompletions(max_retries),
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setattr("crucible.backends.openai_compat.time.sleep", lambda _: None)
    backend = OpenAIBackend(model="x")
    policy = BackendPolicy(
        retry_budget=3,
        timeout_seconds=1.0,
        on_cancellation="raise",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=policy,
    ))

    assert constructed_retries == [0]
    assert len(physical_attempts) == policy.retry_budget
    assert result.error is not None
    assert result.error.http_status == 500
    assert result.error.retriable is True


# ----------------------------------------------------------------------
# Multi-turn tool-use loop
# ----------------------------------------------------------------------


class _StubToolSet:
    """Minimal ToolSet stand-in: records dispatches, returns canned text."""

    domain = "test"
    write_names: set[str] = set()

    @property
    def read_only_names(self):
        return {
            item["function"]["name"] for item in self.to_openai_tools()
        } - self.write_names

    def __init__(self, results=None):
        self._results = results or {}
        self.dispatched = []

    def to_openai_tools(self):
        return [
            {"type": "function", "function": {"name": "lookup", "parameters": {}}},
        ]

    def to_anthropic_tools(self):
        return [
            {
                "name": item["function"]["name"],
                "description": item["function"].get("description", ""),
                "input_schema": item["function"].get("parameters", {}),
            }
            for item in self.to_openai_tools()
        ]

    def dispatch(self, name, args):
        self.dispatched.append((name, args))
        return self._results.get(name, f"result-of-{name}")


def test_exact_served_model_mismatch_prevents_tool_dispatch():
    toolset = _StubToolSet()
    backend = OpenAIBackend(
        model="x",
        expected_served_model="x",
        capability="read_only",
    )
    backend.attach_toolset(toolset)
    backend._client = _stub_client(_stub_response(
        "",
        model="substituted-model",
        tool_calls=[_tool_call("c1", "lookup", "{}")],
    ))

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None,
        max_turns=2, budget_usd=None,
    ))

    assert result.stop_reason == "model_mismatch"
    assert result.error is not None
    assert result.error.kind == "incomplete_evidence"
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert toolset.dispatched == []


def test_late_tool_response_retains_identity_without_dispatch_or_content(
    monkeypatch,
):
    clock = {"now": 200.0}
    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.monotonic",
        lambda: clock["now"],
    )
    toolset = _StubToolSet()
    backend = OpenAIBackend(
        model="x",
        expected_served_model="x",
        capability="read_only",
    )
    backend.attach_toolset(toolset)

    def _create(**_kwargs):
        clock["now"] = 202.0
        return _stub_response(
            "must-not-cross-deadline",
            model="x",
            request_id="late-tool-request",
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    policy = BackendPolicy(
        retry_budget=1,
        timeout_seconds=1.0,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=2,
        budget_usd=None,
        policy=policy,
    ))

    assert result.stop_reason == "timeout"
    assert result.final_message == ""
    assert [message.content for message in result.transcript] == ["u"]
    assert toolset.dispatched == []
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == "x"
    assert result.execution_trace.provider_request_id == "late-tool-request"


def test_exact_served_model_contract_covers_forced_finalization():
    toolset = _StubToolSet()
    backend = OpenAIBackend(
        model="x",
        expected_served_model="x",
        capability="read_only",
    )
    backend.attach_toolset(toolset)
    responses = iter([
        _stub_response(
            "",
            model="x",
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        ),
        _stub_response("must-not-be-admitted", model="substituted-model"),
    ])
    requests = 0

    def _create(**_kwargs):
        nonlocal requests
        requests += 1
        return next(responses)

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None,
        max_turns=1, budget_usd=None,
    ))

    assert requests == 2
    assert toolset.dispatched == [("lookup", {})]
    assert result.stop_reason == "model_mismatch"
    assert result.final_message == ""
    assert result.error is not None
    assert result.error.kind == "incomplete_evidence"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 40
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == "substituted-model"


def test_late_forced_final_response_is_accounted_without_content(
    monkeypatch,
):
    clock = {"now": 300.0}
    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.monotonic",
        lambda: clock["now"],
    )
    toolset = _StubToolSet()
    backend = OpenAIBackend(
        model="x",
        expected_served_model="x",
        capability="read_only",
    )
    backend.attach_toolset(toolset)
    responses = iter([
        _stub_response(
            "",
            model="x",
            request_id="tool-request",
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        ),
        _stub_response(
            "must-not-cross-deadline",
            model="x",
            request_id="late-final-request",
            in_tokens=30,
            out_tokens=40,
        ),
    ])
    calls = 0

    def _create(**_kwargs):
        nonlocal calls
        calls += 1
        response = next(responses)
        if calls == 2:
            clock["now"] = 302.0
        return response

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    policy = BackendPolicy(
        retry_budget=1,
        timeout_seconds=1.0,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=policy,
    ))

    assert calls == 2
    assert toolset.dispatched == [("lookup", {})]
    assert result.stop_reason == "timeout"
    assert result.final_message == ""
    assert all(
        message.content != "must-not-cross-deadline"
        for message in result.transcript
    )
    assert result.usage.input_tokens == 40
    assert result.usage.output_tokens == 60
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == "x"
    assert result.execution_trace.provider_request_id == "late-final-request"


def test_forced_final_transport_failure_poisons_aggregate_accounting(
    monkeypatch,
):
    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.sleep",
        lambda _delay: None,
    )
    toolset = _StubToolSet()
    backend = OpenAIBackend(
        model="x",
        expected_served_model="x",
        capability="read_only",
    )
    backend.attach_toolset(toolset)
    calls = 0

    def _create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stub_response(
                "",
                model="x",
                request_id="tool-request",
                tool_calls=[_tool_call("c1", "lookup", "{}")],
            )
        raise ConnectionError("forced final response envelope was lost")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=_policy(1),
    ))

    assert calls == 2
    assert result.stop_reason == "max_turns"
    assert isinstance(result.error, MaxTurnsExceeded)
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.execution_trace is not None
    assert result.execution_trace.served_model == "x"
    assert result.execution_trace.provider_request_id == "tool-request"


def test_unclassified_forced_final_fault_is_not_treated_as_budget_skip():
    toolset = _StubToolSet()
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(toolset)
    calls = 0

    def _create(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _stub_response(
                "",
                tool_calls=[_tool_call("c1", "lookup", "{}")],
            )
        raise RuntimeError("unknown forced-final SDK contract fault")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    with pytest.raises(RuntimeError, match="unknown forced-final"):
        backend.invoke(Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=None,
            policy=_policy(1),
        ))

    assert calls == 2


def test_attach_toolset_stores_toolset_and_invoke_runs_tool_loop():
    """attach_toolset stores the toolset; invoke() then drives the loop."""
    ts = _StubToolSet(results={"lookup": "the-answer-is-42"})
    b = OpenAIBackend(model="x", capability="read_only")
    b.attach_toolset(ts)
    assert b._toolset is ts

    # Turn 0: model calls a tool. Turn 1: model produces final text.
    responses = iter([
        _stub_response(
            "", tool_calls=[_tool_call("c1", "lookup", '{"q": "life"}')],
        ),
        _stub_response("done: the-answer-is-42"),
    ])
    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        return next(responses)

    b._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    inv = Invocation(
        system="s", user="u", tools=None,
        max_turns=5, budget_usd=None,
    )
    result = b.invoke(inv)

    assert result.stop_reason == "end_turn"
    assert result.final_message == "done: the-answer-is-42"
    assert ts.dispatched == [("lookup", {"q": "life"})]
    assert all("tool_choice" not in call for call in calls)
    assert result.execution_trace is not None
    assert result.execution_trace.tool_authority_digest == (
        b._tool_authority_digest(ts, context_budget=None)  # noqa: SLF001
    )


def test_tool_mutation_indeterminate_stops_without_provider_retry():
    class _IndeterminateToolSet(_StubToolSet):
        def dispatch(self, name, args):
            self.dispatched.append((name, args))
            raise ToolMutationIndeterminate("delete settlement is unknown")

    toolset = _IndeterminateToolSet()
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(toolset)
    requests = 0

    def _create(**_kwargs):
        nonlocal requests
        requests += 1
        if requests > 1:
            pytest.fail("terminal mutation must prevent another request")
        return _stub_response(
            "",
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=3,
        budget_usd=None,
        policy=_policy(retry_budget=2),
    ))

    assert requests == 1
    assert toolset.dispatched == [("lookup", {})]
    assert result.stop_reason == "tool_mutation_indeterminate"
    assert result.error is not None
    assert result.error.kind == "transport"
    assert result.error.retriable is False
    assert result.execution_trace is not None
    assert not any(
        event.kind in {"retry", "limit", "tool_result"}
        for event in result.execution_trace.events
    )


def test_tool_loop_prices_each_subthreshold_response_independently():
    """Aggregate invocation tokens must not trigger a per-request tier."""

    toolset = _StubToolSet(results={"lookup": "evidence"})
    backend = OpenAIBackend(model="gpt-5.6-sol", capability="read_only")
    backend.attach_toolset(toolset)
    responses = iter([
        _stub_response(
            "",
            in_tokens=200_000,
            out_tokens=1_000,
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        ),
        _stub_response(
            "done",
            in_tokens=200_000,
            out_tokens=1_000,
        ),
    ])
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: next(responses)),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=2,
        budget_usd=None,
    ))

    assert result.error is None
    assert result.usage.input_tokens == 400_000
    assert result.usage.output_tokens == 2_000
    assert result.usage.cost_usd == pytest.approx(1.64)


def test_tool_loop_prices_one_long_context_response_at_long_tier():
    """A provider response above 272K input uses the full-request tier."""

    backend = OpenAIBackend(model="gpt-5.6-sol", capability="read_only")
    backend.attach_toolset(_StubToolSet())
    backend._client = _stub_client(_stub_response(
        "done",
        in_tokens=300_001,
        out_tokens=1_000,
    ))

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
    ))

    assert result.error is None
    assert result.usage.input_tokens == 300_001
    assert result.usage.output_tokens == 1_000
    assert result.usage.cost_usd == pytest.approx(2.430008)


@pytest.mark.parametrize(
    "reasoning_fields",
    [
        {"reasoning_content": "opaque-thinking-state"},
        {
            "provider_specific_fields": {
                "reasoning_content": "opaque-thinking-state",
            },
        },
    ],
)
def test_tool_loop_replays_opaque_reasoning_content_without_transcript_leak(
    reasoning_fields,
):
    """Reasoning providers require their assistant state on the next turn."""

    toolset = _StubToolSet(results={"lookup": "evidence"})
    backend = OpenAIBackend(
        model="nvidia/moonshotai/kimi-k3.1-pro",
        capability="read_only",
    )
    backend.attach_toolset(toolset)
    responses = iter([
        _stub_response(
            "",
            tool_calls=[_tool_call("c1", "lookup", "{}")],
            **reasoning_fields,
        ),
        _stub_response("final"),
    ])
    calls = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(responses)

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=2,
        budget_usd=None,
    ))

    assistant_turn = calls[1]["messages"][2]
    assert assistant_turn["reasoning_content"] == "opaque-thinking-state"
    assert assistant_turn["tool_calls"][0]["id"] == "c1"
    assert result.final_message == "final"
    assert all(
        "opaque-thinking-state" not in message.content
        for message in result.transcript
    )


def test_kimi_k31_pro_write_turn_requires_a_tool_call():
    """Write-only turns send ``required``; no model-specific exception."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {"name": "apply_patch", "parameters": {}},
            }]

    toolset = _MutatingToolSet(results={"apply_patch": "ok"})
    backend = OpenAIBackend(
        model="nvidia/moonshotai/kimi-k3.1-pro",
        capability="workspace_write",
    )
    backend.attach_toolset(toolset)
    calls = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if "tools" in kwargs:
            return _stub_response(
                "",
                tool_calls=[_tool_call("write-1", "apply_patch", "{}")],
                reasoning_content="opaque-thinking-state",
            )
        return _stub_response("done")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
    ))

    tool_calls = [call for call in calls if "tools" in call]
    assert [call.get("tool_choice") for call in tool_calls] == ["required"]
    assert toolset.dispatched == [("apply_patch", {})]
    echoed = [
        message for message in calls[-1]["messages"]
        if message.get("role") == "assistant"
    ]
    assert [message.get("reasoning_content") for message in echoed] == [
        "opaque-thinking-state",
    ]
    assert echoed[0]["tool_calls"][0]["id"] == "write-1"
    assert result.final_message == "done"
    assert all(
        "opaque-thinking-state" not in message.content
        for message in result.transcript
    )


def test_tool_loop_timeout_preserves_accumulated_usage():
    """A timeout on round N of the multi-turn tool loop must report the
    tokens spent on rounds 0..N-1, not zeros.

    Without the in-loop catch, the exception unwinds to ``invoke()``,
    which builds a hardcoded ``Usage()`` — dropping every completed-turn
    token. That silently undercounts cost on the most expensive failure
    mode (a long implementer/defender tool loop that times out
    mid-flight). Mirrors ClaudeBackend's ``_partial_usage`` fix.
    """
    ts = _StubToolSet(results={"lookup": "partial"})
    b = OpenAIBackend(model="x", capability="read_only")
    b.attach_toolset(ts)

    # Turn 0: a real tool-call turn carrying usage. Turn 1: timeout.
    def _raise_timeout(**_):
        raise TimeoutError("server took too long")

    call_count = {"n": 0}

    def _create(**kw):
        if call_count["n"] == 0:
            call_count["n"] += 1
            return _stub_response(
                "", in_tokens=1000, out_tokens=200,
                tool_calls=[_tool_call("c1", "lookup", "{}")],
            )
        return _raise_timeout()

    b._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    # retry_budget=1 so we see the first failed attempt's result directly.
    policy = BackendPolicy(
        retry_budget=1, timeout_seconds=0.1,
        on_cancellation="raise", on_max_turns="raise",
    )
    inv = Invocation(
        system="s", user="u", tools=None,
        max_turns=5, budget_usd=None, policy=policy,
    )
    result = b.invoke(inv)

    assert result.error is not None
    assert result.error.kind == "transport"
    assert result.error.retriable is True
    # Usage from the completed turn 0 survives the round-1 timeout.
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 200


def test_max_turns_forces_no_tools_finalization_turn():
    """When the loop exhausts max_turns while the model is STILL calling
    tools (never stops to answer — the deepseek-v4-pro planner failure
    mode), the backend makes one final completion with tools removed so
    the model must convert its investigation into an answer.

    Here every in-loop turn returns a tool_call (the model never stops),
    max_turns=2 is hit, then the forced turn returns the real answer.
    The detached finalization call must surface as final_message.
    """
    ts = _StubToolSet(results={"lookup": "scratch"})
    b = OpenAIBackend(model="x", capability="read_only")
    b.attach_toolset(ts)

    calls = {"with_tools": 0, "no_tools": 0}

    def _create(**kw):
        if "tools" in kw:
            calls["with_tools"] += 1
            # Always ask for another tool — never stop on its own.
            return _stub_response(
                "", tool_calls=[_tool_call("c1", "lookup", "{}")],
            )
        calls["no_tools"] += 1
        return _stub_response('{"tiers": []}')

    b._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    inv = Invocation(
        system="s", user="u", tools=None, max_turns=2, budget_usd=None,
    )
    result = b.invoke(inv)

    assert result.stop_reason == "max_turns"
    # The in-loop turns each carried tools; the forced finalize did not.
    assert calls["with_tools"] == 2
    assert calls["no_tools"] == 1
    # The forced no-tools turn's answer becomes the final message.
    assert result.final_message == '{"tiers": []}'


def test_max_turns_reserves_final_tool_turn_for_workspace_mutation():
    """Mutation-capable roles get an edit-now turn before finalization.

    A read-only no-tools completion cannot apply the patch an implementer was
    asked to produce.  The final bounded tool turn therefore exposes only the
    toolset's explicitly classified write surface when no prior write occurred.
    """

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch", "delete_source"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "delete_source", "parameters": {}},
                },
            ]

    toolset = _MutatingToolSet(results={
        "lookup": "context",
        "apply_patch": "ok",
        "delete_source": "OK: recursively deleted donor",
    })
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(kwargs)
        if "tools" not in kwargs:
            assert "tool_choice" not in kwargs
            return _stub_response("DONE")
        names = {
            descriptor["function"]["name"] for descriptor in kwargs["tools"]
        }
        if names == {"lookup", "apply_patch", "delete_source"}:
            assert "tool_choice" not in kwargs
            return _stub_response(
                "", tool_calls=[_tool_call("read-1", "lookup", "{}")],
            )
        assert names == {"apply_patch", "delete_source"}
        assert kwargs["tool_choice"] == "required"
        assert "Final implementation turn" in kwargs["messages"][-1]["content"]
        return _stub_response(
            "",
            tool_calls=[
                _tool_call(
                    "write-1",
                    "delete_source",
                    '{"file_path":"donor","recursive":true}',
                ),
            ],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=2, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert result.final_message == "DONE"
    assert toolset.dispatched == [
        ("lookup", {}),
        ("delete_source", {"file_path": "donor", "recursive": True}),
    ]
    assert len(calls) == 3


def test_mutating_tool_loop_enforces_bounded_exploration_phase():
    """After one third of the turns, only scoped source edits remain.

    A successful first edit relaxes ``tool_choice`` so the model may either
    make additional edits or finish, but read tools never reappear during the
    invocation.  This bounds replay of large source-tool results and prevents
    a partial patch from reopening investigation.
    """

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
            ]

    toolset = _MutatingToolSet(
        results={"lookup": "source context", "apply_patch": "OK: applied"},
    )
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        index = len(calls) - 1
        names = {
            descriptor["function"]["name"]
            for descriptor in kwargs["tools"]
        }
        if index < 2:
            assert names == {"lookup", "apply_patch"}
            assert "tool_choice" not in kwargs
            return _stub_response(
                "",
                tool_calls=[_tool_call(f"read-{index}", "lookup", "{}")],
            )
        assert names == {"apply_patch"}
        assert any(
            "Implementation phase" in message.get("content", "")
            for message in kwargs["messages"]
            if message.get("role") == "user"
        )
        if index == 2:
            assert kwargs["tool_choice"] == "required"
            return _stub_response(
                "",
                tool_calls=[_tool_call(
                    "write-1", "apply_patch", '{"patch":"first"}',
                )],
            )
        if index == 3:
            assert "tool_choice" not in kwargs
            return _stub_response(
                "",
                tool_calls=[_tool_call(
                    "write-2", "apply_patch", '{"patch":"second"}',
                )],
            )
        assert index == 4
        assert "tool_choice" not in kwargs
        return _stub_response("DONE")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=6, budget_usd=None,
    ))

    assert result.stop_reason == "end_turn"
    assert result.final_message == "DONE"
    assert toolset.dispatched == [
        ("lookup", {}),
        ("lookup", {}),
        ("apply_patch", {"patch": "first"}),
        ("apply_patch", {"patch": "second"}),
    ]
    assert len(calls) == 5


def test_write_phase_rejects_stale_read_tool_call_before_dispatch():
    """Advertising only writes is backed by a dispatch authorization check."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
            ]

    toolset = _MutatingToolSet(
        results={"lookup": "context", "apply_patch": "OK: applied"},
    )
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        index = len(calls) - 1
        if "tools" not in kwargs:
            return _stub_response("DONE")
        names = {
            item["function"]["name"] for item in kwargs["tools"]
        }
        if index == 0:
            assert names == {"lookup", "apply_patch"}
            return _stub_response(
                "", tool_calls=[_tool_call("read", "lookup", "{}")],
            )
        assert names == {"apply_patch"}
        assert kwargs["tool_choice"] == "required"
        if index == 1:
            # A stale provider response names a tool that is absent from the
            # request catalog. It must receive an error, not a dispatch.
            return _stub_response(
                "", tool_calls=[_tool_call("stale", "lookup", "{}")],
            )
        assert index == 2
        return _stub_response(
            "",
            tool_calls=[_tool_call(
                "write", "apply_patch", '{"patch":"targeted"}',
            )],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=3, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert toolset.dispatched == [
        ("lookup", {}),
        ("apply_patch", {"patch": "targeted"}),
    ]
    assert any(
        message.role == "tool" and "unavailable" in message.content
        for message in result.transcript
    )
    assert len(calls) == 4


def test_write_phase_retries_when_required_tool_choice_is_ignored():
    """Provider prose cannot satisfy an unresolved source-edit obligation."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
            ]

    toolset = _MutatingToolSet(
        results={"lookup": "context", "apply_patch": "OK: applied"},
    )
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        index = len(calls) - 1
        if "tools" not in kwargs:
            return _stub_response("DONE")
        if index < 2:
            return _stub_response(
                "", tool_calls=[_tool_call(f"read-{index}", "lookup", "{}")],
            )
        assert {
            item["function"]["name"] for item in kwargs["tools"]
        } == {"apply_patch"}
        assert kwargs["tool_choice"] == "required"
        if index == 2:
            return _stub_response("I should edit this next")
        assert index == 3
        assert any(
            "still required" in message.get("content", "")
            for message in kwargs["messages"]
            if message.get("role") == "user"
        )
        return _stub_response(
            "",
            tool_calls=[_tool_call(
                "write", "apply_patch", '{"patch":"targeted"}',
            )],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=4, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert toolset.dispatched == [
        ("lookup", {}),
        ("lookup", {}),
        ("apply_patch", {"patch": "targeted"}),
    ]
    assert len(calls) == 5


def test_ignored_required_tool_choice_respects_budget():
    """A prose-only required turn cannot schedule recovery past the cap."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {"name": "apply_patch", "parameters": {}},
            }]

    toolset = _MutatingToolSet()
    backend = OpenAIBackend(model="gpt-5.5", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls = 0

    def _create(**_kwargs):
        nonlocal calls
        calls += 1
        return _stub_response(
            "I did not call the required tool",
            in_tokens=10_000_000,
            out_tokens=10_000_000,
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=1, budget_usd=0.01,
    ))

    assert result.stop_reason == "budget_exhausted"
    assert result.error is not None
    assert result.error.kind == "budget_exhausted"
    assert result.usage.cost_usd > 0.01
    assert calls == 1
    assert toolset.dispatched == []


def test_later_rejected_edit_reopens_required_write_obligation():
    """An earlier successful patch cannot mask a later safety rejection."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
            ]

        def dispatch(self, name, args):
            self.dispatched.append((name, args))
            if name == "lookup":
                return "context"
            if args["patch"] == "rejected":
                return "ERROR patch was rejected by the safety contract"
            return "OK: applied"

    toolset = _MutatingToolSet()
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        index = len(calls) - 1
        if index < 2:
            return _stub_response(
                "", tool_calls=[_tool_call(f"read-{index}", "lookup", "{}")],
            )
        assert {
            item["function"]["name"] for item in kwargs["tools"]
        } == {"apply_patch"}
        if index == 2:
            assert kwargs["tool_choice"] == "required"
            patch = "first"
        elif index == 3:
            assert "tool_choice" not in kwargs
            patch = "rejected"
        elif index == 4:
            assert kwargs["tool_choice"] == "required"
            patch = "corrected"
        else:
            assert index == 5
            assert "tool_choice" not in kwargs
            return _stub_response("DONE")
        return _stub_response(
            "",
            tool_calls=[_tool_call(
                f"write-{index}", "apply_patch", f'{{"patch":"{patch}"}}',
            )],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=6, budget_usd=None,
    ))

    assert result.stop_reason == "end_turn"
    assert result.final_message == "DONE"
    assert toolset.dispatched[-3:] == [
        ("apply_patch", {"patch": "first"}),
        ("apply_patch", {"patch": "rejected"}),
        ("apply_patch", {"patch": "corrected"}),
    ]


def test_rejected_early_length_write_uses_bounded_recovery():
    """A rejected forced write gets one correction before verification."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {"name": "apply_patch", "parameters": {}},
            }]

        def dispatch(self, name, args):
            self.dispatched.append((name, args))
            if len(self.dispatched) == 1:
                return "ERROR patch was rejected by the safety contract"
            return "OK: applied"

    toolset = _MutatingToolSet()
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        index = len(calls) - 1
        if index == 0:
            return _stub_response("partial", finish_reason="length")
        assert kwargs["tool_choice"] == "required"
        patch = "rejected" if index == 1 else "corrected"
        return _stub_response(
            "",
            tool_calls=[_tool_call(
                f"write-{index}", "apply_patch", f'{{"patch":"{patch}"}}',
            )],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=6, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert result.final_message == "partial"
    assert toolset.dispatched == [
        ("apply_patch", {"patch": "rejected"}),
        ("apply_patch", {"patch": "corrected"}),
    ]
    assert len(calls) == 3


def test_final_rejected_source_edit_gets_one_write_only_recovery_turn():
    """A last-turn write safety rejection gets one same-context correction."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch", "delete_source", "write_source"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": name, "parameters": {}},
                }
                for name in (
                    "lookup",
                    "apply_patch",
                    "delete_source",
                    "write_source",
                )
            ]

        def dispatch(self, name, args):
            self.dispatched.append((name, args))
            if name == "write_source" and len(self.dispatched) == 1:
                return "OK: created helper.py"
            if name == "write_source":
                return (
                    "Error: refusing to replace existing file without an "
                    "exact expected_sha256; use apply_patch"
                )
            return "OK: applied targeted patch"

    toolset = _MutatingToolSet()
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        call_index = len(calls) - 1
        if call_index == 0:
            return _stub_response(
                "",
                tool_calls=[
                    _tool_call("create", "write_source", '{"path":"helper.py"}'),
                ],
            )
        if call_index == 1:
            return _stub_response(
                "",
                tool_calls=[
                    _tool_call("replace", "write_source", '{"path":"core.py"}'),
                ],
            )
        assert call_index == 2
        assert {
            item["function"]["name"] for item in kwargs["tools"]
        } == {"apply_patch", "delete_source", "write_source"}
        assert kwargs["tool_choice"] == "required"
        assert "Bounded source-edit recovery" in kwargs["messages"][-1]["content"]
        return _stub_response(
            "",
            tool_calls=[
                _tool_call("recover", "apply_patch", '{"patch":"targeted"}'),
            ],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=2, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert isinstance(result.error, MaxTurnsExceeded)
    assert toolset.dispatched == [
        ("write_source", {"path": "helper.py"}),
        ("write_source", {"path": "core.py"}),
        ("apply_patch", {"patch": "targeted"}),
    ]
    assert len(calls) == 3


def test_final_guard_binds_live_like_wrong_path_to_required_create(tmp_path):
    """A model cannot steer the singleton writer toward production source."""

    target = tmp_path / "target.py"
    target.write_text("value = partial\n", encoding="utf-8")
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
    )
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(build_exact_create_recovery_profile(
        context,
        target_path="required.py",
    ))
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if "tools" not in kwargs:
            assert "tool_choice" not in kwargs
            return _stub_response("DONE")
        assert kwargs["tool_choice"] == "required"
        assert "Final implementation turn" in kwargs["messages"][-1][
            "content"
        ]
        assert {
            descriptor["function"]["name"] for descriptor in kwargs["tools"]
        } == {"write_source"}
        schema = kwargs["tools"][0]["function"]["parameters"]
        assert set(schema["properties"]) == {"content"}
        return _stub_response(
            "",
            tool_calls=[_tool_call(
                "required",
                "write_source",
                # Mirrors GLM's prior attempt to choose component.py. The
                # bound handler ignores this non-schema argument.
                '{"file_path":"target.py","content":"REQUIRED = True\\n"}',
            )],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=1, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert isinstance(result.error, MaxTurnsExceeded)
    assert target.read_text(encoding="utf-8") == "value = partial\n"
    assert tmp_path.joinpath("required.py").read_text(encoding="utf-8") == (
        "REQUIRED = True\n"
    )
    assert len(calls) == 2


def test_early_length_uses_one_same_conversation_write_completion():
    """An early truncated answer must not strand unused write-capable turns."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
            ]

    toolset = _MutatingToolSet(
        results={"lookup": "repository context", "apply_patch": "ok"},
    )
    backend = OpenAIBackend(model="x", capability="workspace_write")
    backend.attach_toolset(toolset)
    calls: list[dict] = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        call_index = len(calls) - 1
        if call_index == 0:
            assert "tool_choice" not in kwargs
            return _stub_response(
                "",
                tool_calls=[_tool_call("read-1", "lookup", '{"path":"x"}')],
            )
        if call_index == 1:
            assert "tool_choice" not in kwargs
            return _stub_response(
                "partial implementation plan",
                finish_reason="length",
            )
        assert call_index == 2
        assert {
            descriptor["function"]["name"] for descriptor in kwargs["tools"]
        } == {"apply_patch"}
        assert kwargs["tool_choice"] == "required"
        messages = kwargs["messages"]
        prior_call = next(
            message for message in messages
            if message.get("role") == "assistant"
            and message.get("tool_calls")
        )
        assert prior_call["tool_calls"][0]["id"] == "read-1"
        assert next(
            message for message in messages
            if message.get("role") == "tool"
        ) == {
            "role": "tool",
            "tool_call_id": "read-1",
            "content": "repository context",
        }
        assert messages[-2] == {
            "role": "assistant",
            "content": "partial implementation plan",
        }
        assert "Bounded implementation completion" in messages[-1]["content"]
        return _stub_response(
            "",
            tool_calls=[
                _tool_call("write-1", "apply_patch", '{"patch":"done"}'),
            ],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None, max_turns=4, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert isinstance(result.error, MaxTurnsExceeded)
    assert result.final_message == "partial implementation plan"
    assert toolset.dispatched == [
        ("lookup", {"path": "x"}),
        ("apply_patch", {"patch": "done"}),
    ]
    assert len(calls) == 3


@pytest.mark.parametrize("max_turns", [1, 3])
def test_early_length_does_not_continue_without_remaining_writable_turn(
    max_turns: int,
):
    """Read-only roles stop; a required final write gets one recovery."""

    if max_turns == 1:
        class _ToolSet(_StubToolSet):
            write_names = {"apply_patch"}

            def to_openai_tools(self):
                return [{
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                }]

        toolset = _ToolSet()
        capability = "workspace_write"
    else:
        toolset = _StubToolSet()
        capability = "read_only"
    backend = OpenAIBackend(model="x", capability=capability)
    backend.attach_toolset(toolset)
    calls = 0

    def _create(**_kwargs):
        nonlocal calls
        calls += 1
        return _stub_response("partial", finish_reason="length")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None,
        max_turns=max_turns, budget_usd=None,
    ))

    assert result.stop_reason == "max_turns"
    assert calls == (2 if max_turns == 1 else 1)
    assert toolset.dispatched == []


def test_early_length_write_completion_respects_budget():
    """A truncated turn at the cost cap cannot schedule another request."""

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {}},
                },
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "parameters": {}},
                },
            ]

    toolset = _MutatingToolSet(results={"lookup": "context"})
    backend = OpenAIBackend(model="gpt-5.5", capability="workspace_write")
    backend.attach_toolset(toolset)
    responses = iter([
        _stub_response(
            "",
            tool_calls=[_tool_call("read-1", "lookup", "{}")],
        ),
        _stub_response(
            "partial",
            in_tokens=10_000_000,
            out_tokens=10_000_000,
            finish_reason="length",
        ),
    ])
    calls = 0

    def _create(**_kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s", user="u", tools=None,
        max_turns=4, budget_usd=0.01,
    ))

    assert result.stop_reason == "budget_exhausted"
    assert result.error is not None
    assert result.error.kind == "budget_exhausted"
    assert calls == 2
    assert toolset.dispatched == [("lookup", {})]


def test_max_turns_finalization_skipped_when_budget_exhausted():
    """The forced finalization turn respects the cost cap: if the loop
    already spent the budget, we don't spend more on a finalize call."""
    ts = _StubToolSet(results={"lookup": "scratch"})
    # Priced model so the cost cap actually trips (model "x" is zero-cost).
    b = OpenAIBackend(model="gpt-5.5", capability="read_only")
    b.attach_toolset(ts)

    calls = {"no_tools": 0}

    def _create(**kw):
        if "tools" not in kw:
            calls["no_tools"] += 1
        # Big token spend per turn so the budget trips immediately, but
        # report finish via tool_call so the loop keeps going.
        return _stub_response(
            "", in_tokens=10_000_000, out_tokens=10_000_000,
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        )

    b._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    inv = Invocation(
        system="s", user="u", tools=None, max_turns=2, budget_usd=0.01,
    )
    result = b.invoke(inv)

    # Budget cap breaks the loop before max_turns; no finalize attempted.
    assert result.stop_reason == "budget_exhausted"
    assert result.error is not None
    assert result.error.kind == "budget_exhausted"
    assert calls["no_tools"] == 0


def test_attach_toolset_none_detaches_and_invoke_is_single_shot():
    """Passing None detaches; invoke() falls back to single completion."""
    b = OpenAIBackend(model="x", capability="read_only")
    b.attach_toolset(_StubToolSet())
    b.attach_toolset(None)
    assert b._toolset is None

    b._client = _stub_client(_stub_response("single-shot"))
    inv = Invocation(
        system="s", user="u", tools=None, max_turns=1, budget_usd=None,
    )
    result = b.invoke(inv)
    assert result.final_message == "single-shot"


def test_chat_with_tools_drives_loop_with_explicit_toolset():
    ts = _StubToolSet()
    b = OpenAIBackend(model="x", capability="read_only")

    responses = iter([
        _stub_response(
            "", tool_calls=[_tool_call("c1", "lookup", "{}")],
        ),
        _stub_response("final"),
    ])
    b._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: next(responses)),
        ),
    )
    inv = Invocation(
        system="s", user="u", tools=None, max_turns=1, budget_usd=None,
    )
    result = b.chat_with_tools(inv, ts, max_rounds=5)
    assert result.final_message == "final"
    assert ts.dispatched == [("lookup", {})]


def test_chat_max_rounds_is_a_ceiling_not_a_floor():
    toolset = _StubToolSet()
    backend = OpenAIBackend(model="x", capability="read_only")
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if "tools" in kwargs:
            return _stub_response(
                "",
                tool_calls=[_tool_call(
                    f"call-{len(calls)}",
                    "lookup",
                    "{}",
                )],
            )
        return _stub_response("final")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    result = backend.chat_with_tools(
        Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=2,
            budget_usd=None,
        ),
        toolset,
        max_rounds=5,
    )

    assert result.final_message == "final"
    assert len([call for call in calls if "tools" in call]) == 2
    assert len(toolset.dispatched) == 2


def test_chat_max_response_tokens_reaches_every_completion():
    toolset = _StubToolSet()
    backend = OpenAIBackend(model="x", capability="read_only")
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        if "tools" in kwargs:
            return _stub_response(
                "",
                tool_calls=[_tool_call("call", "lookup", "{}")],
            )
        return _stub_response("final")

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    backend.chat_with_tools(
        Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=None,
        ),
        toolset,
        max_rounds=4,
        max_response_tokens=321,
    )

    assert calls
    assert all(call["max_tokens"] == 321 for call in calls)


def test_non_null_cache_is_rejected_before_transport():
    backend = OpenAIBackend(model="x", capability="read_only")
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: pytest.fail("transport must not run"),
            ),
        ),
    )

    with pytest.raises(NotImplementedError, match="ToolCallCache"):
        backend.chat_with_tools(
            Invocation(
                system="s",
                user="u",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            _StubToolSet(),
            cache=object(),
        )


def test_context_budget_caps_cumulative_utf8_tool_results_and_resets():
    toolset = _StubToolSet(results={"lookup": "ééé"})
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(toolset, context_budget=5)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _stub_response(
            "",
            tool_calls=[_tool_call("call", "lookup", "{}")],
        )

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    invocation = Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=2,
        budget_usd=None,
    )
    first = backend.invoke(invocation)
    second = backend.invoke(invocation)

    for result in (first, second):
        assert result.error is not None
        assert result.error.retriable is False
        assert result.stop_reason == "context_limit"
        assert "6 > 5 UTF-8 bytes" in result.error.message
    assert len(calls) == 2


def test_tool_loop_defers_schema_until_no_tools_finalization():
    """Tools and strict JSON are separate phases for gateway compatibility."""

    ts = _StubToolSet(results={"lookup": "evidence"})
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(ts)
    responses = iter([
        _stub_response(
            "",
            tool_calls=[_tool_call("c1", "lookup", "{}")],
        ),
        _stub_response("draft based on evidence"),
        _stub_response('{"answer":"final"}'),
    ])
    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        return next(responses)

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=5,
        budget_usd=None,
        response_schema=schema,
    ))

    assert result.error is None
    assert result.final_message == '{"answer":"final"}'
    assert ts.dispatched == [("lookup", {})]
    assert len(calls) == 3
    assert all("tools" in call for call in calls[:2])
    assert all("response_format" not in call for call in calls[:2])
    assert "tools" not in calls[2]
    assert calls[2]["response_format"]["json_schema"]["schema"] == schema
    assert calls[2]["messages"][-2] == {
        "role": "assistant",
        "content": "draft based on evidence",
    }
    assert calls[2]["messages"][-1]["role"] == "user"
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 60


def test_structured_finalization_never_reads_reasoning_content():
    """A finalization that returns only reasoning leaves the draft as it is.

    The reasoning here is one complete JSON object, the shape that used to be
    taken as the structured answer.  The unstructured draft stays the final
    message of a normal end turn; Assay's validation of it decides the rest.
    """

    backend = OpenAIBackend(
        model="nvidia/zai-org/glm-5.3",
        base_url="https://inference-api.nvidia.com",
        capability="read_only",
    )
    backend.attach_toolset(_StubToolSet())
    responses = iter([
        _stub_response("draft"),
        _stub_response(
            None,
            provider_specific_fields={
                "reasoning_content": '{"answer":"final"}',
            },
        ),
    ])
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: next(responses)),
        ),
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=2,
        budget_usd=None,
        response_schema=schema,
    ))

    assert result.error is None
    assert result.stop_reason == "end_turn"
    assert result.final_message == "draft"
    assert [message.content for message in result.transcript] == ["u", "draft"]


@pytest.mark.parametrize(
    "site",
    ["required_write_nudge", "length_continuation", "finalization_draft"],
)
@pytest.mark.parametrize(
    ("reasoning_fields", "replayed"),
    [
        ({"reasoning_content": "prose-turn-thinking"}, True),
        (
            {
                "provider_specific_fields": {
                    "reasoning_content": "prose-turn-thinking",
                },
            },
            True,
        ),
        ({}, False),
    ],
)
def test_prose_assistant_turns_replay_reasoning_content(
    site, reasoning_fields, replayed,
):
    """An assistant turn without tool calls goes back with its reasoning.

    Three places put such a turn back into the provider conversation: the
    nudge after an ignored required write, the continuation after an early
    ``length`` stop, and the draft before the constrained finalization.  The
    K3 series requires the complete assistant message on the next request, so
    each carries the provider's reasoning when one was sent and no
    ``reasoning_content`` key otherwise.  None of it reaches Crucible's
    transcript or final message.
    """

    class _MutatingToolSet(_StubToolSet):
        write_names = {"apply_patch"}

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {"name": "apply_patch", "parameters": {}},
            }]

    schema = None
    if site == "finalization_draft":
        toolset = _StubToolSet(results={"lookup": "evidence"})
        capability = "read_only"
        max_turns = 5
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        responses = [
            _stub_response(
                "",
                tool_calls=[_tool_call("c1", "lookup", "{}")],
                reasoning_content="tool-turn-thinking",
            ),
            _stub_response("prose turn", **reasoning_fields),
            _stub_response('{"answer":"final"}'),
        ]
    else:
        toolset = _MutatingToolSet(results={"apply_patch": "ok"})
        capability = "workspace_write"
        # One turn makes the first request a required write; four leave room
        # for the same-conversation completion after an early ``length``.
        max_turns = 1 if site == "required_write_nudge" else 4
        responses = [
            _stub_response(
                "prose turn",
                finish_reason=(
                    "stop" if site == "required_write_nudge" else "length"
                ),
                **reasoning_fields,
            ),
            _stub_response(
                "",
                tool_calls=[_tool_call("write-1", "apply_patch", "{}")],
            ),
        ]
    backend = OpenAIBackend(
        model="nvidia/moonshotai/kimi-k3.1-pro",
        capability=capability,
    )
    backend.attach_toolset(toolset)
    pending = iter(responses)
    calls = []

    def _create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(pending)

    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=max_turns,
        budget_usd=None,
        response_schema=schema,
    ))

    assert len(calls) == len(responses)
    if site == "required_write_nudge":
        assert calls[0]["tool_choice"] == "required"
        assert "write tool was not called" in calls[-1]["messages"][-1]["content"]
    elif site == "length_continuation":
        assert "tool_choice" not in calls[0]
        assert "hit its output limit" in calls[-1]["messages"][-1]["content"]
    else:
        assert "response_format" in calls[-1]
        assert result.final_message == '{"answer":"final"}'
    expected = {"role": "assistant", "content": "prose turn"}
    if replayed:
        expected["reasoning_content"] = "prose-turn-thinking"
    assert calls[-1]["messages"][-2] == expected
    assert all(
        "thinking" not in message.content for message in result.transcript
    )
    assert "thinking" not in result.final_message


def test_outer_invoke_does_not_double_retry_tool_loop_error(monkeypatch):
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(_StubToolSet())
    calls = []

    def _raise_connection(**_kwargs):
        calls.append(1)
        raise ConnectionError("tool-loop transport failure")

    monkeypatch.setattr("crucible.backends.openai_compat.time.sleep", lambda _: None)
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_raise_connection),
        ),
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=2,
        budget_usd=None,
        policy=_policy(3),
    ))

    assert len(calls) == 3
    assert result.error is not None
    assert result.error.retriable is True


def test_chat_with_tools_no_toolset_falls_back_to_invoke():
    b = OpenAIBackend(model="x")
    b._client = _stub_client(_stub_response("no-tools"))
    inv = Invocation(
        system="s", user="u", tools=None, max_turns=1, budget_usd=None,
    )
    result = b.chat_with_tools(inv, None)
    assert result.final_message == "no-tools"


def test_chat_with_tools_strict_no_toolset_raises():
    b = OpenAIBackend(model="x")
    inv = Invocation(
        system="s", user="u", tools=None, max_turns=1, budget_usd=None,
    )
    with pytest.raises(NotImplementedError):
        b.chat_with_tools(inv, None, strict=True)


def test_missing_usage_is_marked_incomplete():
    response = _stub_response("done")
    response.usage = None
    backend = OpenAIBackend(model="x")
    backend._client = _stub_client(response)

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=_policy(1),
    ))

    assert result.error is None
    assert result.usage.input_tokens == 0
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.execution_trace is not None
    assert result.execution_trace.terminal_event_count == 1


def test_single_shot_terminal_over_budget_returns_typed_error():
    backend = OpenAIBackend(model="gpt-5.5")
    backend._client = _stub_client(_stub_response(
        "done",
        in_tokens=1_000_000,
        out_tokens=1_000_000,
    ))

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=0.01,
    ))

    assert result.stop_reason == "budget_exhausted"
    assert result.error is not None
    assert result.error.kind == "budget_exhausted"


def test_trace_records_served_model_endpoint_and_one_terminal():
    backend = OpenAIBackend(
        model="x",
        base_url=(
            "https://operator:supersecret@gateway.example/"
            "v1?token=must-not-persist"
        ),
    )
    backend._client = _stub_client(_stub_response(
        "done",
        model="served-x",
        request_id="provider-request-7",
    ))

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
    ))

    assert result.completeness.complete is True
    assert result.execution_trace is not None
    assert result.execution_trace.requested_model == "x"
    assert result.execution_trace.served_model == "served-x"
    assert result.execution_trace.endpoint == "https://gateway.example/v1"
    assert result.execution_trace.provider_request_id == "provider-request-7"
    assert result.execution_trace.terminal_event_count == 1
    assert backend.features.tool_loop == "adapter_owned"


def test_direct_http_request_does_not_ingest_ambient_instruction_files(
    monkeypatch,
    tmp_path,
):
    marker = "HOSTILE-AMBIENT-MARKER-MUST-NOT-LEAVE-DISK"
    for relative in (
        "AGENTS.md",
        "nested/AGENTS.md",
        "CLAUDE.md",
        ".mcp.json",
        ".codex/config.toml",
        ".claude/settings.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    captured = {}
    backend = OpenAIBackend(model="x", cwd=str(tmp_path))
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (
                    captured.update(kwargs) or _stub_response("safe")
                ),
            ),
        ),
    )

    result = backend.invoke(Invocation(
        system="explicit-system",
        user="explicit-user",
        tools=None,
        max_turns=1,
        budget_usd=None,
    ))

    assert result.error is None
    outbound = json.dumps(captured["messages"], sort_keys=True)
    assert marker not in outbound
    assert outbound == (
        '[{"content": "explicit-system", "role": "system"}, '
        '{"content": "explicit-user", "role": "user"}]'
    )


# ----------------------------------------------------------------------
# close
# ----------------------------------------------------------------------


def test_close_releases_client(monkeypatch):
    closed = []

    class _Client:
        def close(self):
            closed.append(True)

    b = OpenAIBackend(model="x")
    b._client = _Client()
    b.close()
    assert closed == [True]
    assert b._client is None


def test_close_when_no_client_is_safe():
    b = OpenAIBackend(model="x")
    # never used _get_client → _client stays None
    b.close()   # no exception


def test_close_failure_retains_exact_client_for_retry():
    class _Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("still draining")

    client = _Client()
    backend = OpenAIBackend(model="x")
    backend._client = client  # noqa: SLF001

    with pytest.raises(RuntimeError, match="still draining"):
        backend.close()
    assert backend._client is client  # noqa: SLF001

    backend.close()
    assert client.close_calls == 2
    assert backend._client is None  # noqa: SLF001


@pytest.mark.parametrize("close_value", [None, "not-callable"])
def test_close_without_callable_client_close_retains_transport_owner(
    close_value,
) -> None:
    client = SimpleNamespace()
    if close_value is not None:
        client.close = close_value
    backend = OpenAIBackend(model="x")
    backend._client = client  # noqa: SLF001

    with pytest.raises(RuntimeError, match="transport settlement is unproved"):
        backend.close()

    assert backend._client is client  # noqa: SLF001


def test_invocation_deadline_stops_retry_backoff_and_decreases_timeout(
    monkeypatch,
):
    clock = {"now": 100.0}
    calls: list[float] = []

    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.monotonic",
        lambda: clock["now"],
    )
    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.sleep",
        lambda delay: clock.__setitem__("now", clock["now"] + delay),
    )

    def _create(**kwargs):
        calls.append(kwargs["timeout"])
        raise ConnectionError("retryable")

    backend = OpenAIBackend(model="x")
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    policy = BackendPolicy(
        retry_budget=3,
        timeout_seconds=0.12,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=policy,
    ))

    assert len(calls) == 1
    assert calls[0] == pytest.approx(0.12)
    assert result.stop_reason == "timeout"
    assert result.error is not None
    assert result.error.retriable is False
    assert result.execution_trace is not None
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "retry",
        "limit",
        "terminal",
    ]


def test_tool_dispatch_is_bounded_by_same_invocation_deadline(monkeypatch):
    clock = {"now": 200.0}

    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.monotonic",
        lambda: clock["now"],
    )

    class _SlowToolSet(_StubToolSet):
        def dispatch(self, name, args):
            result = super().dispatch(name, args)
            clock["now"] += 1.0
            return result

    toolset = _SlowToolSet(results={"lookup": "late"})
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(toolset)
    calls = []
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (
                    calls.append(kwargs)
                    or _stub_response(
                        "",
                        tool_calls=[_tool_call("call-1", "lookup", "{}")],
                    )
                ),
            ),
        ),
    )
    policy = BackendPolicy(
        retry_budget=2,
        timeout_seconds=0.5,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=3,
        budget_usd=None,
        policy=policy,
    ))

    assert len(calls) == 1
    assert toolset.dispatched == [("lookup", {})]
    assert result.stop_reason == "timeout"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.execution_trace is not None
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "usage",
        "tool_call",
        "tool_result",
        "limit",
        "terminal",
    ]


def test_forced_finalization_uses_remaining_invocation_deadline(monkeypatch):
    clock = {"now": 300.0}
    request_timeouts = []

    monkeypatch.setattr(
        "crucible.backends.openai_compat.time.monotonic",
        lambda: clock["now"],
    )

    class _AdvancingToolSet(_StubToolSet):
        def dispatch(self, name, args):
            clock["now"] += 0.4
            return super().dispatch(name, args)

    state = {"calls": 0}

    def _create(**kwargs):
        request_timeouts.append(kwargs["timeout"])
        state["calls"] += 1
        if state["calls"] == 1:
            return _stub_response(
                "",
                tool_calls=[_tool_call("call-1", "lookup", "{}")],
            )
        clock["now"] += 0.2
        raise TimeoutError("forced finalization exceeded remaining time")

    toolset = _AdvancingToolSet(results={"lookup": "evidence"})
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(toolset)
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        ),
    )
    policy = BackendPolicy(
        retry_budget=2,
        timeout_seconds=0.5,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )

    result = backend.invoke(Invocation(
        system="s",
        user="u",
        tools=None,
        max_turns=1,
        budget_usd=None,
        policy=policy,
    ))

    assert len(request_timeouts) == 2
    assert request_timeouts[0] == pytest.approx(0.5)
    assert request_timeouts[1] == pytest.approx(0.1)
    assert result.stop_reason == "timeout"
    assert result.error is not None
    assert result.error.retriable is False
    assert result.execution_trace is not None
    assert result.execution_trace.events[-2].kind == "limit"
    assert result.execution_trace.events[-2].name == "deadline"


def test_close_cancels_tool_loop_at_dispatch_boundary():
    entered = threading.Event()
    release = threading.Event()

    class _BlockingToolSet(_StubToolSet):
        def dispatch(self, name, args):
            entered.set()
            assert release.wait(timeout=2)
            return super().dispatch(name, args)

    toolset = _BlockingToolSet(results={"lookup": "late"})
    backend = OpenAIBackend(model="x", capability="read_only")
    backend.attach_toolset(toolset)
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: _stub_response(
                    "",
                    tool_calls=[_tool_call("call-1", "lookup", "{}")],
                ),
            ),
        ),
        close=lambda: None,
    )
    policy = BackendPolicy(
        retry_budget=2,
        timeout_seconds=5,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(backend.invoke(Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=3,
            budget_usd=None,
            policy=policy,
        ))),
    )
    worker.start()
    assert entered.wait(timeout=2)
    backend.close()
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(results) == 1
    result = results[0]
    assert result.stop_reason == "cancelled"
    assert result.error is not None
    assert result.error.kind == "cancellation"
    assert result.execution_trace is not None
    assert [event.kind for event in result.execution_trace.events][-2:] == [
        "cancellation",
        "terminal",
    ]


def test_tool_loop_transport_retry_preserves_conversation():
    """A transport blip mid tool-loop must retry the SAME create() with the
    accumulated ``messages`` (the client-side conversation), not unwind to
    invoke() and restart with a fresh [system,user] list. Turn 0 does a
    tool call (messages grows); turn 1's create transport-fails then
    succeeds — the retry must see the GROWN messages (same object), proving
    rounds 0..N-1 were preserved (no context reset)."""
    ts = _StubToolSet(results={"lookup": "v"})
    b = OpenAIBackend(model="x", capability="read_only")
    b.attach_toolset(ts)

    seen: list[tuple[int, int]] = []  # (id(messages), len) per create
    state = {"n": 0}

    def _create(**kw):
        seen.append((id(kw["messages"]), len(kw["messages"])))
        n = state["n"]; state["n"] += 1
        if n == 0:  # turn 0 → tool call (grows messages)
            return _stub_response("", tool_calls=[_tool_call("c1", "lookup", "{}")])
        if n == 1:  # turn 1 → transport drop
            raise ConnectionError("stream dropped mid-loop")
        return _stub_response("DONE")  # in-loop retry → success

    b._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
    )
    inv = Invocation(system="s", user="u", tools=None, max_turns=5, budget_usd=None)
    result = b.invoke(inv)

    assert result.final_message == "DONE"
    assert len(seen) == 3                      # turn0, turn1-fail, turn1-retry
    assert seen[1][0] == seen[2][0]            # SAME messages object (no restart)
    assert seen[2][1] == seen[1][1] >= 4       # grown (tool turn preserved)
    assert seen[2][1] > seen[0][1]             # bigger than the initial turn-0 call
    assert ts.dispatched == [("lookup", {})]
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.completeness.trace_complete is True
    assert result.execution_trace is not None
    assert "retry" in {
        event.kind for event in result.execution_trace.events
    }
