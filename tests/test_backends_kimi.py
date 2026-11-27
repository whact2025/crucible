# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Offline qualification tests for the dedicated Kimi Agent SDK backend."""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import importlib.util
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest
from assay import BackendPolicy, ConversationTransportRequest, Invocation
from assay.backend.protocol import Backend

from crucible.backends import kimi as kimi_mod
from crucible.backends.kimi import (
    KimiBackend,
    KimiSDKQualificationError,
)
from crucible.backends.openai_compat import OpenAIBackend

_MODEL = "kimi-k3.1-pro"
_BASE_URL = "https://api.moonshot.ai/v1"
_TOKEN = "test-runtime-token-not-real"
_HOSTILE_KIMI_PROVIDER_ENV = {
    "KIMI_BASE_URL": "https://hostile.invalid/v1",
    "KIMI_API_KEY": "hostile-native-provider-token",
    "KIMI_MODEL_NAME": "hostile-model",
    "KIMI_MODEL_CAPABILITIES": "thinking",
    "KIMI_MODEL_MAX_CONTEXT_SIZE": "1",
    "KIMI_MODEL_TEMPERATURE": "2",
    "KIMI_MODEL_TOP_P": "0",
    "KIMI_MODEL_MAX_TOKENS": "1",
}


class _ToolSet:
    def __init__(self, *, writes: set[str] | None = None) -> None:
        self.write_names = set(writes or ())
        self.read_only_names = {"echo"} - self.write_names
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "echo",
                "description": "Echo a value.",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return str(arguments.get("value", ""))


def _call_attached_mcp(
    request: kimi_mod._WorkerRequest,
    *,
    value: str = "through-bridge",
) -> str:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    assert request.mcp_url is not None
    assert request.mcp_auth_token is not None

    async def call() -> str:
        async with streamablehttp_client(
            request.mcp_url,
            headers={
                "Authorization": f"Bearer {request.mcp_auth_token}",
            },
        ) as (read, write, _info):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "echo",
                    {"value": value},
                )
                return result.content[0].text

    return asyncio.run(call())


class _ScriptedKimiEndpoint(http.server.BaseHTTPRequestHandler):
    """Two-step OpenAI stream: tool call, then final answer.

    The usage is synthetic. Both final packets carry an explicit
    ``cached_tokens: 0`` so that strict accounting can accept them; Moonshot
    was observed on 2026-11-18 to omit the counter when nothing was read from
    cache. The recorded kimi-k3.1-pro shapes are pinned in
    ``test_backends_kimi_k3_stream_contract.py``.
    """

    server: Any
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)
        turn = len(self.server.requests)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if turn == 1:
            first_prompt_tokens = getattr(
                self.server,
                "first_prompt_tokens",
                20,
            )
            chunks = [
                {
                    "id": "chatcmpl-tool",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": (
                                    "I must use the confined echo tool."
                                ),
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-echo-1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": ('{"value":"scripted-nonce"}'),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-tool",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
                {
                    "id": "chatcmpl-tool",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _MODEL,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": first_prompt_tokens,
                        "completion_tokens": 5,
                        "total_tokens": first_prompt_tokens + 5,
                        "prompt_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 0,
                        },
                    },
                },
            ]
        else:
            chunks = [
                {
                    "id": "chatcmpl-final",
                    "object": "chat.completion.chunk",
                    "created": 2,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "reasoning_content": (
                                    "The function returned the nonce: "
                                    '"scripted-nonce". I need to repeat this '
                                    "in my final answer as instructed by the "
                                    "system prompt."
                                ),
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-final",
                    "object": "chat.completion.chunk",
                    "created": 2,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "scripted-nonce observed"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-final",
                    "object": "chat.completion.chunk",
                    "created": 2,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                },
                {
                    "id": "chatcmpl-final",
                    "object": "chat.completion.chunk",
                    "created": 2,
                    "model": _MODEL,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 4,
                        "total_tokens": 34,
                        "prompt_tokens_details": {
                            "cached_tokens": 0,
                            "cache_write_tokens": 0,
                        },
                    },
                },
            ]
        encoded = (
            b"".join(
                b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n"
                for chunk in chunks
            )
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        self.close_connection = True


class _RetriableMoonshotEndpoint(http.server.BaseHTTPRequestHandler):
    """Return one retriable provider error for every physical POST."""

    server: Any
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)
        encoded = json.dumps(
            {
                "error": {
                    "message": "temporary Moonshot failure",
                    "type": "server_error",
                    "code": "service_unavailable",
                }
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(503)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        self.close_connection = True


class _NativeStreamEndpoint(http.server.BaseHTTPRequestHandler):
    """Serve a caller-provided sequence of native chat-completion chunks."""

    server: Any
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)
        encoded = (
            b"".join(
                b"data: "
                + json.dumps(chunk, separators=(",", ":")).encode()
                + b"\n\n"
                for chunk in self.server.chunks
            )
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        self.close_connection = True


_WORKER_PROJECT = Path(kimi_mod.__file__).with_name("kimi_worker")


def _project_entries(project: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every entry of a worker project.

    The installed environment is left out: nothing here imports from it.
    """

    entries: dict[str, tuple[int, int]] = {}
    for root, directories, files in os.walk(project):
        if Path(root) == project:
            directories[:] = [name for name in directories if name != ".venv"]
        for name in (*directories, *files):
            entry = Path(root) / name
            metadata = entry.lstat()
            entries[entry.relative_to(project).as_posix()] = (
                metadata.st_size,
                metadata.st_mtime_ns,
            )
    return entries


def _load_worker_source(name: str, source: Path, *, project: Path = _WORKER_PROJECT) -> Any:
    """Load a source file of the worker project by path and leave no bytecode there.

    The adapter's pin hashes the worker's source directory with every
    bytecode entry in it and is checked again before every launch. A
    ``__pycache__`` written by this process would change the pin of the
    checkout under every other test that runs in it, while those tests run.
    """

    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    before = _project_entries(project)
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.dont_write_bytecode = write_bytecode
    assert _project_entries(project) == before, "loading worker source changed its project"
    return module


def _load_production_worker_module() -> Any:
    """Load the isolated worker source that the deployed subprocess executes."""

    module_name = "crucible_kimi_worker_test_target"
    if module_name in sys.modules:
        return sys.modules[module_name]
    return _load_worker_source(
        module_name,
        _WORKER_PROJECT / "src" / "crucible_kimi_worker" / "__main__.py",
    )


worker_mod = _load_production_worker_module()


def _native_chunk(
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    model: str = _MODEL,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-native-qualification",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }


def _native_usage_chunk(
    *,
    prompt_tokens: int = 13,
    completion_tokens: int = 4,
    cached_tokens: int = 2,
    cache_write_tokens: int = 0,
    model: str = _MODEL,
) -> dict[str, Any]:
    """Return the K3-series final statistics packet (``choices: []``).

    The counters are synthetic and ``cached_tokens`` is always present. An
    explicit zero is a shape the provider was observed not to send
    (2026-11-18: the counter is omitted when nothing was read from cache).
    """

    return {
        "id": "chatcmpl-native-qualification",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        },
    }


def _run_pinned_native_stream(
    locked_worker_launch: Any,
    chunks: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ``chunks`` through the pinned native stream in the locked interpreter.

    The interpreter is started through the launcher of the suite fixture, so
    it runs under the worker harness with the network guard installed.
    """

    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _NativeStreamEndpoint,
    )
    endpoint.requests = []
    endpoint.chunks = chunks
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    script = """
import asyncio
import json
import sys

from kosong.chat_provider.kimi import Kimi
from crucible_kimi_worker import __main__ as worker


async def main():
    provider = Kimi(
        model=sys.argv[2],
        api_key="test-native-stream-token",
        base_url=sys.argv[1],
        stream=True,
    )
    provider.client = provider.client.with_options(max_retries=0)
    worker._install_moonshot_stream_compat(provider, model=sys.argv[2])
    try:
        stream = await provider.generate("", [], [])
        parts = []
        async for part in stream:
            item = {"type": type(part).__name__}
            if hasattr(part, "id"):
                item["id"] = part.id
            if hasattr(part, "function"):
                item["name"] = part.function.name
                item["arguments"] = part.function.arguments
            if hasattr(part, "text"):
                item["text"] = part.text
            if hasattr(part, "think"):
                item["think"] = part.think
            parts.append(item)
        result = {
            "error_type": None,
            "error": None,
            "parts": parts,
            "finish_reason": stream.finish_reason,
            "served_model": stream.served_model,
        }
        usage = stream.usage
        if usage is not None:
            result["usage"] = {
                "input": usage.input,
                "input_other": usage.input_other,
                "output": usage.output,
                "input_cache_read": usage.input_cache_read,
                "input_cache_creation": usage.input_cache_creation,
            }
    except Exception as exc:
        result = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    print(json.dumps(result, sort_keys=True))


asyncio.run(main())
"""
    try:
        completed = locked_worker_launch.run(
            [
                str(kimi_mod._DEFAULT_WORKER_PYTHON),
                "-c",
                script,
                base_url,
                _MODEL,
            ],
            capture_output=True,
            check=False,
            env=kimi_mod._worker_environment(),
            text=True,
            timeout=10.0,
        )
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout), endpoint.requests


@pytest.fixture(autouse=True)
def _preserve_fake_session_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep fake Session tests independent of the isolated native package."""

    monkeypatch.setattr(
        kimi_mod,
        "_install_moonshot_stream_compat",
        lambda _provider, **_kwargs: None,
    )
    monkeypatch.setattr(
        worker_mod,
        "_install_moonshot_stream_compat",
        lambda _provider, **_kwargs: None,
    )


def _qualified_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "kimi-agent-sdk": "0.0.5",
        "kimi-cli": "1.12.0",
    }
    monkeypatch.setattr(
        kimi_mod,
        "_distribution_version",
        lambda name: versions[name],
    )


def _invocation(
    *,
    response_schema: dict[str, Any] | None = None,
    policy: BackendPolicy | None = None,
) -> Invocation:
    return Invocation(
        system="SYSTEM CONTRACT",
        user="Return the requested result.",
        tools=None,
        max_turns=2,
        budget_usd=None,
        policy=policy
        or BackendPolicy(
            retry_budget=1,
            timeout_seconds=2.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
        response_schema=response_schema,
    )


def _success_outcome(**overrides: Any) -> kimi_mod._WorkerOutcome:
    values: dict[str, Any] = {
        "final_message": '{"ok":true}',
        "input_tokens": 15,
        "output_tokens": 7,
        "cache_creation_tokens": 2,
        "cache_read_tokens": 3,
        "usage_seen": True,
        "usage_valid": True,
        "terminal_seen": True,
        "configured_model": _MODEL,
        "served_model": _MODEL,
        "stop_reason": "end_turn",
        "steps_used": 1,
        "settled_usage_steps": 1,
    }
    values.update(overrides)
    return kimi_mod._WorkerOutcome(**values)


def test_kimi_backend_is_dedicated_and_satisfies_backend_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    backend = KimiBackend()

    assert isinstance(backend, Backend)
    assert backend.name == "kimi"
    assert backend.model == _MODEL
    assert not isinstance(backend, OpenAIBackend)
    assert not issubclass(KimiBackend, OpenAIBackend)


def test_kimi_rejects_conversation_before_credentials_or_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    invocation = replace(
        _invocation(),
        conversation=ConversationTransportRequest(
            mode="workflow_bound",
            action="reuse",
            opaque_id="provider-thread",
            identity_fingerprint="sha256:" + "a" * 64,
            operation_digest="sha256:" + "b" * 64,
        ),
    )

    with pytest.raises(NotImplementedError, match="does not support"):
        KimiBackend().invoke(invocation)


@pytest.mark.parametrize(
    "model",
    [
        # Retired route ids are refused outright: no alias, no fallback.
        "kimi-k3",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.7-code",
        # A current Moonshot id that this route does not admit.
        "kimi-k3.1-pro-highspeed",
        # The NVIDIA gateway id of the same model is another route.
        "nvidia/moonshotai/kimi-k3.1-pro",
    ],
)
def test_kimi_backend_rejects_any_model_substitution(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    _qualified_versions(monkeypatch)

    with pytest.raises(ValueError, match="supports only the qualified model"):
        KimiBackend(model=model)


def test_kimi_missing_isolated_runtime_fails_closed_at_invoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(worker_python=tmp_path / "missing-python")

    result = backend.invoke(_invocation())

    assert result.error.kind == "transport"
    assert result.error.retriable is False
    assert "worker runtime is not installed" in result.error.message


def test_kimi_unreviewed_worker_tuple_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker_python = tmp_path / "python"
    worker_python.write_text("#!/bin/sh\nexit 0\n")
    worker_python.chmod(0o700)
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(
            {
                "protocol": kimi_mod._WORKER_PROTOCOL,
                "type": "probe",
                "sdk_version": "0.0.5",
                "cli_version": "1.12.1",
                "kosong_version": "0.42.0",
                "openai_version": "2.14.0",
            }
        ).encode(),
        stderr=b"",
    )
    commands: list[list[str]] = []

    def run(command, *_args, **_kwargs):
        commands.append(command)
        return completed

    with pytest.raises(
        KimiSDKQualificationError,
        match="does not match the qualified",
    ):
        monkeypatch.setattr(kimi_mod.subprocess, "run", run)
        kimi_mod._qualify_worker_runtime.cache_clear()
        kimi_mod._qualify_worker_runtime(str(worker_python))
    kimi_mod._qualify_worker_runtime.cache_clear()
    assert commands[0][1:3] == ["-B", "-I"]


def test_kimi_worker_probe_attests_complete_pinned_native_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "kimi-agent-sdk": "0.0.5",
        "kimi-cli": "1.12.0",
        "kosong": "0.42.0",
        "openai": "2.14.0",
    }
    monkeypatch.setattr(
        worker_mod,
        "_distribution_version",
        lambda name: versions[name],
    )

    assert worker_mod._runtime_identity() == {
        "protocol": worker_mod._PROTOCOL,
        "type": "probe",
        "sdk_version": "0.0.5",
        "cli_version": "1.12.0",
        "kosong_version": "0.42.0",
        "openai_version": "2.14.0",
    }
    worker_mod._assert_runtime_identity()


@pytest.mark.parametrize("kosong_version", [None, "0.41.0"])
def test_kimi_parent_requires_exact_kosong_probe_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kosong_version: str | None,
) -> None:
    worker_python = tmp_path / "python"
    worker_python.write_text("#!/bin/sh\nexit 0\n")
    worker_python.chmod(0o700)
    payload = {
        "protocol": kimi_mod._WORKER_PROTOCOL,
        "type": "probe",
        "sdk_version": "0.0.5",
        "cli_version": "1.12.0",
        "openai_version": "2.14.0",
    }
    if kosong_version is not None:
        payload["kosong_version"] = kosong_version
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(payload).encode(),
        stderr=b"",
    )
    monkeypatch.setattr(kimi_mod.subprocess, "run", lambda *_a, **_k: completed)

    with pytest.raises(
        KimiSDKQualificationError,
        match="kosong==0.42.0",
    ):
        kimi_mod._qualify_worker_runtime.cache_clear()
        kimi_mod._qualify_worker_runtime(str(worker_python))
    kimi_mod._qualify_worker_runtime.cache_clear()


def test_kimi_credentials_are_read_at_invocation_time_and_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    captured: list[kimi_mod._WorkerRequest] = []

    def runner(request, _timeout, **_kwargs):
        captured.append(request)
        return _success_outcome()

    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    backend = KimiBackend(_worker_runner=runner)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    result = backend.invoke(
        _invocation(
            response_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }
        )
    )

    assert result.error is None
    assert result.final_message == '{"ok":true}'
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 7
    assert result.usage.cost_usd == pytest.approx(
        10 * 2.00 / 1_000_000
        + 2 * 2.00 / 1_000_000
        + 3 * 0.20 / 1_000_000
        + 7 * 10.00 / 1_000_000
    )
    assert result.usage.cost_usd == pytest.approx(0.0000946)
    assert result.completeness.complete is True
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "usage",
        "terminal",
    ]
    assert result.execution_trace.backend == "kimi"
    assert result.execution_trace.requested_model == _MODEL
    assert result.execution_trace.served_model == _MODEL
    assert result.execution_trace.requested_effort == "max"
    assert result.execution_trace.endpoint == _BASE_URL

    assert len(captured) == 1
    request = captured[0]
    assert request.base_url == _BASE_URL
    assert request.auth_token == _TOKEN
    assert request.max_steps == 1
    assert request.max_response_tokens == 32_768
    assert json.loads(request.schema_json)["required"] == ["ok"]
    assert _TOKEN not in repr(request)
    assert _TOKEN not in repr(backend.__dict__)


def test_kimi_route_owned_max_response_tokens_reaches_plain_invoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    captured: list[kimi_mod._WorkerRequest] = []

    backend = KimiBackend(
        max_response_tokens=16_384,
        _worker_runner=lambda request, *_args, **_kwargs: (
            captured.append(request) or _success_outcome()
        ),
    )

    result = backend.invoke(_invocation())

    assert result.error is None
    assert backend.max_response_tokens == 16_384
    assert captured[0].max_response_tokens == 16_384


@pytest.mark.parametrize(
    "value",
    [True, 0, -1, kimi_mod._MODEL_CONTEXT_SIZES[kimi_mod._MODEL] + 1],
)
def test_kimi_route_owned_max_response_tokens_is_closed(value: Any) -> None:
    with pytest.raises(ValueError, match="max_response_tokens"):
        KimiBackend(max_response_tokens=value)


@pytest.mark.parametrize(
    (
        "worker_stop",
        "worker_error",
        "final_message",
        "expected_error_kind",
    ),
    [
        (
            "max_tokens",
            "max_tokens",
            "partial billed response",
            "transport",
        ),
        (
            "content_filter",
            "content_rejected",
            "",
            "content_rejected",
        ),
    ],
)
def test_kimi_parent_preserves_billed_nonstop_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    worker_stop: str,
    worker_error: str,
    final_message: str,
    expected_error_kind: str,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(
            final_message=final_message,
            stop_reason=worker_stop,
            error_kind=worker_error,
            error_message=f"terminal {worker_stop}",
            retriable=False,
        )
    )

    result = backend.invoke(_invocation())

    assert result.final_message == final_message
    assert result.stop_reason == worker_stop
    assert result.error.kind == expected_error_kind
    assert result.error.retriable is False
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 7
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert result.completeness.terminal_complete is True
    assert result.execution_trace.served_model == _MODEL
    assert result.execution_trace.events[-2].kind == "usage"
    assert result.execution_trace.events[-1].kind == "terminal"
    assert result.execution_trace.events[-1].name == worker_stop


def test_kimi_legacy_infer_and_model_env_cannot_redirect_moonshot_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[kimi_mod._WorkerRequest] = []

    def runner(request, _timeout, **_kwargs):
        captured.append(request)
        return _success_outcome()

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    monkeypatch.setenv("INFER_BASE_URL", "https://hostile.invalid/v1")
    monkeypatch.setenv("INFER_AUTH_TOKEN", "legacy-hostile-token")
    monkeypatch.setenv("KIMI_MODEL_NAME", "hostile-model")
    backend = KimiBackend(_worker_runner=runner)

    result = backend.invoke(_invocation())

    assert result.error is None
    assert backend.model == _MODEL
    assert len(captured) == 1
    assert captured[0].base_url == _BASE_URL
    assert captured[0].auth_token == _TOKEN


@pytest.mark.parametrize(
    ("configured_model", "served_model"),
    [
        (None, _MODEL),
        # A sibling id is not proof of this invocation's model.
        ("kimi-k3.1-pro-highspeed", _MODEL),
        ("kimi-k3", _MODEL),
        (7, _MODEL),
        (_MODEL, None),
        (_MODEL, "kimi-k3.1-pro-highspeed"),
        (_MODEL, "kimi-k3"),
        (_MODEL, 7),
    ],
)
def test_kimi_parent_refuses_unproven_configured_or_served_model_identity(
    monkeypatch: pytest.MonkeyPatch,
    configured_model: Any,
    served_model: Any,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(
            configured_model=configured_model,
            served_model=served_model,
        )
    )

    result = backend.invoke(_invocation())

    assert result.error is not None
    assert result.error.kind == "transport"
    assert result.error.retriable is False
    assert "exact configured model identity" in result.error.message
    assert result.final_message == ""
    assert result.completeness.terminal_complete is False


def test_kimi_prompt_abi_v2_reuses_exactly_one_canonical_schema_contract() -> None:
    from assay.backend.features import BackendFeatureSet
    from assay.primitives.prompt_abi import (
        PromptABIProfile,
        PromptOutputContract,
        render_prompt_abi_v2,
    )

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    schema_json = kimi_mod._schema_json(schema)
    assert schema_json is not None
    system = render_prompt_abi_v2(
        "Return one qualification result.",
        profile=PromptABIProfile(
            overlay="kimi",
            features=BackendFeatureSet(
                structured_output="prompt_only",
                tool_loop="sdk_owned",
            ),
        ),
        output=PromptOutputContract(response_schema=schema),
    )
    request = kimi_mod._WorkerRequest(
        system=system,
        user="task",
        schema_json=schema_json,
    )

    assert kimi_mod._prompt_system_text(request) == system
    assert worker_mod._system_text(request) == system
    assert system.count("## Required output contract") == 1
    assert system.count(schema_json) == 1
    assert "CRUCIBLE OUTPUT CONTRACT:" not in system


def test_kimi_prompt_abi_v2_fails_closed_on_mismatched_schema() -> None:
    expected = '{"type":"object"}'
    mismatched_system = (
        "<!-- assay.prompt.v2 -->\n"
        "## Required output contract\n\n"
        "```json\n"
        '{"type":"array"}'
        "\n```\n"
    )
    request = kimi_mod._WorkerRequest(
        system=mismatched_system,
        user="task",
        schema_json=expected,
    )

    with pytest.raises(
        KimiSDKQualificationError,
        match="exactly one matching canonical response schema",
    ):
        kimi_mod._prompt_system_text(request)
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="exactly one matching canonical response schema",
    ):
        worker_mod._system_text(request)


def test_kimi_legacy_prompt_retains_worker_output_contract_append() -> None:
    request = kimi_mod._WorkerRequest(
        system="LEGACY SYSTEM",
        user="task",
        schema_json='{"type":"object"}',
    )

    parent = kimi_mod._prompt_system_text(request)
    production = worker_mod._system_text(request)

    assert parent == production
    assert parent.count("CRUCIBLE OUTPUT CONTRACT:") == 1
    assert parent.endswith('{"type":"object"}')


def test_kimi_missing_runtime_credentials_never_starts_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return _success_outcome()

    backend = KimiBackend(_worker_runner=runner)
    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    with pytest.raises(
        KimiSDKQualificationError,
        match="KIMI_BASE_URL",
    ):
        backend.invoke(_invocation())
    assert called is False


def test_kimi_rejects_embedded_endpoint_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    backend = KimiBackend(_worker_runner=lambda *_a, **_k: _success_outcome())
    monkeypatch.setenv(
        "KIMI_BASE_URL",
        "https://user:password@inference.example.test/v1",
    )
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    with pytest.raises(
        KimiSDKQualificationError,
        match="without embedded credentials",
    ):
        backend.invoke(_invocation())


@pytest.mark.parametrize(
    "base_url",
    [
        "http://inference.example.test/v1",
        "http://localhost:8765/v1",
        "http://[::1]:8765/v1",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:8765/v1?route=dev",
        "http://127.0.0.1:8765/v1#fragment",
    ],
)
def test_kimi_rejects_unencrypted_nonexact_inference_routes(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    with pytest.raises(KimiSDKQualificationError):
        kimi_mod._validate_runtime_credentials()


def test_kimi_accepts_exact_http_ipv4_loopback_test_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "http://127.0.0.1:8765/v1"
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    assert kimi_mod._validate_runtime_credentials() == (base_url, _TOKEN)
    assert kimi_mod._normalize_base_url(base_url, setting="test") == base_url


@pytest.mark.parametrize("codepoint", [*range(0x20), 0x7F])
def test_kimi_rejects_every_ascii_control_character_in_inference_token(
    monkeypatch: pytest.MonkeyPatch,
    codepoint: int,
) -> None:
    token = f"prefix{chr(codepoint)}suffix"
    monkeypatch.setattr(
        kimi_mod.os,
        "environ",
        {
            "KIMI_BASE_URL": _BASE_URL,
            "MOONSHOT_API_KEY": token,
        },
    )

    with pytest.raises(
        KimiSDKQualificationError,
        match="header-safe printable ASCII",
    ):
        kimi_mod._validate_runtime_credentials()


@pytest.mark.parametrize(
    "token",
    [
        " exact-token",
        "exact-token ",
        "exact token",
        "exact\u00a0token",
        "tökén",
    ],
)
def test_kimi_rejects_inference_token_whitespace_or_non_ascii(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", token)

    with pytest.raises(
        KimiSDKQualificationError,
        match="header-safe printable ASCII",
    ):
        kimi_mod._validate_runtime_credentials()


def test_kimi_rejects_oversized_inference_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv(
        "MOONSHOT_API_KEY",
        "t" * (kimi_mod._MAX_AUTH_TOKEN_CHARS + 1),
    )

    with pytest.raises(KimiSDKQualificationError, match="valid bounded"):
        kimi_mod._validate_runtime_credentials()


def test_kimi_expected_endpoint_is_compared_after_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[kimi_mod._WorkerRequest] = []

    def runner(request, _timeout, **_kwargs):
        captured.append(request)
        return _success_outcome()

    backend = KimiBackend(
        expected_base_url="HTTPS://API.MOONSHOT.AI/v1/",
        _worker_runner=runner,
    )
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    assert backend.invoke(_invocation()).error is None
    assert captured[0].base_url == _BASE_URL

    monkeypatch.setenv(
        "KIMI_BASE_URL",
        "https://other-inference.example.test/v1",
    )
    with pytest.raises(
        KimiSDKQualificationError,
        match="does not match the configured",
    ):
        backend.invoke(_invocation())


def test_kimi_worker_python_precedence_and_environment_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from_env = tmp_path / "from-env" / "python"
    explicit = tmp_path / "explicit" / "python"
    monkeypatch.setenv(
        "CRUCIBLE_KIMI_WORKER_PYTHON",
        str(from_env),
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-aws-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-github-token")
    for name, value in _HOSTILE_KIMI_PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    monkeypatch.setenv("PATH", "/hostile/path")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert.pem")

    assert kimi_mod._resolve_worker_python() == from_env
    assert kimi_mod._resolve_worker_python(explicit) == explicit
    child = kimi_mod._worker_environment()
    assert child["LANG"] == "C.UTF-8"
    assert child["SSL_CERT_FILE"] == "/etc/ssl/cert.pem"
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "PATH" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert "ANTHROPIC_API_KEY" not in child
    assert "GITHUB_TOKEN" not in child
    for name in _HOSTILE_KIMI_PROVIDER_ENV:
        assert name not in child
    assert "MOONSHOT_API_KEY" not in child
    assert "CRUCIBLE_KIMI_WORKER_PYTHON" not in child

    with pytest.raises(
        KimiSDKQualificationError,
        match="absolute path",
    ):
        kimi_mod._resolve_worker_python("relative/python")


def test_project_locked_worker_rejects_every_path_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_KIMI_WORKER_PYTHON",
        str(tmp_path / "ambient-python"),
    )
    with pytest.raises(
        KimiSDKQualificationError,
        match="rejects ambient CRUCIBLE_KIMI_WORKER_PYTHON",
    ):
        KimiBackend(worker_source="project_locked")

    monkeypatch.delenv("CRUCIBLE_KIMI_WORKER_PYTHON")
    with pytest.raises(
        KimiSDKQualificationError,
        match="rejects an explicit worker_python override",
    ):
        KimiBackend(
            worker_source="project_locked",
            worker_python=tmp_path / "explicit-python",
        )


def test_project_locked_worker_requires_exact_expected_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pin = SimpleNamespace(runtime_fingerprint="sha256:" + "a" * 64)
    monkeypatch.delenv("CRUCIBLE_KIMI_WORKER_PYTHON", raising=False)
    monkeypatch.setattr(
        kimi_mod._PinnedKimiWorkerRuntime,
        "capture",
        lambda _path: pin,
    )

    backend = KimiBackend(
        worker_source="project_locked",
        expected_worker_runtime_fingerprint=pin.runtime_fingerprint,
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(),
    )
    assert backend.worker_runtime_fingerprint == pin.runtime_fingerprint

    with pytest.raises(
        KimiSDKQualificationError,
        match="does not match the configured expected fingerprint",
    ):
        KimiBackend(
            worker_source="project_locked",
            expected_worker_runtime_fingerprint="sha256:" + "b" * 64,
        )


def _synthetic_worker_project(root: Path) -> Path:
    project = root / "worker"
    (project / ".venv").mkdir(parents=True)
    (project / "src/crucible_kimi_worker").mkdir(parents=True)
    (project / ".venv/lib/python3.12/site-packages/demo").mkdir(parents=True)
    files = {
        "pyproject.toml": "[project]\nname='worker'\n",
        "uv.lock": "version = 1\n",
        ".python-version": "3.12\n",
        ".venv/pyvenv.cfg": "version = 3.12\n",
        "src/crucible_kimi_worker/__init__.py": "VALUE = 1\n",
        ".venv/lib/python3.12/site-packages/demo/__init__.py": "VALUE = 2\n",
    }
    for relative, content in files.items():
        (project / relative).write_text(content, encoding="utf-8")
    return project


@pytest.mark.parametrize(
    "relative",
    [
        "uv.lock",
        "src/crucible_kimi_worker/__init__.py",
        ".venv/lib/python3.12/site-packages/demo/__init__.py",
    ],
)
def test_project_locked_runtime_fingerprint_binds_lock_source_and_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative: str,
) -> None:
    project = _synthetic_worker_project(tmp_path)
    monkeypatch.setattr(kimi_mod, "_WORKER_PROJECT", project)
    original = kimi_mod._worker_runtime_fingerprint()

    path = project / relative
    path.write_text(
        path.read_text(encoding="utf-8") + "# drift\n",
        encoding="utf-8",
    )

    assert kimi_mod._worker_runtime_fingerprint() != original


def test_pinned_worker_hashes_the_invoked_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    importing_project = _synthetic_worker_project(tmp_path / "importing")
    invoked_project = _synthetic_worker_project(tmp_path / "invoked")
    invoked_source = invoked_project / "src/crucible_kimi_worker/__init__.py"
    invoked_source.write_text("VALUE = 3\n", encoding="utf-8")
    target = tmp_path / "python-real"
    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o700)
    invocation = invoked_project / ".venv/bin/python"
    invocation.parent.mkdir(parents=True)
    invocation.symlink_to(target)
    monkeypatch.setattr(kimi_mod, "_WORKER_PROJECT", importing_project)

    pin = kimi_mod._PinnedKimiWorkerRuntime.capture(invocation)

    assert pin.installed_tree_fingerprint == kimi_mod._worker_runtime_fingerprint(
        invoked_project
    )
    assert pin.installed_tree_fingerprint != kimi_mod._worker_runtime_fingerprint(
        importing_project
    )
    pin.revalidate()


def test_pinned_worker_revalidates_symlink_interpreter_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-real"
    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o700)
    invocation = tmp_path / "worker/.venv/bin/python"
    invocation.parent.mkdir(parents=True)
    invocation.symlink_to(target)
    runtime = {"value": "sha256:" + "a" * 64}
    monkeypatch.setattr(
        kimi_mod,
        "_worker_runtime_fingerprint",
        lambda _project=None: runtime["value"],
    )
    pin = kimi_mod._PinnedKimiWorkerRuntime.capture(invocation)
    pin.revalidate()

    replacement = tmp_path / "python-replacement"
    replacement.write_bytes(target.read_bytes())
    replacement.chmod(0o700)
    invocation.unlink()
    invocation.symlink_to(replacement)
    with pytest.raises(KimiSDKQualificationError, match="identity drifted"):
        pin.revalidate()

    invocation.unlink()
    invocation.symlink_to(target)
    pin = kimi_mod._PinnedKimiWorkerRuntime.capture(invocation)
    target.write_bytes(b"#!/bin/sh\nexit 1\n")
    with pytest.raises(KimiSDKQualificationError, match="identity drifted"):
        pin.revalidate()

    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o700)
    pin = kimi_mod._PinnedKimiWorkerRuntime.capture(invocation)
    runtime["value"] = "sha256:" + hashlib.sha256(b"drift").hexdigest()
    with pytest.raises(KimiSDKQualificationError, match="identity drifted"):
        pin.revalidate()


def test_worker_pin_drift_prevents_probe_and_child_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DriftedPin:
        runtime_fingerprint = "sha256:" + "a" * 64

        def revalidate(self) -> None:
            raise KimiSDKQualificationError("injected worker drift")

    monkeypatch.setattr(
        kimi_mod,
        "_qualify_worker_runtime",
        lambda *_args: pytest.fail("probe must not start"),
    )
    monkeypatch.setattr(
        kimi_mod.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )

    outcome = kimi_mod._run_isolated_worker_subprocess(
        kimi_mod._WorkerRequest(
            system="system",
            user="user",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
        ),
        1.0,
        worker_python=tmp_path / "python",
        worker_pin=DriftedPin(),
    )

    assert outcome.error_kind == "qualification"
    assert "injected worker drift" in outcome.error_message


def test_worker_pin_revalidates_again_immediately_before_popen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"

    class SecondCheckDrifts:
        runtime_fingerprint = "sha256:" + "a" * 64

        def __init__(self) -> None:
            self.calls = 0

        def revalidate(self) -> None:
            self.calls += 1
            if self.calls == 2:
                raise KimiSDKQualificationError("drift before Popen")

    pin = SecondCheckDrifts()
    monkeypatch.setattr(
        kimi_mod,
        "_qualify_worker_runtime",
        lambda _path, _fingerprint: None,
    )
    monkeypatch.setattr(
        kimi_mod,
        "_make_private_temp_root",
        lambda: (root.mkdir(mode=0o700), root)[1],
    )
    monkeypatch.setattr(
        kimi_mod.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )

    with pytest.raises(KimiSDKQualificationError, match="drift before Popen"):
        kimi_mod._run_isolated_worker_subprocess(
            kimi_mod._WorkerRequest(
                system="system",
                user="user",
                base_url=_BASE_URL,
                auth_token=_TOKEN,
            ),
            1.0,
            worker_python=tmp_path / "python",
            worker_pin=pin,
        )

    assert pin.calls == 2
    assert root.exists() is False


def test_kimi_rejects_unbounded_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        _worker_runner=lambda *_args, **_kwargs: pytest.fail("worker must not start")
    )
    with pytest.raises(ValueError, match="finite number > 0"):
        BackendPolicy(
            retry_budget=1,
            timeout_seconds=float("inf"),
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        )
    backend.close()


@pytest.mark.parametrize(
    "timeout_seconds",
    [True, 0, -1, float("inf"), float("nan")],
)
def test_kimi_route_default_timeout_is_closed(
    timeout_seconds: Any,
) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        KimiBackend(timeout_seconds=timeout_seconds)


def test_kimi_route_timeout_is_default_and_invocation_policy_is_tightest_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    observed: list[float] = []

    def runner(_request, timeout, **_kwargs):
        observed.append(timeout)
        return _success_outcome()

    backend = KimiBackend(timeout_seconds=1_200, _worker_runner=runner)
    backend.invoke(replace(_invocation(), policy=None))
    backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=1,
                timeout_seconds=3.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )
    tighter_route = KimiBackend(
        timeout_seconds=3,
        _worker_runner=runner,
    )
    tighter_route.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=1,
                timeout_seconds=1_200.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert observed[0] == pytest.approx(1_200, abs=1.0)
    assert observed[1] == pytest.approx(3.0, abs=1.0)
    assert observed[2] == pytest.approx(3.0, abs=1.0)


@dataclass
class _FakeTokenUsage:
    input_other: int
    output: int
    input_cache_creation: int = 0
    input_cache_read: int = 0

    @property
    def input(self) -> int:
        return self.input_other + self.input_cache_creation + self.input_cache_read


class _UsageIdSubclass(str):
    pass


@dataclass
class _FakeTextPart:
    text: str


@dataclass
class _FakeThinkPart:
    think: str


@dataclass
class _FakeStatusUpdate:
    token_usage: _FakeTokenUsage | None = None
    message_id: str | None = None


class _FakeTurnEnd:
    pass


class _FakeStepBegin:
    pass


class _FakeToolCall:
    def __init__(self, tool_call_id: str, name: str) -> None:
        self.id = tool_call_id
        self.function = SimpleNamespace(name=name)


class _FakeToolResult:
    def __init__(
        self,
        tool_call_id: str,
        *,
        error_text: str | None = None,
    ) -> None:
        self.tool_call_id = tool_call_id
        self.return_value = SimpleNamespace(
            is_error=error_text is not None,
            output=(
                []
                if error_text is None
                else [SimpleNamespace(text=error_text)]
            ),
        )


class _FakeStepInterrupted:
    pass


class _FakeApprovalRequest:
    def __init__(
        self,
        *,
        tool_call_id: str = "",
        sender: str = "",
        action: str = "",
    ) -> None:
        self.tool_call_id = tool_call_id
        self.sender = sender
        self.action = action
        self.responses: list[str] = []

    def resolve(self, response: str) -> None:
        self.responses.append(response)


class _FakeSDKError(Exception):
    pass


class _FakeAPIStatusError(_FakeSDKError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeKaosPath(str):
    pass


class _FakeAsyncOpenAI:
    def __init__(
        self,
        *,
        max_retries: int = 2,
        with_options_calls: list[int] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.with_options_calls = (
            with_options_calls if with_options_calls is not None else []
        )

    def with_options(self, *, max_retries: int) -> _FakeAsyncOpenAI:
        self.with_options_calls.append(max_retries)
        return type(self)(
            max_retries=max_retries,
            with_options_calls=self.with_options_calls,
        )


_FakeAsyncOpenAI.__name__ = "AsyncOpenAI"
_FakeAsyncOpenAI.__module__ = "openai"


class _FakeKimiSoul:
    async def compact_context(self) -> None:
        raise AssertionError("qualified tests must replace automatic compaction")


_FakeKimiSoul.__name__ = "KimiSoul"
_FakeKimiSoul.__module__ = "kimi_cli.soul.kimisoul"
_FakeKimiSoul.compact_context.__module__ = "kimi_cli.soul.kimisoul"
_FakeKimiSoul.compact_context.__qualname__ = "KimiSoul.compact_context"


class _FakeSession:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.id = "qualified-session-id"
        self.model_name = _MODEL
        self.status = SimpleNamespace(yolo_enabled=False)
        self.prompt_calls: list[tuple[str, bool]] = []
        self.cancel_calls = 0
        self.close_calls = 0
        provider_type = type("Kimi", (), {})
        provider_type.__module__ = "kosong.chat_provider.kimi"
        self.provider = provider_type()
        self.provider.stream = True
        self.provider.client = _FakeAsyncOpenAI()
        self.provider.thinking_effort = None
        self.provider._generation_kwargs = {"prompt_cache_key": self.id}
        self.provider._crucible_last_stream = SimpleNamespace(
            finish_reason="stop",
            served_model=_MODEL,
        )
        soul = _FakeKimiSoul()
        soul._runtime = SimpleNamespace(
            llm=SimpleNamespace(chat_provider=self.provider)
        )
        self._cli = SimpleNamespace(soul=soul)

    async def prompt(
        self,
        prompt: str,
        *,
        merge_wire_messages: bool,
    ):
        self.prompt_calls.append((prompt, merge_wire_messages))
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event

    def cancel(self) -> None:
        self.cancel_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def _fake_bindings(
    session: _FakeSession,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    config_docs: list[dict[str, Any]] = []
    create_calls: list[dict[str, Any]] = []

    class Config:
        @classmethod
        def model_validate(cls, value):
            config_docs.append(value)
            return value

    class SessionAPI:
        @staticmethod
        async def create(**kwargs):
            create_calls.append(kwargs)
            return session

    bindings = worker_mod._Bindings(
        Session=SessionAPI,
        Config=Config,
        KaosPath=_FakeKaosPath,
        ApprovalRequest=_FakeApprovalRequest,
        StatusUpdate=_FakeStatusUpdate,
        TextPart=_FakeTextPart,
        ThinkPart=_FakeThinkPart,
        ToolCall=_FakeToolCall,
        ToolResult=_FakeToolResult,
        StepBegin=_FakeStepBegin,
        TurnEnd=_FakeTurnEnd,
        StepInterrupted=_FakeStepInterrupted,
        RunCancelled=_FakeSDKError,
        MaxStepsReached=_FakeSDKError,
        APIStatusError=_FakeAPIStatusError,
        APIConnectionError=_FakeSDKError,
        APITimeoutError=_FakeSDKError,
        APIEmptyResponseError=_FakeSDKError,
        ChatProviderError=_FakeSDKError,
    )
    return bindings, config_docs, create_calls


def _worker_paths(root: Path) -> Any:
    paths = worker_mod._Paths(
        root=root,
        home=root / "home",
        share=root / "share",
        work=root / "work",
        skills=root / "skills",
        config=root / "config",
        scratch=root / "tmp",
    )
    for path in (
        paths.home,
        paths.share,
        paths.work,
        paths.skills,
        paths.config,
        paths.scratch,
    ):
        path.mkdir(parents=True)
    return paths


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stream", False, "unqualified chat provider"),
        (
            "thinking_effort",
            object(),
            "unsupported reasoning effort",
        ),
        (
            "_generation_kwargs",
            {},
            "unqualified Moonshot generation options",
        ),
        (
            "_generation_kwargs",
            {
                "prompt_cache_key": "qualified-session-id",
                "max_tokens": 1,
            },
            "unqualified Moonshot generation options",
        ),
    ],
)
def test_kimi_provider_requires_exact_native_generation_shape(
    qualifier: Any,
    error_type: type[Exception],
    field: str,
    value: Any,
    message: str,
) -> None:
    session = _FakeSession([])
    setattr(session.provider, field, value)

    with pytest.raises(error_type, match=message):
        qualifier(session, max_response_tokens=32_768)


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
@pytest.mark.parametrize(
    ("provider_name", "provider_module"),
    [
        ("OpenAILegacy", "kosong.chat_provider.kimi"),
        ("Kimi", "kosong.contrib.chat_provider.openai_legacy"),
    ],
)
def test_kimi_provider_requires_exact_native_type(
    qualifier: Any,
    error_type: type[Exception],
    provider_name: str,
    provider_module: str,
) -> None:
    session = _FakeSession([])
    provider_type = type(provider_name, (), {})
    provider_type.__module__ = provider_module
    replacement = provider_type()
    replacement.stream = session.provider.stream
    replacement.client = session.provider.client
    replacement.thinking_effort = session.provider.thinking_effort
    replacement._generation_kwargs = session.provider._generation_kwargs
    session._cli.soul._runtime.llm.chat_provider = replacement

    with pytest.raises(error_type, match="unqualified chat provider"):
        qualifier(session, max_response_tokens=32_768)


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
@pytest.mark.parametrize(
    ("client_name", "client_module", "max_retries", "with_options"),
    [
        ("OpenAI", "openai", 2, lambda **_kwargs: None),
        ("AsyncOpenAI", "openai._client", 2, lambda **_kwargs: None),
        ("AsyncOpenAI", "openai", 1, lambda **_kwargs: None),
        ("AsyncOpenAI", "openai", 2, None),
    ],
)
def test_kimi_provider_requires_exact_native_http_client(
    qualifier: Any,
    error_type: type[Exception],
    client_name: str,
    client_module: str,
    max_retries: int,
    with_options: Any,
) -> None:
    session = _FakeSession([])
    client_type = type(client_name, (), {})
    client_type.__module__ = client_module
    client = client_type()
    client.max_retries = max_retries
    client.with_options = with_options
    session.provider.client = client

    with pytest.raises(error_type, match="unqualified Moonshot HTTP client"):
        qualifier(session, max_response_tokens=32_768)


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
def test_kimi_provider_fails_if_hidden_retries_remain_enabled(
    qualifier: Any,
    error_type: type[Exception],
) -> None:
    session = _FakeSession([])
    original_client = session.provider.client
    original_client.with_options = lambda **_kwargs: original_client

    with pytest.raises(error_type, match="could not disable hidden"):
        qualifier(session, max_response_tokens=32_768)


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
def test_kimi_provider_disables_unaccounted_auto_compaction(
    qualifier: Any,
    error_type: type[Exception],
) -> None:
    session = _FakeSession([])

    qualifier(session, max_response_tokens=32_768)

    with pytest.raises(
        error_type,
        match="automatic context compaction is disabled",
    ):
        asyncio.run(session._cli.soul.compact_context())


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
def test_kimi_provider_requires_exact_auto_compaction_boundary(
    qualifier: Any,
    error_type: type[Exception],
) -> None:
    session = _FakeSession([])
    session._cli.soul.compact_context = lambda: None

    with pytest.raises(error_type, match="unqualified compaction boundary"):
        qualifier(session, max_response_tokens=32_768)


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_pinned_native_stream_keeps_interleaved_parallel_tool_calls_separate(
    locked_worker_launch: Any,
) -> None:
    result, requests = _run_pinned_native_stream(
        locked_worker_launch,
        [
            _native_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-first",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"fir',
                            },
                        },
                        {
                            "index": 1,
                            "id": "call-second",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"sec',
                            },
                        },
                    ]
                }
            ),
            _native_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {"arguments": 'ond"}'},
                        },
                        {
                            "index": 0,
                            "function": {"arguments": 'st"}'},
                        },
                    ]
                }
            ),
            _native_chunk(finish_reason="tool_calls"),
            _native_usage_chunk(),
        ]
    )

    assert result == {
        "error": None,
        "error_type": None,
        "finish_reason": "tool_calls",
        "parts": [
            {
                "arguments": '{"value":"first"}',
                "id": "call-first",
                "name": "echo",
                "type": "ToolCall",
            },
            {
                "arguments": '{"value":"second"}',
                "id": "call-second",
                "name": "echo",
                "type": "ToolCall",
            },
        ],
        "served_model": _MODEL,
        "usage": {
            "input": 13,
            "input_cache_creation": 0,
            "input_cache_read": 2,
            "input_other": 11,
            "output": 4,
        },
    }
    assert len(requests) == 1
    assert requests[0]["model"] == _MODEL
    assert requests[0]["stream_options"] == {"include_usage": True}


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_pinned_native_stream_retains_length_finish_reason(
    locked_worker_launch: Any,
) -> None:
    result, _ = _run_pinned_native_stream(
        locked_worker_launch,
        [
            _native_chunk(delta={"content": "truncated visible response"}),
            _native_chunk(finish_reason="length"),
            _native_usage_chunk(),
        ]
    )

    assert result == {
        "error": None,
        "error_type": None,
        "finish_reason": "length",
        "parts": [
            {
                "text": "truncated visible response",
                "type": "TextPart",
            }
        ],
        "served_model": _MODEL,
        "usage": {
            "input": 13,
            "input_cache_creation": 0,
            "input_cache_read": 2,
            "input_other": 11,
            "output": 4,
        },
    }


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_pinned_native_stream_preserves_terminal_only_evidence(
    finish_reason: str,
    locked_worker_launch: Any,
) -> None:
    result, _ = _run_pinned_native_stream(
        locked_worker_launch,
        [
            _native_chunk(finish_reason=finish_reason),
            _native_usage_chunk(),
        ]
    )

    assert result == {
        "error": None,
        "error_type": None,
        "finish_reason": finish_reason,
        "parts": [
            {
                "text": "",
                "type": "TextPart",
            }
        ],
        "served_model": _MODEL,
        "usage": {
            "input": 13,
            "input_cache_creation": 0,
            "input_cache_read": 2,
            "input_other": 11,
            "output": 4,
        },
    }


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_pinned_native_stream_discards_partial_tool_on_nonstop_terminal(
    finish_reason: str,
    locked_worker_launch: Any,
) -> None:
    result, _ = _run_pinned_native_stream(
        locked_worker_launch,
        [
            _native_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-must-not-run",
                            "type": "function",
                            "function": {
                                "name": "echo",
                                "arguments": '{"value":"partial',
                            },
                        }
                    ]
                }
            ),
            _native_chunk(finish_reason=finish_reason),
            _native_usage_chunk(),
        ]
    )

    assert [part["type"] for part in result["parts"]] == ["TextPart"]
    assert result["parts"][0]["text"] == ""
    assert result["finish_reason"] == finish_reason
    assert result["served_model"] == _MODEL
    assert result["usage"] == {
        "input": 13,
        "input_cache_creation": 0,
        "input_cache_read": 2,
        "input_other": 11,
        "output": 4,
    }


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        (
            [
                _native_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": -1,
                                "id": "call-invalid",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                ),
                _native_chunk(finish_reason="tool_calls"),
            ],
            "invalid tool-call index",
        ),
        (
            [
                _native_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call-sparse",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                ),
                _native_chunk(finish_reason="tool_calls"),
                # This rejection follows the stream, so the K3-series final
                # statistics packet must have been accepted first.
                _native_usage_chunk(),
            ],
            "sparse tool-call index set",
        ),
        (
            [
                _native_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-original",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": "{",
                                },
                            }
                        ]
                    }
                ),
                _native_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-conflict",
                                "function": {"arguments": "}"},
                            }
                        ]
                    }
                ),
                _native_chunk(finish_reason="tool_calls"),
            ],
            "changed a tool-call id",
        ),
    ],
)
def test_pinned_native_stream_rejects_malformed_tool_call_indices(
    chunks: list[dict[str, Any]],
    message: str,
    locked_worker_launch: Any,
) -> None:
    result, _ = _run_pinned_native_stream(locked_worker_launch, chunks)

    assert result["error_type"] == "WorkerQualificationError"
    assert message in result["error"]


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_pinned_native_stream_rejects_served_model_mismatch(
    locked_worker_launch: Any,
) -> None:
    result, _ = _run_pinned_native_stream(
        locked_worker_launch,
        [
            _native_chunk(
                delta={"content": "untrusted response"},
                model=f"{_MODEL}-substitute",
            ),
            _native_chunk(
                finish_reason="stop",
                model=f"{_MODEL}-substitute",
            ),
            _native_usage_chunk(model=f"{_MODEL}-substitute"),
        ]
    )

    assert result["error_type"] == "WorkerQualificationError"
    assert "did not prove the exact served model" in result["error"]


@pytest.mark.parametrize(
    ("qualifier", "error_type"),
    [
        pytest.param(
            worker_mod._qualify_moonshot_provider,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._qualify_moonshot_provider,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
@pytest.mark.parametrize("session_id", [None, "", "bad\nid", "s" * 257])
def test_kimi_provider_requires_bounded_session_cache_identity(
    qualifier: Any,
    error_type: type[Exception],
    session_id: Any,
) -> None:
    session = _FakeSession([])
    session.id = session_id

    with pytest.raises(error_type, match="bounded session cache identity"):
        qualifier(session, max_response_tokens=32_768)


@pytest.mark.parametrize(
    "extractor",
    [
        pytest.param(
            worker_mod._extract_final_text,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._extract_final_text,
            id="parent-mirror",
        ),
    ],
)
def test_kimi_preserves_ordinary_final_text_byte_for_byte(
    extractor: Any,
) -> None:
    visible = "  <thinking> is an ordinary word here.\n&lt;/think&gt; stays escaped.  "

    assert extractor(visible) == visible


@pytest.mark.parametrize(
    "text",
    [
        "private reasoning</think>",
        "</think>visible final",
        "private reasoning</THINK>visible final",
        "private reasoning</ think >visible final",
        "private reasoning</think>visible</think>tail",
        "<think>private reasoning</think>",
        "prefix <think>private reasoning</think>visible final",
        "<think>private reasoning</think attr>visible final",
        "<think>private reasoning",
        "private reasoning</think",
    ],
)
@pytest.mark.parametrize(
    ("extractor", "error_type"),
    [
        pytest.param(
            worker_mod._extract_final_text,
            worker_mod.WorkerQualificationError,
            id="deployed-worker",
        ),
        pytest.param(
            kimi_mod._extract_final_text,
            KimiSDKQualificationError,
            id="parent-mirror",
        ),
    ],
)
def test_kimi_fails_closed_on_any_visible_reasoning_marker(
    extractor: Any,
    error_type: type[Exception],
    text: str,
) -> None:
    with pytest.raises(error_type, match="reasoning marker in visible content"):
        extractor(text)


def test_kimi_session_rejects_split_reasoning_marker_in_text_parts(
    tmp_path: Path,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("private reasoning</thi"),
            _FakeTextPart("nk>visible final"),
            _FakeTurnEnd(),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="reasoning marker in visible content",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )
    assert session.close_calls == 1


@pytest.mark.parametrize(
    (
        "finish_reason",
        "expected_final",
        "expected_stop",
        "expected_error",
    ),
    [
        (
            "length",
            "partial billed response",
            "max_tokens",
            "max_tokens",
        ),
        (
            "content_filter",
            "",
            "content_filter",
            "content_rejected",
        ),
    ],
)
def test_kimi_worker_normalizes_nonstop_terminal_finish_reasons(
    tmp_path: Path,
    finish_reason: str,
    expected_final: str,
    expected_stop: str,
    expected_error: str,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("partial billed response"),
            _FakeStatusUpdate(
                _FakeTokenUsage(
                    input_other=11,
                    output=4,
                    input_cache_read=2,
                ),
                message_id="message-terminal",
            ),
            _FakeTurnEnd(),
        ]
    )
    session.provider._crucible_last_stream.finish_reason = finish_reason
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            lambda _payload: None,
        )
    )

    assert outcome.final_message == expected_final
    assert outcome.stop_reason == expected_stop
    assert outcome.error_kind == expected_error
    assert outcome.retriable is False
    assert outcome.input_tokens == 13
    assert outcome.output_tokens == 4
    assert outcome.cache_read_tokens == 2
    assert outcome.usage_seen is True
    assert outcome.usage_valid is True
    assert outcome.terminal_seen is True
    assert outcome.configured_model == _MODEL
    assert outcome.served_model == _MODEL
    assert session.close_calls == 1


def test_kimi_worker_rejects_unqualified_terminal_finish_reason(
    tmp_path: Path,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("visible"),
            _FakeTurnEnd(),
        ]
    )
    session.provider._crucible_last_stream.finish_reason = "function_call"
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="unqualified finish reason",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )
    assert session.close_calls == 1


def test_kimi_session_uses_confined_prompt_only_sdk_surface(
    tmp_path: Path,
) -> None:
    raw_reasoning = "private ThinkPart that must never leave the worker"
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart('{"ok":'),
            _FakeThinkPart(raw_reasoning),
            _FakeTextPart("true}"),
            _FakeStatusUpdate(
                _FakeTokenUsage(
                    input_other=10,
                    output=7,
                    input_cache_creation=2,
                    input_cache_read=3,
                ),
                message_id="message-1",
            ),
            # A repeated status for one message replaces rather than doubles
            # the provider's cumulative step snapshot.
            _FakeStatusUpdate(
                _FakeTokenUsage(
                    input_other=10,
                    output=7,
                    input_cache_creation=2,
                    input_cache_read=3,
                ),
                message_id="message-1",
            ),
            _FakeTurnEnd(),
        ]
    )
    bindings, config_docs, create_calls = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    emitted: list[dict[str, Any]] = []
    request = worker_mod._Request(
        root=paths.root,
        system="SYSTEM ${HOSTILE_TEMPLATE_NAME}",
        user="/hostile-slash-command",
        schema_json='{"type":"object"}',
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=1,
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            emitted.append,
        )
    )

    assert outcome.final_message == '{"ok":true}'
    assert outcome.input_tokens == 15
    assert outcome.output_tokens == 7
    assert outcome.cache_creation_tokens == 2
    assert outcome.cache_read_tokens == 3
    assert outcome.usage_seen is True
    assert outcome.usage_valid is True
    assert outcome.terminal_seen is True
    assert outcome.steps_used == 1
    assert raw_reasoning not in outcome.final_message
    assert raw_reasoning not in json.dumps(emitted)
    assert session.provider.client.max_retries == 0
    assert session.provider.client.with_options_calls == [0]
    assert session.provider.thinking_effort is None
    assert session.provider._generation_kwargs == {
        "prompt_cache_key": session.id,
        "max_completion_tokens": 32_768,
        "reasoning_effort": "max",
        "extra_body": {
            "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
        },
    }
    assert [
        event["steps_used"] for event in emitted if event["type"] == "step_progress"
    ] == [1]
    assert session.close_calls == 1
    assert session.prompt_calls == [
        (
            "Complete the following Crucible task:\n\n/hostile-slash-command",
            False,
        )
    ]

    config = config_docs[0]
    assert config["default_model"] == _MODEL
    assert config["default_thinking"] is True
    assert config["default_yolo"] is False
    assert config["providers"][worker_mod._PROVIDER_NAME] == {
        "type": "kimi",
        "base_url": _BASE_URL,
        "api_key": _TOKEN,
    }
    assert config["models"][_MODEL]["model"] == _MODEL
    assert config["models"][_MODEL]["capabilities"] == []
    assert config["loop_control"] == {
        "max_steps_per_turn": 1,
        "max_retries_per_step": 1,
        "max_ralph_iterations": 0,
    }

    create_call = create_calls[0]
    assert create_call["model"] == _MODEL
    assert create_call["thinking"] is True
    assert create_call["yolo"] is False
    assert create_call["mcp_configs"] == []
    assert create_call["skills_dir"] == str(paths.skills)
    assert create_call["max_retries_per_step"] == 1
    agent_doc = json.loads(Path(create_call["agent_file"]).read_text())
    agent = agent_doc["agent"]
    assert agent["tools"] == []
    assert agent["subagents"] == {}
    assert (
        "${HOSTILE_TEMPLATE_NAME}" in (agent["system_prompt_args"]["CRUCIBLE_SYSTEM"])
    )
    assert (
        "CRUCIBLE OUTPUT CONTRACT" in (agent["system_prompt_args"]["CRUCIBLE_SYSTEM"])
    )
    assert Path(agent["system_prompt_path"]).name == "system.md"


def test_worker_accounts_reused_response_id_separately_across_steps(
    tmp_path: Path,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeStatusUpdate(
                _FakeTokenUsage(input_other=5, output=2),
                message_id="response-reused",
            ),
            _FakeStepBegin(),
            _FakeTextPart("OK"),
            _FakeStatusUpdate(
                _FakeTokenUsage(input_other=7, output=3),
                message_id="response-reused",
            ),
            _FakeTurnEnd(),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            lambda _payload: None,
        )
    )

    assert outcome.input_tokens == 12
    assert outcome.output_tokens == 5
    assert outcome.usage_valid is True
    assert outcome.steps_used == 2


@pytest.mark.parametrize(
    "accumulator_type",
    [
        pytest.param(
            kimi_mod._UsageAccumulator,
            id="parent-mirror",
        ),
        pytest.param(
            worker_mod._UsageAccumulator,
            id="deployed-worker",
        ),
    ],
)
def test_usage_accumulator_makes_exact_same_step_repeats_idempotent(
    accumulator_type: Any,
) -> None:
    accumulator = accumulator_type()
    sample = SimpleNamespace(
        input=13,
        output=4,
        input_cache_creation=0,
        input_cache_read=2,
    )

    accumulator.observe(sample, "response-stable", step=1)
    accumulator.observe(sample, "response-stable", step=1)

    assert accumulator.totals() == (13, 4, 0, 2)
    assert accumulator.seen is True
    assert accumulator.valid is True


@pytest.mark.parametrize(
    "accumulator_type",
    [
        pytest.param(
            kimi_mod._UsageAccumulator,
            id="parent-mirror",
        ),
        pytest.param(
            worker_mod._UsageAccumulator,
            id="deployed-worker",
        ),
    ],
)
@pytest.mark.parametrize(
    ("second_id", "second_input"),
    [
        pytest.param("response-stable", 14, id="same-id-conflicting-counters"),
        pytest.param("response-other", 13, id="different-id"),
        pytest.param(None, 13, id="missing-id"),
        pytest.param("", 13, id="empty-id"),
        pytest.param(7, 13, id="non-string-id"),
        pytest.param(
            _UsageIdSubclass("response-subclass"),
            13,
            id="string-subclass-id",
        ),
        pytest.param("bad\nid", 13, id="control-character-id"),
        pytest.param("x" * 257, 13, id="oversized-id"),
    ],
)
def test_usage_accumulator_keeps_first_same_step_sample_on_ambiguity(
    accumulator_type: Any,
    second_id: Any,
    second_input: int,
) -> None:
    accumulator = accumulator_type()
    accumulator.observe(
        SimpleNamespace(
            input=13,
            output=4,
            input_cache_creation=0,
            input_cache_read=2,
        ),
        "response-stable",
        step=1,
    )

    accumulator.observe(
        SimpleNamespace(
            input=second_input,
            output=4,
            input_cache_creation=0,
            input_cache_read=2,
        ),
        second_id,
        step=1,
    )

    assert accumulator.totals() == (13, 4, 0, 2)
    assert accumulator.seen is True
    assert accumulator.valid is False


@pytest.mark.parametrize(
    "accumulator_type",
    [
        pytest.param(
            kimi_mod._UsageAccumulator,
            id="parent-mirror",
        ),
        pytest.param(
            worker_mod._UsageAccumulator,
            id="deployed-worker",
        ),
    ],
)
@pytest.mark.parametrize(
    "message_id",
    [
        None,
        "",
        7,
        _UsageIdSubclass("response-subclass"),
        "bad\nid",
        "x" * 257,
    ],
)
def test_usage_accumulator_accounts_repeated_malformed_id_only_once(
    accumulator_type: Any,
    message_id: Any,
) -> None:
    accumulator = accumulator_type()
    sample = SimpleNamespace(
        input=9,
        output=3,
        input_cache_creation=1,
        input_cache_read=2,
    )

    accumulator.observe(sample, message_id, step=1)
    accumulator.observe(sample, message_id, step=1)

    assert accumulator.totals() == (9, 3, 1, 2)
    assert accumulator.seen is True
    assert accumulator.valid is False


@pytest.mark.parametrize(
    "accumulator_type",
    [
        pytest.param(
            kimi_mod._UsageAccumulator,
            id="parent-mirror",
        ),
        pytest.param(
            worker_mod._UsageAccumulator,
            id="deployed-worker",
        ),
    ],
)
def test_usage_accumulator_counts_reused_id_again_on_later_step(
    accumulator_type: Any,
) -> None:
    accumulator = accumulator_type()
    accumulator.observe(
        SimpleNamespace(
            input=5,
            output=2,
            input_cache_creation=0,
            input_cache_read=0,
        ),
        "response-reused",
        step=1,
    )
    accumulator.observe(
        SimpleNamespace(
            input=7,
            output=3,
            input_cache_creation=0,
            input_cache_read=0,
        ),
        "response-reused",
        step=2,
    )

    assert accumulator.totals() == (12, 5, 0, 0)
    assert accumulator.seen is True
    assert accumulator.valid is True


@pytest.mark.parametrize(
    "accumulator_type",
    [
        pytest.param(
            kimi_mod._UsageAccumulator,
            id="parent-mirror",
        ),
        pytest.param(
            worker_mod._UsageAccumulator,
            id="deployed-worker",
        ),
    ],
)
def test_usage_accumulator_rejects_cache_counters_outside_input_subset(
    accumulator_type: Any,
) -> None:
    accumulator = accumulator_type()

    accumulator.observe(
        SimpleNamespace(
            input=3,
            output=1,
            input_cache_creation=2,
            input_cache_read=2,
        ),
        "response-invalid-cache",
        step=1,
    )

    assert accumulator.totals() == (3, 1, 2, 2)
    assert accumulator.seen is True
    assert accumulator.valid is False


def test_worker_accounts_missing_usage_id_once_and_marks_evidence_invalid(
    tmp_path: Path,
) -> None:
    usage = _FakeTokenUsage(
        input_other=6,
        output=2,
        input_cache_read=1,
    )
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("OK"),
            _FakeStatusUpdate(usage, message_id=None),
            _FakeStatusUpdate(usage, message_id=None),
            _FakeTurnEnd(),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            lambda _payload: None,
        )
    )

    assert outcome.input_tokens == 7
    assert outcome.output_tokens == 2
    assert outcome.cache_read_tokens == 1
    assert outcome.usage_seen is True
    assert outcome.usage_valid is False


def test_worker_marks_cached_input_exceeding_total_as_invalid(
    tmp_path: Path,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("OK"),
            _FakeStatusUpdate(
                SimpleNamespace(
                    input=3,
                    output=1,
                    input_cache_creation=2,
                    input_cache_read=2,
                ),
                message_id="response-invalid-cache",
            ),
            _FakeTurnEnd(),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            lambda _payload: None,
        )
    )

    assert outcome.usage_seen is True
    assert outcome.usage_valid is False


def test_kimi_session_runs_allowlisted_tool_and_returns_only_final_step(
    tmp_path: Path,
) -> None:
    approval = _FakeApprovalRequest(
        tool_call_id="call-1",
        sender="echo",
        action="mcp:echo",
    )
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("intermediate text must be discarded"),
            _FakeToolCall("call-1", "echo"),
            approval,
            _FakeToolResult("call-1"),
            _FakeStepBegin(),
            _FakeTextPart("final answer"),
            _FakeTurnEnd(),
        ]
    )
    bindings, config_docs, create_calls = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    emitted: list[dict[str, Any]] = []
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="use echo",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
        max_response_tokens=1_234,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            emitted.append,
        )
    )

    assert outcome.final_message == "final answer"
    assert outcome.tool_events == (
        ("tool_call", "echo", "call-1"),
        ("tool_result", "echo", "call-1"),
    )
    assert outcome.steps_used == 2
    assert approval.responses == ["approve"]
    assert session.cancel_calls == 0
    assert session.close_calls == 1
    assert session.provider.client.max_retries == 0
    assert session.provider.client.with_options_calls == [0]
    assert session.provider.thinking_effort is None
    assert session.provider._generation_kwargs == {
        "prompt_cache_key": session.id,
        "max_completion_tokens": 1_234,
        "reasoning_effort": "max",
        "extra_body": {
            "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
        },
    }
    assert [
        event["steps_used"] for event in emitted if event["type"] == "step_progress"
    ] == [1, 2]
    assert [
        (event["kind"], event["name"], event["call_id"])
        for event in emitted
        if event["type"] == "tool_event"
    ] == [
        ("tool_call", "echo", "call-1"),
        ("tool_result", "echo", "call-1"),
    ]
    assert config_docs[0]["mcp"]["client"]["tool_call_timeout_ms"] == 1_800_000
    create_call = create_calls[0]
    assert create_call["max_steps_per_turn"] == 2
    assert create_call["mcp_configs"] == [
        {
            "mcpServers": {
                "crucible": {
                    "url": "http://127.0.0.1:8765/mcp",
                    "transport": "streamable-http",
                    "auth": "m" * 48,
                }
            }
        }
    ]
    agent = json.loads(Path(create_call["agent_file"]).read_text())["agent"]
    assert agent["name"] == "crucible-kimi-toolset"
    assert agent["tools"] == []
    assert agent["subagents"] == {}


def test_kimi_tool_dispatch_budget_denial_terminates_sdk_tool_loop(
    tmp_path: Path,
) -> None:
    approval = _FakeApprovalRequest(
        tool_call_id="call-1",
        sender="echo",
        action="mcp:echo",
    )
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeToolCall("call-1", "echo"),
            approval,
            _FakeToolResult(
                "call-1",
                error_text=(
                    "Error: ToolDispatchBudgetExceeded: planner tool-call "
                    "limit exhausted; underlying dispatch was not started"
                ),
            ),
            AssertionError(
                "Kimi requested another paid step after exhaustion"
            ),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    emitted: list[dict[str, Any]] = []
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="use echo",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
        max_response_tokens=1_234,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    with pytest.raises(
        worker_mod.ToolDispatchBudgetTerminal,
        match="dispatch budget is exhausted",
    ) as caught:
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                emitted.append,
            )
        )

    outcome = worker_mod._classify(caught.value, request, bindings)
    assert outcome.error_kind == "tool_budget"
    assert outcome.retriable is False
    assert approval.responses == ["approve"]
    assert session.cancel_calls == 1
    assert session.close_calls == 1
    assert [
        event["kind"]
        for event in emitted
        if event["type"] == "tool_event"
    ] == ["tool_call", "tool_result"]


@pytest.mark.parametrize(
    "implementation",
    ["deployed-worker", "parent-mirror"],
)
@pytest.mark.parametrize("terminal_kind", ["budget", "mutation"])
@pytest.mark.parametrize(
    "result_order",
    [
        pytest.param(
            ("call-denied", "call-ok"),
            id="denied-first",
        ),
        pytest.param(
            ("call-ok", "call-denied"),
            id="denied-last",
        ),
    ],
)
def test_kimi_parallel_tool_budget_denial_drains_issued_results(
    tmp_path: Path,
    implementation: str,
    terminal_kind: str,
    result_order: tuple[str, str],
) -> None:
    approvals = {
        call_id: _FakeApprovalRequest(
            tool_call_id=call_id,
            sender="echo",
            action="mcp:echo",
        )
        for call_id in ("call-ok", "call-denied")
    }
    results = {
        "call-ok": _FakeToolResult("call-ok"),
        "call-denied": _FakeToolResult(
            "call-denied",
            error_text=(
                "Error: TerminalToolDispatchError: "
                + (
                    "ToolDispatchBudgetExceeded: planner tool-call tranche "
                    "exhausted; underlying dispatch was not started"
                    if terminal_kind == "budget"
                    else "ToolMutationIndeterminate: delete settlement is "
                    "unknown"
                )
            ),
        ),
    }
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeToolCall("call-ok", "echo"),
            _FakeToolCall("call-denied", "echo"),
            approvals["call-ok"],
            approvals["call-denied"],
            *(results[call_id] for call_id in result_order),
            AssertionError("Kimi requested another paid step after exhaustion"),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / implementation)
    emitted: list[dict[str, Any]] = []
    if implementation == "deployed-worker":
        request = worker_mod._Request(
            root=paths.root,
            system="system",
            user="use echo",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
            max_steps=2,
            max_response_tokens=1_234,
            mcp_url="http://127.0.0.1:8765/mcp",
            mcp_auth_token="m" * 48,
            mcp_tool_names=("echo",),
        )
        runner = worker_mod._run_session
        terminal_type = (
            worker_mod.ToolDispatchBudgetTerminal
            if terminal_kind == "budget"
            else worker_mod.ToolMutationIndeterminateTerminal
        )
        classifier = worker_mod._classify
    else:
        request = kimi_mod._WorkerRequest(
            system="system",
            user="use echo",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
            max_steps=2,
            max_response_tokens=1_234,
            mcp_url="http://127.0.0.1:8765/mcp",
            mcp_auth_token="m" * 48,
            mcp_tool_names=("echo",),
        )
        runner = kimi_mod._run_sdk_session
        terminal_type = (
            kimi_mod.KimiToolDispatchBudgetTerminal
            if terminal_kind == "budget"
            else kimi_mod.KimiToolMutationIndeterminateTerminal
        )
        classifier = kimi_mod._classify_worker_exception

    with pytest.raises(
        terminal_type,
        match=(
            "dispatch budget is exhausted"
            if terminal_kind == "budget"
            else "indeterminate source mutation"
        ),
    ) as caught:
        asyncio.run(
            runner(
                request,
                paths,
                bindings,
                emitted.append,
            )
        )

    expected_tool_events = (
        ("tool_call", "echo", "call-ok"),
        ("tool_call", "echo", "call-denied"),
        *(("tool_result", "echo", call_id) for call_id in result_order),
    )
    observed_tool_events = tuple(
        (event["kind"], event["name"], event["call_id"])
        for event in emitted
        if event["type"] == "tool_event"
    )
    assert observed_tool_events == expected_tool_events
    assert all(approval.responses == ["approve"] for approval in approvals.values())
    assert session.cancel_calls == 1
    assert session.close_calls == 1

    classified = classifier(
        caught.value,
        request,
        bindings,
        steps_used=1,
    )
    expected_error_kind = (
        "tool_budget" if terminal_kind == "budget" else "tool_mutation"
    )
    assert classified.error_kind == expected_error_kind
    assert classified.retriable is False

    # Exercise the parent process's exact terminal parser, incremental merge,
    # and fail-closed trace validators. A balanced budget terminal must retain
    # its typed classification instead of degrading to a protocol failure.
    terminal = kimi_mod._WorkerOutcome.from_wire(classified.to_wire())
    incremental = kimi_mod._WorkerOutcome(
        steps_used=1,
        tool_events=expected_tool_events,
    )
    qualified = kimi_mod._merge_incremental_outcome(incremental, terminal)
    qualified = kimi_mod._qualify_worker_model_identity(qualified)
    qualified = kimi_mod._qualify_worker_tool_events(
        qualified,
        allowed_names=("echo",),
    )
    qualified = kimi_mod._qualify_worker_steps(
        qualified,
        physical_limit=2,
    )

    assert qualified.error_kind == expected_error_kind
    assert qualified.retriable is False
    assert qualified.tool_events == expected_tool_events
    assert "invalid per-attempt tool event sequence" not in qualified.error_message


def test_worker_step_ledger_emits_incrementally_and_rejects_excess(
    tmp_path: Path,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeStepBegin(),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    emitted: list[dict[str, Any]] = []
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=1,
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="more StepBegin",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                emitted.append,
            )
        )

    assert [
        event["steps_used"] for event in emitted if event["type"] == "step_progress"
    ] == [1]
    assert session.cancel_calls == 1
    assert session.close_calls == 1


def test_production_worker_rejects_malformed_usage_evidence(
    tmp_path: Path,
) -> None:
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeTextPart("OK"),
            _FakeStatusUpdate(
                SimpleNamespace(
                    input=-1,
                    output=True,
                    input_cache_creation="2",
                    input_cache_read=0,
                ),
                message_id="message-invalid",
            ),
            _FakeTurnEnd(),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    emitted: list[dict[str, Any]] = []
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = asyncio.run(
        worker_mod._run_session(
            request,
            paths,
            bindings,
            emitted.append,
        )
    )

    assert outcome.final_message == "OK"
    assert outcome.usage_seen is True
    assert outcome.usage_valid is False
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0
    assert emitted[-1]["usage_valid"] is False
    assert outcome.to_wire()["usage_valid"] is False


@pytest.mark.parametrize("implementation", ["deployed-worker", "parent-mirror"])
def test_max_steps_preserves_exact_pricing_identity_for_settled_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    implementation: str,
) -> None:
    _qualified_versions(monkeypatch)
    approval = _FakeApprovalRequest(
        tool_call_id="call-1",
        sender="echo",
        action="mcp:echo",
    )
    session = _FakeSession(
        [
            _FakeStepBegin(),
            _FakeToolCall("call-1", "echo"),
            approval,
            _FakeStatusUpdate(
                _FakeTokenUsage(
                    input_other=11,
                    output=4,
                    input_cache_read=2,
                ),
                message_id="message-max-step",
            ),
            _FakeToolResult("call-1"),
            _FakeSDKError("exact step limit reached"),
        ]
    )
    session.provider._crucible_last_stream = SimpleNamespace(
        finish_reason="tool_calls",
        served_model=_MODEL,
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / implementation)
    emitted: list[dict[str, Any]] = []
    if implementation == "deployed-worker":
        request = worker_mod._Request(
            root=paths.root,
            system="system",
            user="use echo",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
            max_steps=1,
            mcp_url="http://127.0.0.1:8765/mcp",
            mcp_auth_token="m" * 48,
            mcp_tool_names=("echo",),
        )
        runner = worker_mod._run_session
    else:
        request = kimi_mod._WorkerRequest(
            system="system",
            user="use echo",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
            max_steps=1,
            mcp_url="http://127.0.0.1:8765/mcp",
            mcp_auth_token="m" * 48,
            mcp_tool_names=("echo",),
        )
        runner = kimi_mod._run_sdk_session

    outcome = asyncio.run(
        runner(request, paths, bindings, emitted.append)
    )

    assert outcome.final_message == ""
    assert outcome.stop_reason == "max_turns"
    assert outcome.error_kind == "max_turns"
    assert outcome.terminal_seen is False
    assert outcome.usage_seen is True
    assert outcome.usage_valid is True
    assert outcome.configured_model == _MODEL
    assert outcome.served_model == _MODEL
    assert outcome.steps_used == 1
    assert approval.responses == ["approve"]
    assert session.close_calls == 1

    parsed = kimi_mod._WorkerOutcome.from_wire(outcome.to_wire())
    latest = kimi_mod._WorkerOutcome()
    for payload in emitted:
        if payload.get("type") == "step_progress":
            latest = replace(latest, steps_used=payload["steps_used"])
        elif payload.get("type") == "usage":
            latest = kimi_mod._settle_usage_checkpoint(
                latest,
                (
                    payload["input_tokens"],
                    payload["output_tokens"],
                    payload["cache_creation_tokens"],
                    payload["cache_read_tokens"],
                ),
                usage_valid=payload["usage_valid"],
            )
        elif payload.get("type") == "tool_event":
            latest = replace(
                latest,
                tool_events=(
                    *latest.tool_events,
                    (payload["kind"], payload["name"], payload["call_id"]),
                ),
            )
    merged = kimi_mod._merge_incremental_outcome(latest, parsed)
    merged = kimi_mod._qualify_worker_model_identity(merged)
    merged = kimi_mod._qualify_worker_tool_events(
        merged,
        allowed_names=("echo",),
    )
    merged = kimi_mod._qualify_worker_steps(merged, physical_limit=1)
    assert merged.error_kind == "max_turns"
    assert merged.settled_usage_steps == merged.steps_used == 1

    backend = KimiBackend()
    invocation = _invocation()
    assert invocation.policy is not None
    result = backend._result_from_outcome(
        invocation,
        invocation.policy,
        merged,
        base_url=_BASE_URL,
        events=[],
        tool_authority_digest=None,
    )
    assert result.stop_reason == "max_turns"
    assert result.error is not None
    assert result.error.kind == "max_turns"
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert result.completeness.terminal_complete is False
    assert result.pricing_evidence.complete is True
    assert len(result.pricing_evidence.segments) == 1


@pytest.mark.parametrize("implementation", ["deployed-worker", "parent-mirror"])
@pytest.mark.parametrize(
    "fault",
    [
        "early",
        "pending_tool",
        "interrupted",
        "missing_usage",
        "wrong_finish_reason",
        "wrong_served_model",
    ],
)
def test_max_steps_rejects_non_exact_accounting_boundary(
    tmp_path: Path,
    implementation: str,
    fault: str,
) -> None:
    approval = _FakeApprovalRequest(
        tool_call_id="call-1",
        sender="echo",
        action="mcp:echo",
    )
    events: list[Any] = [
        _FakeStepBegin(),
        _FakeToolCall("call-1", "echo"),
        approval,
    ]
    if fault != "missing_usage":
        events.append(
            _FakeStatusUpdate(
                _FakeTokenUsage(input_other=11, output=4),
                message_id="message-max-step",
            )
        )
    if fault != "pending_tool":
        events.append(_FakeToolResult("call-1"))
    if fault == "interrupted":
        events.append(_FakeStepInterrupted())
    events.append(_FakeSDKError("inexact step limit"))
    session = _FakeSession(events)
    session.provider._crucible_last_stream = SimpleNamespace(
        finish_reason=(
            "stop" if fault == "wrong_finish_reason" else "tool_calls"
        ),
        served_model=(
            f"{_MODEL}-substitute"
            if fault == "wrong_served_model"
            else _MODEL
        ),
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / implementation)
    max_steps = 2 if fault == "early" else 1
    if implementation == "deployed-worker":
        request = worker_mod._Request(
            root=paths.root,
            system="system",
            user="use echo",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
            max_steps=max_steps,
            mcp_url="http://127.0.0.1:8765/mcp",
            mcp_auth_token="m" * 48,
            mcp_tool_names=("echo",),
        )
        runner = worker_mod._run_session
        error_type = worker_mod.WorkerQualificationError
    else:
        request = kimi_mod._WorkerRequest(
            system="system",
            user="use echo",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
            max_steps=max_steps,
            mcp_url="http://127.0.0.1:8765/mcp",
            mcp_auth_token="m" * 48,
            mcp_tool_names=("echo",),
        )
        runner = kimi_mod._run_sdk_session
        error_type = KimiSDKQualificationError

    with pytest.raises(error_type, match="step.limit|settled loop"):
        asyncio.run(runner(request, paths, bindings, lambda _payload: None))
    assert session.close_calls == 1


def test_worker_wire_rejects_identity_bearing_non_max_steps_error() -> None:
    payload = kimi_mod._WorkerOutcome(
        input_tokens=13,
        output_tokens=4,
        usage_seen=True,
        usage_valid=True,
        terminal_seen=False,
        configured_model=_MODEL,
        served_model=_MODEL,
        stop_reason="max_turns",
        error_kind="max_turns",
        error_message="bounded max turns",
        steps_used=1,
    ).to_wire()

    parsed = kimi_mod._WorkerOutcome.from_wire(payload)
    assert parsed.error_kind == "max_turns"
    payload["stop_reason"] = "error"
    payload["error_kind"] = "transport"
    with pytest.raises(ValueError, match="invalid terminal state"):
        kimi_mod._WorkerOutcome.from_wire(payload)


@pytest.mark.parametrize(
    ("physical_limit", "settled_usage_steps"),
    [(2, 1), (1, 0)],
)
def test_parent_rejects_max_steps_without_exact_physical_settlement(
    physical_limit: int,
    settled_usage_steps: int,
) -> None:
    outcome = kimi_mod._WorkerOutcome(
        input_tokens=13,
        output_tokens=4,
        usage_seen=True,
        usage_valid=True,
        terminal_seen=False,
        configured_model=_MODEL,
        served_model=_MODEL,
        stop_reason="max_turns",
        error_kind="max_turns",
        error_message="bounded max turns",
        steps_used=1,
        settled_usage_steps=settled_usage_steps,
    )

    qualified = kimi_mod._qualify_worker_steps(
        outcome,
        physical_limit=physical_limit,
    )
    assert qualified.error_kind == "qualification"
    assert qualified.usage_valid is False


def test_step_interrupted_preserves_retriable_provider_exception(
    tmp_path: Path,
) -> None:
    provider_error = _FakeAPIStatusError(503, "temporary upstream failure")
    session = _FakeSession(
        [
            _FakeStepInterrupted(),
            provider_error,
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    bindings = replace(
        bindings,
        RunCancelled=type("FakeRunCancelled", (Exception,), {}),
        MaxStepsReached=type("FakeMaxStepsReached", (Exception,), {}),
    )
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    with pytest.raises(_FakeAPIStatusError) as caught:
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )
    outcome = worker_mod._classify(caught.value, request, bindings)

    assert outcome.error_kind == "transport"
    assert outcome.http_status == 503
    assert outcome.retriable is True
    assert session.close_calls == 1


def test_step_interrupted_silent_stream_end_fails_qualification(
    tmp_path: Path,
) -> None:
    session = _FakeSession([_FakeStepInterrupted()])
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="without TurnEnd after StepInterrupted",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )

    assert session.close_calls == 1


def test_kimi_session_denies_unknown_approval_and_closes_once(
    tmp_path: Path,
) -> None:
    approval = _FakeApprovalRequest()
    session = _FakeSession([approval])
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="unknown or mismatched tool approval",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )

    assert approval.responses == ["reject"]
    assert session.cancel_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize(
    ("tool_call_id", "sender", "action"),
    [
        ("other-call", "echo", "mcp:echo"),
        ("call-1", "other", "mcp:echo"),
        ("call-1", "echo", "mcp:other"),
    ],
)
def test_kimi_session_rejects_mismatched_tool_approval(
    tmp_path: Path,
    tool_call_id: str,
    sender: str,
    action: str,
) -> None:
    approval = _FakeApprovalRequest(
        tool_call_id=tool_call_id,
        sender=sender,
        action=action,
    )
    session = _FakeSession([_FakeToolCall("call-1", "echo"), approval])
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="unknown or mismatched tool approval",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )

    assert approval.responses == ["reject"]
    assert session.cancel_calls == 1
    assert session.close_calls == 1


def test_production_worker_rejects_completed_tool_call_id_replay(
    tmp_path: Path,
) -> None:
    approval = _FakeApprovalRequest(
        tool_call_id="call-1",
        sender="echo",
        action="mcp:echo",
    )
    session = _FakeSession(
        [
            _FakeToolCall("call-1", "echo"),
            approval,
            _FakeToolResult("call-1"),
            _FakeToolCall("call-1", "echo"),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="outside the attached ToolSet",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )

    assert approval.responses == ["approve"]
    assert session.cancel_calls == 1
    assert session.close_calls == 1


def test_production_worker_enforces_tool_event_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(worker_mod, "_MAX_TOOL_EVENTS", 2)
    approval = _FakeApprovalRequest(
        tool_call_id="call-1",
        sender="echo",
        action="mcp:echo",
    )
    session = _FakeSession(
        [
            _FakeToolCall("call-1", "echo"),
            approval,
            _FakeToolResult("call-1"),
            _FakeToolCall("call-2", "echo"),
        ]
    )
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="outside the attached ToolSet",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )

    assert session.cancel_calls == 1
    assert session.close_calls == 1


@pytest.mark.parametrize("tool_call_id", ["", "x" * 257, "bad call id"])
def test_production_worker_rejects_nonportable_tool_call_id(
    tmp_path: Path,
    tool_call_id: str,
) -> None:
    session = _FakeSession([_FakeToolCall(tool_call_id, "echo")])
    bindings, _, _ = _fake_bindings(session)
    paths = _worker_paths(tmp_path / "worker")
    request = worker_mod._Request(
        root=paths.root,
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        max_steps=2,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="outside the attached ToolSet",
    ):
        asyncio.run(
            worker_mod._run_session(
                request,
                paths,
                bindings,
                lambda _payload: None,
            )
        )

    assert session.cancel_calls == 1
    assert session.close_calls == 1


def test_kimi_worker_environment_excludes_ambient_overrides(
    tmp_path: Path,
) -> None:
    original_env = os.environ.copy()
    original_cwd = Path.cwd()
    original_tempdir = tempfile.tempdir
    original_umask = os.umask(0)
    os.umask(original_umask)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    try:
        os.environ["OPENAI_BASE_URL"] = "https://hostile.invalid/v1"
        os.environ["OPENAI_API_KEY"] = "hostile-openai-token"
        os.environ.update(_HOSTILE_KIMI_PROVIDER_ENV)
        os.environ["MOONSHOT_API_KEY"] = "hostile-moonshot-token"
        os.environ["INFER_BASE_URL"] = "https://legacy-hostile.invalid/v1"
        os.environ["INFER_AUTH_TOKEN"] = _TOKEN
        os.environ["MCP_CONFIG_FILE"] = "/hostile/mcp.json"

        paths = kimi_mod._isolate_worker_environment(root)

        assert "OPENAI_BASE_URL" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ
        for name in _HOSTILE_KIMI_PROVIDER_ENV:
            assert name not in os.environ
        assert "MOONSHOT_API_KEY" not in os.environ
        assert "INFER_BASE_URL" not in os.environ
        assert "INFER_AUTH_TOKEN" not in os.environ
        assert "MCP_CONFIG_FILE" not in os.environ
        assert os.environ["HOME"] == str(paths.home)
        assert os.environ["KIMI_SHARE_DIR"] == str(paths.share)
        assert Path.cwd() == paths.work
        assert list(paths.work.iterdir()) == []
        assert list(paths.skills.iterdir()) == []
    finally:
        os.umask(original_umask)
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)
        tempfile.tempdir = original_tempdir


def test_production_worker_environment_excludes_hostile_ambient_sources(
    tmp_path: Path,
) -> None:
    original_env = os.environ.copy()
    original_cwd = Path.cwd()
    original_tempdir = tempfile.tempdir
    original_umask = os.umask(0)
    os.umask(original_umask)
    hostile_home = tmp_path / "hostile-home"
    hostile_home.mkdir()
    (hostile_home / "AGENTS.md").write_text("HOSTILE ROOT MARKER")
    (hostile_home / ".kimi").mkdir()
    (hostile_home / ".kimi" / "mcp.json").write_text("HOSTILE MCP MARKER")
    root = tmp_path / "private-production-worker"
    root.mkdir(mode=0o700)
    try:
        os.environ.update(
            {
                "HOME": str(hostile_home),
                "XDG_CONFIG_HOME": str(hostile_home / ".config"),
                "OPENAI_BASE_URL": "https://hostile.invalid/v1",
                "OPENAI_API_KEY": "hostile-openai-token",
                **_HOSTILE_KIMI_PROVIDER_ENV,
                "KIMI_SHARE_DIR": str(hostile_home / ".kimi"),
                "MOONSHOT_API_KEY": "hostile-moonshot-token",
                "INFER_BASE_URL": "https://legacy-hostile.invalid/v1",
                "INFER_AUTH_TOKEN": _TOKEN,
                "MCP_CONFIG_FILE": str(hostile_home / ".kimi" / "mcp.json"),
            }
        )

        paths = worker_mod._isolate_environment(root)

        for name in (
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "MOONSHOT_API_KEY",
            "INFER_BASE_URL",
            "INFER_AUTH_TOKEN",
            "MCP_CONFIG_FILE",
            *_HOSTILE_KIMI_PROVIDER_ENV,
        ):
            assert name not in os.environ
        assert os.environ["HOME"] == str(paths.home)
        assert os.environ["XDG_CONFIG_HOME"] == str(paths.config)
        assert os.environ["KIMI_SHARE_DIR"] == str(paths.share)
        assert Path.cwd() == paths.work
        assert list(paths.home.iterdir()) == []
        assert list(paths.share.iterdir()) == []
        assert list(paths.work.iterdir()) == []
        assert list(paths.skills.iterdir()) == []
        assert hostile_home not in paths.work.parents
    finally:
        os.umask(original_umask)
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)
        tempfile.tempdir = original_tempdir


def _crashing_worker(
    _connection: Any,
    request: kimi_mod._WorkerRequest,
    root_text: str,
) -> None:
    artifact = Path(root_text) / "share"
    artifact.mkdir()
    (artifact / "context.jsonl").write_text(
        request.auth_token + "\nraw ThinkPart reasoning"
    )
    os._exit(23)


@pytest.mark.parametrize("worker_state", ["exit", "alive", "cancel"])
def test_kimi_pipe_eof_waits_for_exit_within_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    worker_state: str,
) -> None:
    root = tmp_path / "private-worker"
    clock = [100.0]

    class Connection:
        def __init__(self) -> None:
            self.eof_read = False
            self.closed = False

        def poll(self, _timeout: float) -> bool:
            assert not self.eof_read, "EOF must not become a busy-poll loop"
            return True

        def recv(self) -> Any:
            self.eof_read = True
            raise EOFError

        def close(self) -> None:
            self.closed = True

    class Process:
        def __init__(self) -> None:
            self.alive = True
            self.exitcode: int | None = None
            self.closed = False
            self.terminate_calls = 0

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            if not self.alive:
                return
            assert 0 < timeout <= 0.2
            assert timeout <= 102.0 - clock[0]
            if worker_state == "cancel":
                raise KeyboardInterrupt
            clock[0] += timeout
            if worker_state == "exit":
                self.alive = False
                self.exitcode = 23

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.alive = False
            self.exitcode = -15

        def close(self) -> None:
            self.closed = True

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        (root / "context.jsonl").write_text(
            _TOKEN + "\nraw ThinkPart reasoning"
        )
        return root

    receive, send, process = Connection(), Connection(), Process()
    started: list[Any] = []
    stopped: list[Any] = []
    context = SimpleNamespace(
        Pipe=lambda **_kwargs: (receive, send),
        Process=lambda **_kwargs: process,
    )
    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod.time, "monotonic", lambda: clock[0])
    request = kimi_mod._WorkerRequest(
        system="system", user="user", base_url=_BASE_URL, auth_token=_TOKEN
    )

    def invoke() -> kimi_mod._WorkerOutcome:
        return kimi_mod._run_worker_process(
            request,
            2.0,
            process_context=context,
            on_process_start=started.append,
            on_process_stop=stopped.append,
        )

    if worker_state == "cancel":
        with pytest.raises(KeyboardInterrupt):
            invoke()
    else:
        outcome = invoke()
        assert outcome.error_kind == (
            "worker_exit" if worker_state == "exit" else "timeout"
        )
        assert outcome.retriable is False
        assert _TOKEN not in outcome.error_message
        if worker_state == "alive":
            assert clock[0] == 102.0

    assert process.terminate_calls == (worker_state != "exit")
    assert receive.eof_read and receive.closed and send.closed
    assert started == stopped == [process]
    assert process.closed and not process.alive
    assert not root.exists()


def _hanging_worker(
    _connection: Any,
    request: kimi_mod._WorkerRequest,
    root_text: str,
) -> None:
    artifact = Path(root_text) / "share"
    artifact.mkdir()
    (artifact / "context.jsonl").write_text(
        request.auth_token + "\nraw ThinkPart reasoning"
    )
    time.sleep(60)


def _write_fake_subprocess_worker(
    path: Path,
    *,
    hang: bool,
) -> None:
    final_action = "time.sleep(60)" if hang else "os._exit(23)"
    path.write_text(
        "#!/bin/sh\n"
        "exec /usr/bin/python3 -c '\n"
        "import json, os, pathlib, sys, time\n"
        "request = json.loads(sys.stdin.readline())\n"
        'artifact = pathlib.Path(request["root"]) / "share"\n'
        "artifact.mkdir()\n"
        '(artifact / "context.jsonl").write_text('
        'request["auth_token"] + "\\nraw ThinkPart reasoning")\n'
        f"{final_action}\n"
        "'\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="forced-death isolation probe requires fork",
)
def test_kimi_forced_worker_death_removes_credentials_and_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    request = kimi_mod._WorkerRequest(
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = kimi_mod._run_worker_process(
        request,
        2.0,
        process_context=multiprocessing.get_context("fork"),
        worker_target=_crashing_worker,
    )

    assert outcome.error_kind == "worker_exit"
    assert _TOKEN not in outcome.error_message
    assert root.exists() is False


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="worker deadline probe requires fork",
)
def test_kimi_worker_deadline_terminates_and_scrubs_private_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    request = kimi_mod._WorkerRequest(
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = kimi_mod._run_worker_process(
        request,
        0.05,
        process_context=multiprocessing.get_context("fork"),
        worker_target=_hanging_worker,
    )

    assert outcome.error_kind == "timeout"
    assert outcome.retriable is False
    assert root.exists() is False


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="legacy process-owner retention probe requires fork",
)
def test_legacy_worker_stop_failure_retains_exact_process_and_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    real_stop = kimi_mod._stop_process
    allow_settlement = False
    stop_callbacks: list[Any] = []

    def injected_stop(process: Any) -> None:
        if not allow_settlement:
            raise kimi_mod.KimiWorkerLifecycleError(
                "injected legacy process settlement failure"
            )
        real_stop(process)

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod, "_stop_process", injected_stop)
    backend = KimiBackend()
    request = kimi_mod._WorkerRequest(
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    with pytest.raises(
        kimi_mod.KimiWorkerLifecycleError,
        match="worker lifecycle did not settle",
    ):
        kimi_mod._run_worker_process(
            request,
            2.0,
            process_context=multiprocessing.get_context("fork"),
            worker_target=_crashing_worker,
            on_process_start=backend._register_process,  # noqa: SLF001
            on_process_stop=lambda process: (
                stop_callbacks.append(process),
                backend._unregister_process(process),  # noqa: SLF001
            ),
        )

    assert stop_callbacks == []
    assert len(backend._active_processes) == 1  # noqa: SLF001
    retained_process = next(iter(backend._active_processes))  # noqa: SLF001
    assert getattr(retained_process, "_crucible_private_root") == root
    assert root.is_dir()

    allow_settlement = True
    backend.close()

    assert backend._active_processes == set()  # noqa: SLF001
    assert root.exists() is False


@pytest.mark.parametrize(
    ("hang", "timeout_seconds", "expected_error"),
    [
        (False, 2.0, "worker_exit"),
        (True, 0.05, "timeout"),
    ],
)
def test_deployed_subprocess_failure_reaps_and_scrubs_private_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hang: bool,
    timeout_seconds: float,
    expected_error: str,
) -> None:
    root = tmp_path / "private-worker"
    worker_python = tmp_path / "fake-worker"
    _write_fake_subprocess_worker(worker_python, hang=hang)
    started: list[subprocess.Popen[bytes]] = []

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod, "_qualify_worker_runtime", lambda _path: None)
    request = kimi_mod._WorkerRequest(
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
    )

    outcome = kimi_mod._run_isolated_worker_subprocess(
        request,
        timeout_seconds,
        worker_python=worker_python,
        on_process_start=started.append,
    )

    assert outcome.error_kind == expected_error
    assert _TOKEN not in outcome.error_message
    assert len(started) == 1
    assert started[0].poll() is not None
    assert root.exists() is False


def test_deployed_subprocess_kills_term_ignoring_descendant_after_leader_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"
    worker_python = tmp_path / "descendant-worker"
    worker_python.write_text(
        "#!/usr/bin/python3\n"
        "import json, pathlib, subprocess, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        'root = pathlib.Path(request[\"root\"])\n'
        '(root / \"credential-copy\").write_text(request[\"auth_token\"])\n'
        "child = subprocess.Popen([\n"
        "    sys.executable, \"-c\",\n"
        "    \"import os,signal,time;\"\n"
        "    \"signal.signal(signal.SIGTERM, signal.SIG_IGN);\"\n"
        "    \"print(os.getpid(), flush=True);time.sleep(60)\",\n"
        "], stdout=subprocess.PIPE, text=True)\n"
        "child_pid = int(child.stdout.readline())\n"
        "print(json.dumps({\"protocol\":\"crucible.kimi-worker.v7\","
        "\"type\":\"step_progress\",\"steps_used\":1}), flush=True)\n"
        "print(json.dumps({\"protocol\":\"crucible.kimi-worker.v7\","
        "\"type\":\"usage\",\"input_tokens\":1,\"output_tokens\":1,"
        "\"cache_creation_tokens\":0,\"cache_read_tokens\":0,"
        "\"usage_valid\":True}), flush=True)\n"
        "print(json.dumps({\n"
        "    \"protocol\":\"crucible.kimi-worker.v7\", \"type\":\"terminal\",\n"
        "    \"final_message\":str(child_pid), \"input_tokens\":1,\n"
        "    \"output_tokens\":1, \"cache_creation_tokens\":0,\n"
        "    \"cache_read_tokens\":0, \"usage_seen\":True, \"usage_valid\":True,\n"
        "    \"terminal_seen\":True,\n"
        f"    \"configured_model\":\"{_MODEL}\",\n"
        f"    \"served_model\":\"{_MODEL}\",\n"
        "    \"stop_reason\":\"end_turn\", \"error_kind\":None,\n"
        "    \"error_message\":\"\", \"http_status\":None,\n"
        "    \"retriable\":False, \"steps_used\":1,\n"
        "    \"tool_events\":[],\n"
        "    \"cost_bound_segments\":[],\n"
        "}), flush=True)\n",
        encoding="utf-8",
    )
    worker_python.chmod(0o700)

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod, "_qualify_worker_runtime", lambda _path: None)
    outcome = kimi_mod._run_isolated_worker_subprocess(
        kimi_mod._WorkerRequest(
            system="system",
            user="user",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
        ),
        3.0,
        worker_python=worker_python,
    )

    assert outcome.error_kind is None
    descendant_pid = int(outcome.final_message)
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)
    assert root.exists() is False


def test_deployed_subprocess_retains_credential_root_until_settlement_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"
    worker_python = tmp_path / "fake-worker"
    _write_fake_subprocess_worker(worker_python, hang=False)

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    real_stop = kimi_mod._stop_process_locked
    allow_settlement = False

    def injected_stop(process: Any) -> None:
        if not allow_settlement:
            raise kimi_mod.KimiWorkerLifecycleError(
                "injected unconfirmed process group"
            )
        real_stop(process)

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod, "_qualify_worker_runtime", lambda _path: None)
    monkeypatch.setattr(kimi_mod, "_stop_process_locked", injected_stop)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(worker_python=worker_python)

    with pytest.raises(
        kimi_mod.KimiWorkerLifecycleError,
        match="worker lifecycle did not settle",
    ):
        backend.invoke(_invocation())

    assert len(backend._active_processes) == 1  # noqa: SLF001
    assert root.is_dir()
    credential_copy = root / "share" / "context.jsonl"
    assert credential_copy.is_file()
    assert _TOKEN in credential_copy.read_text(encoding="utf-8")

    allow_settlement = True
    backend.close()
    assert backend._active_processes == set()  # noqa: SLF001
    assert root.exists() is False


def test_kimi_process_group_inspection_oserror_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_inspection(_pgid: int, _signal_number: int) -> None:
        raise OSError("injected process-table failure")

    monkeypatch.setattr(kimi_mod.os, "killpg", fail_inspection)
    with pytest.raises(
        kimi_mod.KimiWorkerLifecycleError,
        match="could not inspect",
    ):
        kimi_mod._process_group_exists(12345)


def test_backend_close_cancels_deployed_subprocess_and_scrubs_private_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"
    worker_python = tmp_path / "fake-worker"
    _write_fake_subprocess_worker(worker_python, hang=True)
    registered = threading.Event()

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod, "_qualify_worker_runtime", lambda _path: None)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(worker_python=worker_python)
    original_register = backend._register_process

    def register(process: Any) -> None:
        original_register(process)
        registered.set()

    monkeypatch.setattr(backend, "_register_process", register)
    results: list[Any] = []
    thread = threading.Thread(
        target=lambda: results.append(backend.invoke(_invocation())),
        daemon=True,
    )
    thread.start()
    assert registered.wait(timeout=2.0)

    backend.close()
    thread.join(timeout=2.0)

    assert thread.is_alive() is False
    assert len(results) == 1
    assert results[0].stop_reason == "cancelled"
    assert results[0].error.kind == "cancellation"
    assert backend._active_processes == set()
    assert root.exists() is False


def test_kimi_subprocess_protocol_keeps_secrets_out_of_argv_and_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private-worker"
    worker_python = tmp_path / "fake-worker"
    worker_python.write_text(
        "#!/bin/sh\n"
        "exec /usr/bin/python3 -c '\n"
        "import json, os, pathlib, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        'token = request["auth_token"]\n'
        'argv_clean = token not in pathlib.Path("/proc/self/cmdline")'
        '.read_bytes().decode(errors="replace")\n'
        "env_clean = all(token not in value for value in os.environ.values())\n"
        'artifact = pathlib.Path(request["root"]) / "secret-artifact"\n'
        "artifact.write_text(token)\n"
        "payload = {\n"
        '  "protocol": "crucible.kimi-worker.v7",\n'
        '  "type": "terminal",\n'
        '  "final_message": json.dumps({"argv_clean": argv_clean, '
        '"env_clean": env_clean}),\n'
        '  "input_tokens": 1,\n'
        '  "output_tokens": 1,\n'
        '  "cache_creation_tokens": 0,\n'
        '  "cache_read_tokens": 0,\n'
        '  "usage_seen": True,\n'
        '  "usage_valid": True,\n'
        '  "terminal_seen": True,\n'
        f'  "configured_model": "{_MODEL}",\n'
        f'  "served_model": "{_MODEL}",\n'
        '  "stop_reason": "end_turn",\n'
        '  "error_kind": None,\n'
        '  "error_message": "",\n'
        '  "http_status": None,\n'
        '  "retriable": False,\n'
        '  "steps_used": 1,\n'
        '  "tool_events": [],\n'
        '  "cost_bound_segments": [],\n'
        "}\n"
        'print(json.dumps({"protocol": "crucible.kimi-worker.v7", '
        '"type": "step_progress", "steps_used": 1}), flush=True)\n'
        'print(json.dumps({"protocol": "crucible.kimi-worker.v7", '
        '"type": "usage", "input_tokens": 1, "output_tokens": 1, '
        '"cache_creation_tokens": 0, "cache_read_tokens": 0, '
        '"usage_valid": True}), flush=True)\n'
        "print(json.dumps(payload), flush=True)\n"
        "'\n"
    )
    worker_python.chmod(0o700)

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(
        kimi_mod,
        "_qualify_worker_runtime",
        lambda _path: None,
    )
    request = kimi_mod._WorkerRequest(
        system="system",
        user="user",
        base_url=_BASE_URL,
        auth_token=_TOKEN,
        mcp_url="http://127.0.0.1:8765/mcp",
        mcp_auth_token="m" * 48,
        mcp_tool_names=("echo",),
    )

    outcome = kimi_mod._run_isolated_worker_subprocess(
        request,
        2.0,
        worker_python=worker_python,
    )

    assert outcome.error_kind is None
    assert json.loads(outcome.final_message) == {
        "argv_clean": True,
        "env_clean": True,
    }
    assert root.exists() is False
    request_wire = kimi_mod._subprocess_request_wire(
        request,
        tmp_path / "root",
    )
    assert request_wire["mcp_url"] == "http://127.0.0.1:8765/mcp"


@pytest.mark.parametrize(
    ("blank_before_step", "trailing_delay", "expected_error"),
    [
        (True, 0.0, "qualification"),
        (False, 0.0, "qualification"),
        (False, kimi_mod._WORKER_DRAIN_SECONDS + 0.1, "lifecycle"),
    ],
)
def test_subprocess_protocol_rejects_blank_trailing_or_unsettled_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    blank_before_step: bool,
    trailing_delay: float,
    expected_error: str,
) -> None:
    root = tmp_path / "private-worker"
    worker_python = tmp_path / "protocol-worker"
    terminal = _success_outcome(
        input_tokens=1,
        output_tokens=1,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    ).to_wire()
    terminal["protocol"] = kimi_mod._WORKER_PROTOCOL
    step = {
        "protocol": kimi_mod._WORKER_PROTOCOL,
        "type": "step_progress",
        "steps_used": 1,
    }
    usage = {
        "protocol": kimi_mod._WORKER_PROTOCOL,
        "type": "usage",
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "usage_valid": True,
    }
    prefix = [""] if blank_before_step else []
    records = [*prefix, json.dumps(step), json.dumps(usage), json.dumps(terminal)]
    trailing = None if blank_before_step else json.dumps(terminal)
    worker_python.write_text(
        "#!/usr/bin/python3\n"
        "import json, sys, time\n"
        "json.loads(sys.stdin.readline())\n"
        f"records = {records!r}\n"
        "for record in records:\n"
        "    print(record, flush=True)\n"
        f"time.sleep({trailing_delay!r})\n"
        f"trailing = {trailing!r}\n"
        "if trailing is not None:\n"
        "    print(trailing, flush=True)\n",
        encoding="utf-8",
    )
    worker_python.chmod(0o700)

    def make_root() -> Path:
        root.mkdir(mode=0o700)
        return root

    monkeypatch.setattr(kimi_mod, "_make_private_temp_root", make_root)
    monkeypatch.setattr(kimi_mod, "_qualify_worker_runtime", lambda _path: None)

    outcome = kimi_mod._run_isolated_worker_subprocess(
        kimi_mod._WorkerRequest(
            system="system",
            user="user",
            base_url=_BASE_URL,
            auth_token=_TOKEN,
        ),
        5.0,
        worker_python=worker_python,
    )

    assert outcome.error_kind == expected_error
    assert outcome.terminal_seen is False
    assert root.exists() is False


def _worker_request_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": worker_mod._PROTOCOL,
        "type": "invoke",
        "root": "/tmp/private-worker",
        "system": "system",
        "user": "user",
        "model": _MODEL,
        "reasoning_effort": "max",
        "diagnostic_cost_upper_bound": False,
        "diagnostic_budget_usd": None,
        "schema_json": None,
        "base_url": _BASE_URL,
        "auth_token": _TOKEN,
        "max_steps": 2,
        "max_response_tokens": 32_768,
        "mcp_url": "http://127.0.0.1:8765/mcp",
        "mcp_auth_token": "m" * 48,
        "mcp_tool_names": ["echo"],
    }
    payload.update(overrides)
    return payload


def test_worker_request_accepts_only_complete_loopback_mcp_authority() -> None:
    request = worker_mod._Request.from_wire(_worker_request_payload())

    assert request.mcp_url == "http://127.0.0.1:8765/mcp"
    assert request.mcp_auth_token == "m" * 48
    assert request.mcp_tool_names == ("echo",)
    assert request.max_steps == 2
    assert request.max_response_tokens == 32_768


@pytest.mark.parametrize(
    "base_url",
    [
        "http://inference.example.test/v1",
        "http://localhost:8765/v1",
        "http://[::1]:8765/v1",
        "http://127.0.0.1/v1",
        "http://user:pass@127.0.0.1:8765/v1",
        "http://127.0.0.1:8765/v1?route=dev",
        "http://127.0.0.1:8765/v1#fragment",
    ],
)
def test_worker_request_rejects_unencrypted_nonexact_inference_routes(
    base_url: str,
) -> None:
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="inference endpoint",
    ):
        worker_mod._Request.from_wire(_worker_request_payload(base_url=base_url))


def test_worker_request_accepts_exact_http_ipv4_loopback_inference_route() -> None:
    base_url = "http://127.0.0.1:8765/v1"

    request = worker_mod._Request.from_wire(_worker_request_payload(base_url=base_url))

    assert request.base_url == base_url


@pytest.mark.parametrize("codepoint", [*range(0x20), 0x7F])
def test_worker_request_rejects_every_ascii_control_in_inference_token(
    codepoint: int,
) -> None:
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="authorization token",
    ):
        worker_mod._Request.from_wire(
            _worker_request_payload(auth_token=f"prefix{chr(codepoint)}suffix")
        )


@pytest.mark.parametrize(
    "token",
    [" token", "token ", "to ken", "to\u00a0ken", "tökén"],
)
def test_worker_request_rejects_token_whitespace_or_non_ascii(
    token: str,
) -> None:
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="authorization token",
    ):
        worker_mod._Request.from_wire(_worker_request_payload(auth_token=token))


@pytest.mark.parametrize("token", [None, True, 123, b"bytes"])
def test_worker_request_requires_str_inference_token(token: Any) -> None:
    with pytest.raises(worker_mod.WorkerQualificationError):
        worker_mod._Request.from_wire(_worker_request_payload(auth_token=token))


def test_worker_request_rejects_oversized_inference_token() -> None:
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="authorization token",
    ):
        worker_mod._Request.from_wire(
            _worker_request_payload(
                auth_token="t" * (worker_mod._MAX_AUTH_TOKEN_CHARS + 1)
            )
        )


@pytest.mark.parametrize(
    "max_response_tokens",
    [None, True, 0, -1, worker_mod._MODEL_CONTEXT_SIZES[worker_mod._MODEL] + 1],
)
def test_worker_request_rejects_invalid_max_response_tokens(
    max_response_tokens: Any,
) -> None:
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="max_response_tokens",
    ):
        worker_mod._Request.from_wire(
            _worker_request_payload(max_response_tokens=max_response_tokens)
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8765/mcp",
        "http://localhost:8765/mcp",
        "http://0.0.0.0:8765/mcp",
        "http://127.0.0.1.example.test:8765/mcp",
        "http://2130706433:8765/mcp",
        "http://0x7f000001:8765/mcp",
        "http://user:pass@127.0.0.1:8765/mcp",
        "http://127.0.0.1:8765/mcp?token=x",
        "http://127.0.0.1:8765/mcp#fragment",
        "http://127.0.0.1/mcp",
        "http://127.0.0.1:8765/other",
        "http://127.0.0.1:0/mcp",
    ],
)
def test_worker_request_rejects_nonexact_mcp_endpoint(url: str) -> None:
    with pytest.raises(
        worker_mod.WorkerQualificationError,
        match="loopback endpoint",
    ):
        worker_mod._Request.from_wire(_worker_request_payload(mcp_url=url))


@pytest.mark.parametrize(
    "overrides",
    [
        {"mcp_auth_token": None},
        {"mcp_auth_token": "short"},
        {"mcp_auth_token": "m" * 32 + "\t"},
        {"mcp_auth_token": "m" * 32 + "\x7f"},
        {"mcp_tool_names": ["z", "a"]},
        {"mcp_tool_names": ["echo", "echo"]},
        {"mcp_tool_names": ["bad name"]},
        {"max_steps": 101},
    ],
)
def test_worker_request_rejects_malformed_mcp_authority(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(worker_mod.WorkerQualificationError):
        worker_mod._Request.from_wire(_worker_request_payload(**overrides))


def test_worker_plain_request_forbids_mcp_fields_and_multiple_steps() -> None:
    plain = _worker_request_payload(
        mcp_url=None,
        mcp_auth_token=None,
        mcp_tool_names=[],
        max_steps=1,
    )
    assert worker_mod._Request.from_wire(plain).mcp_url is None

    with pytest.raises(worker_mod.WorkerQualificationError):
        worker_mod._Request.from_wire({**plain, "max_steps": 2})
    with pytest.raises(worker_mod.WorkerQualificationError):
        worker_mod._Request.from_wire({**plain, "mcp_auth_token": "m" * 48})


def test_kimi_empty_visible_completion_is_retriable_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(
            final_message="",
            output_tokens=200,
        )
    )

    result = backend.invoke(_invocation())

    assert result.error.kind == "transport"
    assert result.error.retriable is True
    assert "content-free visible completion" in result.error.message


@pytest.mark.parametrize("steps_used", [True, "1", 0, 2, -1])
def test_parent_rejects_malformed_incremental_step_progress(
    steps_used: Any,
) -> None:
    latest, terminal = kimi_mod._consume_worker_line(
        json.dumps(
            {
                "protocol": kimi_mod._WORKER_PROTOCOL,
                "type": "step_progress",
                "steps_used": steps_used,
            }
        ).encode(),
        latest_usage=kimi_mod._WorkerOutcome(),
        max_steps=2,
    )

    assert latest.steps_used == 0
    assert terminal is not None
    assert terminal.error_kind == "qualification"
    assert terminal.retriable is False


def test_parent_rejects_terminal_without_step_attestation() -> None:
    payload = _success_outcome().to_wire()
    payload["protocol"] = kimi_mod._WORKER_PROTOCOL
    payload.pop("steps_used")

    _, terminal = kimi_mod._consume_worker_line(
        json.dumps(payload).encode(),
        latest_usage=kimi_mod._WorkerOutcome(),
        max_steps=2,
    )

    assert terminal is not None
    assert terminal.error_kind == "qualification"
    assert terminal.retriable is False


def test_parent_rejects_terminal_incremental_step_disagreement() -> None:
    merged = kimi_mod._merge_incremental_outcome(
        kimi_mod._WorkerOutcome(steps_used=1),
        _success_outcome(steps_used=2),
    )

    assert merged.final_message == ""
    assert merged.error_kind == "qualification"
    assert merged.retriable is False
    assert merged.steps_used == 1


@pytest.mark.parametrize(
    "raw_line",
    [
        b"[]",
        b"null",
        b"\n",
        (
            '{"protocol":"'
            + kimi_mod._WORKER_PROTOCOL
            + '","type":"usage","type":"terminal"}'
        ).encode(),
        json.dumps(
            {
                "protocol": kimi_mod._WORKER_PROTOCOL,
                "type": "future_record",
            }
        ).encode(),
    ],
)
def test_parent_rejects_nonobject_blank_and_unknown_protocol_records(
    raw_line: bytes,
) -> None:
    _, terminal = kimi_mod._consume_worker_line(
        raw_line,
        latest_usage=kimi_mod._WorkerOutcome(),
        max_steps=2,
    )

    assert terminal is not None
    assert terminal.error_kind == "qualification"
    assert terminal.retriable is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("final_message", 1),
        ("input_tokens", "15"),
        ("usage_seen", 1),
        ("configured_model", 7),
        ("stop_reason", None),
        ("error_message", False),
        ("http_status", True),
        ("retriable", 0),
        ("steps_used", True),
    ],
)
def test_parent_strictly_types_terminal_protocol_fields(
    field: str,
    value: Any,
) -> None:
    payload = _success_outcome().to_wire()
    payload["protocol"] = kimi_mod._WORKER_PROTOCOL
    payload[field] = value

    with pytest.raises(ValueError):
        kimi_mod._WorkerOutcome.from_wire(payload)


def test_parent_rejects_missing_or_unknown_terminal_fields() -> None:
    missing = _success_outcome().to_wire()
    missing.pop("error_message")
    unknown = {**_success_outcome().to_wire(), "future_field": True}

    with pytest.raises(ValueError):
        kimi_mod._WorkerOutcome.from_wire(missing)
    with pytest.raises(ValueError):
        kimi_mod._WorkerOutcome.from_wire(unknown)


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop_reason": "error"},
        {"error_kind": "transport", "error_message": "contradiction"},
        {"terminal_seen": False},
        {
            "stop_reason": "max_tokens",
            "error_kind": None,
            "error_message": "limit",
        },
        {
            "stop_reason": "content_filter",
            "error_kind": "content_rejected",
            "error_message": "rejected",
            "final_message": "must be empty",
        },
    ],
)
def test_parent_rejects_invalid_success_terminal_state_combinations(
    overrides: dict[str, Any],
) -> None:
    payload = _success_outcome(**overrides).to_wire()

    with pytest.raises(ValueError):
        kimi_mod._WorkerOutcome.from_wire(payload)


def test_parent_rejects_cached_tokens_exceeding_total_input() -> None:
    usage_record = {
        "protocol": kimi_mod._WORKER_PROTOCOL,
        "type": "usage",
        "input_tokens": 3,
        "output_tokens": 1,
        "cache_creation_tokens": 2,
        "cache_read_tokens": 2,
        "usage_valid": True,
    }
    _, terminal = kimi_mod._consume_worker_line(
        json.dumps(usage_record).encode(),
        latest_usage=kimi_mod._WorkerOutcome(),
        max_steps=2,
    )
    terminal_payload = _success_outcome(
        input_tokens=3,
        cache_creation_tokens=2,
        cache_read_tokens=2,
    ).to_wire()

    assert terminal is not None
    assert terminal.error_kind == "qualification"
    with pytest.raises(ValueError):
        kimi_mod._WorkerOutcome.from_wire(terminal_payload)


def test_parent_rejects_decreasing_or_revalidated_cumulative_usage() -> None:
    latest = kimi_mod._WorkerOutcome(
        input_tokens=10,
        output_tokens=5,
        cache_creation_tokens=2,
        cache_read_tokens=1,
        usage_seen=True,
        usage_valid=False,
        steps_used=1,
    )
    base_record = {
        "protocol": kimi_mod._WORKER_PROTOCOL,
        "type": "usage",
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_tokens": 2,
        "cache_read_tokens": 1,
        "usage_valid": False,
    }

    for overrides in (
        {"input_tokens": 9},
        {"output_tokens": 4},
        {"cache_creation_tokens": 1},
        {"cache_read_tokens": 0},
        {"usage_valid": True},
    ):
        _, terminal = kimi_mod._consume_worker_line(
            json.dumps({**base_record, **overrides}).encode(),
            latest_usage=latest,
            max_steps=2,
        )
        assert terminal is not None
        assert terminal.error_kind == "qualification"


@pytest.mark.parametrize(
    "record",
    [
        {
            "protocol": kimi_mod._WORKER_PROTOCOL,
            "type": "usage",
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        },
        {
            "protocol": kimi_mod._WORKER_PROTOCOL,
            "type": "tool_event",
            "kind": "tool_call",
            "name": "echo",
            "call_id": "call-1",
        },
    ],
)
def test_parent_rejects_usage_or_tool_event_before_step_progress(
    record: dict[str, Any],
) -> None:
    _, terminal = kimi_mod._consume_worker_line(
        json.dumps(record).encode(),
        latest_usage=kimi_mod._WorkerOutcome(),
        max_steps=2,
    )

    assert terminal is not None
    assert terminal.error_kind == "qualification"


def test_parent_derives_one_usage_settlement_per_begun_step() -> None:
    def consume(
        latest: kimi_mod._WorkerOutcome,
        payload: dict[str, Any],
    ) -> tuple[kimi_mod._WorkerOutcome, kimi_mod._WorkerOutcome | None]:
        return kimi_mod._consume_worker_line(
            json.dumps(
                {"protocol": kimi_mod._WORKER_PROTOCOL, **payload}
            ).encode(),
            latest_usage=latest,
            max_steps=2,
        )

    latest, terminal = consume(
        kimi_mod._WorkerOutcome(),
        {"type": "step_progress", "steps_used": 1},
    )
    assert terminal is None
    latest, terminal = consume(
        latest,
        {
            "type": "usage",
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        },
    )
    assert terminal is None
    assert latest.steps_used == 1
    assert latest.settled_usage_steps == 1
    assert len(latest.pricing_segments) == 1
    assert latest.pricing_segments[0].input_tokens == 10
    assert latest.pricing_segments[0].output_tokens == 2

    latest, duplicate_terminal = consume(
        latest,
        {
            "type": "usage",
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        },
    )
    assert duplicate_terminal is None
    assert latest.settled_usage_steps == 1
    assert len(latest.pricing_segments) == 1

    _, changed_duplicate_terminal = consume(
        latest,
        {
            "type": "usage",
            "input_tokens": 11,
            "output_tokens": 2,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        },
    )
    assert changed_duplicate_terminal is not None
    assert changed_duplicate_terminal.error_kind == "qualification"

    latest, terminal = consume(
        latest,
        {"type": "step_progress", "steps_used": 2},
    )
    assert terminal is None
    _, delayed_duplicate_terminal = consume(
        latest,
        {
            "type": "usage",
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        },
    )
    assert delayed_duplicate_terminal is not None
    assert delayed_duplicate_terminal.error_kind == "qualification"
    assert delayed_duplicate_terminal.usage_valid is False

    latest, terminal = consume(
        latest,
        {
            "type": "usage",
            "input_tokens": 25,
            "output_tokens": 6,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        },
    )
    assert terminal is None
    assert latest.steps_used == 2
    assert latest.settled_usage_steps == 2
    assert len(latest.pricing_segments) == 2
    assert latest.pricing_segments[1].input_tokens == 15
    assert latest.pricing_segments[1].output_tokens == 4


def test_parent_step_segments_preserve_cache_deltas_and_exact_cost() -> None:
    latest = kimi_mod._WorkerOutcome(steps_used=1)
    latest = kimi_mod._settle_usage_checkpoint(
        latest,
        (20, 3, 4, 6),
        usage_valid=True,
    )
    latest = replace(latest, steps_used=2)
    latest = kimi_mod._settle_usage_checkpoint(
        latest,
        (35, 8, 7, 10),
        usage_valid=True,
    )

    assert [
        (
            segment.input_tokens,
            segment.output_tokens,
            segment.cache_creation_tokens,
            segment.cache_read_tokens,
        )
        for segment in latest.pricing_segments
    ] == [(20, 3, 4, 6), (15, 5, 3, 4)]
    assert latest.pricing_segments[1].cost_usd == pytest.approx(
        kimi_mod.compute_cost(
            _MODEL,
            input_tokens=15,
            output_tokens=5,
            cache_creation_tokens=3,
            cache_read_tokens=4,
        )
    )


def test_parent_rejects_usage_that_skips_a_begun_step() -> None:
    latest = kimi_mod._WorkerOutcome(steps_used=2)
    raw = json.dumps(
        {
            "protocol": kimi_mod._WORKER_PROTOCOL,
            "type": "usage",
            "input_tokens": 25,
            "output_tokens": 6,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "usage_valid": True,
        }
    ).encode()

    _, terminal = kimi_mod._consume_worker_line(
        raw,
        latest_usage=latest,
        max_steps=2,
    )

    assert terminal is not None
    assert terminal.error_kind == "qualification"
    assert terminal.usage_valid is False


def test_parent_rejects_terminal_usage_or_tool_trace_disagreement() -> None:
    latest = kimi_mod._WorkerOutcome(
        input_tokens=15,
        output_tokens=7,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        usage_seen=True,
        usage_valid=True,
        settled_usage_steps=1,
        tool_events=(("tool_call", "echo", "call-1"),),
        steps_used=1,
    )

    usage_mismatch = kimi_mod._merge_incremental_outcome(
        latest,
        _success_outcome(input_tokens=16, tool_events=latest.tool_events),
    )
    tool_mismatch = kimi_mod._merge_incremental_outcome(
        latest,
        _success_outcome(tool_events=()),
    )

    assert usage_mismatch.error_kind == "qualification"
    assert usage_mismatch.usage_valid is False
    assert tool_mismatch.error_kind == "qualification"
    assert tool_mismatch.usage_valid is False


def test_parent_overlays_incrementals_only_on_empty_nonterminal_classifier() -> None:
    latest = kimi_mod._WorkerOutcome(
        input_tokens=15,
        output_tokens=7,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        usage_seen=True,
        usage_valid=True,
        settled_usage_steps=1,
        tool_events=(
            ("tool_call", "echo", "call-1"),
            ("tool_result", "echo", "call-1"),
        ),
        steps_used=1,
    )
    classifier = kimi_mod._WorkerOutcome(
        stop_reason="error",
        error_kind="qualification",
        error_message="compaction guard",
        retriable=False,
        steps_used=1,
    )

    merged = kimi_mod._merge_incremental_outcome(latest, classifier)

    assert merged.error_message == "compaction guard"
    assert merged.input_tokens == 15
    assert merged.usage_valid is True
    assert merged.settled_usage_steps == 1
    assert merged.tool_events == latest.tool_events


@pytest.mark.parametrize(
    ("error_kind", "settled_usage_steps", "expected_complete"),
    [
        ("timeout", 0, False),
        ("max_turns", 1, True),
    ],
)
def test_nonterminal_usage_is_complete_only_when_every_begun_step_settled(
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
    settled_usage_steps: int,
    expected_complete: bool,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    outcome = kimi_mod._WorkerOutcome(
        input_tokens=25,
        output_tokens=6,
        usage_seen=True,
        usage_valid=True,
        terminal_seen=False,
        configured_model=_MODEL if error_kind == "max_turns" else None,
        served_model=_MODEL if error_kind == "max_turns" else None,
        stop_reason="timeout" if error_kind == "timeout" else "max_turns",
        error_kind=error_kind,
        error_message=f"bounded {error_kind}",
        retriable=False,
        steps_used=1,
        settled_usage_steps=settled_usage_steps,
    )
    backend = KimiBackend(
        _worker_runner=lambda *_args, **_kwargs: outcome,
    )
    result = backend.invoke(_invocation())

    assert result.usage.input_tokens == 25
    assert result.usage.output_tokens == 6
    assert result.completeness.token_usage_complete is expected_complete
    assert result.completeness.cost_complete is expected_complete
    assert result.completeness.terminal_complete is False
    assert result.error is not None


@pytest.mark.parametrize(
    ("settled_usage_steps", "expected_complete"),
    [(1, False), (2, True)],
)
def test_terminal_usage_requires_every_begun_step_settled(
    monkeypatch: pytest.MonkeyPatch,
    settled_usage_steps: int,
    expected_complete: bool,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    outcome = _success_outcome(
        steps_used=2,
        settled_usage_steps=settled_usage_steps,
    )
    backend = KimiBackend()
    invocation = _invocation()
    assert invocation.policy is not None
    result = backend._result_from_outcome(
        invocation,
        invocation.policy,
        outcome,
        base_url=_BASE_URL,
        events=[],
        tool_authority_digest=None,
    )

    assert result.error is None
    assert result.completeness.terminal_complete is True
    assert result.completeness.token_usage_complete is expected_complete
    assert result.completeness.cost_complete is expected_complete


def test_kimi_result_exposes_complete_step_pricing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    outcome = kimi_mod._WorkerOutcome(steps_used=1)
    outcome = kimi_mod._settle_usage_checkpoint(
        outcome,
        (25, 6, 3, 7),
        usage_valid=True,
    )
    outcome = replace(
        outcome,
        final_message="done",
        terminal_seen=True,
        configured_model=_MODEL,
        served_model=_MODEL,
        stop_reason="end_turn",
    )
    backend = KimiBackend()
    invocation = _invocation()
    assert invocation.policy is not None

    result = backend._result_from_outcome(
        invocation,
        invocation.policy,
        outcome,
        base_url=_BASE_URL,
        events=[],
        tool_authority_digest=None,
    )

    assert result.usage.cache_creation_tokens == 3
    assert result.usage.cache_read_tokens == 7
    assert result.pricing_evidence.complete is True
    assert result.pricing_evidence.segments == outcome.pricing_segments


def test_kimi_incomplete_attempt_suppresses_paid_retry_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    monkeypatch.setattr(kimi_mod.time, "sleep", lambda _seconds: None)
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="response envelope was lost",
            retriable=True,
        )

    backend = KimiBackend(_worker_runner=runner)
    result = backend.invoke(
        _invocation(
                policy=BackendPolicy(
                    retry_budget=2,
                    # This test exercises aggregate retry evidence, not the
                    # wall-deadline boundary. Keep enough headroom for the
                    # full coverage suite's process-heavy load.
                    timeout_seconds=30.0,
                    on_cancellation="convert_to_backend_error",
                    on_max_turns="raise",
                )
        )
    )

    assert calls == 1
    assert result.error is not None
    assert result.error.retriable is False
    assert "aggregate accounting" in result.error.message
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.cost_usd == 0.0
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "terminal",
    ]


def test_kimi_complete_attempt_still_suppresses_ambiguous_multi_attempt_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    first = _success_outcome(
        final_message="",
        input_tokens=11,
        output_tokens=5,
        cache_creation_tokens=1,
        cache_read_tokens=4,
        stop_reason="error",
        error_kind="transport",
        error_message="retry after a complete usage envelope",
        retriable=True,
    )
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return first

    backend = KimiBackend(_worker_runner=runner)
    monkeypatch.setattr(
        backend._close_event,
        "wait",
        lambda _seconds: False,
    )

    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=2,
                timeout_seconds=30.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    expected_cost = kimi_mod.compute_cost(
        _MODEL,
        input_tokens=11,
        output_tokens=5,
        cache_creation_tokens=1,
        cache_read_tokens=4,
    )
    assert calls == 1
    assert result.error is not None
    assert result.error.retriable is False
    assert "aggregate accounting" in result.error.message
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 5
    assert result.usage.cost_usd == pytest.approx(expected_cost)
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "usage",
        "terminal",
    ]


def test_kimi_retry_suppression_starts_no_backoff_or_second_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    now = [100.0]
    worker_timeouts: list[float] = []
    def runner(_request, timeout, **_kwargs):
        worker_timeouts.append(timeout)
        now[0] += 0.5
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="retry once",
            retriable=True,
        )

    backend = KimiBackend(_worker_runner=runner)
    monkeypatch.setattr(kimi_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        backend._close_event,
        "wait",
        lambda seconds: now.__setitem__(0, now[0] + seconds) or False,
    )

    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=2,
                timeout_seconds=3.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert result.error is not None
    assert result.error.retriable is False
    assert worker_timeouts == pytest.approx([3.0])
    assert now[0] == pytest.approx(100.5)


def test_kimi_retry_suppression_preserves_first_attempt_step_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    requested_step_limits: list[int] = []
    def runner(request, _timeout, **_kwargs):
        requested_step_limits.append(request.max_steps)
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="retry after two model steps",
            retriable=True,
            steps_used=2,
        )

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=runner,
    )
    backend.attach_toolset(_ToolSet())
    monkeypatch.setattr(
        backend._close_event,
        "wait",
        lambda _seconds: False,
    )
    invocation = replace(
        _invocation(
            policy=BackendPolicy(
                retry_budget=2,
                timeout_seconds=10.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        ),
        max_turns=3,
    )

    result = backend.invoke(invocation)

    assert result.error is not None
    assert result.error.retriable is False
    assert requested_step_limits == [3]
    backend.close()


def test_kimi_step_budget_exhaustion_suppresses_physical_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    requested_step_limits: list[int] = []

    def runner(request, _timeout, **_kwargs):
        requested_step_limits.append(request.max_steps)
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="provider response was lost",
            retriable=True,
            steps_used=2,
        )

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=runner,
    )
    backend.attach_toolset(_ToolSet())

    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=3,
                timeout_seconds=10.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert requested_step_limits == [2]
    assert result.error.retriable is False
    assert "logical step budget is exhausted" in result.error.message
    backend.close()


def test_kimi_accounting_suppression_precedes_retry_backoff_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    now = [50.0]
    calls = 0

    def runner(_request, _timeout, **_kwargs):
        nonlocal calls
        calls += 1
        now[0] += 0.75
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="retry would exceed the deadline",
            retriable=True,
        )

    backend = KimiBackend(_worker_runner=runner)
    monkeypatch.setattr(kimi_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        backend._close_event,
        "wait",
        lambda seconds: now.__setitem__(0, now[0] + seconds) or False,
    )

    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=3,
                timeout_seconds=1.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert calls == 1
    assert result.stop_reason == "error"
    assert result.error.retriable is False
    assert "aggregate accounting" in result.error.message
    assert not any(
        event.kind in {"limit", "retry"}
        for event in result.execution_trace.events
    )


def test_kimi_parent_redacts_worker_error_before_debug_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _qualified_versions(monkeypatch)
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        log_dir=tmp_path,
        _worker_runner=lambda *_args, **_kwargs: kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message=f"Authorization: Bearer {_TOKEN}",
            retriable=False,
        ),
    )

    result = backend.invoke(_invocation())

    assert _TOKEN not in result.error.message
    record = next(tmp_path.glob("sdk_*_kimi.json"))
    assert _TOKEN not in record.read_text()


def test_kimi_text_capability_rejects_toolset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)
    backend = KimiBackend()

    backend.attach_toolset(None)
    with pytest.raises(ValueError, match="text capability"):
        backend.attach_toolset(object())


def test_kimi_attached_toolset_uses_authenticated_ephemeral_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    captured: list[kimi_mod._WorkerRequest] = []

    def runner(request, _timeout, **_kwargs):
        captured.append(request)
        assert _call_attached_mcp(request) == "through-bridge"
        return _success_outcome(
            tool_events=(
                ("tool_call", "echo", "call-bridge-1"),
                ("tool_result", "echo", "call-bridge-1"),
            )
        )

    toolset = _ToolSet()
    backend = KimiBackend(
        capability="read_only",
        _worker_runner=runner,
    )
    backend.attach_toolset(toolset, context_budget=10_000)

    result = backend.invoke(_invocation())

    assert result.error is None
    assert toolset.calls == [("echo", {"value": "through-bridge"})]
    assert len(captured) == 1
    request = captured[0]
    assert request.max_steps == 2
    assert request.mcp_tool_names == ("echo",)
    assert request.mcp_auth_token
    assert request.mcp_auth_token != _TOKEN
    assert request.mcp_auth_token not in request.mcp_url
    parsed = urlsplit(request.mcp_url)
    assert parsed.hostname == "127.0.0.1"
    assert parsed.port is not None
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", parsed.port)) != 0
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "tool_call",
        "tool_result",
        "usage",
        "terminal",
    ]
    request_event, call_event, result_event = result.execution_trace.events[:3]
    assert request_event.child_id == "kimi-attempt-1"
    assert call_event.parent_id == request_event.child_id
    assert result_event.parent_id == request_event.child_id
    assert call_event.child_id == result_event.child_id == "call-bridge-1"
    assert result.execution_trace.tool_authority_digest.startswith("sha256:")
    assert result.completeness.trace_complete is True
    assert backend._active_mcp_servers == set()
    backend.close()


def test_kimi_authority_digest_binds_frozen_exposed_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    class MutableDescriptorToolSet(_ToolSet):
        def __init__(self) -> None:
            super().__init__()
            self.catalogue_reads = 0
            self.history: list[list[dict[str, Any]]] = []

        def to_anthropic_tools(self) -> list[dict[str, Any]]:
            self.catalogue_reads += 1
            descriptors = super().to_anthropic_tools()
            descriptors[0]["description"] = f"mutable-version-{self.catalogue_reads}"
            self.history.append(json.loads(json.dumps(descriptors)))
            return descriptors

    toolset = MutableDescriptorToolSet()
    real_digest = kimi_mod._tool_authority_digest
    frozen_snapshots: list[Any] = []
    original_read_counts: list[tuple[int, int]] = []

    def capture_digest(frozen_toolset, **kwargs):
        before = toolset.catalogue_reads
        frozen_snapshots.append(frozen_toolset)
        digest = real_digest(frozen_toolset, **kwargs)
        original_read_counts.append((before, toolset.catalogue_reads))
        return digest

    monkeypatch.setattr(kimi_mod, "_tool_authority_digest", capture_digest)
    backend = KimiBackend(
        capability="read_only",
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(),
    )
    backend.attach_toolset(toolset, context_budget=321)

    result = backend.invoke(_invocation())

    assert result.error is None
    assert len(frozen_snapshots) == 1
    assert original_read_counts == [(toolset.catalogue_reads, toolset.catalogue_reads)]
    frozen = frozen_snapshots[0]
    assert frozen.to_anthropic_tools() == toolset.history[-1]
    assert result.execution_trace.tool_authority_digest == real_digest(
        frozen,
        capability="read_only",
        context_budget=321,
    )
    backend.close()


def test_kimi_frozen_authority_binds_and_enforces_one_scope_snapshot(
    tmp_path: Path,
) -> None:
    from crucible.backends.tool_authority import snapshot_tool_authority
    from crucible.tools import ToolContext
    from crucible.tools._isolated_mutation_host import (
        build_isolated_toolset_spec,
    )
    from crucible.tools.write import build_write_tools

    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("src",),
    )
    toolset = build_write_tools(context).filter_to({"write_source"})
    names, writes = kimi_mod._tool_inventory(toolset)
    snapshot = snapshot_tool_authority(toolset)
    frozen = kimi_mod._FrozenMcpToolset(
        toolset,
        names=names,
        write_names=writes,
        authority_snapshot=snapshot,
    )
    first_digest = kimi_mod._tool_authority_digest(
        frozen,
        capability="workspace_write",
        context_budget=321,
    )

    # Mutable ToolContext state after the authority freeze must neither change
    # the digest nor expand what the isolated mutation child enforces.
    context.write_scope = ("other",)
    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=writes,
        authority_snapshot=snapshot,
    )
    assert spec.binding_for("write_source").context.write_scope == ("src",)
    assert (
        kimi_mod._tool_authority_digest(
            frozen,
            capability="workspace_write",
            context_budget=321,
        )
        == first_digest
    )

    changed = build_write_tools(context).filter_to({"write_source"})
    changed_names, changed_writes = kimi_mod._tool_inventory(changed)
    changed_frozen = kimi_mod._FrozenMcpToolset(
        changed,
        names=changed_names,
        write_names=changed_writes,
    )
    assert (
        kimi_mod._tool_authority_digest(
            changed_frozen,
            capability="workspace_write",
            context_budget=321,
        )
        != first_digest
    )


def test_kimi_context_budget_resets_after_retry_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    responses: list[str] = []
    calls = 0

    def runner(request, _timeout, **_kwargs):
        nonlocal calls
        calls += 1
        responses.append(_call_attached_mcp(request, value="éé"))
        event_id = f"call-{calls}"
        tool_events = (
            ("tool_call", "echo", event_id),
            ("tool_result", "echo", event_id),
        )
        if calls == 1:
            return kimi_mod._WorkerOutcome(
                stop_reason="error",
                error_kind="transport",
                error_message="retry after complete read",
                retriable=True,
                tool_events=tool_events,
            )
        return _success_outcome(tool_events=tool_events)

    toolset = _ToolSet()
    backend = KimiBackend(
        capability="read_only",
        _worker_runner=runner,
    )
    backend.attach_toolset(toolset, context_budget=5)
    monkeypatch.setattr(
        backend._close_event,
        "wait",
        lambda _seconds: False,
    )
    invocation = _invocation(
        policy=BackendPolicy(
            retry_budget=2,
            timeout_seconds=10.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        )
    )

    suppressed = backend.invoke(invocation)
    fresh = backend.invoke(invocation)

    assert suppressed.error is not None
    assert suppressed.error.retriable is False
    assert calls == 2
    assert len(responses) == 2
    assert all("[TRUNCATED]" not in response for response in responses)
    assert fresh.completeness.trace_complete is True
    backend.close()


@pytest.mark.parametrize(
    "tool_events",
    [
        (
            ("tool_call", "echo", "call-a"),
            ("tool_call", "echo", "call-b"),
            ("tool_result", "echo", "call-a"),
            ("tool_result", "echo", "call-a"),
        ),
        (("tool_result", "echo", "call-a"),),
        (
            ("tool_call", "echo", "call-a"),
            ("tool_result", "other", "call-a"),
        ),
        (
            ("tool_call", "other", "call-a"),
            ("tool_result", "other", "call-a"),
        ),
        (("tool_call", "echo", "call-a"),),
    ],
)
def test_kimi_invalid_per_attempt_tool_trace_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tool_events: tuple[tuple[str, str, str], ...],
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        capability="read_only",
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(
            tool_events=tool_events
        ),
    )
    backend.attach_toolset(_ToolSet())

    result = backend.invoke(_invocation())

    assert result.final_message == ""
    assert result.error.retriable is False
    assert "invalid per-attempt tool event sequence" in result.error.message
    assert result.completeness.trace_complete is False
    backend.close()


def test_kimi_balanced_read_attempt_still_suppresses_physical_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    calls = 0

    def runner(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="retry after complete read",
            retriable=True,
            tool_events=(
                ("tool_call", "echo", "call-first"),
                ("tool_result", "echo", "call-first"),
            ),
        )

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=runner,
    )
    backend.attach_toolset(_ToolSet())
    monkeypatch.setattr(
        backend._close_event,
        "wait",
        lambda _seconds: False,
    )

    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=2,
                timeout_seconds=10.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert calls == 1
    assert result.error is not None
    assert result.error.retriable is False
    assert result.final_message == ""
    assert [
        (event.kind, event.child_id)
        for event in result.execution_trace.events
        if event.kind in {"tool_call", "tool_result"}
    ] == [
        ("tool_call", "call-first"),
        ("tool_result", "call-first"),
    ]
    backend.close()


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_multistep_replays_reasoning_and_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _ScriptedKimiEndpoint,
    )
    endpoint.requests = []
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    toolset = _ToolSet()
    backend = KimiBackend(
        capability="read_only",
        expected_base_url=base_url,
    )
    backend.attach_toolset(toolset)
    invocation = replace(
        _invocation(),
        user="Use echo with scripted-nonce, then report what it returned.",
        max_turns=4,
        policy=BackendPolicy(
            retry_budget=1,
            timeout_seconds=20.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    assert result.error is None
    assert result.final_message == "scripted-nonce observed"
    assert toolset.calls == [("echo", {"value": "scripted-nonce"})]
    assert len(endpoint.requests) == 2
    first, second = endpoint.requests
    for request in (first, second):
        assert request["model"] == _MODEL
        assert request["max_completion_tokens"] == 32_768
        assert request["reasoning_effort"] == "max"
        assert request["prompt_cache_options"] == {
            "mode": "implicit",
            "ttl": "5m",
        }
        assert "max_tokens" not in request
        assert "thinking" not in request
    assert isinstance(first["prompt_cache_key"], str)
    assert first["prompt_cache_key"]
    assert second["prompt_cache_key"] == first["prompt_cache_key"]
    assert [tool["function"]["name"] for tool in first["tools"]] == ["echo"]
    assistant = next(
        message for message in second["messages"] if message["role"] == "assistant"
    )
    assert assistant["reasoning_content"] == "I must use the confined echo tool."
    assert assistant["tool_calls"][0]["function"]["name"] == "echo"
    tool_message = next(
        message for message in second["messages"] if message["role"] == "tool"
    )
    assert "scripted-nonce" in str(tool_message["content"])
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "tool_call",
        "tool_result",
        "usage",
        "terminal",
    ]
    assert result.completeness.trace_complete is True


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_project_locked_worker_preserves_settled_max_steps_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _ScriptedKimiEndpoint,
    )
    endpoint.requests = []
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.delenv("CRUCIBLE_KIMI_WORKER_PYTHON", raising=False)
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    toolset = _ToolSet()
    backend = KimiBackend(
        capability="read_only",
        expected_base_url=base_url,
        worker_source="project_locked",
    )
    assert backend.worker_runtime_fingerprint is not None
    backend.attach_toolset(toolset)
    invocation = replace(
        _invocation(),
        user="Use echo with scripted-nonce, then report what it returned.",
        max_turns=1,
        policy=BackendPolicy(
            retry_budget=1,
            timeout_seconds=20.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    assert result.final_message == ""
    assert result.stop_reason == "max_turns"
    assert result.error is not None
    assert result.error.kind == "max_turns"
    assert result.error.retriable is False
    assert result.execution_trace.requested_model == _MODEL
    assert result.execution_trace.served_model == _MODEL
    assert len(endpoint.requests) == 1
    assert toolset.calls == [("echo", {"value": "scripted-nonce"})]
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 5
    assert result.usage.cache_creation_tokens == 0
    assert result.usage.cache_read_tokens == 0
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert result.completeness.terminal_complete is False
    assert result.completeness.trace_complete is True
    assert result.pricing_evidence.complete is True
    assert len(result.pricing_evidence.segments) == 1
    segment = result.pricing_evidence.segments[0]
    assert segment.input_tokens == result.usage.input_tokens
    assert segment.output_tokens == result.usage.output_tokens
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "tool_call",
        "tool_result",
        "usage",
        "terminal",
    ]


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
@pytest.mark.parametrize("partial_text", ["", "partial billed response"])
def test_real_worker_normalizes_partial_and_terminal_only_finish_reasons(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
    partial_text: str,
) -> None:
    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _NativeStreamEndpoint,
    )
    endpoint.requests = []
    endpoint.chunks = [
        *(
            [_native_chunk(delta={"content": partial_text})]
            if partial_text
            else []
        ),
        _native_chunk(finish_reason=finish_reason),
        _native_usage_chunk(),
    ]
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(expected_base_url=base_url)
    try:
        result = backend.invoke(
            _invocation(
                policy=BackendPolicy(
                    retry_budget=1,
                    timeout_seconds=20.0,
                    on_cancellation="convert_to_backend_error",
                    on_max_turns="raise",
                )
            )
        )
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    expected_stop = "max_tokens" if finish_reason == "length" else "content_filter"
    expected_error = "transport" if finish_reason == "length" else "content_rejected"
    expected_final = partial_text if finish_reason == "length" else ""
    assert result.final_message == expected_final
    assert result.stop_reason == expected_stop
    assert result.error.kind == expected_error
    assert result.error.retriable is False
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 4
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert result.completeness.terminal_complete is True
    assert result.execution_trace.served_model == _MODEL
    assert len(endpoint.requests) == 1


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_real_worker_never_dispatches_partial_tool_on_nonstop_terminal(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _NativeStreamEndpoint,
    )
    endpoint.requests = []
    endpoint.chunks = [
        _native_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-must-not-run",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"value":"partial',
                        },
                    }
                ]
            }
        ),
        _native_chunk(finish_reason=finish_reason),
        _native_usage_chunk(),
    ]
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    toolset = _ToolSet()
    backend = KimiBackend(
        capability="read_only",
        expected_base_url=base_url,
    )
    backend.attach_toolset(toolset)
    try:
        result = backend.invoke(
            _invocation(
                policy=BackendPolicy(
                    retry_budget=1,
                    timeout_seconds=20.0,
                    on_cancellation="convert_to_backend_error",
                    on_max_turns="raise",
                )
            )
        )
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    expected_stop = "max_tokens" if finish_reason == "length" else "content_filter"
    assert result.stop_reason == expected_stop
    assert result.error.retriable is False
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 4
    assert toolset.calls == []
    assert not {
        event.kind
        for event in result.execution_trace.events
    } & {"tool_call", "tool_result"}
    assert len(endpoint.requests) == 1


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_fails_before_hidden_auto_compaction_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _ScriptedKimiEndpoint,
    )
    endpoint.requests = []
    # CLI 1.12 compacts once context + 50,000 reserved tokens reaches the
    # configured 1,048,576-token window.
    endpoint.first_prompt_tokens = 1_000_000
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    toolset = _ToolSet()
    backend = KimiBackend(
        capability="read_only",
        expected_base_url=base_url,
    )
    backend.attach_toolset(toolset)
    invocation = replace(
        _invocation(),
        user="Use echo with scripted-nonce, then report what it returned.",
        max_turns=3,
        policy=BackendPolicy(
            retry_budget=1,
            timeout_seconds=20.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    assert result.error is not None
    assert result.error.retriable is False
    assert "automatic context compaction is disabled" in result.error.message
    assert toolset.calls == [("echo", {"value": "scripted-nonce"})]
    # One accepted tool-calling step is the only physical generation. The
    # automatic compaction and following agent step are both denied locally.
    assert len(endpoint.requests) == 1


@pytest.mark.matrix_worker
@pytest.mark.usefixtures("require_kimi_worker")
def test_real_worker_disables_hidden_retries_for_retriable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _RetriableMoonshotEndpoint,
    )
    endpoint.requests = []
    endpoint_thread = threading.Thread(
        target=endpoint.serve_forever,
        daemon=True,
    )
    endpoint_thread.start()
    base_url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    monkeypatch.setenv("KIMI_BASE_URL", base_url)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(expected_base_url=base_url)
    invocation = _invocation(
        policy=BackendPolicy(
            retry_budget=1,
            timeout_seconds=20.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        )
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()
        endpoint.shutdown()
        endpoint.server_close()
        endpoint_thread.join(timeout=2.0)

    assert result.error is not None
    assert result.error.retriable is True
    assert len(endpoint.requests) == 1


def test_kimi_read_only_capability_rejects_mutating_toolset() -> None:
    backend = KimiBackend(capability="read_only")

    with pytest.raises(
        ValueError,
        match="read_only capability cannot attach mutating tools",
    ):
        backend.attach_toolset(_ToolSet(writes={"echo"}))


def test_kimi_chat_with_tools_bounds_sdk_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    captured: list[kimi_mod._WorkerRequest] = []

    def runner(request, _timeout, **_kwargs):
        captured.append(request)
        return _success_outcome()

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=runner,
    )
    result = backend.chat_with_tools(
        _invocation(),
        _ToolSet(),
        max_rounds=1,
        max_response_tokens=1_234,
        retry_budget=1,
    )

    assert result.error is None
    assert captured[0].max_steps == 1
    assert captured[0].max_response_tokens == 1_234
    with pytest.raises(ValueError, match="max_rounds"):
        backend.chat_with_tools(_invocation(), _ToolSet(), max_rounds=False)
    with pytest.raises(ValueError, match="max_response_tokens"):
        backend.chat_with_tools(
            _invocation(),
            _ToolSet(),
            max_response_tokens=kimi_mod._MODEL_CONTEXT_SIZES[kimi_mod._MODEL] + 1,
        )
    backend.close()


def test_kimi_chat_with_tools_defaults_to_route_owned_response_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    captured: list[kimi_mod._WorkerRequest] = []
    backend = KimiBackend(
        capability="read_only",
        max_response_tokens=12_345,
        _worker_runner=lambda request, *_args, **_kwargs: (
            captured.append(request) or _success_outcome()
        ),
    )

    result = backend.chat_with_tools(
        _invocation(),
        _ToolSet(),
        max_rounds=1,
        retry_budget=1,
    )

    assert result.error is None
    assert captured[0].max_response_tokens == 12_345
    backend.close()


def test_kimi_chat_with_tools_cannot_widen_route_owned_response_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    backend = KimiBackend(
        capability="read_only",
        max_response_tokens=1_000,
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(),
    )

    with pytest.raises(ValueError, match="route-owned bound"):
        backend.chat_with_tools(
            _invocation(),
            _ToolSet(),
            max_response_tokens=1_001,
        )
    backend.close()


def test_kimi_does_not_treat_unexecuted_mutation_request_as_started(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    calls = 0

    def runner(_request, _timeout, **_kwargs):
        nonlocal calls
        calls += 1
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="provider response was lost",
            retriable=True,
            tool_events=(
                ("tool_call", "write_source", "call-write-1"),
                ("tool_result", "write_source", "call-write-1"),
            ),
        )

    backend = KimiBackend(
        capability="workspace_write",
        _worker_runner=runner,
    )
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("output.txt",),
    )
    backend.attach_toolset(
        build_write_tools(context).filter_to({"write_source"})
    )
    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=1,
                timeout_seconds=2.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert calls == 1
    assert result.error.retriable is True
    assert result.error.message == "provider response was lost"
    assert result.completeness.terminal_complete is False
    assert result.completeness.trace_complete is True
    backend.close()


def test_kimi_host_attestation_suppresses_retry_when_worker_loses_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    calls = 0

    async def write_through_bridge(request: kimi_mod._WorkerRequest) -> str:
        assert request.mcp_url is not None
        assert request.mcp_auth_token is not None
        async with streamablehttp_client(
            request.mcp_url,
            headers={
                "Authorization": f"Bearer {request.mcp_auth_token}",
            },
        ) as (read, write, _info):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(
                    "write_source",
                    {
                        "file_path": "output.txt",
                        "content": "settled\n",
                    },
                )
                return response.content[0].text

    def runner(request, _timeout, **_kwargs):
        nonlocal calls
        calls += 1
        assert "OK: wrote" in asyncio.run(write_through_bridge(request))
        # Simulate a provider/worker transport loss after the mutation host
        # attested execution but before the SDK trace reached the parent.
        return kimi_mod._WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message="worker died after write result",
            retriable=True,
            tool_events=(),
        )

    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("output.txt",),
    )
    backend = KimiBackend(
        capability="workspace_write",
        _worker_runner=runner,
    )
    backend.attach_toolset(
        build_write_tools(context).filter_to({"write_source"})
    )
    result = backend.invoke(
        _invocation(
            policy=BackendPolicy(
                retry_budget=3,
                timeout_seconds=5.0,
                on_cancellation="convert_to_backend_error",
                on_max_turns="raise",
            )
        )
    )

    assert calls == 1
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "settled\n"
    assert result.error.retriable is False
    assert "retry suppressed after a mutating tool call" in result.error.message
    assert result.completeness.trace_complete is False
    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    backend.close()


_READ_CALL = ("grep_symbol", {"pattern": "absent"})
_WRITE_CALL = ("write_source", {"file_path": "output.txt", "content": "progress\n"})
# A write outside the write scope: the host refuses it and nothing is written.
_REFUSED_WRITE_CALL = (
    "write_source",
    {"file_path": "elsewhere.txt", "content": "refused\n"},
)


def _progress_bounded_writer(tmp_path: Path, *, max_consecutive_read_calls: int) -> Any:
    from crucible.tools import ToolContext
    from crucible.tools._dispatch_budget import ProgressBoundedMutatingToolSet
    from crucible.tools.profiles import build_autonomous_implementer_profile

    return ProgressBoundedMutatingToolSet(
        build_autonomous_implementer_profile(
            ToolContext(
                workspace_root=tmp_path,
                repo_root=tmp_path,
                write_scope=("output.txt",),
            )
        ),
        max_consecutive_read_calls=max_consecutive_read_calls,
    )


def _worker_that_calls_through_the_bridge(
    script: tuple[tuple[str, dict[str, Any]], ...],
    replies: list[str],
) -> Any:
    """A worker runner that makes the scripted calls on the attached bridge."""

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def call_all(request: kimi_mod._WorkerRequest) -> None:
        assert request.mcp_url is not None
        assert request.mcp_auth_token is not None
        async with streamablehttp_client(
            request.mcp_url,
            headers={"Authorization": f"Bearer {request.mcp_auth_token}"},
        ) as (read, write, _info):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for name, arguments in script:
                    response = await session.call_tool(name, arguments)
                    replies.append(response.content[0].text)

    def runner(request, _timeout, **_kwargs):
        asyncio.run(call_all(request))
        return _success_outcome(
            final_message="done",
            tool_events=tuple(
                (kind, name, f"call-{index}")
                for index, (name, _arguments) in enumerate(script)
                for kind in ("tool_call", "tool_result")
            ),
        )

    return runner


_WRITER_POLICY = BackendPolicy(
    retry_budget=1,
    timeout_seconds=60.0,
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)


def test_kimi_hosted_write_between_reads_resets_the_consecutive_read_streak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    # Two reads reach the bound of two; the write that runs in the isolated
    # host is progress, so the two reads after it are admitted.
    replies: list[str] = []
    bounded = _progress_bounded_writer(tmp_path, max_consecutive_read_calls=2)
    backend = KimiBackend(
        capability="workspace_write",
        _worker_runner=_worker_that_calls_through_the_bridge(
            (_READ_CALL, _READ_CALL, _WRITE_CALL, _READ_CALL, _READ_CALL),
            replies,
        ),
    )
    backend.attach_toolset(bounded)

    result = backend.invoke(_invocation(policy=_WRITER_POLICY))

    assert len(replies) == 5
    assert not any("TerminalToolDispatchError" in reply for reply in replies)
    assert result.error is None
    assert (result.stop_reason, result.final_message) == ("end_turn", "done")
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "progress\n"
    progress = bounded.snapshot()
    assert (progress.read_calls, progress.write_calls) == (4, 1)
    assert (progress.denied_read_calls, progress.consecutive_read_calls) == (0, 2)
    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    backend.close()


@pytest.mark.parametrize(
    "script",
    (
        (_READ_CALL, _READ_CALL, _READ_CALL),
        (_READ_CALL, _READ_CALL, _REFUSED_WRITE_CALL, _READ_CALL),
    ),
    ids=("reads_alone", "refused_write_between_reads"),
)
def test_kimi_reads_alone_still_end_at_the_consecutive_read_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    script: tuple[tuple[str, dict[str, Any]], ...],
) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    replies: list[str] = []
    bounded = _progress_bounded_writer(tmp_path, max_consecutive_read_calls=2)
    backend = KimiBackend(
        capability="workspace_write",
        _worker_runner=_worker_that_calls_through_the_bridge(script, replies),
    )
    backend.attach_toolset(bounded)

    result = backend.invoke(_invocation(policy=_WRITER_POLICY))

    # The third read in a row is denied, and the denial ends the call
    # whatever the worker reports afterwards. A write that the host refused
    # is no progress: it is not credited and the streak goes on across it.
    assert len(replies) == len(script)
    assert "TerminalToolDispatchError" in replies[-1]
    assert not any("TerminalToolDispatchError" in reply for reply in replies[:-1])
    assert result.stop_reason == "tool_budget_exhausted"
    assert result.error is not None and result.error.retriable is False
    assert result.final_message == ""
    progress = bounded.snapshot()
    assert (progress.read_calls, progress.write_calls) == (2, 0)
    assert (progress.denied_read_calls, progress.consecutive_read_calls) == (1, 2)
    assert not (tmp_path / "output.txt").exists()
    assert not (tmp_path / "elsewhere.txt").exists()
    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    backend.close()


def test_kimi_rejects_unsupported_mutators_before_credentials_or_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import ToolContext
    from crucible.tools.profiles import build_implementer_profile

    monkeypatch.delenv("KIMI_BASE_URL", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("output.txt",),
    )
    backend = KimiBackend(
        capability="workspace_write",
        _worker_runner=lambda *_args, **_kwargs: pytest.fail(
            "worker must not start"
        ),
    )
    backend.attach_toolset(build_implementer_profile(context))

    with pytest.raises(
        KimiSDKQualificationError,
        match=(
            "build_project, create_rust_crate, materialize_artifact"
        ),
    ):
        backend.invoke(_invocation())

    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    assert backend._active_mcp_servers == set()  # noqa: SLF001
    backend.close()


def test_kimi_does_not_return_while_mutation_host_settlement_is_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.tools import ToolContext
    from crucible.tools import _isolated_mutation_host as mutation_mod
    from crucible.tools.write import build_write_tools

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    class FailingHost:
        instances: list[FailingHost] = []

        def __init__(self, spec, *, invocation_deadline) -> None:
            del invocation_deadline
            self.spec = spec
            self.allow_settlement = False
            self.stop_calls = 0
            type(self).instances.append(self)

        @property
        def tool_names(self):
            return self.spec.names

        @property
        def ever_started_write(self) -> bool:
            return False

        @property
        def started_write_names(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return

        def stop(self) -> None:
            self.stop_calls += 1
            if not self.allow_settlement:
                raise mutation_mod.MutationHostLifecycleError(
                    "injected live mutation child"
                )

    monkeypatch.setattr(mutation_mod, "IsolatedMutationHost", FailingHost)
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("output.txt",),
    )
    backend = KimiBackend(
        capability="workspace_write",
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(),
    )
    backend.attach_toolset(
        build_write_tools(context).filter_to({"write_source"})
    )

    with pytest.raises(
        KimiSDKQualificationError,
        match="mutation/MCP lifecycle did not settle",
    ):
        backend.invoke(_invocation())

    assert len(FailingHost.instances) == 1
    host = FailingHost.instances[0]
    assert host in backend._active_mutation_hosts  # noqa: SLF001
    assert backend._active_mcp_servers == set()  # noqa: SLF001
    with pytest.raises(
        KimiSDKQualificationError,
        match="could not confirm all confined resources settled",
    ):
        backend.close()
    assert host in backend._active_mutation_hosts  # noqa: SLF001
    host.allow_settlement = True
    backend.close()
    assert host.stop_calls == 4
    assert backend._active_mutation_hosts == set()  # noqa: SLF001


def test_kimi_retains_listener_until_a_failed_stop_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import _mcp_http_server as mcp_mod

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)

    class FailingServer:
        instances: list[FailingServer] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.allow_settlement = False
            self.stop_calls = 0
            type(self).instances.append(self)

        def start(self) -> str:
            return "http://127.0.0.1:49152/mcp"

        def stop(self) -> None:
            self.stop_calls += 1
            if not self.allow_settlement:
                raise mcp_mod.McpServerLifecycleError(
                    "injected live listener thread"
                )

    monkeypatch.setattr(mcp_mod, "CrucibleHttpMcpServer", FailingServer)
    backend = KimiBackend(
        capability="read_only",
        _worker_runner=lambda *_args, **_kwargs: _success_outcome(),
    )
    backend.attach_toolset(_ToolSet())

    with pytest.raises(
        KimiSDKQualificationError,
        match="mutation/MCP lifecycle did not settle",
    ):
        backend.invoke(_invocation())

    assert len(FailingServer.instances) == 1
    server = FailingServer.instances[0]
    assert server in backend._active_mcp_servers  # noqa: SLF001
    with pytest.raises(
        KimiSDKQualificationError,
        match="could not confirm all confined resources settled",
    ):
        backend.close()
    assert server in backend._active_mcp_servers  # noqa: SLF001
    server.allow_settlement = True
    backend.close()
    assert server.stop_calls == 4
    assert backend._active_mcp_servers == set()  # noqa: SLF001


def test_kimi_settled_late_mutation_overrides_completed_worker_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import ToolMutationIndeterminate
    from crucible.tools import _mcp_http_server as mcp_mod

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    terminal = ToolMutationIndeterminate("late delete settlement is unknown")

    class _Server:
        instances: list[_Server] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.terminal_dispatch_error = None
            self.stop_calls = 0
            type(self).instances.append(self)

        def start(self) -> str:
            return "http://127.0.0.1:49152/mcp"

        def stop(self) -> None:
            self.stop_calls += 1
            self.terminal_dispatch_error = terminal

    monkeypatch.setattr(mcp_mod, "CrucibleHttpMcpServer", _Server)
    worker_calls = 0

    def completed_worker(*_args, **_kwargs):
        nonlocal worker_calls
        worker_calls += 1
        return _success_outcome()

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=completed_worker,
    )
    backend.attach_toolset(_ToolSet())
    result = backend.invoke(_invocation(policy=BackendPolicy(
        retry_budget=3,
        timeout_seconds=5.0,
        on_cancellation="convert_to_backend_error",
        on_max_turns="raise",
    )))

    assert worker_calls == 1
    assert result.final_message == ""
    assert result.stop_reason == "tool_mutation_indeterminate"
    assert result.error is not None
    assert result.error.kind == "transport"
    assert result.error.retriable is False
    assert str(terminal) in result.error.message
    assert result.usage.input_tokens == 15
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert len(_Server.instances) == 1
    assert _Server.instances[0].stop_calls == 1
    assert backend._active_mcp_servers == set()  # noqa: SLF001
    backend.close()


def test_kimi_unsettled_cleanup_preserves_mutation_as_primary_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import ToolMutationIndeterminate
    from crucible.tools import _mcp_http_server as mcp_mod

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    terminal = ToolMutationIndeterminate("late delete settlement is unknown")

    class _Server:
        instances: list[_Server] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.terminal_dispatch_error = None
            self.allow_settlement = False
            self.stop_calls = 0
            type(self).instances.append(self)

        def start(self) -> str:
            return "http://127.0.0.1:49152/mcp"

        def stop(self) -> None:
            self.stop_calls += 1
            self.terminal_dispatch_error = terminal
            if not self.allow_settlement:
                raise mcp_mod.McpServerLifecycleError(
                    "injected unsettled dispatch pool"
                )

    monkeypatch.setattr(mcp_mod, "CrucibleHttpMcpServer", _Server)
    worker_calls = 0

    def completed_worker(*_args, **_kwargs):
        nonlocal worker_calls
        worker_calls += 1
        return _success_outcome()

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=completed_worker,
    )
    backend.attach_toolset(_ToolSet())
    with pytest.raises(ToolMutationIndeterminate) as caught:
        backend.invoke(_invocation(policy=BackendPolicy(
            retry_budget=3,
            timeout_seconds=5.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        )))

    assert caught.value is terminal
    assert isinstance(caught.value.__cause__, mcp_mod.McpServerLifecycleError)
    assert worker_calls == 1
    assert len(_Server.instances) == 1
    server = _Server.instances[0]
    assert server in backend._active_mcp_servers  # noqa: SLF001
    server.allow_settlement = True
    backend.close()
    assert server.stop_calls == 2
    assert backend._active_mcp_servers == set()  # noqa: SLF001


def test_kimi_settled_mutation_overrides_worker_lifecycle_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.tools import ToolMutationIndeterminate
    from crucible.tools import _mcp_http_server as mcp_mod

    monkeypatch.setenv("KIMI_BASE_URL", _BASE_URL)
    monkeypatch.setenv("MOONSHOT_API_KEY", _TOKEN)
    terminal = ToolMutationIndeterminate("late delete settlement is unknown")

    class _Server:
        def __init__(self, *_args, **_kwargs) -> None:
            self.terminal_dispatch_error = None

        def start(self) -> str:
            return "http://127.0.0.1:49152/mcp"

        def stop(self) -> None:
            self.terminal_dispatch_error = terminal

    monkeypatch.setattr(mcp_mod, "CrucibleHttpMcpServer", _Server)
    worker_calls = 0

    def failed_worker(*_args, **_kwargs):
        nonlocal worker_calls
        worker_calls += 1
        raise kimi_mod.KimiWorkerLifecycleError(
            "worker failed after dispatch timeout"
        )

    backend = KimiBackend(
        capability="read_only",
        _worker_runner=failed_worker,
    )
    backend.attach_toolset(_ToolSet())
    with pytest.raises(ToolMutationIndeterminate) as caught:
        backend.invoke(_invocation(policy=BackendPolicy(
            retry_budget=3,
            timeout_seconds=5.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        )))

    assert caught.value is terminal
    assert isinstance(caught.value.__cause__, kimi_mod.KimiWorkerLifecycleError)
    assert worker_calls == 1
    assert backend._active_mcp_servers == set()  # noqa: SLF001
    backend.close()


def test_kimi_close_is_idempotent_and_reaps_active_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)

    class Process:
        def __init__(self) -> None:
            self.alive = True
            self.terminate_calls = 0
            self.join_calls = 0
            self.kill_calls = 0

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.alive = False

        def join(self, timeout=None) -> None:
            del timeout
            self.join_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.alive = False

    process = Process()
    backend = KimiBackend()
    backend._active_processes.add(process)

    backend.close()
    backend.close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert backend._active_processes == set()
    with pytest.raises(RuntimeError, match="closed"):
        backend.invoke(_invocation())


def test_kimi_close_replays_sigint_after_final_registry_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final successful pass cannot consume first-pass cancellation."""

    _qualified_versions(monkeypatch)

    class _InterruptingHost:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise KeyboardInterrupt

    host = _InterruptingHost()
    backend = KimiBackend()
    backend._register_mutation_host(host)  # noqa: SLF001

    with pytest.raises(KeyboardInterrupt):
        backend.close()

    assert host.stop_calls == 2
    assert not backend._active_mutation_hosts  # noqa: SLF001
    backend.close()


def test_kimi_close_keeps_mutation_host_until_dispatch_server_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _qualified_versions(monkeypatch)

    class _Server:
        def __init__(self) -> None:
            self.allow_settlement = False
            self.calls = 0

        def stop(self) -> None:
            self.calls += 1
            if not self.allow_settlement:
                raise RuntimeError("dispatch still active")

    class _Host:
        def __init__(self) -> None:
            self.calls = 0

        def stop(self) -> None:
            self.calls += 1

    server = _Server()
    host = _Host()
    backend = KimiBackend()
    backend._register_mcp_server(server)  # noqa: SLF001
    backend._register_mutation_host(host)  # noqa: SLF001

    with pytest.raises(
        KimiSDKQualificationError,
        match="could not confirm all confined resources settled",
    ):
        backend.close()
    assert server.calls == 2
    assert host.calls == 0
    assert host in backend._active_mutation_hosts  # noqa: SLF001

    server.allow_settlement = True
    backend.close()
    assert server.calls == 3
    assert host.calls == 1
    assert not backend._active_mcp_servers  # noqa: SLF001
    assert not backend._active_mutation_hosts  # noqa: SLF001
