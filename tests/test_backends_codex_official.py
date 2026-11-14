# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import concurrent.futures
import signal
import time
from dataclasses import replace

import pytest
from assay import (
    BackendPolicy,
    ConversationTransportAction,
    ConversationTransportRequest,
    Invocation,
    StaleConversation,
)

from crucible.backends import CodexBackend, OfficialCodexBackend, resolve
from crucible.backends.config import load_routing
import crucible.backends.codex_official as official_module
import crucible.backends.codex_official_worker as worker_transport
from crucible.backends.codex_official_worker import (
    CodexOfficialWorkerInvocationError,
)


def _invocation() -> Invocation:
    return Invocation(
        system="Return only the requested JSON.",
        user="Return OK.",
        tools=None,
        max_turns=1,
        budget_usd=10.0,
        policy=BackendPolicy(
            retry_budget=1,
            timeout_seconds=5.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
        response_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )


def _worker_payload():
    return {
        "protocol": "crucible.codex-worker.v1",
        "type": "terminal",
        "ok": True,
        "sdk_version": "0.144.4",
        "thread_id": "thread-official",
        "turn_id": "turn-official",
        "status": "completed",
        "error": None,
        "final_response": '{"ok":true}',
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 20,
            "output_tokens": 10,
            "reasoning_output_tokens": 0,
            "total_tokens": 110,
        },
        "tool_items": [],
    }


def _conversation_request(
    *,
    action: ConversationTransportAction,
    opaque_id: str | None,
) -> ConversationTransportRequest:
    return ConversationTransportRequest(
        mode="workflow_bound",
        action=action,
        opaque_id=opaque_id,
        identity_fingerprint="sha256:" + ("a" * 64),
        operation_digest="sha256:" + ("b" * 64),
    )


def test_resolve_keeps_legacy_default_and_requires_explicit_official_switch(
    tmp_path
):
    legacy = resolve(
        "codex",
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
    )
    official = resolve(
        "codex",
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        transport="official_sdk",
        worker_python=tmp_path / "worker-python",
    )
    try:
        assert type(legacy) is CodexBackend
        assert isinstance(official, OfficialCodexBackend)
        assert official.transport == "official_sdk"
    finally:
        legacy.close()
        official.close()

    with pytest.raises(ValueError, match="codex transport"):
        resolve("codex", transport="automatic_fallback")


def test_routing_schema_rejects_unknown_codex_transport(
    tmp_path,
    monkeypatch,
):
    # The environment route intentionally has higher discovery priority than
    # ``config_path``.  Isolate this parser/schema test from CI's hermetic
    # route so it exercises the malformed fixture below.
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    path = tmp_path / "backends.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: crucible.backend-routing.v1",
                "default:",
                "  backend: codex",
                "  model: gpt-5.6-sol",
                "  effort: ultra",
                "  extra:",
                "    transport: silent_fallback",
                "roles: {}",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra.transport"):
        load_routing(config_path=path)


def test_official_sdk_transport_normalizes_public_terminal_usage_and_trace(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")
    worker = tmp_path / "worker-python"
    captured = {}

    async def fake_run(request, **kwargs):
        captured["request"] = request
        captured["environment"] = dict(kwargs["env"])
        holder = kwargs["thread_id_holder"]
        if holder is not None:
            holder.append("thread-official")
        return _worker_payload()

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=worker,
    )
    try:
        result = backend.invoke(_invocation())
    finally:
        backend.close()

    assert result.error is None
    assert result.final_message == '{"ok":true}'
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 10
    assert result.usage.cost_usd == pytest.approx(0.00152)
    assert result.usage.cache_creation_tokens == 0
    assert result.usage.cache_read_tokens == 20
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert not result.completeness.complete
    assert result.execution_trace is not None
    assert result.execution_trace.provider_request_id == "turn-official"
    assert [event.kind for event in result.execution_trace.events] == [
        "request",
        "usage",
        "terminal",
    ]
    assert result.execution_trace.events[0].name == "codex_official_sdk"
    assert result.execution_trace.events[1].name == "cost_upper_bound"
    assert captured["request"]["resume_thread_id"] is None
    assert captured["request"]["native_shell"] is False
    assert captured["request"]["trusted_project_instructions"] is False
    assert captured["request"]["output_schema"] == (
        _invocation().response_schema
    )
    assert "CODEX_HOME" in captured["environment"]


def test_official_sdk_budget_uses_unobserved_cache_write_upper_bound(
    monkeypatch,
    tmp_path,
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")

    async def fake_run(_request, **kwargs):
        holder = kwargs["thread_id_holder"]
        if holder is not None:
            holder.append("thread-official")
        payload = _worker_payload()
        payload["usage"] = {
            "input_tokens": 10_000,
            "cached_input_tokens": 2_000,
            "output_tokens": 10,
            "reasoning_output_tokens": 0,
            "total_tokens": 10_010,
        }
        return payload

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(replace(_invocation(), budget_usd=0.090))
    finally:
        backend.close()

    # $0.1025 assumes all 8,000 non-read input tokens were cache writes.
    # The no-write estimate ($0.0825) would have missed this budget overrun.
    assert result.error is not None
    assert result.error.kind == "budget_exhausted"
    assert result.usage.cost_usd == pytest.approx(0.1025)
    assert result.usage.cache_read_tokens == 2_000
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete


def test_official_sdk_all_cache_read_forces_exact_zero_write(
    monkeypatch,
    tmp_path,
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")

    async def fake_run(_request, **kwargs):
        holder = kwargs["thread_id_holder"]
        if holder is not None:
            holder.append("thread-official")
        payload = _worker_payload()
        payload["usage"] = {
            "input_tokens": 100,
            "cached_input_tokens": 100,
            "output_tokens": 10,
            "reasoning_output_tokens": 0,
            "total_tokens": 110,
        }
        return payload

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(_invocation())
    finally:
        backend.close()

    assert result.error is None
    assert result.usage.cache_creation_tokens == 0
    assert result.usage.cache_read_tokens == 100
    assert result.completeness.token_usage_complete
    assert result.completeness.cost_complete
    assert result.pricing_evidence.complete
    assert result.pricing_evidence.reconciles(result.usage)
    assert result.pricing_evidence.segments[0].cache_creation_tokens == 0


def test_official_sdk_rejects_unqualified_native_shell(tmp_path):
    with pytest.raises(ValueError, match="not qualified for native_shell"):
        OfficialCodexBackend(
            cwd=str(tmp_path),
            capability="workspace_write",
            native_shell=True,
            worker_python=tmp_path / "worker-python",
        )


def test_official_sdk_unaccounted_failure_suppresses_automatic_retry(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")
    requests = []

    async def fake_run(request, **kwargs):
        requests.append(request)
        holder = kwargs["thread_id_holder"]
        if holder is not None:
            holder.append("thread-retry")
        if len(requests) == 1:
            raise CodexOfficialWorkerInvocationError("BrokenPipeError")
        payload = _worker_payload()
        payload["thread_id"] = "thread-retry"
        return payload

    async def no_backoff(_seconds):
        return None

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    monkeypatch.setattr(
        "crucible.backends.codex.asyncio.sleep",
        no_backoff,
    )
    invocation = _invocation()
    invocation = Invocation(
        system=invocation.system,
        user=invocation.user,
        tools=invocation.tools,
        max_turns=invocation.max_turns,
        budget_usd=invocation.budget_usd,
        policy=BackendPolicy(
            retry_budget=2,
            timeout_seconds=5.0,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
        response_schema=invocation.response_schema,
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()

    assert result.error is not None
    assert result.error.retriable is False
    assert "did not return complete token/cost accounting" in (
        result.error.message
    )
    assert len(requests) == 1
    assert requests[0]["resume_thread_id"] is None
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert not any(
        event.kind == "retry"
        for event in result.execution_trace.events
    )


def test_official_sdk_external_resume_rejection_is_typed_stale(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")
    calls = 0

    async def fake_run(_request, **_kwargs):
        nonlocal calls
        calls += 1
        raise CodexOfficialWorkerInvocationError("InvalidParamsError")

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    invocation = replace(
        _invocation(),
        policy=replace(_invocation().policy, retry_budget=2),
        conversation=_conversation_request(
            action="reuse",
            opaque_id="thread-stale",
        ),
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()

    assert calls == 1
    assert isinstance(result.error, StaleConversation)
    assert result.error.retriable is False
    assert result.stop_reason == "stale_conversation"
    assert result.conversation is not None
    assert result.conversation.state == "stale"
    assert result.conversation.opaque_id is None


def test_official_sdk_post_acceptance_rpc_error_is_not_stale_fallback(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")

    async def fake_run(_request, **kwargs):
        holder = kwargs["thread_id_holder"]
        assert holder is not None
        holder.append("thread-stale")
        raise CodexOfficialWorkerInvocationError("InvalidParamsError")

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    invocation = replace(
        _invocation(),
        conversation=_conversation_request(
            action="reuse",
            opaque_id="thread-stale",
        ),
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()

    assert result.error is not None
    assert result.error.kind == "transport"
    assert result.stop_reason == "error"
    assert result.conversation is not None
    assert result.conversation.state == "active"
    assert result.conversation.opaque_id == "thread-stale"


def test_official_sdk_returns_identity_bound_conversation_outcome(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")

    async def fake_run(_request, **kwargs):
        holder = kwargs["thread_id_holder"]
        assert holder is not None
        holder.append("thread-official")
        return _worker_payload()

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    request = _conversation_request(action="fresh", opaque_id=None)
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(replace(_invocation(), conversation=request))
    finally:
        backend.close()

    assert result.error is None
    assert result.conversation is not None
    assert result.conversation.state == "active"
    assert result.conversation.opaque_id == "thread-official"
    assert (
        result.conversation.identity_fingerprint
        == request.identity_fingerprint
    )
    assert result.conversation.operation_digest == request.operation_digest


def test_official_sdk_deadline_cancels_worker_await_and_records_limit(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")
    cancelled = False

    async def fake_run(_request, **_kwargs):
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        finally:
            cancelled = True

    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        official_module,
        "run_official_worker",
        fake_run,
    )
    invocation = _invocation()
    invocation = Invocation(
        system=invocation.system,
        user=invocation.user,
        tools=invocation.tools,
        max_turns=invocation.max_turns,
        budget_usd=invocation.budget_usd,
        policy=BackendPolicy(
            retry_budget=1,
            timeout_seconds=0.02,
            on_cancellation="convert_to_backend_error",
            on_max_turns="raise",
        ),
        response_schema=invocation.response_schema,
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    try:
        result = backend.invoke(invocation)
    finally:
        backend.close()

    assert cancelled
    assert result.stop_reason == "timeout"
    assert result.error is not None
    assert result.error.retriable is False
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert not any(
        event.kind == "cancellation"
        for event in result.execution_trace.events
    )
    assert any(
        event.kind == "limit" and event.name == "deadline"
        for event in result.execution_trace.events
    )


def test_official_sdk_close_kills_worker_group_before_late_child_mutation(
    monkeypatch, tmp_path
):
    auth = tmp_path / "operator-auth.json"
    auth.write_text("{}", encoding="utf-8")
    started = tmp_path / "started"
    late = tmp_path / "late"
    worker = tmp_path / "worker"
    worker.write_text(
        "#!/bin/sh\n"
        "read ignored\n"
        f"/usr/bin/touch '{started}'\n"
        "(trap '' TERM; "
        f"/bin/sleep 0.4; /usr/bin/touch '{late}'; /bin/sleep 20) &\n"
        "/bin/sleep 20\n",
        encoding="utf-8",
    )
    worker.chmod(0o700)
    monkeypatch.setattr(
        official_module,
        "_resolve_codex_auth_path",
        lambda: auth,
    )
    monkeypatch.setattr(
        worker_transport,
        "qualify_worker_runtime",
        lambda _path: None,
    )
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=worker,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(backend.invoke, _invocation())
        deadline = time.monotonic() + 2.0
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists()
        backend.close()
        result = future.result(timeout=5.0)

    time.sleep(0.6)
    assert not late.exists()
    assert result.stop_reason == "cancelled"
    assert result.error is not None
    assert result.error.kind == "cancellation"
    assert any(
        event.kind == "cancellation"
        for event in result.execution_trace.events
    )


def test_official_sdk_close_retains_unproven_group_for_retry(
    monkeypatch, tmp_path
):
    backend = OfficialCodexBackend(
        cwd=str(tmp_path),
        capability="text",
        native_shell=False,
        worker_python=tmp_path / "worker-python",
    )
    process_id = 987_654_321
    backend._official_process_ids.add(process_id)
    signals = []
    monkeypatch.setattr(
        official_module.os,
        "killpg",
        lambda _process_group_id, sig: signals.append(sig),
    )
    monkeypatch.setattr(
        official_module,
        "_official_process_group_exists",
        lambda _process_group_id: True,
    )
    monkeypatch.setattr(
        official_module,
        "_official_process_exists",
        lambda _process_id: False,
    )
    monkeypatch.setattr(official_module, "_PROCESS_STOP_SECONDS", 0.001)

    try:
        with pytest.raises(
            worker_transport.CodexOfficialSDKQualificationError,
            match="did not settle",
        ):
            backend.close()

        assert backend._official_process_ids == {process_id}
        assert signal.SIGTERM in signals
        assert signal.SIGKILL in signals
    finally:
        monkeypatch.setattr(
            official_module,
            "_official_process_group_exists",
            lambda _process_group_id: False,
        )
        backend.close()

    assert backend._official_process_ids == set()
