# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Offline cancellation paths through the real SDK capture and retry owner."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from assay import Invocation, TransportError, Usage
from crucible.backends import codex as codex_mod
from crucible.backends._physical_telemetry import capture_physical_usage
from crucible.backends.codex import CodexBackend
from crucible.tests.test_backends_codex import (
    _conversation_request,
    _policy,
    _raw_response_completed_event,
    _token_usage_event,
)
from crucible.tools._dispatch_budget import (
    ToolDispatchBudgetExceeded,
    ToolMutationIndeterminate,
)


def _terminal(*, turn_id="turn-1", thread_id="thread-1"):
    return {
        "method": "turn/completed",
        "params": {
            "threadId": thread_id,
            "turn": {"id": turn_id, "status": "completed"},
        },
    }


def _prefix(*, raw=True, resumed=False):
    events = []
    if resumed:
        events.append(
            _token_usage_event(
                input_tokens=100, output_tokens=10, turn_id="old-turn"
            )
        )
    if raw:
        events.append(_raw_response_completed_event(input_tokens=20, output_tokens=5))
    events.append(
        _token_usage_event(
            input_tokens=20,
            output_tokens=5,
            total_input=120 if resumed else 20,
            total_output=15 if resumed else 5,
        )
    )
    return events


@pytest.fixture
def run_captured(monkeypatch, tmp_path):
    """Fake only the SDK transport; retain the actual capture/owner/monitors."""
    auth = tmp_path / "auth.json"
    auth.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(codex_mod, "_resolve_codex_auth_path", lambda: auth)
    backends = []

    def run(
        *,
        events,
        mode="budget",
        timeout=None,
        resumed=False,
        on_cancellation="convert_to_backend_error",
        exit_events=(),
        exit_exception=None,
        prior=False,
        suppress_cancellation=False,
    ):
        observed = SimpleNamespace(
            starts=0, exits=0, joined=False, dumps=[], result=None, exception=None
        )
        started = asyncio.Event()
        denial = (
            ToolMutationIndeterminate("unsettled source write")
            if mode == "mutation"
            else ToolDispatchBudgetExceeded("denied by source budget")
        )
        server = SimpleNamespace(
            terminal_dispatch_error_event=threading.Event(),
            terminal_dispatch_error=None,
        )

        class Handle:
            thread_id = "thread-1"

            def __init__(self, client):
                self.client = client

            async def chat_once(self, _user, turn_overrides=None):
                observed.starts += 1
                for event in events:
                    self.client._apply_event_to_session(None, event)
                started.set()
                if mode in {"budget", "mutation", "race"}:
                    server.terminal_dispatch_error = denial
                    server.terminal_dispatch_error_event.set()
                if mode in {"race", "exit"}:
                    return SimpleNamespace(
                        final_text="finished",
                        raw_events=events,
                        turn_id="turn-1",
                        thread_id="thread-1",
                    )
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    if not suppress_cancellation:
                        raise
                    self.client._apply_event_to_session(None, _terminal())
                    return SimpleNamespace(
                        final_text="returned after cancel", raw_events=events,
                        turn_id="turn-1", thread_id="thread-1",
                    )

        class Client:
            @classmethod
            def connect_stdio(cls, **_kwargs):
                return cls()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                observed.exits += 1
                for event in exit_events:
                    self._apply_event_to_session(None, event)
                # Teardown must finish before the result can settle.
                await asyncio.sleep(0)
                observed.joined = True
                if exit_exception is not None:
                    raise exit_exception

            async def initialize(self, params=None):
                pass

            def set_approval_handler(self, _handler):
                pass

            def _apply_event_to_session(self, _session, _event):
                pass

            async def start_thread(self, config=None):
                return Handle(self)

            async def resume_thread(self, thread_id, overrides=None):
                assert thread_id == "thread-1"
                return Handle(self)

        module = ModuleType("codex_app_server_sdk")
        setattr(module, "CodexClient", Client)
        setattr(module, "ThreadConfig", lambda **kwargs: SimpleNamespace(**kwargs))
        setattr(module, "TurnOverrides", lambda **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setitem(sys.modules, "codex_app_server_sdk", module)
        backend = CodexBackend()
        backends.append(backend)
        monkeypatch.setattr(
            backend._sdk_sink, "dump", lambda **kwargs: observed.dumps.append(kwargs)
        )
        policy = _policy(
            retry_budget=3, timeout_seconds=timeout, on_cancellation=on_cancellation
        )
        invocation = Invocation(
            system="test", user="test", tools=None, max_turns=2, budget_usd=None,
            policy=policy,
            conversation=(
                _conversation_request(action="reuse", opaque_id="thread-1")
                if resumed else None
            ),
        )
        if prior:
            real_once = backend._invoke_once
            first = True

            async def with_prior(*args, **kwargs):
                nonlocal first
                if not first:
                    return await real_once(*args, **kwargs)
                first = False
                kwargs["thread_id_holder"].append("thread-1")
                # A fully accounted transport failure authorizes exactly one
                # existing retry. Its usage must be merged once, not replayed.
                return backend._result_with_evidence(
                    final_message="", transcript=(),
                    usage=Usage(input_tokens=100, output_tokens=10, cost_usd=0.25),
                    stop_reason="error",
                    error=TransportError(message="transport lost", retriable=True),
                    token_usage_complete=True, cost_complete=True,
                    events=[codex_mod.BackendExecutionEvent(
                        sequence=0, kind="usage", name="complete"
                    )],
                )

            monkeypatch.setattr(backend, "_invoke_once", with_prior)
            # Avoid sleeping through the existing retry backoff; retain the
            # real async scheduling and the two cancellation monitors.
            sleep = asyncio.sleep

            async def short_backoff(delay, *args, **kwargs):
                return await sleep(0 if delay >= 1 else delay, *args, **kwargs)

            monkeypatch.setattr(codex_mod.asyncio, "sleep", short_backoff)

        async def drive():
            deadline = None if timeout is None else time.monotonic() + timeout
            task = asyncio.create_task(backend._invoke_with_retries_core(
                invocation, policy=policy, timeout_seconds=timeout,
                deadline=deadline, mcp_url_override=None, mutation_host=None,
                terminal_tool_server=(server if mode in {"budget", "mutation", "race"}
                                      else None),
            ))
            if mode == "cancel":
                await asyncio.wait_for(started.wait(), timeout=2.0)
                task.cancel("operator cancellation")
            try:
                observed.result = await task
            except BaseException as exc:
                observed.exception = exc
            assert task.done()

        with capture_physical_usage("codex") as collector:
            asyncio.run(drive())
        observed.telemetry = collector.freeze()
        return observed

    yield run
    for backend in backends:
        backend.close()


def _assert_partial(result, *, input_tokens=20, output_tokens=5):
    assert result.usage.input_tokens == input_tokens
    assert result.usage.output_tokens == output_tokens
    assert result.usage.cost_usd > 0
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert not result.pricing_evidence.complete
    assert [e.name for e in result.execution_trace.events if e.kind == "usage"] == [
        "partial"
    ]


def _cancellation_names(result) -> list[str | None]:
    return [
        event.name
        for event in result.execution_trace.events
        if event.kind == "cancellation"
    ]


def test_deadline_owner_retrieves_same_turn_child_exception(monkeypatch):
    """Owner interruption still observes a child that failed in the same turn."""

    observed: list[BaseException | None] = []

    async def drive() -> None:
        class TrackingFuture(asyncio.Future):
            def exception(self):
                result = super().exception()
                observed.append(result)
                return result

        child = TrackingFuture()
        child_error = RuntimeError("same-turn child failure")

        async def colliding_wait(tasks, **_kwargs):
            assert tasks == {child}
            child.set_exception(child_error)
            raise asyncio.CancelledError("owner interruption")

        monkeypatch.setattr(codex_mod.asyncio, "wait", colliding_wait)
        with pytest.raises(asyncio.CancelledError, match="owner interruption"):
            await codex_mod._await_with_deadline(child, None)
        assert observed == [child_error]

    asyncio.run(drive())


@pytest.mark.parametrize("mode", ["budget", "mutation"])
@pytest.mark.parametrize("timeout", [None, 2.0])
@pytest.mark.parametrize("raw", [False, True])
def test_dispatch_cancellation_preserves_captured_prefix(
    run_captured, mode, timeout, raw
):
    observed = run_captured(events=_prefix(raw=raw), mode=mode, timeout=timeout)
    assert observed.exception is None
    _assert_partial(observed.result)
    assert observed.result.stop_reason == (
        "tool_budget_exhausted" if mode == "budget" else "tool_mutation_indeterminate"
    )
    assert observed.result.completeness.terminal_complete
    assert _cancellation_names(observed.result) == []
    assert observed.starts == observed.exits == 1
    assert observed.joined
    assert len(observed.telemetry.attempts) == 1
    assert not observed.telemetry.attempts[0].complete
    assert len(observed.dumps) == 1


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_deadline_and_converted_external_cancellation_preserve_prefix(run_captured, mode):
    observed = run_captured(
        events=_prefix(), mode=mode, timeout=0.1 if mode == "timeout" else None
    )
    assert observed.exception is None
    _assert_partial(observed.result)
    assert observed.result.stop_reason == ("timeout" if mode == "timeout" else "cancelled")
    assert _cancellation_names(observed.result) == (
        [] if mode == "timeout" else ["external"]
    )
    assert not observed.result.completeness.terminal_complete
    assert observed.starts == observed.exits == 1
    assert observed.joined
    assert len(observed.telemetry.attempts) == 1


def test_raise_policy_keeps_diagnostic_and_propagates_cancellation(run_captured):
    observed = run_captured(events=_prefix(), mode="cancel", on_cancellation="raise")
    assert isinstance(observed.exception, asyncio.CancelledError)
    assert str(observed.exception) == "operator cancellation"
    assert observed.result is None
    assert observed.joined
    assert len(observed.dumps) == 1
    assert observed.dumps[0]["re_raised"]
    _assert_partial(observed.dumps[0]["result"])
    assert _cancellation_names(observed.dumps[0]["result"]) == ["external"]
    assert len(observed.telemetry.attempts) == 1


@pytest.mark.parametrize("mode", ["budget", "timeout", "cancel"])
def test_resumed_raw_prefix_excludes_replayed_history(run_captured, mode):
    observed = run_captured(
        events=_prefix(resumed=True), resumed=True, mode=mode,
        timeout=0.1 if mode == "timeout" else None,
    )
    assert observed.exception is None
    _assert_partial(observed.result)
    assert observed.result.execution_trace.provider_request_id == "turn-1"
    assert _cancellation_names(observed.result) == (
        ["external"] if mode == "cancel" else []
    )
    assert len(observed.telemetry.attempts) == 1


@pytest.mark.parametrize("case", ["empty", "resume_only", "conflicting_thread", "conflicting_turn"])
def test_unqualified_capture_never_invents_free_or_complete_usage(run_captured, case):
    events = _prefix()
    resumed = False
    if case == "empty":
        events = []
    elif case == "resume_only":
        events = [_token_usage_event(input_tokens=100, output_tokens=10, turn_id="old-turn")]
        resumed = True
    elif case == "conflicting_thread":
        events.append(_token_usage_event(input_tokens=20, output_tokens=5, thread_id="other"))
    else:
        events.append(_terminal(turn_id="other"))
    observed = run_captured(events=events, resumed=resumed)
    assert observed.exception is None
    result = observed.result
    assert result.usage == Usage()
    assert not result.completeness.cost_complete
    assert not result.completeness.token_usage_complete
    assert not result.pricing_evidence.complete
    assert result.error is not None
    assert observed.starts == observed.exits == 1


@pytest.mark.parametrize("grouped", [False, True])
def test_cancellation_during_client_exit_keeps_final_ledger(run_captured, grouped):
    cancelled = asyncio.CancelledError("client exit")
    error = BaseExceptionGroup("cleanup cancelled", [cancelled]) if grouped else cancelled
    observed = run_captured(
        events=_prefix(), mode="exit", exit_events=[_terminal()],
        exit_exception=error, on_cancellation="raise",
    )
    assert observed.exception is error
    assert observed.joined
    assert len(observed.telemetry.attempts) == 1
    _assert_partial(observed.dumps[0]["result"])
    assert not observed.dumps[0]["result"].completeness.terminal_complete


@pytest.mark.parametrize("mode", ["budget", "timeout", "cancel"])
def test_teardown_error_replacing_cancel_still_retains_prefix(run_captured, mode):
    observed = run_captured(
        events=_prefix(), mode=mode, timeout=0.1 if mode == "timeout" else None,
        exit_exception=OSError("teardown failed"),
    )
    assert observed.exception is None
    _assert_partial(observed.result)
    assert observed.joined
    assert observed.starts == observed.exits == 1
    assert _cancellation_names(observed.result) == (
        ["external"] if mode == "cancel" else []
    )
    assert len(observed.telemetry.attempts) == 1


@pytest.mark.parametrize("timeout", [None, 2.0])
def test_external_cancel_raise_survives_replacing_teardown_error(run_captured, timeout):
    observed = run_captured(
        events=_prefix(), mode="cancel", timeout=timeout,
        exit_exception=OSError("teardown failed"), on_cancellation="raise",
    )
    assert isinstance(observed.exception, asyncio.CancelledError)
    assert str(observed.exception) == "operator cancellation"
    assert observed.result is None
    assert observed.joined
    assert observed.dumps[0]["re_raised"]
    _assert_partial(observed.dumps[0]["result"])
    assert _cancellation_names(observed.dumps[0]["result"]) == ["external"]


@pytest.mark.parametrize("mode", ["budget", "timeout", "cancel"])
@pytest.mark.parametrize("on_cancellation", ["raise", "convert_to_backend_error"])
def test_pending_provider_return_after_cancel_retains_usage_and_owner_outcome(
    run_captured, mode, on_cancellation
):
    observed = run_captured(
        events=_prefix(), mode=mode, timeout=0.1 if mode == "timeout" else None,
        suppress_cancellation=True, on_cancellation=on_cancellation,
    )
    result = observed.result
    if mode == "cancel" and on_cancellation == "raise":
        assert isinstance(observed.exception, asyncio.CancelledError)
        assert str(observed.exception) == "operator cancellation"
        result = observed.dumps[0]["result"]
    else:
        assert observed.exception is None
        assert result.stop_reason == {
            "budget": "tool_budget_exhausted", "timeout": "timeout", "cancel": "cancelled"
        }[mode]
    _assert_partial(result)
    assert _cancellation_names(result) == (
        ["external"] if mode == "cancel" else []
    )
    assert observed.starts == observed.exits == 1
    assert observed.joined
    assert len(observed.telemetry.attempts) == 1
    assert not observed.telemetry.attempts[0].complete


def test_final_exit_identity_conflict_rejects_provisional_attribution(run_captured):
    observed = run_captured(
        events=_prefix(), mode="exit", exit_events=[_terminal(thread_id="other")],
        exit_exception=asyncio.CancelledError("exit"),
    )
    assert observed.result.usage == Usage()
    assert not observed.result.completeness.cost_complete
    assert len(observed.telemetry.attempts) == 1


def test_natural_completion_racing_denial_is_not_reharvested(run_captured):
    observed = run_captured(events=[*_prefix(), _terminal()], mode="race")
    result = observed.result
    assert result.stop_reason == "tool_budget_exhausted"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 5
    assert result.completeness.cost_complete
    assert result.pricing_evidence.complete
    assert _cancellation_names(result) == []
    assert len(observed.telemetry.attempts) == 1
    assert observed.telemetry.attempts[0].complete


@pytest.mark.parametrize("mode", ["budget", "timeout", "cancel"])
@pytest.mark.parametrize("on_cancellation", ["raise", "convert_to_backend_error"])
def test_prior_accounted_attempt_is_merged_once(run_captured, mode, on_cancellation):
    observed = run_captured(
        events=_prefix(resumed=True), prior=True, mode=mode,
        timeout=0.1 if mode == "timeout" else None,
        on_cancellation=on_cancellation,
    )
    result = observed.result
    if mode == "cancel" and on_cancellation == "raise":
        assert isinstance(observed.exception, asyncio.CancelledError)
        result = observed.dumps[-1]["result"]
    else:
        assert observed.exception is None
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 15
    assert 0.25 < result.usage.cost_usd < 0.26
    assert not result.completeness.cost_complete
    assert not result.pricing_evidence.complete
    assert _cancellation_names(result) == (
        ["external"] if mode == "cancel" else []
    )
    assert observed.starts == observed.exits == 1
    assert len(observed.telemetry.attempts) == 1


def test_private_home_retained_with_modes_until_mutation_host_settles(tmp_path):
    class Host:
        calls = 0

        def stop(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("mutation still owned")

    backend = CodexBackend()
    home = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(home.name)
    evidence = root / "rollout.jsonl"
    evidence.write_text("private bounded test evidence", encoding="utf-8")
    evidence.chmod(0o600)
    original = (root.stat(), evidence.stat())
    host = Host()
    backend._register_mutation_host(host)
    backend._isolated_home = home
    with pytest.raises(RuntimeError, match="mutation still owned"):
        backend.close()
    assert backend._isolated_home is home
    assert evidence.read_text(encoding="utf-8") == "private bounded test evidence"
    for before, after in zip(original, (root.stat(), evidence.stat()), strict=True):
        assert (before.st_mode, before.st_uid, before.st_gid) == (
            after.st_mode, after.st_uid, after.st_gid
        )
    backend.close()
    assert not root.exists()
    assert backend._isolated_home is None
    assert host.calls == 2
