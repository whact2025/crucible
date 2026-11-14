# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for the in-process Claude MCP bridge."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

import pytest

from crucible.tools._dispatch_budget import (
    ToolDispatchBudgetExceeded,
    ToolMutationIndeterminate,
)
from crucible.tools._mcp_bridge import (
    ContextBudgetTracker,
    McpDispatchLifecycleError,
    _DispatchWorker,
    build_mcp_server,
)


def test_context_budget_counts_cumulative_utf8_bytes_and_truncates_safely():
    tracker = ContextBudgetTracker(5)

    assert tracker.apply("a") == "a"
    truncated = tracker.apply("ééé")
    exhausted = tracker.apply("later")

    # One ASCII byte was already consumed, so only two complete two-byte
    # codepoints fit. Never split a UTF-8 sequence or emit U+FFFD.
    assert "\n\néé\n\n[TRUNCATED]" in truncated
    assert "4 UTF-8 bytes" in truncated
    assert "\ufffd" not in truncated
    assert "[BUDGET EXHAUSTED]" in exhausted
    assert "5 UTF-8 bytes" in exhausted


def test_context_budget_drops_partial_multibyte_codepoint():
    tracker = ContextBudgetTracker(5)

    assert tracker.apply("aa") == "aa"
    truncated = tracker.apply("éé")

    # Three bytes remain, but only one complete two-byte character may be
    # returned from the four-byte payload.
    assert "\n\né\n\n[TRUNCATED]" in truncated
    assert "\ufffd" not in truncated


def test_context_budget_reset_restores_full_byte_allowance():
    tracker = ContextBudgetTracker(4)

    tracker.apply("éé")
    assert "[BUDGET EXHAUSTED]" in tracker.apply("x")
    tracker.reset()

    assert tracker.apply("éé").endswith("éé")


def test_context_budget_is_atomic_across_concurrent_dispatches():
    tracker = ContextBudgetTracker(10)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(tracker.apply, ("éééé", "éééé")))

    # The two payloads request 16 bytes. Exactly five complete two-byte
    # characters fit under the shared ten-byte ceiling.
    assert sum(result.count("é") for result in results) == 5
    assert sum("[TRUNCATED]" in result for result in results) == 1
    assert all("\ufffd" not in result for result in results)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "100"])
def test_context_budget_requires_positive_integer_bytes(budget):
    with pytest.raises(ValueError, match="context_budget"):
        ContextBudgetTracker(budget)


@pytest.mark.parametrize(
    "terminal",
    (
        ToolDispatchBudgetExceeded("discovery tranche exhausted"),
        ToolMutationIndeterminate("delete settlement is unknown"),
    ),
    ids=("budget", "mutation"),
)
def test_dispatch_budget_denial_publishes_terminal_signal(
    monkeypatch,
    terminal,
):
    import claude_agent_sdk

    captured = {}

    def fake_create_sdk_mcp_server(*, name, tools, version="1.0.0"):
        captured.update(name=name, tools=tools, version=version)
        return {"name": name, "tools": tools}

    monkeypatch.setattr(
        claude_agent_sdk,
        "create_sdk_mcp_server",
        fake_create_sdk_mcp_server,
    )

    class DenyingToolSet:
        calls = 0

        @staticmethod
        def to_anthropic_tools():
            return [{
                "name": "read_value",
                "description": "read",
                "input_schema": {"type": "object", "properties": {}},
            }]

        def dispatch(self, _name, _arguments):
            self.calls += 1
            raise terminal

    toolset = DenyingToolSet()
    server, _names, worker, _budget = build_mcp_server(toolset)

    async def drive():
        worker.reset_terminal_denial()
        result = await server["tools"][0].handler({})
        repeated = await server["tools"][0].handler({})
        denial = await asyncio.wait_for(
            worker.wait_for_terminal_denial(),
            timeout=1.0,
        )
        return result, repeated, denial

    try:
        result, repeated, denial = asyncio.run(drive())
    finally:
        worker.shutdown()

    assert toolset.calls == 1
    assert result["is_error"] is True
    assert repeated["is_error"] is True
    expected_marker = (
        "terminal tool-dispatch budget exhausted"
        if isinstance(terminal, ToolDispatchBudgetExceeded)
        else type(terminal).__name__
    )
    assert expected_marker in result["content"][0]["text"]
    assert isinstance(denial, type(terminal))
    assert worker.terminal_denial is denial
    assert captured["name"] == "crucible"


def test_running_mutator_timeout_latches_indeterminate_before_response(
    monkeypatch,
):
    import claude_agent_sdk

    captured = {}

    def fake_create_sdk_mcp_server(*, name, tools, version="1.0.0"):
        captured.update(name=name, tools=tools, version=version)
        return {"name": name, "tools": tools}

    monkeypatch.setattr(
        claude_agent_sdk,
        "create_sdk_mcp_server",
        fake_create_sdk_mcp_server,
    )
    started = threading.Event()
    release = threading.Event()

    class _SlowMutator:
        write_names = {"write_source"}
        read_only_names: set[str] = set()
        calls = 0

        @staticmethod
        def to_anthropic_tools():
            return [{
                "name": "write_source",
                "description": "write",
                "input_schema": {"type": "object", "properties": {}},
            }]

        def dispatch(self, _name, _arguments):
            self.calls += 1
            started.set()
            assert release.wait(timeout=2.0)
            return "mutation eventually settled"

    toolset = _SlowMutator()
    server, _names, worker, _budget = build_mcp_server(
        toolset,
        tool_timeout=0.02,
        max_workers=2,
    )

    async def drive():
        worker.reset_terminal_denial()
        first = await server["tools"][0].handler({})
        second = await server["tools"][0].handler({})
        denial = await asyncio.wait_for(
            worker.wait_for_terminal_denial(),
            timeout=1.0,
        )
        return first, second, denial

    try:
        first, second, denial = asyncio.run(drive())
        assert started.is_set()
        assert first["is_error"] is True
        assert second["is_error"] is True
        assert "ToolMutationIndeterminate" in first["content"][0]["text"]
        assert "ToolMutationIndeterminate" in second["content"][0]["text"]
        assert isinstance(denial, ToolMutationIndeterminate)
        assert toolset.calls == 1
    finally:
        release.set()
        worker.shutdown(timeout=1.0)


def test_dispatch_shutdown_is_bounded_and_retries_exact_stuck_worker() -> None:
    started = threading.Event()
    release = threading.Event()

    class _BlockingToolSet:
        @staticmethod
        def dispatch(_name, _arguments) -> str:
            started.set()
            release.wait()
            return "settled"

    worker = _DispatchWorker(_BlockingToolSet(), max_workers=1)
    future = worker.submit("write_source", {})
    assert started.wait(timeout=1)
    try:
        with pytest.raises(McpDispatchLifecycleError, match="remain live"):
            worker.shutdown(timeout=0.01)
        assert not future.done()
        assert len(worker._threads) == 1  # noqa: SLF001
        assert worker._threads[0].is_alive()  # noqa: SLF001
        assert worker._threads[0].daemon  # noqa: SLF001
    finally:
        release.set()
    assert future.result(timeout=1) == "settled"
    worker.shutdown(timeout=1)
    assert not worker._threads[0].is_alive()  # noqa: SLF001


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf")])
def test_dispatch_shutdown_requires_a_finite_bound(timeout: float) -> None:
    worker = _DispatchWorker(object(), max_workers=1)
    try:
        with pytest.raises(ValueError, match="finite"):
            worker.shutdown(timeout=timeout)
    finally:
        worker.shutdown(timeout=1)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="interpreter-exit proof requires fork",
)
def test_stuck_dispatch_has_no_cpython_executor_exit_hook_owner() -> None:
    """A retained daemon worker must not silently become an atexit join."""

    def child() -> None:
        started = threading.Event()
        never_release = threading.Event()

        class _BlockingToolSet:
            @staticmethod
            def dispatch(_name, _arguments) -> str:
                started.set()
                never_release.wait()
                return "unreachable"

        worker = _DispatchWorker(_BlockingToolSet(), max_workers=1)
        worker.submit("write_source", {})
        if not started.wait(timeout=1):
            raise RuntimeError("dispatch never started")
        try:
            worker.shutdown(timeout=0.01)
        except McpDispatchLifecycleError:
            return
        raise RuntimeError("stuck dispatch falsely settled")

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
        pytest.fail("stuck MCP dispatch acquired an interpreter-exit join owner")
    assert process.exitcode == 0
