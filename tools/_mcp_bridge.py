# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bridge any ``ToolSet`` into an in-process Claude Agent SDK MCP server.

The SDK's ``create_sdk_mcp_server`` runs MCP tool dispatch directly inside
the Python process — no subprocess, no stdio JSON-RPC. We adapt each
tool descriptor exposed by ``toolset.to_anthropic_tools()`` into an
``SdkMcpTool``, route every call through ``toolset.dispatch(name, args)``,
and hand the resulting server config to ``ClaudeAgentOptions.mcp_servers``.
The agent SDK exposes the tools to the model as ``mcp__<server>__<tool>``
names.

Works with anything satisfying the ``assay.ToolSet`` protocol —
``BuiltToolSet`` (single domain), ``CompositeToolSet`` (multiple
domains via ``composite(...)``), or custom adapters. The bridge talks
to the protocol surface, not the concrete class internals.

Tool dispatch is offloaded to a dedicated thread pool so a slow tool
can't stall the SDK's event loop. Without that isolation, a blocking
``subprocess.run`` inside a tool can deadlock
the agent's control-message reader.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import logging
import math
import queue
import threading
import time
from typing import Any

from ._dispatch_budget import (
    TerminalToolDispatchError,
    ToolMutationIndeterminate,
)

logger = logging.getLogger(__name__)

MCP_SERVER_NAME = "crucible"

TOOL_TIMEOUT_S: float = 300.0
BUILD_TOOL_TIMEOUT_S: float = 1800.0
DISPATCH_SHUTDOWN_TIMEOUT_S: float = 5.0

# Tools whose runtime is dominated by external subprocess work (cargo
# build and repository scaffolding) and routinely exceed the
# default 5-minute timeout. Each gets the 30-minute long-running budget
# in ``_handler``. Adding a new build-class tool? Append its name here.
_LONG_RUNNING_TOOL_NAMES = frozenset({
    "build_project",       # tools/build — project build command
    "create_rust_crate",   # tools/write — `cargo new` + initial deps
})

# Cap individual tool-error result strings before they reach the agent.
# Without a cap, an exception with a multi-MB payload (e.g. a build log
# attached to a CalledProcessError) would saturate the SDK conversation
# buffer in one shot. Successful results go through ``ContextBudgetTracker``
# instead; errors bypass that path so the agent always sees the failure
# untracked against the budget, but we still need a per-result ceiling.
_TOOL_ERROR_MAX_CHARS = 16 * 1024


class McpDispatchLifecycleError(RuntimeError):
    """Exact dispatch workers remain live after bounded shutdown."""


class ContextBudgetTracker:
    """Cap cumulative tool-result bytes returned to the agent.

    Tool payloads are accounted in UTF-8 bytes, matching the HTTP MCP and
    direct adapter bridges. Budget-control annotations (warning, truncation,
    and exhaustion text) are intentionally excluded so reporting the limit
    cannot recursively consume it.

    ``reset()`` zeroes the consumption counter; callers that reuse the
    tracker across invocations (e.g. attach once, invoke many) should
    call ``reset()`` between invocations to keep the cap per-invocation
    rather than per-attachment.
    """

    def __init__(self, budget_bytes: int = 400_000) -> None:
        if (
            isinstance(budget_bytes, bool)
            or not isinstance(budget_bytes, int)
            or budget_bytes <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        self._budget = budget_bytes
        self._consumed = 0
        self._warning_threshold = 0.85
        self._warned = False
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Zero the consumption counter for a new invocation."""
        with self._lock:
            self._consumed = 0
            self._warned = False

    @property
    def budget_bytes(self) -> int:
        """Configured payload allowance in UTF-8 bytes."""

        return self._budget

    @property
    def consumed_bytes(self) -> int:
        """Thread-safe snapshot of payload bytes consumed so far."""

        with self._lock:
            return self._consumed

    @property
    def exhausted(self) -> bool:
        """Whether the complete payload allowance has been consumed."""

        with self._lock:
            return self._consumed >= self._budget

    def apply(self, result_text: str) -> str:
        payload = result_text.encode("utf-8")
        with self._lock:
            if self._consumed >= self._budget:
                return (
                    f"[BUDGET EXHAUSTED] Context budget of "
                    f"{self._budget:,} UTF-8 bytes is fully consumed. "
                    "No more tool results will be returned. Produce your "
                    "final output now."
                )
            available = self._budget - self._consumed
            warning = ""
            if (
                not self._warned
                and (
                    (self._consumed + len(payload)) / self._budget
                    >= self._warning_threshold
                )
            ):
                self._warned = True
                warning = (
                    f"\n\n[BUDGET WARNING] {available:,} UTF-8 bytes "
                    f"remaining of {self._budget:,}. Prioritize essential "
                    "tool calls.\n\n"
                )
            if len(payload) > available:
                self._consumed = self._budget
                prefix = payload[:available].decode(
                    "utf-8",
                    errors="ignore",
                )
                return (
                    warning
                    + prefix
                    + "\n\n[TRUNCATED] Result truncated to fit the remaining "
                    f"context budget ({available:,} UTF-8 bytes)."
                )
            self._consumed += len(payload)
            return warning + result_text


class _DispatchWorker:
    """Dedicated thread pool for synchronous tool handlers.

    ``ToolSet.dispatch`` is sync — running it on the SDK's event loop
    thread blocks the control-message reader. Using a *separate* pool
    (not the default ``asyncio.get_event_loop().run_in_executor``
    pool) keeps tool dispatch isolated from anyio internals.

    ``max_workers`` defaults to 2 (one build slot + one read slot
    historically). Workflows expecting many concurrent in-flight tool
    calls — e.g. an agent that fans out 5 ``find_callers`` queries —
    should raise this via ``ClaudeBackend.attach_toolset(...,
    mcp_max_workers=8)``.
    """

    def __init__(
        self,
        toolset: Any,
        *,
        max_workers: int = 2,
    ) -> None:
        self._toolset = toolset
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer")
        self._jobs: queue.Queue[
            tuple[concurrent.futures.Future[str], Any] | object
        ] = queue.Queue()
        self._sentinel = object()
        self._lifecycle_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_settled = False
        self._threads: tuple[threading.Thread, ...] = tuple(
            threading.Thread(
                target=self._run_worker,
                name=f"crucible-mcp_{index}",
                daemon=True,
            )
            for index in range(max_workers)
        )
        started: list[threading.Thread] = []
        try:
            for thread in self._threads:
                thread.start()
                started.append(thread)
        except BaseException:
            with self._lifecycle_lock:
                self._shutdown_started = True
            for _thread in started:
                self._jobs.put(self._sentinel)
            for thread in started:
                thread.join(timeout=DISPATCH_SHUTDOWN_TIMEOUT_S)
            raise
        self._terminal_denial_lock = threading.Lock()
        self._terminal_denial: TerminalToolDispatchError | None = None
        self._terminal_denial_event: asyncio.Event | None = None
        self._terminal_denial_loop: asyncio.AbstractEventLoop | None = None

    def _run_worker(self) -> None:
        """Execute jobs until this exact daemon worker receives its sentinel.

        A custom daemon pool is intentional. ``ThreadPoolExecutor`` registers
        its workers in CPython's pre-atexit join table, so even
        ``shutdown(wait=False)`` can hang process exit forever after a stuck
        tool call. These workers remain explicit retry-owned objects without
        acquiring that hidden interpreter-exit owner.
        """

        while True:
            job = self._jobs.get()
            try:
                if job is self._sentinel:
                    return
                future, dispatch = job
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = dispatch()
                except BaseException as exc:  # noqa: BLE001 - Future owns it
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            finally:
                self._jobs.task_done()

    def submit(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> concurrent.futures.Future[str]:
        denial = self.terminal_denial
        if denial is not None:
            denied_future: concurrent.futures.Future[str] = (
                concurrent.futures.Future()
            )
            denied_future.set_exception(denial)
            return denied_future

        def _dispatch() -> str:
            prior_denial = self.terminal_denial
            if prior_denial is not None:
                raise prior_denial
            try:
                return self._toolset.dispatch(tool_name, arguments)
            except TerminalToolDispatchError as exc:
                # Record at the worker boundary, before the MCP SDK converts
                # handler failures into model-visible protocol errors.  The
                # Claude adapter races its stream against this signal and
                # terminates the physical query instead of letting the model
                # request another denied tool call.
                self.record_terminal_denial(exc)
                raise

        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        with self._lifecycle_lock:
            if self._shutdown_started:
                raise RuntimeError("MCP dispatch worker is shutting down")
            self._jobs.put((future, _dispatch))
        return future

    def reset_terminal_denial(self) -> None:
        """Arm a fresh terminal-denial signal for one logical invocation."""

        loop = asyncio.get_running_loop()
        with self._terminal_denial_lock:
            self._terminal_denial = None
            self._terminal_denial_loop = loop
            self._terminal_denial_event = asyncio.Event()

    def record_terminal_denial(
        self,
        exc: TerminalToolDispatchError,
    ) -> None:
        """Publish the first dispatch-budget denial to the adapter."""

        with self._terminal_denial_lock:
            if self._terminal_denial is not None:
                return
            self._terminal_denial = exc
            loop = self._terminal_denial_loop
            event = self._terminal_denial_event
        if loop is not None and event is not None:
            loop.call_soon_threadsafe(event.set)

    @property
    def terminal_denial(self) -> TerminalToolDispatchError | None:
        """Return the first denial observed in the current invocation."""

        with self._terminal_denial_lock:
            return self._terminal_denial

    async def wait_for_terminal_denial(
        self,
    ) -> TerminalToolDispatchError:
        """Wait until a dispatch-budget denial must terminate the tool loop."""

        with self._terminal_denial_lock:
            denial = self._terminal_denial
            event = self._terminal_denial_event
            loop = self._terminal_denial_loop
        if denial is not None:
            return denial
        if event is None or loop is not asyncio.get_running_loop():
            raise RuntimeError(
                "terminal-denial tracker was not reset for this invocation"
            )
        await event.wait()
        denial = self.terminal_denial
        if denial is None:  # pragma: no cover - event/state invariant
            raise RuntimeError("terminal-denial event has no exception")
        return denial

    def shutdown(
        self,
        *,
        timeout: float = DISPATCH_SHUTDOWN_TIMEOUT_S,
    ) -> None:
        """Boundedly join every exact worker or retain them for retry."""

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("dispatch shutdown timeout must be numeric")
        if not math.isfinite(float(timeout)):
            raise ValueError("dispatch shutdown timeout must be finite")
        if timeout < 0:
            raise ValueError("dispatch shutdown timeout must be non-negative")
        with self._lifecycle_lock:
            if self._shutdown_settled:
                return
            if not self._shutdown_started:
                self._shutdown_started = True
                # Cancel only jobs that never began. Running jobs retain their
                # Future and exact worker until a later successful retry.
                while True:
                    try:
                        queued = self._jobs.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if queued is not self._sentinel:
                            future, _dispatch = queued
                            future.cancel()
                    finally:
                        self._jobs.task_done()
                for _thread in self._threads:
                    self._jobs.put(self._sentinel)
            threads = self._threads
        deadline = time.monotonic() + float(timeout)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = tuple(thread.name for thread in threads if thread.is_alive())
        if alive:
            raise McpDispatchLifecycleError(
                "MCP dispatch workers remain live after bounded shutdown: "
                f"{alive!r}"
            )
        with self._lifecycle_lock:
            self._shutdown_settled = True


def _cap_error(text: str) -> str:
    """Truncate an error string to ``_TOOL_ERROR_MAX_CHARS`` with a tail."""
    if len(text) <= _TOOL_ERROR_MAX_CHARS:
        return text
    head_budget = _TOOL_ERROR_MAX_CHARS - 200
    return (
        text[:head_budget]
        + f"\n\n[error truncated; original was {len(text):,} chars]"
    )


def build_mcp_server(
    toolset: Any,
    *,
    name: str = MCP_SERVER_NAME,
    context_budget: int | None = None,
    tool_timeout: float = TOOL_TIMEOUT_S,
    max_workers: int = 2,
) -> tuple[Any, list[str], _DispatchWorker, ContextBudgetTracker | None]:
    """Wrap ``toolset`` as an in-process MCP server config.

    ``toolset`` must satisfy the ``assay.ToolSet`` protocol:
    ``to_anthropic_tools() -> list[dict]`` (each dict carries
    ``name`` / ``description`` / ``input_schema``) and
    ``dispatch(name, arguments) -> str``. Works with single-domain
    ``BuiltToolSet``, ``CompositeToolSet``, or any other adapter.
    When set, ``context_budget`` is a cumulative UTF-8 byte ceiling across
    successful tool-result payloads; bridge-generated control annotations do
    not consume that allowance.

    Returns
    -------
    (server_config, mcp_tool_names, worker, budget_tracker)
        ``server_config`` is the ``McpSdkServerConfig`` dict to drop
        into ``ClaudeAgentOptions.mcp_servers``. ``mcp_tool_names`` is
        the list of ``mcp__<name>__<tool>`` strings to extend
        ``allowed_tools`` with. ``worker`` is the dispatch thread pool
        — caller should call ``worker.shutdown()`` on teardown.
        ``budget_tracker`` is the ``ContextBudgetTracker`` if
        ``context_budget`` was set, else None — caller may call
        ``budget_tracker.reset()`` between invocations to make the cap
        per-invocation rather than per-attachment.
    """
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

    worker = _DispatchWorker(toolset, max_workers=max_workers)
    try:
        budget_tracker = (
            ContextBudgetTracker(context_budget) if context_budget else None
        )
    except BaseException:
        worker.shutdown()
        raise
    _seq = itertools.count(1)
    try:
        tool_descriptors = toolset.to_anthropic_tools()
        write_names = frozenset(
            str(tool_name)
            for tool_name in getattr(toolset, "write_names", ())
        )
    except BaseException:
        worker.shutdown()
        raise

    def _make_handler(tool_name: str):
        async def _handler(arguments: dict[str, Any]) -> dict[str, Any]:
            seq = next(_seq)
            arg_summary = ", ".join(
                f"{k}={str(v)[:80]}"
                for k, v in list((arguments or {}).items())[:4]
            )
            logger.info("MCP tool #%d: %s(%s)", seq, tool_name, arg_summary)
            effective_timeout = (
                BUILD_TOOL_TIMEOUT_S
                if tool_name in _LONG_RUNNING_TOOL_NAMES
                else tool_timeout
            )
            is_error = False
            t0 = time.monotonic()
            try:
                fut = worker.submit(tool_name, arguments or {})
                deadline = t0 + effective_timeout
                while not fut.done():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        cancelled_before_start = fut.cancel()
                        if not cancelled_before_start and fut.done():
                            # The dispatch settled on the deadline edge.  Its
                            # exact result/exception remains authoritative.
                            break
                        if (
                            tool_name in write_names
                            and not cancelled_before_start
                        ):
                            raise ToolMutationIndeterminate(
                                f"mutating tool {tool_name!r} exceeded its "
                                "MCP dispatch timeout while still running; "
                                "source settlement is indeterminate"
                            )
                        raise TimeoutError(tool_name)
                    await asyncio.sleep(min(1.0, remaining))
                result_text = fut.result(timeout=0)
            except (TimeoutError, concurrent.futures.TimeoutError):
                elapsed = time.monotonic() - t0
                is_error = True
                logger.error(
                    "MCP tool #%d: %s TIMED OUT after %.1fs (limit %ss)",
                    seq, tool_name, elapsed, effective_timeout,
                )
                result_text = (
                    f"Error: tool {tool_name!r} timed out after "
                    f"{effective_timeout:.0f}s. Try a narrower query."
                )
            except TerminalToolDispatchError as exc:
                elapsed = time.monotonic() - t0
                worker.record_terminal_denial(exc)
                is_error = True
                logger.warning(
                    "MCP tool #%d: %s TERMINAL DISPATCH after "
                    "%.1fs: %s",
                    seq, tool_name, elapsed, exc,
                )
                if isinstance(exc, ToolMutationIndeterminate):
                    result_text = _cap_error(
                        f"Error: {type(exc).__name__}: {exc}"
                    )
                else:
                    result_text = _cap_error(
                        "Error: terminal tool-dispatch budget exhausted: "
                        f"{exc}"
                    )
            except Exception as exc:  # noqa: BLE001 — surfaced to agent
                elapsed = time.monotonic() - t0
                is_error = True
                logger.warning(
                    "MCP tool #%d: %s ERROR after %.1fs: %s",
                    seq, tool_name, elapsed, exc,
                )
                result_text = _cap_error(
                    f"Error: {type(exc).__name__}: {exc}"
                )
            else:
                elapsed = time.monotonic() - t0
                logger.info(
                    "MCP tool #%d: %s -> %d chars (%.1fs)",
                    seq, tool_name, len(result_text), elapsed,
                )
            # Budget tracking applies only to successful results — errors
            # must always reach the agent untracked so it can adapt
            # (errors are size-capped via ``_cap_error`` instead, so a
            # multi-MB exception payload can't blow up the conversation).
            if budget_tracker is not None and not is_error:
                result_text = budget_tracker.apply(result_text)
            return {
                "content": [{"type": "text", "text": result_text}],
                "is_error": is_error,
            }

        return _handler

    try:
        sdk_tools = [
            SdkMcpTool(
                name=t["name"],
                description=t["description"],
                input_schema=t["input_schema"],
                handler=_make_handler(t["name"]),
            )
            for t in tool_descriptors
        ]
        server = create_sdk_mcp_server(name=name, tools=sdk_tools)
        mcp_tool_names = [
            f"mcp__{name}__{t['name']}" for t in tool_descriptors
        ]
    except BaseException:
        worker.shutdown()
        raise
    return server, mcp_tool_names, worker, budget_tracker


__all__ = [
    "MCP_SERVER_NAME",
    "ContextBudgetTracker",
    "McpDispatchLifecycleError",
    "build_mcp_server",
]
