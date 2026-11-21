# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Lifecycle + dispatch tests for ``crucible.tools._mcp_http_server``.

The HTTP MCP server is the parity-restoration layer for non-Claude
backends. Its lifecycle guarantees are load-bearing — every test here
either checks that:

1. The server starts cleanly on an OS-assigned port and serves a real
   MCP handshake + tool call, or
2. ``stop()`` reliably releases that port (graceful → forced → atexit
   backstop) so a long-running crucible process can't accumulate
   leaked sockets.

A failing test here would either drop tools the operator routed to
Codex (silent correctness regression) or leak listening sockets
across invocations (operational regression).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import socket
import subprocess
import sys
import threading
import time
import textwrap

import pytest

from crucible.tools import _mcp_http_server as _server_module
from crucible.tools._mcp_bridge import ContextBudgetTracker
from crucible.tools._dispatch_budget import (
    ToolDispatchBudgetExceeded,
    ToolMutationIndeterminate,
)
from crucible.tools._mcp_http_server import (
    BUILD_TOOL_TIMEOUT_S,
    TOOL_TIMEOUT_S,
    CrucibleHttpMcpServer,
    McpServerLifecycleError,
)


# ----------------------------------------------------------------------
# Toolset fixtures
# ----------------------------------------------------------------------


class _RecordingToolset:
    """Minimal ToolSet satisfying the protocol the bridge consumes.

    Exposes a single ``echo`` tool that returns the JSON-stringified
    args + records every dispatch for the test to assert on.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    @property
    def read_only_names(self) -> set[str]:
        return {"echo", "fail", "slow"}

    @property
    def write_names(self) -> set[str]:
        return set()

    def to_anthropic_tools(self) -> list[dict]:
        return [
            {
                "name": "echo",
                "description": "Return the JSON-stringified arguments.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "payload": {"type": "string"},
                    },
                    "required": ["payload"],
                },
            },
            {
                "name": "fail",
                "description": "Raise on dispatch.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "slow",
                "description": "Block past the test's tool_timeout.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def dispatch(self, name: str, arguments: dict) -> str:
        self.calls.append((name, dict(arguments)))
        if name == "echo":
            return f"echo:{arguments.get('payload', '')}"
        if name == "fail":
            raise RuntimeError("boom from toolset")
        if name == "slow":
            time.sleep(5.0)
            return "slow:done"
        return f"ERROR: unknown tool {name!r}"


def _port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if TCP connect succeeds within ``timeout`` seconds."""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except (ConnectionRefusedError, OSError):
        return False
    finally:
        s.close()


def _port_from(url: str) -> int:
    # http://host:port/mcp -> port
    rest = url.split("://", 1)[1]
    hostport = rest.split("/", 1)[0]
    return int(hostport.split(":", 1)[1])


# ----------------------------------------------------------------------
# start() / stop() lifecycle
# ----------------------------------------------------------------------


def test_start_returns_loopback_mcp_url():
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    try:
        url = srv.start()
        assert url.startswith("http://127.0.0.1:")
        assert url.endswith("/mcp")
        assert srv.is_running
    finally:
        srv.stop()


def test_stop_releases_the_port():
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    url = srv.start()
    port = _port_from(url)
    listener = srv._listener_socket  # noqa: SLF001
    assert listener is not None
    assert listener.fileno() >= 0
    assert listener.get_inheritable() is False
    assert _port_listening("127.0.0.1", port), "server should be reachable"

    srv.stop()
    # Graceful shutdown should release the listening socket within
    # the default 5s grace window. Poll briefly to avoid a flaky
    # race against uvicorn's teardown.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _port_listening("127.0.0.1", port, timeout=0.1):
            break
        time.sleep(0.05)
    assert not _port_listening("127.0.0.1", port), (
        f"port {port} still listening after stop() — socket leak"
    )
    assert listener.fileno() == -1
    assert not srv.is_running


def test_stop_releases_listener_for_immediate_rebind():
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    port = _port_from(srv.start())
    srv.stop()

    replacement = socket.socket()
    replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        replacement.bind(("127.0.0.1", port))
    finally:
        replacement.close()


def test_stop_is_idempotent():
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    srv.start()
    srv.stop()
    # Second stop must not raise + must remain a no-op.
    srv.stop()
    assert not srv.is_running


def test_stubborn_uvicorn_thread_fails_closed_and_can_be_retried():
    class _StubbornThread:
        daemon = True

        def __init__(self) -> None:
            self.alive = True
            self.join_timeouts: list[float | None] = []

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)

    class _FakeUvicornServer:
        should_exit = False
        force_exit = False

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = int(listener.getsockname()[1])

    srv = CrucibleHttpMcpServer(
        _RecordingToolset(),
        graceful_timeout=0.0,
        force_timeout=0.0,
    )
    thread = _StubbornThread()
    fake_uvicorn = _FakeUvicornServer()
    srv._started = True  # noqa: SLF001
    srv._thread = thread  # type: ignore[assignment]  # noqa: SLF001
    srv._server = fake_uvicorn  # noqa: SLF001
    srv._listener_socket = listener  # noqa: SLF001
    CrucibleHttpMcpServer._live_instances.add(srv)  # noqa: SLF001

    with pytest.raises(
        McpServerLifecycleError,
        match="daemon thread remains alive",
    ):
        srv.stop()

    assert fake_uvicorn.should_exit is True
    assert fake_uvicorn.force_exit is True
    assert thread.join_timeouts == [0.0, 0.0]
    assert listener.fileno() == -1
    assert not _port_listening("127.0.0.1", port)
    assert srv.is_running
    assert not srv._stopped  # noqa: SLF001
    assert srv in CrucibleHttpMcpServer._live_instances  # noqa: SLF001
    with pytest.raises(RuntimeError, match="single-use"):
        srv.start()

    # Once the external lifecycle condition settles, retrying stop confirms
    # the dead thread and transitions the single-use instance exactly once.
    thread.alive = False
    srv.stop()
    srv.stop()
    assert not srv.is_running
    assert srv._stopped  # noqa: SLF001
    assert srv not in CrucibleHttpMcpServer._live_instances  # noqa: SLF001


def test_start_after_stop_raises():
    """Single-use guarantee — re-starting a stopped server is a bug."""
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    srv.start()
    srv.stop()
    with pytest.raises(RuntimeError, match="single-use"):
        srv.start()


def test_double_start_returns_same_url():
    """Idempotent start: calling start() twice returns the cached URL."""
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    try:
        url1 = srv.start()
        url2 = srv.start()
        assert url1 == url2
    finally:
        srv.stop()


def test_context_manager_starts_and_stops():
    ts = _RecordingToolset()
    with CrucibleHttpMcpServer(ts) as srv:
        assert srv.is_running
        port = _port_from(srv.url)
        assert _port_listening("127.0.0.1", port)
    assert not srv.is_running


def test_url_property_raises_before_start():
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    with pytest.raises(RuntimeError, match="before start"):
        srv.url


# ----------------------------------------------------------------------
# Dispatch via MCP client
# ----------------------------------------------------------------------


def _mcp_call(
    url: str,
    tool_name: str,
    args: dict,
    *,
    headers: dict[str, str] | None = None,
) -> str:
    """Run a single tool call against the server via the streamable-http
    MCP client; return the first text content string."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    async def go() -> str:
        async with streamablehttp_client(
            url,
            headers=headers,
        ) as (read, write, _info):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(tool_name, args)
                return res.content[0].text if res.content else ""

    return asyncio.run(go())


def _mcp_list(url: str) -> list[str]:
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

    async def go() -> list[str]:
        async with streamablehttp_client(url) as (read, write, _info):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.list_tools()
                return [t.name for t in res.tools]

    return asyncio.run(go())


def _mcp_list_descriptors(url: str):
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def go():
        async with streamablehttp_client(url) as (read, write, _info):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    return asyncio.run(go())


def test_list_tools_returns_toolset_descriptors():
    ts = _RecordingToolset()
    with CrucibleHttpMcpServer(ts) as srv:
        names = _mcp_list(srv.url)
    assert names == ["echo", "fail", "slow"]


def test_list_tools_carries_capability_annotations():
    class _MixedToolset:
        read_only_names = {"inspect"}
        write_names = {"apply_patch"}

        def to_anthropic_tools(self):
            schema = {"type": "object", "properties": {}}
            return [
                {
                    "name": "inspect",
                    "description": "Read a confined file.",
                    "input_schema": schema,
                },
                {
                    "name": "apply_patch",
                    "description": "Patch a confined file.",
                    "input_schema": schema,
                },
            ]

        def dispatch(self, name, arguments):
            return f"{name}:{arguments}"

    with CrucibleHttpMcpServer(_MixedToolset()) as srv:
        descriptors = {
            tool.name: tool for tool in _mcp_list_descriptors(srv.url)
        }

    read = descriptors["inspect"].annotations
    assert read is not None
    assert read.readOnlyHint is True
    assert read.destructiveHint is False
    assert read.idempotentHint is True
    assert read.openWorldHint is None

    write = descriptors["apply_patch"].annotations
    assert write is not None
    assert write.readOnlyHint is False
    assert write.destructiveHint is True
    assert write.idempotentHint is False
    assert write.openWorldHint is None


@pytest.mark.parametrize(
    ("read_names", "write_names", "match"),
    [
        ({"tool"}, {"tool"}, "both read-only and mutating"),
        (set(), set(), "unclassified=\\['tool'\\]"),
        ({"tool", "ghost"}, set(), "unknown=\\['ghost'\\]"),
    ],
)
def test_tool_capability_annotations_fail_closed(
    read_names, write_names, match,
):
    class _BadMetadataToolset:
        def __init__(self):
            self.read_only_names = read_names
            self.write_names = write_names

        def to_anthropic_tools(self):
            return [{
                "name": "tool",
                "description": "test",
                "input_schema": {"type": "object", "properties": {}},
            }]

        def dispatch(self, name, arguments):
            return "ok"

    server = CrucibleHttpMcpServer(_BadMetadataToolset())
    with pytest.raises(ValueError, match=match):
        server.start()
    assert not server.is_running


def test_dispatch_round_trips_arguments():
    ts = _RecordingToolset()
    with CrucibleHttpMcpServer(ts) as srv:
        text = _mcp_call(srv.url, "echo", {"payload": "hello"})
    assert text == "echo:hello"
    assert ts.calls == [("echo", {"payload": "hello"})]


def test_bearer_auth_is_required_and_never_enters_url():
    import httpx

    ts = _RecordingToolset()
    token = "private-mcp-token-" + "x" * 32
    with CrucibleHttpMcpServer(ts, auth_token=token) as srv:
        assert token not in srv.url
        assert httpx.post(srv.url).status_code == 401
        assert (
            httpx.post(
                srv.url,
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        text = _mcp_call(
            srv.url,
            "echo",
            {"payload": "authorized"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert text == "echo:authorized"
    assert ts.calls == [("echo", {"payload": "authorized"})]


@pytest.mark.parametrize("token", ["abc\tdef", "abc\x7fdef"])
def test_bearer_auth_rejects_control_characters(token: str):
    with pytest.raises(ValueError, match="control characters"):
        CrucibleHttpMcpServer(_RecordingToolset(), auth_token=token)


def test_context_budget_caps_cumulative_utf8_result_bytes():
    ts = _RecordingToolset()
    with CrucibleHttpMcpServer(ts, context_budget=5) as srv:
        text = _mcp_call(srv.url, "echo", {"payload": "ééé"})
        exhausted = _mcp_call(srv.url, "echo", {"payload": "later"})

    assert "[TRUNCATED]" in text
    assert "\ufffd" not in text
    assert "[BUDGET EXHAUSTED]" in exhausted


def test_context_budget_tracker_survives_http_bridge_restart():
    ts = _RecordingToolset()
    tracker = ContextBudgetTracker(8)

    with CrucibleHttpMcpServer(
        ts,
        context_budget_tracker=tracker,
    ) as first:
        assert _mcp_call(first.url, "echo", {"payload": "a"}).endswith(
            "echo:a"
        )
    with CrucibleHttpMcpServer(
        ts,
        context_budget_tracker=tracker,
    ) as second:
        truncated = _mcp_call(second.url, "echo", {"payload": "b"})
        exhausted = _mcp_call(second.url, "echo", {"payload": "c"})

    assert tracker.consumed_bytes == 8
    assert tracker.exhausted is True
    assert "[TRUNCATED]" in truncated
    assert "[BUDGET EXHAUSTED]" in exhausted


def test_expired_invocation_deadline_never_dispatches_tool():
    ts = _RecordingToolset()
    with CrucibleHttpMcpServer(
        ts,
        invocation_deadline=time.monotonic() - 1.0,
    ) as srv:
        text = _mcp_call(srv.url, "echo", {"payload": "must-not-run"})

    assert "deadline is exhausted" in text
    assert ts.calls == []


def test_dispatch_exception_surfaces_capped_error():
    ts = _RecordingToolset()
    with CrucibleHttpMcpServer(ts) as srv:
        text = _mcp_call(srv.url, "fail", {})
        assert not srv.terminal_dispatch_error_event.is_set()
        assert srv.terminal_dispatch_error_message is None
    assert text.startswith("Error: RuntimeError: boom from toolset")


@pytest.mark.parametrize(
    "terminal",
    (
        ToolDispatchBudgetExceeded(
            "planner tool-call limit exhausted; dispatch not started"
        ),
        ToolMutationIndeterminate("delete settlement is unknown"),
    ),
    ids=("budget", "mutation"),
)
def test_source_budget_denial_sets_terminal_adapter_signal(terminal):
    class _DeniedToolset(_RecordingToolset):
        def dispatch(self, name: str, arguments: dict) -> str:
            self.calls.append((name, dict(arguments)))
            raise terminal

    ts = _DeniedToolset()
    with CrucibleHttpMcpServer(ts) as srv:
        text = _mcp_call(srv.url, "echo", {"payload": "denied"})
        repeated = _mcp_call(srv.url, "echo", {"payload": "repeated"})

        assert type(terminal).__name__ in text
        assert type(terminal).__name__ in repeated
        assert len(ts.calls) == 1
        assert srv.terminal_dispatch_error_event.is_set()
        assert srv.terminal_dispatch_error is terminal
        assert srv.terminal_dispatch_error_message is not None
        assert str(terminal) in (
            srv.terminal_dispatch_error_message
        )

        srv.reset_terminal_dispatch_error()
        if isinstance(terminal, ToolMutationIndeterminate):
            assert srv.terminal_dispatch_error_event.is_set()
            assert srv.terminal_dispatch_error is terminal
            assert srv.terminal_dispatch_error_message is not None
        else:
            assert not srv.terminal_dispatch_error_event.is_set()
            assert srv.terminal_dispatch_error is None
            assert srv.terminal_dispatch_error_message is None


def test_late_terminal_after_http_timeout_is_published_and_latched():
    started = threading.Event()
    release = threading.Event()
    terminal = ToolMutationIndeterminate("late settlement is unknown")

    class _LateTerminalToolset(_RecordingToolset):
        def dispatch(self, name: str, arguments: dict) -> str:
            self.calls.append((name, dict(arguments)))
            started.set()
            assert release.wait(timeout=2.0)
            raise terminal

    toolset = _LateTerminalToolset()
    server = CrucibleHttpMcpServer(
        toolset,
        tool_timeout=0.02,
        force_timeout=1.0,
    )
    try:
        server.start()
        timed_out = _mcp_call(server.url, "echo", {"payload": "once"})
        assert "timed out" in timed_out.lower()
        assert started.is_set()
        assert not server.terminal_dispatch_error_event.is_set()
        # Simulate a reusable bridge crossing an invocation boundary while
        # the timed-out worker is still unsettled. Mutation uncertainty must
        # survive the captured generation mismatch.
        server.reset_terminal_dispatch_error()

        release.set()
        assert server.terminal_dispatch_error_event.wait(timeout=1.0)
        assert server.terminal_dispatch_error is terminal
        repeated = _mcp_call(
            server.url,
            "echo",
            {"payload": "must-not-dispatch"},
        )
        assert "ToolMutationIndeterminate" in repeated
        assert len(toolset.calls) == 1
    finally:
        release.set()
        server.stop()


def test_running_mutator_timeout_latches_indeterminate_before_response():
    started = threading.Event()
    release = threading.Event()

    class _SlowMutatingToolset(_RecordingToolset):
        @property
        def read_only_names(self) -> set[str]:
            return {"fail", "slow"}

        @property
        def write_names(self) -> set[str]:
            return {"echo"}

        def dispatch(self, name: str, arguments: dict) -> str:
            self.calls.append((name, dict(arguments)))
            started.set()
            assert release.wait(timeout=2.0)
            return "mutation eventually settled"

    toolset = _SlowMutatingToolset()
    server = CrucibleHttpMcpServer(
        toolset,
        tool_timeout=0.02,
        max_workers=2,
        force_timeout=1.0,
    )
    try:
        server.start()
        response = _mcp_call(server.url, "echo", {"payload": "once"})
        assert started.is_set()
        assert "ToolMutationIndeterminate" in response
        assert server.terminal_dispatch_error_event.is_set()
        assert isinstance(
            server.terminal_dispatch_error,
            ToolMutationIndeterminate,
        )
        repeated = _mcp_call(
            server.url,
            "echo",
            {"payload": "must-not-dispatch"},
        )
        assert "ToolMutationIndeterminate" in repeated
        assert len(toolset.calls) == 1
    finally:
        release.set()
        server.stop()


def test_dispatch_timeout_returns_timeout_message():
    ts = _RecordingToolset()
    # Tight timeout so the test doesn't take 5s; ``slow`` sleeps 5s.
    srv = CrucibleHttpMcpServer(
        ts,
        tool_timeout=0.25,
        force_timeout=6.0,
    )
    try:
        srv.start()
        text = _mcp_call(srv.url, "slow", {})
    finally:
        srv.stop()
    assert "timed out" in text.lower()
    assert "slow" in text


# ----------------------------------------------------------------------
# Session idle limit
# ----------------------------------------------------------------------

# The margin the idle limit keeps above a turn. Stated here, independently of
# the server module, so that shrinking the margin there fails these tests.
_IDLE_MARGIN_S = 900.0

_PLAIN_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


class _HoldingToolset(_RecordingToolset):
    """``slow`` holds its call for a chosen time instead of five seconds."""

    def __init__(self, hold_seconds: float) -> None:
        super().__init__()
        self._hold_seconds = hold_seconds

    def dispatch(self, name: str, arguments: dict) -> str:
        if name != "slow":
            return super().dispatch(name, arguments)
        self.calls.append((name, dict(arguments)))
        time.sleep(self._hold_seconds)
        return "slow:done"


def _sse_messages(response) -> list[dict]:
    """Decode the JSON-RPC messages of one streamable-HTTP answer."""
    return [
        json.loads(line[len("data:"):])
        for line in response.text.splitlines()
        if line.startswith("data:") and line[len("data:"):].strip()
    ]


def _plain_open_session(client, url: str) -> dict[str, str]:
    """Open one MCP session with plain POST requests.

    mcp's own client opens a GET stream right after the handshake, and an
    open GET stream alone holds a session. These requests open none, so
    only the requests a test makes itself can hold the session.
    """
    opened = client.post(
        url,
        headers=_PLAIN_MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "idle-limit-test", "version": "0"},
            },
        },
    )
    assert opened.status_code == 200
    (handshake,) = _sse_messages(opened)
    headers = {
        **_PLAIN_MCP_HEADERS,
        "mcp-session-id": opened.headers["mcp-session-id"],
        "mcp-protocol-version": handshake["result"]["protocolVersion"],
    }
    notified = client.post(
        url,
        headers=headers,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert notified.status_code == 202
    return headers


def _plain_call(
    client,
    url: str,
    headers: dict[str, str],
    request_id: int,
    name: str,
    arguments: dict,
):
    return client.post(
        url,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )


def _plain_call_text(response) -> str:
    assert response.status_code == 200
    (message,) = _sse_messages(response)
    return message["result"]["content"][0]["text"]


def _answer_after_idle(
    client,
    url: str,
    headers: dict[str, str],
    limit: float,
):
    """Return the session's answer once it was idle for longer than ``limit``.

    A request that reaches a live session restarts its idle period, so this
    cannot poll quickly. Each try waits a whole multiple of the limit, the
    later ones longer in case a loaded machine delayed the first expiry.
    ``tools/list`` is used because it never reaches the toolset.
    """
    response = None
    for request_id, multiple in ((901, 2.0), (902, 4.0), (903, 8.0)):
        time.sleep(multiple * limit)
        response = client.post(
            url,
            headers=headers,
            json={"jsonrpc": "2.0", "id": request_id, "method": "tools/list"},
        )
        if response.status_code != 200:
            break
    return response


def _recorded_idle_limits(monkeypatch) -> list[object]:
    """Record the idle limit mcp holds for each FastMCP a server builds."""
    import mcp.server.fastmcp as fastmcp_module

    real_fastmcp = fastmcp_module.FastMCP
    seen: list[object] = []

    def recording_fastmcp(*args, **kwargs):
        instance = real_fastmcp(*args, **kwargs)
        seen.append(instance.settings.session_idle_timeout)
        return instance

    monkeypatch.setattr(fastmcp_module, "FastMCP", recording_fastmcp)
    return seen


def test_session_idle_limit_exceeds_default_turns_and_tool_waits():
    """Tie the limit to the adapter defaults that one session must outlast.

    Raising one of these defaults without raising the limit has to fail here
    instead of silently letting a session expire inside a turn.
    """
    from crucible.backends import claude, codex, kimi

    limit = _server_module.SESSION_IDLE_TIMEOUT_S
    assert type(limit) is float
    assert math.isfinite(limit) and limit > 0

    default_turns = {
        "codex._DEFAULT_POLICY": codex._DEFAULT_POLICY.timeout_seconds,
        "CodexBackend(timeout_seconds=...)": inspect.signature(
            codex.CodexBackend.__init__
        ).parameters["timeout_seconds"].default,
        "claude._DEFAULT_POLICY": claude._DEFAULT_POLICY.timeout_seconds,
    }
    for name, seconds in default_turns.items():
        assert isinstance(seconds, float), name
        assert limit >= seconds + _IDLE_MARGIN_S, name

    shorter_waits = {
        "kimi._DEFAULT_TIMEOUT_SECONDS": kimi._DEFAULT_TIMEOUT_SECONDS,
        "TOOL_TIMEOUT_S": TOOL_TIMEOUT_S,
        "BUILD_TOOL_TIMEOUT_S": BUILD_TOOL_TIMEOUT_S,
        "kimi._MCP_TOOL_TIMEOUT_MS": kimi._MCP_TOOL_TIMEOUT_MS / 1000.0,
    }
    for name, seconds in shorter_waits.items():
        assert limit > seconds, name


def test_server_hands_its_finite_idle_limit_to_mcp(monkeypatch):
    seen = _recorded_idle_limits(monkeypatch)

    with CrucibleHttpMcpServer(_RecordingToolset()):
        pass
    # A deadline nearer than the constant never lowers the limit.
    with CrucibleHttpMcpServer(
        _RecordingToolset(),
        invocation_deadline=time.monotonic() + 60.0,
    ):
        pass

    assert seen == [
        _server_module.SESSION_IDLE_TIMEOUT_S,
        _server_module.SESSION_IDLE_TIMEOUT_S,
    ]


def test_idle_limit_follows_a_deadline_beyond_the_constant(monkeypatch):
    seen = _recorded_idle_limits(monkeypatch)
    to_deadline = 3.0 * _server_module.SESSION_IDLE_TIMEOUT_S

    before = time.monotonic()
    server = CrucibleHttpMcpServer(
        _RecordingToolset(),
        invocation_deadline=before + to_deadline,
    )
    constructed_within = time.monotonic() - before
    with server:
        pass

    (limit,) = seen
    assert isinstance(limit, float) and math.isfinite(limit)
    # Beyond the deadline by the margin, and by no more than that.
    assert limit >= to_deadline - constructed_within + _IDLE_MARGIN_S - 1e-3
    assert limit <= to_deadline + _IDLE_MARGIN_S + 1e-3


@pytest.mark.parametrize(
    "unbounded",
    [None, math.inf, math.nan, 0.0, -1.0, True],
    ids=["none", "inf", "nan", "zero", "negative", "bool"],
)
def test_idle_limit_can_never_be_unset_or_unbounded(monkeypatch, unbounded):
    """mcp reads ``None`` as "never expire"; the server must refuse it."""
    monkeypatch.setattr(_server_module, "SESSION_IDLE_TIMEOUT_S", unbounded)

    with pytest.raises(ValueError, match="SESSION_IDLE_TIMEOUT_S"):
        CrucibleHttpMcpServer(_RecordingToolset())
    with pytest.raises(ValueError, match="SESSION_IDLE_TIMEOUT_S"):
        CrucibleHttpMcpServer(
            _RecordingToolset(),
            invocation_deadline=time.monotonic() + 36_000.0,
        )


def test_requests_hold_a_session_and_idleness_past_the_limit_ends_it(
    monkeypatch,
):
    """One session, plain requests only, with the limit shortened to 2 s.

    A call in flight for longer than the limit returns its result, calls
    spaced closer than the limit keep the session alive past the limit, and
    only then does a silence longer than the limit end the session.
    """
    import httpx

    limit = 2.0
    monkeypatch.setattr(_server_module, "SESSION_IDLE_TIMEOUT_S", limit)
    toolset = _HoldingToolset(hold_seconds=2.5 * limit)

    with CrucibleHttpMcpServer(toolset) as srv, httpx.Client(
        timeout=60.0,
        trust_env=False,
    ) as client:
        headers = _plain_open_session(client, srv.url)

        started = time.monotonic()
        held = _plain_call(client, srv.url, headers, 1, "slow", {})
        in_flight_for = time.monotonic() - started
        assert _plain_call_text(held) == "slow:done"
        assert in_flight_for > limit

        started = time.monotonic()
        for request_id in (2, 3, 4):
            time.sleep(limit / 2.0)
            spaced = _plain_call(
                client,
                srv.url,
                headers,
                request_id,
                "echo",
                {"payload": str(request_id)},
            )
            assert _plain_call_text(spaced) == f"echo:{request_id}"
        assert time.monotonic() - started > limit

        expired = _answer_after_idle(client, srv.url, headers, limit)
        assert expired.status_code == 404
        assert "Session not found" in expired.text
        # The server does not revive the session for a later request.
        again = _plain_call(
            client,
            srv.url,
            headers,
            5,
            "echo",
            {"payload": "5"},
        )
        assert again.status_code == 404

    assert toolset.calls == [
        ("slow", {}),
        ("echo", {"payload": "2"}),
        ("echo", {"payload": "3"}),
        ("echo", {"payload": "4"}),
    ]


def test_abandoned_session_is_closed_after_the_idle_limit(monkeypatch):
    """Failure path: a client leaves without ending its session.

    Driven with mcp's own client, which is told not to send the closing
    DELETE. The session must stop answering once the limit this server set
    has passed; with the library default it would stay for half an hour.
    """
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    limit = 2.0
    monkeypatch.setattr(_server_module, "SESSION_IDLE_TIMEOUT_S", limit)

    async def open_and_abandon(url: str) -> tuple[str, str, str]:
        async with streamablehttp_client(
            url,
            terminate_on_close=False,
        ) as (read, write, get_session_id):
            async with ClientSession(read, write) as session:
                handshake = await session.initialize()
                result = await session.call_tool("echo", {"payload": "kept"})
                session_id = get_session_id()
                assert session_id
                return (
                    session_id,
                    str(handshake.protocolVersion),
                    result.content[0].text,
                )

    toolset = _RecordingToolset()
    with CrucibleHttpMcpServer(toolset) as srv:
        session_id, protocol_version, text = asyncio.run(
            open_and_abandon(srv.url)
        )
        assert text == "echo:kept"
        headers = {
            **_PLAIN_MCP_HEADERS,
            "mcp-session-id": session_id,
            "mcp-protocol-version": protocol_version,
        }
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            expired = _answer_after_idle(client, srv.url, headers, limit)

    assert expired.status_code == 404
    assert "Session not found" in expired.text
    assert toolset.calls == [("echo", {"payload": "kept"})]


# ----------------------------------------------------------------------
# Atexit fallback
# ----------------------------------------------------------------------


def test_atexit_hook_is_registered_exactly_once():
    """Two server instances must not register two atexit hooks.

    Otherwise the hook would walk the WeakSet twice per process exit,
    which is wasteful and (more importantly) means the registration
    pattern is leaky.
    """
    # Reset the class-level flag so the test sees the registration
    # decision from a known state. The atexit module itself doesn't
    # expose a "was this function registered" API; we observe the
    # class-side flag that gates the decision.
    CrucibleHttpMcpServer._atexit_registered = False  # noqa: SLF001
    s1 = CrucibleHttpMcpServer(_RecordingToolset())
    s2 = CrucibleHttpMcpServer(_RecordingToolset())
    try:
        s1.start()
        registered_after_first = CrucibleHttpMcpServer._atexit_registered  # noqa: SLF001
        s2.start()
        registered_after_second = CrucibleHttpMcpServer._atexit_registered  # noqa: SLF001
    finally:
        s1.stop()
        s2.stop()
    assert registered_after_first is True
    assert registered_after_second is True


def test_atexit_stop_all_walks_live_instances():
    """The class-level atexit handler must call stop() on every live
    instance, but tolerate instances that already stopped."""
    s1 = CrucibleHttpMcpServer(_RecordingToolset())
    s2 = CrucibleHttpMcpServer(_RecordingToolset())
    s1.start()
    s2.start()
    s1.stop()  # one of them already stopped before atexit fires
    # Directly invoke the class-level hook + assert both end up stopped.
    CrucibleHttpMcpServer._atexit_stop_all()
    assert not s1.is_running
    assert not s2.is_running


# ----------------------------------------------------------------------
# Thread isolation
# ----------------------------------------------------------------------


def test_uvicorn_runs_in_daemon_thread():
    """If the uvicorn-hosting thread isn't a daemon, the process can't
    exit cleanly when stop() misses (e.g. os._exit bypassing atexit)."""
    srv = CrucibleHttpMcpServer(_RecordingToolset())
    try:
        srv.start()
        thread = srv._thread  # noqa: SLF001
        assert thread is not None
        assert thread.daemon, "uvicorn thread must be daemon for clean process exit"
    finally:
        srv.stop()


def test_timed_out_dispatch_cannot_hold_interpreter_shutdown():
    """A permanently stuck sync tool must not keep Python alive.

    ``ThreadPoolExecutor`` joins workers at interpreter exit even after
    ``shutdown(wait=False)``. Exercise this in a child process so the test
    fails by timeout if the dispatch implementation ever regresses to that
    lifecycle.
    """
    script = textwrap.dedent(
        """
        import asyncio
        import threading

        from crucible.tools._mcp_http_server import (
            CrucibleHttpMcpServer,
            McpServerLifecycleError,
            _DaemonDispatchPool,
        )

        class StuckToolset:
            def dispatch(self, name, arguments):
                threading.Event().wait()

        server = CrucibleHttpMcpServer(StuckToolset(), tool_timeout=0.05)
        server._pool = _DaemonDispatchPool(
            max_workers=1,
            thread_name_prefix="crucible-mcp-exit-test",
        )
        result = asyncio.run(server._dispatch_one("hang", {}, {"hang"}))
        assert "timed out" in result.content[0].text
        try:
            server._pool.shutdown(cancel_futures=True, timeout=0.01)
        except McpServerLifecycleError:
            pass
        else:
            raise AssertionError("stuck dispatch was reported as settled")
        print("main-finished", flush=True)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "main-finished" in completed.stdout


def test_stop_retains_dispatch_pool_until_exact_worker_exits():
    entered = threading.Event()
    release = threading.Event()
    late_write = threading.Event()

    class _LateWriter:
        read_only_names: set[str] = set()
        write_names = {"write"}

        def to_anthropic_tools(self):
            return [{
                "name": "write",
                "description": "write after release",
                "input_schema": {"type": "object", "properties": {}},
            }]

        def dispatch(self, name, arguments):
            entered.set()
            release.wait()
            late_write.set()
            return "done"

    srv = CrucibleHttpMcpServer(
        _LateWriter(),
        tool_timeout=0.01,
        graceful_timeout=0.05,
        force_timeout=0.01,
    )
    srv.start()
    call = threading.Thread(
        target=_mcp_call,
        args=(srv.url, "write", {}),
        daemon=True,
    )
    call.start()
    assert entered.wait(timeout=2.0)

    with pytest.raises(McpServerLifecycleError, match="dispatch pool"):
        srv.stop()

    retained_pool = srv._pool  # noqa: SLF001
    assert retained_pool is not None
    assert not srv._stopped  # noqa: SLF001
    assert not late_write.is_set()

    release.set()
    assert late_write.wait(timeout=2.0)
    srv._graceful_timeout = 2.0  # noqa: SLF001
    srv._force_timeout = 2.0  # noqa: SLF001
    srv.stop()

    assert srv._pool is None  # noqa: SLF001
    assert srv._stopped  # noqa: SLF001
    call.join(timeout=2.0)
