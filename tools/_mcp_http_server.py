# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""HTTP MCP server bridge for subprocess-backed backends.

The Claude Agent SDK ships an *in-process* MCP primitive
(``create_sdk_mcp_server`` — see ``tools/_mcp_bridge.py``). Codex CLI and
the isolated Kimi worker instead connect to a real MCP server over the
streamable-http transport. The bridge can be long-lived for a serialized
Codex backend or authenticated and single-attempt for Kimi.

Topology
--------
The HTTP server runs in a **daemon thread of the parent crucible
process** — not a subprocess. The toolset stays referenced by the
parent (no serialization, no IPC), the daemon thread runs uvicorn,
and the backend subprocess connects back over loopback HTTP. Tool calls flow:

    backend subprocess -> HTTP -> uvicorn (this proc, daemon thread)
                                -> FastMCP handler
                                -> toolset.dispatch(name, args)
                                -> response back through FastMCP -> HTTP

Lifecycle guarantees
--------------------
Three independent shutdown paths so the listening socket never
outlives the parent Python process:

1. **Explicit** — ``CodexBackend.close()`` (and the context manager
   exit) call ``stop()``. This is the primary path; workflows wrap
   the backend in ``contextlib.closing(...)`` to guarantee it fires.
2. **atexit** — every ``CrucibleHttpMcpServer`` registers an
   ``atexit`` finalizer at ``start()`` time so a normal interpreter
   exit triggers ``stop()`` even if the caller forgot
   ``backend.close()``. Idempotent.
3. **Daemon thread** — the uvicorn-hosting thread is a daemon. If
   both above paths somehow miss (e.g. ``os._exit()``), the kernel
   reaps the thread when the process exits, releasing the port.

``stop()`` is staged: ``should_exit=True`` (graceful, default 5s),
then ``force_exit=True`` plus direct closure of the bridge-owned listener
(forced, default 2s). If the uvicorn thread still survives, ``stop()``
raises ``McpServerLifecycleError`` instead of claiming that teardown
completed; the daemon-thread guarantee remains the interpreter-exit
backstop.

Dispatch isolation
------------------
Tool handlers are sync; running them on uvicorn's asyncio loop
thread would stall the HTTP transport. We submit each
``toolset.dispatch(...)`` call to a dedicated daemon worker pool and
``await`` on the future. The workers deliberately do not use
``ThreadPoolExecutor``: CPython joins that executor's workers during
interpreter shutdown, so a timed-out tool that never returns would
otherwise keep the parent process alive forever.

Session lifetime
----------------
One MCP session belongs to one backend attempt. The session manager closes
a session that has had no HTTP request in flight for the server's idle
limit: ``SESSION_IDLE_TIMEOUT_S``, or the time left to the invocation
deadline plus a margin when the server was given a later deadline. A
closed session answers 404 and is not revived. The limit is always finite,
so a client that leaves without ending its session cannot keep one open.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import hmac
import logging
import math
import queue
import socket
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

from ._dispatch_budget import (
    TerminalToolDispatchError,
    ToolMutationIndeterminate,
)
from ._mcp_bridge import ContextBudgetTracker

logger = logging.getLogger(__name__)

# Default budgets for tool dispatch. Mirrors ``_mcp_bridge.py``:
# - Most tools (reads, greps, file writes) finish in seconds.
# - A small set of build/test/scaffold tools can legitimately run
#   minutes; those names get the long-running timeout in ``_handler``.
TOOL_TIMEOUT_S: float = 300.0
BUILD_TOOL_TIMEOUT_S: float = 1800.0

# Idle limit of one streamable-HTTP session. The mcp session manager closes
# a session that has had no HTTP request in flight for this long (library
# default: 1800 s). Its id then answers 404 and the server does not revive
# it, so the tools of the attempt that owned it are gone. One session belongs
# to one backend attempt, so the limit must exceed
# - the default wall time of an invocation: 2700 s for Codex
#   (``backends.codex._DEFAULT_POLICY`` and the ``timeout_seconds`` default of
#   ``CodexBackend``) and for Claude (``backends.claude._DEFAULT_POLICY``;
#   Claude's tools run in process and hold no HTTP session), 600 s for Kimi
#   (``backends.kimi._DEFAULT_TIMEOUT_SECONDS``); and
# - the longest wait for one tool call, ``BUILD_TOOL_TIMEOUT_S`` above and
#   Kimi's client-side ``backends.kimi._MCP_TOOL_TIMEOUT_MS``, so that a call
#   whose client went away still ends by its own dispatch timeout, which is
#   what records an indeterminate mutation, and not by session expiry.
# It stays finite so that abandoned sessions are reclaimed. Those defaults
# are not ceilings: a route or the command line may admit a longer turn. A
# server constructed with ``invocation_deadline`` therefore raises its limit
# to the time left to that deadline plus ``_SESSION_IDLE_MARGIN_S``; a server
# constructed without one keeps this constant.
# ``tests/test_mcp_http_server.py`` ties the constant to the defaults above.
SESSION_IDLE_TIMEOUT_S: float = 3600.0
_SESSION_IDLE_MARGIN_S: float = 900.0

# Same long-running set as ``_mcp_bridge._LONG_RUNNING_TOOL_NAMES``.
# Kept in sync intentionally — both bridges dispatch the same ToolSet
# shapes and should treat the same tools as long-running.
_LONG_RUNNING_TOOL_NAMES = frozenset({
    "build_project",
    "create_rust_crate",
})

# Per-error truncation cap. This bounds an error before the optional
# invocation-wide context budget is applied, preventing a multi-MB
# CalledProcessError payload from blowing up the conversation.
_TOOL_ERROR_MAX_CHARS = 16 * 1024


class McpServerLifecycleError(RuntimeError):
    """The HTTP listener could not be fully settled within its deadline."""


def _validate_auth_token(token: str) -> str:
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 1024
        or any(
            ord(character) < 0x20 or ord(character) == 0x7F
            for character in token
        )
    ):
        raise ValueError(
            "MCP auth_token must be a non-empty string without control "
            "characters"
        )
    return token


def _session_idle_timeout(invocation_deadline: float | None) -> float:
    """Return the finite idle limit for the sessions of one server.

    ``invocation_deadline`` is the validated monotonic deadline of the one
    invocation a server serves, or ``None`` for a server without one. mcp
    reads ``None`` as "never expire", so the limit is refused here unless
    it is a finite positive number.
    """
    base = SESSION_IDLE_TIMEOUT_S
    if (
        isinstance(base, bool)
        or not isinstance(base, (int, float))
        or not math.isfinite(base)
        or base <= 0
    ):
        raise ValueError(
            "SESSION_IDLE_TIMEOUT_S must be a finite positive number of "
            "seconds"
        )
    limit = float(base)
    if invocation_deadline is not None:
        limit = max(
            limit,
            invocation_deadline - time.monotonic() + _SESSION_IDLE_MARGIN_S,
        )
    return limit


class _BearerAuthMiddleware:
    """Minimal ASGI bearer gate for the private loopback listener."""

    def __init__(self, app: Any, expected: bytes) -> None:
        self._app = app
        self._expected = expected

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        supplied = b""
        for name, value in scope.get("headers", ()):
            if name.lower() == b"authorization":
                supplied = value
                break
        if not hmac.compare_digest(supplied, self._expected):
            body = b"Unauthorized"
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"text/plain; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self._app(scope, receive, send)


def _cap_error(text: str) -> str:
    if len(text) <= _TOOL_ERROR_MAX_CHARS:
        return text
    head = _TOOL_ERROR_MAX_CHARS - 200
    return text[:head] + f"\n\n[error truncated; original was {len(text):,} chars]"


_DispatchItem = tuple[
    concurrent.futures.Future[str],
    Callable[..., str],
    tuple[Any, ...],
    dict[str, Any],
]


class _DaemonDispatchPool:
    """Minimal future-returning pool whose workers cannot block process exit.

    ``ThreadPoolExecutor.shutdown(wait=False)`` sounds suitable for abandoned
    tool calls, but CPython still registers every executor worker with its
    interpreter-exit hook and joins it unconditionally. A tool stuck in a
    subprocess, driver call, or third-party library can therefore hang the
    entire Crucible process after the HTTP server has stopped.

    Python cannot safely kill a running thread. These workers are daemon
    threads and are intentionally absent from ``concurrent.futures``' global
    join registry. Shutdown cancels queued futures, wakes idle workers, and
    waits for every exact worker for a bounded interval. It raises unless all
    workers exit, so a caller cannot mistake interpreter-exit behavior for a
    positive mutation-settlement receipt.
    """

    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self._queue: queue.Queue[_DispatchItem | None] = queue.Queue()
        self._lock = threading.Lock()
        self._shutdown = False
        threads = tuple(
            threading.Thread(
                target=self._worker,
                args=(self._queue,),
                daemon=True,
                name=f"{thread_name_prefix}_{index}",
            )
            for index in range(max_workers)
        )
        started: list[threading.Thread] = []
        try:
            for thread in threads:
                thread.start()
                started.append(thread)
        except BaseException:
            # Construction failed before the pool could be published to its
            # owner, so wake every worker we did manage to start.
            for _thread in started:
                self._queue.put(None)
            raise
        self._threads = threads

    @staticmethod
    def _worker(work_queue: queue.Queue[_DispatchItem | None]) -> None:
        while True:
            item = work_queue.get()
            if item is None:
                return
            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # mirror Future executor semantics
                future.set_exception(exc)
            else:
                future.set_result(result)

    def submit(
        self,
        fn: Callable[..., str],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> concurrent.futures.Future[str]:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            future: concurrent.futures.Future[str] = concurrent.futures.Future()
            self._queue.put((future, fn, args, kwargs))
            return future

    def shutdown(
        self,
        *,
        cancel_futures: bool = True,
        timeout: float,
    ) -> None:
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a finite non-negative number")
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if cancel_futures:
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is not None:
                            item[0].cancel()
                for _thread in self._threads:
                    self._queue.put(None)

        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        live_threads = tuple(thread for thread in self._threads if thread.is_alive())
        if live_threads:
            raise McpServerLifecycleError(
                "MCP dispatch pool did not settle "
                f"{len(live_threads)} worker(s) within {timeout:.1f}s"
            )


class CrucibleHttpMcpServer:
    """Loopback HTTP MCP server hosting a Crucible ``ToolSet``.

    Codex may reuse an instance across serialized invocations; Kimi creates an
    authenticated instance for each physical attempt. ``start()`` and
    successful ``stop()`` calls are idempotent. Shutdown stages graceful →
    forced listener closure, and fails closed if the HTTP thread survives.

    Parameters
    ----------
    toolset
        Any object satisfying the assay ``ToolSet`` protocol:
        ``to_anthropic_tools() -> list[dict]`` + ``dispatch(name, args)
        -> str``. Works with ``BuiltToolSet``, ``CompositeToolSet``, or
        a custom adapter — the bridge talks to the protocol surface,
        not the concrete class.
    name
        Logical MCP server name; surfaces in tool names as
        ``mcp__<name>__<tool>`` on the Codex CLI side.
    host
        Loopback address to bind. Default ``127.0.0.1``; do not bind
        to ``0.0.0.0`` — these tools execute arbitrary code on local
        paths.
    max_workers
        Dispatch thread pool size. Matches ``_mcp_bridge.py``
        default of 2 (one build slot + one read slot historically).
    auth_token
        Optional bearer credential required on every HTTP request. Kimi uses
        a fresh high-entropy token for each physical attempt.
    context_budget
        Optional cumulative UTF-8 byte ceiling for tool results served by this
        instance.
    context_budget_tracker
        Optional tracker owned by the logical invocation. Supplying the same
        tracker to replacement bridge instances preserves one cumulative
        allowance across physical retries.
    invocation_deadline
        Optional absolute ``time.monotonic()`` deadline. Calls received after
        it are rejected before dispatch, and in-flight waits are capped by it.
    """

    # Class-level set of every live instance — used by the atexit
    # hook to fire stop() for any server still up at interpreter exit.
    # WeakSet so finalized instances drop out without us tracking them.
    _live_instances: "weakref.WeakSet[CrucibleHttpMcpServer]" = (
        weakref.WeakSet()
    )
    _atexit_registered = False
    _atexit_lock = threading.Lock()

    def __init__(
        self,
        toolset: Any,
        *,
        name: str = "crucible",
        host: str = "127.0.0.1",
        max_workers: int = 2,
        tool_timeout: float = TOOL_TIMEOUT_S,
        graceful_timeout: float = 5.0,
        force_timeout: float = 2.0,
        start_timeout: float = 10.0,
        auth_token: str | None = None,
        context_budget: int | None = None,
        context_budget_tracker: ContextBudgetTracker | None = None,
        invocation_deadline: float | None = None,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError(
                "CrucibleHttpMcpServer must bind exactly to 127.0.0.1"
            )
        self._toolset = toolset
        self._name = name
        self._host = host
        self._max_workers = max_workers
        self._tool_timeout = tool_timeout
        self._graceful_timeout = graceful_timeout
        self._force_timeout = force_timeout
        self._start_timeout = start_timeout
        self._auth_header = (
            None
            if auth_token is None
            else f"Bearer {_validate_auth_token(auth_token)}".encode("utf-8")
        )
        if context_budget is not None and context_budget_tracker is not None:
            raise ValueError(
                "context_budget and context_budget_tracker are mutually exclusive"
            )
        if context_budget_tracker is not None and not isinstance(
            context_budget_tracker,
            ContextBudgetTracker,
        ):
            raise TypeError(
                "context_budget_tracker must be a ContextBudgetTracker"
            )
        self._context_budget = (
            context_budget_tracker
            if context_budget_tracker is not None
            else (
                None
                if context_budget is None
                else ContextBudgetTracker(context_budget)
            )
        )
        if invocation_deadline is not None and (
            isinstance(invocation_deadline, bool)
            or not isinstance(invocation_deadline, (int, float))
            or not math.isfinite(float(invocation_deadline))
        ):
            raise ValueError("invocation_deadline must be a finite monotonic time")
        self._invocation_deadline = (
            None
            if invocation_deadline is None
            else float(invocation_deadline)
        )
        self._session_idle_timeout = _session_idle_timeout(
            self._invocation_deadline
        )

        self._lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._stop_attempted = False
        self._thread: threading.Thread | None = None
        self._server: Any = None  # uvicorn.Server, lazily imported
        self._listener_socket: socket.socket | None = None
        self._port: int | None = None
        self._pool: _DaemonDispatchPool | None = None
        self._thread_error: BaseException | None = None
        self._terminal_dispatch_error_event = threading.Event()
        self._terminal_dispatch_error_lock = threading.Lock()
        self._terminal_dispatch_error_message: str | None = None
        self._terminal_dispatch_error: TerminalToolDispatchError | None = None
        self._terminal_dispatch_generation = 0
        self._write_names: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Streamable-HTTP MCP endpoint URL. Raises if not started."""
        if not self._started or self._port is None:
            raise RuntimeError("CrucibleHttpMcpServer.url accessed before start()")
        return f"http://{self._host}:{self._port}/mcp"

    @property
    def is_running(self) -> bool:
        thread = self._thread
        # Report the physical lifecycle, including a thread that survived a
        # failed stop attempt. State flags alone must never let callers infer
        # that a listener-hosting thread was settled when it was not.
        return thread is not None and thread.is_alive()

    def reset_context_budget(self, context_budget: int | None) -> None:
        """Reset the cumulative tool-result budget for a new invocation."""

        self._context_budget = (
            None
            if context_budget is None
            else ContextBudgetTracker(context_budget)
        )

    def reset_terminal_dispatch_error(self) -> None:
        """Clear the prior invocation's terminal dispatch signal.

        Codex reuses one read-only HTTP bridge across serialized logical
        invocations.  A workflow-owned source-budget denial is terminal only
        for the invocation that issued it, so the adapter resets this signal
        at the same invocation boundary as the context budget.
        """

        with self._terminal_dispatch_error_lock:
            # A mutation whose settlement is unknown is a bridge-lifecycle
            # failure, not an invocation-local model budget.  Never let a
            # reusable bridge boundary erase it before the owning dispatch
            # worker has settled.
            if isinstance(
                self._terminal_dispatch_error,
                ToolMutationIndeterminate,
            ):
                self._terminal_dispatch_error_event.set()
                return
            self._terminal_dispatch_generation += 1
            self._terminal_dispatch_error_message = None
            self._terminal_dispatch_error = None
            self._terminal_dispatch_error_event.clear()

    @property
    def terminal_dispatch_error_event(self) -> threading.Event:
        """Event set when dispatch reaches a workflow-terminal boundary."""

        return self._terminal_dispatch_error_event

    @property
    def terminal_dispatch_error_message(self) -> str | None:
        """Return the first workflow-terminal dispatch error, if any."""

        with self._terminal_dispatch_error_lock:
            return self._terminal_dispatch_error_message

    @property
    def terminal_dispatch_error(self) -> TerminalToolDispatchError | None:
        """Return the typed workflow-terminal dispatch error, if any."""

        with self._terminal_dispatch_error_lock:
            return self._terminal_dispatch_error

    def _record_terminal_dispatch_error(
        self,
        exc: TerminalToolDispatchError,
        *,
        generation: int,
    ) -> None:
        """Publish the first terminal dispatch error to the owning adapter."""

        message = f"{type(exc).__name__}: {exc}"
        with self._terminal_dispatch_error_lock:
            mutation_indeterminate = isinstance(
                exc,
                ToolMutationIndeterminate,
            )
            if (
                generation != self._terminal_dispatch_generation
                and not mutation_indeterminate
            ):
                return
            if mutation_indeterminate and not isinstance(
                self._terminal_dispatch_error,
                ToolMutationIndeterminate,
            ):
                # Mutation uncertainty has process-lifecycle scope.  It also
                # takes precedence over a prior invocation-local budget
                # terminal recorded concurrently on another worker.
                self._terminal_dispatch_error_message = message
                self._terminal_dispatch_error = exc
                self._terminal_dispatch_error_event.set()
            elif self._terminal_dispatch_error_message is None:
                self._terminal_dispatch_error_message = message
                self._terminal_dispatch_error = exc
                self._terminal_dispatch_error_event.set()

    def start(self) -> str:
        """Start the HTTP server. Idempotent. Returns the MCP URL."""
        with self._lock:
            if self._started and not self._stop_attempted:
                return self.url
            if self._stop_attempted or self._stopped:
                raise RuntimeError(
                    "CrucibleHttpMcpServer.start() called after stop(); "
                    "the instance is single-use. Create a new one."
                )
            try:
                self._do_start()
            except BaseException:
                # ``_do_start`` allocates the dispatch pool before building
                # the ASGI app and binding the socket. Any failure along that
                # path must tear down partial resources; otherwise a failed
                # lazy start leaks idle workers and may leave a half-started
                # listener behind.
                self._stop_attempted = True
                try:
                    self._do_stop()
                except BaseException as shutdown_error:
                    type(self)._live_instances.add(self)
                    self._ensure_atexit_registered()
                    shutdown_error.add_note(
                        "CrucibleHttpMcpServer startup also failed before "
                        "shutdown became necessary."
                    )
                    raise
                self._stopped = True
                raise
            self._started = True
            type(self)._live_instances.add(self)
            self._ensure_atexit_registered()
            return self.url

    def stop(self) -> None:
        """Stop the HTTP server and confirm its listener thread exited.

        Successful calls are idempotent. A thread that survives both bounded
        shutdown stages raises :class:`McpServerLifecycleError`; the instance
        remains registered for an explicit or atexit teardown retry.
        """
        with self._lock:
            if self._stopped:
                return
            self._stop_attempted = True
            self._do_stop()
            self._stopped = True
            type(self)._live_instances.discard(self)

    def __enter__(self) -> "CrucibleHttpMcpServer":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_atexit_registered(cls) -> None:
        """Register the class-level atexit hook exactly once.

        Walks every live instance at interpreter exit and calls
        ``stop()`` on each. Each ``stop()`` is itself idempotent, so
        re-entering through an instance that already stopped via
        ``backend.close()`` is a cheap no-op.
        """
        with cls._atexit_lock:
            if cls._atexit_registered:
                return
            atexit.register(cls._atexit_stop_all)
            cls._atexit_registered = True

    @classmethod
    def _atexit_stop_all(cls) -> None:
        # Snapshot first — stop() removes from WeakSet via __del__ path
        # (instances may dealloc while we iterate).
        for inst in list(cls._live_instances):
            try:
                inst.stop()
            except Exception:  # noqa: BLE001 — last-ditch teardown
                logger.warning(
                    "CrucibleHttpMcpServer atexit stop() raised; "
                    "daemon thread will die with the interpreter.",
                    exc_info=True,
                )

    def _do_start(self) -> None:
        """Spawn the daemon thread + wait for ``server.started``."""
        # Lazy imports — keep module import cheap; uvicorn + FastMCP
        # are sizable.
        import uvicorn  # noqa: PLC0415
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415

        # Build the dispatch pool first so handlers can reference it.
        self._pool = _DaemonDispatchPool(
            max_workers=self._max_workers,
            thread_name_prefix=f"crucible-mcp-{self._name}",
        )

        # FastMCP carries the Starlette + StreamableHTTPSessionManager
        # plumbing we need; we use it strictly for the transport. Tool
        # registration goes through the underlying lowlevel server
        # (``mcp._mcp_server``) so we author tool descriptors verbatim
        # from ``toolset.to_anthropic_tools()`` instead of having
        # FastMCP synthesize a pydantic arg model from a Python
        # function signature (which can't represent the toolset's
        # JSON Schema faithfully). The idle limit is always passed: the
        # library default is shorter than a default invocation.
        mcp = FastMCP(
            self._name,
            session_idle_timeout=self._session_idle_timeout,
        )
        self._register_tools(mcp)

        app: Any = mcp.streamable_http_app()
        if self._auth_header is not None:
            app = _BearerAuthMiddleware(app, self._auth_header)
        cfg = uvicorn.Config(
            app,
            host=self._host,
            port=0,  # OS-assigned
            log_level="warning",
            timeout_graceful_shutdown=self._graceful_timeout,
            # Don't let uvicorn install signal handlers — we're inside
            # the parent process's signal scope and stealing SIGINT
            # would mask Ctrl-C for the operator.
        )
        cfg.install_signal_handlers = False  # type: ignore[attr-defined]
        self._server = uvicorn.Server(cfg)
        # Bind in the parent thread and retain this exact socket. Uvicorn takes
        # transport ownership once ``serve(sockets=...)`` starts, but keeping
        # the Python socket object lets forced teardown close the underlying
        # descriptor even if uvicorn's loop stops making progress.
        listener_socket = cfg.bind_socket()
        # ``Config.bind_socket()`` marks sockets inheritable for uvicorn's
        # multi-process launchers. This bridge runs one in-process thread, so
        # fail closed against leaking listener ownership into backend children.
        listener_socket.set_inheritable(False)
        self._listener_socket = listener_socket
        self._port = int(listener_socket.getsockname()[1])

        def _run() -> None:
            try:
                asyncio.run(self._server.serve(sockets=[listener_socket]))
            except BaseException as exc:
                # Uvicorn reports bind failure as ``SystemExit``. Capture it
                # rather than letting pytest/interpreter thread-exception
                # hooks treat it as an unhandled background failure; start()
                # re-raises a lifecycle error with this as its cause.
                self._thread_error = exc
                if not isinstance(exc, SystemExit):
                    logger.exception(
                        "CrucibleHttpMcpServer[%s] uvicorn loop crashed",
                        self._name,
                    )

        self._thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f"crucible-mcp-{self._name}-http",
        )
        self._thread.start()

        # Wait for server.started flip OR start_timeout.
        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if self._server.started:
                break
            if not self._thread.is_alive():
                error = RuntimeError(
                    "CrucibleHttpMcpServer: uvicorn thread died before "
                    "server.started flipped — check logs above."
                )
                if self._thread_error is not None:
                    raise error from self._thread_error
                raise error
            time.sleep(0.025)
        else:
            raise RuntimeError(
                f"CrucibleHttpMcpServer: server did not start within "
                f"{self._start_timeout}s"
            )

        # Read the OS-assigned port from the bound socket.
        servers = getattr(self._server, "servers", None)
        if not servers or not servers[0].sockets:
            raise RuntimeError(
                "CrucibleHttpMcpServer: server started but exposed no "
                "bound socket (uvicorn API drift?)"
            )
        logger.info(
            "CrucibleHttpMcpServer[%s] listening on http://%s:%d/mcp",
            self._name, self._host, self._port,
        )

    def _do_stop(self) -> None:
        server = self._server
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=self._graceful_timeout)
            if thread.is_alive():
                logger.warning(
                    "CrucibleHttpMcpServer[%s] graceful shutdown timed "
                    "out after %.1fs; escalating to force_exit and "
                    "direct listener closure.",
                    self._name, self._graceful_timeout,
                )
                if server is not None:
                    server.force_exit = True
                self._close_listener_socket()
                thread.join(timeout=self._force_timeout)
        # On a normal uvicorn exit this is an idempotent close of an already
        # closed descriptor. On forced exit it is the fail-closed guarantee
        # that the bridge no longer owns an accepting loopback socket.
        self._close_listener_socket()
        pool = self._pool
        if pool is not None:
            # A listener shutdown is not a mutation-settlement receipt. Keep
            # ownership of this exact pool until every dispatch worker exits;
            # a failed stop can then retry the same physical workers.
            pool.shutdown(
                cancel_futures=True,
                timeout=self._force_timeout,
            )
            self._pool = None
        if thread is not None and thread.is_alive():
            error = McpServerLifecycleError(
                "CrucibleHttpMcpServer shutdown did not settle the uvicorn "
                f"thread within {self._graceful_timeout + self._force_timeout:.1f}s; "
                "the listener socket was closed directly, but the daemon "
                "thread remains alive"
            )
            logger.error("%s", error)
            raise error

    def _close_listener_socket(self) -> None:
        """Close the retained bound socket; safe to call repeatedly."""

        listener = self._listener_socket
        if listener is None:
            return
        try:
            listener.close()
        except OSError as exc:
            raise McpServerLifecycleError(
                "CrucibleHttpMcpServer could not close its bound listener "
                "socket"
            ) from exc

    def _register_tools(self, mcp: Any) -> None:
        """Register ``list_tools`` / ``call_tool`` on the lowlevel server.

        FastMCP's ``add_tool()`` builds a pydantic argument model from
        the handler's Python signature, which can't faithfully
        represent the toolset's hand-authored JSON Schema (default
        values, enums, nested object shapes). Bypass that by
        registering handlers on the lowlevel ``MCPServer`` directly —
        we author ``mcp.types.Tool`` descriptors verbatim from
        ``toolset.to_anthropic_tools()`` and the call dispatcher
        receives the raw ``(name, arguments)`` from the wire.
        """
        from mcp import types  # noqa: PLC0415

        # FastMCP's __init__ already registers its own list_tools and
        # call_tool handlers on the lowlevel server's request_handlers
        # dict (see ``FastMCP._setup_handlers``). Re-registering replaces
        # them — last write wins.
        server = mcp._mcp_server  # noqa: SLF001

        descriptors = list(self._toolset.to_anthropic_tools())
        descriptor_names = {str(d["name"]) for d in descriptors}
        if len(descriptor_names) != len(descriptors):
            raise ValueError(
                "MCP toolset contains duplicate tool descriptors"
            )

        write_metadata = getattr(self._toolset, "write_names", None)
        if write_metadata is None:
            raise ValueError(
                "MCP toolset has no write_names capability metadata"
            )
        write_names = {str(name) for name in write_metadata}
        self._write_names = frozenset(write_names)

        read_metadata = getattr(self._toolset, "read_only_names", None)
        read_only_names = (
            descriptor_names - write_names
            if read_metadata is None
            else {str(name) for name in read_metadata}
        )
        overlap = read_only_names & write_names
        if overlap:
            raise ValueError(
                "MCP tool capability metadata classifies tools as both "
                f"read-only and mutating: {sorted(overlap)}"
            )
        classified_names = read_only_names | write_names
        if classified_names != descriptor_names:
            missing = descriptor_names - classified_names
            unknown = classified_names - descriptor_names
            details = []
            if missing:
                details.append(f"unclassified={sorted(missing)}")
            if unknown:
                details.append(f"unknown={sorted(unknown)}")
            raise ValueError(
                "MCP tool capability metadata does not match descriptors: "
                + ", ".join(details)
            )

        tool_descriptors = [
            types.Tool(
                name=d["name"],
                description=d.get("description", ""),
                inputSchema=d.get("input_schema") or {},
                annotations=types.ToolAnnotations(
                    readOnlyHint=d["name"] in read_only_names,
                    # ``write_names`` is deliberately conservative: a
                    # mutating tool makes no non-destructive or idempotent
                    # promise, while a read-only tool has no side effect.
                    destructiveHint=d["name"] in write_names,
                    idempotentHint=d["name"] in read_only_names,
                ),
            )
            for d in descriptors
        ]
        # Index by name so call_tool can look up the per-tool timeout
        # bucket and surface a sensible error if the wire asks for
        # something the toolset doesn't export.
        known_names = {d.name for d in tool_descriptors}

        @server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            return tool_descriptors

        @server.call_tool(validate_input=False)
        async def _call_tool(
            tool_name: str, arguments: dict,
        ) -> types.CallToolResult:
            return await self._dispatch_one(tool_name, arguments, known_names)

    async def _dispatch_one(
        self,
        tool_name: str,
        arguments: dict,
        known_names: set[str],
    ) -> Any:
        """Run one tool call on the dispatch pool with per-tool timeout.

        Mirrors the timeout + truncation policy in
        ``tools/_mcp_bridge.py`` so the two bridges report identical
        error shapes back to the agent.
        """
        from mcp.types import CallToolResult, TextContent  # noqa: PLC0415

        def result(text: str, *, is_error: bool) -> CallToolResult:
            if self._context_budget is not None and not is_error:
                text = self._context_budget.apply(text)
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                isError=is_error,
            )

        if tool_name not in known_names:
            return result(
                f"Error: unknown tool {tool_name!r}",
                is_error=True,
            )

        effective_timeout = (
            BUILD_TOOL_TIMEOUT_S
            if tool_name in _LONG_RUNNING_TOOL_NAMES
            else self._tool_timeout
        )

        t0 = time.monotonic()
        with self._terminal_dispatch_error_lock:
            dispatch_generation = self._terminal_dispatch_generation
            prior_terminal = self._terminal_dispatch_error
        if prior_terminal is not None:
            return result(
                _cap_error(
                    "Error: TerminalToolDispatchError: "
                    f"{type(prior_terminal).__name__}: {prior_terminal}"
                ),
                is_error=True,
            )
        if (
            self._invocation_deadline is not None
            and t0 >= self._invocation_deadline
        ):
            return result(
                "Error: logical invocation deadline is exhausted; "
                "tool dispatch was not started.",
                is_error=True,
            )
        assert self._pool is not None
        try:
            fut = self._pool.submit(
                self._toolset.dispatch, tool_name, arguments or {},
            )

            def publish_late_terminal(
                completed: concurrent.futures.Future[str],
            ) -> None:
                """Publish terminal dispatches even after the HTTP wait ends."""

                try:
                    dispatch_error = completed.exception()
                except concurrent.futures.CancelledError:
                    return
                if isinstance(dispatch_error, TerminalToolDispatchError):
                    self._record_terminal_dispatch_error(
                        dispatch_error,
                        generation=dispatch_generation,
                    )

            # A running future cannot be cancelled when the HTTP tool timeout
            # expires.  Its terminal mutation result may therefore arrive
            # after this coroutine has already returned a timeout envelope.
            # Future callbacks run on that exact worker completion and keep
            # the adapter-visible terminal signal authoritative.
            fut.add_done_callback(publish_late_terminal)
            tool_deadline = t0 + effective_timeout
            deadline = (
                tool_deadline
                if self._invocation_deadline is None
                else min(tool_deadline, self._invocation_deadline)
            )
            while not fut.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    cancelled_before_start = fut.cancel()
                    if not cancelled_before_start and fut.done():
                        # The dispatch settled on the deadline edge.  Consume
                        # its exact outcome instead of manufacturing timeout.
                        break
                    if (
                        tool_name in self._write_names
                        and not cancelled_before_start
                    ):
                        raise ToolMutationIndeterminate(
                            f"mutating tool {tool_name!r} exceeded its MCP "
                            "dispatch timeout while still running; source "
                            "settlement is indeterminate"
                        )
                    elapsed = time.monotonic() - t0
                    logger.error(
                        "CrucibleHttpMcpServer[%s] tool %s TIMED OUT "
                        "after %.1fs (limit %ss)",
                        self._name, tool_name, elapsed,
                        effective_timeout,
                    )
                    if (
                        self._invocation_deadline is not None
                        and self._invocation_deadline <= tool_deadline
                    ):
                        return result(
                            "Error: logical invocation deadline was exhausted "
                            f"while tool {tool_name!r} was running.",
                            is_error=True,
                        )
                    return result(
                        (
                            f"Error: tool {tool_name!r} timed out after "
                            f"{effective_timeout:.0f}s."
                        ),
                        is_error=True,
                    )
                await asyncio.sleep(min(0.25, remaining))
            result_text = fut.result(timeout=0)
        except TerminalToolDispatchError as exc:
            elapsed = time.monotonic() - t0
            self._record_terminal_dispatch_error(
                exc,
                generation=dispatch_generation,
            )
            logger.warning(
                "CrucibleHttpMcpServer[%s] tool %s TERMINAL DISPATCH "
                "after %.1fs: %s",
                self._name, tool_name, elapsed, exc,
            )
            return result(
                _cap_error(
                    "Error: TerminalToolDispatchError: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to agent
            elapsed = time.monotonic() - t0
            logger.warning(
                "CrucibleHttpMcpServer[%s] tool %s ERROR after %.1fs: %s",
                self._name, tool_name, elapsed, exc,
            )
            return result(
                _cap_error(f"Error: {type(exc).__name__}: {exc}"),
                is_error=True,
            )
        elapsed = time.monotonic() - t0
        logger.info(
            "CrucibleHttpMcpServer[%s] tool %s -> %d chars (%.1fs)",
            self._name, tool_name, len(result_text), elapsed,
        )
        return result(str(result_text), is_error=False)


__all__ = ["CrucibleHttpMcpServer"]
