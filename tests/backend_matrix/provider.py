# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``LoopbackProvider``: the scripted model side, behind a real loopback socket.

The provider plays a :class:`~.script.ProviderScript` over HTTP on
``127.0.0.1:0`` and keeps the provider ledger: every request it saw, what it
billed for every response, which steps were never asked for. Replay is strict
in order and count and loose in content: a request matches the next step on
method, path, its position in the script and, when it answers tool calls, the
tool-call ids. An unmatched request, of any HTTP method, gets an error status
and is counted as extra. What a request *contains* is recorded for the
scenario that is about it, never used to pick a reply.

A dialect turns a script step into one provider's reply and reads a request
back into the neutral :class:`SeenRequest`. This module holds the toy dialect,
the minimal one the toy driver speaks; a vendor dialect lives with its driver
and renders its replies from envelopes (:mod:`.envelopes`).

Because replay is loose in content, the provider answers a request that a
real stateless route may refuse: one that drops an item of an earlier round.
A dialect of such a wire states what each request must replay
(:class:`StatelessDialect`); the provider applies the rule when the scripted
exchange ends and the ledger keeps what it found, for the scenarios that
attach a ToolSet.

A dialect may name a served model that differs from the requested one only
for a ``ServedAs`` step. Anything else is a seam problem: a dialect that
answers under a fixed id would switch off the "served equals requested"
check for its whole kind.

Where a route prices a response by what the reply says of the provider's
cache, that statement is part of the bill. A dialect of such a wire reads the
cache counts back from every billed reply as it is sent
(:class:`CacheCountingDialect`), and the ledger keeps them beside the tokens:
a count the reply did not carry is recorded as missing, never as zero.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .drivers.base import MatrixSetupError
from .script import (
    Call,
    Loop,
    ProviderScript,
    Refuse,
    Say,
    ServedAs,
    Step,
    TokenUsage,
)
from .toolsets import BINDINGS

UNSCRIPTED_STATUS = 409
UNSCRIPTED_CODE = "matrix_unscripted_request"
POLICY_ERROR_CODE = "content_policy_violation"
PROBE_TEXT = "matrix probe answer"


@dataclass(frozen=True)
class SeenRequest:
    """One request as the provider saw it, in dialect-neutral terms."""

    method: str
    path: str
    user_agent: str
    credential_ok: bool
    model: str
    system: str
    user: str
    tools: tuple[str, ...]
    tool_results: tuple[tuple[str, str], ...]
    turn_control: int | None
    budget_control: float | None
    schema_field: bool
    matched: bool = False
    # The Host header as the provider received it: the address the client
    # believed it was talking to.
    host: str = ""
    # What the dialect knows about how the request came to be sent: the
    # process that sent it and the exchange that preceded it. Read by the
    # dialect's own seam check and by the proof of a column's canary; never
    # used to pick a reply.
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentCall:
    """A tool call the provider sent: slot, id, arguments, and the request it answered."""

    slot: str
    call_id: str
    request_index: int
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class CacheCounts:
    """What one billed reply stated of the provider's cache, as it travelled.

    ``None`` is a count the reply did not carry. It is never zero: a route
    whose results are cost intervals prices a response with an unknown count
    of cache reads as a range and one with a known count exactly, so the two
    are different bills.
    """

    read_tokens: int | None
    write_tokens: int | None


@dataclass
class ProviderLedger:
    """What the provider side saw and billed during the scripted exchange."""

    requests: list[SeenRequest] = field(default_factory=list)
    billed: list[TokenUsage] = field(default_factory=list)
    # The cache counts the reply of every billed response stated, in the order
    # of ``billed``: read back from the reply as it was sent, by a dialect
    # that can (:class:`CacheCountingDialect`). ``None`` where the dialect of
    # the exchange reads none back; clause C6 then refuses to judge a cost
    # interval, because it could not tell a count the provider sent from one
    # nobody sent.
    cache_counts: list[CacheCounts] | None = None
    served_models: list[str] = field(default_factory=list)
    sent_calls: list[SentCall] = field(default_factory=list)
    # The envelope each matched reply was rendered from (vendor dialects).
    envelopes: list[str] = field(default_factory=list)
    extra: int = 0
    unconsumed: int = 0
    # Whether the optional tail of the script was asked for and answered.
    tail_served: bool = False
    probe_requests: int = 0
    late_requests: int = 0
    seam_problems: list[str] = field(default_factory=list)
    # What the replay rule of a stateless dialect says of the scripted
    # requests (:class:`StatelessDialect`); empty where the dialect has none.
    replay_problems: list[str] = field(default_factory=list)
    # Whether the wire of the exchange has a field for the turn count of a
    # call (:class:`TurnControlDialect`). Where it has none, a request
    # without a turn control says nothing about the adapter.
    turn_control_field: bool = False

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def billed_total(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=sum(item.input_tokens for item in self.billed),
            output_tokens=sum(item.output_tokens for item in self.billed),
        )


@dataclass(frozen=True)
class Reply:
    """A rendered reply: status, JSON body, and what the provider bills."""

    status: int
    # The JSON document of the reply; for a stream, the events it holds.
    body: Any
    usage: TokenUsage | None = None
    served_model: str | None = None
    call: tuple[str, str, Mapping[str, Any]] | None = None  # slot, call id, arguments
    envelope: str | None = None  # the envelope the body was rendered from
    # Every envelope of a body that holds several frames, in the order sent.
    envelopes: tuple[str, ...] = ()
    # The bytes to send where the wire is not one JSON document (a stream of
    # server-sent events), with their content type. ``body`` then holds what
    # the bytes were rendered from.
    raw: bytes | None = None
    content_type: str = "application/json"


class Dialect(Protocol):
    """One provider's wire: how a request reads and how a step is answered."""

    name: str
    path: str

    def parse(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        credential: str,
    ) -> SeenRequest: ...

    def seam_problems(self, seen: SeenRequest) -> list[str]: ...

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply: ...


@runtime_checkable
class StatelessDialect(Protocol):
    """A dialect of a wire that keeps no conversation: the client replays it.

    What a later request of a call must carry of the earlier rounds is the
    wire's own rule, so the dialect states it. The provider applies the rule
    to the scripted exchange when it ends and the ledger keeps what it found;
    the scenarios with a ToolSet report it. Replies are still picked on
    position and call ids alone.
    """

    def replay_problems(
        self, requests: Sequence[SeenRequest], calls: Sequence[SentCall]
    ) -> list[str]:
        """Why the matched requests do not replay the earlier rounds in full.

        ``requests`` are all requests of the scripted exchange, in order;
        ``calls`` name the request each tool call answered by its index there.
        """


@runtime_checkable
class CacheCountingDialect(Protocol):
    """A dialect that says which cache counts a reply of its wire carried.

    The dialect reads them from the reply as it is sent, never from the step
    it rendered or from an envelope: a reply that a test dialect changes on
    its way out is recorded as it travelled. The provider asks for every
    billed reply and the ledger keeps the answers
    (:attr:`ProviderLedger.cache_counts`).
    """

    def cache_counts(self, reply: Reply) -> CacheCounts:
        """The counts of cache reads and writes ``reply`` states; ``None`` for none."""


@runtime_checkable
class TurnControlDialect(Protocol):
    """A dialect of a wire that has a field for the turn count of a call.

    Such a dialect reads the field into :attr:`SeenRequest.turn_control`, and
    the ledger keeps that the wire has one
    (:attr:`ProviderLedger.turn_control_field`). On such a wire a request
    without the count is an adapter that kept the count to itself; on a wire
    without the field every request is without it. The scenario of the
    suite's turn count asks for the count on the first kind of wire only.
    """

    turn_control_field: bool


def answered_from(dialect: Dialect, script: ProviderScript) -> tuple[str, ...]:
    """The envelopes ``dialect`` renders the steps of ``script`` from, in order.

    Read without an exchange: each step is rendered against a blank request,
    as the provider renders it when it is built. What a dialect adds only for
    a later state of a session (a terminal that the request of a turn
    selects) is not in it, so this names envelopes a play of the script uses
    and may not name them all.
    """

    blank = dialect.parse("POST", dialect.path, {}, b"{}", credential="")
    names: list[str] = []
    for step in (*script.steps, *((script.tail,) if script.tail else ())):
        reply = dialect.render(step.step if isinstance(step, Loop) else step, blank, "")
        if reply.envelope is not None:
            names.append(reply.envelope)
        names.extend(reply.envelopes)
    return tuple(names)


class ToyDialect:
    """The toy driver's own wire: one JSON request and reply per model turn."""

    name = "toy"
    path = "/toy/v1/turn"
    user_agent = "crucible-matrix-toy/1"
    turn_control_field = True  # ``max_turns`` in the body of a request

    def parse(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        credential: str,
    ) -> SeenRequest:
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return SeenRequest(
            method=method,
            path=path,
            user_agent=headers.get("User-Agent", ""),
            credential_ok=headers.get("Authorization", "") == f"Bearer {credential}",
            model=str(payload.get("model", "")),
            system=str(payload.get("system", "")),
            user=str(payload.get("user", "")),
            tools=tuple(str(name) for name in payload.get("tools", ())),
            tool_results=tuple(
                (str(item.get("id", "")), str(item.get("text", "")))
                for item in payload.get("tool_results", ())
                if isinstance(item, dict)
            ),
            turn_control=payload.get("max_turns"),
            budget_control=payload.get("budget_usd"),
            schema_field="response_schema" in payload,
            host=headers.get("Host", ""),
            # The calls a request names as answered by earlier requests.
            evidence=(
                {"answered": [str(item) for item in payload["answered"]]}
                if isinstance(payload.get("answered"), list)
                else {}
            ),
        )

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        """Why this request cannot have come from the toy adapter's client."""

        problems: list[str] = []
        if seen.path != self.path:
            problems.append(f"request path {seen.path!r}, not {self.path!r}")
        if seen.user_agent != self.user_agent:
            problems.append(f"user agent {seen.user_agent!r}, not {self.user_agent!r}")
        if not seen.credential_ok:
            problems.append("the canary credential is not in the Authorization header")
        return problems

    def replay_problems(
        self, requests: Sequence[SeenRequest], calls: Sequence[SentCall]
    ) -> list[str]:
        """Why a request does not name every call that earlier requests answered.

        The toy wire keeps no conversation. A request carries the results of
        the calls of the last reply, and names, in order, the calls whose
        results earlier requests of the call carried.
        """

        problems: list[str] = []
        for index, request in enumerate(requests):
            if not request.matched:
                continue  # an unscripted request is clause C0's business
            now = {call_id for call_id, _text in request.tool_results}
            expected = [
                call.call_id
                for call in calls
                if call.request_index < index and call.call_id not in now
            ]
            named = list(request.evidence.get("answered", ()))
            if named != expected:
                problems.append(
                    f"request {index + 1} names {named} as answered by earlier "
                    f"requests; {expected} were"
                )
        return problems

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        served = seen.model
        if isinstance(step, ServedAs):
            served = seen.model + step.suffix
            step = step.step
        usage = {
            "input_tokens": getattr(step, "usage", TokenUsage(0, 0)).input_tokens,
            "output_tokens": getattr(step, "usage", TokenUsage(0, 0)).output_tokens,
        }
        if isinstance(step, Say):
            return Reply(
                200,
                {
                    "model": served,
                    "type": "message",
                    "text": step.text,
                    "finish": "stop",
                    "usage": usage,
                },
                step.usage,
                served,
            )
        if isinstance(step, Call):
            binding = BINDINGS[step.slot]
            arguments = dict(binding.arguments if step.args is None else step.args)
            return Reply(
                200,
                {
                    "model": served,
                    "type": "tool_calls",
                    "calls": [
                        {"id": call_id, "name": binding.name, "arguments": arguments}
                    ],
                    "usage": usage,
                },
                step.usage,
                served,
                (step.slot, call_id, arguments),
            )
        if isinstance(step, Refuse):
            if step.form == "content_filter":
                body = {
                    "model": served,
                    "type": "message",
                    "text": "",
                    "finish": "content_filter",
                    "usage": usage,
                }
                return Reply(200, body, step.usage, served)
            if step.form == "refusal_part":
                body = {
                    "model": served,
                    "type": "refusal",
                    "refusal": "the request was declined",
                    "usage": usage,
                }
                return Reply(200, body, step.usage, served)
            return Reply(
                400,
                {
                    "error": {
                        "code": POLICY_ERROR_CODE,
                        "message": "the request violates the usage policy",
                    }
                },
            )
        raise MatrixSetupError(
            f"the {self.name} dialect does not render {type(step).__name__} yet; "
            "the scenario that needs it adds the rendering with its liar"
        )


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self) -> None:
        # HTTPServer.server_bind resolves the host name; the matrix resolves nothing.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class LoopbackProvider:
    """Play one script on ``127.0.0.1`` and keep the ledger of the exchange."""

    def __init__(
        self,
        script: ProviderScript,
        *,
        credential: str,
        dialect: Dialect | None = None,
    ) -> None:
        self._script = script
        self._dialect: Dialect = dialect or ToyDialect()
        self._credential = credential
        self._lock = threading.Lock()
        self._ledger = ProviderLedger()
        self._position = 0
        self._loop_served = 0
        self._tail_served = False
        self._pending_call_ids: tuple[str, ...] = ()
        self._calls = 0
        self._probing = False
        self._closed_for_cell = False
        self._served_seam_reported = False
        self._frozen: ProviderLedger | None = None
        if isinstance(self._dialect, CacheCountingDialect):
            self._ledger.cache_counts = []
        if isinstance(self._dialect, TurnControlDialect):
            self._ledger.turn_control_field = self._dialect.turn_control_field
        # A step the dialect cannot render, or a reply whose cache counts it
        # cannot read back, is a setup error here, in the test's thread, not a
        # dropped connection inside the server's.
        blank = self._dialect.parse("POST", self._dialect.path, {}, b"{}", credential="")
        for step in (*script.steps, *((script.tail,) if script.tail else ())):
            self._stated(
                self._dialect.render(step.step if isinstance(step, Loop) else step, blank, "")
            )
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:  # silence the server log
                return None

            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(length) if length else b""
                reply = provider._handle(
                    self.command, self.path, dict(self.headers), body
                )
                payload = (
                    json.dumps(reply.body).encode("utf-8") if reply.raw is None else reply.raw
                )
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

            # Every method reaches the ledger: a request that is not the scripted
            # POST (a models probe, a health check) is an extra request, not a
            # 501 the ledger never sees.
            do_POST = do_GET = do_HEAD = do_PUT = do_PATCH = do_DELETE = _serve  # noqa: N815
            do_OPTIONS = _serve  # noqa: N815

        self._server = _Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="matrix-loopback-provider",
            daemon=True,
        )
        self._thread.start()

    # -- addresses -------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # -- replay ----------------------------------------------------------

    def _stated(self, reply: Reply) -> CacheCounts | None:
        """What a billed reply says of the cache, where the dialect reads it back."""

        if reply.usage is None or not isinstance(self._dialect, CacheCountingDialect):
            return None
        return self._dialect.cache_counts(reply)

    def _next_step(self, seen: SeenRequest) -> Step | None:
        """Return the step this request consumes, or ``None`` when unscripted."""

        if seen.method != "POST" or seen.path != self._dialect.path:
            return None
        if tuple(call_id for call_id, _text in seen.tool_results) != (
            self._pending_call_ids
        ):
            return None
        steps = self._script.steps
        if self._position < len(steps):
            step = steps[self._position]
            if not isinstance(step, Loop):
                self._position += 1
                return step
            if self._loop_served < step.limit and seen.tools:
                self._loop_served += 1
                return step.step
            if self._loop_served >= step.at_least:
                # The loop is satisfied; what follows it decides.
                self._position += 1
                return self._next_step(seen)
            return None
        if self._script.tail is not None and not self._tail_served and not seen.tools:
            self._tail_served = self._ledger.tail_served = True
            return self._script.tail
        return None

    def _handle(
        self, method: str, path: str, headers: Mapping[str, str], body: bytes
    ) -> Reply:
        with self._lock:
            seen = self._dialect.parse(
                method, path, headers, body, credential=self._credential
            )
            unscripted = Reply(
                UNSCRIPTED_STATUS,
                {"error": {"code": UNSCRIPTED_CODE, "message": "no scripted reply"}},
            )
            if self._closed_for_cell:
                self._ledger.late_requests += 1
                return unscripted
            if self._probing:
                self._ledger.probe_requests += 1
                return self._dialect.render(Say(PROBE_TEXT), seen, "probe")
            step = self._next_step(seen)
            if step is None:
                self._ledger.requests.append(seen)
                self._ledger.extra += 1
                return unscripted
            self._ledger.requests.append(replace(seen, matched=True))
            if not self._ledger.seam_problems:
                self._ledger.seam_problems.extend(self._dialect.seam_problems(seen))
            self._calls += 1
            reply = self._dialect.render(step, seen, f"call-{self._calls}")
            if (
                reply.served_model is not None
                and reply.served_model != seen.model
                and not isinstance(step, ServedAs)
                and not self._served_seam_reported
            ):
                self._served_seam_reported = True
                self._ledger.seam_problems.append(
                    f"the {self._dialect.name} dialect answered under model "
                    f"{reply.served_model!r} a request for {seen.model!r} although "
                    "the script has no ServedAs step there"
                )
            if reply.usage is not None:
                self._ledger.billed.append(reply.usage)
            stated = self._stated(reply)
            if stated is not None and self._ledger.cache_counts is not None:
                # One for every billed response: read from the reply that is
                # about to be sent, whatever rendered it.
                self._ledger.cache_counts.append(stated)
            if reply.served_model is not None:
                self._ledger.served_models.append(reply.served_model)
            if reply.envelope is not None:
                self._ledger.envelopes.append(reply.envelope)
            self._ledger.envelopes.extend(reply.envelopes)
            if reply.call is None:
                self._pending_call_ids = ()
            else:
                slot, call_id, arguments = reply.call
                self._pending_call_ids = (call_id,)
                self._ledger.sent_calls.append(
                    SentCall(slot, call_id, len(self._ledger.requests) - 1, arguments)
                )
            return reply

    def _unconsumed(self) -> int:
        count = 0
        for index, step in enumerate(self._script.steps):
            if index < self._position:
                continue
            satisfied = isinstance(step, Loop) and self._loop_served >= step.at_least
            if index == self._position and satisfied:
                continue
            count += 1
        return count

    # -- the cell's view -------------------------------------------------

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._ledger.request_count

    def freeze(self) -> ProviderLedger:
        """End the scripted exchange and return its ledger."""

        with self._lock:
            if self._frozen is None:
                self._ledger.unconsumed = self._unconsumed()
                if isinstance(self._dialect, StatelessDialect):
                    # The exchange is over: every request of it can be judged.
                    self._ledger.replay_problems = list(
                        self._dialect.replay_problems(
                            self._ledger.requests, self._ledger.sent_calls
                        )
                    )
                self._frozen = replace(
                    self._ledger,
                    requests=list(self._ledger.requests),
                    billed=list(self._ledger.billed),
                    cache_counts=(
                        None
                        if self._ledger.cache_counts is None
                        else list(self._ledger.cache_counts)
                    ),
                    served_models=list(self._ledger.served_models),
                    sent_calls=list(self._ledger.sent_calls),
                    envelopes=list(self._ledger.envelopes),
                    seam_problems=list(self._ledger.seam_problems),
                    replay_problems=list(self._ledger.replay_problems),
                )
            return self._frozen

    def begin_probe(self) -> None:
        """From now on every request is the guarded lens's second plain call."""

        self.freeze()
        with self._lock:
            self._probing = True

    def end_of_cell(self) -> None:
        """The backend is closed: any request from now on is a late request."""

        self.freeze()
        with self._lock:
            self._probing = False
            self._closed_for_cell = True

    def final_counts(self) -> tuple[int, int]:
        """``(probe requests, late requests)`` once the cell is over."""

        with self._lock:
            return self._ledger.probe_requests, self._ledger.late_requests

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> LoopbackProvider:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
