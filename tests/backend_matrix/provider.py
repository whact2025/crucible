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

A request that is accepted and never answered (``Stall``) is held: the
provider reads it, records it and keeps the connection open without a byte of
reply, until the cell is over or the test lets it go. No clock of the
provider ends it. On a wire whose requests come from a scripted tool the
reply tells the tool to stall instead, and the tool keeps its session open
and says nothing more. Either way the provider notes the moment under the
marker the step names, for a cell that acts on it from another thread.

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
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .drivers.base import MatrixSetupError
from .script import (
    Call,
    Cut,
    Drop,
    Empty,
    Loop,
    Malformed,
    ProviderScript,
    Reason,
    Refuse,
    Say,
    ServedAs,
    Stall,
    Step,
    TokenUsage,
)
from .toolsets import BINDINGS

UNSCRIPTED_STATUS = 409
UNSCRIPTED_CODE = "matrix_unscripted_request"
POLICY_ERROR_CODE = "content_policy_violation"
PROBE_TEXT = "matrix probe answer"
# The moment the provider notes when it has taken the first request of a
# cell. No marker of a script step: a step names its own.
FIRST_REQUEST_MARKER = "matrix:first-request"


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
    # The cap on the tokens of the response that the request carries, where
    # the dialect reads one; ``None`` for a request without it.
    response_cap: int | None = None
    # Whether the request carried the scripted credential as its bearer, read
    # by the provider itself from the header as it arrived, whatever the
    # dialect accepts as a credential (``credential_ok``): a dialect of a
    # route without a key of its own accepts any bearer.
    scripted_credential: bool = False


@dataclass(frozen=True)
class SentCall:
    """A tool call the provider sent: slot, id, arguments, and the request it answered."""

    slot: str
    call_id: str
    request_index: int
    arguments: Mapping[str, Any]
    # Whether the argument text of the call was no JSON object: such a call
    # names no arguments, so nothing a ToolSet was handed can be its dispatch.
    malformed: bool = False


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
    # What the dialect of a wire that carries the conversation says of the
    # messages of the scripted requests that are neither the workflow's
    # prompt, a turn of the model nor a tool result
    # (:class:`ConversationDialect`); empty where the dialect has no such rule.
    foreign_messages: list[str] = field(default_factory=list)
    # Whether the wire of the exchange has a field for a cap on the tokens of
    # a response (:class:`ResponseCapDialect`). Where it has none, a request
    # without a cap says nothing about the adapter.
    response_cap_field: bool = False
    # Whether a tool beside the adapter keeps the conversations of the wire
    # (:class:`KeeperDialect`), and every handle that tool was asked to
    # continue and refused because it holds no conversation under it, in the
    # order asked. On a wire without such a tool nobody but the adapter can
    # be asked.
    conversation_keeper: bool = False
    refused_handles: list[str] = field(default_factory=list)

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
    # Where the connection goes away: the headers announce the whole reply,
    # this many bytes of it are sent, and the socket is closed. ``None`` for
    # a reply that travels whole.
    cut: int | None = None
    # Whether the argument text of ``call`` is no JSON object (its mapping is
    # then empty).
    call_malformed: bool = False
    # The request is accepted and never answered: the provider keeps the
    # connection open and sends nothing, until it is released.
    held: bool = False
    # The moment this reply stands for, noted when the provider has taken
    # the request: what a cell waits for before it acts from another thread.
    marker: str | None = None


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


@runtime_checkable
class ResponseCapDialect(Protocol):
    """A dialect of a wire that has a field for a cap on the tokens of a response.

    Such a dialect reads the field into :attr:`SeenRequest.response_cap`, and
    the ledger keeps that the wire has one
    (:attr:`ProviderLedger.response_cap_field`). On such a wire a request
    without the cap is an adapter that kept the cap of its route to itself;
    on a wire without the field every request is without it. The scenario of
    the output-token cut asks for the cap on the first kind of wire only. A
    dialect may read a cap that shows somewhere without stating that its
    wire has the field: a cap it reads is then held against a route that
    declares none, and none is asked of a route that declares one.
    """

    response_cap_field: bool


def stated_cap(value: Any) -> int | None:
    """A cap on the tokens of a response as a request states it; ``None`` for none."""

    return value if type(value) is int else None


@runtime_checkable
class KeeperDialect(Protocol):
    """A dialect of a wire on which a tool beside the adapter keeps the conversations.

    On such a wire the adapter cannot tell by itself whether a handle is
    held: it asks the tool, and the peer of the cell stands where the tool
    is. The peer holds no conversation of an earlier call, so a handle that
    a cell names is one it refuses. What the tool says then is no reply of a
    model, and nothing of it was observed. It is rendered here, from an
    envelope with its provenance and its grade, and not written into the
    peer: the peer asks the provider side at ``resume_path`` and relays the
    answer. The exchange is no model request, so clause C0 does not count
    it. The ledger keeps the handle
    (:attr:`ProviderLedger.refused_handles`) and the envelope, so a cell
    that rests on the refusal shows that the tool was asked, and for which
    handle, and carries the grade of what it was answered with.
    """

    resume_path: str

    def refuse_resume(self, body: bytes) -> tuple[str, Reply]:
        """The handle the tool was asked to continue, and what it answers."""


@runtime_checkable
class ConversationDialect(Protocol):
    """A dialect of a wire on which every request carries its conversation.

    Such a dialect can say what each message of a request is. A request of a
    call holds the workflow's prompt (the opening), the turns of the model
    and the results of the tools it called; a message of any other kind is
    the adapter's own word to the model, which the workflow never sees. The
    provider asks when the scripted exchange ends and the ledger keeps the
    answer (:attr:`ProviderLedger.foreign_messages`), for the scenario that
    is about it. A wire that keeps the conversation itself has no such rule.
    """

    def foreign_messages(self, requests: Sequence[SeenRequest]) -> list[str]:
        """The messages of the matched requests that are none of the three kinds."""


def require_decoded_arguments(dialect: str, step: Call) -> None:
    """A wire that is a session of a vendor's tool carries no malformed call.

    What such a dialect scripts is what the tool tells its client, and there
    a tool call is a request to a tool server with a decoded object for its
    arguments. A dialect that scripts the provider's own reply, with the
    arguments of a call in a string, renders the text instead.
    """

    if step.malformed is not None:
        raise MatrixSetupError(
            f"the {dialect} dialect renders no tool call whose argument text is no "
            "JSON object: the wire it scripts is a session of the vendor's tool, "
            "on which a call reaches the adapter as a decoded object, so the cell "
            "is pinned not expressible in the inventory"
        )


def require_cut_in_body(dialect: str, step: Drop) -> None:
    """A dialect cuts a reply inside its body, after part of the answer."""

    if step.after != "partial_body":
        raise MatrixSetupError(
            f"the {dialect} dialect drops a reply inside its body only; the "
            "scenario that drops one earlier adds the rendering with its liar"
        )


def cut_inside(payload: bytes, text: str) -> int:
    """Where a reply that carries ``text`` is cut: in the middle of that text.

    The offset of the middle of the first place the text stands in the bytes
    of the reply, so that what travels ends inside the answer.
    """

    words = text.encode("utf-8")
    if len(words) < 2 or words not in payload:
        raise MatrixSetupError(
            "a reply can be cut inside its answer text only where it carries "
            f"that text, of two bytes or more: {text!r}"
        )
    return payload.index(words) + len(words) // 2


def cut_after(frames: Sequence[bytes], text: str) -> int:
    """Where a stream that carries ``text`` is cut: inside the frame after it.

    The frames up to the first one that carries the text travel whole, so the
    client holds the text; the connection goes away in the middle of the
    frame that follows, before anything that ends the turn.
    """

    whole, partial = whole_and_partial(frames, text)
    return sum(len(frame) for frame in frames[:whole]) + len(partial)


def whole_and_partial(frames: Sequence[bytes], text: str) -> tuple[int, bytes]:
    """What travels of a stream that is cut after its answer text.

    How many frames travel whole, the first that carries ``text`` being the
    last of them, and the leading half of the frame that follows it: what
    the other side had written of that frame when it went away. The one
    statement of where a stream of frames is cut, for a wire whose frames
    are parts of an HTTP body and for one whose frames are lines of a tool's
    output.
    """

    words = text.encode("utf-8")
    carrying = next((index for index, frame in enumerate(frames) if words in frame), None)
    if not words or carrying is None or carrying + 1 >= len(frames):
        raise MatrixSetupError(
            "a stream can be cut after its answer text only where a frame "
            f"carries that text and another follows it: {text!r}"
        )
    following = frames[carrying + 1]
    return carrying + 1, following[: len(following) // 2]


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


def refused_from(dialect: KeeperDialect) -> tuple[str, ...]:
    """The envelopes ``dialect`` renders the refusal of a handle from, in order.

    Read without an exchange: the refusal is rendered for a request that
    names no handle, and these are the names the provider keeps in its
    ledger when the peer of a cell asks for a handle.
    """

    _handle, refusal = dialect.refuse_resume(b"{}")
    return refusal.envelopes


class ToyDialect:
    """The toy driver's own wire: one JSON request and reply per model turn."""

    name = "toy"
    path = "/toy/v1/turn"
    user_agent = "crucible-matrix-toy/1"
    turn_control_field = True  # ``max_turns`` in the body of a request
    response_cap_field = True  # ``max_output_tokens`` in the body of a request

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
            # The calls a request names as answered by earlier requests, and
            # what the adapter says to the model in words of its own.
            evidence={
                name: [str(item) for item in payload[name]]
                for name in ("answered", "notes")
                if isinstance(payload.get(name), list)
            },
            response_cap=stated_cap(payload.get("max_output_tokens")),
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

    def foreign_messages(self, requests: Sequence[SeenRequest]) -> list[str]:
        """The notes of the adapter's own that the matched requests carry.

        The toy wire carries the workflow's prompt in two fields, the results
        of the last calls and the names of the earlier ones. Anything the
        adapter says to the model beside them travels as a note.
        """

        return [
            f"request {index + 1} carries {len(notes)} note(s) of the adapter's own"
            for index, request in enumerate(requests)
            for notes in (request.evidence.get("notes") or (),)
            if request.matched and notes
        ]

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        served = seen.model
        if isinstance(step, ServedAs):
            served = seen.model + step.suffix
            step = step.step
        if isinstance(step, Stall):
            # The request is taken and never answered: nothing travels,
            # nothing is billed and no model is named.
            return Reply(200, None, held=True, marker=step.marker)
        if isinstance(step, Malformed):
            # A 200 whose document holds no turn: no type, no finish and no
            # usage. Nothing is billed for it.
            return Reply(200, {"model": served}, served_model=served)
        if isinstance(step, Drop):
            require_cut_in_body(self.name, step)
            # The document of a plain answer, of which the wire carries the
            # part that ends inside the text. No document arrives, so nothing
            # is billed and no model is named.
            whole = {"model": served, "type": "message", "text": step.text, "finish": "stop"}
            payload = json.dumps(whole).encode("utf-8")
            return Reply(200, whole, raw=payload, cut=cut_inside(payload, step.text))
        usage = {
            "input_tokens": getattr(step, "usage", TokenUsage(0, 0)).input_tokens,
            "output_tokens": getattr(step, "usage", TokenUsage(0, 0)).output_tokens,
        }
        if isinstance(step, Cut):
            # The document of an answer that the provider ended at the cap:
            # what was produced so far, and the mark of a cut as the finish.
            return Reply(
                200,
                {
                    "model": served,
                    "type": "message",
                    "text": step.text,
                    "finish": "length",
                    "usage": usage,
                },
                step.usage,
                served,
            )
        if isinstance(step, Empty):
            # A clean stop that produced nothing: an ordinary finish, the
            # empty text and no output token.
            return Reply(
                200,
                {
                    "model": served,
                    "type": "message",
                    "text": "",
                    "finish": "stop",
                    "usage": usage,
                },
                step.usage,
                served,
            )
        if isinstance(step, Reason):
            # A clean stop with reasoning and no visible text.
            return Reply(
                200,
                {
                    "model": served,
                    "type": "message",
                    "text": "",
                    "reasoning": "toy reasoning that is never shown",
                    "finish": "stop",
                    "usage": usage,
                },
                step.usage,
                served,
            )
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
            malformed = step.malformed is not None
            if malformed:
                arguments = {}
            return Reply(
                200,
                {
                    "model": served,
                    "type": "tool_calls",
                    "calls": [
                        {
                            "id": call_id,
                            "name": binding.name,
                            # The toy wire carries the arguments as an object;
                            # where the model wrote none, it carries the text
                            # the model wrote in its place.
                            "arguments": step.malformed if malformed else arguments,
                        }
                    ],
                    "usage": usage,
                },
                step.usage,
                served,
                (step.slot, call_id, arguments),
                call_malformed=malformed,
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
        # The moments of the cell that were reached, by marker, with the time
        # on the monotonic clock; and whether held requests were let go.
        self._marks: dict[str, threading.Event] = {}
        self._marked_at: dict[str, float] = {}
        self._released = threading.Event()
        if isinstance(self._dialect, CacheCountingDialect):
            self._ledger.cache_counts = []
        if isinstance(self._dialect, TurnControlDialect):
            self._ledger.turn_control_field = self._dialect.turn_control_field
        if isinstance(self._dialect, ResponseCapDialect):
            self._ledger.response_cap_field = self._dialect.response_cap_field
        if isinstance(self._dialect, KeeperDialect):
            self._ledger.conversation_keeper = True
        # A step the dialect cannot render, or a reply whose cache counts it
        # cannot read back, is a setup error here, in the test's thread, not a
        # dropped connection inside the server's.
        blank = self._dialect.parse("POST", self._dialect.path, {}, b"{}", credential="")
        for step in (*script.steps, *((script.tail,) if script.tail else ())):
            rendered = self._dialect.render(
                step.step if isinstance(step, Loop) else step, blank, ""
            )
            self._stated(rendered)
            self._require_cut_inside(rendered)
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
                if reply.held:
                    # Accepted and never answered: the connection stays open
                    # without a byte until the provider lets it go, then it
                    # is closed. The other side sees no reply at all.
                    provider._released.wait()
                    self.close_connection = True
                    return
                payload = (
                    json.dumps(reply.body).encode("utf-8") if reply.raw is None else reply.raw
                )
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                if reply.cut is not None:
                    # The connection goes away: the headers announced the
                    # whole reply, a leading part is sent, the socket closes.
                    self.wfile.write(payload[: reply.cut])
                    self.wfile.flush()
                    self.close_connection = True
                elif self.command != "HEAD":
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

    @staticmethod
    def _require_cut_inside(reply: Reply) -> None:
        """A reply that is cut has bytes of its own, and loses at least one of them."""

        if reply.cut is None:
            return
        whole = len(json.dumps(reply.body).encode("utf-8") if reply.raw is None else reply.raw)
        if not 0 < reply.cut < whole:
            raise MatrixSetupError(
                f"a reply of {whole} byte(s) cannot be cut after {reply.cut}: the "
                "connection goes away inside the reply, after its first byte and "
                "before its last"
            )

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
                # The loop is satisfied; what follows it decides, and a loop
                # that follows counts its own requests.
                self._position += 1
                self._loop_served = 0
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
            bearer = next(
                (value for name, value in headers.items() if name.lower() == "authorization"),
                "",
            )
            seen = replace(seen, scripted_credential=bearer == f"Bearer {self._credential}")
            unscripted = Reply(
                UNSCRIPTED_STATUS,
                {"error": {"code": UNSCRIPTED_CODE, "message": "no scripted reply"}},
            )
            if self._closed_for_cell:
                self._ledger.late_requests += 1
                return unscripted
            if (
                isinstance(self._dialect, KeeperDialect)
                and method == "POST"
                and path == self._dialect.resume_path
            ):
                # The tool that keeps the conversations is asked to continue
                # one it does not hold. No model is asked: the exchange is
                # kept apart from the requests of the script.
                handle, refusal = self._dialect.refuse_resume(body)
                self._ledger.refused_handles.append(handle)
                self._ledger.envelopes.extend(refusal.envelopes)
                return refusal
            if self._probing:
                self._ledger.probe_requests += 1
                return self._dialect.render(Say(PROBE_TEXT), seen, "probe")
            # The first request of the cell, scripted or not: a moment a
            # cell that plays a deadline holds against that deadline.
            self._mark(FIRST_REQUEST_MARKER)
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
            if reply.marker is not None:
                self._mark(reply.marker)
            if reply.envelope is not None:
                self._ledger.envelopes.append(reply.envelope)
            self._ledger.envelopes.extend(reply.envelopes)
            if reply.call is None:
                self._pending_call_ids = ()
            else:
                slot, call_id, arguments = reply.call
                self._pending_call_ids = (call_id,)
                self._ledger.sent_calls.append(
                    SentCall(
                        slot,
                        call_id,
                        len(self._ledger.requests) - 1,
                        arguments,
                        malformed=reply.call_malformed,
                    )
                )
            return reply

    def _unconsumed(self) -> int:
        count = 0
        for index, step in enumerate(self._script.steps):
            if index < self._position:
                continue
            # A loop that was never reached has served nothing: it is
            # satisfied only where it requires nothing.
            served = self._loop_served if index == self._position else 0
            if isinstance(step, Loop) and served >= step.at_least:
                continue
            count += 1
        return count

    # -- the moments of a cell --------------------------------------------

    def _mark(self, marker: str) -> None:
        self._marked_at.setdefault(marker, time.monotonic())
        self._marks.setdefault(marker, threading.Event()).set()

    def reached(self, marker: str) -> threading.Event:
        """The event that is set once the cell has reached ``marker``."""

        with self._lock:
            return self._marks.setdefault(marker, threading.Event())

    def reached_at(self, marker: str) -> float | None:
        """When ``marker`` was reached, on the monotonic clock; ``None`` if never."""

        with self._lock:
            return self._marked_at.get(marker)

    def reach(self, marker: str) -> None:
        """Note a moment that is not the provider's own to see.

        The runner calls it where the moment a cell acts on is no request:
        the entry of a tool handler it holds.
        """

        with self._lock:
            self._mark(marker)

    def release(self) -> None:
        """Let every held request go: its connection is closed without a reply."""

        self._released.set()

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
                if isinstance(self._dialect, ConversationDialect):
                    self._ledger.foreign_messages = list(
                        self._dialect.foreign_messages(self._ledger.requests)
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
                    foreign_messages=list(self._ledger.foreign_messages),
                    refused_handles=list(self._ledger.refused_handles),
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
        self.release()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)

    def __enter__(self) -> LoopbackProvider:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
