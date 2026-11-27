# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Driver of the Kimi kind: the Moonshot route behind its locked worker.

The backend is built exactly as a workflow builds it, with ``resolve_for_role``
on the column's shipped routing file, and nothing of that file is overlaid.
The route's endpoint variable expands to the exact production endpoint, so
routing admission, the writer gate and the route-qualified feature profile all
apply. Everything from routing down to the socket runs: the project-locked
pin of the worker (captured when the adapter is built, checked again before
every launch), the probe of the installed runtime, the request with the
credential on the worker's standard input, the locked interpreter with the
vendor's agent SDK, its command-line runtime and the pinned ``openai`` client,
the worker's own check of the raw stream, the parent's reading of the worker
protocol, the per-attempt tool bridge with its bearer token and the mutation
host.

The seam is the launch interposer (:mod:`..peers.launch`) at ``subprocess`` as
the adapter module sees it. A launch of the locked worker interpreter runs
under the worker harness: the request launch in mode ``redirect``, so that
the client's request for the production origin is answered by the scripted
loopback provider, and the probe in mode ``guard``. Any other executable
passes through untouched, and so does loopback traffic inside the worker (the
tool bridge). Bypassed: the ``-m`` entry of the worker (the harness runs the
same module as ``__main__``), DNS, TLS and the provider.

A route that opts into the diagnostic cost upper bound is called as its
routing file says it must be: the guarded lens binds the cost policy that
file names, with a durable ledger, where every other route gets the plain
diagnostic limit policy. (The budget such a route asks of every invocation is
the runner's business: the invocation is not a driver's to write.) Such a
route prices a response by what its reply says of the provider's cache, so
the dialect reads the cache counts back from the bytes of every reply it
sends, for the provider ledger of the cell: a count a reply does not carry
is missing there, never zero.

The seam canary of a column fails unless the request came from the locked
worker behind the redirect: the harness journalled a redirected launch of the
adapter's own worker interpreter, in another process, isolated and without
bytecode, before the request arrived; the request has the documented path,
the user agent of the pinned command-line runtime, the version of the pinned
``openai`` client, the canary credential as a bearer value and the production
host; and it asks for a stream with its usage.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from crucible.backends import kimi as adapter
from crucible.backends.config import load_routing, resolve_for_role
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY
from crucible.backends.limits import INVOCATION_LIMIT_LEDGER_LEAF, InvocationLimitLedger

from .. import columns as matrix_columns
from .. import declarations
from .. import envelopes as envelope_files
from ..columns import CANARY_CREDENTIAL, COLUMN_PRODUCT, Column
from ..peers import launch
from ..provider import (
    CacheCounts,
    LoopbackProvider,
    Reply,
    SeenRequest,
    SentCall,
    answered_from,
    cut_after,
    require_cut_in_body,
)
from ..script import (
    Call,
    Cut,
    Drop,
    Empty,
    Malformed,
    ProviderScript,
    Reason,
    Refuse,
    Say,
    ServedAs,
    Stall,
    Step,
)
from ..toolsets import BINDINGS
from . import openai_compat as chat
from .base import (
    GUARD_POLICY,
    Cancellation,
    Lens,
    MatrixSetupError,
    close_when,
    register,
)

# The two pinned distributions a request of the worker names: the client that
# sends it and the command-line runtime that names itself in the user agent.
_SDK = "openai"
_RUNTIME = "kimi-cli"
_WORKER_PROJECT = launch.REPOSITORY / "backends" / "kimi_worker"
# What the adapter starts for a request: ``-m`` this module, no argument.
_WORKER_MODULE = "crucible_kimi_worker"
# The adapter's launch seam: ``subprocess`` as its module sees it.
(_SEAM,) = (name for module, name in launch.SUBPROCESS_SEAMS if module == adapter.__name__)
# The adapter caches a passed probe of the worker for the life of the process.
_PROBE_CACHE = "_qualify_worker_runtime"
_CREATED = 1_700_000_000
# The command of the CI step, in the shape of the DeepSeek worker's. Copy
# mode is a convention here: the Kimi pin takes any regular file.
INSTALL = "uv sync --frozen --reinstall --link-mode copy --project backends/kimi_worker"
# What the model is scripted to have thought before it answers or calls a
# tool. The provider's turn is replayed with it; it never reaches a result.
REASONING_TEXT = "matrix reasoning before the reply [canary-reasoning-8e2b]"
_TOOL_CALL = "tool_call"
_DONE = "[DONE]"


@functools.cache
def pinned_version(distribution: str) -> str:
    """The version of a distribution the worker's lock pins: read, never typed."""

    lock = tomllib.loads((_WORKER_PROJECT / "uv.lock").read_text(encoding="utf-8"))
    versions = [
        str(package.get("version"))
        for package in lock.get("package", ())
        if package.get("name") == distribution
    ]
    if len(versions) != 1:
        raise MatrixSetupError(
            f"the Kimi worker lock pins {distribution!r} {len(versions)} times"
        )
    return versions[0]


def production_endpoint() -> str:
    """The endpoint the shipped routes of the kind name, as routing expands it."""

    return matrix_columns.SCRIPTED_ENV[matrix_columns.ENDPOINT_ENV[matrix_columns.KIMI_KIND]]


def _text(content: Any) -> str:
    """The text of one chat message's content: a string, or text parts."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _shape(message: Mapping[str, Any]) -> dict[str, Any]:
    """What one chat message is to the conversation: role, call ids, reasoning."""

    role = str(message.get("role", ""))
    calls = message.get("tool_calls")
    return {
        "role": role,
        # The calls an assistant turn asks for, in its order.
        "asks": [
            str(call.get("id", ""))
            for call in (calls if isinstance(calls, list) else ())
            if isinstance(call, dict)
        ],
        # The call a tool message answers.
        "answers": str(message.get("tool_call_id", "")) if role == "tool" else None,
        # Whether the turn carries reasoning content (never its text).
        "reasoning": bool(message.get("reasoning_content")),
    }


class MoonshotChatDialect:
    """The chat-completions wire of the Moonshot route: one stream per request."""

    name = "moonshot_chat"
    provider = "moonshot"
    # ``max_completion_tokens``, or ``max_tokens``, in the body of a request.
    response_cap_field = True

    def __init__(
        self,
        *,
        journal: launch.Journal | None = None,
        interpreter: str | None = None,
    ) -> None:
        # The journal of the cell's worker launches and the interpreter the
        # adapter starts; without them nothing can pass the seam.
        self._journal = journal
        self._interpreter = interpreter
        endpoint = urlsplit(production_endpoint())
        self._host = endpoint.netloc
        self.path = endpoint.path.rstrip("/") + "/chat/completions"

    @property
    def user_agent(self) -> str:
        """What the worker's pinned command-line runtime names itself."""

        return f"KimiCLI/{pinned_version(_RUNTIME)}"

    def envelope_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}/{self.provider}.{behaviour}"
            for behaviour in (
                "completion",
                _TOOL_CALL,
                chat.REASONING_ONLY,
                *chat.REFUSALS.values(),
                chat.MALFORMED,
                chat.OUTPUT_CUT,
                chat.EMPTY_COMPLETION,
            )
        )

    def parse(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        credential: str,
    ) -> SeenRequest:
        lowered = {name.lower(): value for name, value in headers.items()}
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        messages = [
            item for item in payload.get("messages") or () if isinstance(item, dict)
        ]
        last_model_turn = max(
            (index for index, item in enumerate(messages) if item.get("role") == "assistant"),
            default=-1,
        )
        response_format = payload.get("response_format")
        options = payload.get("stream_options")
        return SeenRequest(
            method=method,
            path=path,
            user_agent=lowered.get("user-agent", ""),
            credential_ok=lowered.get("authorization", "") == f"Bearer {credential}",
            model=str(payload.get("model", "")),
            system="\n".join(
                _text(item.get("content")) for item in messages if item.get("role") == "system"
            ),
            user="\n".join(
                _text(item.get("content")) for item in messages if item.get("role") == "user"
            ),
            tools=tuple(
                str(item.get("function", {}).get("name", ""))
                for item in payload.get("tools") or ()
                if isinstance(item, dict)
            ),
            # The history is replayed in full on every request: what answers
            # the last reply is what follows the last model turn.
            tool_results=tuple(
                (str(item.get("tool_call_id", "")), _text(item.get("content")))
                for item in messages[last_model_turn + 1 :]
                if item.get("role") == "tool"
            ),
            # The wire has no field for a turn count or a spend ceiling: the
            # step limit of the route is kept by the client.
            turn_control=None,
            budget_control=None,
            schema_field=(
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
            ),
            host=lowered.get("host", ""),
            evidence={
                "stream": payload.get("stream"),
                "include_usage": isinstance(options, dict) and options.get("include_usage"),
                # The version the sending client reports of itself.
                "sdk": lowered.get("x-stainless-package-version", ""),
                # The shape of the conversation: roles, call ids and whether a
                # turn carries reasoning; no text.
                "messages": [_shape(item) for item in messages],
                "launches": [] if self._journal is None else self._journal.launches(),
            },
            response_cap=chat.response_cap(payload),
        )

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        """Why this request cannot be the locked worker's own, behind the redirect."""

        problems: list[str] = []
        if seen.path != self.path:
            problems.append(f"request path {seen.path!r}, not {self.path!r}")
        if seen.user_agent != self.user_agent:
            problems.append(
                f"user agent {seen.user_agent!r}, not the pinned runtime's "
                f"{self.user_agent!r}"
            )
        if seen.evidence.get("sdk") != pinned_version(_SDK):
            problems.append(
                f"the client reports version {seen.evidence.get('sdk')!r} of itself, "
                f"not the pinned SDK's {pinned_version(_SDK)!r}"
            )
        if not seen.credential_ok:
            problems.append("the canary credential is not the bearer value of the request")
        if seen.host != self._host:
            problems.append(
                f"the client addressed {seen.host!r}, not the production host "
                f"{self._host!r}: the request did not come through the redirect"
            )
        if not (
            seen.evidence.get("stream") is True and seen.evidence.get("include_usage") is True
        ):
            problems.append("the request does not ask for a stream with its usage")
        redirected = [
            record
            for record in seen.evidence.get("launches") or ()
            if record.get("mode") == "redirect"
        ]
        if not redirected:
            problems.append(
                "the harness journalled no redirected worker launch before the "
                "request arrived"
            )
            return problems
        record = redirected[-1]
        if record.get("pid") == os.getpid():
            problems.append("the worker did not run in a process of its own")
        if self._interpreter is None or record.get("executable") != self._interpreter:
            problems.append(
                f"the redirected launch ran {record.get('executable')!r}, not the "
                f"adapter's locked worker interpreter {self._interpreter!r}"
            )
        if (record.get("form"), record.get("target"), record.get("arguments")) != (
            "-m",
            _WORKER_MODULE,
            [],
        ):
            problems.append("the redirected launch is not the request launch of the worker")
        if not (record.get("guard") and record.get("isolated") and record.get("no_bytecode")):
            problems.append(
                "the worker was not launched isolated, without bytecode and under the "
                "network guard"
            )
        return problems

    def replay_problems(
        self, requests: Sequence[SeenRequest], calls: Sequence[SentCall]
    ) -> list[str]:
        """The stateless-replay rule of the route, for the cells of the kind.

        The scripted provider answers on position and call ids alone, so it
        answers a conversation that the production route refuses. The provider
        applies this rule to the scripted exchange when it ends, and the
        scenarios with a ToolSet report what it names.
        """

        return replay_problems(requests, calls)

    def foreign_messages(self, requests: Sequence[SeenRequest]) -> list[str]:
        """What the requests of a call carry beside prompt, model turns and tool results.

        The rule of any chat-completions wire
        (:func:`.openai_compat.foreign_messages`).
        """

        return chat.foreign_messages(requests)

    def _dropped(
        self,
        step: Drop,
        envelope: envelope_files.Envelope,
        slots: dict[str, Any],
        served: str,
        call_id: str,
    ) -> Reply:
        """The stream of a plain completion, cut after its answer text.

        The chunks that arrive whole name the model, so the reply names it.
        The usage travels in the chunks that never arrive: none is billed,
        and the counters of those chunks are fillers.
        """

        slots.update(
            response_id=f"chatcmpl-{call_id}",
            created=_CREATED,
            served_model=served,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            reasoning_text=REASONING_TEXT,
            reasoning_tokens=0,
        )
        require_cut_in_body(self.name, step)
        chunks = envelope.fill(slots)
        frames = [server_sent_events([chunk]) for chunk in chunks]
        return Reply(
            envelope.status,
            chunks,
            None,
            served,
            None,
            envelope.name,
            raw=b"".join(frames),
            content_type="text/event-stream",
            cut=cut_after(frames, step.text),
        )

    def cache_counts(self, reply: Reply) -> CacheCounts:
        """The cache counts a reply states, for the provider ledger of a cell.

        The diagnostic routes of the kind price a response by whether its
        statistics carry a count of cache reads, so clause C6 holds the
        counts a result records to what the reply said. Read from the bytes
        of the reply, whichever dialect rendered them.
        """

        return stated_cache_counts(reply.raw or b"")

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        served = seen.model
        if isinstance(step, ServedAs):
            served = seen.model + step.suffix
            step = step.step
        if isinstance(step, Stall):
            # The request is taken and never answered: not a byte travels,
            # nothing is billed and no model is named. No envelope stands
            # behind it, because there is no reply.
            return Reply(200, None, held=True, marker=step.marker)
        if isinstance(step, Malformed):
            # A stream that ends without its terminal: the reasoning and the
            # answer arrive, then the body ends in order, before the finish
            # chunk, the statistics and the closing message. The usage
            # travels in the chunks that never come, so none is billed.
            envelope = envelope_files.load(f"{self.name}/{self.provider}.{chat.MALFORMED}")
            chunks = envelope.fill(
                {
                    "response_id": f"chatcmpl-{call_id}",
                    "created": _CREATED,
                    "served_model": served,
                    "reasoning_text": REASONING_TEXT,
                    "text": step.text,
                }
            )
            return Reply(
                envelope.status,
                chunks,
                None,
                served,
                None,
                envelope.name,
                raw=server_sent_events(chunks),
                content_type="text/event-stream",
            )
        if isinstance(step, (Say, Drop)):
            behaviour, call = "completion", None
            slots: dict[str, Any] = {"text": step.text}
        elif isinstance(step, Reason):
            behaviour, call, slots = chat.REASONING_ONLY, None, {}
        elif isinstance(step, Call):
            binding = BINDINGS[step.slot]
            arguments = dict(binding.arguments if step.args is None else step.args)
            if step.malformed is not None:
                arguments = {}
            behaviour, call = _TOOL_CALL, (step.slot, call_id, arguments)
            slots = {
                "call_id": call_id,
                "tool_name": binding.name,
                # The wire carries the arguments as a JSON document in a
                # string, so it also carries a text that is none. The reply
                # is read by the vendor's SDK in the worker; what reaches
                # the adapter of such a call is what that SDK makes of it.
                "tool_arguments": (
                    json.dumps(arguments, sort_keys=True)
                    if step.malformed is None
                    else step.malformed
                ),
            }
        elif isinstance(step, Refuse):
            behaviour, call, slots = chat.REFUSALS[step.form], None, {}
        elif isinstance(step, Cut):
            # The stream of an answer that ends at the cap: the reasoning and
            # the text that was produced, then the cap as the finish reason.
            behaviour, call, slots = chat.OUTPUT_CUT, None, {"text": step.text}
        elif isinstance(step, Empty):
            # The stream of an answer in which nothing was produced.
            behaviour, call, slots = chat.EMPTY_COMPLETION, None, {}
        else:
            raise MatrixSetupError(
                f"the {self.name} dialect does not render {type(step).__name__} yet; "
                "the scenario that needs it adds the envelope with its liar"
            )
        envelope = envelope_files.load(f"{self.name}/{self.provider}.{behaviour}")
        if envelope.status != 200:
            # A request the service rejects is answered with its error object
            # alone, no stream: nothing is billed and no model is named.
            return Reply(envelope.status, envelope.fill(slots), envelope=envelope.name)
        if isinstance(step, Drop):
            return self._dropped(step, envelope, slots, served, call_id)
        usage = step.usage
        slots.update(
            response_id=f"chatcmpl-{call_id}",
            created=_CREATED,
            served_model=served,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        if not isinstance(step, (Refuse, Empty)):
            # A reply that answers or calls a tool reasons first; reasoning
            # is part of the completion tokens the provider bills.
            # Where the reply shows no text, every one of them is reasoning.
            # A reply in which nothing was produced did not reason either.
            slots.update(
                reasoning_text=REASONING_TEXT,
                reasoning_tokens=(
                    usage.output_tokens
                    if isinstance(step, Reason)
                    else usage.output_tokens // 2
                ),
            )
        chunks = envelope.fill(slots)
        return Reply(
            envelope.status,
            chunks,
            usage,
            served,
            call,
            envelope.name,
            raw=server_sent_events(chunks),
            content_type="text/event-stream",
            call_malformed=isinstance(step, Call) and step.malformed is not None,
        )


def server_sent_events(chunks: Sequence[Any]) -> bytes:
    """The chunks of one reply as they travel; the closing message is no JSON."""

    return b"".join(
        (
            "data: "
            + (chunk if isinstance(chunk, str) else json.dumps(chunk, separators=(",", ":")))
            + "\n\n"
        ).encode()
        for chunk in chunks
    )


_EVENT = b"data: "
# Where a usage block states its counts of cache reads and of cache writes.
_READS, _WRITES, _DETAILS = "cached_tokens", "cache_write_tokens", "prompt_tokens_details"


def _count(holder: Mapping[str, Any], key: str) -> int | None:
    """One cache count as a usage block states it; ``None`` where the key is absent."""

    if key not in holder:
        return None
    if type(holder[key]) is not int:
        raise MatrixSetupError(
            f"a reply states {key!r} as {type(holder[key]).__name__}, which the "
            "provider ledger cannot record as a count; the scenario that needs "
            "such a reply adds the reading with its liar"
        )
    return int(holder[key])


def stated_cache_counts(stream: bytes) -> CacheCounts:
    """What the usage blocks of one streamed reply say of the provider's cache.

    Read from the events as they travel. A reply states its usage in the
    statistics chunk and again inside the finish choice. Each block states
    the count of cache writes under the prompt details and, only for a
    request that read from the cache, the count of cache reads there and as
    an equal alias beside the totals. A count the reply does not carry is
    ``None``, never zero. A reply whose blocks disagree states no count the
    ledger could record, and is no reply of this wire.
    """

    stated: set[CacheCounts] = set()
    for event in stream.split(b"\n\n"):
        if not event.startswith(_EVENT):
            continue
        try:
            chunk = json.loads(event[len(_EVENT) :])
        except json.JSONDecodeError:
            continue  # the closing message of the stream
        if not isinstance(chunk, dict):
            continue
        for holder in (chunk, *(chunk.get("choices") or ())):
            usage = holder.get("usage") if isinstance(holder, dict) else None
            if not isinstance(usage, dict):
                continue
            details = usage.get(_DETAILS)
            details = details if isinstance(details, dict) else {}
            reads = {
                count
                for count in (_count(details, _READS), _count(usage, _READS))
                if count is not None
            }
            if len(reads) > 1:
                raise MatrixSetupError(
                    f"a usage block of a reply states two counts of cache reads: {sorted(reads)}"
                )
            stated.add(CacheCounts(next(iter(reads), None), _count(details, _WRITES)))
    if len(stated) > 1:
        raise MatrixSetupError(
            "the usage blocks of one reply state different cache counts: "
            f"{sorted(map(repr, stated))}"
        )
    return next(iter(stated), CacheCounts(None, None))


def reasons_before_a_call(envelope: str) -> bool:
    """Whether a reply of ``envelope`` carries reasoning content with its tool call.

    Read from the envelope: what a client must send back with the turn before
    it answers the call.
    """

    return any(
        "reasoning_content" in choice.get("delta", {})
        for chunk in envelope_files.load(envelope).body
        if isinstance(chunk, dict)
        for choice in chunk.get("choices") or ()
    )


def replay_problems(
    requests: Sequence[SeenRequest], calls: Sequence[SentCall]
) -> list[str]:
    """Why the requests of one call do not replay the conversation in full.

    The wire keeps no conversation, as any chat-completions wire: every
    request after the first begins with the opening messages and carries each
    earlier reply that asked for a tool, in order, as the assistant turn that
    names the call directly followed by the tool message that answers it (the
    rule of :func:`.openai_compat.replay_problems`). On this route the
    assistant turn must also come back whole: a reply that reasoned before it
    called a tool is replayed with its reasoning content.

    A request that matched no scripted step is not judged: it is an extra
    request, which clause C0 reports.
    """

    problems = list(chat.replay_problems(requests, calls))
    if not reasons_before_a_call(
        f"{MoonshotChatDialect.name}/{MoonshotChatDialect.provider}.{_TOOL_CALL}"
    ):
        return problems
    for index, request in enumerate(requests):
        if not request.matched:
            continue
        bare = [
            message["asks"]
            for message in request.evidence.get("messages") or ()
            if message["role"] == "assistant" and message["asks"] and not message["reasoning"]
        ]
        if bare:
            problems.append(
                f"request {index + 1} carries {', '.join(chat.conversation(request))}: "
                f"the assistant turn(s) that asked for {bare} are replayed without "
                "their reasoning content"
            )
    return problems


@dataclass
class _Bound:
    backend: Any
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    journal: launch.Journal
    # The bearer token of the tool bridge is made per attempt inside the
    # adapter and crosses to the worker on standard input only: the driver
    # cannot name it, so clause C9 looks for the canary credential alone.
    secrets: tuple[str, ...] = ()
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "mcp_bridge"
    # What the driver left for the last moment before the call: on a play
    # without the credential of the route, taking its variable away.
    before_call: Callable[[], Any] | None = None

    def cancel_when(self, marker: str) -> Cancellation:
        return close_when(self.provider, self.backend, marker)

    def children(self) -> tuple[int, ...]:
        return self.journal.pids()  # every worker process of the cell


class KimiDriver:
    """The shipped Kimi routes, their locked worker answered on loopback."""

    kind = matrix_columns.KIMI_KIND
    dialect: str = MoonshotChatDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    bypassed_layers: tuple[str, ...] = (
        "the -m entry of the worker (the worker harness runs the same module as "
        "__main__, in the locked interpreter with the launch's own options)",
        "DNS and TLS (the origin redirect inside the worker answers the "
        "production origin on loopback)",
        "the provider (replies are scripted envelopes)",
    )
    # The launch seam, patched through the launch interposer for as long as a
    # backend is open; and the adapter's cache of a passed probe, cleared once
    # per run so that the probe of the installed runtime runs under the harness.
    private_seams: tuple[str, ...] = (
        f"{adapter.__name__}.{_SEAM}",
        f"{adapter.__name__}.{_PROBE_CACHE}",
    )
    # A locked worker interpreter is started for every provider request, and
    # its pin is hashed before each start: the request of a plain call is on
    # the wire about five seconds after the call on an idle machine. Between
    # four and five times that, for a busy one.
    startup_seconds: float = 23.0

    @property
    def inexpressible(self) -> Mapping[str, str]:
        """Script steps this kind's wire cannot carry: the inventory's rows only."""

        from .. import cells  # noqa: PLC0415 - cells imports the drivers

        return {
            f"{row['scenario']}/{row['step']}": row["reason"]
            for row in cells.load_inventory().not_expressible
            if row["kind"] == self.kind
        }

    @property
    def envelopes(self) -> tuple[str, ...]:
        return MoonshotChatDialect().envelope_names()

    def answers(self, script: ProviderScript) -> tuple[str, ...]:
        return answered_from(MoonshotChatDialect(), script)

    @property
    def interpreter(self) -> Path:
        """Where the worker environment puts its interpreter."""

        return _WORKER_PROJECT / ".venv" / "bin" / "python"

    def capture(self, journal: launch.Journal) -> Any:
        """The runtime identity of the installed worker, probed under the guard."""

        interposer = launch.Interposer(lambda: journal)
        # A probe this process passed earlier is not asked again; ask it now,
        # under the harness of this journal.
        getattr(adapter, _PROBE_CACHE).cache_clear()
        try:
            with launch.at_subprocess(adapter, interposer, _SEAM):
                identity = adapter.capture_project_locked_kimi_runtime_identity()
        except (adapter.KimiSDKUnavailable, adapter.KimiSDKQualificationError) as exc:
            raise MatrixSetupError(
                f"kind {self.kind}: the locked worker environment cannot be pinned "
                f"({exc}); install it with `{INSTALL}`"
            ) from exc
        if Path(identity.logical_worker_python) != self.interpreter or not (
            launch.is_locked_interpreter(identity.logical_worker_python)
        ):
            raise MatrixSetupError(
                f"kind {self.kind}: the adapter starts {identity.logical_worker_python!r}, "
                "which is not the interpreter of its worker environment: the launch "
                "interposer would not guard it"
            )
        violations = journal.violations()
        if violations or not journal.launches():
            raise MatrixSetupError(
                f"kind {self.kind}: the probe of the worker did not run cleanly under "
                f"the harness: {'; '.join(violations) or 'no launch was journalled'}"
            )
        return identity

    def _require_interpreter(self) -> None:
        """A worker environment that is not installed is an error that says so."""

        if not self.interpreter.is_file():
            raise MatrixSetupError(
                f"kind {self.kind} needs the locked worker environment, and its "
                f"interpreter ({'/'.join(self.interpreter.parts[-5:])}) does not exist; "
                f"install it with `{INSTALL}`"
            )

    def preflight(self) -> None:
        if sys.platform != "linux":
            raise MatrixSetupError(f"kind {self.kind} runs on Linux only")
        self._require_interpreter()
        with tempfile.TemporaryDirectory(prefix="matrix-preflight-") as directory:
            self.capture(launch.Journal(Path(directory)))

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        return {}  # the shipped route is used as it is

    def _dialect(self, journal: launch.Journal) -> MoonshotChatDialect:
        return MoonshotChatDialect(journal=journal, interpreter=str(self.interpreter))

    def _redirect(self, provider: LoopbackProvider) -> launch.Redirect | None:
        """What the request launch of the worker is redirected to."""

        endpoint = urlsplit(production_endpoint())
        return launch.Redirect(
            module=_WORKER_MODULE,
            origin=f"{endpoint.scheme}://{endpoint.netloc}",
            target=provider.url,
        )

    def _interposer(
        self, journal: launch.Journal, provider: LoopbackProvider
    ) -> launch.Interposer:
        """What decides every launch of the adapter while a backend is open."""

        return launch.Interposer(lambda: journal, redirect=self._redirect(provider))

    def limits(self, column: Column, role: str, lens: Lens, tmp_path: Path) -> dict[str, Any]:
        """What routing is told about limits: the lens, as the route asks for it.

        A route that opts into the diagnostic cost upper bound binds a limit
        guard only with the cost policy its routing file names and a durable
        ledger; the policy in turn requires a budget on every invocation.
        """

        if lens != "guarded":
            return {}
        if not declarations.route_extra(column, role, matrix_columns.COST_BOUND_EXTRA):
            return {"limit_policy": GUARD_POLICY}
        ledger = tmp_path / "limit-ledger"
        ledger.mkdir(exist_ok=True)
        return {
            "limit_policy": dataclasses.replace(
                GUARD_POLICY,
                require_budget_usd=True,
                diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
            ),
            "limit_ledger": InvocationLimitLedger(ledger / INVOCATION_LIMIT_LEDGER_LEAF),
        }

    @contextlib.contextmanager
    def open(
        self,
        column: Column,
        *,
        capability: str,
        lens: Lens,
        script: ProviderScript,
        tmp_path: Path,
        without_credential: bool = False,
    ) -> Iterator[_Bound]:
        if column.kind != self.kind:
            raise MatrixSetupError(f"column {column.id} is not of kind {self.kind}")
        # A proof opens a column without asking the preflight of the session.
        self._require_interpreter()
        log_dir, cwd = tmp_path / "backend-logs", tmp_path / "backend-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        journal = launch.Journal(tmp_path / "worker-harness")
        journal.directory.mkdir(exist_ok=True)
        role = matrix_columns.ROLE_BY_CAPABILITY[capability]
        with contextlib.ExitStack() as stack:
            provider = stack.enter_context(
                LoopbackProvider(
                    script, credential=CANARY_CREDENTIAL, dialect=self._dialect(journal)
                )
            )
            # The adapter reads its endpoint and credential from the
            # environment at each invocation, and routing expands the same
            # two variables when it loads the file.
            stack.enter_context(matrix_columns.scripted_environment())
            interposer = self._interposer(journal, provider)
            stack.enter_context(launch.at_subprocess(adapter, interposer, _SEAM))
            limits = self.limits(column, role, lens, tmp_path)
            # The boundary of the matrix: exactly what a workflow calls.
            backend = resolve_for_role(
                role,
                routing=load_routing(config_path=column.routing_path(COLUMN_PRODUCT)),
                capability=capability,
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(cwd),
                log_dir=log_dir,
                **limits,
            )
            before_call = None
            if without_credential:
                # The route names its key as a variable. Routing reads it
                # whenever it loads the file, and the adapter reads it again
                # at each call: that is the last moment, so the variable is
                # taken away there and not before. The scripted environment
                # puts it back when the backend is closed.
                variable = declarations.credential_variable(
                    column.routing_path(COLUMN_PRODUCT), role
                )
                if variable is None or variable not in os.environ:
                    raise MatrixSetupError(
                        f"column {column.id}: the route of role {role!r} names no "
                        "variable for its key that is set while the backend is "
                        "built: there is nothing to take away at the call"
                    )
                before_call = functools.partial(os.environ.pop, variable, None)
            try:
                yield _Bound(
                    backend,
                    provider,
                    # The journal holds the command line of every worker launch
                    # and the limit ledger what the guard wrote of each call:
                    # clause C9 reads both with the adapter's own records.
                    log_dirs=(log_dir, journal.directory, tmp_path / "limit-ledger"),
                    journal=journal,
                    before_call=before_call,
                )
            finally:
                with contextlib.suppress(Exception):
                    backend.close()  # the runner has observed close() already
                violations = journal.violations()
                if violations:
                    # A worker swallows what it raises and its adapter types the
                    # failure; the journal decides.
                    raise MatrixSetupError(
                        f"column {column.id}: a worker of the cell reached beyond the "
                        f"scripted provider: {'; '.join(violations)}"
                    )


register(KimiDriver())
