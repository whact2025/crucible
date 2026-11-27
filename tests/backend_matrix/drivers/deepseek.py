# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Driver of the DeepSeek kind: the Responses route behind its locked worker.

The backend is built exactly as a workflow builds it, with ``resolve_for_role``
on the column's shipped routing file. The one overlay is
``extra.expected_worker_runtime_fingerprint``, captured inside the cell with
the public capture function, as an operator binds a run-local copy of the
route. Everything from routing admission down to the socket runs: the pin
capture and its revalidation before every launch, the request encoding with
the credential on the worker's standard input, the Linux supervisor, the
locked interpreter with the pinned ``openai`` SDK and its Responses stream
client, the worker's validation and normalisation, the parent's parser of the
worker protocol, the parent's tool loop.

The seam is the launch interposer (:mod:`..peers.launch`) at the adapter's
supervisor function, the one private name this driver patches. A launch of the
locked worker interpreter runs under the worker harness: the request launch in
mode ``redirect``, so that the SDK's request for the production origin is
answered by the scripted loopback provider, and the probe in mode ``guard``.
Any other executable passes through untouched. Bypassed: the ``-m`` entry of
the worker (the harness runs the same module as ``__main__``), DNS, TLS and
the provider.

The seam canary of the column fails unless the request came from the locked
worker behind the redirect: the harness journalled a redirected launch of the
adapter's own worker interpreter, in another process, isolated and without
bytecode, before the request arrived; the request has the documented path,
the pinned SDK's user agent, the canary credential as a bearer value and the
production host; and it asks for a stream.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from crucible.backends import deepseek as adapter
from crucible.backends.config import load_routing, resolve_for_role

from .. import columns as matrix_columns
from .. import envelopes as envelope_files
from ..columns import CANARY_CREDENTIAL, COLUMN_PRODUCT, Column
from ..peers import launch
from ..provider import (
    LoopbackProvider,
    Reply,
    SeenRequest,
    SentCall,
    answered_from,
    cut_after,
    require_cut_in_body,
    stated_cap,
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
from .base import (
    GUARD_POLICY,
    Cancellation,
    Lens,
    MatrixSetupError,
    close_when,
    register,
)
from .openai_compat import (
    EMPTY_COMPLETION,
    MALFORMED,
    OUTPUT_CUT,
    REASONING_ONLY,
    REFUSALS,
)

_SDK = "openai"
_WORKER_PROJECT = launch.REPOSITORY / "backends" / "deepseek_worker"
# What the adapter starts for a request: ``-m`` this module, no argument.
_WORKER_MODULE = "crucible_deepseek_worker"
_OVERLAY_FIELD = "extra.expected_worker_runtime_fingerprint"
# The adapter's launch seam: its supervisor function, a module global.
(_SEAM,) = (name for module, name in launch.SUPERVISOR_SEAMS if module == adapter.__name__)
_CREATED = 1_700_000_000
INSTALL = (
    "uv sync --frozen --reinstall --link-mode copy --project backends/deepseek_worker"
)
# What the model is scripted to have thought before it calls a tool. The
# worker hands it back to the parent for the stateless replay only.
REASONING_TEXT = "matrix reasoning before the tool call [canary-reasoning-3d7a]"
# The items of a provider turn that a client must replay, by envelope.
_TOOL_CALL = "tool_call"


@functools.cache
def pinned_sdk_version() -> str:
    """The version of the SDK the worker's lock pins: read, never typed."""

    lock = tomllib.loads((_WORKER_PROJECT / "uv.lock").read_text(encoding="utf-8"))
    versions = [
        str(package.get("version"))
        for package in lock.get("package", ())
        if package.get("name") == _SDK
    ]
    if len(versions) != 1:
        raise MatrixSetupError(f"the DeepSeek worker lock pins {_SDK!r} {len(versions)} times")
    return versions[0]


def _kind(item: Mapping[str, Any]) -> str:
    """What an input item is: its type, or the role of a plain message."""

    return str(item.get("type") or item.get("role") or "")


class DeepSeekResponsesDialect:
    """The Responses wire of the DeepSeek route: one streamed response per request."""

    name = "deepseek_responses"
    path = "/responses"
    provider = "deepseek"
    response_cap_field = True  # ``max_output_tokens`` in the body of a request

    def __init__(
        self,
        *,
        journal: launch.Journal | None = None,
        interpreter: str | None = None,
    ) -> None:
        # The journal of the cell's worker launches and the interpreter the
        # adapter says it starts; without them nothing can pass the seam.
        self._journal = journal
        self._interpreter = interpreter
        self._host = urlsplit(matrix_columns.WORKER_ORIGIN[matrix_columns.DEEPSEEK_KIND]).netloc

    @property
    def user_agent(self) -> str:
        """What the worker's pinned SDK names itself."""

        return f"OpenAI/Python {pinned_sdk_version()}"

    def envelope_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}/{self.provider}.{behaviour}"
            for behaviour in (
                "completion",
                _TOOL_CALL,
                REASONING_ONLY,
                *REFUSALS.values(),
                MALFORMED,
                OUTPUT_CUT,
                EMPTY_COMPLETION,
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
        items = [item for item in payload.get("input") or () if isinstance(item, dict)]
        # The route is stateless: the history is replayed in full, and what
        # answers the last reply is what follows the last item of the model.
        last_model_item = max(
            (
                index
                for index, item in enumerate(items)
                if item.get("type") in ("reasoning", "message", "function_call")
            ),
            default=-1,
        )
        text_format = payload.get("text")
        schema = text_format.get("format") if isinstance(text_format, dict) else None
        return SeenRequest(
            method=method,
            path=path,
            user_agent=lowered.get("user-agent", ""),
            credential_ok=lowered.get("authorization", "") == f"Bearer {credential}",
            model=str(payload.get("model", "")),
            system="\n".join(
                str(item.get("content", "")) for item in items if _kind(item) == "system"
            ),
            user="\n".join(
                str(item.get("content", "")) for item in items if _kind(item) == "user"
            ),
            tools=tuple(
                str(item.get("name", ""))
                for item in payload.get("tools") or ()
                if isinstance(item, dict)
            ),
            tool_results=tuple(
                (str(item.get("call_id", "")), str(item.get("output", "")))
                for item in items[last_model_item + 1 :]
                if item.get("type") == "function_call_output"
            ),
            # The wire has no field for a turn count or a spend ceiling.
            turn_control=None,
            budget_control=None,
            schema_field=isinstance(schema, dict) and schema.get("type") == "json_schema",
            host=lowered.get("host", ""),
            evidence={
                "stream": payload.get("stream"),
                # The shape of the replayed history: kinds and call ids, no text.
                "input": [
                    {
                        "kind": _kind(item),
                        "call_id": item.get("call_id"),
                        "role": item.get("role"),
                    }
                    for item in items
                ],
                "launches": [] if self._journal is None else self._journal.launches(),
            },
            response_cap=stated_cap(payload.get("max_output_tokens")),
        )

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        """Why this request cannot be the locked worker's own, behind the redirect."""

        problems: list[str] = []
        if seen.path != self.path:
            problems.append(f"request path {seen.path!r}, not {self.path!r}")
        if seen.user_agent != self.user_agent:
            problems.append(
                f"user agent {seen.user_agent!r}, not the pinned SDK's {self.user_agent!r}"
            )
        if not seen.credential_ok:
            problems.append("the canary credential is not the bearer value of the request")
        if seen.host != self._host:
            problems.append(
                f"the client addressed {seen.host!r}, not the production host "
                f"{self._host!r}: the request did not come through the redirect"
            )
        if seen.evidence.get("stream") is not True:
            problems.append("the request does not ask for a stream")
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
        if record.get("pid") == os.getpid() or record.get("parent") == os.getpid():
            problems.append("the worker did not run in a supervised process of its own")
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
        """The stateless-replay rule of the route, for the cells of the column.

        The scripted provider answers on position and call ids alone, so it
        answers a replay that the production route may refuse. The provider
        applies this rule to the scripted exchange when it ends, and the
        scenarios with a ToolSet report what it names.
        """

        return replay_problems(requests, calls)

    def foreign_messages(self, requests: Sequence[SeenRequest]) -> list[str]:
        """What the requests of a call carry beside prompt, model turns and tool results."""

        return foreign_messages(requests)

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
            # A stream that ends without its terminal: every event of a plain
            # answer arrives but the terminal, then the body ends in order.
            # The usage is null until the terminal, so none is billed.
            envelope = envelope_files.load(f"{self.name}/{self.provider}.{MALFORMED}")
            events = envelope.fill(
                {
                    "response_id": f"resp-{call_id}",
                    "created": _CREATED,
                    "served_model": served,
                    "item_id": f"msg-{call_id}",
                    "text": step.text,
                }
            )
            return Reply(
                envelope.status,
                events,
                None,
                served,
                None,
                envelope.name,
                raw=server_sent_events(events),
                content_type="text/event-stream",
            )
        if isinstance(step, (Say, Drop)):
            behaviour, call = "completion", None
            slots: dict[str, Any] = {"text": step.text, "item_id": f"msg-{call_id}"}
        elif isinstance(step, Reason):
            behaviour, call = REASONING_ONLY, None
            slots = {
                "reasoning_item_id": f"rs-{call_id}",
                "reasoning_text": REASONING_TEXT,
                # Every output token of such a response is a reasoning token.
                "reasoning_tokens": step.usage.output_tokens,
            }
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
                # string, so it also carries a text that is none.
                "tool_arguments": (
                    json.dumps(arguments, sort_keys=True)
                    if step.malformed is None
                    else step.malformed
                ),
                "item_id": f"fc-{call_id}",
                "reasoning_item_id": f"rs-{call_id}",
                "reasoning_text": REASONING_TEXT,
                # Reasoning is part of the output tokens the provider bills.
                "reasoning_tokens": step.usage.output_tokens // 2,
            }
        elif isinstance(step, Refuse):
            behaviour, call = REFUSALS[step.form], None
            # Only the form that declines in an output message has an item.
            slots = {"item_id": f"msg-{call_id}"} if step.form == "refusal_part" else {}
        elif isinstance(step, Cut):
            # The stream of an answer that ends at the cap: the events of the
            # text that was produced, then the terminal of an incomplete
            # response with the cap as its reason.
            behaviour, call = OUTPUT_CUT, None
            slots = {"text": step.text, "item_id": f"msg-{call_id}"}
        elif isinstance(step, Empty):
            # The stream of a response in which nothing was produced: the
            # terminal of a completed response without an output item.
            behaviour, call, slots = EMPTY_COMPLETION, None, {}
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
            # The stream of a plain answer, cut after its text delta. The
            # events that arrive whole name the model; the usage is null
            # until the terminal, which never arrives, so none is billed and
            # the counters of that event are fillers.
            slots.update(
                response_id=f"resp-{call_id}",
                created=_CREATED,
                served_model=served,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
            require_cut_in_body(self.name, step)
            events = envelope.fill(slots)
            frames = [server_sent_events([event]) for event in events]
            return Reply(
                envelope.status,
                events,
                None,
                served,
                None,
                envelope.name,
                raw=b"".join(frames),
                content_type="text/event-stream",
                cut=cut_after(frames, step.text),
            )
        usage = step.usage
        slots.update(
            response_id=f"resp-{call_id}",
            created=_CREATED,
            served_model=served,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        events = envelope.fill(slots)
        return Reply(
            envelope.status,
            events,
            usage,
            served,
            call,
            envelope.name,
            raw=server_sent_events(events),
            content_type="text/event-stream",
            call_malformed=isinstance(step, Call) and step.malformed is not None,
        )


def server_sent_events(events: Sequence[Mapping[str, Any]]) -> bytes:
    """The events of one response as they travel: no closing message."""

    return b"".join(
        (
            f"event: {event['type']}\n"
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        ).encode()
        for event in events
    )


def replayed_turn(envelope: str) -> tuple[str, ...]:
    """The kinds of output item a response of ``envelope`` holds, in order.

    Read from the terminal event of the envelope: what a stateless client must
    replay before it answers the response.
    """

    terminal = envelope_files.load(envelope).body[-1]
    return tuple(str(item["type"]) for item in terminal["response"]["output"])


_PROMPT_KINDS = ("system", "user")
# The items of a turn of the model, and the item that answers a call of it.
_MODEL_KINDS = ("reasoning", "function_call")
_RESULT_KIND = "function_call_output"


def _is_of_the_model(item: Mapping[str, Any]) -> bool:
    kind = item.get("kind")
    return kind in _MODEL_KINDS or (kind == "message" and item.get("role") == "assistant")


def foreign_messages(requests: Sequence[SeenRequest]) -> list[str]:
    """The input items of a call that are neither its prompt, a model turn nor a tool result.

    Every request of the route carries the conversation as input items. It
    opens with the workflow's prompt: at most one system and one user
    message, the same in every request of the call. After the first item of
    the model come only items of the model (its reasoning, its function
    calls, a message of the assistant) and the outputs of the calls. Any
    other item there, or an opening that is not the first request's, is the
    client's own word to the model.

    A request that matched no scripted step is not judged: it is an extra
    request, which clause C0 reports.
    """

    problems: list[str] = []
    opening: list[str] | None = None
    for index, request in enumerate(requests):
        if not request.matched:
            continue
        items = [item for item in request.evidence.get("input") or () if isinstance(item, dict)]
        first_turn = next(
            (
                place
                for place, item in enumerate(items)
                if _is_of_the_model(item) or item.get("kind") == _RESULT_KIND
            ),
            len(items),
        )
        head = [str(item.get("kind")) for item in items[:first_turn]]
        why: list[str] = []
        if opening is None:
            opening = head
            if any(kind not in _PROMPT_KINDS or head.count(kind) > 1 for kind in head):
                why.append(
                    f"it opens with {head}: more than the one system and the one "
                    "user message of a workflow's prompt"
                )
        elif head != opening:
            why.append(f"it opens with {head}, the first request of the call with {opening}")
        other = [
            str(item.get("kind"))
            for item in items[first_turn:]
            if not _is_of_the_model(item) and item.get("kind") != _RESULT_KIND
        ]
        if other:
            why.append(
                f"{len(other)} item(s) of the kind(s) {sorted(set(other))} follow the "
                "first item of the model"
            )
        if why:
            kinds = [str(item.get("kind")) for item in items]
            problems.append(f"request {index + 1} carries {kinds}: " + "; ".join(why))
    return problems


def replay_problems(
    requests: Sequence[SeenRequest], calls: Sequence[SentCall]
) -> list[str]:
    """Why the requests of one call do not replay the provider's turns in full.

    The route is stateless. Every request after the first must carry, after
    the system and user messages, each earlier response of the call that asked
    for a tool: its items in order (the reasoning item, then the function
    call with its call id) followed by the output of that call.

    ``requests`` are all requests of the exchange in order, as the provider
    ledger keeps them; a tool call names the request it answered by its index
    there. A request that matched no scripted step is not judged: it is an
    extra request, which clause C0 reports.
    """

    turn = replayed_turn(
        f"{DeepSeekResponsesDialect.name}/{DeepSeekResponsesDialect.provider}.{_TOOL_CALL}"
    )
    problems: list[str] = []
    for index, request in enumerate(requests):
        if not request.matched:
            continue
        answered = [call for call in calls if call.request_index < index]
        expected: list[tuple[str, str | None]] = [("system", None), ("user", None)]
        for call in answered:
            expected.extend(
                (kind, call.call_id if kind == "function_call" else None) for kind in turn
            )
            expected.append(("function_call_output", call.call_id))
        seen = [
            (str(item.get("kind")), item.get("call_id"))
            for item in request.evidence.get("input") or ()
        ]
        if seen != expected:
            problems.append(
                f"request {index + 1} carries {seen}; a stateless replay of "
                f"{len(answered)} tool round(s) is {expected}"
            )
    return problems


@dataclass
class _Bound:
    backend: Any
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    journal: launch.Journal
    # The runtime identity captured in the cell, which the route was built
    # with; ``None`` where the routing file was read as shipped.
    identity: Any
    # The adapter holds no token beside the credential it reads at dispatch.
    secrets: tuple[str, ...] = ()
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "function_call"

    def cancel_when(self, marker: str) -> Cancellation:
        return close_when(self.provider, self.backend, marker)

    def children(self) -> tuple[int, ...]:
        return self.journal.pids()  # every worker process of the cell


class DeepSeekDriver:
    """The shipped DeepSeek route, its locked worker answered on loopback."""

    kind = matrix_columns.DEEPSEEK_KIND
    dialect: str = DeepSeekResponsesDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    bypassed_layers: tuple[str, ...] = (
        "the -m entry of the worker (the worker harness runs the same module as "
        "__main__, in the locked interpreter with the launch's own options)",
        "DNS and TLS (the origin redirect inside the worker answers the "
        "production origin on loopback)",
        "the provider (replies are scripted envelopes)",
    )
    # Patched through the launch interposer for as long as a backend is open.
    private_seams: tuple[str, ...] = (f"{adapter.__name__}.{_SEAM}",)
    # A locked worker interpreter is started for every provider request, and
    # its pin is revalidated before each start: the request of a plain call
    # is on the wire about five seconds after the call on an idle machine.
    # Between four and five times that, for a busy one.
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
        return DeepSeekResponsesDialect().envelope_names()

    def answers(self, script: ProviderScript) -> tuple[str, ...]:
        return answered_from(DeepSeekResponsesDialect(), script)

    @property
    def interpreter(self) -> Path:
        """Where the worker environment puts its interpreter."""

        return _WORKER_PROJECT / ".venv" / "bin" / "python"

    def _capture(self, journal: launch.Journal) -> Any:
        """The runtime identity of the installed worker, probed under the guard."""

        interposer = launch.Interposer(lambda: journal)
        try:
            with launch.at_supervisor(adapter, interposer, _SEAM):
                identity = adapter.capture_project_locked_deepseek_runtime_identity()
        except (adapter.DeepSeekSDKUnavailable, adapter.DeepSeekSDKQualificationError) as exc:
            raise MatrixSetupError(
                f"kind {self.kind}: the locked worker environment cannot be pinned "
                f"({exc}); install it with `{INSTALL}`, in a clone inside the Linux "
                "filesystem"
            ) from exc
        if not launch.is_locked_interpreter(identity.logical_worker_python):
            raise MatrixSetupError(
                f"kind {self.kind}: the adapter starts {identity.logical_worker_python!r}, "
                "which is outside every worker environment: the launch interposer "
                "would not guard it"
            )
        violations = journal.violations()
        if violations or not journal.launches():
            raise MatrixSetupError(
                f"kind {self.kind}: the probe of the worker did not run cleanly under "
                f"the harness: {'; '.join(violations) or 'no launch was journalled'}"
            )
        return identity

    def preflight(self) -> None:
        if sys.platform != "linux":
            raise MatrixSetupError(f"kind {self.kind} runs on Linux only")
        if not self.interpreter.is_file():
            raise MatrixSetupError(
                f"kind {self.kind} needs the locked worker environment, and its "
                f"interpreter ({'/'.join(self.interpreter.parts[-5:])}) does not exist; "
                f"install it with `{INSTALL}`"
            )
        with tempfile.TemporaryDirectory(prefix="matrix-preflight-") as directory:
            self._capture(launch.Journal(Path(directory)))

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        if column.shipped:
            # The file as shipped: nothing is bound to its routes, so routing
            # is asked what it does with a route no run has bound yet.
            return {}
        return {_OVERLAY_FIELD: "<captured in the cell from the installed worker>"}

    def _dialect(self, journal: launch.Journal, interpreter: str) -> DeepSeekResponsesDialect:
        return DeepSeekResponsesDialect(journal=journal, interpreter=interpreter)

    def _redirect(self, provider: LoopbackProvider) -> launch.Redirect | None:
        """What the request launch of the worker is redirected to."""

        return launch.Redirect(
            module=_WORKER_MODULE,
            origin=matrix_columns.WORKER_ORIGIN[self.kind],
            target=provider.url,
        )

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
        log_dir, cwd = tmp_path / "backend-logs", tmp_path / "backend-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        journal = launch.Journal(tmp_path / "worker-harness")
        journal.directory.mkdir()
        with contextlib.ExitStack() as stack:
            # The fingerprint the route is bound to is captured here, in the
            # cell. On the file as shipped nothing is bound and nothing is
            # captured: a worker that routing starts there all the same runs
            # under the harness below, as every other one does.
            fields = self.overlay(column, {})
            identity = self._capture(journal) if fields else None
            interpreter = self.interpreter if identity is None else identity.logical_worker_python
            provider = stack.enter_context(
                LoopbackProvider(
                    script,
                    credential=CANARY_CREDENTIAL,
                    dialect=self._dialect(journal, str(interpreter)),
                )
            )
            interposer = launch.Interposer(lambda: journal, redirect=self._redirect(provider))
            stack.enter_context(launch.at_supervisor(adapter, interposer, _SEAM))
            # The route names no key: the adapter reads the variable of the
            # kind at each call. Without its credential the route is built as
            # ever and that variable is absent for as long as the backend is
            # open.
            stack.enter_context(
                matrix_columns.adapter_credential(self.kind, absent=without_credential)
            )
            overlay = (
                {}
                if identity is None
                else {name: identity.runtime_fingerprint for name in fields}
            )
            # The boundary of the matrix: exactly what a workflow calls.
            backend = resolve_for_role(
                matrix_columns.ROLE_BY_CAPABILITY[capability],
                routing=matrix_columns.overlaid_routing(
                    load_routing(config_path=column.routing_path(COLUMN_PRODUCT)),
                    overlay,
                    kind=self.kind,
                    backend=column.backend,
                ),
                capability=capability,
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(cwd),
                log_dir=log_dir,
                limit_policy=GUARD_POLICY if lens == "guarded" else None,
            )
            try:
                yield _Bound(
                    backend,
                    provider,
                    # The journal holds the command line of every worker launch:
                    # clause C9 reads it with the adapter's own records.
                    log_dirs=(log_dir, journal.directory),
                    journal=journal,
                    identity=identity,
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


register(DeepSeekDriver())
