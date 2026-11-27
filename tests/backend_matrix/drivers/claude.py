# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Driver of the Claude kind: the reference columns of the session route.

The backend is built exactly as a workflow builds it, with ``resolve_for_role``
on the column's shipped routing file. The one overlay is ``extra.cli_path``,
which points the route at the scripted peer of the cell
(:mod:`..peers.claude_cli`). Everything from the adapter down to the process
boundary runs: the session route's checks, the sanitised session transport,
the pinned Agent SDK's process transport, control protocol and frame parser,
the adapter's stream handling, and the SDK-hosted tool server with the real
dispatch bridge.

The vendor tool and the provider are what the peer stands in for. The peer
takes every model turn from the scripted loopback provider, in the
``claude_cli`` dialect: one request per model turn, answered with the frames
the tool emits for that turn, each rendered from an envelope. So the provider
ledger, the replay rules and clause C0 are the same as on every other column.

The two session checks of the adapter are met and no product name is patched:
``-v`` is answered with a version the adapter has reviewed for the route's
model (read from the adapter's own table, the one private name this driver
touches) and ``auth status --json`` with the organisation the routing file
names.

The seam canary of a column fails unless the SDK started the peer: a child of
the test process with the stream-json command line, the SDK's entry-point
variables and none of the scripted route variables in its environment, both
probes before the session, and the ``initialize`` and ``set_model`` control
requests before the first prompt.

The peer's processes are network clients outside the test process. Their
launcher installs the network guard of the matrix before the peer runs, so
the rule of the package holds there too, and the journal they write names
the variables of their environment without the values.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from crucible.backends._claude_session_transport import (
    CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL,
)
from crucible.backends.config import load_routing, resolve_for_role

from .. import columns as matrix_columns
from .. import envelopes as envelope_files
from .. import netguard
from ..columns import CANARY_CREDENTIAL, COLUMN_PRODUCT, Column
from ..peers import claude_cli as peer_program
from ..provider import (
    LoopbackProvider,
    Reply,
    SeenRequest,
    answered_from,
    refused_from,
    require_cut_in_body,
    require_decoded_arguments,
    whole_and_partial,
)
from ..script import (
    Call,
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

_SDK = "claude-agent-sdk"
_SDK_MODULE = "claude_agent_sdk"
_ENTRYPOINT = "sdk-py"
# What the SDK's process transport puts into the tool's environment.
_SDK_VERSION_ENV = "CLAUDE_AGENT_SDK_VERSION"
_ENTRYPOINT_ENV = "CLAUDE_CODE_ENTRYPOINT"
_STREAM_FLAGS = ("--output-format", "--input-format")
_OVERLAY_FIELD = "extra.cli_path"
_PEER_FAILURES = ("unsupported", "crashed")
# The forms of refusal the frames of the tool can carry: a model turn that
# ends on the refusal stop reason, and a request the provider rejected. A
# message the tool relays has no refusal part.
REFUSALS = ("content_filter", "http_policy")
# The kernel refuses an interpreter line longer than this.
_MAX_INTERPRETER_LINE = 250

# What a session with a response schema shows as text, in the last model turn
# and in the result text of the terminal frame. The answer itself travels in
# the structured output of that frame and nowhere else, so that a cell can
# tell an adapter that reads it from one that reads either text.
SCHEMA_SESSION_TEXT = "The answer is in the structured output of the result."
# What the model is scripted to have thought in a turn that shows no text.
# It never reaches a result.
REASONING_TEXT = "matrix reasoning that is never shown [canary-reasoning-b47e]"

_LAUNCHER = """#!{interpreter} -IS
# Written by the backend matrix for one cell: the scripted Claude CLI peer.
import json
import sys


def load(path, name):
    with open(path, encoding="utf-8") as stream:
        code = compile(stream.read(), path, "exec")
    scope = {{"__name__": name, "__file__": path}}
    exec(code, scope)
    return scope


# The peer is a network client. The network guard of the matrix is installed
# before a line of it runs: in this process too, only loopback may be
# connected to or resolved.
load({guard!r}, "crucible_matrix_netguard")["install"]()
peer = load({source!r}, "crucible_matrix_claude_peer")
with open({config!r}, encoding="utf-8") as stream:
    config = json.load(stream)
raise SystemExit(peer["main"](sys.argv[1:], config))
"""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _number(value: str | None, kind: type) -> Any:
    try:
        return None if value is None else kind(value)
    except ValueError:
        return None


class ClaudeCliDialect:
    """The frames of the command-line tool, one model turn per request."""

    name = "claude_cli"
    path = "/claude-cli/v1/turn"
    provider = "cli"
    turn_control_field = True  # ``--max-turns`` on the command line of the tool
    # Where the peer asks what the tool says of a session id it holds no
    # conversation for: the tool keeps the conversations of this wire.
    resume_path = peer_program.RESUME_PATH
    behaviours: tuple[str, ...] = (
        "init",
        "assistant_text",
        # A model turn that thinks and writes no text.
        "assistant_thinking",
        "assistant_tool_use",
        "tool_result",
        "result",
        "result_structured",
        # The tool's own turn terminal, and the two refusals its frames carry.
        "result_max_turns",
        # The tool's own budget terminal, between two model turns.
        "result_max_budget",
        "assistant_refusal",
        "result_refusal",
        "assistant_api_error",
        "result_api_error",
        # No frame: what the tool prints, and the status it exits with, when
        # it is asked to continue a session it holds no conversation for.
        "resume_rejected",
    )

    @property
    def user_agent(self) -> str:
        """How the installed SDK marks a tool it starts: read, never typed."""

        return f"{_SDK}/{importlib.metadata.version(_SDK)} ({_ENTRYPOINT})"

    def envelope_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}/{self.provider}.{item}" for item in self.behaviours)

    # -- reading a turn request ------------------------------------------------

    @staticmethod
    def _session(evidence: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
        """The journal record of the session that sent a turn, and the roles before it."""

        journal = _records(evidence.get("journal"))
        for index, record in enumerate(journal):
            if record.get("role") == "session" and record.get("pid") == evidence.get("pid"):
                return record, [str(item.get("role")) for item in journal[:index]]
        return {}, []

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
            evidence = _mapping(json.loads(body.decode("utf-8")) if body else {})
        except (UnicodeDecodeError, json.JSONDecodeError):
            evidence = {}
        session, _before = self._session(evidence)
        argv = [str(item) for item in session.get("argv", ())]
        environment = _mapping(session.get("env"))
        models = [
            str(request.get("model"))
            for request in _records(evidence.get("control"))
            if request.get("subtype") == "set_model"
        ]
        # The scripted credential belongs to other routes: it must not reach
        # the tool, in any process the adapter started. The journal keeps no
        # value of an environment, so each process says whether it saw the
        # credential; one that does not say has not shown that it did not.
        journal = _records(evidence.get("journal"))
        reached = any(record.get("forbidden_seen") is not False for record in journal)
        written = bool(credential) and credential in json.dumps(journal, sort_keys=True)
        return SeenRequest(
            method=method,
            path=path,
            user_agent=(
                f"{_SDK}/{environment.get(_SDK_VERSION_ENV, '')} "
                f"({environment.get(_ENTRYPOINT_ENV, '')})"
            ),
            credential_ok=not reached and not written,
            # The model the session runs on: the last one the SDK set, or the
            # one on the command line.
            model=models[-1] if models else str(peer_program.argument(argv, "--model") or ""),
            system=str(peer_program.argument(argv, "--system-prompt") or ""),
            user=str(evidence.get("user", "")),
            tools=tuple(
                str(tool)
                for tools in _mapping(evidence.get("tools")).values()
                for tool in tools
            ),
            tool_results=tuple(
                (str(item.get("tool_use_id", "")), str(item.get("content", "")))
                for item in _records(evidence.get("tool_results"))
            ),
            turn_control=_number(peer_program.argument(argv, "--max-turns"), int),
            budget_control=_number(peer_program.argument(argv, "--max-budget-usd"), float),
            schema_field=peer_program.argument(argv, "--json-schema") is not None,
            host=lowered.get("host", ""),
            evidence=evidence,
            # The tool takes a cap on the tokens of a response from its
            # environment and from nowhere on its command line. The dialect
            # does not state that the wire has the field: a cap that shows
            # is held against a route that declares none.
            response_cap=_number(session.get("response_cap"), int),
        )

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        """Why this turn cannot come from a tool that the SDK started for the route."""

        problems: list[str] = []
        if seen.path != self.path:
            problems.append(f"request path {seen.path!r}, not {self.path!r}")
        session, before = self._session(seen.evidence)
        if not session:
            return [*problems, "the turn names no session process of the peer's journal"]
        if session.get("pid") == os.getpid() or session.get("ppid") != os.getpid():
            problems.append(
                f"the peer ran as process {session.get('pid')} with parent "
                f"{session.get('ppid')}: not a child that the SDK of this "
                f"process ({os.getpid()}) started"
            )
        if seen.user_agent != self.user_agent:
            problems.append(
                f"the tool was started as {seen.user_agent!r}, not by the "
                f"installed SDK's {self.user_agent!r}"
            )
        if not seen.credential_ok:
            problems.append(
                "a scripted credential reached the tool: its environment is not sanitised"
            )
        names = session.get("env_names")
        if not isinstance(names, list):
            problems.append("the session's journal record does not name its environment")
        inherited = sorted(set(matrix_columns.SCRIPTED_ENV) & set(map(str, names or ())))
        if inherited:
            problems.append(
                f"the scripted route variables {inherited} reached the tool: its "
                "environment is not sanitised"
            )
        if before[-2:] != ["auth_status", "version"]:
            problems.append(
                "the session did not follow the route's two probes (auth "
                f"status, version); before it ran {before[-2:]}"
            )
        argv = [str(item) for item in session.get("argv", ())]
        if [peer_program.argument(argv, flag) for flag in _STREAM_FLAGS] != ["stream-json"] * 2:
            problems.append("the command line is not the SDK's stream-json command line")
        control = [str(item.get("subtype")) for item in _records(seen.evidence.get("control"))]
        if control[:1] != ["initialize"] or "set_model" not in control:
            problems.append(
                "the first prompt did not follow the SDK's initialize request and "
                f"the adapter's set_model request; the control requests were {control}"
            )
        return problems

    # -- answering it ------------------------------------------------------------

    def refuse_resume(self, body: bytes) -> tuple[str, Reply]:
        """What the tool says of a session id it holds no conversation for.

        The line it writes to its standard error and the status it exits
        with, rendered from their envelope. Emulated, not observed: the
        envelope says what it rests on.
        """

        try:
            asked = _mapping(json.loads(body.decode("utf-8")) if body else {})
        except (UnicodeDecodeError, json.JSONDecodeError):
            asked = {}
        session_id = str(asked.get("resumed", ""))
        envelope = envelope_files.load(f"{self.name}/{self.provider}.resume_rejected")
        said = envelope.fill({"session_id": session_id})
        return session_id, Reply(
            200,
            {"stderr": "".join(said["stderr"]), "exit": said["exit"]},
            envelopes=(envelope.name,),
        )

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        served = seen.model
        if isinstance(step, ServedAs):
            served = seen.model + step.suffix
            step = step.step
        if isinstance(step, Refuse) and step.form not in REFUSALS:
            raise MatrixSetupError(
                f"the {self.name} dialect cannot render a refusal of the form "
                f"{step.form!r}: a message the tool relays has no refusal part, so "
                "the cell is pinned not expressible in the inventory"
            )
        if not isinstance(step, (Say, Call, Refuse, Reason, Malformed, Drop, Empty, Stall)):
            raise MatrixSetupError(
                f"the {self.name} dialect does not render {type(step).__name__} yet; "
                "the scenario that needs it adds the envelope with its liar"
            )
        if isinstance(step, Drop):
            require_cut_in_body(self.name, step)
        evidence = seen.evidence
        session_id = str(evidence.get("session_id", ""))
        turn = int(evidence.get("turn", 1))
        spent = _mapping(evidence.get("spent"))
        rate = matrix_columns.matrix_rate(seen.model)
        frames: list[Any] = []
        used: list[str] = []

        def frame(behaviour: str, **slots: Any) -> None:
            envelope = envelope_files.load(f"{self.name}/{self.provider}.{behaviour}")
            used.append(envelope.name)
            frames.append(
                envelope.fill({**slots, "session_id": session_id, "frame_id": str(uuid.uuid4())})
            )

        # The tool's own turn terminal is emulated by the peer, which ends the
        # session after the last turn it is allowed (see below): a turn beyond
        # the allowance is a turn the peer must never ask for. So is its
        # budget terminal: a turn that is asked for when the cost of the
        # session has reached the budget is a turn the peer must never ask
        # for either.
        unplayed = None
        if seen.turn_control is not None and turn > seen.turn_control:
            unplayed = (
                f"turn {turn} was asked for beyond --max-turns {seen.turn_control}: "
                "the peer ends a session at its turn terminal"
            )
        elif seen.budget_control is not None and seen.budget_control <= rate.compute_cost(
            input_tokens=int(spent.get("input_tokens", 0)),
            output_tokens=int(spent.get("output_tokens", 0)),
        ):
            unplayed = (
                f"turn {turn} was asked for with --max-budget-usd "
                f"{seen.budget_control} already spent: the peer ends a session at "
                "its budget terminal"
            )
        if unplayed is not None:
            return Reply(200, {"frames": [], "call": None, "unsupported": unplayed})

        if turn <= 1:
            frame("init")
        if isinstance(step, Refuse) and step.form == "http_policy":
            # The provider rejected the request: the tool emits a local
            # message in the place of the model turn, then an error terminal,
            # and exits with a status other than zero. Nothing is billed.
            frame("assistant_api_error", response_id=f"msg_{call_id}")
            frame("result_api_error", turns=turn)
            return Reply(
                200, {"frames": frames, "call": None, "exit": 1}, envelopes=tuple(used)
            )
        for item in _records(evidence.get("tool_results")):
            frame(
                "tool_result",
                call_id=str(item.get("tool_use_id", "")),
                tool_result=str(item.get("content", "")),
                tool_is_error=bool(item.get("is_error")),
            )
        if isinstance(step, Stall):
            # The provider takes the request of the turn and never answers
            # it. The tool has said what it says before a turn (the frame
            # that opens the session, the results of the tools it ran) and
            # says nothing more: the reply tells the peer to keep the
            # session open and silent. Nothing is billed and no model is
            # named.
            return Reply(
                200,
                {"frames": frames, "call": None, "stall": True},
                envelopes=tuple(used),
                marker=step.marker,
            )
        usage = step.usage
        turn_slots = {
            "response_id": f"msg_{call_id}",
            "served_model": served,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        if isinstance(step, Malformed):
            # The stream of the tool ends without its terminal: the frame of
            # the model turn arrives whole, then the tool ends its output and
            # exits with status 0 without a terminal frame. The turn frame
            # names the model and states the usage of the turn, which the
            # provider therefore bills.
            frame("assistant_text", **turn_slots, text=step.text)
            return Reply(
                200,
                {"frames": frames, "call": None, "gone": {"partial": "", "killed": False}},
                usage,
                served,
                envelopes=tuple(used),
            )
        if isinstance(step, Call):
            require_decoded_arguments(self.name, step)
            binding = BINDINGS[step.slot]
            arguments = dict(binding.arguments if step.args is None else step.args)
            listed = _mapping(evidence.get("tools"))
            # The server that lists the tool; a tool no server lists goes to
            # the first one, which refuses it.
            server = next(
                (name for name, tools in listed.items() if binding.name in tools),
                next(iter(listed), ""),
            )
            frame(
                "assistant_tool_use",
                **turn_slots,
                call_id=call_id,
                tool_name=f"mcp__{server}__{binding.name}",
                tool_input=arguments,
            )
            body: dict[str, Any] = {
                "frames": frames,
                "call": {
                    "id": call_id,
                    "server": server,
                    "name": binding.name,
                    "arguments": arguments,
                },
            }
            spent_in = int(spent.get("input_tokens", 0)) + usage.input_tokens
            spent_out = int(spent.get("output_tokens", 0)) + usage.output_tokens
            spent_usd = rate.compute_cost(input_tokens=spent_in, output_tokens=spent_out)
            at_turn_limit = seen.turn_control is not None and turn >= seen.turn_control
            if (
                not at_turn_limit
                and seen.budget_control is not None
                and seen.budget_control <= spent_usd
            ):
                # The cost of the session has reached its budget with this
                # turn. The tool then reports the budget terminal instead of
                # asking the model again, and exits with a status other than
                # zero. The peer plays it after the tool call of this turn,
                # as it plays the turn terminal below.
                played, frames = frames, []
                frame(
                    "result_max_budget",
                    turns=turn,
                    served_model=served,
                    input_tokens=spent_in,
                    output_tokens=spent_out,
                    cost_usd=spent_usd,
                )
                body["frames"], body["budget_limit"] = played, {"frames": frames, "exit": 1}
            if seen.turn_control is not None and turn >= seen.turn_control:
                # The session has used up its turn allowance with this turn.
                # The tool then reports the turn terminal instead of asking
                # the model again, and exits with a status other than zero.
                # The peer plays it after the tool call of this turn; the
                # frame is rendered here, with the accounting of the session
                # so far, so that no reply shape lives in the peer.
                played, frames = frames, []
                frame(
                    "result_max_turns",
                    # One more than the allowance: the turn that was not run.
                    turns=turn + 1,
                    served_model=served,
                    input_tokens=spent_in,
                    output_tokens=spent_out,
                    cost_usd=rate.compute_cost(
                        input_tokens=spent_in, output_tokens=spent_out
                    ),
                )
                body["frames"], body["turn_limit"] = played, {"frames": frames, "exit": 1}
            return Reply(
                200, body, usage, served, (step.slot, call_id, arguments), envelopes=tuple(used)
            )
        if isinstance(step, Refuse):
            # The provider's safety system stopped the turn: a billed message
            # without content, then the terminal of the session.
            frame("assistant_refusal", **turn_slots)
            refused_in = int(spent.get("input_tokens", 0)) + usage.input_tokens
            refused_out = int(spent.get("output_tokens", 0)) + usage.output_tokens
            frame(
                "result_refusal",
                turns=turn,
                served_model=served,
                input_tokens=refused_in,
                output_tokens=refused_out,
                cost_usd=rate.compute_cost(
                    input_tokens=refused_in, output_tokens=refused_out
                ),
            )
            return Reply(
                200,
                {"frames": frames, "call": None, "exit": 0},
                usage,
                served,
                envelopes=tuple(used),
            )
        answer: Any = None
        if seen.schema_field:
            try:
                answer = (
                    None if isinstance(step, (Reason, Empty)) else json.loads(step.text)
                )
            except json.JSONDecodeError:
                answer = None
            if not isinstance(answer, dict):
                return Reply(
                    200,
                    {
                        "frames": [],
                        "call": None,
                        "unsupported": "a session with a response schema ended on an "
                        "answer that is no JSON object: the tool's schema retries "
                        "are not emulated",
                    },
                )
        # In a schema session the answer travels in the structured output of
        # the terminal frame only; both texts carry the tool's note instead.
        if isinstance(step, Reason):
            # The turn thinks and writes nothing: the frame holds a thinking
            # block, and the result text of the terminal frame is empty.
            text = ""
            frame("assistant_thinking", **turn_slots, reasoning_text=REASONING_TEXT)
        elif isinstance(step, Empty):
            # The turn produced nothing: the frames of a plain turn with the
            # empty string as its text and as the result text, and a usage
            # without an output token.
            text = ""
            frame("assistant_text", **turn_slots, text=text)
        else:
            text = SCHEMA_SESSION_TEXT if seen.schema_field else step.text
            frame("assistant_text", **turn_slots, text=text)
        total_in = int(spent.get("input_tokens", 0)) + usage.input_tokens
        total_out = int(spent.get("output_tokens", 0)) + usage.output_tokens
        totals = {
            "text": text,
            "turns": turn,
            "served_model": served,
            "input_tokens": total_in,
            "output_tokens": total_out,
            # The rate the adapter is configured with, as the tool's own
            # price table is the vendor's in production.
            "cost_usd": rate.compute_cost(input_tokens=total_in, output_tokens=total_out),
        }
        if seen.schema_field:
            frame("result_structured", **totals, structured_output=answer)
        else:
            frame("result", **totals)
        if isinstance(step, Drop):
            # The tool goes away inside the turn. The wire is a stream of
            # frames, one line each, so it is cut as an HTTP stream is: the
            # frame that carries the answer travels whole and the client of
            # the tool holds the text; the line of the frame after it, the
            # terminal of the session, ends in its middle without a line
            # end, and the tool is killed. The turn frame that arrived names
            # the model and states the usage of the turn, which the provider
            # therefore bills.
            lines = [json.dumps(item).encode("utf-8") for item in frames]
            whole, partial = whole_and_partial(lines, step.text)
            return Reply(
                200,
                {
                    "frames": frames[:whole],
                    "call": None,
                    "gone": {"partial": partial.decode("utf-8"), "killed": True},
                },
                usage,
                served,
                envelopes=tuple(used[: whole + 1]),
            )
        return Reply(
            200, {"frames": frames, "call": None, "exit": 0}, usage, served, envelopes=tuple(used)
        )


@dataclass(frozen=True)
class Peer:
    """The scripted peer of one cell: its launcher, its configuration, its journal."""

    launcher: Path
    config: Path
    journal: Path

    def records(self) -> list[Mapping[str, Any]]:
        if not self.journal.is_file():
            return []
        lines = self.journal.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def pids(self) -> tuple[int, ...]:
        return tuple(sorted({int(record["pid"]) for record in self.records()}))

    def failures(self) -> list[str]:
        """What the peer refused to play, or crashed on."""

        return [
            f"{record.get('role')}: {record.get('detail')}"
            for record in self.records()
            if record.get("role") in _PEER_FAILURES
        ]


def reviewed_cli_version(model: str) -> str:
    """The newest tool version the adapter has reviewed for ``model``."""

    reviewed = CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL.get(model)
    if not reviewed:
        raise MatrixSetupError(
            f"the adapter has reviewed no command-line tool version for model {model!r}"
        )
    return max(reviewed, key=lambda version: tuple(int(part) for part in version.split(".")))


def write_peer(
    directory: Path,
    *,
    provider: str,
    organisation: Mapping[str, str],
    cli_version: str,
    faults: Sequence[str] = (),
    signed_in: bool = True,
    tool_request: str = peer_program.JOINED,
) -> Peer:
    """Write the launcher and the configuration of a cell's peer.

    With ``signed_in`` false the peer answers the sign-in probe of the
    session route as a tool that is not signed in; the rest of its answer is
    as on every other cell. ``tool_request`` is the order in which the peer
    sends the request for a tool call after the frame of the model turn that
    makes it, one of the two the peer states.
    """

    directory.mkdir(parents=True, exist_ok=True)
    peer = Peer(directory / "claude", directory / "config.json", directory / "journal.jsonl")
    peer.config.write_text(
        json.dumps(
            {
                "provider": provider,
                "journal": str(peer.journal),
                # What the two probes of the session route are answered with.
                "auth_status": {
                    "loggedIn": signed_in,
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    **organisation,
                },
                "cli_version": cli_version,
                "version_line": f"{cli_version} (Claude Code)",
                # What must not reach the tool. The peer says whether it saw
                # it and never writes it into its journal.
                "forbidden": [CANARY_CREDENTIAL],
                "faults": list(faults),
                "tool_request": tool_request,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    peer.launcher.write_text(
        _LAUNCHER.format(
            interpreter=sys.executable,
            guard=str(Path(netguard.__file__).resolve()),
            source=str(Path(peer_program.__file__).resolve()),
            config=str(peer.config),
        ),
        encoding="utf-8",
    )
    peer.launcher.chmod(0o700)
    return peer


# How a routing file says that a route takes its credential from the
# session of the vendor tool.
_SESSION = "claude_session"


@dataclass
class _Bound:
    backend: Any
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    peer: Peer
    # The control channel carries no bearer value of its own.
    secrets: tuple[str, ...] = ()
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "control_channel"

    def cancel_when(self, marker: str) -> Cancellation:
        return close_when(self.provider, self.backend, marker)

    def children(self) -> tuple[int, ...]:
        return self.peer.pids()  # every process the SDK started for the cell


class ClaudeDriver:
    """The session route of the Claude adapter, answered by the scripted peer."""

    kind = matrix_columns.CLAUDE_KIND
    dialect: str = ClaudeCliDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    bypassed_layers: tuple[str, ...] = (
        "the vendor command-line tool (a scripted peer is started in its place; "
        "its tool loop, session accounting, schema extraction and turn terminal "
        "are emulated)",
        "the provider (the peer takes each model turn from scripted envelopes)",
    )
    # Read, not patched: the version the peer answers the version probe with.
    private_seams: tuple[str, ...] = (
        "crucible.backends._claude_session_transport."
        "CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL",
    )
    # One scripted tool is started for the call and opens its session before
    # the request of the turn: some tenths of a second on an idle machine,
    # two seconds where the test shares one hardware thread with busy loops.
    startup_seconds: float = 5.0
    faults: tuple[str, ...] = ()
    # How the peer of a cell sends the request for a tool call after the
    # frame of the model turn that makes it. Every cell plays the tool that
    # asks its tool server at once, frame and request in one write: the
    # adapter then has the request in hand when it reads the frame, and a
    # cell shows the same thing in every play. The timing is the peer's and
    # is constructed: no source in reach states the vendor tool's.
    tool_request: str = peer_program.JOINED

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
        return self._dialect().envelope_names()

    def answers(self, script: ProviderScript) -> tuple[str, ...]:
        return answered_from(self._dialect(), script)

    def refuses(self) -> tuple[str, ...]:
        return refused_from(self._dialect())

    def _dialect(self) -> ClaudeCliDialect:
        return ClaudeCliDialect()

    def preflight(self) -> None:
        if os.name != "posix":
            raise MatrixSetupError(f"kind {self.kind} runs on POSIX only")
        if importlib.util.find_spec(_SDK_MODULE) is None:
            raise MatrixSetupError(
                f"kind {self.kind} needs the root environment's {_SDK!r}; "
                "run `uv sync --frozen --group dev`"
            )
        if len(sys.executable) > _MAX_INTERPRETER_LINE:
            raise MatrixSetupError(
                f"kind {self.kind}: the interpreter path is too long for the "
                "first line of the peer's launcher"
            )

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        return {_OVERLAY_FIELD: "<the scripted peer of the cell>"}

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
        role = matrix_columns.ROLE_BY_CAPABILITY[capability]
        with contextlib.ExitStack() as stack:
            dialect = self._dialect()
            provider = stack.enter_context(
                LoopbackProvider(script, credential=CANARY_CREDENTIAL, dialect=dialect)
            )
            shipped = load_routing(config_path=column.routing_path(COLUMN_PRODUCT))
            route = shipped.for_role(role)
            if without_credential and route.extra.get("credential_source") != _SESSION:
                raise MatrixSetupError(
                    f"column {column.id}: the route of role {role!r} does not name "
                    "the session of the vendor tool as the source of its credential, "
                    "so the peer has no sign-in to take away"
                )
            # The organisation of the session comes from the routing file.
            organisation = {
                wire: str(route.extra[name])
                for wire, name in (("orgId", "expected_org_id"), ("orgName", "expected_org_name"))
                if route.extra.get(name) is not None
            }
            peer = write_peer(
                tmp_path / "claude-peer",
                provider=provider.url + dialect.path,
                organisation=organisation,
                cli_version=reviewed_cli_version(str(route.model)),
                faults=self.faults,
                # The route takes its credential from the session of the
                # tool and reads it at the call, through the sign-in probe
                # of the tool: without it the tool reports that it is not
                # signed in.
                signed_in=not without_credential,
                tool_request=self.tool_request,
            )
            overlay = {name: str(peer.launcher) for name in self.overlay(column, {})}
            # The boundary of the matrix: exactly what a workflow calls.
            backend = resolve_for_role(
                role,
                routing=matrix_columns.overlaid_routing(
                    shipped, overlay, kind=self.kind, backend=column.backend
                ),
                capability=capability,
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(cwd),
                log_dir=log_dir,
                limit_policy=GUARD_POLICY if lens == "guarded" else None,
            )
            try:
                yield _Bound(backend, provider, log_dirs=(log_dir,), peer=peer)
            finally:
                with contextlib.suppress(Exception):
                    backend.close()  # the runner has observed close() already
                failures = peer.failures()
                if failures:
                    # Not an observation of the adapter: the stand-in for the
                    # vendor tool could not play what it was asked for.
                    raise MatrixSetupError(
                        f"column {column.id}: the scripted peer did not play the "
                        f"session: {'; '.join(failures)}"
                    )


register(ClaudeDriver())
