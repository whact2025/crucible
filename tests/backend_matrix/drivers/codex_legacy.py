# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Driver of the Codex app-server kind: the reference column of the session route.

The backend is built exactly as a workflow builds it, with ``resolve_for_role``
on the column's shipped routing file. The one overlay is ``extra.cli_path``,
which points the route at the scripted peer of the cell
(:mod:`..peers.codex_app_server`). Everything from the adapter down to the
process boundary runs: the pin of the tool and its revalidation, the version
probe before every call, the validation of the sign-in file and its private
copy, the allow-listed environment, the pinned ``codex-app-server-sdk`` with
its standard-stream client, the adapter's event capture and accounting, and
the loopback tool server with the real ToolSet.

The vendor tool and the provider are what the peer stands in for. The peer
takes every model response from the scripted loopback provider, in the
``codex_app_server`` dialect: one request per model response, answered with
the notifications the app-server emits for it, each rendered from an envelope.
So the provider ledger, the replay rules and clause C0 are the same as on
every other column. The app-server keeps the conversation of a thread itself,
so the dialect states no replay rule.

The session checks of the adapter are met as follows:

- the route accepts only a native executable. The peer is a script, so for as
  long as a backend is open the driver replaces the adapter's native-binary
  predicate by one that also accepts that one path: the one private name this
  driver patches;
- ``--version`` is answered with a version the adapter has reviewed for the
  route's model, read from the adapter's own table (read, not patched);
- the sign-in file the route validates is a fake one, written into a home
  directory of the cell that the driver makes ``HOME`` for as long as a
  backend is open. No other home is ever written to.

The seam canary of a column fails unless the SDK started the peer: a child of
the test process whose command line ends in the app-server subcommand, after
the version probe, with none of the scripted route variables in its
environment, the sign-in file in a private home of its own, the SDK's
``initialize`` request first and a thread that was started with the exact
usage notification asked for.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from crucible.backends import codex as adapter
from crucible.backends.config import load_routing, resolve_for_role

from .. import columns as matrix_columns
from .. import envelopes as envelope_files
from .. import netguard
from ..columns import CANARY_CREDENTIAL, COLUMN_PRODUCT, Column
from ..peers import codex_app_server as peer_program
from ..provider import LoopbackProvider, Reply, SeenRequest
from ..script import Call, ProviderScript, Say, Step
from ..toolsets import BINDINGS
from .base import GUARD_POLICY, Lens, MatrixSetupError, register

_SDK_MODULE = "codex_app_server_sdk"
_OVERLAY_FIELD = "extra.cli_path"
_PEER_FAILURES = ("unsupported", "crashed")
# The three requests that open a session, in the order a client sends them.
_OPENING = ("initialize", "thread/start", "turn/start")
# The kernel refuses an interpreter line longer than this.
_MAX_INTERPRETER_LINE = 250
# How the peer names itself in the answer to ``initialize``. No client of the
# matrix reads the name; the official SDK requires that one is there.
_TOOL_LABEL = "codex_cli_rs"
# The word in the sign-in file of a cell. It is a fake; clause C9 looks for it
# wherever it looks for the canary credential.
LOGIN_WORD = "matrix-fake-session-word"

_LAUNCHER = """#!{interpreter} -IS
# Written by the backend matrix for one cell: the scripted Codex app-server peer.
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
peer = load({source!r}, "crucible_matrix_codex_peer")
with open({config!r}, encoding="utf-8") as stream:
    config = json.load(stream)
raise SystemExit(peer["main"](sys.argv[1:], config))
"""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(content: Any) -> str:
    """The text of a tool result: the text blocks of its content list."""

    return "".join(
        str(block.get("text", "")) for block in _records(content) if block.get("type") == "text"
    )


def _turn_control(*requests: Mapping[str, Any]) -> int | None:
    """A turn ceiling in a request of the client, under any spelling of the key."""

    for params in requests:
        for key, value in params.items():
            if key.replace("_", "").lower() == "maxturns" and type(value) is int:
                return value
    return None


class CodexAppServerDialect:
    """The notifications of the app-server, one model response per request."""

    name = "codex_app_server"
    path = "/codex-app-server/v1/response"
    provider = "app_server"
    behaviours: tuple[str, ...] = (
        "agent_message",
        "tool_call_started",
        "tool_call_completed",
        "raw_response_completed",
        "token_usage_updated",
        "turn_completed",
    )

    def __init__(self, *, client: str = "") -> None:
        # How the client of the route names itself in its first request.
        self._client = client

    @property
    def user_agent(self) -> str:
        return self._client

    def envelope_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}/{self.provider}.{item}" for item in self.behaviours)

    # -- reading a request for a model response ----------------------------------

    @staticmethod
    def session(evidence: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
        """The journal record of the session that asked, and the roles before it."""

        journal = _records(evidence.get("journal"))
        for index, record in enumerate(journal):
            if record.get("role") == "app_server" and record.get("pid") == evidence.get("pid"):
                return record, [str(item.get("role")) for item in journal[:index]]
        return {}, []

    @staticmethod
    def request(evidence: Mapping[str, Any], method: str) -> Mapping[str, Any]:
        """The parameters of the first request of the client with this method."""

        for item in _records(evidence.get("requests")):
            if item.get("method") == method:
                return _mapping(item.get("params"))
        return {}

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
        client = _mapping(self.request(evidence, "initialize").get("clientInfo"))
        thread = self.request(evidence, "thread/start")
        turn = self.request(evidence, "turn/start")
        # A scripted credential belongs to other routes and the sign-in word to
        # a file: neither may reach the tool through its environment or its
        # command line. The journal keeps no value of an environment, so each
        # process says whether it saw one; a process that does not say has not
        # shown that it did not.
        journal = _records(evidence.get("journal"))
        reached = any(record.get("forbidden_seen") is not False for record in journal)
        written = bool(credential) and credential in json.dumps(journal, sort_keys=True)
        return SeenRequest(
            method=method,
            path=path,
            user_agent=f"{client.get('name', '')}/{client.get('version', '')}",
            credential_ok=not reached and not written,
            # The model the thread was started on: the wire names it once.
            model=str(thread.get("model", "")),
            system=str(thread.get("baseInstructions", "")),
            user="".join(
                str(item.get("text", ""))
                for item in _records(turn.get("input"))
                if item.get("type") == "text"
            ),
            tools=tuple(
                str(tool)
                for tools in _mapping(evidence.get("tools")).values()
                for tool in tools
            ),
            tool_results=tuple(
                (str(item.get("call_id", "")), _text(item.get("content")))
                for item in _records(evidence.get("tool_results"))
            ),
            # The protocol has no turn ceiling and no spend ceiling; a request
            # that carried a turn ceiling under any spelling would show here.
            turn_control=_turn_control(thread, turn),
            budget_control=None,
            schema_field=turn.get("outputSchema") is not None,
            host=lowered.get("host", ""),
            evidence=evidence,
        )

    def launch_problems(
        self, session: Mapping[str, Any], before: Sequence[str], seen: SeenRequest
    ) -> list[str]:
        """Why this session was not started by the client of the route."""

        raise NotImplementedError

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        """Why this response cannot be asked for by a tool the route's client started."""

        problems: list[str] = []
        if seen.path != self.path:
            problems.append(f"request path {seen.path!r}, not {self.path!r}")
        session, before = self.session(seen.evidence)
        if not session:
            return [*problems, "the request names no session process of the peer's journal"]
        problems.extend(self.launch_problems(session, before, seen))
        if seen.user_agent != self.user_agent:
            problems.append(
                f"the session was opened by {seen.user_agent!r}, not by the "
                f"route's client {self.user_agent!r}"
            )
        if not seen.credential_ok:
            problems.append(
                "a scripted credential or the sign-in word reached the tool through "
                "its environment or command line"
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
        command = [str(item) for item in _mapping(seen.evidence.get("command")).get("command", ())]
        if command[:1] != [peer_program.APP_SERVER]:
            problems.append("the command line does not start the app-server")
        login = _mapping(session.get("login"))
        home = _mapping(session.get("env"))
        if not (login.get("present") and login.get("private")) or home.get(
            "CODEX_HOME"
        ) in (None, "", os.path.join(str(home.get("HOME", "")), ".codex")):
            problems.append(
                "the tool did not find the sign-in file as a private copy in a home "
                "of its own"
            )
        methods = [str(item.get("method")) for item in _records(seen.evidence.get("requests"))]
        if methods[:1] != [_OPENING[0]] or [m for m in methods if m in _OPENING] != [*_OPENING]:
            problems.append(
                "the session was not opened by initialize, thread/start and turn/start "
                f"in that order; the client sent {methods}"
            )
        return problems

    # -- answering it ------------------------------------------------------------

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        if not isinstance(step, (Say, Call)):
            raise MatrixSetupError(
                f"the {self.name} dialect does not render {type(step).__name__}: the "
                "wire names no served model, and every other step lands with the "
                "scenario that needs its envelope and its liar"
            )
        evidence = seen.evidence
        ids = {
            "thread_id": str(evidence.get("thread_id", "")),
            "turn_id": str(evidence.get("turn_id", "")),
        }
        spent = _mapping(evidence.get("spent"))
        frames: list[Any] = []
        used: list[str] = []

        def frame(behaviour: str, **slots: Any) -> None:
            envelope = envelope_files.load(f"{self.name}/{self.provider}.{behaviour}")
            used.append(envelope.name)
            frames.append(envelope.fill({**ids, **slots}))

        def now() -> int:
            return int(time.time() * 1000)

        for item in _records(evidence.get("tool_results")):
            if item.get("is_error"):
                # How the tool reports a failed call is not known.
                return Reply(
                    200,
                    {
                        "frames": [],
                        "call": None,
                        "unsupported": "a tool call that its server answered with an "
                        "error has no envelope: the scenario that needs it adds one",
                    },
                )
            frame(
                "tool_call_completed",
                created=now(),
                call_id=str(item.get("call_id", "")),
                tool_server=str(item.get("server", "")),
                tool_name=str(item.get("name", "")),
                tool_input=item.get("arguments"),
                tool_result=item.get("content"),
            )
        usage = step.usage

        def account() -> None:
            total_in = int(spent.get("input_tokens", 0)) + usage.input_tokens
            total_out = int(spent.get("output_tokens", 0)) + usage.output_tokens
            if evidence.get("raw_events") is True:
                frame(
                    "raw_response_completed",
                    response_id=f"resp_{call_id}",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            frame(
                "token_usage_updated",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                cumulative_input_tokens=total_in,
                cumulative_output_tokens=total_out,
                cumulative_total_tokens=total_in + total_out,
            )

        if isinstance(step, Call):
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
                "tool_call_started",
                created=now(),
                call_id=call_id,
                tool_server=server,
                tool_name=binding.name,
                tool_input=arguments,
            )
            account()
            body = {
                "frames": frames,
                "call": {
                    "id": call_id,
                    "server": server,
                    "name": binding.name,
                    "arguments": arguments,
                },
            }
            # The wire names no served model, so the reply names none.
            return Reply(
                200, body, usage, None, (step.slot, call_id, arguments), envelopes=tuple(used)
            )
        frame("agent_message", created=now(), item_id=f"msg_{call_id}", text=step.text)
        account()
        frame("turn_completed")
        return Reply(200, {"frames": frames, "call": None}, usage, None, envelopes=tuple(used))


class LegacyDialect(CodexAppServerDialect):
    """The app-server as the pinned ``codex-app-server-sdk`` of this process starts it."""

    def launch_problems(
        self, session: Mapping[str, Any], before: Sequence[str], seen: SeenRequest
    ) -> list[str]:
        problems: list[str] = []
        if session.get("pid") == os.getpid() or session.get("ppid") != os.getpid():
            problems.append(
                f"the peer ran as process {session.get('pid')} with parent "
                f"{session.get('ppid')}: not a child that the SDK of this process "
                f"({os.getpid()}) started"
            )
        if list(before[-1:]) != ["version"]:
            problems.append(
                "the session did not follow the route's version probe; before it "
                f"ran {list(before[-1:])}"
            )
        if seen.evidence.get("raw_events") is not True:
            problems.append(
                "the thread was not started with the exact usage notification asked "
                "for (experimentalApi and experimentalRawEvents)"
            )
        return problems


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


def write_peer(
    directory: Path,
    *,
    provider: str,
    cli_version: str,
    faults: Sequence[str] = (),
) -> Peer:
    """Write the launcher and the configuration of a cell's peer."""

    directory.mkdir(parents=True, exist_ok=True)
    peer = Peer(directory / "codex", directory / "config.json", directory / "journal.jsonl")
    peer.config.write_text(
        json.dumps(
            {
                "provider": provider,
                "journal": str(peer.journal),
                # What the version probe of the session route is answered with.
                "cli_version": cli_version,
                "version_line": f"codex-cli {cli_version}",
                "user_agent": f"{_TOOL_LABEL}/{cli_version}",
                # What must not reach the tool through its environment or its
                # command line. The peer says whether it saw one of them and
                # never writes it into its journal.
                "forbidden": [CANARY_CREDENTIAL, LOGIN_WORD],
                "faults": list(faults),
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


def write_login(home: Path) -> Path:
    """Write the fake sign-in file of a cell into ``home``, private to its owner."""

    directory = home / ".codex"
    directory.mkdir(parents=True, exist_ok=True)
    login = directory / peer_program.LOGIN_FILE
    login.write_text(
        json.dumps(
            {"auth_mode": "chatgpt", "OPENAI_API_KEY": None, "tokens": {"word": LOGIN_WORD}}
        ),
        encoding="utf-8",
    )
    login.chmod(0o600)
    return login


@contextlib.contextmanager
def private_home(tmp_path: Path) -> Iterator[Path]:
    """Make a home directory of the cell ``HOME`` for the block, with its sign-in file.

    The adapters read the sign-in file from the home directory of the process.
    The driver never writes into the home it finds: it makes a directory of
    the cell the home for as long as a backend is open, and puts back what
    was there. No ambient tool home is in force meanwhile.
    """

    home = tmp_path / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    write_login(home)
    previous = {name: os.environ.get(name) for name in ("HOME", "CODEX_HOME")}
    os.environ["HOME"] = str(home)
    os.environ.pop("CODEX_HOME", None)
    try:
        yield home
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def reviewed_cli_version(model: str) -> str:
    """The newest tool version the adapter has reviewed for ``model``."""

    reviewed = adapter._CODEX_CHATGPT_SESSION_CLI_VERSIONS_BY_MODEL.get(model)
    if not reviewed:
        raise MatrixSetupError(
            f"the adapter binds model {model!r} to no reviewed tool version: the "
            "peer would have to invent the answer to the version probe"
        )
    return max(reviewed, key=lambda version: tuple(int(part) for part in version.split(".")))


def sdk_client() -> str:
    """How the installed ``codex-app-server-sdk`` names itself: read, never typed."""

    from codex_app_server_sdk import client as sdk  # noqa: PLC0415 - preflight reports its absence

    info = sdk._default_initialize_params()["clientInfo"]
    return f"{info['name']}/{info['version']}"


@contextlib.contextmanager
def _native(launcher: Path) -> Iterator[None]:
    """Let the session route take the peer's launcher for a native executable.

    The route accepts a native binary or the vendor's package launcher and
    nothing else. The peer is a script, so for the block the adapter's
    predicate also accepts that one path; every other path is judged as
    before. The predicate's own tests stay where they are.
    """

    original = adapter._is_native_codex_binary
    accepted = launcher.resolve()

    def is_native_codex_binary(path: Path) -> bool:
        return path == accepted or original(path)

    adapter._is_native_codex_binary = is_native_codex_binary
    try:
        yield
    finally:
        adapter._is_native_codex_binary = original


@dataclass
class _Bound:
    backend: Any
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    peer: Peer
    # The tool server of the route takes no bearer value; the sign-in word is
    # the one other secret a cell holds.
    secrets: tuple[str, ...] = (LOGIN_WORD,)
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "mcp_bridge"

    def cancel_when(self, marker: str) -> None:
        raise MatrixSetupError(
            f"the Codex driver has no marker {marker!r}: cancellation scenarios add "
            "it with their liars"
        )

    def children(self) -> tuple[int, ...]:
        return self.peer.pids()  # every process the route started for the cell


class CodexLegacyDriver:
    """The session route of the Codex adapter, answered by the scripted peer."""

    kind = matrix_columns.CODEX_LEGACY_KIND
    dialect: str = CodexAppServerDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    bypassed_layers: tuple[str, ...] = (
        "the vendor command-line tool (a scripted peer is started in its place; "
        "its tool loop and the usage counters of its thread are emulated)",
        "the native-binary predicate of the session route (it also accepts the "
        "peer's launcher; its unit tests stay)",
        "the provider (the peer takes each model response from scripted envelopes)",
    )
    # The first is patched for as long as a backend is open; the second is
    # read for the answer of the peer to the version probe.
    private_seams: tuple[str, ...] = (
        f"{adapter.__name__}._is_native_codex_binary",
        f"{adapter.__name__}._CODEX_CHATGPT_SESSION_CLI_VERSIONS_BY_MODEL",
    )
    faults: tuple[str, ...] = ()

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
        return CodexAppServerDialect().envelope_names()

    def _dialect(self) -> CodexAppServerDialect:
        return LegacyDialect(client=sdk_client())

    def preflight(self) -> None:
        if os.name != "posix":
            raise MatrixSetupError(f"kind {self.kind} runs on POSIX only")
        if importlib.util.find_spec(_SDK_MODULE) is None:
            raise MatrixSetupError(
                f"kind {self.kind} needs the root environment's {_SDK_MODULE!r}; "
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
    ) -> Iterator[_Bound]:
        if column.kind != self.kind:
            raise MatrixSetupError(f"column {column.id} is not of kind {self.kind}")
        log_dir, cwd = tmp_path / "backend-logs", tmp_path / "backend-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        role = matrix_columns.ROLE_BY_CAPABILITY[capability]
        with contextlib.ExitStack() as stack:
            stack.enter_context(private_home(tmp_path))
            dialect = self._dialect()
            provider = stack.enter_context(
                LoopbackProvider(script, credential=CANARY_CREDENTIAL, dialect=dialect)
            )
            shipped = load_routing(config_path=column.routing_path(COLUMN_PRODUCT))
            peer = write_peer(
                tmp_path / "codex-peer",
                provider=provider.url + dialect.path,
                cli_version=reviewed_cli_version(str(shipped.for_role(role).model)),
                faults=self.faults,
            )
            stack.enter_context(_native(peer.launcher))
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


register(CodexLegacyDriver())
