# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The scripted peer that stands where a Codex route looks for its vendor tool.

A Codex column's driver writes a small launcher per cell and puts it where the
route looks for the vendor command-line tool: as the route's ``cli_path`` on
the app-server route, and where the official SDK looks for its bundled runtime
on the worker route. Whoever starts the tool then starts this program:

- ``--version``, the probe of the session route. The answer comes from the
  cell's configuration: a version the adapter has reviewed for the model.
- ``[options] app-server [--listen stdio://]``, the session. The peer speaks
  the app-server's JSON-RPC on its standard streams, one JSON object per
  line, and takes every model response from the scripted provider of the
  cell over loopback HTTP. The provider answers with the notifications to
  emit, rendered from envelopes, so no reply shape of a model response lives
  in this file.

What the peer does on its own is what an app-server owns: it answers the
requests that open a session (``initialize``, ``thread/start``, a
``thread/resume`` of the thread it holds, ``turn/start``) and a
``thread/read`` of that thread, connects the tool
servers its command line configures (``mcp_servers.<name>.url``) over
streamable HTTP, sends each tool call the
model makes to the server that lists the tool, feeds the result into the next
model response and keeps the usage counters of its thread: what the usage
notification of the last response it relayed states as the thread's total,
which it hands to the provider with the next request. It reads nothing else
of that notification, so what the notification says of one response is the
provider's statement alone. It emits
the exact per-response usage notification only to a client that asked for it,
as the app-server does: ``experimentalApi`` in ``initialize`` and
``experimentalRawEvents`` in ``thread/start``. And it goes away where a reply
says so (``gone``): after the notifications of the reply it writes the
leading part of a line the reply names, without a line end, and then either
ends its output and exits with status 0, as a tool whose stream ends without
the notification that ends the turn, or kills itself, as a tool that is gone
in the middle of a response.

The answers that open a session are not envelopes. They are the response
shapes of the app-server protocol as the vendor's generated types state them
(``openai_codex.generated.v2_all`` of the pinned SDK: ``ThreadStartResponse``
and ``ThreadResumeResponse`` with their ``Thread``, ``TurnStartResponse`` with its
``Turn``; and ``openai_codex.models.InitializeResponse``), with their required
fields and
nothing else; the official SDK validates them against those types in every
cell of its column. Every value in them is taken from the request, generated
(ids, a timestamp) or a filler that no client reads (the provider label).

The answer to ``thread/read`` is the same ``Thread`` (``ThreadReadResponse``),
with the turns of the session when the client asks for them: each ``Turn``
holds the items of the ``item/completed`` notifications the peer relayed in
it and the status its ``turn/completed`` notification stated. So what a
client reads back of a thread is what the provider's notifications said of
it, and nothing the peer made up. A client asks for it after a turn that
completed without a message of the agent.

Every process appends one record to the journal of the cell: its role, its
pids, its command line, the names of its environment and whether it saw a
value the cell forbids. The session also records what it found where the tool
keeps its sign-in (``$CODEX_HOME/auth.json``): that the file is there, private
and of the session kind, never a value of it. The session sends the journal
with each model response, so the provider side can tell how the peer was
started.

Standard library only and no package-relative import: the launcher runs this
file with ``-I -S``, outside the test environment, after it has installed the
network guard of the matrix in the process. Only loopback may be connected to
or resolved there, as in the test process.

Protocol sources, each at the release its lock pins (``codex-app-server-sdk``
in the root lock, ``openai-codex`` in the Codex worker's):

- command line: ``crucible/backends/codex.py`` (the app-server route builds
  ``--disable <feature>``, ``-c <key>=<value>``, ``app-server``) and
  ``openai_codex.client.CodexClient.start`` (``--config <key>=<value>``,
  ``app-server --listen stdio://``);
- requests of the two clients: ``codex_app_server_sdk.client``
  (``initialize``, ``thread/start``, ``thread/resume`` before a turn on a
  thread it holds a handle of, ``turn/start``) and ``openai_codex.client``
  (the same, and the ``initialized`` notification);
- the tool servers: the streamable HTTP transport of the Model Context
  Protocol (``initialize``, ``notifications/initialized``, ``tools/list``,
  ``tools/call``; one POST per message, the answer as JSON or as one event
  stream, the session id in ``Mcp-Session-Id``).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, BinaryIO

VERSION_COMMAND = ["--version"]
APP_SERVER = "app-server"
LISTEN_STDIO = ["--listen", "stdio://"]
# The protocol revision the peer offers a tool server; the server names the
# one it speaks.
MCP_PROTOCOL_VERSION = "2025-03-26"
CLIENT_NAME = "crucible-matrix-codex-peer"
SKIP_TOOL_SERVER = "skip_tool_server"
FABRICATED_RESULT = "a result the peer made up without asking the tool server"
SKIP_ADDRESS_CHECK = "skip_address_check"
# The only variables the journal keeps by value: where the tool keeps its
# sign-in, the home directory and the search path it was given.
JOURNALLED_ENV = ("CODEX_HOME", "HOME", "PATH")
MASKED = "<forbidden value>"
LOGIN_FILE = "auth.json"
# What a thread is started with when the request names none (the defaults of
# the protocol's own enumerations).
SANDBOX_POLICIES = {"read-only": {"type": "readOnly"}}
PROVIDER_LABEL = "openai"

EXIT_PROVIDER_FAILED = 1
EXIT_UNSUPPORTED = 3
EXIT_CRASHED = 4


class _InputClosed(Exception):
    """The client closed the peer's standard input."""


class _Unsupported(Exception):
    """The peer was asked for something it does not play."""


class _Gone(Exception):
    """The tool goes away now: its output ends here, in order or cut off."""

    def __init__(self, partial: str, killed: bool) -> None:
        super().__init__("the scripted tool goes away")
        self.partial = partial
        self.killed = killed


def go_away(gone: _Gone, stdout: BinaryIO) -> int:
    """Write what is left of the output and leave; the status of a tool that is not killed.

    A tool that is cut off has written part of a line and no line end, and
    is killed where it stands: nothing of this process runs after that.
    """

    stdout.write(gone.partial.encode("utf-8"))
    stdout.flush()
    if gone.killed:
        import signal  # noqa: PLC0415 - only a tool that is cut off needs it

        os.kill(os.getpid(), signal.SIGKILL)
    return 0


def parse_command(argv: list[str]) -> dict[str, Any]:
    """The options before the subcommand, the subcommand and what follows it."""

    disabled: list[str] = []
    settings: list[str] = []
    index = 0
    while index + 1 < len(argv):
        if argv[index] == "--disable":
            disabled.append(argv[index + 1])
        elif argv[index] in ("-c", "--config"):
            settings.append(argv[index + 1])
        else:
            break
        index += 2
    return {"disabled": disabled, "settings": settings, "command": argv[index:]}


def tool_servers(settings: list[str]) -> dict[str, str]:
    """The tool servers the configuration overrides leave: name -> URL."""

    servers: dict[str, str] = {}
    for setting in settings:
        key, _, value = setting.partition("=")
        if key == "mcp_servers":
            if value.strip() != "{}":
                raise _Unsupported("mcp_servers is set to something other than {}")
            servers.clear()
        elif key.startswith("mcp_servers.") and key.endswith(".url"):
            try:
                url = json.loads(value)
            except ValueError as exc:
                raise _Unsupported(f"{key} is not a quoted string") from exc
            servers[key[len("mcp_servers.") : -len(".url")]] = str(url)
    return servers


def login_state() -> dict[str, Any]:
    """What the tool finds where it keeps its sign-in; never a value of it."""

    home = os.environ.get("CODEX_HOME", "")
    if not home:
        return {"home": False, "present": False}
    path = os.path.join(home, LOGIN_FILE)
    try:
        status = os.stat(path, follow_symlinks=False)
        with open(path, encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, ValueError):
        return {"home": True, "present": False}
    record = record if isinstance(record, dict) else {}
    return {
        "home": True,
        "present": True,
        "private": not status.st_mode & 0o077,
        "auth_mode": str(record.get("auth_mode")),
        "api_key": record.get("OPENAI_API_KEY") not in (None, ""),
        "session": isinstance(record.get("tokens"), dict),
    }


def started_as(argv: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """What the journal says about how this process was started.

    The command line, the names of the environment, the values of
    :data:`JOURNALLED_ENV`, and whether a value the cell forbids was seen in
    the environment or on the command line. Whatever holds a forbidden value
    is masked, so the journal cannot show one.
    """

    forbidden = [str(word) for word in config.get("forbidden", ()) if word]

    def holds(text: str) -> bool:
        return any(word in text for word in forbidden)

    def shown(text: str) -> str:
        return MASKED if holds(text) else text

    environment = dict(os.environ)
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": shown(os.getcwd()),
        "argv": [shown(item) for item in argv],
        "env_names": sorted(shown(name) for name in environment),
        "env": {
            name: shown(environment[name]) for name in JOURNALLED_ENV if name in environment
        },
        "forbidden_seen": any(
            holds(item) for item in (*argv, *environment, *environment.values())
        ),
    }


def append_journal(config: dict[str, Any], record: dict[str, Any]) -> None:
    with open(config["journal"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def read_journal(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        with open(config["journal"], encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []


def _stream_messages(kind: str, body: bytes) -> list[dict[str, Any]]:
    """The JSON-RPC messages of one HTTP answer: a document or an event stream."""

    text = body.decode("utf-8")
    documents: list[str] = []
    if kind.split(";", 1)[0].strip().lower() == "text/event-stream":
        data: list[str] = []
        for line in [*text.splitlines(), ""]:
            if line.startswith("data:"):
                data.append(line[len("data:") :].lstrip(" "))
            elif not line and data:
                documents.append("\n".join(data))
                data = []
    elif text.strip():
        documents.append(text)
    messages: list[dict[str, Any]] = []
    for document in documents:
        decoded = json.loads(document)
        for item in decoded if isinstance(decoded, list) else [decoded]:
            if isinstance(item, dict):
                messages.append(item)
    return messages


class ToolServer:
    """One tool server of the command line, reached over streamable HTTP."""

    def __init__(self, name: str, url: str, config: dict[str, Any]) -> None:
        from urllib.parse import urlsplit  # noqa: PLC0415 - the probe never needs it

        target = urlsplit(url)
        # A lying counterpart for the matrix's own tests skips this check. The
        # network guard that the launcher installed is then what stands
        # between the peer and an address that is not loopback.
        unchecked = SKIP_ADDRESS_CHECK in config.get("faults", ())
        if not unchecked and (target.scheme != "http" or target.hostname != "127.0.0.1"):
            raise _Unsupported(f"tool server {name!r} is not on loopback")
        self.name = name
        self._host = target.hostname or ""
        self._port = target.port
        self._path = target.path or "/"
        self._client_version = str(config.get("cli_version", ""))
        self._session: str | None = None
        self._protocol: str | None = None
        self._messages = 0
        self.tools: list[str] = []
        self.notes: list[dict[str, Any]] = []

    def _post(self, message: dict[str, Any]) -> tuple[int, str, bytes]:
        import http.client  # noqa: PLC0415 - the probe never needs it

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": CLIENT_NAME,
        }
        if self._session is not None:
            headers["Mcp-Session-Id"] = self._session
        if self._protocol is not None:
            headers["MCP-Protocol-Version"] = self._protocol
        connection = http.client.HTTPConnection(self._host, self._port, timeout=120.0)
        try:
            connection.request(
                "POST", self._path, body=json.dumps(message).encode("utf-8"), headers=headers
            )
            response = connection.getresponse()
            body = response.read()
            session = response.getheader("Mcp-Session-Id")
            kind = response.getheader("Content-Type") or ""
        finally:
            connection.close()
        if session:
            self._session = session
        return response.status, kind, body

    def _ask(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one request; return its answer (``result`` or ``error``)."""

        self._messages += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._messages, "method": method}
        if params is not None:
            message["params"] = params
        note: dict[str, Any] = {"server": self.name, "method": method}
        if method == "tools/call":
            note["name"] = (params or {}).get("name")
        self.notes.append(note)
        status, kind, body = self._post(message)
        if status >= 400:
            return {"error": {"message": f"the tool server answered HTTP {status}"}}
        for answer in _stream_messages(kind, body):
            if answer.get("id") == message["id"]:
                return answer
        return {"error": {"message": "the tool server sent no answer to the request"}}

    def connect(self) -> None:
        """The handshake of a tool server: initialise, confirm, list the tools."""

        opened = self._ask(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": self._client_version},
            },
        )
        if "error" in opened:
            raise _Unsupported(
                f"tool server {self.name!r} refused the handshake: "
                f"{opened['error'].get('message')}"
            )
        self._protocol = str(opened.get("result", {}).get("protocolVersion", "")) or None
        self.notes.append({"server": self.name, "method": "notifications/initialized"})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = self._ask("tools/list")
        tools = listed.get("result", {}).get("tools", [])
        self.tools = [str(tool.get("name", "")) for tool in tools if isinstance(tool, dict)]

    def call(self, name: str, arguments: Any) -> tuple[list[Any], bool]:
        """Run one tool call; return the content the server sent and its error flag."""

        answer = self._ask("tools/call", {"name": name, "arguments": arguments})
        if "error" in answer:
            message = str(answer["error"].get("message", ""))
            return [{"type": "text", "text": message}], True
        result = answer.get("result", {})
        content = result.get("content")
        return (content if isinstance(content, list) else []), bool(result.get("isError"))


class AppServer:
    """One app-server session, driven by a client on the standard streams."""

    def __init__(
        self,
        command: dict[str, Any],
        config: dict[str, Any],
        stdin: BinaryIO,
        stdout: BinaryIO,
    ) -> None:
        self._command = command
        self._config = config
        self._stdin = stdin
        self._stdout = stdout
        self._requests: list[dict[str, Any]] = []
        self._servers: dict[str, ToolServer] = {}
        self._experimental = False
        self._thread: dict[str, Any] | None = None
        self._turn_id = ""
        # The turns of the thread, as a client reads them back: the items
        # that were completed in each and the status it ended with.
        self._turns: list[dict[str, Any]] = []
        self._responses = 0
        self._spent = {"input_tokens": 0, "output_tokens": 0}

    # -- the standard streams ------------------------------------------------

    def _emit(self, message: dict[str, Any]) -> None:
        self._stdout.write(json.dumps(message).encode("utf-8") + b"\n")
        self._stdout.flush()

    def _read(self) -> dict[str, Any]:
        while True:
            line = self._stdin.readline()
            if not line:
                raise _InputClosed
            if line.strip():
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise _Unsupported("a message on standard input is not an object")
                return message

    def _refuse(self, request_id: Any, method: str) -> None:
        self._emit(
            {
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"the scripted peer does not play {method!r}",
                },
            }
        )
        raise _Unsupported(f"request {method!r}")

    # -- the three requests that open a session ------------------------------

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        capabilities = params.get("capabilities")
        self._experimental = (
            isinstance(capabilities, dict) and capabilities.get("experimentalApi") is True
        )
        return {
            "userAgent": str(self._config["user_agent"]),
            "platformFamily": "unix",
            "platformOs": "linux",
        }

    def _start_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        import time  # noqa: PLC0415 - the probe never needs it
        import uuid  # noqa: PLC0415

        if self._thread is not None:
            raise _Unsupported("a second thread in one session")
        sandbox = SANDBOX_POLICIES.get(str(params.get("sandbox", "read-only")))
        if sandbox is None:
            raise _Unsupported(f"sandbox {params.get('sandbox')!r}")
        for name, url in tool_servers(self._command["settings"]).items():
            server = ToolServer(name, url, self._config)
            server.connect()
            self._servers[name] = server
        now = int(time.time())
        cwd = str(params.get("cwd") or os.getcwd())
        self._thread = {
            "id": str(uuid.uuid4()),
            "model": str(params.get("model", "")),
            # The exact per-response usage is sent only where it was asked for.
            "raw_events": self._experimental and params.get("experimentalRawEvents") is True,
            # What the thread was started as: the answer to a request that
            # starts it or resumes it.
            "described": {
                "thread": {
                    "id": "",
                    "sessionId": str(uuid.uuid4()),
                    "preview": "",
                    "ephemeral": params.get("ephemeral") is True,
                    "modelProvider": PROVIDER_LABEL,
                    "createdAt": now,
                    "updatedAt": now,
                    "status": {"type": "idle"},
                    "cwd": cwd,
                    "cliVersion": str(self._config["cli_version"]),
                    "source": "appServer",
                    "turns": [],
                },
                "model": str(params.get("model", "")),
                "modelProvider": PROVIDER_LABEL,
                "cwd": cwd,
                "approvalPolicy": params.get("approvalPolicy") or "on-request",
                "approvalsReviewer": params.get("approvalsReviewer") or "user",
                "sandbox": sandbox,
            },
        }
        self._thread["described"]["thread"]["id"] = self._thread["id"]
        return dict(self._thread["described"])

    def _resume_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """A client asks again for the thread this session holds."""

        if self._thread is None or params.get("threadId") != self._thread["id"]:
            raise _Unsupported("the resumption of a thread this session did not start")
        return dict(self._thread["described"])

    def _start_turn(self, params: dict[str, Any]) -> dict[str, Any]:
        import uuid  # noqa: PLC0415

        if self._thread is None or params.get("threadId") != self._thread["id"]:
            raise _Unsupported("a turn on a thread this session did not start")
        self._turn_id = str(uuid.uuid4())
        self._responses = 0
        self._turns.append(
            {"id": self._turn_id, "items": [], "status": "inProgress", "error": None}
        )
        return {
            "turn": {"id": self._turn_id, "items": [], "status": "inProgress", "error": None}
        }

    def _read_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """A client reads back the thread this session holds."""

        if self._thread is None or params.get("threadId") != self._thread["id"]:
            raise _Unsupported("the reading of a thread this session did not start")
        turns = self._turns if params.get("includeTurns") is True else []
        thread = dict(self._thread["described"]["thread"])
        thread["turns"] = [{**turn, "items": list(turn["items"])} for turn in turns]
        return {"thread": thread}

    def _keep(self, frame: dict[str, Any]) -> None:
        """Keep what a notification of the turn in flight says of the thread."""

        if not self._turns:
            return
        params = frame.get("params")
        params = params if isinstance(params, dict) else {}
        if frame.get("method") == "item/completed" and isinstance(params.get("item"), dict):
            self._turns[-1]["items"].append(params["item"])
        elif frame.get("method") == "turn/completed" and isinstance(params.get("turn"), dict):
            self._turns[-1]["status"] = params["turn"].get("status")
            self._turns[-1]["error"] = params["turn"].get("error")

    # -- the model side --------------------------------------------------------

    def _ask_provider(self, body: dict[str, Any]) -> dict[str, Any]:
        import http.client  # noqa: PLC0415 - the probe never needs it
        from urllib.parse import urlsplit  # noqa: PLC0415

        target = urlsplit(str(self._config["provider"]))
        unchecked = SKIP_ADDRESS_CHECK in self._config.get("faults", ())
        host = target.hostname or ""
        if not unchecked and (target.scheme != "http" or host != "127.0.0.1"):
            raise _Unsupported("the scripted provider is not on loopback")
        connection = http.client.HTTPConnection(host, target.port, timeout=30.0)
        try:
            connection.request(
                "POST",
                target.path,
                body=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": CLIENT_NAME},
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        if response.status != 200:
            sys.stderr.write(f"scripted peer: the provider answered HTTP {response.status}\n")
            raise SystemExit(EXIT_PROVIDER_FAILED)
        reply = json.loads(payload)
        if not isinstance(reply, dict):
            raise _Unsupported("the provider's reply is not an object")
        return reply

    def _account(self, frame: dict[str, Any]) -> None:
        """Keep the thread's counters as the usage notification of a response states them.

        Only the thread's total is read. What ``last`` holds beside it (one
        response, or the turn so far) is no business of the peer's.
        """

        if frame.get("method") != "thread/tokenUsage/updated":
            return
        total = frame.get("params", {}).get("tokenUsage", {}).get("total") or {}
        self._spent = {
            "input_tokens": int(total.get("inputTokens", 0) or 0),
            "output_tokens": int(total.get("outputTokens", 0) or 0),
        }

    def _dispatch(self, call: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call of the model on the server that lists the tool."""

        record = {
            "call_id": call["id"],
            "server": call["server"],
            "name": call["name"],
            "arguments": call["arguments"],
        }
        if SKIP_TOOL_SERVER in self._config.get("faults", ()):
            # A lying counterpart for the matrix's own tests: the tool server
            # is never asked.
            content: list[Any] = [{"type": "text", "text": FABRICATED_RESULT}]
            return {**record, "content": content, "is_error": False}
        server = self._servers.get(str(call["server"]))
        if server is None:
            message = f"no tool server {call['server']!r} is configured"
            return {**record, "content": [{"type": "text", "text": message}], "is_error": True}
        content, failed = server.call(str(call["name"]), call["arguments"])
        return {**record, "content": content, "is_error": failed}

    def _play(self) -> None:
        """Answer one turn: model responses and tool calls until the turn ends."""

        assert self._thread is not None
        results: list[dict[str, Any]] = []
        while True:
            self._responses += 1
            reply = self._ask_provider(
                {
                    "pid": os.getpid(),
                    "journal": read_journal(self._config),
                    "command": self._command,
                    "requests": self._requests,
                    "thread_id": self._thread["id"],
                    "turn_id": self._turn_id,
                    "raw_events": self._thread["raw_events"],
                    "response": self._responses,
                    "spent": dict(self._spent),
                    "tools": {name: server.tools for name, server in self._servers.items()},
                    "mcp": [note for server in self._servers.values() for note in server.notes],
                    "tool_results": results,
                }
            )
            if reply.get("unsupported"):
                raise _Unsupported(str(reply["unsupported"]))
            for frame in reply.get("frames", ()):
                self._account(frame)
                self._keep(frame)
                self._emit(frame)
            gone = reply.get("gone")
            if isinstance(gone, dict):
                raise _Gone(str(gone.get("partial", "")), bool(gone.get("killed")))
            call = reply.get("call")
            if call is None:
                return
            results = [self._dispatch(call)]

    def run(self) -> int:
        """Serve the session until the client closes the input."""

        try:
            while True:
                message = self._read()
                method = message.get("method")
                if not isinstance(method, str):
                    continue  # an answer to a request; the peer sends none
                params = message.get("params")
                params = params if isinstance(params, dict) else {}
                notification = "id" not in message
                self._requests.append(
                    {"method": method, "params": params, "notification": notification}
                )
                if notification:
                    continue
                if method == "initialize":
                    self._emit({"id": message["id"], "result": self._initialize(params)})
                elif method == "thread/start":
                    self._emit({"id": message["id"], "result": self._start_thread(params)})
                elif method == "thread/resume":
                    self._emit({"id": message["id"], "result": self._resume_thread(params)})
                elif method == "turn/start":
                    self._emit({"id": message["id"], "result": self._start_turn(params)})
                    self._play()
                elif method == "thread/read":
                    self._emit({"id": message["id"], "result": self._read_thread(params)})
                else:
                    self._refuse(message["id"], method)
        except (_InputClosed, BrokenPipeError):
            return 0


def main(argv: list[str], config: dict[str, Any]) -> int:
    """Play one command line of the tool; return the exit code."""

    try:
        record = started_as(argv, config)
        if argv == VERSION_COMMAND:
            append_journal(config, {**record, "role": "version"})
            sys.stdout.write(str(config["version_line"]) + "\n")
            sys.stdout.flush()
            return 0
        command = parse_command(argv)
        if command["command"][:1] != [APP_SERVER] or command["command"][1:] not in (
            [],
            LISTEN_STDIO,
        ):
            raise _Unsupported(f"the command line {record['argv'][:3]} ...")
        append_journal(config, {**record, "role": "app_server", "login": login_state()})
        return AppServer(command, config, sys.stdin.buffer, sys.stdout.buffer).run()
    except _Gone as gone:
        return go_away(gone, sys.stdout.buffer)
    except _Unsupported as exc:
        append_journal(
            config, {"role": "unsupported", "pid": os.getpid(), "detail": str(exc)}
        )
        sys.stderr.write(f"scripted peer: not played: {exc}\n")
        return EXIT_UNSUPPORTED
    except Exception:  # noqa: BLE001 - the journal is how the cell learns of it
        import traceback  # noqa: PLC0415

        append_journal(
            config,
            {"role": "crashed", "pid": os.getpid(), "detail": traceback.format_exc()},
        )
        sys.stderr.write("scripted peer: crashed; see its journal\n")
        return EXIT_CRASHED
