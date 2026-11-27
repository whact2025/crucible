# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The scripted peer that stands where the Claude adapter looks for its CLI.

The Claude column's driver writes a small launcher per cell and overlays its
path as the route's ``extra.cli_path``. The adapter's own Agent SDK then
starts this program exactly as it starts the vendor tool, three times per
call:

- ``auth status --json`` and ``-v``, the two probes of the session route.
  The answers come from the cell's configuration: the organisation the
  routing file names, and a version the adapter has reviewed for the route's
  model.
- the session itself, with the SDK's stream-json command line. The peer
  speaks the SDK's control protocol on its standard streams and takes every
  model turn from the scripted provider of the cell over loopback HTTP. The
  provider answers with the frames to emit, rendered from envelopes, so no
  reply shape lives in this file.

What the peer does on its own is the tool loop a command-line agent owns: it
asks the SDK-hosted tool server which tools exist, sends each tool call the
model makes to that server through the control channel, feeds the result into
the next turn and adds up the usage of the turns it relayed. It also ends a
session that has used up its turn allowance: the reply to the last allowed
turn carries the frames of the tool's turn terminal, and the peer emits them
after the tool call of that turn in the place of a further model turn, then
exits with the status the reply names. A session whose cost has reached its
budget ends in the same way, at the frames of the tool's budget terminal. A
reply that says ``unsupported`` ends the session with a journal entry, which
fails the cell as a setup error. And it goes away where a reply says so
(``gone``): after the frames of the reply it writes the leading part of a
line the reply names, without a line end, and then either ends its output and
exits with status 0, as a tool whose stream ends without its terminal frame,
or kills itself, as a tool that is gone in the middle of a turn. And it
refuses to continue a session it holds no conversation for: the tool keeps
its conversations in its home, the home of a cell is new, and a command line
that names a session id which no earlier process of the cell opened is
answered with one line on the standard error, no frame and the status of a
failure, before a prompt is read or the model is asked anything. That
behaviour of the tool is emulated, not observed, and neither the line nor
the status is in this file: the peer asks the provider side for them, which
renders them from an envelope that says what it rests on. Continuing a
session that an earlier process of the cell did open is not played.

Every process appends one record to the journal of the cell: its role, its
pids, its command line, the names of its environment and whether it saw a
value the cell forbids (the canary credential of the other routes). The
session sends the journal with each turn, so the provider side can tell how
the peer was started. The journal is a file that outlives the test, and a
machine's environment holds credentials: of the values only the three that
the seam check and the canary proof read are kept, and a forbidden value is
reported as a fact and never written.

Standard library only and no package-relative import: the launcher runs this
file with ``-I -S``, outside the test environment, after it has installed the
network guard of the matrix in the process. The peer is a network client, and
only loopback may be connected to or resolved there, as in the test process.

Protocol sources, all in the pinned ``claude-agent-sdk``:

- command line and environment: ``_internal/transport/subprocess_cli.py``
  (``_build_command``, ``connect``). Every flag that the peer or its dialect
  reads is followed by its value as a token of its own. Three others are
  passed as one ``--flag=value`` token each (``--resume``, ``--session-id``,
  ``--setting-sources``), and :func:`argument` would not find them. Of the
  three the peer reads ``--resume``, in either form
  (:func:`resumed_session`); nothing here reads the two others;
- control frames: ``types.py`` (``SDKControlRequest``, ``SDKControlResponse``,
  ``SDKControlMcpMessageRequest``) and ``_internal/query.py``
  (``_send_control_request``, ``_handle_control_request``);
- the tool server's JSON-RPC methods: ``_internal/query.py``
  (``_handle_sdk_mcp_request``: ``initialize``, ``notifications/initialized``,
  ``tools/list``, ``tools/call``);
- the prompt frame: ``client.py`` (``ClaudeSDKClient.query``).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, BinaryIO

AUTH_STATUS_COMMAND = ["auth", "status", "--json"]
VERSION_COMMAND = ["-v"]
# The protocol revision the SDK's in-process tool server answers with.
MCP_PROTOCOL_VERSION = "2024-11-05"
SKIP_BRIDGE = "skip_bridge"
FABRICATED_RESULT = "a result the peer made up without asking the tool server"
SKIP_ADDRESS_CHECK = "skip_address_check"
SKIP_TURN_TERMINAL = "skip_turn_terminal"
SKIP_BUDGET_TERMINAL = "skip_budget_terminal"
# The only variables the journal keeps by value: how the SDK marks a tool it
# starts, and the home directory the tool was given.
JOURNALLED_ENV = ("CLAUDE_AGENT_SDK_VERSION", "CLAUDE_CODE_ENTRYPOINT", "HOME")
# Where the tool takes a cap on the tokens of a response from: a variable of
# its environment (the vendor's list of the tool's environment variables).
# The journal says what it was started with there, under a key of its own.
RESPONSE_CAP_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
# How a session is asked to continue an earlier one, and where the peer asks
# what the tool says of a session id it holds no conversation for: the
# origin of the scripted provider, this path. The answer is rendered from an
# envelope; what the real tool prints was not observed.
RESUME_FLAG = "--resume"
RESUME_PATH = "/claude-cli/v1/resume"
# A lying counterpart for the matrix's own tests: the same refusal in other
# words. A client that types a refused session by what the tool printed may
# type this one otherwise.
REJECT_IN_OTHER_WORDS = "reject_in_other_words"
OTHER_WORDS = "Error: the conversation {session_id} could not be restored"
MASKED = "<forbidden value>"

EXIT_PROVIDER_FAILED = 1
EXIT_UNSUPPORTED = 3
EXIT_CRASHED = 4


class _InputClosed(Exception):
    """The SDK closed the peer's standard input."""


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


def argument(argv: list[str], flag: str) -> str | None:
    """The value that follows ``flag`` on the command line, if it is there."""

    for index, item in enumerate(argv[:-1]):
        if item == flag:
            return argv[index + 1]
    return None


def resumed_session(argv: list[str]) -> str | None:
    """The session a command line asks to continue: one token with its value, or two."""

    for index, item in enumerate(argv):
        if item.startswith(RESUME_FLAG + "="):
            return item.partition("=")[2]
        if item == RESUME_FLAG and index + 1 < len(argv):
            return argv[index + 1]
    return None


def sdk_servers(argv: list[str]) -> list[str]:
    """The SDK-hosted tool servers the command line configures, by name."""

    raw = argument(argv, "--mcp-config")
    if raw is None:
        return []
    try:
        configured = json.loads(raw)["mcpServers"]
    except (ValueError, KeyError, TypeError) as exc:
        raise _Unsupported(f"--mcp-config is not an inline server mapping: {exc}") from exc
    servers: list[str] = []
    for name, config in configured.items():
        if not isinstance(config, dict) or config.get("type") != "sdk":
            raise _Unsupported(f"tool server {name!r} is not hosted by the SDK")
        servers.append(str(name))
    return servers


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
        "argv": [shown(item) for item in argv],
        "env_names": sorted(shown(name) for name in environment),
        "env": {
            name: shown(environment[name]) for name in JOURNALLED_ENV if name in environment
        },
        "response_cap": (
            shown(environment[RESPONSE_CAP_ENV]) if RESPONSE_CAP_ENV in environment else None
        ),
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


def post_to_provider(
    config: dict[str, Any], body: dict[str, Any], path: str | None = None
) -> tuple[int, bytes]:
    """Send one JSON document to the scripted provider; the status and the answer.

    To the address of the configuration, or to ``path`` at its origin.
    """

    import http.client  # noqa: PLC0415 - the two probes never need it
    from urllib.parse import urlsplit  # noqa: PLC0415

    target = urlsplit(str(config["provider"]))
    # A lying counterpart for the matrix's own tests skips this check.
    # The network guard that the launcher installed is then what stands
    # between the peer and an address that is not loopback.
    unchecked = SKIP_ADDRESS_CHECK in config.get("faults", ())
    host = target.hostname or ""
    if not unchecked and (target.scheme != "http" or host != "127.0.0.1"):
        raise _Unsupported("the scripted provider is not on loopback")
    connection = http.client.HTTPConnection(host, target.port, timeout=30.0)
    try:
        connection.request(
            "POST",
            target.path if path is None else path,
            body=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "crucible-matrix-claude-peer",
            },
        )
        response = connection.getresponse()
        payload = response.read()
    finally:
        connection.close()
    return response.status, payload


def refusal_of(config: dict[str, Any], resumed: str) -> tuple[str, int]:
    """What the tool prints for a session id it holds no conversation for, and its status.

    Asked of the provider side, which renders both from an envelope: no
    sentence of the tool lives in this file.
    """

    status, payload = post_to_provider(
        config, {"pid": os.getpid(), "resumed": resumed}, RESUME_PATH
    )
    said = json.loads(payload) if status == 200 else None
    if (
        not isinstance(said, dict)
        or not isinstance(said.get("stderr"), str)
        or type(said.get("exit")) is not int
    ):
        raise _Unsupported(
            "the provider side does not say what the tool answers a session id it "
            "holds no conversation for"
        )
    if REJECT_IN_OTHER_WORDS in config.get("faults", ()):
        return OTHER_WORDS.format(session_id=resumed), said["exit"]
    return said["stderr"], said["exit"]


def _text_of(content: Any) -> str:
    """The text of a message or tool result: a string, or its text blocks."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


class Session:
    """One stream-json session, driven by the SDK on the standard streams."""

    def __init__(
        self,
        argv: list[str],
        config: dict[str, Any],
        stdin: BinaryIO,
        stdout: BinaryIO,
    ) -> None:
        self._argv = argv
        self._config = config
        self._stdin = stdin
        self._stdout = stdout
        import uuid  # noqa: PLC0415 - the two probes never need it

        self.session_id = str(uuid.uuid4())
        self._control: list[dict[str, Any]] = []
        self._mcp: list[dict[str, Any]] = []
        self._tools: dict[str, list[str]] = {}
        self._backlog: list[dict[str, Any]] = []
        # Control requests and JSON-RPC messages are numbered apart.
        self._requests = 0
        self._messages = 0
        self._turn = 0
        self._spent = {"input_tokens": 0, "output_tokens": 0}

    # -- the standard streams ------------------------------------------------

    def _emit(self, frame: dict[str, Any]) -> None:
        self._stdout.write(json.dumps(frame).encode("utf-8") + b"\n")
        self._stdout.flush()

    def _read(self) -> dict[str, Any]:
        while True:
            line = self._stdin.readline()
            if not line:
                raise _InputClosed
            if line.strip():
                frame = json.loads(line)
                if not isinstance(frame, dict):
                    raise _Unsupported("a frame on standard input is not an object")
                return frame

    # -- the control protocol --------------------------------------------------

    def _respond(self, request_id: Any, *, error: str | None = None) -> None:
        response: dict[str, Any] = {"request_id": request_id}
        if error is None:
            response.update(subtype="success", response={})
        else:
            response.update(subtype="error", error=error)
        self._emit({"type": "control_response", "response": response})

    def _serve(self, frame: dict[str, Any]) -> None:
        """Answer one control request of the SDK."""

        request = frame.get("request")
        if not isinstance(request, dict):
            raise _Unsupported("a control request carries no request object")
        self._control.append(request)
        subtype = request.get("subtype")
        if subtype == "initialize":
            # A session connects its tool servers while it initialises.
            for server in sdk_servers(self._argv):
                self._connect(server)
            self._respond(frame.get("request_id"))
        elif subtype == "set_model":
            self._respond(frame.get("request_id"))
        else:
            self._respond(
                frame.get("request_id"),
                error=f"the scripted peer does not play control request {subtype!r}",
            )
            raise _Unsupported(f"control request {subtype!r}")

    def _ask_sdk(self, server: str, message: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON-RPC message to an SDK-hosted tool server; return its answer."""

        self._requests += 1
        request_id = f"peer_{self._requests}"
        note = {"server": server, "method": message.get("method")}
        if message.get("method") == "tools/call":
            note["name"] = message.get("params", {}).get("name")
        self._mcp.append(note)
        self._emit(
            {
                "type": "control_request",
                "request_id": request_id,
                "request": {
                    "subtype": "mcp_message",
                    "server_name": server,
                    "message": message,
                },
            }
        )
        while True:
            frame = self._read()
            kind = frame.get("type")
            if kind == "control_request":
                self._serve(frame)
            elif kind != "control_response":
                self._backlog.append(frame)
            elif frame.get("response", {}).get("request_id") == request_id:
                response = frame["response"]
                if response.get("subtype") != "success":
                    return {"error": {"message": str(response.get("error", ""))}}
                answer = (response.get("response") or {}).get("mcp_response")
                return answer if isinstance(answer, dict) else {}

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._messages += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._messages, "method": method}
        if params is not None:
            message["params"] = params
        return message

    def _connect(self, server: str) -> None:
        """The tool-server handshake: initialise, confirm, list the tools."""

        self._ask_sdk(
            server,
            self._rpc(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "crucible-matrix-claude-peer",
                        "version": str(self._config["cli_version"]),
                    },
                },
            ),
        )
        self._ask_sdk(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = self._ask_sdk(server, self._rpc("tools/list"))
        tools = listed.get("result", {}).get("tools", [])
        self._tools[server] = [str(tool.get("name", "")) for tool in tools]

    def _dispatch(self, call: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call of the model through the control channel."""

        if SKIP_BRIDGE in self._config.get("faults", ()):
            # A lying counterpart for the matrix's own tests: the tool server
            # is never asked.
            return {
                "tool_use_id": call["id"],
                "content": FABRICATED_RESULT,
                "is_error": False,
            }
        answer = self._ask_sdk(
            str(call["server"]),
            self._rpc(
                "tools/call", {"name": call["name"], "arguments": call["arguments"]}
            ),
        )
        if "error" in answer:
            return {
                "tool_use_id": call["id"],
                "content": str(answer["error"].get("message", "")),
                "is_error": True,
            }
        result = answer.get("result", {})
        return {
            "tool_use_id": call["id"],
            "content": _text_of(result.get("content")),
            "is_error": bool(result.get("isError")),
        }

    # -- the model side --------------------------------------------------------

    def _ask_provider(self, body: dict[str, Any]) -> dict[str, Any]:
        status, payload = post_to_provider(self._config, body)
        if status != 200:
            sys.stderr.write(f"scripted peer: the provider answered HTTP {status}\n")
            raise SystemExit(EXIT_PROVIDER_FAILED)
        reply = json.loads(payload)
        if not isinstance(reply, dict):
            raise _Unsupported("the provider's reply is not an object")
        return reply

    def _account(self, frame: dict[str, Any]) -> None:
        """Add the usage of a model turn the peer relays to the session's total."""

        if frame.get("type") != "assistant":
            return
        usage = frame.get("message", {}).get("usage") or {}
        for name in self._spent:
            self._spent[name] += int(usage.get(name, 0) or 0)

    def _play(self, prompt: dict[str, Any]) -> int:
        """Answer one prompt: model turns and tool calls until the session ends."""

        user = _text_of(prompt.get("message", {}).get("content"))
        results: list[dict[str, Any]] = []
        while True:
            self._turn += 1
            reply = self._ask_provider(
                {
                    "pid": os.getpid(),
                    "journal": read_journal(self._config),
                    "control": self._control,
                    "mcp": self._mcp,
                    "tools": self._tools,
                    "session_id": self.session_id,
                    "turn": self._turn,
                    "spent": dict(self._spent),
                    "user": user,
                    "tool_results": results,
                }
            )
            if reply.get("unsupported"):
                raise _Unsupported(str(reply["unsupported"]))
            for frame in reply.get("frames", ()):
                self._account(frame)
                self._emit(frame)
            gone = reply.get("gone")
            if isinstance(gone, dict):
                raise _Gone(str(gone.get("partial", "")), bool(gone.get("killed")))
            call = reply.get("call")
            if call is None:
                return int(reply.get("exit", 0) or 0)
            results = [self._dispatch(call)]
            terminal = reply.get("turn_limit")
            # A lying counterpart for the matrix's own tests does not stop at
            # the turn allowance: it asks the provider for one more turn.
            if terminal and SKIP_TURN_TERMINAL not in self._config.get("faults", ()):
                # The turn allowance is used up: the tool does not ask the
                # model again. It reports its turn terminal and exits.
                for frame in terminal.get("frames", ()):
                    self._emit(frame)
                return int(terminal.get("exit", 0) or 0)
            spent = reply.get("budget_limit")
            # A lying counterpart for the matrix's own tests does not stop at
            # the budget either: it asks the provider for one more turn.
            if spent and SKIP_BUDGET_TERMINAL not in self._config.get("faults", ()):
                # The cost of the session has reached its budget: the tool
                # does not ask the model again. It reports its budget
                # terminal and exits.
                for frame in spent.get("frames", ()):
                    self._emit(frame)
                return int(spent.get("exit", 0) or 0)

    def run(self) -> int:
        """Serve the session until the SDK closes the input."""

        code = 0
        try:
            while True:
                frame = self._backlog.pop(0) if self._backlog else self._read()
                kind = frame.get("type")
                if kind == "control_request":
                    self._serve(frame)
                elif kind == "user":
                    code = self._play(frame)
                elif kind != "control_response":
                    raise _Unsupported(f"input frame of type {kind!r}")
        except (_InputClosed, BrokenPipeError):
            return code


def main(argv: list[str], config: dict[str, Any]) -> int:
    """Play one command line of the tool; return the exit code."""

    try:
        record = started_as(argv, config)
        if argv == AUTH_STATUS_COMMAND:
            append_journal(config, {**record, "role": "auth_status"})
            sys.stdout.write(json.dumps(config["auth_status"]) + "\n")
            sys.stdout.flush()
            return 0
        if argv == VERSION_COMMAND:
            append_journal(config, {**record, "role": "version"})
            sys.stdout.write(str(config["version_line"]) + "\n")
            sys.stdout.flush()
            return 0
        if (
            argument(argv, "--output-format") != "stream-json"
            or argument(argv, "--input-format") != "stream-json"
        ):
            raise _Unsupported(f"the command line {record['argv'][:3]} ...")
        resumed = resumed_session(argv)
        if resumed is not None:
            # The tool keeps its conversations in its home, and the home of a
            # cell is new: it holds a session only if an earlier process of
            # the cell opened it. Continuing such a session is not played.
            # For any other id the tool finds no conversation: it says so on
            # its standard error and exits before it asks the model anything.
            # What it says is emulated: the provider side renders it.
            if any(item.get("session_id") == resumed for item in read_journal(config)):
                raise _Unsupported(
                    "the resumption of a session that an earlier process of the "
                    "cell opened"
                )
            append_journal(config, {**record, "role": "resume_rejected", "resumed": resumed})
            line, code = refusal_of(config, resumed)
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            return code
        session = Session(argv, config, sys.stdin.buffer, sys.stdout.buffer)
        append_journal(config, {**record, "role": "session", "session_id": session.session_id})
        return session.run()
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
