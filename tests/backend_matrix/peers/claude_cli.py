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
the next turn and adds up the usage of the turns it relayed. It emulates no
turn or budget terminal yet: a reply that says ``unsupported`` ends the
session with a journal entry, which fails the cell as a setup error.

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

Protocol sources, all in the pinned ``claude-agent-sdk`` (0.2.110):

- command line and environment: ``_internal/transport/subprocess_cli.py``
  (``_build_command``, ``connect``);
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
# The only variables the journal keeps by value: how the SDK marks a tool it
# starts, and the home directory the tool was given.
JOURNALLED_ENV = ("CLAUDE_AGENT_SDK_VERSION", "CLAUDE_CODE_ENTRYPOINT", "HOME")
MASKED = "<forbidden value>"

EXIT_PROVIDER_FAILED = 1
EXIT_UNSUPPORTED = 3
EXIT_CRASHED = 4


class _InputClosed(Exception):
    """The SDK closed the peer's standard input."""


class _Unsupported(Exception):
    """The peer was asked for something it does not play."""


def argument(argv: list[str], flag: str) -> str | None:
    """The value that follows ``flag`` on the command line, if it is there."""

    for index, item in enumerate(argv[:-1]):
        if item == flag:
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

        self._session_id = str(uuid.uuid4())
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
        import http.client  # noqa: PLC0415 - the two probes never need it
        from urllib.parse import urlsplit  # noqa: PLC0415

        target = urlsplit(str(self._config["provider"]))
        # A lying counterpart for the matrix's own tests skips this check.
        # The network guard that the launcher installed is then what stands
        # between the peer and an address that is not loopback.
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
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "crucible-matrix-claude-peer",
                },
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        if response.status != 200:
            sys.stderr.write(
                f"scripted peer: the provider answered HTTP {response.status}\n"
            )
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
                    "session_id": self._session_id,
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
            call = reply.get("call")
            if call is None:
                return int(reply.get("exit", 0) or 0)
            results = [self._dispatch(call)]

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
        append_journal(config, {**record, "role": "session"})
        return Session(argv, config, sys.stdin.buffer, sys.stdout.buffer).run()
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
