# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The Claude kind: its cells, and the proof of its driver.

The cells run the starter set on both Claude columns through the real adapter,
its session route and the pinned Agent SDK, with a scripted peer where the
adapter looks for the vendor tool. The remaining tests show that the driver's
seam can fail: a tool that the SDK did not start, an environment that was not
sanitised, a session without the route's probes or without the control
handshake, a peer that answers a tool call without asking the tool server, a
dialect that does not read the model. They also pin the frames to the
installed SDK's parser, show that the session accounting of the peer is judged
against the provider ledger, that a schema answer is the provider's only when
it came through the structured output of the terminal frame, and that what
the peer does not emulate fails a cell as a setup error. The peer is a program
of its own: its journal keeps no value of the environment it was started in,
and the network guard of the package runs inside it.

What the cells of these columns show today is the ledger's business. One test
is required by name by the verifier all the same, because it holds a known
gap of the reference to its cause: on a write the trace binds a
tool-authority digest that is not the shared one, the one code of the cell
does not say why, and the test shows that the value is the digest of the same
tools without a path scope and that nothing else stands behind the code. Its
lying counterparts are beside it. It replaces one private name of the
adapter in this process, which is listed beside the required proofs.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.metadata
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import pytest
from anthropic.types import Message as ProviderMessage
from claude_agent_sdk import types as sdk_types
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk._internal.transport import subprocess_cli as sdk_cli

from assay import Invocation
from crucible.backends.claude import ClaudeBackend
from crucible.backends.config import load_routing
from crucible.backends.tool_authority import (
    APPROVAL_POLICY_PER_CALL_EXACT,
    TOOL_AUTHORITY_BRIDGE_PROTOCOL,
    TOOL_AUTHORITY_BRIDGE_VERSION,
    tool_authority_digest,
)

from . import cells, contract, declarations, envelopes, ledger, netguard, runner, toolsets
from . import columns as matrix_columns
from .cells import Cell
from .columns import CANARY_CREDENTIAL, CLAUDE_KIND, COLUMN_PRODUCT, Column
from .drivers import MatrixSetupError
from .drivers import claude as claude_driver
from .peers import claude_cli as peer_program
from .provider import LoopbackProvider, Reply, SeenRequest
from .scenarios import (
    BASE_POLICY,
    SCENARIOS,
    SCHEMA,
    SCHEMA_ANSWER,
    SCHEMA_KEY,
    SYSTEM,
    TEXT,
    TURN_LIMIT,
    USER,
    suite_default_max_turns,
)
from .script import (
    READ_A,
    READ_B,
    Call,
    Cut,
    ProviderScript,
    Refuse,
    Say,
    ServedAs,
    Step,
    TokenUsage,
)

LEDGER = ledger.load()
KINDS = (CLAUDE_KIND,)
CELLS = cells.cells_of(KINDS)
COLUMNS = tuple(column for column in matrix_columns.real_columns() if column.kind in KINDS)
SDK_DISTRIBUTION = "claude-agent-sdk"
# The product's dispatch bridge logs under its module name.
BRIDGE_LOGGER = "crucible.tools._mcp_bridge"


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)


def _column(role: str = "primary") -> Column:
    return next(item for item in COLUMNS if item.role == role)


def _played(
    column: Column, scenario: str, tmp_path: Path, lens: str = "raw", **more: Any
) -> runner.Execution:
    cell = Cell(column, scenario, lens)
    more.setdefault("differences", LEDGER.differences_for(cell.id))
    return runner.execute(cell, tmp_path, **more)


def _session(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """The journal record of the session process that sent a turn."""

    return next(
        record
        for record in evidence["journal"]
        if record["role"] == "session" and record["pid"] == evidence["pid"]
    )


# -- the seam canary: the real layers ran -----------------------------------------


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_canary_shows_that_the_sdk_started_the_peer_for_the_route(
    column: Column, tmp_path: Path
) -> None:
    execution = _played(column, "S00", tmp_path)
    (request,) = execution.observation.provider.requests
    journal = request.evidence["journal"]
    session = _session(request.evidence)
    argv = session["argv"]

    assert execution.report.seam_problems == () and execution.report.codes == frozenset()
    assert request.matched and request.path == claude_driver.ClaudeCliDialect.path
    # The installed SDK started the tool: its own version in the child's
    # environment, read from the installed distribution, not a typed one.
    assert request.user_agent == (
        f"{SDK_DISTRIBUTION}/{importlib.metadata.version(SDK_DISTRIBUTION)} (sdk-py)"
    )
    # The session route ran its two probes first, and each of the three was a
    # process of its own that this process started.
    assert [record["role"] for record in journal] == ["auth_status", "version", "session"]
    assert [record["argv"] for record in journal[:2]] == [
        peer_program.AUTH_STATUS_COMMAND,
        peer_program.VERSION_COMMAND,
    ]
    pids = [record["pid"] for record in journal]
    assert len(set(pids)) == 3 and os.getpid() not in pids
    assert {record["ppid"] for record in journal} == {os.getpid()}
    # The sanitised transport ran: this process holds the scripted route
    # variables and their credential, and none of them reached the tool.
    assert all(os.environ[name] == value for name, value in matrix_columns.SCRIPTED_ENV.items())
    assert request.credential_ok
    for record in journal:
        assert not set(matrix_columns.SCRIPTED_ENV) & set(record["env_names"])
        assert record["forbidden_seen"] is False
        # Of the values of its environment the journal keeps these three.
        assert set(record["env"]) == set(peer_program.JOURNALLED_ENV)
        assert record["env"]["HOME"] == os.environ["HOME"]
    kept = (tmp_path / "claude-peer" / "journal.jsonl").read_text(encoding="utf-8")
    leaked = CANARY_CREDENTIAL in kept or CANARY_CREDENTIAL in json.dumps(journal)
    assert not leaked
    # The SDK's stream-json command line, for the model of the catalog.
    assert peer_program.argument(argv, "--output-format") == "stream-json"
    assert peer_program.argument(argv, "--input-format") == "stream-json"
    assert peer_program.argument(argv, "--model") == column.model != ""
    assert peer_program.argument(argv, "--max-turns") == str(execution.plan.max_turns)
    assert request.system == peer_program.argument(argv, "--system-prompt") == SYSTEM
    # The control exchange before the prompt: the SDK's initialize request,
    # then the adapter's model preflight for the route's model.
    control = request.evidence["control"]
    assert [item["subtype"] for item in control] == ["initialize", "set_model"]
    assert control[1]["model"] == request.model == column.model
    assert request.user == USER and request.tools == ()
    # What the probes were answered with: the organisation of the routing
    # file and a tool version the adapter has reviewed for the model.
    config = json.loads((tmp_path / "claude-peer" / "config.json").read_text(encoding="utf-8"))
    route = load_routing(config_path=column.routing_path(COLUMN_PRODUCT)).for_role(
        matrix_columns.ROLE_BY_CAPABILITY["text"]
    )
    assert config["auth_status"]["orgId"] == route.extra["expected_org_id"] != ""
    assert config["auth_status"].get("orgName") == route.extra.get("expected_org_name")
    assert config["cli_version"] == claude_driver.reviewed_cli_version(column.model)
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    assert result.execution_trace.served_model == column.model
    # The route names no endpoint, so the trace's is not compared.
    assert execution.reference.endpoint is None


def test_the_route_is_the_shipped_one_with_the_peer_as_its_only_overlay(tmp_path: Path) -> None:
    driver = claude_driver.ClaudeDriver()
    for column in COLUMNS:
        overlay = driver.overlay(column, matrix_columns.SCRIPTED_ENV)
        assert tuple(overlay) == matrix_columns.ALLOWED_OVERLAY[CLAUDE_KIND]
        with driver.open(
            column,
            capability="text",
            lens="raw",
            script=ProviderScript((Say("unused"),)),
            tmp_path=tmp_path / column.role,
        ) as bound:
            backend = bound.backend
            assert (backend.name, backend.model) == (column.adapter, column.model)
            assert backend.features == declarations.routed(column)
            assert bound.tool_path == "control_channel" and bound.children() == ()
            # No process exists before a call: the tool is started per call.
            assert bound.peer.records() == [] and bound.peer.launcher.is_file()
            assert os.access(bound.peer.launcher, os.X_OK)
            if column.role != "primary":
                continue
            # After a call the driver names every process the SDK started for
            # it, the two probes and the session, for the settlement probe.
            result = backend.invoke(
                Invocation(
                    system=SYSTEM,
                    user=USER,
                    tools=None,
                    max_turns=1,
                    budget_usd=None,
                    policy=BASE_POLICY,
                )
            )
            started = bound.children()
            assert result.error is None and result.final_message == "unused"
            assert len(started) == 3 and os.getpid() not in started
            assert started == tuple(sorted(item["pid"] for item in bound.peer.records()))
    with pytest.raises(MatrixSetupError, match="reviewed no command-line tool version"):
        claude_driver.reviewed_cli_version("a-model-nobody-reviewed")
    # The session route refuses to run while this variable of the pinned SDK
    # is set, so a machine that holds it never shows it to a cell.
    assert "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT" in matrix_columns.CLEARED_ENV


def _seam(evidence: Mapping[str, Any], **replaced: Any) -> list[str]:
    dialect = claude_driver.ClaudeCliDialect()
    seen = dialect.parse(
        "POST",
        dialect.path,
        {},
        json.dumps(evidence).encode("utf-8"),
        credential=CANARY_CREDENTIAL,
    )
    return dialect.seam_problems(dataclasses.replace(seen, **replaced))


def test_seam_names_what_is_not_a_tool_the_sdk_started_for_the_route(tmp_path: Path) -> None:
    execution = _played(_column(), "S00", tmp_path)
    healthy = execution.observation.provider.requests[0].evidence
    position = healthy["journal"].index(_session(healthy))

    def changed(edit: Any) -> list[str]:
        evidence = copy.deepcopy(dict(healthy))
        edit(evidence, evidence["journal"][position])
        return _seam(evidence)

    def expect(edit: Any, *named: str) -> None:
        problems = changed(edit)
        assert len(problems) == len(named), problems
        for problem, text in zip(problems, named, strict=True):
            assert text in problem, problems

    def swap_probes(evidence: dict[str, Any], _session: dict[str, Any]) -> None:
        evidence["journal"][:2] = evidence["journal"][1::-1]

    def in_this_process(evidence: dict[str, Any], session: dict[str, Any]) -> None:
        session["pid"] = evidence["pid"] = os.getpid()

    def plain_command_line(_evidence: dict[str, Any], session: dict[str, Any]) -> None:
        session["argv"] = [item for item in session["argv"] if item != "stream-json"]

    assert _seam(healthy) == []
    # Not started by the installed SDK.
    expect(lambda _e, s: s["env"].update(CLAUDE_AGENT_SDK_VERSION="0.0.1"), "installed SDK's")
    expect(lambda _e, s: s["env"].pop("CLAUDE_CODE_ENTRYPOINT"), "installed SDK's")
    # A variable of another route, and a credential under any name, in any
    # process the adapter started: a process that says it saw the credential,
    # one that does not say, and a journal that shows it.
    expect(lambda _e, s: s["env_names"].append("KIMI_BASE_URL"), "scripted route variables")
    expect(lambda _e, s: s.pop("env_names"), "does not name its environment")
    expect(lambda _e, s: s.update(forbidden_seen=True), "scripted credential")
    expect(lambda _e, s: s.pop("forbidden_seen"), "scripted credential")
    expect(lambda _e, s: s["argv"].append(CANARY_CREDENTIAL), "scripted credential")
    expect(lambda e, _s: e["journal"][0].update(forbidden_seen=True), "scripted credential")
    # The probes are part of the session route: one missing, or the other
    # order, and the route did not run as it does.
    expect(lambda e, _s: e["journal"].pop(0), "two probes")
    expect(swap_probes, "two probes")
    # A process of this test, or one that something else started, or none.
    expect(in_this_process, "not a child")
    expect(lambda _e, s: s.update(ppid=1), "not a child")
    expect(lambda e, _s: e.update(pid=-1), "no session process")
    # Not the SDK's command line, not the SDK's control exchange.
    expect(plain_command_line, "stream-json")
    expect(lambda e, _s: e["control"].pop(), "set_model")
    expect(lambda e, _s: e["control"].reverse(), "initialize request")
    (path,) = _seam(healthy, path="/v1/messages")
    assert "request path" in path


class _NoProbes(claude_driver.ClaudeCliDialect):
    """Reads a session as if the route's two probes had not run before it."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        seen = super().parse(*arguments, **named)
        journal = [
            item for item in seen.evidence.get("journal", ()) if item["role"] == "session"
        ]
        return dataclasses.replace(seen, evidence={**seen.evidence, "journal": journal})


class _FixedServedId(claude_driver.ClaudeCliDialect):
    """Answers every turn under one id, whatever model was asked for."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        return dataclasses.replace(super().render(step, seen, call_id), served_model="a-fixed-id")


class _ModelBlind(claude_driver.ClaudeCliDialect):
    """Does not read the model a session runs on."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return dataclasses.replace(super().parse(*arguments, **named), model="")


def _with_dialect(dialect: type[claude_driver.ClaudeCliDialect]) -> claude_driver.ClaudeDriver:
    class Driver(claude_driver.ClaudeDriver):
        def _dialect(self) -> claude_driver.ClaudeCliDialect:
            return dialect()

    return Driver()


@pytest.mark.parametrize(
    ("dialect", "scenario", "problem"),
    [
        (_NoProbes, "S00", "did not follow the route's two probes"),
        (_FixedServedId, "S01", "answered under model 'a-fixed-id'"),
    ],
    ids=["no-probes", "fixed-served-id"],
)
def test_a_broken_seam_fails_the_cell_whatever_it_shows(
    dialect: type[claude_driver.ClaudeCliDialect], scenario: str, problem: str, tmp_path: Path
) -> None:
    with pytest.raises(pytest.fail.Exception, match=problem):
        runner.run_cell(
            Cell(_column(), scenario, "raw"),
            tmp_path,
            preflight={CLAUDE_KIND: None},
            known=LEDGER,
            driver=_with_dialect(dialect),
        )


def test_c7_is_live_on_the_reference_column_through_what_the_dialect_reads(
    tmp_path: Path,
) -> None:
    honest = _played(_column(), "S01", tmp_path / "honest")
    blind = runner.execute(
        Cell(_column(), "S01", "raw"), tmp_path / "blind", driver=_with_dialect(_ModelBlind)
    )

    assert honest.report.outcomes["C7"].status == "ok"
    assert {request.model for request in honest.observation.provider.requests} == {
        honest.reference.model
    }
    assert "request.content:model" in blind.report.codes
    # Clause C9 read what the adapter wrote into its log directory for the
    # call, and what was logged at every level, and found no credential.
    written = [item for item in (tmp_path / "honest" / "backend-logs").rglob("*") if item.is_file()]
    dump = honest.observation.debug_dump_text
    assert written and all(item.read_text(encoding="utf-8") in dump for item in written)
    assert CANARY_CREDENTIAL not in dump and honest.report.outcomes["C9"].status == "ok"


# -- the tool path: the control channel and the real bridge ---------------------------


class _Listener(logging.Handler):
    """Keeps the records of the logger it is attached to."""

    def __init__(self, heard: list[logging.LogRecord]) -> None:
        super().__init__()
        self._heard = heard

    def emit(self, record: logging.LogRecord) -> None:
        self._heard.append(record)


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_tool_calls_travel_the_control_channel_to_the_real_bridge(
    column: Column, tmp_path: Path
) -> None:
    heard: list[logging.LogRecord] = []
    listener = _Listener(heard)
    bridge = logging.getLogger(BRIDGE_LOGGER)
    bridge.addHandler(listener)
    try:
        execution = _played(column, "S03", tmp_path)
    finally:
        bridge.removeHandler(listener)
    requests = execution.observation.provider.requests
    evidence = requests[-1].evidence
    argv = _session(evidence)["argv"]
    (server,) = peer_program.sdk_servers(argv)
    bindings = [toolsets.BINDINGS[slot] for slot in (READ_A, READ_B)]
    listed = evidence["tools"][server]

    assert execution.report.codes == frozenset()
    # The peer has no way to the ToolSet but the SDK: it asked the SDK-hosted
    # server which tools exist, and sent each call of the model as an
    # mcp_message control request.
    assert evidence["mcp"] == [
        {"server": server, "method": "initialize"},
        {"server": server, "method": "notifications/initialized"},
        {"server": server, "method": "tools/list"},
        *({"server": server, "method": "tools/call", "name": item.name} for item in bindings),
    ]
    assert set(listed) == {item.name for item in bindings}
    assert all(request.tools == tuple(listed) for request in requests)
    # The adapter allowed exactly those tools of that server on the command line.
    assert sorted((peer_program.argument(argv, "--allowedTools") or "").split(",")) == sorted(
        f"mcp__{server}__{name}" for name in listed
    )
    # The calls reached the ToolSet in this process with the model's arguments ...
    log = execution.observation.toolset_log
    assert [(item.name, dict(item.arguments)) for item in log] == [
        (item.name, dict(item.arguments)) for item in bindings
    ]
    # ... through the product's dispatch bridge, which logged each of them ...
    assert len(heard) >= len(bindings) and {record.name for record in heard} == {BRIDGE_LOGGER}
    # ... and what the tools returned went back down the channel into the
    # next model turn, unchanged.
    assert [request.tool_results for request in requests] == [
        (),
        (("call-1", log[0].result),),
        (("call-2", log[1].result),),
    ]
    assert execution.report.outcomes["C10"].status == "ok"
    # One session of the tool played all three turns.
    assert len({request.evidence["pid"] for request in requests}) == 1


class _AroundTheBridge(claude_driver.ClaudeDriver):
    """Its peer answers a tool call itself: the tool server is never asked."""

    faults = (peer_program.SKIP_BRIDGE,)


def test_a_peer_that_answers_a_tool_call_without_the_bridge_fails_the_cell(
    tmp_path: Path,
) -> None:
    cell = Cell(_column(), "S03", "raw")
    execution = runner.execute(cell, tmp_path / "played", driver=_AroundTheBridge())
    requests = execution.observation.provider.requests

    # The model turns went by as scripted and got a tool result each ...
    assert execution.report.outcomes["C0"].status == "ok"
    assert [text for request in requests for _id, text in request.tool_results] == [
        peer_program.FABRICATED_RESULT
    ] * 2
    assert all(item["method"] != "tools/call" for item in requests[-1].evidence["mcp"])
    # ... and no call reached the ToolSet: clause C10 says so, exactly.
    assert execution.observation.toolset_log == ()
    assert execution.report.codes == {"dispatch.missing:read_a", "dispatch.missing:read_b"}
    assert execution.report.tripped() == {"C10"}
    with pytest.raises(ledger.LedgerMismatch, match=r"unexpected deviation .*dispatch\.missing"):
        runner.run_cell(
            cell,
            tmp_path / "judged",
            preflight={CLAUDE_KIND: None},
            known=LEDGER,
            driver=_AroundTheBridge(),
        )


# -- the frames: envelopes through the installed SDK -----------------------------------


def _turn(
    *,
    argv: list[str] | None = None,
    turn: int = 1,
    spent: tuple[int, int] = (0, 0),
    tool_results: list[dict[str, Any]] | None = None,
    control: list[dict[str, Any]] | None = None,
) -> SeenRequest:
    """A turn request as the peer of one session sends it, read by the dialect."""

    dialect = claude_driver.ClaudeCliDialect()
    command = ["--model", "asked", "--max-turns", "5", *(argv or [])]
    started = {"argv": command, "env_names": [], "env": {}, "forbidden_seen": False}
    evidence = {
        "pid": 7,
        "journal": [{"role": "session", "pid": 7, "ppid": 1, **started}],
        "control": control or [],
        "mcp": [],
        "tools": {"tools-of-the-test": [toolsets.BINDINGS[READ_A].name]},
        "session_id": "session-1",
        "turn": turn,
        "spent": {"input_tokens": spent[0], "output_tokens": spent[1]},
        "user": "ask",
        "tool_results": tool_results or [],
    }
    return dialect.parse(
        "POST", dialect.path, {"Host": "127.0.0.1:1"}, json.dumps(evidence).encode(), credential=""
    )


def test_frames_parse_through_the_installed_sdk() -> None:
    dialect = claude_driver.ClaudeCliDialect()
    binding = toolsets.BINDINGS[READ_A]
    rate = matrix_columns.matrix_rate("asked")

    called = dialect.render(Call(READ_A, usage=TokenUsage(11, 5)), _turn(), "call-1")
    opened, tool_turn = (parse_message(frame) for frame in called.body["frames"])
    answered = _turn(
        turn=2,
        spent=(11, 5),
        tool_results=[{"tool_use_id": "call-1", "content": "the result", "is_error": False}],
    )
    said = dialect.render(Say("text", TokenUsage(7, 3)), answered, "call-2")
    echoed, final_turn, terminal = (parse_message(frame) for frame in said.body["frames"])

    # A session opens with its id; a tool turn names the tool of the server.
    assert isinstance(opened, sdk_types.SystemMessage) and opened.subtype == "init"
    assert opened.data["session_id"] == "session-1"
    assert isinstance(tool_turn, sdk_types.AssistantMessage)
    assert tool_turn.content == [
        sdk_types.ToolUseBlock(
            id="call-1",
            name=f"mcp__tools-of-the-test__{binding.name}",
            input=dict(binding.arguments),
        )
    ]
    assert (tool_turn.model, tool_turn.session_id) == ("asked", "session-1")
    assert tool_turn.usage is not None
    assert (tool_turn.usage["input_tokens"], tool_turn.usage["output_tokens"]) == (11, 5)
    # The peer is told which call to send down the control channel.
    assert called.body["call"] == {
        "id": "call-1",
        "server": "tools-of-the-test",
        "name": binding.name,
        "arguments": dict(binding.arguments),
    }
    assert called.call == (READ_A, "call-1", dict(binding.arguments))
    assert called.envelopes == ("claude_cli/cli.init", "claude_cli/cli.assistant_tool_use")
    # The next turn shows the tool result, answers, and ends the session.
    assert isinstance(echoed, sdk_types.UserMessage)
    assert echoed.content == [
        sdk_types.ToolResultBlock(
            tool_use_id="call-1",
            content=[{"type": "text", "text": "the result"}],
            is_error=False,
        )
    ]
    assert isinstance(final_turn, sdk_types.AssistantMessage)
    assert final_turn.content == [sdk_types.TextBlock(text="text")]
    assert (final_turn.message_id, final_turn.stop_reason) == ("msg_call-2", "end_turn")
    assert isinstance(terminal, sdk_types.ResultMessage)
    assert (terminal.subtype, terminal.is_error, terminal.result) == ("success", False, "text")
    assert (terminal.num_turns, terminal.stop_reason, terminal.structured_output) == (
        2,
        "end_turn",
        None,
    )
    # The terminal carries the session's totals: both turns, priced at the
    # rate card the adapter is configured with, once per served model.
    cost = rate.compute_cost(input_tokens=18, output_tokens=8)
    assert terminal.usage is not None and terminal.model_usage is not None
    assert (terminal.usage["input_tokens"], terminal.usage["output_tokens"]) == (18, 8)
    assert terminal.total_cost_usd == cost > 0
    assert list(terminal.model_usage) == ["asked"]
    per_model = terminal.model_usage["asked"]
    assert (per_model["inputTokens"], per_model["outputTokens"]) == (18, 8)
    assert per_model["costUSD"] == cost
    assert per_model["canonicalModel"] == "asked"
    assert said.envelopes == (
        "claude_cli/cli.tool_result",
        "claude_cli/cli.assistant_text",
        "claude_cli/cli.result",
    )
    assert said.body["call"] is None and said.body["exit"] == 0 and said.usage == TokenUsage(7, 3)
    # The inner message of a model turn is the provider's message object.
    for frame in (called.body["frames"][1], said.body["frames"][1]):
        message = ProviderMessage.model_validate(frame["message"])
        assert (message.role, message.model, message.type) == ("assistant", "asked", "message")
    # Every frame has an id of its own and names its session.
    frames = [*called.body["frames"], *said.body["frames"]]
    assert len({frame["uuid"] for frame in frames}) == len(frames)
    assert {frame["session_id"] for frame in frames} == {"session-1"}


def test_the_dialect_reads_the_command_line_and_answers_only_what_it_has_an_envelope_for() -> None:
    dialect = claude_driver.ClaudeCliDialect()
    plain = _turn()
    schema = json.dumps(SCHEMA)
    full = _turn(
        argv=["--max-budget-usd", "0.5", "--json-schema", schema, "--system-prompt", "sys"],
        control=[
            {"subtype": "initialize"},
            {"subtype": "set_model", "model": "set-first"},
            {"subtype": "set_model", "model": "set-later"},
        ],
    )

    assert (plain.model, plain.turn_control, plain.budget_control) == ("asked", 5, None)
    assert not plain.schema_field and plain.system == "" and plain.user == "ask"
    assert plain.tools == (toolsets.BINDINGS[READ_A].name,) and plain.host == "127.0.0.1:1"
    # The last model the SDK set is the one the session runs on.
    assert (full.model, full.budget_control, full.system) == ("set-later", 0.5, "sys")
    assert full.schema_field
    # The wire is JSON: a body in another encoding reads as nothing.
    assert dialect.parse("POST", dialect.path, {}, b"model: m", credential="").evidence == {}
    # A session with a schema returns the decoded answer in the structured
    # output of its terminal frame and nowhere else: the text of the last
    # turn and the result text are the tool's note, so a cell can tell which
    # of the three an adapter reads.
    structured = dialect.render(Say(SCHEMA_ANSWER), full, "call-1")
    last_turn, terminal = (parse_message(frame) for frame in structured.body["frames"][-2:])
    assert isinstance(terminal, sdk_types.ResultMessage)
    assert isinstance(last_turn, sdk_types.AssistantMessage)
    assert terminal.structured_output == json.loads(SCHEMA_ANSWER)
    assert terminal.result == claude_driver.SCHEMA_SESSION_TEXT
    assert last_turn.content == [sdk_types.TextBlock(text=claude_driver.SCHEMA_SESSION_TEXT)]
    elsewhere = [dict(frame) for frame in structured.body["frames"]]
    assert elsewhere[-1].pop("structured_output") == json.loads(SCHEMA_ANSWER)
    assert json.loads(SCHEMA_ANSWER)[SCHEMA_KEY] not in json.dumps(elsewhere)
    assert structured.envelopes[-1] == "claude_cli/cli.result_structured"
    # Without a schema the same step is relayed as the text it is.
    relayed = dialect.render(Say(SCHEMA_ANSWER), plain, "call-1").body["frames"]
    assert relayed[-1]["result"] == relayed[-2]["message"]["content"][0]["text"] == SCHEMA_ANSWER
    assert "structured_output" not in relayed[-1]
    # A scripted ServedAs step is the one way to name another model.
    assert dialect.render(Say("x"), plain, "1").served_model == "asked"
    renamed = dialect.render(ServedAs(Say("x")), plain, "1")
    assert renamed.served_model == renamed.body["frames"][1]["message"]["model"] == "asked-served"
    # A step without an envelope is refused, and so is the one form of
    # refusal the frames of the tool cannot carry (it is pinned not
    # expressible in the inventory).
    with pytest.raises(MatrixSetupError, match="does not render Cut yet"):
        dialect.render(Cut(), plain, "1")
    with pytest.raises(MatrixSetupError, match="has no refusal part"):
        dialect.render(Refuse("refusal_part"), plain, "1")
    assert list(claude_driver.ClaudeDriver().inexpressible) == ["S23b/Refuse(refusal_part)"]
    assert set(dialect.envelope_names()) == {
        name for name in envelopes.all_envelopes() if name.startswith(f"{dialect.name}/")
    }


def test_what_the_peer_does_not_emulate_is_refused_not_improvised() -> None:
    dialect = claude_driver.ClaudeCliDialect()
    cases: list[tuple[SeenRequest, Step, str]] = [
        # A turn beyond the turn limit: the tool would have ended the session.
        (_turn(turn=6), Call(READ_A), "--max-turns 5"),
        # A turn after the budget is spent.
        (_turn(argv=["--max-budget-usd", "0.001"], turn=2, spent=(1000, 1000)), Say("x"), "budget"),
        # A schema session whose answer is no JSON object.
        (_turn(argv=["--json-schema", "{}"]), Say("not json"), "schema"),
    ]

    for seen, step, named in cases:
        reply = dialect.render(step, seen, "call-1")
        assert named in reply.body["unsupported"], reply.body
        assert reply.body["frames"] == [] and reply.body["call"] is None
        assert reply.usage is None and reply.envelopes == ()
    # Within the limits the same turns are answered.
    assert "unsupported" not in dialect.render(Call(READ_A), _turn(turn=5), "call-1").body
    within = _turn(argv=["--max-budget-usd", "1"], turn=2, spent=(1000, 1000))
    assert "unsupported" not in dialect.render(Say("x"), within, "call-1").body


def test_the_turn_terminal_and_the_two_refusals_parse_through_the_installed_sdk() -> None:
    dialect = claude_driver.ClaudeCliDialect()
    rate = matrix_columns.matrix_rate("asked")

    # The last allowed turn (the fifth of five) asks for a tool: the reply
    # carries the frames of that turn and, apart from them, the turn terminal
    # the peer plays after the tool call.
    last = dialect.render(
        Call(READ_A, usage=TokenUsage(11, 5)), _turn(turn=5, spent=(40, 20)), "call-5"
    )
    assert [frame["type"] for frame in last.body["frames"]] == ["assistant"]
    assert last.body["turn_limit"]["exit"] == 1
    (limit,) = (parse_message(frame) for frame in last.body["turn_limit"]["frames"])
    assert isinstance(limit, sdk_types.ResultMessage)
    assert (limit.subtype, limit.is_error, limit.result, limit.errors) == (
        "error_max_turns",
        True,
        None,
        None,
    )
    # The stop reason of the last model turn, and one turn more than allowed.
    assert (limit.stop_reason, limit.num_turns) == ("tool_use", 6)
    # The accounting of the session so far, this turn included.
    assert limit.usage is not None and limit.model_usage is not None
    assert (limit.usage["input_tokens"], limit.usage["output_tokens"]) == (51, 25)
    assert limit.total_cost_usd == rate.compute_cost(input_tokens=51, output_tokens=25)
    assert list(limit.model_usage) == ["asked"]
    assert last.envelopes == (
        "claude_cli/cli.assistant_tool_use",
        "claude_cli/cli.result_max_turns",
    )
    # A turn before the last carries no terminal.
    assert "turn_limit" not in dialect.render(Call(READ_A), _turn(turn=4), "call-4").body

    # A turn the safety system of the provider stopped: a billed message
    # without content, then a terminal that names the same stop reason.
    stopped = dialect.render(Refuse("content_filter", TokenUsage(7, 3)), _turn(), "call-1")
    _opened, refused_turn, ended = (parse_message(frame) for frame in stopped.body["frames"])
    assert isinstance(refused_turn, sdk_types.AssistantMessage)
    assert (refused_turn.content, refused_turn.stop_reason) == ([], "refusal")
    message = ProviderMessage.model_validate(stopped.body["frames"][1]["message"])
    assert (message.stop_reason, message.model, message.content) == ("refusal", "asked", [])
    assert isinstance(ended, sdk_types.ResultMessage)
    assert (ended.subtype, ended.is_error, ended.stop_reason, ended.result) == (
        "success",
        False,
        "refusal",
        "",
    )
    assert ended.usage is not None
    assert (ended.usage["input_tokens"], ended.usage["output_tokens"]) == (7, 3)
    assert (stopped.usage, stopped.served_model, stopped.body["exit"]) == (
        TokenUsage(7, 3),
        "asked",
        0,
    )

    # A request the provider rejected: a local message that no model wrote,
    # marked with the error category, then an error terminal with the HTTP
    # status. Nothing is billed, no model is named, and the tool exits with
    # a status other than zero.
    rejected = dialect.render(Refuse("http_policy"), _turn(), "call-1")
    _opened, local, failed = (parse_message(frame) for frame in rejected.body["frames"])
    assert isinstance(local, sdk_types.AssistantMessage)
    assert (local.error, local.model) == ("invalid_request", "<synthetic>")
    assert isinstance(failed, sdk_types.ResultMessage)
    assert (failed.subtype, failed.is_error, failed.api_error_status) == ("success", True, 400)
    # The accounting of the terminal as the tests of the product state it: a
    # usage of zero, a cost of zero, the text of the local message as the
    # result text, and a usage record that names the placeholder model only.
    assert failed.usage == dict.fromkeys(
        (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ),
        0,
    )
    assert (failed.model_usage, failed.total_cost_usd) == ({"<synthetic>": {}}, 0.0)
    assert [sdk_types.TextBlock(text=str(failed.result))] == local.content
    assert failed.stop_reason is None
    assert (rejected.usage, rejected.served_model, rejected.body["exit"]) == (None, None, 1)
    assert rejected.envelopes == (
        "claude_cli/cli.init",
        "claude_cli/cli.assistant_api_error",
        "claude_cli/cli.result_api_error",
    )


class _PastTheTurnLimit(claude_driver.ClaudeDriver):
    """Its peer does not stop at the turn allowance: it asks for one more turn."""

    faults = (peer_program.SKIP_TURN_TERMINAL,)


def _one_turn_session(played_by: claude_driver.ClaudeDriver, tmp_path: Path) -> Any:
    """One turn is allowed, and the model asks for a tool in it."""

    fixture = toolsets.build(toolsets.READ, tmp_path / "workspace")
    opened = played_by.open(
        _column(),
        capability="read_only",
        lens="raw",
        # A second model turn is scripted and must never be asked for.
        script=ProviderScript((Call(READ_A), Say("never reached"))),
        tmp_path=tmp_path,
    )
    with opened as bound:
        bound.backend.attach_toolset(fixture.toolset)
        result = bound.backend.invoke(
            Invocation(
                system=SYSTEM,
                user=USER,
                tools=None,
                max_turns=1,
                budget_usd=None,
                policy=BASE_POLICY,
            )
        )
        seen = bound.provider.freeze()
    return fixture, result, seen


def test_a_session_that_uses_up_its_turns_ends_at_the_turn_terminal_of_the_tool(
    tmp_path: Path,
) -> None:
    fixture, result, seen = _one_turn_session(claude_driver.ClaudeDriver(), tmp_path / "honest")

    # The tool ran the call of its one allowed turn, reported its turn
    # terminal and was not asked for another model turn: the adapter types
    # the limit from that terminal, with the accounting the terminal carries.
    assert [item.name for item in fixture.log] == [toolsets.BINDINGS[READ_A].name]
    assert (seen.request_count, seen.extra, seen.unconsumed) == (1, 0, 1)
    assert seen.envelopes[-1] == "claude_cli/cli.result_max_turns"
    assert result.error is not None and result.error.kind == "max_turns"
    assert result.stop_reason == "max_turns"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert result.completeness.terminal_complete

    # The lying counterpart: a peer that does not stop there asks for a turn
    # beyond the allowance. That is no observation of the adapter: the
    # provider side refuses to answer it, and the cell is a setup error.
    with pytest.raises(MatrixSetupError, match=r"did not play the session.*--max-turns 1"):
        _one_turn_session(_PastTheTurnLimit(), tmp_path / "lying")


class _AnotherTurnCount:
    """The backend of a cell, called with another turn count than the cell asks for."""

    def __init__(self, inner: Any, count: Callable[[int], int]) -> None:
        self._inner, self._count = inner, count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def invoke(self, invocation: Invocation) -> Any:
        return self._inner.invoke(
            dataclasses.replace(invocation, max_turns=self._count(invocation.max_turns))
        )


class _EnforcesAnotherTurnCount(claude_driver.ClaudeDriver):
    """Stands for an adapter that keeps a turn count of its own and says so nowhere."""

    def __init__(self, count: Callable[[int], int]) -> None:
        self._count = count

    @contextlib.contextmanager
    def open(self, column: Column, **named: Any) -> Iterator[Any]:
        with super().open(column, **named) as bound:
            bound.backend = _AnotherTurnCount(bound.backend, self._count)
            yield bound


def test_an_adapter_that_keeps_a_turn_count_of_its_own_fails_the_turn_cells(
    tmp_path: Path,
) -> None:
    def controls(execution: runner.Execution) -> list[int | None]:
        return [request.turn_control for request in execution.observation.provider.requests]

    # The turn limit. Asked for two turns, the adapter hands the count to the
    # tool, the model is asked twice and the limit is typed after the second
    # turn. One turn fewer is a typed limit too, with the same events and
    # flags: only the turn the provider was never asked for shows it.
    honest = _played(_column(), "S10a", tmp_path / "honest")
    fewer = _played(
        _column(),
        "S10a",
        tmp_path / "fewer",
        driver=_EnforcesAnotherTurnCount(lambda asked: asked - 1),
    )
    assert honest.report.codes == frozenset()
    assert controls(honest) == [TURN_LIMIT] * TURN_LIMIT
    assert controls(fewer) == [TURN_LIMIT - 1]
    for execution in (honest, fewer):
        result = execution.observation.result
        assert result is not None and result.error is not None
        assert (result.stop_reason, result.error.kind) == ("max_turns", "max_turns")
    assert fewer.report.codes == {"request.unconsumed"}

    # The suite's turn count. One completion needs one turn, so a ceiling of
    # the adapter's own below the count changes nothing of the result: the
    # turn control on the wire is the one place that shows it.
    asked = suite_default_max_turns()
    plain = _played(_column(), "S11", tmp_path / "plain")
    clamped = _played(
        _column(),
        "S11",
        tmp_path / "clamped",
        driver=_EnforcesAnotherTurnCount(lambda count: min(count, 5)),
    )
    assert plain.report.codes == frozenset() and controls(plain) == [asked]
    assert controls(clamped) == [5] and asked > 5
    assert clamped.report.codes == {"request.content:turn_control"}


def test_an_adapter_that_tells_the_tool_no_turn_count_fails_the_suites_turn_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The route declares that its provider stops a session at the turn count,
    # so the tool is owed the count on its command line. An adapter that keeps
    # the count to itself is played by the SDK's own command line without it:
    # the route takes that command line twice, when it builds its transport
    # and before it starts the tool, and compares the two.
    build = sdk_cli.SubprocessCLITransport._build_command

    def without_turn_count(transport: Any) -> list[str]:
        command = build(transport)
        at = command.index("--max-turns")
        return [*command[:at], *command[at + 2 :]]

    monkeypatch.setattr(sdk_cli.SubprocessCLITransport, "_build_command", without_turn_count)
    untold = _played(_column(), "S11", tmp_path / "untold")

    # One completion under the suite's turn count: the result is the clean
    # finish of the honest play, and only the request shows that the tool was
    # told no count.
    result = untold.observation.result
    assert result is not None and result.error is None
    assert (result.stop_reason, result.final_message) == ("end_turn", TEXT)
    assert [request.turn_control for request in untold.observation.provider.requests] == [None]
    assert untold.report.codes == {"request.content:turn_control"}
    # The count is owed because of what the route declares and because the
    # command line of the tool has a place for it.
    assert SCENARIOS["S11"].governing == ("turn_limit",)
    assert untold.current == {"turn_limit": "provider_stop"}
    assert untold.observation.provider.turn_control_field


# -- the peer as a program ---------------------------------------------------------------

_SESSION_ARGV = [
    "--output-format", "stream-json", "--verbose", "--system-prompt", "sys",
    "--max-turns", "2", "--model", "asked", "--input-format", "stream-json",
]  # fmt: skip


def _control(request_id: str, **request: Any) -> dict[str, Any]:
    return {"type": "control_request", "request_id": request_id, "request": request}


# What the SDK writes to the tool before and with the first prompt.
_PROMPT: list[dict[str, Any]] = [
    _control("r1", subtype="initialize", hooks=None),
    _control("r2", subtype="set_model", model="asked"),
    {
        "type": "user",
        "message": {"role": "user", "content": "ask"},
        "parent_tool_use_id": None,
        "session_id": "",
    },
]


def _peer(
    directory: Path,
    provider: str = "http://127.0.0.1:9/none",
    faults: tuple[str, ...] = (),
) -> claude_driver.Peer:
    return claude_driver.write_peer(
        directory,
        provider=provider,
        organisation={"orgId": "an-organisation", "orgName": "An Organisation"},
        cli_version="9.8.7",
        faults=faults,
    )


def _run(
    peer: claude_driver.Peer,
    argv: list[str],
    frames: list[dict[str, Any]] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(peer.launcher), *argv],
        input="".join(json.dumps(frame) + "\n" for frame in frames or ()).encode("utf-8"),
        capture_output=True,
        timeout=60,
        check=False,
        # The launcher names its interpreter by an absolute path and needs
        # nothing of the environment: a peer started by hand gets what the
        # test names and nothing of this process or of the machine.
        env=dict(env or {}),
    )


def test_the_peer_answers_the_two_probes_and_refuses_any_other_command(tmp_path: Path) -> None:
    peer = _peer(tmp_path)
    status = _run(peer, peer_program.AUTH_STATUS_COMMAND)
    version = _run(peer, peer_program.VERSION_COMMAND)

    assert (status.returncode, status.stderr) == (0, b"")
    assert json.loads(status.stdout) == {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "orgId": "an-organisation",
        "orgName": "An Organisation",
    }
    assert (version.returncode, version.stderr) == (0, b"")
    assert version.stdout == b"9.8.7 (Claude Code)\n"
    assert [record["role"] for record in peer.records()] == ["auth_status", "version"]
    assert peer.failures() == [] and len(peer.pids()) == 2
    # Started by hand, the peer got nothing of this process's environment,
    # which holds the scripted route variables and their credential.
    assert all(os.environ[name] == value for name, value in matrix_columns.SCRIPTED_ENV.items())
    for record in peer.records():
        assert not set(matrix_columns.SCRIPTED_ENV) & set(record["env_names"])
        assert record["forbidden_seen"] is False
    # Anything else is not played: the journal says so, and a cell that
    # reached it ends as a setup error.
    helped = _run(peer, ["--help"])
    assert helped.returncode == peer_program.EXIT_UNSUPPORTED and b"not played" in helped.stderr
    (failure,) = peer.failures()
    assert failure.startswith("unsupported: the command line ['--help']")
    # A session whose input ends before a prompt ends quietly.
    quiet = _run(_peer(tmp_path / "quiet"), _SESSION_ARGV, _PROMPT[:2])
    assert (quiet.returncode, quiet.stderr) == (0, b"")
    answers = [json.loads(line)["type"] for line in quiet.stdout.splitlines()]
    assert answers == ["control_response"] * 2


def test_the_peer_takes_its_turns_from_the_scripted_provider_on_loopback_only(
    tmp_path: Path,
) -> None:
    dialect = claude_driver.ClaudeCliDialect()
    with LoopbackProvider(
        ProviderScript((Say("the answer"),)), credential=CANARY_CREDENTIAL, dialect=dialect
    ) as provider:
        peer = _peer(tmp_path / "scripted", provider.url + dialect.path)
        played = _run(peer, _SESSION_ARGV, _PROMPT)
        again = _run(peer, _SESSION_ARGV, _PROMPT)
        shown = provider.freeze()
    frames = [json.loads(line) for line in played.stdout.splitlines()]

    # Both control requests are answered, then the frames of the one turn.
    assert (played.returncode, played.stderr) == (0, b"")
    assert [frame["type"] for frame in frames] == [
        "control_response", "control_response", "system", "assistant", "result",
    ]  # fmt: skip
    assert [frame["response"] for frame in frames[:2]] == [
        {"subtype": "success", "request_id": "r1", "response": {}},
        {"subtype": "success", "request_id": "r2", "response": {}},
    ]
    assert frames[-1]["result"] == "the answer"
    assert frames[2]["session_id"] == frames[-1]["session_id"] != ""
    # A second session has no scripted turn left: it fails as a tool whose
    # provider failed, and that is an extra request, not a failure of the peer.
    assert again.returncode == peer_program.EXIT_PROVIDER_FAILED and b"HTTP 409" in again.stderr
    assert (shown.request_count, shown.extra, peer.failures()) == (2, 1, [])
    # Started by hand, the peer is visibly not a tool the SDK started.
    assert any("installed SDK's" in problem for problem in shown.seam_problems)
    assert any("two probes" in problem for problem in shown.seam_problems)
    # A provider that is not on loopback is never contacted.
    away = _peer(tmp_path / "away", "http://provider.invalid/turn")
    refused = _run(away, _SESSION_ARGV, _PROMPT)
    assert refused.returncode == peer_program.EXIT_UNSUPPORTED
    assert away.failures() == ["unsupported: the scripted provider is not on loopback"]
    # A control request the peer does not play is answered with an error and
    # ends the session as unsupported.
    odd = _peer(tmp_path / "odd")
    interrupted = _run(odd, _SESSION_ARGV, [_control("r9", subtype="interrupt")])
    (answer,) = [json.loads(line) for line in interrupted.stdout.splitlines()]
    assert answer["response"]["subtype"] == "error" and answer["response"]["request_id"] == "r9"
    assert interrupted.returncode == peer_program.EXIT_UNSUPPORTED and len(odd.failures()) == 1


def _answer(request: int, answer: dict[str, Any]) -> dict[str, Any]:
    """What the SDK writes back for the peer's control request number ``request``."""

    return {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": f"peer_{request}",
            "response": {"mcp_response": answer},
        },
    }


def test_the_peer_passes_a_tool_error_on_to_the_next_model_turn(tmp_path: Path) -> None:
    dialect = claude_driver.ClaudeCliDialect()
    tool = toolsets.BINDINGS[READ_A].name
    servers = json.dumps({"mcpServers": {"hosted": {"type": "sdk", "name": "hosted"}}})
    # What an SDK that hosts one tool server writes, in the order the peer
    # reads it: the handshake's three answers while the session initialises,
    # then the prompt, then the answer to each of the model's two tool calls.
    # The first is a result the server flags as an error, the second a
    # JSON-RPC error.
    frames = [
        _PROMPT[0],
        _answer(1, {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {"tools": {}}}}),
        _answer(2, {"jsonrpc": "2.0", "result": {}}),
        _answer(3, {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": tool}]}}),
        *_PROMPT[1:],
        _answer(
            4,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"content": [{"type": "text", "text": "it failed"}], "isError": True},
            },
        ),
        _answer(5, {"jsonrpc": "2.0", "id": 4, "error": {"code": -32601, "message": "no such"}}),
    ]
    script = ProviderScript((Call(READ_A), Call(READ_A), Say("done")))
    with LoopbackProvider(script, credential=CANARY_CREDENTIAL, dialect=dialect) as provider:
        peer = _peer(tmp_path, provider.url + dialect.path)
        played = _run(peer, ["--mcp-config", servers, *_SESSION_ARGV[:6], "--max-turns", "5",
                             *_SESSION_ARGV[8:]], frames)  # fmt: skip
        shown = provider.freeze()
    written = [json.loads(line) for line in played.stdout.splitlines()]
    sent = [frame["request"] for frame in written if frame["type"] == "control_request"]
    echoed = [
        frame["message"]["content"][0] for frame in written if frame["type"] == "user"
    ]

    assert (played.returncode, played.stderr, peer.failures()) == (0, b"", [])
    # The peer asked the hosted server as a tool client does, numbering its
    # control requests and its JSON-RPC messages apart.
    assert [frame["request_id"] for frame in written if frame["type"] == "control_request"] == [
        f"peer_{number}" for number in range(1, 6)
    ]
    assert {item["subtype"] for item in sent} == {"mcp_message"}
    assert {item["server_name"] for item in sent} == {"hosted"}
    assert [(item["message"]["method"], item["message"].get("id")) for item in sent] == [
        ("initialize", 1),
        ("notifications/initialized", None),
        ("tools/list", 2),
        ("tools/call", 3),
        ("tools/call", 4),
    ]
    binding = toolsets.BINDINGS[READ_A]
    assert sent[3]["message"]["params"] == {"name": tool, "arguments": dict(binding.arguments)}
    # Both failures reached the next model turn as failed tool results, and
    # the frame that shows each says so.
    assert [request.tool_results for request in shown.requests] == [
        (),
        (("call-1", "it failed"),),
        (("call-2", "no such"),),
    ]
    assert [(block["tool_use_id"], block["is_error"]) for block in echoed] == [
        ("call-1", True),
        ("call-2", True),
    ]
    assert [block["content"][0]["text"] for block in echoed] == ["it failed", "no such"]
    assert (shown.extra, shown.unconsumed) == (0, 0) and written[-1]["result"] == "done"
    assert shown.requests[0].tools == (tool,)


def test_the_journal_names_the_environment_and_keeps_none_of_its_values(tmp_path: Path) -> None:
    started_with = {
        "CLAUDE_AGENT_SDK_VERSION": "0.0.1",
        "CLAUDE_CODE_ENTRYPOINT": "by-hand",
        "HOME": str(tmp_path / "a-home"),
        "MATRIX_PLANTED_NAME": "a-planted-value",
    }
    peer = _peer(tmp_path / "plain")
    _run(peer, peer_program.VERSION_COMMAND, env=started_with)
    written = peer.journal.read_text(encoding="utf-8")
    (record,) = peer.records()

    # The journal is a file that outlives the test. Whatever the machine holds
    # in its environment stays out of it: every variable is named, and only
    # the three the seam check and the canary proof read are kept by value.
    assert "a-planted-value" not in written
    # (The interpreter may add a locale variable of its own to what it was
    # started with; the journal names what the process holds.)
    names = record["env_names"]
    assert names == sorted(names) and set(started_with) <= set(names)
    assert record["env"] == {name: started_with[name] for name in peer_program.JOURNALLED_ENV}
    assert record["forbidden_seen"] is False
    # What a cell forbids its tool to see, the canary credential, is in the
    # peer's configuration. The peer looks for it wherever it was started
    # with something, and reports a fact: the word itself is never written.
    config = json.loads(peer.config.read_text(encoding="utf-8"))
    assert config["forbidden"] == [CANARY_CREDENTIAL]
    cases: list[tuple[str, dict[str, str], list[str]]] = [
        ("a value", {**started_with, "OTHER_NAME": CANARY_CREDENTIAL}, []),
        ("a kept value", {**started_with, "HOME": str(tmp_path / CANARY_CREDENTIAL)}, []),
        ("a name", {**started_with, CANARY_CREDENTIAL: "x"}, []),
        ("the command line", started_with, ["--append-system-prompt", CANARY_CREDENTIAL]),
    ]
    for name, environment, more in cases:
        tainted = _peer(tmp_path / name.replace(" ", "-"))
        played = _run(tainted, [*_SESSION_ARGV, *more], _PROMPT[:2], env=environment)
        (session,) = tainted.records()
        leaked = CANARY_CREDENTIAL in tainted.journal.read_text(encoding="utf-8")
        assert (played.returncode, session["role"], leaked) == (0, "session", False), name
        assert session["forbidden_seen"] is True, name
        assert session["argv"][: len(_SESSION_ARGV)] == _SESSION_ARGV, name
        if more:
            assert session["argv"][-1] == peer_program.MASKED
    # The dialect takes the peer's word for it: a process that saw the word,
    # or does not say, is a tool whose environment was not sanitised.
    dialect = claude_driver.ClaudeCliDialect()

    def credential_ok(journal: list[Mapping[str, Any]]) -> bool:
        body = json.dumps({"pid": journal[-1]["pid"], "journal": journal}).encode("utf-8")
        return dialect.parse(
            "POST", dialect.path, {}, body, credential=CANARY_CREDENTIAL
        ).credential_ok

    assert credential_ok([record]) and not credential_ok([record, session])
    assert not credential_ok([{name: record[name] for name in record if name != "forbidden_seen"}])


def test_the_network_guard_is_installed_in_every_process_of_the_peer(tmp_path: Path) -> None:
    # The peer's own check of its provider's address is one comparison. A
    # peer that skips it, told of a provider elsewhere, is stopped by the
    # package's network guard, which its launcher installs before the peer
    # runs: no connection is made and no name resolved.
    unchecked = _peer(
        tmp_path, "http://provider.invalid/turn", faults=(peer_program.SKIP_ADDRESS_CHECK,)
    )
    stopped = _run(unchecked, _SESSION_ARGV, _PROMPT)
    answers = [json.loads(line)["type"] for line in stopped.stdout.splitlines()]
    (crash,) = unchecked.failures()

    assert answers == ["control_response"] * 2
    assert stopped.returncode == peer_program.EXIT_CRASHED
    # The guard's exception is no transport error, so the peer does not take
    # it for a provider that failed: it is a crash of the stand-in.
    assert crash.startswith("crashed: ") and "NetworkGuardViolation" in crash
    assert "create_connection -> ('provider.invalid', 80)" in crash
    assert str(Path(netguard.__file__).resolve()) in unchecked.launcher.read_text(encoding="utf-8")
    # The probes run under the same guard and need no network.
    assert _run(unchecked, peer_program.VERSION_COMMAND).returncode == 0


def test_a_peer_stopped_by_the_guard_ends_the_cell_as_a_setup_error(tmp_path: Path) -> None:
    opened = claude_driver.ClaudeDriver().open(
        _column(),
        capability="text",
        lens="raw",
        script=ProviderScript((Say("never asked for"),)),
        tmp_path=tmp_path,
    )
    seen: list[LoopbackProvider] = []

    # The real route, the real adapter and SDK, and a peer that was told of a
    # provider that is not on loopback and not to check the address itself.
    # Whatever the adapter makes of the dead session is not evidence.
    stopped = pytest.raises(
        MatrixSetupError, match=r"(?s)did not play the session: crashed: .*NetworkGuardViolation"
    )
    with stopped, opened as bound:
        seen.append(bound.provider)
        config = json.loads(bound.peer.config.read_text(encoding="utf-8"))
        config.update(
            provider="http://provider.invalid/turn", faults=[peer_program.SKIP_ADDRESS_CHECK]
        )
        bound.peer.config.write_text(json.dumps(config), encoding="utf-8")
        bound.backend.invoke(
            Invocation(
                system=SYSTEM,
                user=USER,
                tools=None,
                max_turns=1,
                budget_usd=None,
                policy=BASE_POLICY,
            )
        )
    # Nothing reached the scripted provider either: the peer never got as far.
    assert seen[0].freeze().request_count == 0


# -- envelopes, grades and the session accounting ------------------------------------------


def test_every_frame_of_a_cell_counts_towards_its_grade(tmp_path: Path) -> None:
    column = _column()
    played = {
        scenario: _played(column, scenario, tmp_path / scenario)
        for scenario in ("S01", "S03", "S04")
    }
    used = {
        scenario: set(execution.observation.provider.envelopes)
        for scenario, execution in played.items()
    }

    def named(*behaviours: str) -> set[str]:
        return {f"claude_cli/cli.{behaviour}" for behaviour in behaviours}

    # A reply holds several frames, and each is an envelope of its own.
    assert used["S01"] == named("init", "assistant_text", "result")
    tool_turn = ("init", "assistant_tool_use", "tool_result", "assistant_text")
    assert used["S03"] == named(*tool_turn, "result")
    assert used["S04"] == named(*tool_turn, "result_structured")
    # Nothing was observed of the real tool: every frame is constructed, and
    # so is every cell of the kind.
    assert {execution.grade for execution in played.values()} == {"constructed"}
    assert {envelopes.load(name).grade for name in claude_driver.ClaudeDriver().envelopes} == {
        "constructed"
    }
    # The starter cells use the frames of a session that ends on an answer;
    # the turn terminal of the tool and the frames of the two refusals are
    # used by the limit and refusal cells alone.
    assert set(claude_driver.ClaudeDriver().envelopes) - set().union(*used.values()) == named(
        "result_max_turns",
        "assistant_refusal",
        "result_refusal",
        "assistant_api_error",
        "result_api_error",
    )
    # The schema travels on the command line of every turn's session, and the
    # decoded answer comes back as the final message. Only the structured
    # output of the terminal frame carried it: both texts were the tool's note.
    schema_cell = played["S04"]
    requests = schema_cell.observation.provider.requests
    argv = _session(requests[0].evidence)["argv"]
    result = schema_cell.observation.result
    assert all(request.schema_field for request in requests)
    assert json.loads(peer_program.argument(argv, "--json-schema") or "") == dict(SCHEMA)
    assert result is not None and json.loads(result.final_message) == json.loads(SCHEMA_ANSWER)
    assert SCENARIOS["S04"].governing == ("structured_output",)
    assert schema_cell.current == {"structured_output": "native"}


class _UnderBilled(claude_driver.ClaudeCliDialect):
    """Its terminal frame reports fewer input tokens than the turns it relayed."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        for frame in reply.body["frames"]:
            if frame["type"] == "result":
                frame["usage"]["input_tokens"] -= 10
                for record in frame["modelUsage"].values():
                    record["inputTokens"] -= 10
        return reply


def test_the_peers_session_accounting_is_judged_against_the_provider_ledger(
    tmp_path: Path,
) -> None:
    honest = _played(_column(), "S03", tmp_path / "honest")
    short = runner.execute(
        Cell(_column(), "S03", "raw"), tmp_path / "short", driver=_with_dialect(_UnderBilled)
    )
    result = honest.observation.result
    billed = honest.observation.provider.billed_total()

    # The adapter takes the totals from the tool's terminal frame; clause C6
    # compares them with what the scripted provider billed turn by turn.
    assert result is not None and honest.report.outcomes["C6"].status == "ok"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (
        billed.input_tokens,
        billed.output_tokens,
    )
    assert len(honest.observation.provider.billed) == 3
    assert "usage.input_tokens:lt" in short.report.codes and "C6" in short.report.tripped()


class _WithoutStructuredOutput(claude_driver.ClaudeCliDialect):
    """Its terminal frame of a schema session carries no structured output."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        for frame in reply.body["frames"]:
            if frame["type"] == "result":
                frame.pop("structured_output", None)
        return reply


class _OtherStructuredOutput(claude_driver.ClaudeCliDialect):
    """Its terminal frame carries another object of the schema than the model answered."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        for frame in reply.body["frames"]:
            if frame["type"] == "result" and "structured_output" in frame:
                frame["structured_output"] = {
                    key: "not what the provider said" for key in frame["structured_output"]
                }
        return reply


@pytest.mark.parametrize(
    "dialect", [_WithoutStructuredOutput, _OtherStructuredOutput], ids=["dropped", "replaced"]
)
def test_a_schema_answer_that_is_not_the_one_the_provider_gave_fails_the_cell(
    dialect: type[claude_driver.ClaudeCliDialect], tmp_path: Path
) -> None:
    cell = Cell(_column(), "S04", "raw")
    execution = runner.execute(cell, tmp_path / "played", driver=_with_dialect(dialect))
    result = execution.observation.result

    # The exchange went by as scripted and the adapter returned a result ...
    assert execution.report.outcomes["C0"].status == "ok"
    assert result is not None and result.error is None
    # ... whose final message is not the answer the provider gave. Without
    # the structured output the adapter is left with the text of the tool,
    # which is not the answer; with another object in it the adapter returns
    # an answer of the right shape that the provider never sent. The scenario
    # says so in both cases, and no clause says anything else.
    assert json.loads(SCHEMA_ANSWER) != _decoded(result.final_message)
    assert execution.report.codes == {"text:mismatch"}
    assert execution.report.tripped() == {"S"}
    with pytest.raises(ledger.LedgerMismatch, match=r"unexpected deviation .*text:mismatch"):
        runner.run_cell(
            cell,
            tmp_path / "judged",
            preflight={CLAUDE_KIND: None},
            known=LEDGER,
            driver=_with_dialect(dialect),
        )


def _decoded(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# -- the digest a write binds: a known gap held to its cause -------------------------------

WRITE_DIGEST_CODE = "digest.mismatch:tool_authority"
# The record of the ledger whose cells the proof below plays.
WRITE_DIGEST_GAP = "R13"
PROOF_MODULE = cells.MODULE_BY_KIND[CLAUDE_KIND]


def _write_cell(column: Column) -> Cell:
    """The cell of a Claude column on which the proof of the write digest plays."""

    (listed,) = (
        item
        for item in cells.WRITES_ON_THE_CLAUDE_COLUMNS
        if ledger.parse_cell_id(item)[2] == column.role
    )
    _kind, scenario, _role, lens = ledger.parse_cell_id(listed)
    return Cell(column, scenario, lens)


def _trace_digest(execution: runner.Execution) -> str | None:
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    return result.execution_trace.tool_authority_digest


def _unscoped_digest(execution: runner.Execution, workspace: Path) -> str:
    """The shared digest of the tools a played cell attached, with no path scope stated."""

    scenario = SCENARIOS[execution.cell.scenario]
    assert scenario.toolset is not None
    twin = toolsets.build(scenario.toolset, workspace, scoped=False)
    return runner.shared_tool_authority_digest(
        twin.toolset,
        capability=scenario.capability,
        context_budget=execution.plan.context_budget,
    )


def _not_the_digest_without_a_path_scope(execution: runner.Execution, unscoped: str) -> list[str]:
    """The first half: why the digest of a played write is not the one of the known gap."""

    problems: list[str] = []
    carried = _trace_digest(execution)
    if carried is None:
        problems.append("the trace binds no tool-authority digest")
    if carried == execution.reference.tool_authority_digest:
        problems.append("the trace binds the shared digest of the attached ToolSet")
    if carried != unscoped:
        problems.append(
            "the digest of the trace is not the shared digest of the same tools "
            "with no path scope stated"
        )
    return problems


def _more_than_the_one_cause(
    as_is: runner.Execution, removed: runner.Execution, met: list[str]
) -> list[str]:
    """The second half: why the write shows more than the one cause of the known gap.

    ``removed`` is the same cell played with the adapter's hash bound to the
    ToolSet that was attached; ``met`` names what the adapter handed to its
    hash instead, once for every time it did.
    """

    problems: list[str] = []
    if not met:
        problems.append("the adapter handed the attached ToolSet to its hash")
    if removed.report.codes:
        problems.append(
            "with the hash bound to the attached ToolSet the cell still shows "
            f"{sorted(removed.report.codes)}"
        )
    if _trace_digest(removed) != removed.reference.tool_authority_digest:
        problems.append(
            "with the hash bound to the attached ToolSet the trace does not bind "
            "the shared digest"
        )
    before, after = _statuses(as_is), _statuses(removed)
    moved = sorted(name for name in {*before, *after} if before.get(name) != after.get(name))
    if moved != [contract.clause_of(WRITE_DIGEST_CODE)]:
        problems.append(
            f"the outcome of the clauses {moved} changed with the cause, not that "
            "of the clause that compares the digest alone"
        )
    return problems


def _statuses(execution: runner.Execution) -> dict[str, str]:
    return {name: str(item.status) for name, item in execution.report.outcomes.items()}


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_the_trace_of_a_write_binds_the_digest_of_the_same_tools_without_a_path_scope(
    column: Column, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """A known gap of the reference, held to its one cause from both sides.

    On a write the adapter hands its hash the wrapper that routes the
    mutators to their isolated process, where it attached a ToolSet. The
    cell then shows one code, which says that the digest of the trace is not
    the shared one and nothing about why. This test says why, by what the
    cell shows and without naming the wrapper: the digest is the shared one
    of the same tools with no path scope stated; and with the hash bound to
    the ToolSet that was attached, and nothing else of the adapter changed,
    the cell shows no code at all. The ledger carries the code as a known
    gap (R13) on the cells this test plays and on no other.
    """

    # The verifier requires this test by name on each column: the entries of
    # the record lean on it, and a run without it is refused. No entry
    # attributes a code to that record on a cell this test does not play.
    assert request.node.name in cells.REQUIRED_PROOFS[PROOF_MODULE]
    cell = _write_cell(column)
    attributed = {
        entry.cell for entry in LEDGER.entries if WRITE_DIGEST_GAP in entry.codes.values()
    }
    assert attributed <= set(cells.WRITES_ON_THE_CLAUDE_COLUMNS)
    assert cell.id in cells.WRITES_ON_THE_CLAUDE_COLUMNS

    as_is = _played(column, cell.scenario, tmp_path / "as-is", cell.lens)

    # The one cause, removed: the adapter's hash bound to the ToolSet that was
    # attached, whatever the adapter hands it. The one private name of the
    # product this replaces is listed beside the required proofs.
    attached: dict[int, Any] = {}
    met: list[str] = []
    attach, digest = ClaudeBackend.attach_toolset, ClaudeBackend._tool_authority_digest

    def remembering(self: Any, toolset: Any, **more: Any) -> None:
        attached[id(self)] = toolset
        attach(self, toolset, **more)

    def bound_to_what_was_attached(
        self: Any, toolset: Any, *, context_budget: int | None
    ) -> str:
        if toolset is not attached[id(self)]:
            met.append(type(toolset).__name__)
        return digest(self, attached[id(self)], context_budget=context_budget)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(ClaudeBackend, "attach_toolset", remembering)
        patched.setattr(ClaudeBackend, "_tool_authority_digest", bound_to_what_was_attached)
        removed = _played(column, cell.scenario, tmp_path / "cause-removed", cell.lens)
    assert f"{ClaudeBackend.__module__}.{ClaudeBackend.__qualname__}._tool_authority_digest" in (
        cells.PROOF_SEAMS[PROOF_MODULE]
    )

    # The cell as the ledger has it: a clean result of a write that is on
    # disk, one code, no governing declared value.
    for execution in (as_is, removed):
        assert execution.report.seam_problems == ()
        result = execution.observation.result
        assert result is not None and result.error is None
        assert execution.observation.write_applied is True
    assert as_is.report.codes == {WRITE_DIGEST_CODE} and dict(as_is.current) == {}
    # The first half: which digest the trace binds.
    unscoped = _unscoped_digest(as_is, tmp_path / "twin")
    assert unscoped != as_is.reference.tool_authority_digest
    assert _not_the_digest_without_a_path_scope(as_is, unscoped) == []
    # The second half: with that one cause removed nothing is left, and no
    # clause but the one that compares the digest changed its outcome.
    assert _more_than_the_one_cause(as_is, removed, met) == []


def _hashing(change: Callable[..., str]) -> Callable[..., str]:
    """A hash for the Claude adapter that ``change`` computes from what it is handed."""

    def replaced(self: Any, toolset: Any, *, context_budget: int | None) -> str:
        return change(self, toolset, context_budget)

    return replaced


def test_the_first_half_of_the_write_proof_fails_on_a_digest_that_is_wrong_for_another_reason(
    tmp_path: Path,
) -> None:
    # The lying counterparts of the first half: the trace of the same write
    # binds a digest that is not the shared one for another reason than the
    # known gap names: the adapter hashes what it is handed under another
    # context budget, or under another version of the bridge. The cell shows
    # the same one code either way, so the entry alone would call both the
    # known gap; the first half does not. (Another capability cannot be
    # played on a write: the shared function refuses a writer under every
    # capability but the one of the cell.)
    column = _column()
    cell = _write_cell(column)
    digest = ClaudeBackend._tool_authority_digest

    def another_budget(self: Any, toolset: Any, context_budget: int | None) -> str:
        return digest(self, toolset, context_budget=(context_budget or 0) + 1)

    def another_bridge(self: Any, toolset: Any, context_budget: int | None) -> str:
        return tool_authority_digest(
            toolset,
            capability=self.capability,
            approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
            bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
            bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION + 1,
            context_budget=context_budget,
        )

    honest = _played(column, cell.scenario, tmp_path / "honest", cell.lens)
    unscoped = _unscoped_digest(honest, tmp_path / "twin")
    assert _not_the_digest_without_a_path_scope(honest, unscoped) == []
    for name, change in (("budget", another_budget), ("bridge", another_bridge)):
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(ClaudeBackend, "_tool_authority_digest", _hashing(change))
            lying = _played(column, cell.scenario, tmp_path / name, cell.lens)
        assert lying.report.seam_problems == ()
        assert lying.report.codes == honest.report.codes == {WRITE_DIGEST_CODE}, name
        assert _trace_digest(lying) != _trace_digest(honest), name
        assert _not_the_digest_without_a_path_scope(lying, unscoped) != [], name


def test_the_second_half_of_the_write_proof_fails_on_a_write_that_keeps_a_second_deviation(
    tmp_path: Path,
) -> None:
    # The lying counterparts of the second half, each with the one cause of
    # the known gap removed as the proof removes it. One keeps a deviation of
    # another clause: the session accounting of the peer is short. One keeps
    # the same code from a second cause: the hash is bound to the ToolSet that
    # was attached and to another context budget. One never met the cause:
    # the hash was handed the attached ToolSet all along.
    column = _column()
    cell = _write_cell(column)
    honest = _played(column, cell.scenario, tmp_path / "honest", cell.lens)
    attach, digest = ClaudeBackend.attach_toolset, ClaudeBackend._tool_authority_digest

    def play(name: str, *, budget: int = 0, driver: Any = None) -> tuple[runner.Execution, list[str]]:
        attached: dict[int, Any] = {}
        met: list[str] = []

        def remembering(self: Any, toolset: Any, **more: Any) -> None:
            attached[id(self)] = toolset
            attach(self, toolset, **more)

        def bound(self: Any, toolset: Any, context_budget: int | None) -> str:
            if toolset is not attached[id(self)]:
                met.append(type(toolset).__name__)
            asked = context_budget if not budget else (context_budget or 0) + budget
            return digest(self, attached[id(self)], context_budget=asked)

        more = {} if driver is None else {"driver": driver}
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(ClaudeBackend, "attach_toolset", remembering)
            patched.setattr(ClaudeBackend, "_tool_authority_digest", _hashing(bound))
            return _played(column, cell.scenario, tmp_path / name, cell.lens, **more), met

    removed, met = play("removed")
    assert _more_than_the_one_cause(honest, removed, met) == []
    short, short_met = play("short", driver=_with_dialect(_UnderBilled))
    assert short.report.seam_problems == () and WRITE_DIGEST_CODE not in short.report.codes
    assert short.report.codes and _more_than_the_one_cause(honest, short, short_met) != []
    twice, twice_met = play("twice", budget=1)
    assert twice.report.codes == {WRITE_DIGEST_CODE}
    assert _more_than_the_one_cause(honest, twice, twice_met) != []
    # A play of a cell whose ToolSet the adapter does not wrap says nothing
    # about the cause: the hash was handed what was attached.
    assert _more_than_the_one_cause(honest, removed, []) != []


# -- the module ends with the canary check --------------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
