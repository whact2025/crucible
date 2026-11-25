# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The Codex app-server kind: its cells, and the proof of its driver and its peer.

The cells run the starter set on the session route through the real adapter
and the pinned ``codex-app-server-sdk``, with a scripted peer where the route
looks for the vendor tool. The remaining tests show that the driver's seam
can fail: a tool that the SDK did not start, an environment that was not
sanitised, a session without the route's version probe, a sign-in file that
is not a private copy, a thread without the exact usage notification, a peer
that answers a tool call without asking the tool server. They show that the
accounting of the peer is judged against the provider ledger, what the green
of the column rests on (the one notification no vendor type in reach states)
and what it does not rest on (how the tool fills the ``last`` counters of a
usage notification, which the adapter never reads on that path), and that a
trace which presents a served model fails a cell although the wire names
none. The peer is a program of its own: its journal keeps no value of the
environment it was started in, and the network guard of the package runs
inside it.

One test is held by name for the verifier, because no code of its cell shows
what it holds: the result of a dispatch-budget denial reports every response
the provider billed. That result flags its accounting incomplete, which the
ledger carries as a known gap, and while it does no clause compares what it
reports. The lying counterparts of that test are beside it.

Two lying counterparts are shared with the module of the official-SDK column,
which plays them on its own driver: :class:`LastAsTheTurnSoFar` and
:func:`presenting_a_served_model`.

No worker environment is needed here. What the official-SDK column adds to
the peer (the stage that puts it where the SDK looks for its bundled runtime,
and the rule that a worker starts no program of its own environment) is shown
on a stand-in worker environment; the column itself is another module.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest

from assay import Usage
from crucible.backends import codex as adapter
from crucible.backends.config import load_routing

from . import cells, contract, declarations, ledger, runner, toolsets
from . import columns as matrix_columns
from .cells import Cell
from .columns import CANARY_CREDENTIAL, CODEX_LEGACY_KIND, COLUMN_PRODUCT, Column
from .drivers import MatrixSetupError
from .drivers import codex_legacy as codex_driver
from .peers import codex_app_server as peer_program
from .peers import codex_worker as stage
from .peers import launch, worker_harness
from .provider import LoopbackProvider, Reply, SeenRequest
from .scenarios import D_CDX_SERVED, SCENARIOS, SYSTEM, USER
from .script import READ_A, READ_B, Call, ProviderScript, Refuse, Say, ServedAs, Step

LEDGER = ledger.load()
KINDS = (CODEX_LEGACY_KIND,)
# The envelope of the one refusal the notifications of the app-server carry.
_REFUSED_TURN = "codex_app_server/app_server.turn_failed_policy"
CELLS = cells.cells_of(KINDS)
(COLUMN,) = (column for column in matrix_columns.real_columns() if column.kind in KINDS)
# The product's loopback tool server logs under its module name.
TOOL_SERVER_LOGGER = "crucible.tools._mcp_http_server"
RAW_USAGE = "rawResponse/completed"
THREAD_USAGE = "thread/tokenUsage/updated"


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)


def _played(scenario: str, tmp_path: Path, lens: str = "raw", **more: Any) -> runner.Execution:
    cell = Cell(COLUMN, scenario, lens)
    more.setdefault("differences", LEDGER.differences_for(cell.id))
    return runner.execute(cell, tmp_path, **more)


def _session(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    """The journal record of the session process that asked for a response."""

    return next(
        record
        for record in evidence["journal"]
        if record["role"] == "app_server" and record["pid"] == evidence["pid"]
    )


def _request(evidence: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    return codex_driver.CodexAppServerDialect.request(evidence, method)


def _with_dialect(dialect: type[codex_driver.LegacyDialect]) -> codex_driver.CodexLegacyDriver:
    class Driver(codex_driver.CodexLegacyDriver):
        def _dialect(self) -> codex_driver.CodexAppServerDialect:
            return dialect(client=codex_driver.sdk_client())

    return Driver()


# -- the seam canary: the real layers ran -----------------------------------------


def test_canary_shows_that_the_sdk_started_the_peer_for_the_route(tmp_path: Path) -> None:
    execution = _played("S00", tmp_path)
    (request,) = execution.observation.provider.requests
    evidence = request.evidence
    journal = evidence["journal"]
    session = _session(evidence)
    home = str(tmp_path / "codex-home")

    assert execution.report.seam_problems == () and execution.report.codes == frozenset()
    assert request.matched and request.path == codex_driver.CodexAppServerDialect.path
    # The session route ran its version probe first, in a process of its own,
    # with a fresh directory of the cell's home as its home; then the session.
    assert [record["role"] for record in journal] == ["version", "app_server"]
    probe = journal[0]
    assert probe["argv"] == peer_program.VERSION_COMMAND
    assert probe["env"]["HOME"] == probe["env"]["CODEX_HOME"] == probe["cwd"]
    assert Path(probe["cwd"]).parent == Path(home)
    pids = [record["pid"] for record in journal]
    assert len(set(pids)) == 2 and os.getpid() not in pids
    assert {record["ppid"] for record in journal} == {os.getpid()}
    # The allow-listed environment: this process holds the scripted route
    # variables and their credential, and none of them reached the tool.
    assert all(os.environ[name] == value for name, value in matrix_columns.SCRIPTED_ENV.items())
    assert request.credential_ok
    for record in journal:
        assert not set(matrix_columns.SCRIPTED_ENV) & set(record["env_names"])
        assert record["forbidden_seen"] is False
        assert set(record["env"]) == set(peer_program.JOURNALLED_ENV)
        assert "PYTEST_CURRENT_TEST" not in record["env_names"]
    kept = (tmp_path / "codex-peer" / "journal.jsonl").read_text(encoding="utf-8")
    for word in (CANARY_CREDENTIAL, codex_driver.LOGIN_WORD):
        assert word not in kept and word not in json.dumps(journal)
    # The sign-in file was validated and reached the tool as a private copy in
    # a home of the tool's own, not as the file of the cell's home.
    assert session["env"]["HOME"] == home
    assert session["env"]["CODEX_HOME"] not in (home, os.path.join(home, ".codex"))
    assert session["login"] == {
        "home": True,
        "present": True,
        "private": True,
        "auth_mode": "chatgpt",
        "api_key": False,
        "session": True,
    }
    # The adapter's command line: no ambient tool server, no project
    # instructions, no native executor, then the app-server.
    command = evidence["command"]
    assert command["command"] == [peer_program.APP_SERVER]
    assert {"shell_tool", "unified_exec", "multi_agent"} <= set(command["disabled"])
    assert command["settings"] == ["mcp_servers={}", "project_doc_max_bytes=0"]
    # The SDK opened the session under its own name, and the adapter asked
    # for the exact usage of every response.
    opening = evidence["requests"]
    assert [item["method"] for item in opening] == [
        "initialize",
        "thread/start",
        "thread/resume",
        "turn/start",
    ]
    assert not any(item["notification"] for item in opening)
    assert request.user_agent == codex_driver.sdk_client() != "/"
    assert _request(evidence, "initialize")["capabilities"]["experimentalApi"] is True
    thread = _request(evidence, "thread/start")
    route = load_routing(config_path=COLUMN.routing_path(COLUMN_PRODUCT)).for_role(
        matrix_columns.ROLE_BY_CAPABILITY["text"]
    )
    assert thread["model"] == request.model == COLUMN.model == route.model != ""
    assert thread["baseInstructions"] == request.system == SYSTEM
    assert (thread["approvalPolicy"], thread["sandbox"]) == ("never", "read-only")
    assert thread["experimentalRawEvents"] is True and evidence["raw_events"] is True
    assert thread["cwd"] == session["cwd"] == str(tmp_path / "backend-cwd")
    turn = _request(evidence, "turn/start")
    assert request.user == USER and turn["effort"] == route.effort
    assert request.tools == () and request.turn_control is None
    # What the probe was answered with: a version the adapter has reviewed
    # for the model of the route.
    config = json.loads((tmp_path / "codex-peer" / "config.json").read_text(encoding="utf-8"))
    assert config["cli_version"] == codex_driver.reviewed_cli_version(COLUMN.model)
    assert config["version_line"] == f"codex-cli {config['cli_version']}"
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    # The request id of the trace is the turn the peer started.
    assert result.execution_trace.provider_request_id == evidence["turn_id"]
    assert result.execution_trace.served_model is None
    assert execution.reference.endpoint is None


def test_the_route_is_the_shipped_one_with_the_peer_as_its_only_overlay(tmp_path: Path) -> None:
    driver = codex_driver.CodexLegacyDriver()
    overlay = driver.overlay(COLUMN, matrix_columns.SCRIPTED_ENV)
    home_before = os.environ["HOME"]
    predicate = adapter._is_native_codex_binary
    script = tmp_path / "another-script"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o700)

    assert tuple(overlay) == matrix_columns.ALLOWED_OVERLAY[CODEX_LEGACY_KIND]
    with driver.open(
        COLUMN,
        capability="text",
        lens="raw",
        script=ProviderScript((Say("unused"),)),
        tmp_path=tmp_path / "cell",
    ) as bound:
        backend = bound.backend
        launcher = bound.peer.launcher
        assert (backend.name, backend.model) == (COLUMN.adapter, COLUMN.model)
        assert backend.features == declarations.routed(COLUMN)
        assert bound.tool_path == "mcp_bridge" and bound.children() == ()
        assert bound.secrets == (codex_driver.LOGIN_WORD,)
        # For the block the route takes the launcher of the peer, and nothing
        # else, for a native executable; the home is the cell's own.
        assert adapter._is_native_codex_binary(launcher.resolve())
        assert not adapter._is_native_codex_binary(script)
        assert os.environ["HOME"] == str(tmp_path / "cell" / "codex-home")
        assert "CODEX_HOME" not in os.environ
        assert launcher.read_text(encoding="utf-8").startswith(f"#!{sys.executable} -IS\n")
    # Afterwards the predicate and the home are what they were, and the home
    # the test started with was never written to.
    assert adapter._is_native_codex_binary is predicate
    assert not adapter._is_native_codex_binary(launcher.resolve())
    assert os.environ["HOME"] == home_before
    assert not (Path(home_before) / ".codex").exists()
    route = load_routing(config_path=COLUMN.routing_path(COLUMN_PRODUCT)).default
    assert "cli_path" not in route.extra and route.base_url is None
    with pytest.raises(MatrixSetupError, match="may overlay"):
        matrix_columns.overlaid_routing(
            load_routing(config_path=COLUMN.routing_path(COLUMN_PRODUCT)),
            {"extra.credential_source": "environment"},
            kind=CODEX_LEGACY_KIND,
            backend=COLUMN.backend,
        )
    with pytest.raises(MatrixSetupError, match="no reviewed tool version"):
        codex_driver.reviewed_cli_version("a-model-nobody-reviewed")
    assert {"CODEX_HOME", "CRUCIBLE_CODEX_WORKER_PYTHON"} <= set(matrix_columns.CLEARED_ENV)


def _seam(evidence: Mapping[str, Any], **replaced: Any) -> list[str]:
    dialect = codex_driver.LegacyDialect(client=codex_driver.sdk_client())
    seen = dialect.parse(
        "POST",
        dialect.path,
        {},
        json.dumps(evidence).encode("utf-8"),
        credential=CANARY_CREDENTIAL,
    )
    return dialect.seam_problems(dataclasses.replace(seen, **replaced))


def test_seam_names_what_is_not_a_tool_the_sdk_started_for_the_route(tmp_path: Path) -> None:
    execution = _played("S00", tmp_path)
    healthy = execution.observation.provider.requests[0].evidence
    position = healthy["journal"].index(_session(healthy))

    def expect(edit: Any, *named: str) -> None:
        evidence = copy.deepcopy(dict(healthy))
        edit(evidence, evidence["journal"][position])
        problems = _seam(evidence)
        assert len(problems) == len(named), problems
        for problem, text in zip(problems, named, strict=True):
            assert text in problem, problems

    def in_this_process(evidence: dict[str, Any], session: dict[str, Any]) -> None:
        session["pid"] = evidence["pid"] = os.getpid()

    def other_client(evidence: dict[str, Any], _session: dict[str, Any]) -> None:
        evidence["requests"][0]["params"]["clientInfo"]["name"] = "another-client"

    def turn_before_thread(evidence: dict[str, Any], _session: dict[str, Any]) -> None:
        evidence["requests"].reverse()

    def home_of_the_cell(_evidence: dict[str, Any], session: dict[str, Any]) -> None:
        session["env"]["CODEX_HOME"] = os.path.join(session["env"]["HOME"], ".codex")

    assert _seam(healthy) == []
    # A process of this test, or one that something else started, or none.
    expect(in_this_process, "not a child")
    expect(lambda _e, s: s.update(ppid=1), "not a child")
    expect(lambda e, _s: e.update(pid=-1), "no session process")
    # The version probe is part of the session route.
    expect(lambda e, _s: e["journal"].pop(0), "version probe")
    # Not the client of the route, or not its opening of a session.
    expect(other_client, "route's client")
    expect(turn_before_thread, "in that order")
    expect(lambda e, _s: e["requests"].pop(0), "route's client", "in that order")
    expect(lambda e, _s: e["command"].update(command=["exec"]), "does not start the app-server")
    # The adapter asks for the exact usage of every response.
    expect(lambda e, _s: e.update(raw_events=False), "exact usage notification")
    # A variable of another route, and a credential under any name, in any
    # process the adapter started.
    expect(lambda _e, s: s["env_names"].append("KIMI_BASE_URL"), "scripted route variables")
    expect(lambda _e, s: s.pop("env_names"), "does not name its environment")
    expect(lambda _e, s: s.update(forbidden_seen=True), "reached the tool")
    expect(lambda _e, s: s.pop("forbidden_seen"), "reached the tool")
    expect(lambda _e, s: s["argv"].append(CANARY_CREDENTIAL), "reached the tool")
    expect(lambda e, _s: e["journal"][0].update(forbidden_seen=True), "reached the tool")
    # The sign-in file: absent, readable by others, or the file of the home
    # the cell wrote instead of the adapter's private copy.
    expect(lambda _e, s: s["login"].update(present=False), "sign-in file")
    expect(lambda _e, s: s["login"].update(private=False), "sign-in file")
    expect(home_of_the_cell, "sign-in file")
    (path,) = _seam(healthy, path="/v1/responses")
    assert "request path" in path


class _NoProbe(codex_driver.LegacyDialect):
    """Reads a session as if the route's version probe had not run before it."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        seen = super().parse(*arguments, **named)
        journal = [
            item for item in seen.evidence.get("journal", ()) if item["role"] == "app_server"
        ]
        return dataclasses.replace(seen, evidence={**seen.evidence, "journal": journal})


class _ModelBlind(codex_driver.LegacyDialect):
    """Does not read the model a thread was started on."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return dataclasses.replace(super().parse(*arguments, **named), model="")


def test_a_broken_seam_fails_the_cell_whatever_it_shows(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="did not follow the route's version probe"):
        runner.run_cell(
            Cell(COLUMN, "S00", "raw"),
            tmp_path,
            preflight={CODEX_LEGACY_KIND: None},
            known=LEDGER,
            driver=_with_dialect(_NoProbe),
        )


def test_c7_and_c9_are_live_on_the_column(tmp_path: Path) -> None:
    honest = _played("S01", tmp_path / "honest")
    blind = runner.execute(
        Cell(COLUMN, "S01", "raw"), tmp_path / "blind", driver=_with_dialect(_ModelBlind)
    )

    assert honest.report.outcomes["C7"].status == "ok"
    assert {request.model for request in honest.observation.provider.requests} == {
        honest.reference.model
    }
    assert "request.content:model" in blind.report.codes
    # Clause C9 read what the adapter wrote into its log directory for the
    # call and found neither the canary credential nor the sign-in word,
    # which it looks for because the driver names it as a secret of the cell.
    written = [item for item in (tmp_path / "honest" / "backend-logs").rglob("*") if item.is_file()]
    dump = honest.observation.debug_dump_text
    assert written and all(item.read_text(encoding="utf-8") in dump for item in written)
    assert codex_driver.LOGIN_WORD in honest.reference.secrets
    assert CANARY_CREDENTIAL not in dump and codex_driver.LOGIN_WORD not in dump
    assert honest.report.outcomes["C9"].status == "ok"
    login = tmp_path / "honest" / "codex-home" / ".codex" / peer_program.LOGIN_FILE
    assert codex_driver.LOGIN_WORD in login.read_text(encoding="utf-8")
    assert login.stat().st_mode & 0o777 == 0o600


# -- the tool path: the loopback tool server of the route -------------------------


class _Listener(logging.Handler):
    """Keeps the records of the logger it is attached to."""

    def __init__(self, heard: list[logging.LogRecord]) -> None:
        super().__init__()
        self._heard = heard

    def emit(self, record: logging.LogRecord) -> None:
        self._heard.append(record)


def test_tool_calls_travel_to_the_tool_server_of_the_route(tmp_path: Path) -> None:
    heard: list[logging.LogRecord] = []
    listener = _Listener(heard)
    server_log = logging.getLogger(TOOL_SERVER_LOGGER)
    server_log.addHandler(listener)
    try:
        execution = _played("S03", tmp_path)
    finally:
        server_log.removeHandler(listener)
    requests = execution.observation.provider.requests
    evidence = requests[-1].evidence
    (server,) = peer_program.tool_servers(evidence["command"]["settings"])
    url = peer_program.tool_servers(evidence["command"]["settings"])[server]
    bindings = [toolsets.BINDINGS[slot] for slot in (READ_A, READ_B)]
    listed = evidence["tools"][server]

    assert execution.report.codes == frozenset()
    # The peer has no way to the ToolSet but the server its command line
    # names: a loopback address that the adapter approved for its tools.
    assert url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
    assert 'mcp_servers.crucible.default_tools_approval_mode="approve"' in (
        evidence["command"]["settings"]
    )
    assert evidence["mcp"] == [
        {"server": server, "method": "initialize"},
        {"server": server, "method": "notifications/initialized"},
        {"server": server, "method": "tools/list"},
        *({"server": server, "method": "tools/call", "name": item.name} for item in bindings),
    ]
    assert set(listed) == {item.name for item in bindings}
    assert all(request.tools == tuple(listed) for request in requests)
    # The calls reached the ToolSet in this process with the model's arguments ...
    log = execution.observation.toolset_log
    assert [(item.name, dict(item.arguments)) for item in log] == [
        (item.name, dict(item.arguments)) for item in bindings
    ]
    # ... through the product's tool server, which logged each of them ...
    served = [record for record in heard if " tool " in record.getMessage()]
    assert len(served) >= len(bindings) and {record.name for record in heard} == {
        TOOL_SERVER_LOGGER
    }
    # ... and what the tools returned went back into the next model response,
    # unchanged.
    assert [request.tool_results for request in requests] == [
        (),
        (("call-1", log[0].result),),
        (("call-2", log[1].result),),
    ]
    assert execution.report.outcomes["C10"].status == "ok"
    # One session of the tool played all three responses, in one turn.
    assert len({request.evidence["pid"] for request in requests}) == 1
    assert len({request.evidence["turn_id"] for request in requests}) == 1
    assert execution.observation.result is not None
    # Every notification of the cell was rendered from an envelope: all of
    # them but the terminal of a turn the provider refused.
    assert set(execution.observation.provider.envelopes) == set(
        codex_driver.CodexLegacyDriver().envelopes
    ) - {_REFUSED_TURN}
    assert execution.grade == "constructed"


class _AroundTheToolServer(codex_driver.CodexLegacyDriver):
    """Its peer answers a tool call itself: the tool server is never asked."""

    faults = (peer_program.SKIP_TOOL_SERVER,)


def test_a_peer_that_answers_a_tool_call_without_the_tool_server_fails_the_cell(
    tmp_path: Path,
) -> None:
    cell = Cell(COLUMN, "S03", "raw")
    execution = runner.execute(cell, tmp_path / "played", driver=_AroundTheToolServer())
    requests = execution.observation.provider.requests

    # The model responses went by as scripted and got a tool result each ...
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
            preflight={CODEX_LEGACY_KIND: None},
            known=LEDGER,
            driver=_AroundTheToolServer(),
        )


# -- the accounting: what the peer emulates, and what the green rests on ----------


def _frames(reply: Reply, method: str) -> list[dict[str, Any]]:
    return [frame for frame in reply.body["frames"] if frame["method"] == method]


class _UnderBilled(codex_driver.LegacyDialect):
    """Reports ten input tokens fewer per response than the provider bills."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        for frame in _frames(reply, THREAD_USAGE):
            for part in ("total", "last"):
                frame["params"]["tokenUsage"][part]["inputTokens"] -= 10
                frame["params"]["tokenUsage"][part]["totalTokens"] -= 10
        for frame in _frames(reply, RAW_USAGE):
            frame["params"]["usage"]["inputTokens"] -= 10
        return reply


class _WithoutExactUsage(codex_driver.LegacyDialect):
    """Sends the usage of the thread and never the exact usage of a response."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        kept = [frame for frame in reply.body["frames"] if frame["method"] != RAW_USAGE]
        return dataclasses.replace(reply, body={**reply.body, "frames": kept})


class LastAsTheTurnSoFar:
    """A dialect of a Codex column that plays the other reading of ``last``.

    The usage notification of a thread carries two sets of counters. The
    envelope fills ``total`` with the thread so far and ``last`` with one
    model response. This dialect fills ``last`` with the turn so far, which
    is the thread so far on a thread with one turn, as every cell plays it.
    Both are instances of the vendor's type, which names the two fields and
    states the meaning of neither. It wraps the dialect of either Codex
    driver and changes nothing else of what that dialect renders.
    """

    def __init__(self, dialect: Any) -> None:
        self._dialect = dialect
        # Per model response of a session: ``last`` as the wrapped dialect
        # rendered it, and the usage the notification went out with.
        self.responses: list[tuple[dict[str, int], dict[str, dict[str, int]]]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dialect, name)

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply: Reply = self._dialect.render(step, seen, call_id)
        for frame in _frames(reply, THREAD_USAGE):
            usage = frame["params"]["tokenUsage"]
            one_response = dict(usage["last"])
            usage["last"] = dict(usage["total"])
            if seen.evidence.get("turn_id"):
                self.responses.append((one_response, usage))
        return reply

    def differs_from_the_second_response_on(self, responses: int) -> bool:
        """Whether a session of ``responses`` responses was told the other reading.

        Every notification went out with ``last`` equal to ``total``, and from
        the second response on that is not what one response used.
        """

        return len(self.responses) == responses > 1 and all(
            usage["last"] == usage["total"] and (one == usage["total"]) is (index == 0)
            for index, (one, usage) in enumerate(self.responses)
        )


class _OtherReadingOfLast(codex_driver.CodexLegacyDriver):
    """Its dialect sends ``last`` as the counters of the turn so far."""

    def __init__(self) -> None:
        self.dialects: list[LastAsTheTurnSoFar] = []

    def _dialect(self) -> Any:
        self.dialects.append(LastAsTheTurnSoFar(super()._dialect()))
        return self.dialects[-1]


def test_the_peers_thread_accounting_is_judged_against_the_provider_ledger(
    tmp_path: Path,
) -> None:
    honest = _played("S01", tmp_path / "honest")
    short = runner.execute(
        Cell(COLUMN, "S01", "raw"), tmp_path / "short", driver=_with_dialect(_UnderBilled)
    )

    assert honest.report.outcomes["C6"].status == "ok"
    billed = honest.observation.provider.billed_total()
    assert honest.observation.result is not None
    assert honest.observation.result.usage.input_tokens == billed.input_tokens
    # The adapter believes the tool; the provider ledger does not.
    assert short.observation.result is not None
    assert short.observation.result.completeness.token_usage_complete
    assert short.report.codes == {"usage.input_tokens:lt", "usage.cost_usd:lt"}
    assert short.report.tripped() == {"C6"}


@pytest.mark.parametrize("scenario", ["S01", "S03"])
def test_the_green_of_the_accounting_rests_on_the_exact_usage_notification(
    scenario: str, tmp_path: Path
) -> None:
    """What the one envelope that is written to the adapter carries.

    No vendor type in reach states the exact usage notification of a response.
    With it the adapter proves its accounting; without it the same usage of
    the thread is a lower bound that the enforcing route refuses to pass on.
    The cells of the column are green only as far as the real tool sends it.
    """

    cell = Cell(COLUMN, scenario, "raw")
    honest = _played(scenario, tmp_path / "honest")
    without = runner.execute(
        cell, tmp_path / "without", driver=_with_dialect(_WithoutExactUsage)
    )

    assert "codex_app_server/app_server.raw_response_completed" in (
        honest.observation.provider.envelopes
    )
    assert honest.report.codes == frozenset()
    # The exchange went by as scripted, the seam held, the same tokens were
    # reported ...
    assert without.report.seam_problems == ()
    assert without.report.outcomes["C0"].status == "ok"
    assert without.observation.result is not None and honest.observation.result is not None
    assert without.observation.result.usage == honest.observation.result.usage
    # ... and the result is no longer one the route lets through as complete.
    assert without.report.codes == {
        "flag:token_usage_complete=False",
        "kind:incomplete_evidence!=none",
        "pricing:unreconciled",
    }


@pytest.mark.parametrize("scenario", ["S03", "S04"])
def test_the_green_of_the_accounting_does_not_rest_on_what_last_holds(
    scenario: str, tmp_path: Path
) -> None:
    """What the green of the column says nothing about: how the tool fills ``last``.

    With the exact usage notification the adapter holds the exact rows of a
    turn against the thread's total and reads nothing else of the usage
    notification. So the column is as green under the other reading of
    ``last``, and its green is no evidence for the reading the peer plays.
    The official-SDK column reports what its worker reads from ``last``:
    there the choice decides what a turn of several responses reports, which
    the module of that column shows.
    """

    cell = Cell(COLUMN, scenario, "raw")
    honest = _played(scenario, tmp_path / "one-response")
    driver = _OtherReadingOfLast()
    other = runner.execute(
        cell, tmp_path / "turn-so-far", driver=driver, differences=LEDGER.differences_for(cell.id)
    )
    (dialect,) = driver.dialects
    billed = honest.observation.provider.billed_total()

    # The turn has several responses, so the two readings differ on it from
    # the second response on, and the thread's total is what the provider
    # billed under either (the peer keeps the total, never ``last``).
    assert dialect.differs_from_the_second_response_on(len(other.observation.provider.requests))
    total = dialect.responses[-1][1]["total"]
    assert (total["inputTokens"], total["outputTokens"]) == (
        billed.input_tokens,
        billed.output_tokens,
    )
    assert other.observation.provider.billed_total() == billed
    assert other.report.seam_problems == ()
    # The column shows nothing of it: no code, and the same usage, flags and
    # pricing evidence.
    assert honest.report.codes == other.report.codes == frozenset()
    one, turn = honest.observation.result, other.observation.result
    assert one is not None and turn is not None
    assert (turn.usage, turn.completeness, turn.pricing_evidence) == (
        one.usage,
        one.completeness,
        one.pricing_evidence,
    )


# -- the usage of a denied turn: what the entry of its cell does not show ---------

DENIAL = cells.DISPATCH_DENIAL_ON_THE_APP_SERVER_CODEX_ROUTE
# The record of the ledger whose cell the proof below plays.
DENIAL_GAP = "R08"


def _judged(execution: runner.Execution, **changed: Any) -> frozenset[str]:
    """The codes of a play as the contract judges it with fields of its result changed."""

    result = execution.observation.result
    assert result is not None
    observation = dataclasses.replace(
        execution.observation, result=dataclasses.replace(result, **changed)
    )
    scenario = SCENARIOS[execution.cell.scenario]
    return contract.evaluate(
        observation,
        execution.plan.expect,
        execution.reference,
        lens=execution.cell.lens,
        toolset=scenario.toolset is not None,
        canary=scenario.canary,
        checks=execution.plan.checks,
    ).codes


def _usage_codes_once_flagged_complete(execution: runner.Execution) -> set[str]:
    """What clause C6 says of the usage a result reports, once the result claims it complete.

    Clause C6 holds a result that flags its token usage and its cost complete
    to the tokens of every response the provider billed and to their price on
    the rate card; of one that flags them incomplete it takes any number below
    the bill for honest. This judges the same observation again with the two
    flags of its result set, by the contract itself, and keeps the codes that
    are about the size of the usage.
    """

    result = execution.observation.result
    assert result is not None
    claimed = dataclasses.replace(
        result.completeness, token_usage_complete=True, cost_complete=True
    )
    return {
        code
        for code in _judged(execution, completeness=claimed)
        if code.startswith("usage.")
    }


def _short_of_the_bill(execution: runner.Execution) -> list[str]:
    """Why a play is not a result that reports every response the provider billed."""

    result = execution.observation.result
    assert result is not None
    problems: list[str] = []
    billed = execution.observation.provider.billed
    if len(billed) < 2 or not all(item.input_tokens and item.output_tokens for item in billed):
        problems.append("the provider billed fewer than two responses, or one for nothing")
    short = _usage_codes_once_flagged_complete(execution)
    if short:
        problems.append(
            f"flagged complete, the usage the result reports would show {sorted(short)}"
        )
    trace = result.execution_trace
    if trace is None or not any(event.kind == "usage" for event in trace.events):
        problems.append("the trace of the result holds no usage event, or there is no trace")
    return problems


def test_the_result_of_a_dispatch_budget_denial_reports_every_response_the_provider_billed(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Held here by name, because no code of the cell shows it.

    When a workflow's tool-dispatch budget denies a call, the adapter cancels
    the attempt and builds the result of the denial from what that attempt
    had captured: every response the provider billed. It flags its token
    usage and its cost incomplete, which is all its cell shows (known gap
    R08), and clause C6 takes any number below the bill for honest on a
    result that is flagged so, down to none. So while the entry of the cell
    stands no clause compares what the result reports, and a result that had
    lost its usage would be the same expected failure. This test compares
    it: judged as a result that flags its accounting complete, the result of
    the denial shows no code about the size of its usage, and its trace
    holds a usage event. A fix that lets the route flag the accounting of a
    settled turn complete lets the cell compare these numbers itself; this
    test and its lying counterparts then go with the entry.
    """

    # The verifier requires this test by name: the entry of the record leans
    # on it, and a run without it is refused. No entry attributes a code to
    # that record on a cell this test does not play.
    assert request.node.name in cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[CODEX_LEGACY_KIND]]
    kind, scenario, role, lens = ledger.parse_cell_id(DENIAL)
    assert (kind, role) == (COLUMN.kind, COLUMN.role)
    attributed = {entry.cell for entry in LEDGER.entries if DENIAL_GAP in entry.codes.values()}
    assert attributed <= {DENIAL}

    execution = _played(scenario, tmp_path, lens)
    result = execution.observation.result

    # The play is the denial the scenario scripts: the kind and the stop
    # reason are the ones it asks for, one call reached the ToolSet, and the
    # provider was asked for nothing after the denied call.
    assert execution.report.seam_problems == () and result is not None
    for clause in ("C0", "C2", "C3", "C10"):
        assert execution.report.outcomes[clause].status == "ok", clause
    assert result.error is not None and execution.observation.provider.extra == 0
    assert _short_of_the_bill(execution) == []


def _without_its_usage(result: Any) -> Any:
    """``result`` as an adapter that had lost the usage of the turn would return it.

    No token, no cost, no usage event in the trace and no priced segment:
    what the result of the same denial is on the official-SDK route today.
    """

    trace, evidence = result.execution_trace, result.pricing_evidence
    kept = [event for event in trace.events if event.kind != "usage"]
    return dataclasses.replace(
        result,
        usage=Usage(),
        execution_trace=dataclasses.replace(
            trace,
            events=tuple(
                dataclasses.replace(event, sequence=index) for index, event in enumerate(kept)
            ),
        ),
        pricing_evidence=None if evidence is None else dataclasses.replace(evidence, segments=()),
    )


class _LosesTheUsageOfTheTurn(codex_driver.CodexLegacyDriver):
    """Its backend returns every result without the usage the adapter had built into it."""

    @contextlib.contextmanager
    def open(self, column: Column, **named: Any) -> Iterator[Any]:
        with super().open(column, **named) as bound:
            invoke = bound.backend.invoke

            def invoke_and_lose(invocation: Any) -> Any:
                return _without_its_usage(invoke(invocation))

            bound.backend.invoke = invoke_and_lose
            yield bound


def _handed_to_the_comparator(execution: runner.Execution) -> tuple[Any, ...]:
    """Everything of a play that the comparator of the ledger decides its verdict by."""

    report = execution.report
    return report.codes, report.evaluated(), report.unasked, dict(execution.current)


def test_the_proof_of_the_billed_denial_fails_on_a_result_that_lost_its_usage(
    tmp_path: Path,
) -> None:
    # The lying counterparts of the proof above. One is a play: a backend
    # that returns the result of the same denial without the usage its
    # attempt had captured. The cell does not see it. The result flags its
    # accounting incomplete either way, so the comparator is handed the same
    # for both plays and says of the lying one what it says of the honest
    # one; on a cell whose result flags its accounting complete the same
    # backend is named by clause C6. The others are the honest result with
    # one thing made up: one response short, one token short, the cost gone,
    # the usage event gone; and a provider that billed one response, or
    # nothing.
    _kind, scenario, _role, lens = ledger.parse_cell_id(DENIAL)
    cell = Cell(COLUMN, scenario, lens)
    in_force = LEDGER.differences_for(cell.id)
    honest = _played(scenario, tmp_path / "honest", lens)
    lost = runner.execute(
        cell, tmp_path / "lost", driver=_LosesTheUsageOfTheTurn(), differences=in_force
    )
    plain = Cell(COLUMN, "S01", "raw")
    seen = runner.execute(
        plain,
        tmp_path / "seen",
        driver=_LosesTheUsageOfTheTurn(),
        differences=LEDGER.differences_for(plain.id),
    )
    result = honest.observation.result
    billed = honest.observation.provider.billed
    assert result is not None and result.execution_trace is not None and len(billed) == 2

    def with_result(**changed: Any) -> runner.Execution:
        observation = dataclasses.replace(
            honest.observation, result=dataclasses.replace(result, **changed)
        )
        return dataclasses.replace(honest, observation=observation)

    def billing(*responses: Any) -> runner.Execution:
        provider = dataclasses.replace(honest.observation.provider, billed=list(responses))
        return dataclasses.replace(
            honest, observation=dataclasses.replace(honest.observation, provider=provider)
        )

    assert _short_of_the_bill(honest) == []
    # The play: the same denial, and a result that reports nothing of it.
    dropped = lost.observation.result
    assert lost.report.seam_problems == () and dropped is not None
    assert lost.observation.provider.billed == billed and dropped.usage == Usage()
    assert dropped.completeness == result.completeness
    assert _handed_to_the_comparator(lost) == _handed_to_the_comparator(honest)
    assert _short_of_the_bill(lost) != []
    # Where a result flags its accounting complete the cell names the loss.
    assert seen.observation.result is not None
    assert seen.observation.result.completeness.token_usage_complete
    assert {"usage.input_tokens:lt", "usage.output_tokens:lt"} <= seen.report.codes
    # One response of the two, in either order; one token; the cost.
    usage = result.usage
    for response in billed:
        one = dataclasses.replace(
            usage, input_tokens=response.input_tokens, output_tokens=response.output_tokens
        )
        assert _short_of_the_bill(with_result(usage=one)) != []
    for name in ("input_tokens", "output_tokens"):
        fewer = dataclasses.replace(usage, **{name: getattr(usage, name) - 1})
        assert _short_of_the_bill(with_result(usage=fewer)) != []
    assert usage.cost_usd > 0
    assert _short_of_the_bill(with_result(usage=dataclasses.replace(usage, cost_usd=0.0))) != []
    # The numbers as billed, and no usage event in the trace.
    silent = dataclasses.replace(
        result.execution_trace, events=_without_its_usage(result).execution_trace.events
    )
    assert _short_of_the_bill(with_result(execution_trace=silent)) != []
    # A provider that billed one response, or nothing: "every response" would
    # then say nothing of a turn of several.
    assert _short_of_the_bill(billing(billed[0])) != []
    assert _short_of_the_bill(billing()) != []
    # More than the bill is no lower bound: the cell itself names that,
    # whatever the result flags.
    more = dataclasses.replace(usage, input_tokens=usage.input_tokens + 1)
    assert _judged(honest) == honest.report.codes
    assert _judged(honest, usage=more) == honest.report.codes | {"usage.input_tokens:gt"}


def presenting_a_served_model(driver: type[Any]) -> Any:
    """A driver like ``driver`` whose backend presents the requested model as the served one.

    The lying counterpart of declared difference ``D-CDX-SERVED``, for a
    column whose wire names no served model.
    """

    class Presenting(driver):  # type: ignore[misc,valid-type]
        @contextlib.contextmanager
        def open(self, column: Column, **named: Any) -> Iterator[Any]:
            with super().open(column, **named) as bound:
                invoke = bound.backend.invoke

                def invoke_and_present(invocation: Any) -> Any:
                    result = invoke(invocation)
                    trace = result.execution_trace
                    return dataclasses.replace(
                        result,
                        execution_trace=dataclasses.replace(
                            trace, served_model=trace.requested_model
                        ),
                    )

                bound.backend.invoke = invoke_and_present
                yield bound

    return Presenting()


def test_a_trace_that_presents_a_served_model_fails_the_cell(tmp_path: Path) -> None:
    """The declared difference of the wire is asserted, not waved through."""

    cell = Cell(COLUMN, "S01", "raw")
    in_force = LEDGER.differences_for(cell.id)
    honest = _played("S01", tmp_path / "honest")
    liar = presenting_a_served_model(codex_driver.CodexLegacyDriver)
    presenting = runner.execute(cell, tmp_path / "presenting", driver=liar, differences=in_force)
    unasserted = runner.execute(cell, tmp_path / "unasserted", driver=liar)

    # In force on every cell of the column that asserts a result: the wire of
    # the kind names no served model (static cell H03 holds the ledger to it).
    assert COLUMN.kind in matrix_columns.UNNAMED_SERVED_MODEL_KINDS
    assert in_force == (D_CDX_SERVED,)
    assert all(
        LEDGER.differences_for(item.id) == (() if SCENARIOS[item.scenario].canary else in_force)
        for item in CELLS
    )
    # The wire names no served model, so the provider side names none ...
    assert honest.observation.provider.served_models == []
    assert honest.observation.result is not None
    assert honest.observation.result.execution_trace is not None
    assert honest.observation.result.execution_trace.served_model is None
    assert honest.report.codes == frozenset()
    # ... and clause C7 alone has nothing to hold a made-up one against.
    assert unasserted.report.codes == frozenset()
    assert presenting.report.codes == {"digest.mismatch:served_model"}
    with pytest.raises(ledger.LedgerMismatch, match=r"unexpected deviation .*served_model"):
        runner.run_cell(
            cell,
            tmp_path / "judged",
            preflight={CODEX_LEGACY_KIND: None},
            known=LEDGER,
            driver=liar,
        )


# -- the dialect ------------------------------------------------------------------


def _evidence(**more: Any) -> dict[str, Any]:
    return {
        "pid": 7,
        "journal": [{"role": "app_server", "pid": 7, "ppid": 1, "forbidden_seen": False}],
        "requests": [
            {"method": "initialize", "params": {"clientInfo": {"name": "c", "version": "1"}}},
            {"method": "thread/start", "params": {"model": "m", "baseInstructions": "sys"}},
            {
                "method": "turn/start",
                "params": {"input": [{"type": "text", "text": "ask"}], "outputSchema": {"a": 1}},
            },
        ],
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "raw_events": True,
        "spent": {"input_tokens": 100, "output_tokens": 10},
        "tools": {"crucible": [toolsets.BINDINGS[READ_A].name]},
        "tool_results": [],
        **more,
    }


def _parsed(
    dialect: codex_driver.CodexAppServerDialect, evidence: Mapping[str, Any]
) -> SeenRequest:
    body = json.dumps(evidence).encode("utf-8")
    return dialect.parse("POST", dialect.path, {"Host": "127.0.0.1:1"}, body, credential="secret")


def test_the_dialect_reads_the_session_and_answers_only_what_it_has_an_envelope_for() -> None:
    dialect = codex_driver.LegacyDialect(client="c/1")
    seen = _parsed(dialect, _evidence())

    assert (seen.model, seen.system, seen.user) == ("m", "sys", "ask")
    assert seen.user_agent == "c/1" and seen.schema_field and seen.credential_ok
    assert seen.tools == (toolsets.BINDINGS[READ_A].name,) and seen.host == "127.0.0.1:1"
    assert (seen.turn_control, seen.budget_control) == (None, None)
    # A turn ceiling under any spelling would show: the protocol has none.
    capped = _evidence()
    capped["requests"][2]["params"]["max_turns"] = 2
    assert _parsed(dialect, capped).turn_control == 2
    # A process that saw a forbidden value, or one that does not say.
    for record in ({"forbidden_seen": True}, {}):
        tainted = _evidence(journal=[{"role": "app_server", "pid": 7, **record}])
        assert not _parsed(dialect, tainted).credential_ok
    # A call: the item before it runs, the usage of the response, no terminal.
    call = dialect.render(Call(READ_A), seen, "call-1")
    assert [frame["method"] for frame in call.body["frames"]] == [
        "item/started",
        RAW_USAGE,
        THREAD_USAGE,
    ]
    assert call.body["call"]["server"] == "crucible" and call.call is not None
    assert call.served_model is None
    usage = _frames(call, THREAD_USAGE)[0]["params"]["tokenUsage"]
    assert (usage["last"]["inputTokens"], usage["total"]["inputTokens"]) == (120, 220)
    assert usage["total"]["totalTokens"] == 220 + 40
    # The answer to the result of a call, then the final message and the terminal.
    answered = _evidence(
        tool_results=[
            {
                "call_id": "call-1",
                "server": "crucible",
                "name": "t",
                "arguments": {},
                "content": [{"type": "text", "text": "found"}],
                "is_error": False,
            }
        ]
    )
    after = _parsed(dialect, answered)
    assert after.tool_results == (("call-1", "found"),)
    final = dialect.render(Say("done"), after, "call-2")
    assert [frame["method"] for frame in final.body["frames"]] == [
        "item/completed",
        "item/completed",
        RAW_USAGE,
        THREAD_USAGE,
        "turn/completed",
    ]
    assert final.body["call"] is None and final.call is None
    assert final.body["frames"][0]["params"]["item"]["result"]["content"] == [
        {"type": "text", "text": "found"}
    ]
    # A client that did not ask for the exact usage gets none.
    quiet = dialect.render(Say("done"), _parsed(dialect, _evidence(raw_events=False)), "call-1")
    assert _frames(quiet, RAW_USAGE) == [] and len(_frames(quiet, THREAD_USAGE)) == 1
    # What has no envelope is refused, never improvised: a failed tool call,
    # a served model on a wire that names none, any other step.
    answered["tool_results"][0]["is_error"] = True
    failed = dialect.render(Say("done"), _parsed(dialect, answered), "call-2")
    assert failed.body["frames"] == [] and "has no envelope" in failed.body["unsupported"]
    with pytest.raises(MatrixSetupError, match="does not render"):
        dialect.render(ServedAs(Say("x"), "-other"), seen, "call-1")
    # The notifications carry one form of refusal: a failed turn whose error
    # names a policy of the provider. It is billed nothing and names no
    # model. The two other forms have no place on this wire and are pinned
    # not expressible in the inventory.
    refused = dialect.render(Refuse("http_policy"), seen, "call-1")
    (terminal,) = refused.body["frames"]
    assert terminal["method"] == "turn/completed"
    assert terminal["params"]["turn"]["status"] == "failed"
    assert terminal["params"]["turn"]["error"]["codexErrorInfo"] == "cyberPolicy"
    assert terminal["params"]["turn"]["error"]["message"]
    assert (refused.usage, refused.served_model, refused.call) == (None, None, None)
    assert refused.envelopes == (_REFUSED_TURN,) and refused.body["call"] is None
    for form in ("content_filter", "refusal_part"):
        with pytest.raises(MatrixSetupError, match="cannot render a refusal of the form"):
            dialect.render(Refuse(form), seen, "call-1")  # type: ignore[arg-type]
    assert list(codex_driver.CodexLegacyDriver().inexpressible) == [
        "S23a/Refuse(content_filter)",
        "S23b/Refuse(refusal_part)",
    ]
    with pytest.raises(NotImplementedError):
        codex_driver.CodexAppServerDialect().launch_problems({}, [], seen)
    assert dialect.envelope_names() == codex_driver.CodexLegacyDriver().envelopes


class _AnswersWhereItShouldRefuse(codex_driver.LegacyDialect):
    """Renders a refusal as a plain answer: the provider never refuses."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        if isinstance(step, Refuse):
            step = Say("an answer where the script refuses")
        return super().render(step, seen, call_id)


def test_a_refused_turn_is_typed_as_rejected_content_and_a_turn_that_is_not_refused_fails(
    tmp_path: Path,
) -> None:
    cell = Cell(COLUMN, "S23c", "raw")
    honest = runner.run_cell(cell, tmp_path / "honest", preflight={KINDS[0]: None}, known=LEDGER)

    # The cell passes with no entry: the turn failed with the policy of the
    # provider as its error, in one request for a model response, and the
    # adapter types that as rejected content that is not to be retried.
    result = honest.observation.result
    assert honest.report.codes == frozenset() and LEDGER.entry(cell.id) is None
    assert honest.observation.provider.envelopes == [_REFUSED_TURN]
    assert honest.observation.provider.request_count == 1
    assert result is not None and result.error is not None
    assert (result.error.kind, result.error.retriable) == ("content_rejected", False)
    # The lying counterpart: a provider side that answers instead of refusing
    # is a success where the scenario expects a refusal, and the cell fails.
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation.*content_rejected"):
        runner.run_cell(
            cell,
            tmp_path / "lying",
            preflight={KINDS[0]: None},
            known=LEDGER,
            driver=_with_dialect(_AnswersWhereItShouldRefuse),
        )


class _Unplayable(codex_driver.LegacyDialect):
    """Answers every request for a model response with something the peer does not play."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        return Reply(200, {"frames": [], "call": None, "unsupported": "not played in this test"})


def test_what_the_peer_does_not_play_is_a_setup_error_not_an_observation(tmp_path: Path) -> None:
    # The adapter sees a tool that went away and types that as it types any
    # failure; the cell must not judge it. The journal of the peer decides.
    with pytest.raises(
        MatrixSetupError,
        match="did not play the session: unsupported: not played in this test",
    ):
        runner.execute(Cell(COLUMN, "S01", "raw"), tmp_path, driver=_with_dialect(_Unplayable))
    records = [
        item
        for item in codex_driver.Peer(
            tmp_path / "codex-peer" / "codex",
            tmp_path / "codex-peer" / "config.json",
            tmp_path / "codex-peer" / "journal.jsonl",
        ).records()
        if item["role"] == "unsupported"
    ]
    assert records and {item["detail"] for item in records} == {"not played in this test"}


# -- the peer as a program --------------------------------------------------------


def _peer(tmp_path: Path, provider: str = "http://127.0.0.1:9/x", **more: Any) -> codex_driver.Peer:
    return codex_driver.write_peer(
        tmp_path / "peer", provider=provider, cli_version="9.9.9", **more
    )


def _run(
    peer: codex_driver.Peer, *argv: str, lines: list[dict[str, Any]] | None = None, **env: str
) -> subprocess.CompletedProcess[str]:
    text = "".join(json.dumps(line) + "\n" for line in lines or ())
    return subprocess.run(
        [str(peer.launcher), *argv],
        input=text,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={"PATH": "/usr/bin:/bin", **env},
    )


def _opening(*, raw: bool = True, thread: str | None = None) -> list[dict[str, Any]]:
    """What a client sends to open a session; ``thread`` resumes that id before the turn."""

    lines: list[dict[str, Any]] = [
        {
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "by-hand", "version": "1"},
                "capabilities": {"experimentalApi": raw},
            },
        },
        {
            "id": 2,
            "method": "thread/start",
            "params": {"model": "a-model", "experimentalRawEvents": raw, "sandbox": "read-only"},
        },
    ]
    if thread is not None:
        lines.append({"id": 3, "method": "thread/resume", "params": {"threadId": thread}})
    return lines


def _dialogue(
    peer: codex_driver.Peer, *, raw: bool = True, resume: bool = True, turn: bool = True
) -> tuple[int, list[dict[str, Any]]]:
    """Open a session by hand and run one turn on the thread it starts.

    The thread id is the session's own, so the turn is written once the
    session has named it. Returns the exit status and every line of output.
    """

    process = subprocess.Popen(
        [str(peer.launcher), "app-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert process.stdin is not None and process.stdout is not None
    out: list[dict[str, Any]] = []
    for line in _opening(raw=raw):
        process.stdin.write(json.dumps(line) + "\n")
        process.stdin.flush()
        out.append(json.loads(process.stdout.readline()))
    mine = out[1]["result"]["thread"]["id"]
    more: list[dict[str, Any]] = []
    if resume:
        more.append({"id": 3, "method": "thread/resume", "params": {"threadId": mine}})
    if turn:
        more.append(
            {
                "id": 4,
                "method": "turn/start",
                "params": {"threadId": mine, "input": [{"type": "text", "text": "ask"}]},
            }
        )
    for line in more:
        process.stdin.write(json.dumps(line) + "\n")
        process.stdin.flush()
    process.stdin.close()
    out.extend(json.loads(line) for line in process.stdout if line.strip())
    return process.wait(timeout=30), out


def test_the_peer_answers_the_version_probe_and_refuses_any_other_command(
    tmp_path: Path,
) -> None:
    peer = _peer(tmp_path)

    probe = _run(peer, "--version", SOME_SECRET=CANARY_CREDENTIAL)
    assert (probe.returncode, probe.stdout, probe.stderr) == (0, "codex-cli 9.9.9\n", "")
    refused = _run(peer, "exec", "something")
    assert refused.returncode == peer_program.EXIT_UNSUPPORTED and refused.stdout == ""
    listening = _run(peer, "app-server", "--listen", "ws://127.0.0.1:1")
    assert listening.returncode == peer_program.EXIT_UNSUPPORTED
    # A session whose input closes at once ends quietly.
    quiet = _run(peer, "-c", "mcp_servers={}", "app-server")
    assert (quiet.returncode, quiet.stdout, quiet.stderr) == (0, "", "")
    records = peer.records()
    assert [record["role"] for record in records] == [
        "version",
        "unsupported",
        "unsupported",
        "app_server",
    ]
    # The journal names the environment and keeps none of its values but the
    # three it is asked for; a forbidden value is reported and never written.
    assert records[0]["forbidden_seen"] is True and "SOME_SECRET" in records[0]["env_names"]
    assert set(records[0]["env"]) == {"PATH"}
    assert CANARY_CREDENTIAL not in peer.journal.read_text(encoding="utf-8")
    assert records[3]["forbidden_seen"] is False
    assert records[3]["login"] == {"home": False, "present": False}
    assert peer.failures() == [
        f"unsupported: {record['detail']}" for record in records if "detail" in record
    ]


def test_the_peer_plays_a_session_against_the_scripted_provider_on_loopback_only(
    tmp_path: Path,
) -> None:
    dialect = codex_driver.LegacyDialect(client="by-hand/1")
    script = ProviderScript((Say("first answer"), Say("second answer")))
    with LoopbackProvider(script, credential=CANARY_CREDENTIAL, dialect=dialect) as provider:
        peer = _peer(tmp_path, provider=provider.url + dialect.path)
        # A first run to learn nothing but that a thread id is generated.
        started = _run(peer, "app-server", lines=_opening())
        first = [json.loads(line) for line in started.stdout.splitlines()]
        thread = first[1]["result"]["thread"]["id"]

        assert started.returncode == 0 and [item["id"] for item in first] == [1, 2]
        assert first[0]["result"]["userAgent"].endswith("/9.9.9")
        assert first[1]["result"]["model"] == "a-model"
        assert first[1]["result"]["sandbox"] == {"type": "readOnly"}
        assert first[1]["result"]["thread"]["cliVersion"] == "9.9.9"

        status, with_raw = _dialogue(peer, raw=True)
        quiet_status, without_raw = _dialogue(peer, raw=False)
        ledger_ = provider.freeze()

    assert (status, quiet_status) == (0, 0)
    # Every session names a thread of its own.
    threads = {out[1]["result"]["thread"]["id"] for out in (with_raw, without_raw)}
    assert len({thread, *threads}) == 3

    methods = [item.get("method") for item in with_raw]
    turn = with_raw[3]["result"]["turn"]
    # The resumed thread is described as it was started; the turn is in
    # progress when it is answered; then the notifications of the response.
    assert with_raw[2]["result"] == with_raw[1]["result"]
    assert (turn["status"], turn["items"]) == ("inProgress", [])
    assert methods[4:] == ["item/completed", RAW_USAGE, THREAD_USAGE, "turn/completed"]
    assert with_raw[4]["params"]["item"]["text"] == "first answer"
    assert with_raw[-1]["params"]["turn"]["id"] == turn["id"]
    # The exact usage of a response is sent only to a client that asked.
    assert [item.get("method") for item in without_raw][4:] == [
        "item/completed",
        THREAD_USAGE,
        "turn/completed",
    ]
    assert [request.evidence["raw_events"] for request in ledger_.requests] == [True, False]
    assert ledger_.unconsumed == 0 and ledger_.extra == 0
    # A peer started by hand is visibly not the route's: the seam says so.
    assert any("version probe" in problem for problem in ledger_.seam_problems)
    assert any("sign-in file" in problem for problem in ledger_.seam_problems)

    # A thread the session does not hold, a request it does not play and a
    # provider that is not on loopback end the session as unsupported.
    foreign = _run(peer, "app-server", lines=_opening(thread="another-thread"))
    assert foreign.returncode == peer_program.EXIT_UNSUPPORTED
    interrupt = _run(
        peer, "app-server", lines=[*_opening(), {"id": 9, "method": "turn/interrupt", "params": {}}]
    )
    assert interrupt.returncode == peer_program.EXIT_UNSUPPORTED
    assert json.loads(interrupt.stdout.splitlines()[-1])["error"]["code"] == -32601
    assert [failure.split(":")[0] for failure in peer.failures()] == ["unsupported"] * 2


def test_the_network_guard_is_installed_in_every_process_of_the_peer(tmp_path: Path) -> None:
    elsewhere = "http://192.0.2.1:9/response"
    checked = codex_driver.write_peer(tmp_path / "checked", provider=elsewhere, cli_version="1")
    unchecked = codex_driver.write_peer(
        tmp_path / "unchecked",
        provider=elsewhere,
        cli_version="1",
        faults=[peer_program.SKIP_ADDRESS_CHECK],
    )

    # The peer refuses a provider that is not on loopback before it connects.
    status, _out = _dialogue(checked)
    assert status == peer_program.EXIT_UNSUPPORTED
    (refused,) = checked.failures()
    assert "not on loopback" in refused
    # Without its own check the peer would connect: the guard that the
    # launcher installed stops it, in the process of the peer.
    status, _out = _dialogue(unchecked)
    assert status == peer_program.EXIT_CRASHED
    (failure,) = unchecked.failures()
    assert failure.startswith("crashed:") and "NetworkGuardViolation" in failure
    assert "192.0.2.1" in failure
    # A tool server that is not on loopback is refused when the thread starts,
    # before anything is sent to it.
    served = _peer(tmp_path)
    setting = 'mcp_servers.elsewhere.url="http://192.0.2.1:9/mcp"'
    started = _run(served, "-c", "mcp_servers={}", "-c", setting, "app-server", lines=_opening())
    assert started.returncode == peer_program.EXIT_UNSUPPORTED
    assert len(started.stdout.splitlines()) == 1  # the answer to initialize, and no thread
    (elsewhere_refused,) = served.failures()
    assert "tool server 'elsewhere' is not on loopback" in elsewhere_refused
    assert peer_program.tool_servers(["mcp_servers.a.url=\"u\"", "mcp_servers={}"]) == {}


# -- the stage of the official-SDK column, on a stand-in worker environment -------

_RUNTIME = '''
from pathlib import Path


def bundled_codex_path():
    return Path(__file__).resolve().parent / "bin" / "codex"
'''
# What the stand-in worker does with its runtime, as the official SDK does:
# it asks where the bundled runtime is and starts it.
_WORKER = '''
import json
import os
import subprocess
import sys

import codex_cli_bin

report = {"argv": sys.argv[1:], "stage_variables": sorted(n for n in os.environ if "STAGE" in n)}
try:
    path = codex_cli_bin.bundled_codex_path()
    done = subprocess.run([str(path), "--version"], capture_output=True, text=True)
    report.update(started=str(path), stdout=done.stdout)
except BaseException as exc:  # a worker reports errors by type
    report.update(error=type(exc).__name__)
print(json.dumps(report, sort_keys=True))
'''


def _standin(root: Path) -> tuple[Path, Path, Path]:
    """A stand-in worker environment: its interpreter, its runtime, the runtime's mark."""

    environment = root / "backends" / "standin_worker" / ".venv"
    (environment / "bin").mkdir(parents=True)
    base = Path(sys.executable).resolve()
    (environment / "bin" / "python").symlink_to(base)
    (environment / "pyvenv.cfg").write_text(
        f"home = {base.parent}\ninclude-system-site-packages = false\n", encoding="utf-8"
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = environment / "lib" / version / "site-packages"
    runtime = site / stage.RUNTIME_MODULE
    (runtime / "bin").mkdir(parents=True)
    (runtime / "__init__.py").write_text(_RUNTIME.lstrip(), encoding="utf-8")
    mark = root / "the-bundled-runtime-ran"
    bundled = runtime / "bin" / "codex"
    bundled.write_text(f"#!/bin/sh\necho ran > {mark}\necho bundled\n", encoding="utf-8")
    bundled.chmod(0o700)
    worker = site / "matrix_standin_codex_worker"
    worker.mkdir()
    (worker / "__init__.py").write_text("", encoding="utf-8")
    (worker / "__main__.py").write_text(_WORKER.lstrip(), encoding="utf-8")
    return environment / "bin" / "python", bundled, mark


def _launch(
    interpreter: Path,
    journal: Path,
    target: list[str],
    settings: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    journal.mkdir(exist_ok=True)
    environments = (Path(os.path.realpath(interpreter.parents[1])),)
    interposer = launch.Interposer(lambda: launch.Journal(journal), environments=environments)
    env = {"PATH": "/usr/bin:/bin"}
    if settings is not None:
        env[stage.SETTINGS_ENV] = json.dumps(settings)
    return interposer.run(
        [str(interpreter), "-B", "-I", *target],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )


def test_the_stage_puts_the_peer_where_the_sdk_looks_for_its_bundled_runtime(
    tmp_path: Path,
) -> None:
    interpreter, bundled, mark = _standin(tmp_path / "repository")
    peer = _peer(tmp_path)
    journal = tmp_path / "journal"
    settings = {
        "module": "matrix_standin_codex_worker",
        "executable": str(peer.launcher),
        "journal": str(journal),
    }

    staged = _launch(interpreter, journal, [str(Path(stage.__file__).resolve())], settings)
    report = json.loads(staged.stdout)

    # The worker ran as the module it is, with no argument and no setting of
    # the stage left in its environment, and what it started was the peer.
    assert staged.returncode == 0 and staged.stderr == ""
    assert report == {
        "argv": [],
        "stage_variables": [],
        "started": str(peer.launcher),
        "stdout": "codex-cli 9.9.9\n",
    }
    assert not mark.exists()
    (record,) = [
        json.loads(line) for line in (journal / stage.STAGED).read_text().splitlines()
    ]
    assert record["bundled"] == str(bundled.resolve()) and record["executable"] == str(
        peer.launcher
    )
    assert record["replaced"] == f"{stage.RUNTIME_MODULE}.{stage.RUNTIME_NAME}"
    (launched,) = launch.Journal(journal).launches()
    assert (launched["mode"], launched["form"]) == ("guard", "script")
    assert launched["pid"] == record["pid"] and launch.Journal(journal).violations() == []
    assert [item["role"] for item in peer.records()] == ["version"]
    assert peer.records()[0]["ppid"] == record["pid"]
    # Without its settings, or with a peer that does not exist, the stage runs
    # nothing: the worker never gets to the vendor's runtime.
    for broken in (None, {**settings, "executable": str(tmp_path / "absent")}, {"module": "m"}):
        refused = _launch(interpreter, tmp_path / "refused", [str(stage.__file__)], broken)
        assert refused.returncode != 0 and refused.stdout == ""
        assert "codex worker stage" in refused.stderr
    assert not mark.exists()


def test_a_worker_that_starts_a_program_of_its_own_environment_is_a_violation(
    tmp_path: Path,
) -> None:
    interpreter, bundled, mark = _standin(tmp_path / "repository")
    journal = tmp_path / "journal"

    # In mode guard, with no stage in front of it, the worker finds the
    # bundled runtime and tries to start it, as the SDK would.
    guarded = _launch(interpreter, journal, ["-m", "matrix_standin_codex_worker"])
    report = json.loads(guarded.stdout)

    # The program never ran; the worker saw an error it could swallow, and the
    # journal of the test that started the worker holds the violation.
    assert guarded.returncode == 0 and not mark.exists()
    assert report["error"] == worker_harness.UnguardedProgram.__name__
    assert not issubclass(worker_harness.UnguardedProgram, OSError)
    (violation,) = launch.Journal(journal).violations()
    assert "worker harness" in violation and str(bundled.resolve()) in violation
    assert "does not see" in violation
    # Outside the harness the same worker starts it: the rule is the harness's.
    plain = subprocess.run(
        [str(interpreter), "-B", "-I", "-m", "matrix_standin_codex_worker"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert json.loads(plain.stdout)["stdout"] == "bundled\n" and mark.exists()


def test_a_locked_interpreter_started_through_asyncio_is_wrapped_and_nothing_else_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter, _bundled, _mark = _standin(tmp_path / "repository")
    directory = tmp_path / "journal"
    directory.mkdir()
    interposer = launch.Interposer(
        lambda: launch.Journal(directory),
        environments=(Path(os.path.realpath(interpreter.parents[1])),),
    )
    module = SimpleNamespace(asyncio=asyncio)
    shows = "import os, sys; print(sorted(n for n in os.environ if 'HARNESS' in n), sys.argv)"

    async def started(*argv: str, **named: Any) -> tuple[int | None, str]:
        process = await module.asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **named
        )
        out, _err = await process.communicate()
        return process.returncode, out.decode("utf-8")

    with launch.at_asyncio(module, interposer.command, "asyncio"):
        view = module.asyncio
        # Every other name of the module is the real one.
        assert isinstance(view, launch.AsyncioView) and view.sleep is asyncio.sleep
        # The locked interpreter ran under the harness: its target saw the
        # command line of the original launch and no setting of the harness.
        locked = asyncio.run(
            started(str(interpreter), "-B", "-I", "-c", shows, "x", env={"PATH": "/usr/bin:/bin"})
        )
        assert locked == (0, "[] ['-c', 'x']\n")
        (launched,) = launch.Journal(directory).launches()
        assert (launched["mode"], launched["form"], launched["arguments"]) == ("guard", "-c", ["x"])
        # Any other executable is started by the callable of the moment, with
        # the very arguments of the call: nothing is journalled for it.
        calls: list[tuple[Any, ...]] = []
        real = asyncio.create_subprocess_exec

        async def recording(*argv: Any, **named: Any) -> Any:
            calls.append((argv, sorted(named)))
            return await real(*argv, **named)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", recording)
        plain = asyncio.run(started(sys.executable, "-c", "print('plain')"))
        assert plain == (0, "plain\n")
        assert calls == [((sys.executable, "-c", "print('plain')"), ["stderr", "stdout"])]
        assert len(launch.Journal(directory).launches()) == 1
        # An interposer that is already there is replaced, not stacked.
        with launch.at_asyncio(module, interposer.command, "asyncio"):
            assert module.asyncio.matrix_real is asyncio
        assert module.asyncio is view
    assert module.asyncio is asyncio


# -- the module ends with the canary check ----------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
