# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The Codex official-SDK kind: its cells, and the proof of its driver.

Every test here needs the locked worker environment of ``backends/codex_worker``
and carries the ``matrix_worker`` marker; a machine without it gets an error,
not a skip. The stage and the rule that a worker starts no program of its own
environment are shown without it, on a stand-in, in the module of the
app-server kind.

The cells run the starter set on the shipped official-SDK route through the
real adapter, its worker client, the locked worker with the real worker
module and the pinned official SDK, with the scripted peer of the app-server
kind where the SDK looks for its bundled runtime. The remaining tests show
that the driver's seam can fail: a peer that no journalled worker started, a
worker that was not launched on the stage or not under the guard, a stage
that put nothing or something else in the place of the runtime, a client
that is not the pinned SDK. They pin every notification of the dialect to the
vendor's generated types inside the locked interpreter, and show that a
request launch without the stage never starts the bundled runtime. They show
that a trace which presents a served model fails a cell of this column too,
on every cell that asserts a result.

What the cells of this column show today is the ledger's business and the
log's, not this module's: a test here asserts shapes and rules, never a gap.
One rule is held by name for the verifier, because no cell shows when it
stops: a turn of one response reports the tokens the provider billed.
One test holds an open point instead of a rule: what the column reports for a
turn of several model responses is decided by how the envelope fills the
``last`` counters of a usage notification, which no source in reach states.
It plays both readings and leaves with whatever settles the point. The
ledger carries the point as a known gap that is unconfirmed and its record
leans on this test, so the verifier requires it by name too, on both cells.

The two lying counterparts it shares with the app-server column are defined
in that column's module.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from crucible.backends import codex_official_worker as worker_client

from . import cells, declarations, envelopes, ledger, runner, toolsets
from . import columns as matrix_columns
from .cells import Cell
from .columns import CANARY_CREDENTIAL, CODEX_OFFICIAL_KIND
from .drivers import MatrixSetupError
from .drivers import codex_legacy, codex_official
from .peers import codex_app_server as peer_program
from .peers import codex_worker as stage
from .peers import launch, worker_harness
from .scenarios import D_CDX_SERVED, SCENARIOS, SYSTEM, USER
from .script import READ_A, READ_B, Call, ProviderScript, Say
from .test_matrix_codex import LastAsTheTurnSoFar, presenting_a_served_model

pytestmark = pytest.mark.matrix_worker

LEDGER = ledger.load()
KINDS = (CODEX_OFFICIAL_KIND,)
CELLS = cells.cells_of(KINDS)
(COLUMN,) = (column for column in matrix_columns.real_columns() if column.kind in KINDS)
WORKER_MODULE = "crucible_codex_worker"
STAGE = str(Path(stage.__file__).resolve())


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)


def _played(scenario: str, tmp_path: Path, lens: str = "raw", **more: Any) -> runner.Execution:
    cell = Cell(COLUMN, scenario, lens)
    more.setdefault("differences", LEDGER.differences_for(cell.id))
    return runner.execute(cell, tmp_path, **more)


def _session(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(
        record
        for record in evidence["journal"]
        if record["role"] == "app_server" and record["pid"] == evidence["pid"]
    )


def _request(evidence: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    return codex_legacy.CodexAppServerDialect.request(evidence, method)


# -- the seam canary: the real layers ran -----------------------------------------


def test_canary_shows_that_the_sdk_in_the_locked_worker_started_the_peer(tmp_path: Path) -> None:
    driver = codex_official.CodexOfficialDriver()
    execution = _played("S00", tmp_path)
    (request,) = execution.observation.provider.requests
    evidence = request.evidence
    session = _session(evidence)
    journal = launch.Journal(tmp_path / "worker-harness")
    probe, staged_launch = journal.launches()
    (record,) = codex_official.staged(journal)
    launcher = str(tmp_path / "codex-peer" / "codex")

    assert execution.report.seam_problems == () and execution.report.codes == frozenset()
    assert request.matched and request.path == codex_legacy.CodexAppServerDialect.path
    # The adapter's probe and its request both started the locked interpreter
    # under the harness: isolated, without bytecode, under the network guard.
    for item in (probe, staged_launch):
        assert item["executable"] == str(driver.interpreter)
        assert (item["mode"], item["isolated"], item["no_bytecode"], item["guard"]) == (
            "guard",
            True,
            True,
            True,
        )
        assert item["pid"] != os.getpid()
    assert (probe["form"], probe["target"]) == ("-m", WORKER_MODULE)
    assert probe["arguments"] == ["--probe"]
    assert (staged_launch["form"], staged_launch["target"]) == ("script", STAGE)
    assert staged_launch["arguments"] == [] and journal.violations() == []
    # The stage put the peer of the cell where the SDK looks for the runtime
    # that the worker environment bundles, and ran the worker module.
    assert record == {
        "pid": staged_launch["pid"],
        "module": WORKER_MODULE,
        "replaced": f"{stage.RUNTIME_MODULE}.{stage.RUNTIME_NAME}",
        "bundled": record["bundled"],
        "executable": launcher,
    }
    assert Path(record["bundled"]).is_relative_to(driver.interpreter.parents[1])
    # The SDK in that worker started the peer: no version probe on this route.
    assert [item["role"] for item in evidence["journal"]] == ["app_server"]
    assert session["ppid"] == staged_launch["pid"] != session["pid"]
    # The pinned SDK opened the session under its own name and version, and
    # confirmed the handshake; the worker asked for no raw events.
    opening = evidence["requests"]
    assert [item["method"] for item in opening] == [
        "initialize",
        "initialized",
        "thread/start",
        "turn/start",
    ]
    assert [item["notification"] for item in opening] == [False, True, False, False]
    sdk = codex_official.pinned_version("openai-codex")
    assert request.user_agent == f"codex_python_sdk/{sdk}"
    assert sdk == codex_official.pinned_version("openai-codex-cli-bin")
    assert sdk == worker_client._SDK_VERSION
    assert evidence["raw_events"] is False
    # The SDK's command line: the worker's configuration overrides, then the
    # app-server on the standard streams.
    command = evidence["command"]
    assert command["command"] == [peer_program.APP_SERVER, *peer_program.LISTEN_STDIO]
    assert command["disabled"] == []
    assert {"mcp_servers={}", "project_doc_max_bytes=0", "features.shell_tool=false"} <= set(
        command["settings"]
    )
    thread = _request(evidence, "thread/start")
    assert thread["model"] == request.model == COLUMN.model != ""
    assert thread["baseInstructions"] == request.system == SYSTEM
    assert (thread["approvalPolicy"], thread["sandbox"]) == ("never", "read-only")
    assert request.user == USER and request.tools == () and request.turn_control is None
    # The worker's allow-listed environment: none of the scripted route
    # variables, no credential; the sign-in file as a private copy.
    assert request.credential_ok and session["forbidden_seen"] is False
    assert not set(matrix_columns.SCRIPTED_ENV) & set(session["env_names"])
    assert "PYTEST_CURRENT_TEST" not in session["env_names"]
    home = str(tmp_path / "codex-home")
    assert session["env"]["CODEX_HOME"] not in (home, os.path.join(home, ".codex"))
    assert session["login"]["present"] and session["login"]["private"]
    for word in (CANARY_CREDENTIAL, codex_legacy.LOGIN_WORD):
        for path in (*journal.directory.iterdir(), tmp_path / "codex-peer" / "journal.jsonl"):
            assert word not in path.read_text(encoding="utf-8")
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    assert result.execution_trace.provider_request_id == evidence["turn_id"]
    assert result.execution_trace.served_model is None
    assert execution.reference.endpoint is None


def test_the_route_is_the_shipped_one_and_nothing_is_overlaid(tmp_path: Path) -> None:
    driver = codex_official.CodexOfficialDriver()
    seams = {name: getattr(worker_client, name) for name in ("subprocess", "asyncio")}

    assert driver.overlay(COLUMN, matrix_columns.SCRIPTED_ENV) == {}
    assert CODEX_OFFICIAL_KIND not in matrix_columns.ALLOWED_OVERLAY
    assert launch.is_locked_interpreter(driver.interpreter)
    with driver.open(
        COLUMN,
        capability="text",
        lens="raw",
        script=ProviderScript((Say("unused"),)),
        tmp_path=tmp_path,
    ) as bound:
        backend = bound.backend
        assert (backend.name, backend.model) == (COLUMN.adapter, COLUMN.model)
        assert backend.transport == "official_sdk"
        assert backend.features == declarations.routed(COLUMN)
        assert bound.tool_path == "mcp_bridge" and bound.children() == ()
        # For the block the two launch seams of the worker client are the
        # driver's, and the interpreter is the one of the locked environment.
        assert isinstance(worker_client.subprocess, launch.SubprocessView)
        assert isinstance(worker_client.asyncio, launch.AsyncioView)
        assert "CRUCIBLE_CODEX_WORKER_PYTHON" not in os.environ
    assert {name: getattr(worker_client, name) for name in seams} == seams
    assert set(driver.private_seams) == {
        f"{worker_client.__name__}.{name}" for name in seams
    }


def _seam(
    evidence: Mapping[str, Any], journal: launch.Journal, launcher: str, **replaced: Any
) -> list[str]:
    driver = codex_official.CodexOfficialDriver()
    dialect = driver._dialect(journal, Path(launcher))
    seen = dialect.parse(
        "POST",
        dialect.path,
        {},
        json.dumps(evidence).encode("utf-8"),
        credential=CANARY_CREDENTIAL,
    )
    return dialect.seam_problems(dataclasses.replace(seen, **replaced))


def test_seam_names_what_is_not_a_tool_the_sdk_of_the_locked_worker_started(
    tmp_path: Path,
) -> None:
    execution = _played("S00", tmp_path / "played")
    healthy = execution.observation.provider.requests[0].evidence
    source = tmp_path / "played" / "worker-harness"
    launcher = str(tmp_path / "played" / "codex-peer" / "codex")

    def journal(name: str, edit: Any) -> launch.Journal:
        """A copy of the cell's journal with one record of one file changed."""

        directory = tmp_path / name
        directory.mkdir()
        for path in source.iterdir():
            records = [json.loads(line) for line in path.read_text().splitlines()]
            edit(path.name, records)
            (directory / path.name).write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
        return launch.Journal(directory)

    def launched(**changed: Any) -> Any:
        def edit(name: str, records: list[dict[str, Any]]) -> None:
            if name == "launches.jsonl":
                records[-1].update(changed)

        return edit

    def staged(**changed: Any) -> Any:
        def edit(name: str, records: list[dict[str, Any]]) -> None:
            if name == stage.STAGED:
                records[-1].update(changed)

        return edit

    def expect(name: str, edit: Any, text: str, evidence: Mapping[str, Any] = healthy) -> None:
        (problem,) = _seam(evidence, journal(name, edit), launcher)
        assert text in problem, problem

    assert _seam(healthy, journal("healthy", lambda _name, _records: None), launcher) == []
    # A peer whose parent is no worker of the journal.
    expect("no-worker", launched(pid=1), "no worker launch")
    # A worker that is not the adapter's locked interpreter, or not on the
    # stage, or not under the conditions of the harness.
    expect("other-python", launched(executable="/usr/bin/python3"), "locked interpreter")
    expect("module", launched(form="-m", target=WORKER_MODULE), "not launched on the stage")
    for flag in ("isolated", "no_bytecode", "guard"):
        expect(f"not-{flag}", launched(**{flag: False}), "under the network guard")
    # A stage that recorded nothing, another executable, or another module.
    expect("not-staged", staged(pid=1), "did not record")
    expect("other-peer", staged(executable="/bin/true"), "did not record")
    expect("other-module", staged(module="another"), "worker module of the locked environment")
    expect("ambient-runtime", staged(bundled="/usr/bin/codex"), "worker module of the locked")
    # A client that is not the pinned SDK: no confirmation of the handshake,
    # or another name.
    silent = copy.deepcopy(dict(healthy))
    silent["requests"] = [item for item in silent["requests"] if item["method"] != "initialized"]
    expect("silent", lambda _name, _records: None, "did not confirm", silent)
    renamed = copy.deepcopy(dict(healthy))
    renamed["requests"][0]["params"]["clientInfo"]["version"] = "0.0.0"
    expect("renamed", lambda _name, _records: None, "route's client", renamed)


class _NotStaged(codex_official.CodexOfficialDriver):
    """Its request launch runs the worker module under the guard, with no stage."""

    def _stage(self, peer: Path, journal: launch.Journal) -> Any:
        interposer = launch.Interposer(lambda: journal)
        return interposer, interposer.command


# What a worker would do with a program of its environment that does not exist.
_TRIPWIRE = """
import subprocess
import sys

try:
    subprocess.Popen([sys.prefix + "/bin/matrix-no-such-program"])
except BaseException as exc:
    print(type(exc).__name__)
"""


def test_a_worker_that_is_not_staged_never_starts_the_bundled_runtime(tmp_path: Path) -> None:
    cell = Cell(COLUMN, "S00", "raw")
    driver = codex_official.CodexOfficialDriver()

    # First on a path of the environment that holds no program: the harness
    # refuses it in this interpreter before anything is looked up. Only a
    # harness that was just seen to refuse is then given the real runtime to
    # refuse; one that does not would have started nothing here.
    tripwire = launch.Journal(tmp_path / "tripwire")
    tripwire.directory.mkdir()
    tripped = launch.Interposer(lambda: tripwire).run(
        [str(driver.interpreter), "-B", "-I", "-c", _TRIPWIRE],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert tripped.stdout.strip() == worker_harness.UnguardedProgram.__name__, tripped.stderr
    (refused,) = tripwire.violations()
    assert "matrix-no-such-program" in refused and "worker harness" in refused

    # The SDK in the worker finds the runtime its environment bundles and
    # tries to start it. The harness refuses, the worker reports an error by
    # type and the adapter returns a typed result: only the journal shows it,
    # and the cell ends as a setup error, never as an observation.
    with pytest.raises(MatrixSetupError, match="a program of the worker environment"):
        runner.execute(cell, tmp_path, driver=_NotStaged())
    journal = launch.Journal(tmp_path / "worker-harness")
    (violation,) = journal.violations()
    assert "codex_cli_bin/bin/codex" in violation and "worker harness" in violation
    assert codex_official.staged(journal) == []
    # No peer was started and nothing reached the scripted provider.
    assert not (tmp_path / "codex-peer" / "journal.jsonl").exists()


# -- the tool path ----------------------------------------------------------------


def test_tool_calls_travel_from_the_worker_to_the_tool_server_of_the_route(
    tmp_path: Path,
) -> None:
    execution = _played("S03", tmp_path)
    requests = execution.observation.provider.requests
    evidence = requests[-1].evidence
    (server,) = peer_program.tool_servers(evidence["command"]["settings"])
    bindings = [toolsets.BINDINGS[slot] for slot in (READ_A, READ_B)]

    # The worker configured the one tool server of the route for the SDK, the
    # peer reached it, and the calls arrived at the ToolSet in this process.
    assert server == "crucible" and execution.report.outcomes["C10"].status == "ok"
    assert [item["method"] for item in evidence["mcp"]] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
        "tools/call",
    ]
    log = execution.observation.toolset_log
    assert [(item.name, dict(item.arguments)) for item in log] == [
        (item.name, dict(item.arguments)) for item in bindings
    ]
    assert [request.tool_results for request in requests] == [
        (),
        (("call-1", log[0].result),),
        (("call-2", log[1].result),),
    ]
    # The vendor's types let every item of the turn through to the worker,
    # and the worker's projection of the tool items through to the adapter:
    # the trace pairs a tool event with each call.
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    events = [(event.kind, event.name, event.child_id) for event in result.execution_trace.events]
    for index, item in enumerate(bindings, start=1):
        assert ("tool_call", item.name, f"call-{index}") in events
        assert ("tool_result", item.name, f"call-{index}") in events
    assert result.final_message == execution.plan.expect.final_text


# -- the usage of a turn: what rests on the peer's reading of ``last`` ------------


class _OtherReadingOfLast(codex_official.CodexOfficialDriver):
    """Its dialect sends ``last`` as the counters of the turn so far."""

    def __init__(self) -> None:
        self.dialects: list[LastAsTheTurnSoFar] = []

    def _dialect(self, journal: launch.Journal, launcher: Path) -> Any:
        self.dialects.append(LastAsTheTurnSoFar(super()._dialect(journal, launcher)))
        return self.dialects[-1]


def _tokens(execution: runner.Execution) -> tuple[int, int]:
    result = execution.observation.result
    assert result is not None
    return result.usage.input_tokens, result.usage.output_tokens


def test_a_turn_of_one_response_reports_the_tokens_the_provider_billed(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Held here by name, because no cell of the column shows when it stops.

    Clause C6 compares the reported tokens with the provider's ledger only
    while the result flags its token usage complete; flagged incomplete, any
    lower bound passes, down to none. A route that never sets the flag on a
    reply that reads nothing from a cache is therefore compared by no cell.
    The comparison itself needs no flag: the turn has one response and the
    worker reports it.
    """

    execution = _played("S01", tmp_path)
    (billed,) = execution.observation.provider.billed

    # The verifier requires this test by name: a run without it is refused.
    assert request.node.name in cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[CODEX_OFFICIAL_KIND]]
    assert execution.report.seam_problems == ()
    assert execution.report.outcomes["C0"].status == "ok"
    assert _tokens(execution) == (billed.input_tokens, billed.output_tokens)
    assert billed.input_tokens > 0 < billed.output_tokens


@pytest.mark.parametrize("scenario", ["S03", "S04"])
def test_what_a_turn_of_several_responses_reports_rests_on_the_reading_of_last(
    scenario: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """An open point, held from both sides: not a rule, and a gap unconfirmed.

    The worker reports ``last`` of the turn's usage. The envelope fills
    ``last`` with one model response, and the column then reports the last
    response of a turn the provider billed several for. Filled with the turn
    so far, which is as much an instance of the vendor's type, the column
    reports the whole turn. No source in reach says which the real tool sends,
    and the green app-server column does not say either: its adapter never
    reads ``last`` there. So a usage code on these cells is, for now, the
    envelope's choice and not an observation of the worker: the ledger carries
    it as a known gap that is unconfirmed (R05). Whatever settles it (a
    recording of the real tool, a ruling on the reading, a change of the
    worker) rewrites this test.
    """

    # The verifier requires this test by name on both cells: the record of the
    # ledger leans on it, and a run without it is refused.
    assert request.node.name in cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[CODEX_OFFICIAL_KIND]]
    cell = Cell(COLUMN, scenario, "raw")
    one = _played(scenario, tmp_path / "one-response")
    driver = _OtherReadingOfLast()
    turn = runner.execute(
        cell, tmp_path / "turn-so-far", driver=driver, differences=LEDGER.differences_for(cell.id)
    )
    (dialect,) = driver.dialects
    billed = one.observation.provider.billed
    whole = one.observation.provider.billed_total()

    # The same exchange through the same seam, and the two readings differ on
    # it: the turn has several responses. The vendor's types validated every
    # notification of both plays inside the worker.
    for execution in (one, turn):
        assert execution.report.seam_problems == ()
        assert execution.report.outcomes["C0"].status == "ok"
    assert turn.observation.provider.billed == billed and len(billed) > 1
    assert dialect.differs_from_the_second_response_on(len(billed))
    # ``last`` as one response: the column reports the last response.
    assert _tokens(one) == (billed[-1].input_tokens, billed[-1].output_tokens)
    assert _tokens(one) != (whole.input_tokens, whole.output_tokens)
    # ``last`` as the turn so far: the column reports the turn the provider
    # billed, and no usage code.
    assert _tokens(turn) == (whole.input_tokens, whole.output_tokens)
    assert not {code for code in turn.report.codes if code.startswith("usage.")}
    # Nothing else moves with the reading: the codes of the column that are
    # not about the size of the usage are the same under both.
    assert turn.report.codes <= one.report.codes
    assert all(code.startswith("usage.") for code in one.report.codes - turn.report.codes)


# -- the served model: the wire names none ----------------------------------------


def test_a_trace_that_presents_a_served_model_fails_the_cell(tmp_path: Path) -> None:
    """The declared difference of the wire is asserted on this column too.

    Clause C7 compares the served model only where the provider named one, and
    this wire names none: the alternative of the difference is the only check
    of the field, and only on the cells the ledger lists.
    """

    cell = Cell(COLUMN, "S01", "raw")
    in_force = LEDGER.differences_for(cell.id)
    liar = presenting_a_served_model(codex_official.CodexOfficialDriver)
    honest = _played("S01", tmp_path / "honest")
    presenting = runner.execute(cell, tmp_path / "presenting", driver=liar, differences=in_force)
    unasserted_honest = runner.execute(cell, tmp_path / "unasserted-honest")
    unasserted = runner.execute(cell, tmp_path / "unasserted", driver=liar)

    # In force on every cell of the column that asserts a result: the wire of
    # the kind names no served model (static cell H03 holds the ledger to it).
    assert COLUMN.kind in matrix_columns.UNNAMED_SERVED_MODEL_KINDS
    assert in_force == (D_CDX_SERVED,)
    assert all(
        LEDGER.differences_for(item.id) == (() if SCENARIOS[item.scenario].canary else in_force)
        for item in CELLS
    )
    # The wire names no served model, so the provider side names none and the
    # honest trace records none.
    assert honest.observation.provider.served_models == []
    assert honest.observation.result is not None
    assert honest.observation.result.execution_trace is not None
    assert honest.observation.result.execution_trace.served_model is None
    assert "digest.mismatch:served_model" not in honest.report.codes
    # Clause C7 alone has nothing to hold a made-up one against: without the
    # difference the liar shows what the honest driver shows ...
    assert unasserted.report.codes == unasserted_honest.report.codes == honest.report.codes
    # ... and with it exactly one code more, which no entry of the ledger
    # covers on any cell.
    assert presenting.report.codes - honest.report.codes == {"digest.mismatch:served_model"}
    assert honest.report.codes <= presenting.report.codes
    with pytest.raises(ledger.LedgerMismatch, match=r"digest\.mismatch:served_model"):
        runner.run_cell(
            cell,
            tmp_path / "judged",
            preflight={CODEX_OFFICIAL_KIND: None},
            known=LEDGER,
            driver=liar,
        )


# -- the notifications: envelopes against the vendor's types ----------------------

_VALIDATE = """
import json
import sys

from openai_codex.generated.notification_registry import NOTIFICATION_MODELS

report = {}
for name, frame in json.load(sys.stdin).items():
    model = NOTIFICATION_MODELS.get(frame["method"])
    if model is None:
        report[name] = "no type"
        continue
    try:
        model.model_validate(frame["params"])
        report[name] = "valid"
    except Exception as exc:
        report[name] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, sort_keys=True))
"""


def test_the_notifications_validate_against_the_vendor_types_in_the_locked_interpreter(
    tmp_path: Path, locked_worker_launch: launch.Interposer
) -> None:
    driver = codex_official.CodexOfficialDriver()
    dialect = codex_legacy.LegacyDialect(client="c/1")
    evidence = {
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "raw_events": True,
        "tools": {"crucible": [toolsets.BINDINGS[READ_A].name]},
        "tool_results": [
            {
                "call_id": "call-1",
                "server": "crucible",
                "name": toolsets.BINDINGS[READ_A].name,
                "arguments": dict(toolsets.BINDINGS[READ_A].arguments),
                "content": [{"type": "text", "text": "found"}],
                "is_error": False,
            }
        ],
    }
    seen = dialect.parse("POST", dialect.path, {}, json.dumps(evidence).encode(), credential="")
    frames: dict[str, Any] = {}
    replies = (dialect.render(Call(READ_A), seen, "call-2"), dialect.render(Say("x"), seen, "c"))
    for reply in replies:
        for name, frame in zip(reply.envelopes, reply.body["frames"], strict=True):
            frames[name] = frame
    assert set(frames) == set(driver.envelopes)

    # Full validation, not the SDK's lenient construction, by the pinned SDK
    # in the locked interpreter, which runs under the harness of the suite.
    completed = locked_worker_launch.run(
        [str(driver.interpreter), "-B", "-I", "-c", _VALIDATE],
        input=json.dumps(frames),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={"PYTHONNOUSERSITE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    raw = "codex_app_server/app_server.raw_response_completed"
    # Every notification is a complete instance of the type its method names,
    # except the one no installed vendor type states: the exact usage of a
    # response, which is therefore written to the adapter and says so.
    assert report == {name: ("no type" if name == raw else "valid") for name in frames}
    stored = envelopes.load(raw)
    assert stored.grade == "constructed" and "sdk_type" not in stored.provenance
    assert "Written to the adapter" in stored.provenance["reason"]
    assert all(
        "sdk_type" in envelopes.load(name).provenance for name in frames if name != raw
    )
    assert locked_worker_launch.journal is not None
    assert locked_worker_launch.violations() == []


def test_a_missing_worker_environment_is_an_error_that_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absent = tmp_path / "backends" / "codex_worker" / ".venv" / "bin" / "python"

    class Elsewhere(codex_official.CodexOfficialDriver):
        @property
        def interpreter(self) -> Path:
            return absent

    with pytest.raises(MatrixSetupError, match="does not exist") as missing:
        Elsewhere().preflight()
    assert codex_official.INSTALL in str(missing.value)
    # The preflight of the session runs the adapter's own probe under the
    # guard, with the variables a cell never sees deleted.
    monkeypatch.setenv("CRUCIBLE_CODEX_WORKER_PYTHON", str(absent))
    assert runner.preflight_all()[CODEX_OFFICIAL_KIND] is None
    assert os.environ["CRUCIBLE_CODEX_WORKER_PYTHON"] == str(absent)


# -- the module ends with the canary check ----------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
