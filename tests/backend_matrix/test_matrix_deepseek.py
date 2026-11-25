# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The DeepSeek column: its cells, and the proof of its driver.

The cells run the starter set through the real adapter, its supervisor, the
locked worker interpreter with the pinned SDK and a loopback socket. Every
test of this module needs the locked worker environment
(``backends/deepseek_worker/.venv``); without it they fail with the command
that installs it, and they carry the marker a developer deselects them by.

The remaining tests show that the driver's seam can fail: a worker that is
not redirected, a redirect for another origin, a dialect that does not see
the worker launches, that answers under a fixed model id or does not read the
model on the wire. They show that the probe and the request launch both run
under the harness and leave the environment as it was, that a tool round is
replayed as the stateless route needs and that a cell deviates when it is not,
that the preflight of the session sees the environment a cell sees, and that
the envelopes parse through the pinned SDK in the locked interpreter.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

from crucible.backends import deepseek as adapter

from . import cells, declarations, envelopes, ledger, netguard, runner
from . import columns as matrix_columns
from .cells import Cell
from .columns import DEEPSEEK_KIND
from .drivers import MatrixSetupError
from .drivers import deepseek as driver
from .peers import launch
from .provider import LoopbackProvider, Reply, SeenRequest, SentCall
from .scenarios import SCENARIOS, TEXT, history_replayed
from .script import READ_A, Call, ProviderScript, Refuse, Say, ServedAs, Step

# Every test here runs the locked worker: a missing environment is an error.
pytestmark = pytest.mark.matrix_worker

LEDGER = ledger.load()
CELLS = cells.cells_of((DEEPSEEK_KIND,))
(COLUMN,) = (column for column in matrix_columns.real_columns() if column.kind == DEEPSEEK_KIND)
PRODUCTION = matrix_columns.WORKER_ORIGIN[DEEPSEEK_KIND]
HOST = PRODUCTION.removeprefix("https://")
DRIVER = driver.DeepSeekDriver()


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)


def _played(scenario: str, tmp_path: Path, lens: str = "raw", **more: Any) -> runner.Execution:
    cell = Cell(COLUMN, scenario, lens)
    more.setdefault("differences", LEDGER.differences_for(cell.id))
    return runner.execute(cell, tmp_path, **more)


# A cell takes several seconds here: every launch of the worker is preceded
# by a probe of its pin. The proofs below read one play of a scenario.
_PLAYED: dict[str, runner.Execution] = {}


@pytest.fixture()
def played(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], runner.Execution]:
    """The raw cell of a scenario, played once for the proofs of this module."""

    def play(scenario: str) -> runner.Execution:
        if scenario not in _PLAYED:
            _PLAYED[scenario] = _played(scenario, tmp_path_factory.mktemp(f"deepseek-{scenario}"))
        return _PLAYED[scenario]

    return play


def _matched(execution: runner.Execution) -> list[SeenRequest]:
    return [request for request in execution.observation.provider.requests if request.matched]


def _tree(root: Path) -> dict[str, tuple[int, int, int]]:
    """Every entry under ``root``: mode, size and modification time."""

    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_mode,
            path.lstat().st_size,
            path.lstat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
    }


# -- the seam canary: the locked worker ran behind the redirect -------------------


def test_canary_shows_the_locked_workers_own_request_behind_the_redirect(
    tmp_path: Path,
) -> None:
    project = DRIVER.interpreter.parents[2]
    before = _tree(project)

    execution = _played("S00", tmp_path)

    # The probes and the request launch left the worker project as it was.
    assert _tree(project) == before
    (request,) = execution.observation.provider.requests
    pinned = driver.pinned_sdk_version()

    assert execution.report.seam_problems == () and execution.report.codes == frozenset()
    assert request.matched and request.path == "/responses"
    # The request names the SDK version the worker's lock pins. That the
    # worker built it, and not this process, is what the launch journal shows
    # below: another process, the locked interpreter, behind the redirect.
    # (Today the test environment locks another version of the SDK as well.
    # Nothing here rests on that: the two locks may be aligned.)
    assert request.user_agent == f"OpenAI/Python {pinned}"
    assert request.credential_ok
    # The client addressed production; the redirect in the worker answered it.
    assert request.host == HOST and not netguard.is_loopback_host(request.host)
    assert request.model == COLUMN.model != ""
    assert request.evidence["stream"] is True

    # The harness journalled every launch before its target ran: the probes of
    # the pin in mode guard, then the request launch in mode redirect.
    launches = request.evidence["launches"]
    assert [item["mode"] for item in launches][-1] == "redirect"
    assert {item["mode"] for item in launches[:-1]} == {"guard"} and len(launches) >= 3
    for item in launches:
        assert item["pid"] != os.getpid() and item["parent"] != os.getpid()
        assert launch.is_locked_interpreter(item["executable"])
        assert Path(item["executable"]) == DRIVER.interpreter
        assert item["guard"] and item["isolated"] and item["no_bytecode"]
        assert (item["form"], item["target"]) == ("-m", "crucible_deepseek_worker")
    assert [item["arguments"] for item in launches[:-1]] == [["--probe"]] * (len(launches) - 1)
    request_launch = launches[-1]
    assert request_launch["arguments"] == []
    assert request_launch["redirect"]["origin"] == PRODUCTION
    assert netguard.is_loopback_host(
        request_launch["redirect"]["target"].removeprefix("http://").split(":")[0]
    )
    assert len({item["pid"] for item in launches}) == len(launches)

    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    assert result.execution_trace.served_model == COLUMN.model
    # The route names no endpoint, so C7 compares none; the adapter records
    # the one its worker addressed.
    assert execution.reference.endpoint is None
    assert result.execution_trace.endpoint == f"{PRODUCTION}/responses"


def test_the_route_is_the_shipped_one_with_the_captured_fingerprint_as_its_only_overlay(
    tmp_path: Path, locked_worker_launch: launch.Interposer
) -> None:
    project = DRIVER.interpreter.parents[2]
    before = _tree(project)
    suite_seam = adapter._run_supervised_command
    assert suite_seam.matrix_interposer is locked_worker_launch
    assert set(DRIVER.overlay(COLUMN, {})) == set(matrix_columns.ALLOWED_OVERLAY[DEEPSEEK_KIND])
    assert os.environ.get("DEEPSEEK_API_KEY") is None

    with DRIVER.open(
        COLUMN,
        capability="text",
        lens="raw",
        script=ProviderScript((Say("unused"),)),
        tmp_path=tmp_path,
    ) as bound:
        identity = bound.identity
        # Built by routing on the shipped file, bound to what the cell captured.
        assert bound.backend.name == COLUMN.adapter == "deepseek"
        assert bound.backend.features == declarations.routed(COLUMN)
        assert bound.backend.worker_runtime_fingerprint == identity.runtime_fingerprint
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", identity.runtime_fingerprint)
        assert identity.openai_version == driver.pinned_sdk_version()
        assert Path(identity.logical_worker_python) == DRIVER.interpreter
        assert bound.tool_path == "function_call" and bound.secrets == ()
        assert bound.journal.directory in bound.log_dirs and len(bound.log_dirs) == 2
        # The adapter reads its credential from the environment at dispatch.
        assert os.environ["DEEPSEEK_API_KEY"] == matrix_columns.CANARY_CREDENTIAL
        # One harness per launch: the driver's interposer replaced the suite's.
        seam = adapter._run_supervised_command
        assert seam.matrix_interposer is not locked_worker_launch
        assert seam.__wrapped__ is suite_seam.__wrapped__
        # Two probes so far, each in a process of its own, both gone.
        launches = bound.journal.launches()
        assert [(item["mode"], item["arguments"]) for item in launches] == [
            ("guard", ["--probe"])
        ] * 2
        assert len(bound.children()) == 2 and os.getpid() not in bound.children()
        # Captured again, the identity is the same one.
        again = DRIVER._capture(bound.journal)
        assert again == identity
        # A probe that did not run under the harness of the journal it is
        # captured for is refused: the pin would rest on an unguarded launch.
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(launch, "at_supervisor", lambda *_args: contextlib.nullcontext())
            with pytest.raises(MatrixSetupError, match="no launch was journalled"):
                DRIVER._capture(launch.Journal(tmp_path / "another-journal"))

    assert os.environ.get("DEEPSEEK_API_KEY") is None
    assert adapter._run_supervised_command is suite_seam
    # The harness left the worker project as it was: nothing was written
    # there, no bytecode either, so the pin still holds.
    assert _tree(project) == before
    # Nothing of the cell went through the suite's own interposer.
    assert locked_worker_launch.journal is None


def test_the_preflight_of_the_session_sees_the_environment_a_cell_sees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A developer's override of the worker interpreter is one of the variables
    # the per-test fixture deletes, so no cell sees it; the adapter's capture
    # function refuses to pin the worker while it is set.
    override = "CRUCIBLE_DEEPSEEK_WORKER_PYTHON"
    assert override in matrix_columns.CLEARED_ENV and override not in os.environ
    monkeypatch.setenv(override, str(DRIVER.interpreter))

    # Asked where the variable is set, the driver cannot pin the environment ...
    with pytest.raises(MatrixSetupError, match="rejects ambient interpreter overrides"):
        DRIVER.preflight()
    # ... and the session fixture, which runs before the per-test fixture, asks
    # under the cleared environment: the column can be driven, as its cells
    # will find it. The variable is back afterwards.
    assert runner.preflight_all()[DEEPSEEK_KIND] is None
    assert os.environ[override] == str(DRIVER.interpreter)


def test_the_probe_and_the_request_launch_run_under_the_harness(
    played: Callable[[str], runner.Execution],
) -> None:
    execution = played("S03")

    requests = _matched(execution)
    assert len(requests) == 3 and execution.report.seam_problems == ()
    launches = requests[-1].evidence["launches"]
    modes = [item["mode"] for item in launches]
    # The pin is probed for the overlay, at construction and before every
    # request launch; each of the three rounds is one redirected launch.
    assert modes == [*["guard"] * 3, "redirect", "guard", "redirect", "guard", "redirect"]
    assert all(
        item["arguments"] == ([] if item["mode"] == "redirect" else ["--probe"])
        for item in launches
    )
    # Every worker of the cell is gone, and the adapter found its pin
    # unchanged before each launch: the call ended without an error.
    assert execution.observation.settlement.children_alive == ()
    assert execution.observation.result is not None
    assert execution.observation.result.error is None


class _GuardOnly(driver.DeepSeekDriver):
    """No launch is redirected: the worker would reach for the real provider."""

    def _redirect(self, provider: LoopbackProvider) -> None:
        return None


class _OtherOrigin(driver.DeepSeekDriver):
    """The redirect answers another origin than the one the worker addresses."""

    def _redirect(self, provider: LoopbackProvider) -> launch.Redirect:
        redirect = super()._redirect(provider)
        assert redirect is not None
        return dataclasses.replace(redirect, origin="https://api.deepseek.invalid")


@pytest.mark.parametrize(
    ("liar", "stopped_by"),
    [
        (_GuardOnly, rf"network guard in worker process \d+: .*{re.escape(HOST)}"),
        (_OtherOrigin, rf"origin redirect in worker process \d+: .*{re.escape(HOST)}"),
    ],
    ids=["no-redirect", "redirect-for-another-origin"],
)
def test_a_worker_that_is_not_redirected_never_reaches_the_provider_and_fails_the_cell(
    liar: type[driver.DeepSeekDriver], stopped_by: str, tmp_path: Path
) -> None:
    # The worker swallows what the guard raises and the adapter types the
    # failure as a result. The journal of the harness is what decides.
    with pytest.raises(MatrixSetupError, match="reached beyond the scripted provider"):
        runner.execute(Cell(COLUMN, "S00", "raw"), tmp_path, driver=liar())

    violations = launch.Journal(tmp_path / "worker-harness").violations()
    assert len(violations) == 1 and re.search(stopped_by, violations[0])
    # Nothing left this process either.
    assert netguard.take_violations() == ()


# -- the dialect ------------------------------------------------------------------


class _WithoutLaunches(driver.DeepSeekResponsesDialect):
    """Does not look at what the harness journalled."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        seen = super().parse(*arguments, **named)
        return dataclasses.replace(seen, evidence={**seen.evidence, "launches": []})


class _FixedServedId(driver.DeepSeekResponsesDialect):
    """Answers every request under one id, whatever model was asked for."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        return dataclasses.replace(reply, served_model="a-fixed-model-id")


class _ModelBlind(driver.DeepSeekResponsesDialect):
    """Does not read the model a request names."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return dataclasses.replace(super().parse(*arguments, **named), model="")


def _with_dialect(dialect: type[driver.DeepSeekResponsesDialect]) -> driver.DeepSeekDriver:
    class Driver(driver.DeepSeekDriver):
        def _dialect(
            self, journal: launch.Journal, interpreter: str
        ) -> driver.DeepSeekResponsesDialect:
            return dialect(journal=journal, interpreter=interpreter)

    return Driver()


@pytest.mark.parametrize(
    ("scenario", "dialect", "problem"),
    [
        ("S00", _WithoutLaunches, "journalled no redirected worker launch"),
        ("S01", _FixedServedId, "answered under model 'a-fixed-model-id'"),
    ],
    ids=["no-launch-evidence", "fixed-served-id"],
)
def test_a_broken_seam_fails_the_cell_whatever_it_shows(
    scenario: str,
    dialect: type[driver.DeepSeekResponsesDialect],
    problem: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(pytest.fail.Exception, match=f"the driver seam is broken.*{problem}"):
        runner.run_cell(
            Cell(COLUMN, scenario, "raw"),
            tmp_path,
            preflight={DEEPSEEK_KIND: None},
            known=LEDGER,
            driver=_with_dialect(dialect),
        )


def test_c7_and_c9_are_live_on_the_column_through_what_the_dialect_reads(
    tmp_path: Path, played: Callable[[str], runner.Execution]
) -> None:
    blind = runner.execute(
        Cell(COLUMN, "S01", "raw"), tmp_path, driver=_with_dialect(_ModelBlind)
    )
    assert "request.content:model" in blind.report.codes

    execution = played("S01")
    assert "request.content:model" not in execution.report.codes
    assert {request.model for request in _matched(execution)} == {execution.reference.model}
    # Clause C9 read the debug records the adapter wrote for the call and the
    # command line of every worker launch, and found no credential in them,
    # in the result or in the trace.
    dump = execution.observation.debug_dump_text
    assert "crucible.sdk-debug-result-artifact" in dump
    assert '"target": "crucible_deepseek_worker"' in dump
    assert matrix_columns.CANARY_CREDENTIAL not in dump
    assert execution.report.outcomes["C9"].status == "ok"


def _healthy(tmp_path: Path) -> tuple[driver.DeepSeekResponsesDialect, dict[str, Any]]:
    """A dialect over a journal that holds one probe and one request launch."""

    directory = tmp_path / "journal"
    directory.mkdir()
    interpreter = str(DRIVER.interpreter)
    record: dict[str, Any] = {
        "pid": os.getpid() + 1,
        "parent": os.getpid() + 2,
        "mode": "redirect",
        "form": "-m",
        "target": "crucible_deepseek_worker",
        "arguments": [],
        "executable": interpreter,
        "isolated": True,
        "no_bytecode": True,
        "guard": True,
        "redirect": {"origin": PRODUCTION, "target": "http://127.0.0.1:1"},
    }
    probe = {**record, "mode": "guard", "arguments": ["--probe"], "redirect": None}
    lines = [json.dumps(probe), json.dumps(record)]
    (directory / "launches.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dialect = driver.DeepSeekResponsesDialect(
        journal=launch.Journal(directory), interpreter=interpreter
    )
    return dialect, record


def test_seam_names_what_is_not_the_locked_workers_request_behind_the_redirect(
    tmp_path: Path,
) -> None:
    dialect, record = _healthy(tmp_path)
    credential = matrix_columns.CANARY_CREDENTIAL

    def seen(
        headers: Mapping[str, str] | None = None, *, body: Mapping[str, Any] | None = None
    ) -> SeenRequest:
        sent = {
            "User-Agent": dialect.user_agent,
            "Authorization": f"Bearer {credential}",
            "Host": HOST,
            **(headers or {}),
        }
        payload = {"model": "m", "stream": True, "input": []} if body is None else body
        return dialect.parse(
            "POST", dialect.path, sent, json.dumps(payload).encode("utf-8"), credential=credential
        )

    def with_launch(**changed: Any) -> SeenRequest:
        healthy = seen()
        launches = [*healthy.evidence["launches"][:-1], {**record, **changed}]
        return dataclasses.replace(healthy, evidence={**healthy.evidence, "launches": launches})

    assert dialect.seam_problems(seen()) == []
    # A version that no lock pins, so that the liar differs from the worker's
    # SDK whatever version the test environment locks.
    other_sdk = "OpenAI/Python 0.0.0+not.the.pinned.version"
    assert other_sdk != dialect.user_agent
    for request, named in (
        (seen({"User-Agent": other_sdk}), "not the pinned SDK's"),
        (seen({"Authorization": "Bearer another-key"}), "canary credential"),
        (seen({"Authorization": credential}), "canary credential"),
        (seen({"Host": "127.0.0.1:8000"}), "did not come through the redirect"),
        (seen(body={"model": "m", "stream": False, "input": []}), "does not ask for a stream"),
        (dataclasses.replace(seen(), path="/v1/responses"), "request path"),
        (with_launch(mode="guard"), "journalled no redirected worker launch"),
        (with_launch(pid=os.getpid()), "a supervised process of its own"),
        (with_launch(parent=os.getpid()), "a supervised process of its own"),
        (with_launch(executable="/usr/bin/python3"), "not the adapter's locked worker"),
        (with_launch(arguments=["--probe"]), "not the request launch"),
        (with_launch(form="-c", target="<code>"), "not the request launch"),
        (with_launch(isolated=False), "not launched isolated"),
        (with_launch(no_bytecode=False), "not launched isolated"),
        (with_launch(guard=False), "not launched isolated"),
    ):
        (problem,) = dialect.seam_problems(request)
        assert named in problem, problem
    # A dialect that knows no journal or no interpreter passes nothing.
    assert driver.DeepSeekResponsesDialect().seam_problems(seen()) != []
    unknown = driver.DeepSeekResponsesDialect(journal=launch.Journal(tmp_path / "journal"))
    (problem,) = unknown.seam_problems(seen())
    assert "not the adapter's locked worker" in problem


def test_the_dialect_reads_a_responses_request_and_renders_only_from_envelopes() -> None:
    dialect = driver.DeepSeekResponsesDialect()
    body = {
        "model": "asked",
        "stream": True,
        "input": [
            {"role": "system", "content": "the system text"},
            {"role": "user", "content": "the user text"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thought"}]},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "first"},
            {"type": "function_call", "call_id": "c2", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c2", "output": "second"},
        ],
        "tools": [{"type": "function", "name": "t", "description": "", "parameters": {}}],
    }
    seen = dialect.parse("POST", "/responses", {}, json.dumps(body).encode(), credential="x")

    assert (seen.model, seen.system, seen.user) == ("asked", "the system text", "the user text")
    assert seen.tools == ("t",) and not seen.schema_field
    # What answers the last reply is what follows the last item of the model.
    assert seen.tool_results == (("c2", "second"),)
    assert [item["kind"] for item in seen.evidence["input"]] == [
        "system",
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
        "function_call",
        "function_call_output",
    ]
    assert seen.turn_control is None and seen.budget_control is None
    schema = {"text": {"format": {"type": "json_schema", "name": "n", "schema": {}}}}
    assert dialect.parse(
        "POST", "/responses", {}, json.dumps({**body, **schema}).encode(), credential="x"
    ).schema_field

    # A scripted ServedAs step is the one way to name another model.
    assert dialect.render(Say("x"), seen, "1").served_model == "asked"
    assert dialect.render(ServedAs(Say("x")), seen, "1").served_model == "asked-served"
    # The stream is the envelope's events with their slots filled, in the
    # server-sent-event form, and it has no closing message.
    reply = dialect.render(Call(READ_A), seen, "call-7")
    assert reply.content_type == "text/event-stream" and reply.raw is not None
    frames = reply.raw.decode("utf-8").split("\n\n")
    assert frames[-1] == "" and len(frames) == len(reply.body) + 1
    for frame, event in zip(frames[:-1], reply.body, strict=True):
        kind, data = frame.split("\n")
        assert kind == f"event: {event['type']}"
        assert json.loads(data.removeprefix("data: ")) == event
    assert "[DONE]" not in reply.raw.decode("utf-8")
    assert [event["sequence_number"] for event in reply.body] == list(range(len(reply.body)))
    terminal = reply.body[-1]
    assert terminal["type"] == "response.completed"
    assert terminal["response"]["model"] == "asked"
    assert reply.call is not None and reply.call[1] == "call-7"
    assert reply.envelope == "deepseek_responses/deepseek.tool_call"
    with pytest.raises(MatrixSetupError, match="does not render"):
        dialect.render(object(), seen, "1")  # type: ignore[arg-type]


# -- a tool round is replayed as the stateless route needs -------------------------


def test_a_tool_round_is_replayed_with_its_reasoning_item_and_its_call(
    played: Callable[[str], runner.Execution],
) -> None:
    execution = played("S03")
    requests = _matched(execution)
    calls = execution.observation.provider.sent_calls
    turn = driver.replayed_turn("deepseek_responses/deepseek.tool_call")

    # Every request is the worker's own: one request per round, nothing else.
    assert len(requests) == 3 == execution.observation.provider.request_count
    assert execution.observation.provider.extra == 0
    for request in requests:
        assert (request.path, request.host, request.credential_ok) == ("/responses", HOST, True)
    # The provider's turn held a reasoning item and a function call, and the
    # next request carries both before the output of the call, round by round.
    # The cell asserts it: the dialect states the rule, the provider applied
    # it to the exchange and the scenario reports what it names.
    assert turn == ("reasoning", "function_call")
    assert history_replayed in execution.plan.checks
    assert execution.observation.provider.replay_problems == []
    assert "request.content:replay" not in execution.report.codes
    assert driver.replay_problems(requests, calls) == []
    second = requests[1].evidence["input"]
    assert [item["kind"] for item in second] == [
        "system",
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert second[3]["call_id"] == second[4]["call_id"] == calls[0].call_id
    assert len(requests[2].evidence["input"]) == 8
    # The results the tools returned are what the next request answers with.
    assert [request.tool_results[0][0] for request in requests[1:]] == [
        call.call_id for call in calls
    ]


_PARSE = adapter._parse_worker_stdout


def _without_reasoning(stdout: bytes, *, returncode: int) -> Any:
    """The parent's reading of a worker's output, the reasoning item dropped.

    What the worker hands back for the stateless replay is the provider's
    turn in full. Here the parent loses the reasoning item of it, so its next
    request replays the function call and its output only.
    """

    terminal = _PARSE(stdout, returncode=returncode)
    kept = [
        item for item in json.loads(terminal.continuation_json) if item["type"] != "reasoning"
    ]
    return dataclasses.replace(terminal, continuation_json=json.dumps(kept))


@pytest.mark.parametrize(("scenario", "lens"), [("S04", "raw"), ("S03", "guarded")])
def test_a_replay_without_the_reasoning_item_is_a_deviation_of_the_cell(
    scenario: str, lens: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = Cell(COLUMN, scenario, lens)
    entry = LEDGER.entry(cell.id)
    assert entry is not None
    monkeypatch.setattr(adapter, "_parse_worker_stdout", _without_reasoning)

    execution = _played(scenario, tmp_path, lens)

    # The scripted provider answered all the same: it picks a reply on
    # position and call ids. Nothing else of the cell changed either ...
    report = execution.report
    assert report.seam_problems == () and execution.observation.provider.extra == 0
    second = _matched(execution)[1].evidence["input"]
    assert [item["kind"] for item in second] == [
        "system",
        "user",
        "function_call",
        "function_call_output",
    ]
    # ... so the codes are the ledgered ones and the one of the replay rule,
    # which names the request. As a test item the cell fails: its entry no
    # longer describes it.
    wide = {scope.code for scope in LEDGER.column_wide if scope.kind == DEEPSEEK_KIND}
    assert report.codes == {*entry.codes, *wide, "request.content:replay"}
    (problem, *_later) = execution.observation.provider.replay_problems
    assert problem.startswith("request 2 carries")
    with pytest.raises(ledger.LedgerMismatch, match="different deviation") as refused:
        ledger.compare(
            LEDGER,
            cell.id,
            report.codes,
            current=execution.current,
            evaluated=report.evaluated(),
            unasked=report.unasked,
            report=report.describe(),
        )
    assert "unexpected ['request.content:replay']" in str(refused.value)


def _request(kinds: Sequence[tuple[str, str | None]], *, matched: bool = True) -> SeenRequest:
    dialect = driver.DeepSeekResponsesDialect()
    blank = dialect.parse("POST", dialect.path, {}, b"{}", credential="")
    evidence = {"input": [{"kind": kind, "call_id": call_id} for kind, call_id in kinds]}
    return dataclasses.replace(blank, evidence=evidence, matched=matched)


def test_a_replay_that_drops_or_reorders_an_item_is_named() -> None:
    call = SentCall(READ_A, "call-1", 0, {})
    opening = [("system", None), ("user", None)]
    replay = [("reasoning", None), ("function_call", "call-1"), ("function_call_output", "call-1")]
    first = _request(opening)

    assert driver.replay_problems([first, _request([*opening, *replay])], [call]) == []
    for broken in (
        [*opening, *replay[1:]],  # the reasoning item is dropped
        [*opening, replay[1], replay[0], replay[2]],  # the call before its reasoning
        [*opening, replay[0], ("function_call", "call-9"), replay[2]],  # another call id
        [*opening, *replay[:2]],  # the call has no output
        [*opening],  # nothing is replayed
        [opening[1], opening[0], *replay],  # the user message before the system message
    ):
        (problem,) = driver.replay_problems([first, _request(broken)], [call])
        assert problem.startswith("request 2 carries")
    # The first request of a call replays nothing.
    (early,) = driver.replay_problems([_request([*opening, *replay])], [call])
    assert early.startswith("request 1 carries")
    # A request that matched no scripted step is an extra request, which
    # clause C0 reports; the rule judges the scripted ones, by their place.
    stray = _request([], matched=False)
    later = SentCall(READ_A, "call-1", 1, {})
    assert driver.replay_problems([stray, first, _request([*opening, *replay])], [later]) == []
    # The dialect states this rule, which is how a cell comes to assert it.
    assert driver.DeepSeekResponsesDialect().replay_problems([first], []) == []
    (stated,) = driver.DeepSeekResponsesDialect().replay_problems(
        [first, _request([*opening, *replay[1:]])], [call]
    )
    assert stated.startswith("request 2 carries")


# -- the envelopes -----------------------------------------------------------------

# Run by the locked interpreter: every event of a stream against the event
# type of the pinned SDK that its ``type`` names, with full validation.
_VALIDATE = """
import json
import sys
import typing

import openai
import pydantic
from openai.types.responses import ResponseStreamEvent

members = typing.get_args(typing.get_args(ResponseStreamEvent)[0])
by_type = {}
for member in members:
    for name in typing.get_args(member.model_fields["type"].annotation):
        by_type[name] = member
report = {"sdk": openai.__version__, "events": []}
for event in json.load(sys.stdin):
    member = by_type[event["type"]]
    try:
        member.model_validate(event)
        errors = []
    except pydantic.ValidationError as exc:
        errors = sorted(
            ".".join(str(part) for part in error["loc"]) + ": " + error["type"]
            for error in exc.errors()
        )
    report["events"].append([event["type"], member.__name__, errors])
print(json.dumps(report))
"""


def test_the_envelopes_validate_against_the_pinned_sdk_in_the_locked_interpreter(
    locked_worker_launch: launch.Interposer,
) -> None:
    dialect = driver.DeepSeekResponsesDialect()
    blank = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")
    seen_types: set[str] = set()

    terminals = {
        "Say": ("response.completed", "ResponseCompletedEvent"),
        "Call": ("response.completed", "ResponseCompletedEvent"),
        # A response a content filter stopped ends as an incomplete one; a
        # response that declines in a refusal part is a completed one.
        "content_filter": ("response.incomplete", "ResponseIncompleteEvent"),
        "refusal_part": ("response.completed", "ResponseCompletedEvent"),
    }
    steps: tuple[Step, ...] = (
        Say("the answer"),
        Call(READ_A),
        Refuse("content_filter"),
        Refuse("refusal_part"),
    )
    for step in steps:
        events = dialect.render(step, blank, "call-1").body
        # The test starts the locked interpreter itself, through the launcher
        # of the suite fixture: it runs under the harness, in mode guard.
        completed = locked_worker_launch.run(
            [str(DRIVER.interpreter), "-B", "-I", "-c", _VALIDATE],
            input=json.dumps(events),
            capture_output=True,
            text=True,
            timeout=100,
            check=False,
            env={"PYTHONNOUSERSITE": "1"},
        )
        assert (completed.returncode, completed.stderr) == (0, "")
        report = json.loads(completed.stdout)
        assert report["sdk"] == driver.pinned_sdk_version()
        assert [name for name, _member, _errors in report["events"]] == [
            event["type"] for event in events
        ]
        # Every event is a complete instance of the pinned SDK's own type ...
        for name, member, errors in report["events"][:-1]:
            assert errors == [], (name, member, errors)
            seen_types.add(member)
        # ... but for one field of the terminal: the pinned SDK requires a
        # cache-write counter that the vendor documents none of and that the
        # live route did not send. The envelope leaves it out on purpose.
        name, member, errors = report["events"][-1]
        assert (name, member) == terminals[getattr(step, "form", type(step).__name__)]
        assert errors == ["response.usage.input_tokens_details.cache_write_tokens: missing"]

    assert {"ResponseTextDeltaEvent", "ResponseFunctionCallArgumentsDoneEvent"} <= seen_types
    # The refusal part travels in the two events the pinned SDK types for it.
    assert {"ResponseRefusalDeltaEvent", "ResponseRefusalDoneEvent"} <= seen_types
    assert locked_worker_launch.journal is not None
    assert [(item["mode"], item["form"]) for item in locked_worker_launch.journal.launches()] == [
        ("guard", "-c")
    ] * len(steps)
    assert locked_worker_launch.violations() == []
    # The third refusal is no stream: a rejected request is answered with an
    # error object of the SDK's four fields, bills nothing and names no model.
    rejected = dialect.render(Refuse("http_policy"), blank, "call-1")
    assert (rejected.status, rejected.raw, rejected.usage, rejected.served_model) == (
        400,
        None,
        None,
        None,
    )
    assert set(rejected.body) == {"error"}
    assert set(rejected.body["error"]) == {"code", "message", "param", "type"}
    assert rejected.envelope == "deepseek_responses/deepseek.refusal_http_policy"


class _AnswersWhereItShouldRefuse(driver.DeepSeekResponsesDialect):
    """Renders a refusal as a plain answer: the provider never refuses."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        if isinstance(step, Refuse):
            step = Say("an answer where the script refuses", step.usage)
        return super().render(step, seen, call_id)


def test_a_refusal_that_is_never_sent_fails_the_cell_whatever_its_entry(
    tmp_path: Path,
) -> None:
    class Lying(driver.DeepSeekDriver):
        def _dialect(self, journal: launch.Journal, interpreter: str) -> Any:
            return _AnswersWhereItShouldRefuse(journal=journal, interpreter=interpreter)

    # The entry of the cell is strict: a provider side that answers instead
    # of refusing yields other codes, and the cell fails as a different
    # deviation instead of staying an expected failure.
    with pytest.raises(ledger.LedgerMismatch, match="different deviation"):
        runner.run_cell(
            Cell(COLUMN, "S23a", "raw"),
            tmp_path,
            preflight={DEEPSEEK_KIND: None},
            known=LEDGER,
            driver=Lying(),
        )


class _CutsTheTurnCount:
    """The backend of a cell behind a ceiling on the turn count that it states nowhere."""

    def __init__(self, inner: Any, ceiling: int) -> None:
        self._inner, self._ceiling = inner, ceiling

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def invoke(self, invocation: Any) -> Any:
        return self._inner.invoke(
            dataclasses.replace(invocation, max_turns=min(invocation.max_turns, self._ceiling))
        )


def test_a_silent_turn_ceiling_does_not_close_the_gap_of_the_suites_turn_count(
    played: Callable[[str], runner.Execution], tmp_path: Path
) -> None:
    ceiling = adapter._MAX_TOOL_ROUNDS

    class Clamping(driver.DeepSeekDriver):
        @contextlib.contextmanager
        def open(self, column: Any, **named: Any) -> Any:
            with super().open(column, **named) as bound:
                bound.backend = _CutsTheTurnCount(bound.backend, ceiling)
                yield bound

    cell = Cell(COLUMN, "S11", "raw")
    entry = LEDGER.entry(cell.id)
    assert entry is not None and set(entry.codes.values()) == {"G15"}

    # Today the adapter refuses the suite's turn count before it asks the
    # provider anything: the cell is the expected failure of the gap.
    honest = played("S11")
    assert honest.plan.max_turns > ceiling
    assert honest.observation.raised is not None
    assert honest.observation.provider.request_count == 0
    assert honest.report.codes == set(entry.codes)

    # The lying counterpart: the count is cut to the adapter's ceiling and
    # the call goes ahead. One completion needs one turn, the wire of the
    # route carries no turn control, and the result is a clean finish: no
    # code of the gap is left, and nothing says that a ceiling was applied.
    clamped = _played("S11", tmp_path, driver=Clamping())
    result = clamped.observation.result
    assert result is not None and result.error is None
    assert (result.stop_reason, result.final_message) == ("end_turn", TEXT)
    assert [request.turn_control for request in _matched(clamped)] == [None]
    assert not clamped.observation.provider.turn_control_field
    assert clamped.current == {"turn_limit": "hard"}
    column_wide = {item.code for item in LEDGER.column_wide if item.kind == DEEPSEEK_KIND}
    clean_finish = LEDGER.entry(Cell(COLUMN, "S01", "raw").id)
    assert clean_finish is not None
    # What the cell then shows is what every clean finish of the column
    # shows: the terminal name of another record.
    left = clamped.report.codes - column_wide
    assert left and left == set(clean_finish.codes) and set(clean_finish.codes.values()) == {"M05"}

    # So the ledger that such a change leaves has no code of the gap on its
    # one cell. Whether a ceiling that is applied in silence is the closing
    # state of the gap is not ruled, and until it is the integrity check
    # holds the gap open whatever the cell shows.
    document = yaml.safe_load(ledger.LEDGER_PATH.read_text(encoding="utf-8"))
    (known,) = (item for item in document["known_gaps"] if item.get("cell") == cell.id)
    known["codes"] = {code: "M05" for code in sorted(left)}
    document["registry"]["M05"]["cells"].append(
        {
            "cell": cell.id,
            "observed_under": dict(clamped.current),
            "closes_under": dict(clamped.current),
        }
    )
    found = ledger.integrity_problems(ledger.parse(document), cells.world())
    assert {item.code for item in found} == {"static:registry_closed_before_ruling"}
    assert [cell.id in item.detail for item in found] == [True]


def test_every_cell_of_the_column_is_graded_by_the_envelopes_it_used(
    played: Callable[[str], runner.Execution],
) -> None:
    text, tools = played("S01"), played("S04")

    loop = {"deepseek_responses/deepseek.completion", "deepseek_responses/deepseek.tool_call"}
    refusals = set(DRIVER.envelopes) - loop
    assert set(text.observation.provider.envelopes) == {"deepseek_responses/deepseek.completion"}
    assert set(tools.observation.provider.envelopes) == loop
    assert set(DRIVER.envelopes) == {
        name for name in envelopes.all_envelopes() if name.startswith("deepseek_responses/")
    }
    # The two envelopes of the tool loop rest on the vendor's documentation
    # and on the live adapter-level check; neither is a recording.
    assert {envelopes.load(name).grade for name in loop} == {"documented"}
    assert text.grade == tools.grade == "documented"
    # No refusal of the route was observed or is documented: the three
    # envelopes of a refusal are built to the types of the pinned SDK, and a
    # cell that is answered from one is graded so.
    assert len(refusals) == 3
    assert {envelopes.load(name).grade for name in refusals} == {"constructed"}
    assert played("S23a").grade == "constructed"
    # The schema answer of the planner shape travels in the prompt, once, and
    # in no provider field: what the route declares.
    assert SCENARIOS["S04"].governing == ("structured_output",)
    assert tools.current == {"structured_output": "prompt_only"}
    assert not any(request.schema_field for request in _matched(tools))
    assert not {code for code in tools.report.codes if code.startswith("request.content")}


# -- the module ends with the canary check --------------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
