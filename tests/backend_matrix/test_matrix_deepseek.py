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

from assay import Usage
from crucible.backends import deepseek as adapter
from crucible.backends.features import FeatureAdmissionError

from . import cells, declarations, envelopes, ledger, netguard, runner
from . import columns as matrix_columns
from .cells import Cell
from .columns import DEEPSEEK_KIND
from .drivers import MatrixSetupError
from .drivers import deepseek as driver
from .peers import launch
from .provider import LoopbackProvider, Reply, SeenRequest, SentCall
from .scenarios import SCENARIOS, TEXT, history_replayed
from .script import (
    READ_A,
    Call,
    Cut,
    Drop,
    Empty,
    Malformed,
    ProviderScript,
    Refuse,
    Say,
    ServedAs,
    Step,
)

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


def test_the_file_as_shipped_is_refused_for_its_missing_fingerprint_before_a_worker_starts(
    tmp_path: Path,
) -> None:
    # The admission cell asks for every capability a second time on the
    # routing file as shipped, and its pin says that routing refuses all
    # three. The pin holds the type of the refusal; this is what it is about.
    cell = Cell(COLUMN, "S51", "raw")
    pinned = cells.expected_admission(cell)
    capabilities = tuple(matrix_columns.ROLE_BY_CAPABILITY)
    assert {pinned[cells.as_shipped(name)] for name in capabilities} == {
        FeatureAdmissionError.__name__
    }
    assert [name for name, _path in COLUMN.routing] == sorted(matrix_columns.PRODUCTS)

    for product in matrix_columns.PRODUCTS:
        view = COLUMN.routed_by(product).as_shipped()
        # There the driver binds nothing to the routes.
        assert DRIVER.overlay(view, {}) == {} != DRIVER.overlay(COLUMN.routed_by(product), {})
        for capability in capabilities:
            directory = tmp_path / product / capability
            directory.mkdir(parents=True)
            # The one thing the shipped route lacks is the fingerprint of the
            # worker runtime, which it leaves to the run.
            with pytest.raises(
                FeatureAdmissionError,
                match=r"rejected: an exact expected_worker_runtime_fingerprint is required$",
            ):
                with DRIVER.open(
                    view,
                    capability=capability,
                    lens="raw",
                    script=ProviderScript(()),
                    tmp_path=directory,
                ):
                    raise AssertionError("routing built the adapter from the file as shipped")
            # Routing refused before it built anything: no worker was
            # started, not even to be probed.
            assert launch.Journal(directory / "worker-harness").launches() == []
    assert os.environ.get("DEEPSEEK_API_KEY") is None


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
    assert not reply.call_malformed
    # The arguments of a function call travel as text in a string, so a text
    # that is no JSON object travels too, through the same envelope; the
    # reply says that the call names no arguments.
    cut = dialect.render(Call(READ_A, malformed='{"file_path": "notes'), seen, "call-8")
    carried = [
        item["arguments"]
        for item in cut.body[-1]["response"]["output"]
        if item["type"] == "function_call"
    ]
    assert carried == ['{"file_path": "notes']
    assert cut.call_malformed and cut.call is not None and cut.call[2] == {}
    assert cut.envelope == reply.envelope
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
        # A response the provider ended at the cap on its output tokens ends
        # as an incomplete one too; a response in which nothing was produced
        # is a completed one.
        "Cut": ("response.incomplete", "ResponseIncompleteEvent"),
        "Empty": ("response.completed", "ResponseCompletedEvent"),
    }
    steps: tuple[Step, ...] = (
        Say("the answer"),
        Call(READ_A),
        Refuse("content_filter"),
        Refuse("refusal_part"),
        Cut("what the cap left"),
        Empty(),
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
    # Two replies end on the terminal of an incomplete response, and only
    # the reason in that terminal tells them apart: the cap on the output
    # tokens, or the content filter. The cells of the column cannot stand
    # for this: the adapter is handed no reason, so the cell of a cut shows
    # the same codes for either. A response the cap ended carries the text
    # that was produced in the output of its terminal; a response in which
    # nothing was produced completes with no output item and no output
    # token.
    cut = dialect.render(Cut("what the cap left"), blank, "call-1")
    filtered = dialect.render(Refuse("content_filter"), blank, "call-1")
    silent = dialect.render(Empty(), blank, "call-1")
    ended, stopped, nothing = (reply.body[-1]["response"] for reply in (cut, filtered, silent))
    assert (ended["status"], ended["incomplete_details"]) == (
        "incomplete",
        {"reason": "max_output_tokens"},
    )
    assert (stopped["status"], stopped["incomplete_details"]) == (
        "incomplete",
        {"reason": "content_filter"},
    )
    (message,) = ended["output"]
    assert (message["status"], message["content"][0]["text"]) == ("incomplete", "what the cap left")
    assert cut.envelope == f"deepseek_responses/deepseek.{driver.OUTPUT_CUT}"
    assert cut.usage == Cut("what the cap left").usage
    assert (nothing["status"], nothing["output"]) == ("completed", [])
    assert "incomplete_details" not in nothing
    assert nothing["usage"]["output_tokens"] == 0
    assert silent.envelope == f"deepseek_responses/deepseek.{driver.EMPTY_COMPLETION}"
    assert silent.usage == Empty().usage and len(silent.body) == 3
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
    reasoning = {"deepseek_responses/deepseek.reasoning_only"}
    unended = f"deepseek_responses/deepseek.{driver.MALFORMED}"
    cut = f"deepseek_responses/deepseek.{driver.OUTPUT_CUT}"
    silent = f"deepseek_responses/deepseek.{driver.EMPTY_COMPLETION}"
    refusals = set(DRIVER.envelopes) - loop - reasoning - {unended, cut, silent}
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
    # Nor was a response that reasons and holds no message: no reply of the
    # route was recorded.
    assert {envelopes.load(name).grade for name in reasoning} == {"constructed"}
    # That a stream of the route ends in order before its terminal is stated
    # by no source either: the envelope of that reply is constructed.
    assert envelopes.load(unended).grade == "constructed"
    # The vendor names the terminal of an incomplete response and does not
    # say when it is sent, and no source states a response without output:
    # the envelopes of a reply cut at the cap and of a reply in which
    # nothing was produced are constructed.
    assert {envelopes.load(name).grade for name in (cut, silent)} == {"constructed"}
    # The schema answer of the planner shape travels in the prompt, once, and
    # in no provider field: what the route declares.
    assert SCENARIOS["S04"].governing == ("structured_output",)
    assert tools.current == {"structured_output": "prompt_only"}
    assert not any(request.schema_field for request in _matched(tools))
    assert not {code for code in tools.report.codes if code.startswith("request.content")}


# -- what a request carries beside prompt, model turns and tool results ----------------


def _input(*items: tuple[str, str | None], matched: bool = True) -> SeenRequest:
    """A request whose input items are ``(kind, role)``, as the dialect reads them."""

    dialect = driver.DeepSeekResponsesDialect()
    blank = dialect.parse("POST", dialect.path, {}, b"{}", credential="")
    evidence = {"input": [{"kind": kind, "call_id": None, "role": role} for kind, role in items]}
    return dataclasses.replace(blank, evidence=evidence, matched=matched)


def test_an_input_item_that_is_not_the_workflows_is_named() -> None:
    opening = [("system", "system"), ("user", "user")]
    round_ = [("reasoning", None), ("function_call", None), ("function_call_output", None)]
    said = ("message", "assistant")
    word = ("user", "user")
    first = _input(*opening)

    def named(*items: tuple[str, str | None], **more: Any) -> list[str]:
        return driver.foreign_messages([first, _input(*items, **more)])

    # The prompt, then items of the model and outputs of its calls: nothing.
    assert named(*opening, *round_, *round_) == []
    assert named(*opening, *round_, said) == []
    # A word of the client's own after a round, as a plain message or as a
    # typed one, and another opening than the first request's: each is named.
    for told in (
        [*opening, *round_, word],
        [*opening, *round_, ("message", "user")],
        [*opening, *round_, ("message", "developer")],
        [*opening, word, *round_],
        [opening[0], *round_],
    ):
        (problem,) = named(*told)
        assert problem.startswith("request 2 carries ")
    (wide,) = driver.foreign_messages([_input(*opening, word)])
    assert "more than the one system and the one user message" in wide
    # A request that matched no scripted step is not judged.
    assert named(*opening, *round_, word, matched=False) == []
    # The dialect reads the role of an item off the wire, beside its kind.
    dialect = driver.DeepSeekResponsesDialect()
    body = {"input": [{"role": "user", "content": "x"}, {"type": "message", "role": "assistant"}]}
    seen = dialect.parse("POST", dialect.path, {}, json.dumps(body).encode(), credential="x")
    assert [(item["kind"], item["role"]) for item in seen.evidence["input"]] == [
        ("user", "user"),
        ("message", "assistant"),
    ]
    assert dialect.foreign_messages([first]) == []


# -- broken terminals: a stream without its terminal, a stream that is cut ----------


def test_a_stream_without_its_terminal_and_a_cut_stream_are_what_the_wire_carries() -> None:
    dialect = driver.DeepSeekResponsesDialect()
    asked = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")
    text = "an answer that is cut off"

    unended = dialect.render(Malformed(text), asked, "call-1")
    cut = dialect.render(Drop(text), asked, "call-2")
    whole = dialect.render(Say(text), asked, "call-2")

    # A stream that ends without its terminal: every event of a plain answer
    # but the terminal, and the body ends in order. The usage is null until
    # the terminal, so none is billed; the first events name the model.
    kinds = [event["type"] for event in unended.body]
    assert kinds == [event["type"] for event in whole.body][:-1]
    assert not set(kinds) & set(adapter._TERMINAL_TYPES)
    assert unended.envelope == f"deepseek_responses/deepseek.{driver.MALFORMED}"
    assert (unended.status, unended.usage, unended.served_model, unended.cut) == (
        200,
        None,
        "asked",
        None,
    )
    assert all(event.get("response", {}).get("usage") is None for event in unended.body)
    assert unended.raw == driver.server_sent_events(unended.body)
    # A stream that is cut: the bytes of a plain answer, of which the events
    # up to the text delta arrive whole and the event after it is cut. The
    # terminal never arrives: nothing is billed, the model is named.
    assert cut.envelope == "deepseek_responses/deepseek.completion"
    assert (cut.status, cut.usage, cut.served_model) == (200, None, "asked")
    assert cut.raw is not None and cut.cut is not None and 0 < cut.cut < len(cut.raw)
    arrived = cut.raw[: cut.cut].split(b"\n\n")
    events = [json.loads(frame.split(b"data: ", 1)[1]) for frame in arrived[:-1]]
    assert events[-1]["type"] == "response.output_text.delta" and events[-1]["delta"] == text
    assert not {event["type"] for event in events} & set(adapter._TERMINAL_TYPES)
    assert arrived[-1].startswith(b"event: ") and b"response.completed" not in cut.raw[: cut.cut]
    with pytest.raises(json.JSONDecodeError):
        json.loads(arrived[-1].split(b"data: ", 1)[1])
    with pytest.raises(MatrixSetupError, match="inside its body only"):
        dialect.render(Drop(text, after="headers"), asked, "call-3")


class _AnswersWhole(driver.DeepSeekResponsesDialect):
    """Renders a broken terminal as a plain answer: the stream ends on its terminal."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        if isinstance(step, (Malformed, Drop)):
            step = Say(step.text)
        return super().render(step, seen, call_id)


@pytest.mark.parametrize("scenario", ["S26", "S27"])
def test_a_stream_that_ends_on_its_terminal_fails_the_cell_of_a_broken_terminal(
    scenario: str, tmp_path: Path, played: Callable[[str], runner.Execution]
) -> None:
    cell = Cell(COLUMN, scenario, "raw")
    entry = LEDGER.entry(cell.id)
    wide = {scope.code for scope in LEDGER.column_wide if scope.kind == DEEPSEEK_KIND}
    honest = played(scenario)
    provider, result = honest.observation.provider, honest.observation.result

    # The honest cell shows what the ledger expects of it, its entry or
    # nothing, beside what the column owes on every cell, ...
    assert honest.report.seam_problems == ()
    assert honest.report.codes == (set() if entry is None else set(entry.codes)) | wide
    # ... from one request, answered with the stream the scenario scripts:
    # the events of a plain answer before the terminal, or the stream of a
    # plain answer that is cut. The events that arrived named the model and
    # stated no usage ...
    assert (provider.request_count, provider.extra, provider.unconsumed) == (1, 0, 0)
    behaviour = driver.MALFORMED if scenario == "S26" else "completion"
    assert provider.envelopes == [f"deepseek_responses/deepseek.{behaviour}"]
    assert (provider.billed, provider.served_models) == ([], [honest.reference.model])
    # ... and for the cause the scenario scripts, as far as a result can
    # show one. The codes of an entry here are those of any result the
    # adapter builds from a failure of its worker, whatever failed. Of this
    # one the adapter says, in the error and in the terminal of the trace,
    # that the worker failed the qualification of what it read. It does not
    # say why, so the stream the one request was answered with is what
    # tells the two cells apart, and a call that ended in another branch of
    # the adapter fails here by name.
    assert result is not None and result.error is not None
    assert type(result.error).__name__ == "TransportError" and not result.error.retriable
    assert (
        str(result.error.message)
        == "DeepSeek worker qualification failed with DeepSeekSDKQualificationError"
    )
    trace = result.execution_trace
    assert trace is not None
    assert [(event.kind, event.name) for event in trace.events][-1] == (
        "terminal",
        "worker_qualification_error",
    )

    # The lying counterpart. The entry of the cell is strict: a provider
    # side whose stream ends on its terminal yields a clean finish where the
    # scenario expects the transport error of a turn without its terminal,
    # and the cell fails as a different deviation instead of staying an
    # expected failure.
    with pytest.raises(ledger.LedgerMismatch, match="different deviation") as refused:
        runner.run_cell(
            cell,
            tmp_path,
            preflight={DEEPSEEK_KIND: None},
            known=LEDGER,
            driver=_with_dialect(_AnswersWhole),
        )
    assert "'kind:none!=transport'" in str(refused.value).splitlines()[0]


# -- a route without its credential ---------------------------------------------------


def test_a_play_without_the_credential_has_no_key_variable_while_its_backend_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variable = matrix_columns.ADAPTER_CREDENTIAL_ENV[DEEPSEEK_KIND]
    role = matrix_columns.ROLE_BY_CAPABILITY["text"]
    path = COLUMN.routing_path(matrix_columns.COLUMN_PRODUCT)

    # The route names no key: the adapter reads the variable of the kind at
    # each call, and routing never reads it.
    assert declarations.credential_variable(path, role) is None
    assert variable not in matrix_columns.SCRIPTED_ENV
    # Whatever a machine holds under that name is taken away for the play
    # and put back after it.
    monkeypatch.setenv(variable, "a-key-of-this-machine")
    with DRIVER.open(
        COLUMN,
        capability="text",
        lens="raw",
        script=ProviderScript(()),
        tmp_path=tmp_path / "without",
        without_credential=True,
    ):
        assert variable not in os.environ
    assert os.environ[variable] == "a-key-of-this-machine"
    with DRIVER.open(
        COLUMN,
        capability="text",
        lens="raw",
        script=ProviderScript(()),
        tmp_path=tmp_path / "with",
    ):
        assert os.environ[variable] == matrix_columns.CANARY_CREDENTIAL


# -- the state of the served-model pin on the routes of the kind ----------------------


def test_the_state_of_the_served_model_pin_is_read_from_every_routing_file() -> None:
    # A reply under another model is judged by the state of the route extra
    # that pins the served model, as routing states it: read here from the
    # shipped files, on every role a cell builds and on both products. A
    # change of this state changes what that scenario asks of the column.
    extra = declarations.SERVED_MODEL_EXTRA
    states = {
        (column.id, product, role): declarations.route_state(column, role, extra, product)
        for column in (COLUMN,)
        for product, _path in column.routing
        for role in matrix_columns.ROLE_BY_CAPABILITY.values()
    }

    assert states and set(states.values()) == {declarations.NOT_ADMITTED}
    assert len(states) == 6


# -- a reply under another model: what the entry of its cell does not show ------------

# The record of the ledger whose cell the proof below plays, and the code of
# that record that keeps the accounting clause from comparing the tokens.
REFUSED_REPLY_GAP = "G25"
_TOKENS_FLAGGED = "flag:token_usage_complete=False"
_REFUSED_REPLY = Cell(COLUMN, cells.SERVED_UNDER_ANOTHER_MODEL, "raw").id


def test_the_result_of_a_reply_refused_for_its_model_reports_the_tokens_the_provider_billed(
    played: Callable[[str], runner.Execution], request: pytest.FixtureRequest
) -> None:
    """Held here by name, because no code of the cell shows it.

    The adapter refuses a reply under another model than its route asked
    for, types the refusal a transport error and flags the token usage of
    the result incomplete: codes of the cell's entry (known gap G25). Of a
    result that is flagged so, clause C6 takes any number below the bill
    for honest, down to none: while the entry stands no clause compares
    what the result reports, and a result that had lost the usage of the
    billed reply would be the same expected failure. This test compares
    it: the provider billed one reply, and the result reports its tokens.
    It holds before and after a fix of the record, and goes with its lying
    counterparts once no entry lists the flag.
    """

    # The verifier requires this test by name: the entry of the record on
    # this scenario leans on it, and a run without it is refused. No entry
    # attributes the flag to that record on a cell this test does not play.
    assert request.node.name in cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[DEEPSEEK_KIND]]
    flagged = {
        entry.cell
        for entry in LEDGER.entries
        if entry.codes.get(_TOKENS_FLAGGED) == REFUSED_REPLY_GAP
    }
    assert flagged <= {_REFUSED_REPLY}

    execution = played(cells.SERVED_UNDER_ANOTHER_MODEL)
    result = execution.observation.result

    # The play is a refusal of the one reply: an error, the provider asked
    # once, nothing dispatched.
    assert execution.report.seam_problems == () and result is not None
    for clause in ("C0", "C10"):
        assert execution.report.outcomes[clause].status == "ok", clause
    assert result.error is not None
    assert runner.not_the_tokens_of_the_one_billed_reply(execution) == []


def test_the_proof_of_the_refused_reply_fails_on_a_result_that_lost_its_usage(
    played: Callable[[str], runner.Execution],
) -> None:
    # The lying counterparts of the proof above: the result of the cell with
    # its usage made up. Dropped whole, halved, one token short: the result
    # flags its token usage incomplete, so the contract judges each to the
    # codes of the honest play and the comparator is handed the same, which
    # makes the lying result the same expected failure; the proof names
    # each. A provider that billed nothing, or two replies, is not the one
    # billed reply the proof is about.
    honest = played(cells.SERVED_UNDER_ANOTHER_MODEL)
    result = honest.observation.result
    assert result is not None and _TOKENS_FLAGGED in honest.report.codes, (
        "the result no longer flags its token usage incomplete, so the cell compares "
        "the tokens itself: these counterparts go with the entry"
    )
    usage = result.usage
    assert runner.not_the_tokens_of_the_one_billed_reply(honest) == []
    for lost in (
        Usage(),
        dataclasses.replace(
            usage, input_tokens=usage.input_tokens // 2, output_tokens=usage.output_tokens // 2
        ),
        dataclasses.replace(usage, input_tokens=usage.input_tokens - 1),
        dataclasses.replace(usage, output_tokens=usage.output_tokens - 1),
    ):
        lying = runner.judged_again(honest, usage=lost)
        assert runner.handed_to_the_comparator(lying) == runner.handed_to_the_comparator(honest)
        assert runner.not_the_tokens_of_the_one_billed_reply(lying) != []
    (billed,) = honest.observation.provider.billed
    for bill in ([], [billed, billed]):
        provider = dataclasses.replace(honest.observation.provider, billed=bill)
        other = dataclasses.replace(
            honest, observation=dataclasses.replace(honest.observation, provider=provider)
        )
        assert runner.not_the_tokens_of_the_one_billed_reply(other) != []
    # More than the bill is no lower bound: the cell itself names that,
    # whatever the result flags.
    more = runner.judged_again(
        honest, usage=dataclasses.replace(usage, input_tokens=usage.input_tokens + 1)
    )
    assert more.report.codes == honest.report.codes | {"usage.input_tokens:gt"}
    assert runner.not_the_tokens_of_the_one_billed_reply(more) != []


# -- the module ends with the canary check --------------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
