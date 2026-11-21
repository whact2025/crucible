# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The Kimi columns: their cells, and the proof of their driver.

The cells run through the real adapter, its project-locked pin, the locked
worker interpreter with the vendor's agent SDK and a loopback socket: the
starter set on the two diagnostic columns, and the canary, one plain call and
the admission cell on the blocked column. Every test of this module needs the
locked worker environment (``backends/kimi_worker/.venv``); without it they
fail with the command that installs it, and they carry the marker a developer
deselects them by.

The remaining tests show that the driver's seam can fail: a worker that is
not redirected, a redirect for another origin, a dialect that does not see
the worker launches, that answers under a fixed model id or does not read the
model on the wire. They show that a route which opts into the cost upper
bound is called as its routing file asks, and what happens when it is not;
that a guarded result of such a route is not called rankable and its cell
says so, with or without a cache-read count in the reply; that the cost
interval of such a route is held to what the provider billed, and that
accounting which loses a response is a deviation of the cell; why the strict
route rejects the plain reply, and that it answers once the reply carries a
cache-read count; that the probe and the request launch run under the
harness and leave the environment as it was; that a TLS trust path of the
machine reaches no worker; that a tool round is replayed with its reasoning
and that a cell deviates when it is not; what the admission cell is held to;
and that the envelopes parse through the pinned SDK in the locked
interpreter.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from assay import Usage
from crucible.backends import kimi as adapter
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY

from . import cells, declarations, envelopes, ledger, netguard, runner
from . import columns as matrix_columns
from .cells import Cell
from .columns import KIMI_KIND
from .contract import ADMITTED, SecondCall
from .drivers import MatrixSetupError
from .drivers import kimi as driver
from .drivers import openai_compat as chat
from .drivers.base import GUARD_POLICY
from .peers import launch, worker_harness
from .provider import LoopbackProvider, Reply, SeenRequest, SentCall
from .scenarios import SCENARIOS, history_replayed
from .script import READ_A, Call, ProviderScript, Say, ServedAs, Step

# Every test here runs the locked worker: a missing environment is an error.
pytestmark = pytest.mark.matrix_worker

LEDGER = ledger.load()
CELLS = cells.cells_of((KIMI_KIND,))
COLUMNS = {
    column.role: column for column in matrix_columns.real_columns() if column.kind == KIMI_KIND
}
PRIMARY, SECONDARY, BLOCKED = (COLUMNS[role] for role in ("primary", "secondary", "blocked"))
PRODUCTION = driver.production_endpoint()
HOST = PRODUCTION.removeprefix("https://").split("/")[0]
DRIVER = driver.KimiDriver()
WORKER = "crucible_kimi_worker"
WIDE = {scope.code for scope in LEDGER.column_wide if scope.kind == KIMI_KIND}


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)


def _played(
    scenario: str, tmp_path: Path, lens: str = "raw", column: Any = PRIMARY, **more: Any
) -> runner.Execution:
    cell = Cell(column, scenario, lens)
    more.setdefault("differences", LEDGER.differences_for(cell.id))
    return runner.execute(cell, tmp_path, **more)


# A cell takes several seconds here: the worker imports the vendor's stack
# without bytecode. The proofs below read one play of a cell.
_PLAYED: dict[tuple[str, str, str], runner.Execution] = {}
# The code of a result the limit guard let stand and does not call rankable.
UNRANKED = "guard:not_rankable"


@pytest.fixture()
def played(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., runner.Execution]:
    """A cell as the matrix plays it, played once for the proofs of this module."""

    def play(scenario: str, lens: str = "raw", column: Any = PRIMARY) -> runner.Execution:
        key = (column.role, scenario, lens)
        if key not in _PLAYED:
            _PLAYED[key] = _played(
                scenario,
                tmp_path_factory.mktemp(f"kimi-{column.role}-{scenario}-{lens}"),
                lens,
                column,
            )
        return _PLAYED[key]

    return play


def _verdict(execution: runner.Execution, known: ledger.Ledger = LEDGER) -> str:
    """What the comparator says of a play that is not a pass or an expected failure."""

    report = execution.report
    with pytest.raises(ledger.LedgerMismatch) as refused:
        ledger.compare(
            known,
            execution.cell.id,
            report.codes,
            current=execution.current,
            evaluated=report.evaluated(),
            unasked=report.unasked,
            report=report.describe(),
        )
    return str(refused.value).splitlines()[0]


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

    # The request launch, and any probe, left the worker project as it was.
    assert _tree(project) == before
    (request,) = execution.observation.provider.requests
    assert execution.report.seam_problems == () and execution.report.codes == frozenset()
    assert request.matched and request.path == "/v1/chat/completions"
    # The request names the command-line runtime and the client the worker's
    # lock pins. That the worker built it, and not this process, is what the
    # launch journal shows below.
    assert request.user_agent == f"KimiCLI/{driver.pinned_version('kimi-cli')}"
    assert request.evidence["sdk"] == driver.pinned_version("openai")
    assert request.credential_ok
    # The client addressed production; the redirect in the worker answered it.
    assert request.host == HOST and not netguard.is_loopback_host(request.host)
    assert request.model == PRIMARY.model != ""
    assert request.evidence["stream"] is True and request.evidence["include_usage"] is True

    # The harness journalled every launch before its target ran: the request
    # launch in mode redirect, and before it at most the probe of the runtime,
    # which the adapter asks once per process.
    launches = request.evidence["launches"]
    assert [item["mode"] for item in launches] in (["redirect"], ["guard", "redirect"])
    for item in launches:
        assert item["pid"] != os.getpid()
        assert launch.is_locked_interpreter(item["executable"])
        assert Path(item["executable"]) == DRIVER.interpreter
        assert item["guard"] and item["isolated"] and item["no_bytecode"]
        assert (item["form"], item["target"]) == ("-m", WORKER)
    request_launch = launches[-1]
    assert request_launch["arguments"] == []
    assert PRODUCTION.startswith(request_launch["redirect"]["origin"])
    assert netguard.is_loopback_host(
        request_launch["redirect"]["target"].removeprefix("http://").split(":")[0]
    )

    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    assert result.execution_trace.served_model == PRIMARY.model
    # The route names its endpoint, and the trace records that one.
    assert execution.reference.endpoint == PRODUCTION == result.execution_trace.endpoint


def test_the_route_is_the_shipped_one_and_its_pin_is_the_installed_workers(
    tmp_path: Path, locked_worker_launch: launch.Interposer
) -> None:
    project = DRIVER.interpreter.parents[2]
    before = _tree(project)
    suite_seam = adapter.subprocess
    assert suite_seam.matrix_interposer is locked_worker_launch
    assert DRIVER.overlay(PRIMARY, {}) == {} and KIMI_KIND not in matrix_columns.ALLOWED_OVERLAY

    with DRIVER.open(
        PRIMARY,
        capability="text",
        lens="raw",
        script=ProviderScript((Say("unused"),)),
        tmp_path=tmp_path,
    ) as bound:
        # Built by routing on the shipped file, with nothing overlaid.
        assert bound.backend.name == PRIMARY.adapter == "kimi"
        assert bound.backend.features == declarations.routed(PRIMARY)
        assert bound.tool_path == "mcp_bridge" and bound.secrets == ()
        assert bound.journal.directory in bound.log_dirs
        # One harness per launch: the driver's interposer replaced the suite's.
        seam = adapter.subprocess
        assert seam.matrix_interposer is not locked_worker_launch
        assert seam.matrix_real is suite_seam.matrix_real
        # The adapter pinned the installed worker when it was built, and the
        # pin is the one the public capture function reports: probed under
        # the harness, in a process of its own.
        identity = DRIVER.capture(bound.journal)
        assert bound.backend.worker_runtime_fingerprint == identity.runtime_fingerprint
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", identity.runtime_fingerprint)
        assert Path(identity.logical_worker_python) == DRIVER.interpreter
        assert identity.package_versions.openai == driver.pinned_version("openai")
        assert identity.package_versions.kimi_cli == driver.pinned_version("kimi-cli")
        (probe,) = bound.journal.launches()
        assert (probe["mode"], probe["target"], probe["arguments"]) == (
            "guard",
            WORKER,
            ["--probe"],
        )
        assert bound.children() == (probe["pid"],) and probe["pid"] != os.getpid()
        # Captured again, the worker is probed again and the identity is the
        # same one: a probe this process passed before is not taken on trust.
        assert DRIVER.capture(bound.journal) == identity
        assert [item["arguments"] for item in bound.journal.launches()] == [["--probe"]] * 2
        # A probe that did not run under the harness of the journal it is
        # captured for is refused: the pin would rest on an unguarded launch.
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(launch, "at_subprocess", lambda *_args: _Nothing())
            with pytest.raises(MatrixSetupError, match="no launch was journalled"):
                DRIVER.capture(launch.Journal(tmp_path / "another-journal"))

    assert adapter.subprocess is suite_seam
    # The harness left the worker project as it was: nothing was written
    # there, no bytecode either, so the pin still holds.
    assert _tree(project) == before
    # Nothing of the cell went through the suite's own interposer.
    assert locked_worker_launch.journal is None


class _Nothing:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


def test_the_preflight_of_the_session_sees_the_environment_a_cell_sees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A developer's override of the worker interpreter is one of the variables
    # the per-test fixture deletes, so no cell sees it; the adapter's capture
    # function refuses to pin the worker while it is set.
    override = "CRUCIBLE_KIMI_WORKER_PYTHON"
    assert override in matrix_columns.CLEARED_ENV and override not in os.environ
    monkeypatch.setenv(override, str(DRIVER.interpreter))

    with pytest.raises(MatrixSetupError, match="rejects ambient"):
        DRIVER.preflight()
    assert runner.preflight_all()[KIMI_KIND] is None
    assert os.environ[override] == str(DRIVER.interpreter)

    # An environment that is not installed is an error that names the command.
    class Absent(driver.KimiDriver):
        @property
        def interpreter(self) -> Path:
            return Path("/nowhere/backends/kimi_worker/.venv/bin/python")

    monkeypatch.delenv(override)
    with pytest.raises(MatrixSetupError, match="does not exist; install it with") as missing:
        Absent().preflight()
    assert driver.INSTALL in str(missing.value)
    # A proof that opens a column itself gets the same error, not the
    # adapter's own.
    with (
        pytest.raises(MatrixSetupError, match="does not exist; install it with") as unopened,
        Absent().open(
            PRIMARY,
            capability="text",
            lens="raw",
            script=ProviderScript(()),
            tmp_path=tmp_path,
        ),
    ):
        pass
    assert driver.INSTALL in str(unopened.value)

    # And an interpreter the adapter does not start is not driven: the launch
    # interposer would not be in front of what the adapter runs.
    class Elsewhere(driver.KimiDriver):
        @property
        def interpreter(self) -> Path:
            return Path(sys.executable)

    with pytest.raises(MatrixSetupError, match="not the interpreter of its worker environment"):
        Elsewhere().preflight()


def test_a_tls_trust_path_of_the_machine_never_reaches_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The adapter hands the TLS trust paths of its environment on to the
    # worker, and the client in the worker reads them when it is built. The
    # per-test environment deletes them, so no cell sees one.
    trust = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    assert set(trust) <= set(matrix_columns.CLEARED_ENV)
    assert not set(trust) & set(os.environ)

    # What a cell would show with one: a path of the machine that names no
    # file ends the worker before its first request, on a route that is fine.
    monkeypatch.setenv(trust[0], str(tmp_path / "a-bundle-of-the-machine.pem"))
    with_path = _played("S00", tmp_path / "machine")
    assert with_path.observation.provider.request_count == 0
    assert with_path.report.codes == {"request.unconsumed"}
    # Cleared as the fixture of every cell clears it, the canary is the
    # worker's own request again.
    runner.clear_environment(monkeypatch)
    assert not set(trust) & set(os.environ)
    cleared = _played("S00", tmp_path / "cleared")
    assert cleared.report.codes == frozenset() and cleared.report.seam_problems == ()


def test_the_probe_and_the_request_launch_run_under_the_harness(tmp_path: Path) -> None:
    # The adapter keeps a passed probe for the life of the process. With that
    # memory cleared, the cell shows both launches of a call.
    adapter._qualify_worker_runtime.cache_clear()

    execution = _played("S03", tmp_path)

    requests = _matched(execution)
    assert len(requests) == 3 and execution.report.seam_problems == ()
    launches = requests[-1].evidence["launches"]
    # One probe in mode guard, then one worker for the whole call: the SDK in
    # the worker owns the tool loop, so three provider rounds are one launch.
    assert [(item["mode"], item["arguments"]) for item in launches] == [
        ("guard", ["--probe"]),
        ("redirect", []),
    ]
    assert len({item["pid"] for item in launches}) == 2
    assert execution.observation.settlement.children_alive == ()
    assert execution.observation.result is not None
    assert execution.observation.result.error is None


class _GuardOnly(driver.KimiDriver):
    """No launch is redirected: the worker would reach for the real provider."""

    def _redirect(self, provider: LoopbackProvider) -> None:
        return None


class _OtherOrigin(driver.KimiDriver):
    """The redirect answers another origin than the one the worker addresses."""

    def _redirect(self, provider: LoopbackProvider) -> launch.Redirect:
        redirect = super()._redirect(provider)
        assert redirect is not None
        return dataclasses.replace(redirect, origin="https://api.moonshot.invalid")


@pytest.mark.parametrize(
    ("liar", "stopped_by"),
    [
        (_GuardOnly, rf"network guard in worker process \d+: .*{re.escape(HOST)}"),
        (_OtherOrigin, rf"origin redirect in worker process \d+: .*{re.escape(HOST)}"),
    ],
    ids=["no-redirect", "redirect-for-another-origin"],
)
def test_a_worker_that_is_not_redirected_never_reaches_the_provider_and_fails_the_cell(
    liar: type[driver.KimiDriver], stopped_by: str, tmp_path: Path
) -> None:
    # The worker swallows what the guard raises and the adapter types the
    # failure as a result. The journal of the harness is what decides.
    with pytest.raises(MatrixSetupError, match="reached beyond the scripted provider"):
        runner.execute(Cell(PRIMARY, "S00", "raw"), tmp_path, driver=liar())

    violations = launch.Journal(tmp_path / "worker-harness").violations()
    assert violations and all(re.search(stopped_by, item) for item in violations)
    # Nothing left this process either.
    assert netguard.take_violations() == ()


# -- a route that opts into the cost upper bound is called as it asks --------------


class _PlainGuard(driver.KimiDriver):
    """Binds the plain diagnostic limit policy, whatever the route asks for."""

    def limits(self, column: Any, role: str, lens: str, tmp_path: Path) -> dict[str, Any]:
        return {"limit_policy": GUARD_POLICY} if lens == "guarded" else {}


def test_a_route_with_the_cost_bound_gets_its_budget_and_its_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, played: Callable[..., runner.Execution]
) -> None:
    extra, role = matrix_columns.COST_BOUND_EXTRA, matrix_columns.ROLE_BY_CAPABILITY["text"]
    # Read from the routing files: both diagnostic routes carry the opt-in on
    # every role a cell builds, the strict route on none.
    for column, opted in ((PRIMARY, True), (SECONDARY, True), (BLOCKED, None)):
        for name in matrix_columns.ROLE_BY_CAPABILITY.values():
            assert declarations.route_extra(column, name, extra) is opted, (column.id, name)
        budget = matrix_columns.ROUTE_BUDGET_USD if opted else None
        assert declarations.route_budget(column, role) == budget

    raw, guarded = played("S01"), played("S01", "guarded")
    assert raw.route_budget == guarded.route_budget == matrix_columns.ROUTE_BUDGET_USD
    # The guard was bound with the policy the routing file names and kept its
    # authority: one physical attempt, a settled bound, a second call answered.
    terminal = guarded.observation.limit_terminal
    assert terminal is not None and terminal.terminal_reason == "diagnostic_cost_upper_bound"
    assert terminal.resolution.diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
    assert terminal.physical_attempts == 1 and terminal.diagnostic_cost_authority
    for execution in (raw, guarded):
        result = execution.observation.result
        assert result is not None and result.error is None
        # A cost interval: the lower bound as cost, the upper bound charged.
        bounds = result.usage.cost_bounds
        assert bounds is not None and bounds.coverage_complete
        assert result.usage.cost_usd == bounds.lower_bound_usd < bounds.upper_bound_usd
        assert [segment.cache_read_tokens for segment in bounds.segments] == [None]
    # The flags of the interval, and in the guarded lens what the guard says
    # of a result under this policy (the tests below).
    assert raw.report.codes == WIDE and guarded.report.codes == WIDE | {UNRANKED}

    # Called without the budget the route asks for, the adapter refuses
    # before transport: no request, and a raise where a result is owed.
    with monkeypatch.context() as patched:
        patched.setattr(declarations, "route_budget", lambda *_args, **_more: None)
        unbudgeted = _played("S01", tmp_path / "no-budget")
    assert unbudgeted.route_budget is None
    assert unbudgeted.report.codes == {
        f"raise:ValueError@before_request/{adapter.__name__}",
        "request.unconsumed",
    }
    # Bound with the plain diagnostic policy, the limit guard refuses the
    # adapter's opt-in when routing binds it: nothing is built, so there is
    # no backend to call a second time either.
    plain = _played("S01", tmp_path / "plain-guard", "guarded", driver=_PlainGuard())
    assert plain.report.codes == {
        "raise:InvocationLimitContractError@before_request/crucible.backends.limits",
        "request.unconsumed",
        "guard:second_call_failed",
    }
    # The strict route asks for neither, and gets neither.
    strict = played("S01", "guarded", BLOCKED)
    assert strict.route_budget is None
    assert strict.observation.limit_terminal is not None
    assert strict.observation.limit_terminal.resolution.diagnostic_cost_policy is None


# -- under the cost policy the guard never calls a result rankable -----------------


class _WithCacheReadCount(driver.MoonshotChatDialect):
    """Answers as the provider answers a request that read from its cache.

    The final statistics then carry the count of cache reads, under the
    prompt details and as an equal alias beside the totals: the key names of
    the cache-hit reply recorded live for the model of the columns (the
    recorded statistics are in ``tests/test_backends_kimi_k3_stream_contract.py``).
    The count is scripted as zero, so what the provider bills stays what the
    scripted usage says. No cell is answered this way: a first request reads
    nothing from the cache, and the envelopes of the columns are that reply.
    """

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        chunks = copy.deepcopy(reply.body)
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            blocks = [chunk.get("usage"), *(item.get("usage") for item in chunk["choices"])]
            for usage in blocks:
                if isinstance(usage, dict):
                    usage["cached_tokens"] = 0
                    usage["prompt_tokens_details"]["cached_tokens"] = 0
        return dataclasses.replace(reply, body=chunks, raw=driver.server_sent_events(chunks))


class _CountedWithoutDone(_WithCacheReadCount):
    """The same reply, and the stream ends without its closing message."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        chunks = [chunk for chunk in reply.body if isinstance(chunk, dict)]
        return dataclasses.replace(reply, body=chunks, raw=driver.server_sent_events(chunks))


@pytest.mark.parametrize("scenario", ["S01", "S03", "S04"])
def test_a_guarded_result_under_the_cost_policy_is_not_rankable_and_the_cell_says_so(
    scenario: str, played: Callable[..., runner.Execution]
) -> None:
    guarded = played(scenario, "guarded")
    result, terminal = guarded.observation.result, guarded.observation.limit_terminal
    assert result is not None and terminal is not None

    # The guard let the result stand: no error, the provider's own stop, no
    # overshoot. Under the plain diagnostic policy that implies a rankable
    # result; under the cost policy of the route the guard never says so.
    assert result.error is None and result.stop_reason == "end_turn"
    assert terminal.overshoots == () and not terminal.rankable
    assert terminal.resolution.diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
    # The guarded contract of the scenario says "rankable": clause C5 reads
    # it from the terminal and reports the one code, beside the two flags.
    c5 = guarded.report.outcomes["C5"]
    assert c5.status == "deviation" and [item.code for item in c5.deviations] == [UNRANKED]
    assert guarded.report.codes == WIDE | {UNRANKED}
    entry = LEDGER.entry(guarded.cell.id)
    assert entry is not None and dict(entry.codes) == {UNRANKED: "M08"}
    # What the guard does say of such a result, and what stays true of the
    # backend: the terminal of the cost policy, one physical attempt, the
    # authority kept, a second call answered from one request.
    assert terminal.terminal_reason == "diagnostic_cost_upper_bound"
    assert terminal.physical_attempts == 1 and terminal.diagnostic_cost_authority
    assert guarded.observation.second_call == SecondCall(requests=1)


def test_a_cache_read_count_in_the_reply_leaves_the_guarded_result_unrankable(
    tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    # The reply of a request that hit the cache carries the count: the
    # accounting of the result is then complete, and nothing of M06 is left.
    execution = _played("S01", tmp_path, "guarded", driver=_with_dialect(_WithCacheReadCount))
    result, terminal = execution.observation.result, execution.observation.limit_terminal
    assert result is not None and terminal is not None and result.error is None
    assert execution.report.seam_problems == ()
    assert result.completeness.complete

    # The guard under the cost policy still does not call the result
    # rankable, and the cell still says so: a resolution of M06 that leaves
    # the policy bound turns no guarded cell green.
    assert not terminal.rankable
    assert terminal.terminal_reason == "diagnostic_cost_upper_bound"
    assert UNRANKED in execution.report.codes
    # The interval is exact now, and it is priced with the rates of the
    # route's cost policy, not with the rate card of the matrix: the cost is
    # above what the rate card prices the billed usage at.
    bounds = result.usage.cost_bounds
    assert bounds is not None and bounds.covers(result.usage)
    assert result.usage.cost_usd == bounds.lower_bound_usd == bounds.upper_bound_usd
    billed = execution.observation.provider.billed_total()
    assert result.usage.cost_usd > execution.reference.pricing.compute_cost(
        input_tokens=billed.input_tokens, output_tokens=billed.output_tokens
    )
    assert execution.report.codes == {UNRANKED, "usage.cost_usd:gt"}
    # As a test item the cell fails either way: with the ledger as it is the
    # flags of M06 are gone, and without M06 the rest is not what M08 lists.
    assert "gap fixed: column-wide code" in _verdict(execution) and "(M06)" in _verdict(execution)
    without_m06 = dataclasses.replace(
        LEDGER, column_wide=tuple(scope for scope in LEDGER.column_wide if scope.gap != "M06")
    )
    assert _verdict(execution, without_m06).endswith(
        "different deviation: missing [], unexpected ['usage.cost_usd:gt']"
    )
    # The honest play of the cell has neither that code nor complete flags.
    honest = played("S01", "guarded")
    assert honest.report.codes == WIDE | {UNRANKED}


# -- the cost interval is held to what the provider billed ---------------------------


@pytest.mark.parametrize(
    ("scenario", "lens"),
    [("S03", "raw"), ("S03", "guarded"), ("S04", "raw"), ("S04", "guarded")],
)
def test_the_cost_interval_of_a_tool_loop_covers_every_response_the_provider_billed(
    scenario: str, lens: str, played: Callable[..., runner.Execution]
) -> None:
    execution = played(scenario, lens)
    result = execution.observation.result
    billed = execution.observation.provider.billed
    assert result is not None and result.error is None
    usage, bounds = result.usage, result.usage.cost_bounds

    # One worker, several provider responses: the usage is their sum ...
    assert len(billed) == {"S03": 3, "S04": 2}[scenario]
    assert (usage.input_tokens, usage.output_tokens) == (
        sum(item.input_tokens for item in billed),
        sum(item.output_tokens for item in billed),
    )
    # ... and the interval has a segment for each, with its tokens and an
    # unknown count of cache reads, covers the usage, and its lower bound is
    # the reported cost (the upper bound is what a budget is charged).
    assert bounds is not None and bounds.covers(usage)
    assert [
        (item.input_tokens, item.output_tokens, item.cache_read_tokens)
        for item in bounds.segments
    ] == [(item.input_tokens, item.output_tokens, None) for item in billed]
    assert usage.cost_usd == bounds.lower_bound_usd < bounds.upper_bound_usd
    assert usage.budget_cost_usd == bounds.upper_bound_usd
    # The cell holds all of it: the route opts into the cost bound, so clause
    # C6 asks for the interval, and names nothing but the two flags of M06.
    assert execution.reference.cost_bounded
    assert {item.code for item in execution.report.outcomes["C6"].deviations} == WIDE


def _nothing_accounted(result: Any) -> Any:
    """The result with no usage at all: no tokens, no cost, no interval."""

    return dataclasses.replace(result, usage=Usage(), pricing_evidence=None)


def _last_response_only(outcome: Any) -> Any:
    """What the worker reported, with the usage of its last provider response alone."""

    segments, priced = outcome.cost_bound_segments[-1:], outcome.pricing_segments[-1:]
    return dataclasses.replace(
        outcome,
        input_tokens=sum(item["input_tokens"] for item in segments),
        output_tokens=sum(item["output_tokens"] for item in segments),
        cost_bound_segments=segments,
        pricing_segments=priced,
        steps_used=len(segments),
        settled_usage_steps=len(segments),
    )


_SHORT = {"usage.input_tokens:lt", "usage.output_tokens:lt"}


@pytest.mark.parametrize(
    ("fault", "scenario", "lens", "named"),
    [
        ("nothing", "S01", "raw", "usage.cost_bounds:none"),
        ("last", "S03", "raw", "usage.cost_bounds:segments"),
        ("last", "S04", "guarded", "usage.cost_bounds:segments"),
    ],
    ids=["no-usage-at-all", "last-response-only", "last-response-only-under-the-guard"],
)
def test_accounting_that_loses_a_response_is_a_deviation_of_the_cell(
    fault: str,
    scenario: str,
    lens: str,
    named: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    played: Callable[..., runner.Execution],
) -> None:
    honest = played(scenario, lens)
    built = adapter.KimiBackend._result_from_outcome

    # One fault in the parent adapter, in this process: it builds its result
    # from less than the worker reported. The worker and its request are
    # untouched, and the result still flags its accounting incomplete, as
    # every result of the route does.
    def lossy(self: Any, invocation: Any, policy: Any, outcome: Any, **more: Any) -> Any:
        if fault == "last":
            return built(self, invocation, policy, _last_response_only(outcome), **more)
        return _nothing_accounted(built(self, invocation, policy, outcome, **more))

    monkeypatch.setattr(adapter.KimiBackend, "_result_from_outcome", lossy)
    execution = _played(scenario, tmp_path, lens)

    result = execution.observation.result
    assert result is not None and result.error is None
    assert execution.report.seam_problems == ()
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    billed = execution.observation.provider.billed
    assert len(billed) == len(honest.observation.provider.billed)
    assert result.usage.input_tokens < sum(item.input_tokens for item in billed)
    # The two flags alone would let it stand as an honest lower bound. The
    # route's results are cost intervals, so the tokens and the interval are
    # owed exactly: the cell names what is short.
    assert execution.report.codes == honest.report.codes | _SHORT | {named}
    assert _SHORT | {named} <= {
        item.code for item in execution.report.outcomes["C6"].deviations
    }
    verdict = _verdict(execution)
    assert "unexpected" in verdict and named in verdict


# -- the strict route: why its plain call is rejected, and when it is not ------------


def _rejection(execution: runner.Execution) -> str:
    """The text of the error a rejected call returned."""

    result = execution.observation.result
    assert result is not None and result.error is not None
    assert type(result.error).__name__ == "TransportError" and not result.error.retriable
    return str(result.error.message)


def test_the_strict_route_rejects_the_plain_reply_for_its_missing_cache_read_count(
    tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    for lens in ("raw", "guarded"):
        cell = Cell(BLOCKED, "S01", lens)
        entry = LEDGER.entry(cell.id)
        assert entry is not None and set(entry.codes.values()) == {"M07"}

        execution = played("S01", lens, BLOCKED)

        # The expected failure the ledger records for the cell ...
        assert execution.report.seam_problems == ()
        assert execution.report.codes == set(entry.codes) | WIDE
        assert execution.observation.provider.request_count == 1
        assert len(execution.observation.provider.billed) == 1
        # ... for the cause M07 names. The codes are those of any rejection
        # after the request; the cause is in the text of the error: the
        # worker refused the billed reply because its statistics carry no
        # count of cache reads.
        text = _rejection(execution)
        assert "code=counter_missing" in text and "field=R" in text

        # Another rejection leaves the cell exactly as the ledger expects it:
        # a reply that does carry the count, in a stream without its closing
        # message. Only the text tells the two apart.
        other = _played(
            "S01",
            tmp_path / lens,
            lens,
            column=BLOCKED,
            driver=_with_dialect(_CountedWithoutDone),
        )
        assert other.report.seam_problems == ()
        assert other.report.codes == execution.report.codes
        assert "code=missing_done" in _rejection(other)
        assert "counter_missing" not in _rejection(other)


def test_the_strict_route_answers_once_the_reply_carries_a_cache_read_count(
    tmp_path: Path,
) -> None:
    counted = _with_dialect(_WithCacheReadCount)
    without_m06 = dataclasses.replace(
        LEDGER, column_wide=tuple(scope for scope in LEDGER.column_wide if scope.gap != "M06")
    )
    for lens in ("raw", "guarded"):
        execution = _played("S01", tmp_path / lens, lens, column=BLOCKED, driver=counted)
        result = execution.observation.result

        # The same cell, answered with the count: a clean result, complete
        # accounting at the rate card, no code at all. So the two entries of
        # M07 rest on the missing count and on nothing else.
        assert result is not None and result.error is None
        assert execution.report.seam_problems == ()
        assert result.completeness.complete and result.usage.cost_bounds is None
        assert execution.report.codes == frozenset()
        # The comparator says what a fix of the two records has to delete.
        assert "gap fixed: column-wide code" in _verdict(execution)
        assert "gap fixed, delete the entry" in _verdict(execution, without_m06)
        if lens == "guarded":
            terminal = execution.observation.limit_terminal
            assert terminal is not None and terminal.rankable
            assert terminal.resolution.diagnostic_cost_policy is None
            assert execution.observation.second_call == SecondCall(requests=1)


def test_the_verifier_requires_the_two_proofs_of_the_strict_route_by_name() -> None:
    # No clause states why the strict route rejects a reply, so no cell's
    # verdict shows when the cause changes: a run without these two proofs is
    # not a complete run.
    assert set(cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[KIMI_KIND]]) == {
        test_the_strict_route_rejects_the_plain_reply_for_its_missing_cache_read_count.__name__,
        test_the_strict_route_answers_once_the_reply_carries_a_cache_read_count.__name__,
    }


# -- the dialect ------------------------------------------------------------------


class _WithoutLaunches(driver.MoonshotChatDialect):
    """Does not look at what the harness journalled."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        seen = super().parse(*arguments, **named)
        return dataclasses.replace(seen, evidence={**seen.evidence, "launches": []})


class _FixedServedId(driver.MoonshotChatDialect):
    """Answers every request under one id, whatever model was asked for."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        return dataclasses.replace(reply, served_model="a-fixed-model-id")


class _ModelBlind(driver.MoonshotChatDialect):
    """Does not read the model a request names."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return dataclasses.replace(super().parse(*arguments, **named), model="")


def _with_dialect(dialect: type[driver.MoonshotChatDialect]) -> driver.KimiDriver:
    class Driver(driver.KimiDriver):
        def _dialect(self, journal: launch.Journal) -> driver.MoonshotChatDialect:
            return dialect(journal=journal, interpreter=str(self.interpreter))

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
    dialect: type[driver.MoonshotChatDialect],
    problem: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(pytest.fail.Exception, match=f"the driver seam is broken.*{problem}"):
        runner.run_cell(
            Cell(PRIMARY, scenario, "raw"),
            tmp_path,
            preflight={KIMI_KIND: None},
            known=LEDGER,
            driver=_with_dialect(dialect),
        )


def test_c7_and_c9_are_live_on_the_column_through_what_the_dialect_reads(
    tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    blind = runner.execute(Cell(PRIMARY, "S01", "raw"), tmp_path, driver=_with_dialect(_ModelBlind))
    assert "request.content:model" in blind.report.codes

    execution = played("S01")
    assert "request.content:model" not in execution.report.codes
    assert {request.model for request in _matched(execution)} == {execution.reference.model}
    # Clause C9 read the debug records the adapter wrote for the call and the
    # command line of every worker launch, and found no credential in them,
    # in the result or in the trace. The reasoning the provider sent reached
    # neither the result nor a record.
    dump = execution.observation.debug_dump_text
    assert "crucible.sdk-debug-result-artifact" in dump
    assert f'"target": "{WORKER}"' in dump
    assert matrix_columns.CANARY_CREDENTIAL not in dump
    assert driver.REASONING_TEXT not in dump
    assert execution.report.outcomes["C9"].status == "ok"


def _healthy(tmp_path: Path) -> tuple[driver.MoonshotChatDialect, dict[str, Any]]:
    """A dialect over a journal that holds one probe and one request launch."""

    directory = tmp_path / "journal"
    directory.mkdir()
    interpreter = str(DRIVER.interpreter)
    record: dict[str, Any] = {
        "pid": os.getpid() + 1,
        "parent": os.getpid(),
        "mode": "redirect",
        "form": "-m",
        "target": WORKER,
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
    dialect = driver.MoonshotChatDialect(
        journal=launch.Journal(directory), interpreter=interpreter
    )
    return dialect, record


def test_seam_names_what_is_not_the_locked_workers_request_behind_the_redirect(
    tmp_path: Path,
) -> None:
    dialect, record = _healthy(tmp_path)
    credential = matrix_columns.CANARY_CREDENTIAL
    streamed = {"model": "m", "stream": True, "stream_options": {"include_usage": True}}

    def seen(
        headers: Mapping[str, str] | None = None, *, body: Mapping[str, Any] | None = None
    ) -> SeenRequest:
        sent = {
            "User-Agent": dialect.user_agent,
            "X-Stainless-Package-Version": driver.pinned_version("openai"),
            "Authorization": f"Bearer {credential}",
            "Host": HOST,
            **(headers or {}),
        }
        payload = streamed if body is None else body
        return dialect.parse(
            "POST", dialect.path, sent, json.dumps(payload).encode("utf-8"), credential=credential
        )

    def with_launch(**changed: Any) -> SeenRequest:
        healthy = seen()
        launches = [*healthy.evidence["launches"][:-1], {**record, **changed}]
        return dataclasses.replace(healthy, evidence={**healthy.evidence, "launches": launches})

    assert dialect.seam_problems(seen()) == []
    for request, named in (
        (seen({"User-Agent": "KimiCLI/0.0.0+not.the.pinned.version"}), "not the pinned runtime's"),
        (seen({"X-Stainless-Package-Version": "0.0.0"}), "not the pinned SDK's"),
        (seen({"Authorization": "Bearer another-key"}), "canary credential"),
        (seen({"Authorization": credential}), "canary credential"),
        (seen({"Host": "127.0.0.1:8000"}), "did not come through the redirect"),
        (seen(body={**streamed, "stream": False}), "a stream with its usage"),
        (seen(body={**streamed, "stream_options": {}}), "a stream with its usage"),
        (dataclasses.replace(seen(), path="/chat/completions"), "request path"),
        (with_launch(mode="guard"), "journalled no redirected worker launch"),
        (with_launch(pid=os.getpid()), "a process of its own"),
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
    assert driver.MoonshotChatDialect().seam_problems(seen()) != []
    unknown = driver.MoonshotChatDialect(journal=launch.Journal(tmp_path / "journal"))
    (problem,) = unknown.seam_problems(seen())
    assert "not the adapter's locked worker" in problem


def test_the_dialect_reads_a_chat_request_and_renders_only_from_envelopes() -> None:
    dialect = driver.MoonshotChatDialect()
    body = {
        "model": "asked",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": "the system text"},
            {"role": "user", "content": [{"type": "text", "text": "the user text"}]},
            {
                "role": "assistant",
                "reasoning_content": "thought",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "first"},
            {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function"}]},
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "text", "text": "second"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
    }
    seen = dialect.parse("POST", dialect.path, {}, json.dumps(body).encode(), credential="x")

    assert dialect.path == "/v1/chat/completions"
    assert (seen.model, seen.system, seen.user) == ("asked", "the system text", "the user text")
    assert seen.tools == ("t",) and not seen.schema_field
    # What answers the last reply is what follows the last turn of the model.
    assert seen.tool_results == (("c2", "second"),)
    assert chat.conversation(seen) == [
        "system",
        "user",
        "assistant[c1]",
        "tool[c1]",
        "assistant[c2]",
        "tool[c2]",
    ]
    # The shape says whether a turn carries reasoning, and never its text.
    assert [item["reasoning"] for item in seen.evidence["messages"]] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert "thought" not in json.dumps(seen.evidence)
    assert seen.turn_control is None and seen.budget_control is None
    schema = {"response_format": {"type": "json_schema", "json_schema": {"name": "n"}}}
    assert dialect.parse(
        "POST", dialect.path, {}, json.dumps({**body, **schema}).encode(), credential="x"
    ).schema_field

    # A scripted ServedAs step is the one way to name another model.
    assert dialect.render(Say("x"), seen, "1").served_model == "asked"
    assert dialect.render(ServedAs(Say("x")), seen, "1").served_model == "asked-served"
    # The stream is the envelope's chunks with their slots filled, in the
    # server-sent-event form, and its closing message is no JSON document.
    reply = dialect.render(Call(READ_A), seen, "call-7")
    assert reply.content_type == "text/event-stream" and reply.raw is not None
    frames = reply.raw.decode("utf-8").split("\n\n")
    assert frames[-1] == "" and len(frames) == len(reply.body) + 1
    assert frames[-2] == "data: [DONE]" == f"data: {reply.body[-1]}"
    chunks = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-2]]
    assert chunks == reply.body[:-1]
    assert {chunk["model"] for chunk in chunks} == {"asked"}
    # Reasoning first, then the call, then the finish and the statistics with
    # empty choices; no count of cache reads anywhere.
    assert [sorted(chunk["choices"][0]["delta"]) for chunk in chunks[:-1]] == [
        ["reasoning_content", "role"],
        ["tool_calls"],
        [],
    ]
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and "cached_tokens" not in json.dumps(chunks)
    assert chunks[-1]["usage"] == chunks[-2]["choices"][0]["usage"]
    assert reply.call is not None and reply.call[1] == "call-7"
    assert reply.envelope == "moonshot_chat/moonshot.tool_call"
    with pytest.raises(MatrixSetupError, match="does not render"):
        dialect.render(object(), seen, "1")  # type: ignore[arg-type]


# -- a tool round is replayed with the reasoning the provider sent -----------------


def test_a_tool_round_is_replayed_as_the_turn_that_asked_with_its_reasoning(
    played: Callable[..., runner.Execution],
) -> None:
    execution = played("S03")
    requests = _matched(execution)
    calls = execution.observation.provider.sent_calls

    # Every request is the worker's own: one request per round, nothing else.
    assert len(requests) == 3 == execution.observation.provider.request_count
    assert execution.observation.provider.extra == 0
    for request in requests:
        assert (request.path, request.host, request.credential_ok) == (
            "/v1/chat/completions",
            HOST,
            True,
        )
    # The provider's turn reasoned before it called, and the next request
    # carries the turn whole. The cell asserts it: the dialect states the
    # rule, the provider applied it to the exchange and the scenario reports
    # what it names.
    assert driver.reasons_before_a_call("moonshot_chat/moonshot.tool_call")
    assert history_replayed in execution.plan.checks
    assert execution.observation.provider.replay_problems == []
    assert "request.content:replay" not in execution.report.codes
    first_round = ["assistant[call-1]", "tool[call-1]"]
    assert [chat.conversation(request) for request in requests] == [
        ["system", "user"],
        ["system", "user", *first_round],
        ["system", "user", *first_round, "assistant[call-2]", "tool[call-2]"],
    ]
    for request in requests:
        turns = [item for item in request.evidence["messages"] if item["role"] == "assistant"]
        assert all(item["reasoning"] for item in turns)
    # The results the tools returned are what the next request answers with.
    assert [request.tool_results[0][0] for request in requests[1:]] == [
        call.call_id for call in calls
    ]


# Run in place of the worker module by the request launch of the liar below:
# the same worker, with one fault in the vendor's client. A turn of the model
# is sent back without the reasoning it came with.
_FORGETFUL = """
import runpy
import sys

from kosong.chat_provider import kimi as provider

convert = provider._convert_message


def without_reasoning(message):
    converted = convert(message)
    converted.pop("reasoning_content", None)
    return converted


provider._convert_message = without_reasoning
sys.argv = ["-m"]
runpy.run_module("crucible_kimi_worker", run_name="__main__", alter_sys=True)
"""


class _ForgetfulWorker(launch.Interposer):
    """The request launch runs the worker with the fault above in its client."""

    def command(
        self, argv: Sequence[str], env: Mapping[str, str] | None
    ) -> tuple[tuple[str, ...], dict[str, str]] | None:
        wrapped = super().command(argv, env)
        if wrapped is None:
            return None
        command, environment = wrapped
        if json.loads(environment[worker_harness.SETTINGS_ENV])["mode"] != "redirect":
            return wrapped
        target = command.index(str(launch.HARNESS)) + 1
        return (*command[:target], "-c", _FORGETFUL), environment


class _Forgetful(driver.KimiDriver):
    def _interposer(
        self, journal: launch.Journal, provider: LoopbackProvider
    ) -> launch.Interposer:
        return _ForgetfulWorker(lambda: journal, redirect=self._redirect(provider))


@pytest.mark.parametrize(("scenario", "lens"), [("S03", "raw"), ("S04", "guarded")])
def test_a_replay_without_the_reasoning_is_a_deviation_of_the_cell(
    scenario: str, lens: str, tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    honest = played(scenario, lens)

    execution = _played(scenario, tmp_path, lens, driver=_Forgetful())

    # The scripted provider answered all the same: it picks a reply on
    # position and call ids. The requests are the worker's own, and the one
    # thing the seam names is the launch that carried the fault.
    report = execution.report
    (seam,) = report.seam_problems
    assert "not the request launch of the worker" in seam
    assert execution.observation.provider.extra == 0
    second = _matched(execution)[1]
    assert chat.conversation(second) == ["system", "user", "assistant[call-1]", "tool[call-1]"]
    assert not any(item["reasoning"] for item in second.evidence["messages"])
    # Nothing else of the cell changed, so the codes are those of the honest
    # play and the one of the replay rule, which names the request. As a test
    # item the cell fails: the ledger does not describe it (the raw cell has
    # no entry of its own; the guarded one has the entry of M08).
    assert report.codes == honest.report.codes | {"request.content:replay"}
    (problem, *_later) = execution.observation.provider.replay_problems
    assert problem.startswith("request 2 carries") and "without their reasoning" in problem
    assert _verdict(execution).endswith(
        "unexpected deviation ['request.content:replay']"
        if LEDGER.entry(execution.cell.id) is None
        else "different deviation: missing [], unexpected ['request.content:replay']"
    )
    assert (LEDGER.entry(execution.cell.id) is None) == (lens == "raw")


def _request(shapes: Sequence[tuple[str, Any, bool]], *, matched: bool = True) -> SeenRequest:
    """A request whose conversation is ``(role, call id or ids, reasoning)`` per message."""

    dialect = driver.MoonshotChatDialect()
    blank = dialect.parse("POST", dialect.path, {}, b"{}", credential="")
    messages = [
        {
            "role": role,
            "asks": list(ids) if role == "assistant" and ids else [],
            "answers": ids if role == "tool" else None,
            "reasoning": reasoning,
        }
        for role, ids, reasoning in shapes
    ]
    return dataclasses.replace(blank, evidence={"messages": messages}, matched=matched)


def test_a_replay_that_drops_the_reasoning_or_a_round_is_named() -> None:
    call = SentCall(READ_A, "call-1", 0, {})
    opening = [("system", None, False), ("user", None, False)]
    asked, answered = ("assistant", ("call-1",), True), ("tool", "call-1", False)
    first = _request(opening)

    assert driver.replay_problems([first, _request([*opening, asked, answered])], [call]) == []
    for broken, named in (
        ([*opening, ("assistant", ("call-1",), False), answered], "without their reasoning"),
        ([*opening, answered], "follows no assistant turn"),
        ([*opening, asked], "is followed by the results of []"),
        ([*opening], "it replays the tool rounds []"),
        ([opening[1], asked, answered], "does not begin with the opening messages"),
    ):
        (problem,) = driver.replay_problems([first, _request(broken)], [call])
        assert problem.startswith("request 2 carries") and named in problem, problem
    # A request that matched no scripted step is an extra request, which
    # clause C0 reports; the rule judges the scripted ones, by their place.
    stray = _request([("assistant", ("call-9",), False)], matched=False)
    later = SentCall(READ_A, "call-1", 1, {})
    assert (
        driver.replay_problems([stray, first, _request([*opening, asked, answered])], [later])
        == []
    )
    # The dialect states this rule, which is how a cell comes to assert it.
    (stated,) = driver.MoonshotChatDialect().replay_problems(
        [first, _request([*opening, ("assistant", ("call-1",), False), answered])], [call]
    )
    assert "without their reasoning" in stated


# -- the admission cell of the blocked column ----------------------------------------


def test_the_admission_cell_is_held_to_its_pin_on_the_routing_files_of_both_products(
    tmp_path: Path,
) -> None:
    cell = Cell(BLOCKED, "S51", "raw")
    pinned = cells.expected_admission(cell)
    assert set(pinned) == set(cells.ADMISSION_SUBJECTS)
    assert [name for name, _path in BLOCKED.routing] == sorted(matrix_columns.PRODUCTS)

    execution = runner.execute(cell, tmp_path / "pinned")

    # Routing refuses the writer role and builds the two others, on both
    # products; nothing was called, so the provider saw nothing.
    assert execution.report.codes == frozenset() and execution.report.seam_problems == ()
    assert set(execution.report.outcomes) == {"C0", "C8", "C9", "S"}
    assert execution.observation.provider.request_count == 0
    assert pinned["workspace_write"] != ADMITTED == pinned["text"] == pinned["read_only"]
    refusal = execution.observation.debug_dump_text
    assert refusal.count("dedicated Kimi workspace_write routing is promotion-blocked") == 2

    # A pin that says otherwise is a deviation that names the subject.
    def held_to(**changed: str) -> frozenset[str]:
        return runner.execute(
            cell, tmp_path / "-".join(changed), admission={**pinned, **changed}
        ).report.codes

    refused = pinned["workspace_write"]
    assert held_to(workspace_write=ADMITTED) == {f"admission:workspace_write:{refused}!=admitted"}
    assert held_to(text=refused) == {f"admission:text:admitted!={refused}"}
    assert held_to(availability="diagnostic") == {"admission:availability:blocked!=diagnostic"}
    # The cell is about something: a diagnostic column, held to the same pin,
    # admits the writer role (its route opts into confined diagnostic writes)
    # and the catalogs have another word for it.
    diagnostic = runner.execute(
        Cell(PRIMARY, "S51", "raw"), tmp_path / "diagnostic", admission=pinned
    )
    assert diagnostic.report.codes == {
        f"admission:workspace_write:admitted!={refused}",
        "admission:availability:diagnostic!=blocked",
    }


# -- the envelopes -----------------------------------------------------------------

# Run by the locked interpreter: every chunk of a stream against the chunk
# type of the pinned SDK, with full validation.
_VALIDATE = """
import json
import sys

import openai
import pydantic
from openai.types.chat import ChatCompletionChunk

report = {"sdk": openai.__version__, "chunks": []}
for chunk in json.load(sys.stdin):
    try:
        ChatCompletionChunk.model_validate(chunk, strict=True)
        errors = []
    except pydantic.ValidationError as exc:
        errors = sorted(
            ".".join(str(part) for part in error["loc"]) + ": " + error["type"]
            for error in exc.errors()
        )
    report["chunks"].append(errors)
print(json.dumps(report))
"""


def test_the_envelopes_validate_against_the_pinned_sdk_in_the_locked_interpreter(
    locked_worker_launch: launch.Interposer,
) -> None:
    dialect = driver.MoonshotChatDialect()
    blank = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")

    for step in (Say("the answer"), Call(READ_A)):
        chunks = dialect.render(step, blank, "call-1").body
        assert chunks[-1] == "[DONE]"
        # The test starts the locked interpreter itself, through the launcher
        # of the suite fixture: it runs under the harness, in mode guard.
        completed = locked_worker_launch.run(
            [str(DRIVER.interpreter), "-B", "-I", "-c", _VALIDATE],
            input=json.dumps(chunks[:-1]),
            capture_output=True,
            text=True,
            timeout=100,
            check=False,
            env={"PYTHONNOUSERSITE": "1"},
        )
        assert (completed.returncode, completed.stderr) == (0, "")
        report = json.loads(completed.stdout)
        assert report["sdk"] == driver.pinned_version("openai")
        # Every chunk is an instance of the pinned SDK's own type.
        assert report["chunks"] == [[]] * (len(chunks) - 1)

    assert locked_worker_launch.journal is not None
    assert [(item["mode"], item["form"]) for item in locked_worker_launch.journal.launches()] == [
        ("guard", "-c")
    ] * 2
    assert locked_worker_launch.violations() == []


def test_every_cell_of_the_columns_is_graded_by_the_envelopes_it_used(
    played: Callable[..., runner.Execution],
) -> None:
    text, tools = played("S01"), played("S04")

    assert set(text.observation.provider.envelopes) == {"moonshot_chat/moonshot.completion"}
    assert set(tools.observation.provider.envelopes) == set(DRIVER.envelopes)
    assert set(DRIVER.envelopes) == {
        name for name in envelopes.all_envelopes() if name.startswith("moonshot_chat/")
    }
    # Both envelopes rest on the vendor's documentation and on what the live
    # checks of the route recorded; neither is a recording.
    assert {envelopes.load(name).grade for name in DRIVER.envelopes} == {"documented"}
    assert text.grade == tools.grade == "documented"
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
