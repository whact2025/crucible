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
that a guarded result of such a route is held to the marks of a diagnostic
result and never called rankable, with or without a cache-read count in the
reply, and that a mark which is missing is a deviation; that the cost
interval of such a route is held to what the provider billed and to the
shipped price of the route's model, and that accounting which loses a
response or misprices one is a deviation of the cell; that the cache counts
a segment records are the ones the replies carried, read back from their
bytes, and that a count no reply carried is a deviation in both lenses and
on a limit exit; why the strict
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
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY, valid_diagnostic_cost_bounds

from . import cells, contract, declarations, envelopes, ledger, netguard, runner
from . import columns as matrix_columns
from .cells import Cell
from .columns import KIMI_KIND
from .contract import ADMITTED, SecondCall
from .drivers import MatrixSetupError
from .drivers import kimi as driver
from .drivers import openai_compat as chat
from .drivers.base import GUARD_POLICY, at_the_call
from .peers import launch, worker_harness
from .provider import CacheCounts, LoopbackProvider, Reply, SeenRequest, SentCall
from .scenarios import D_COST_BOUND_UNRANKED, SCENARIOS, history_replayed
from .script import (
    READ_A,
    UNATTACHED,
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
# The code of a result the limit guard let stand and does not call rankable,
# on a route that does not opt into a cost upper bound.
UNRANKED = "guard:not_rankable"
# The code of an interval whose bounds are not the shipped price of its usage.
MISPRICED = "usage.cost_bounds:amounts"
# The code of an interval whose cache counts are not the ones the replies carried.
MISCOUNTED = "usage.cost_bounds:counts"
# What the reply of the columns says of the provider's cache: nothing was
# written, and no count of reads (the request read nothing from the cache).
UNREAD = CacheCounts(read_tokens=None, write_tokens=0)


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


def _rejudged(execution: runner.Execution, **changes: Any) -> frozenset[str]:
    """The codes of a play with facts of its observation or its reference changed."""

    reference = changes.pop("reference", execution.reference)
    return contract.evaluate(
        dataclasses.replace(execution.observation, **changes),
        execution.plan.expect,
        reference,
        lens=execution.cell.lens,
        toolset=SCENARIOS[execution.cell.scenario].toolset is not None,
        checks=execution.plan.checks,
    ).codes


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
        # The file that opts in names the cost policy a caller has to pass,
        # on both products, and it is the one policy the product knows.
        for product, _path in column.routing:
            named = declarations.cost_policy(column, role, product)
            assert named == (DIAGNOSTIC_COST_POLICY if opted else None), (column.id, product)

    raw, guarded = played("S01"), played("S01", "guarded")
    assert raw.route_budget == guarded.route_budget == matrix_columns.ROUTE_BUDGET_USD
    # What the contract is handed of such a route, none of it from the
    # adapter: the opt-in, the policy the file names and the shipped price
    # row of the route's model, which is not the rate card of the matrix.
    for execution in (raw, guarded):
        reference = execution.reference
        assert reference.cost_bounded and reference.cost_policy == DIAGNOSTIC_COST_POLICY
        assert reference.bound_pricing == matrix_columns.shipped_price(reference.model)
        assert reference.bound_pricing not in (None, reference.pricing)
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
        # The count is unknown because the reply carried none: the provider
        # side of the cell read that back from the bytes it sent.
        assert execution.observation.provider.cache_counts == [UNREAD]
    # The flags of the interval, in both lenses: what the guard says of a
    # result under this policy is no deviation (the tests below).
    assert raw.report.codes == guarded.report.codes == WIDE

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
    assert not strict.reference.cost_bounded and strict.reference.cost_policy is None


# -- under the cost policy a guarded result is diagnostic, never rankable ----------


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
def test_a_guarded_result_under_the_cost_policy_is_held_to_the_marks_of_a_diagnostic_result(
    scenario: str, played: Callable[..., runner.Execution]
) -> None:
    guarded = played(scenario, "guarded")
    result, terminal = guarded.observation.result, guarded.observation.limit_terminal
    reference = guarded.reference
    assert result is not None and terminal is not None

    # The guard let the result stand: no error, the provider's own stop, no
    # overshoot. Under the plain diagnostic policy that implies a rankable
    # result; under the cost policy of the route the guard never says so.
    assert result.error is None and result.stop_reason == "end_turn"
    assert terminal.overshoots == () and not terminal.rankable
    # The routing file opts into the cost upper bound, so the ledger lists
    # the cell under the declared difference of such routes, and clause C5
    # does not ask "rankable" there: it holds the result to the marks of a
    # diagnostic one and names nothing. The cell has no entry of its own;
    # what is left are the two flags of M06.
    assert reference.cost_bounded and reference.cost_policy == DIAGNOSTIC_COST_POLICY
    assert guarded.differences == LEDGER.differences_for(guarded.cell.id)
    assert guarded.differences == (D_COST_BOUND_UNRANKED,)
    assert guarded.report.outcomes["C5"].status == "ok"
    assert guarded.report.codes == WIDE and LEDGER.entry(guarded.cell.id) is None
    # The marks, each as the terminal has it: the terminal of the cost
    # policy under the policy the routing file names, one physical attempt,
    # the authority kept, and a second call answered from one request.
    assert terminal.terminal_reason == contract.COST_BOUND_REASON
    assert terminal.resolution.diagnostic_cost_policy == reference.cost_policy
    assert terminal.physical_attempts == 1 and terminal.diagnostic_cost_authority
    assert guarded.observation.second_call == SecondCall(requests=1)

    # Each mark is asserted, not waived: with one fact of the terminal
    # changed, C5 names the mark that is missing. (A terminal that calls the
    # result rankable cannot be built under the policy: the type refuses it.
    # The self-test holds a rankable terminal against such a route.)
    def marks(**changed: Any) -> frozenset[str]:
        changed_terminal = dataclasses.replace(terminal, **changed)
        return _rejudged(guarded, limit_terminal=changed_terminal) - WIDE

    late = terminal.resolution.deadline_seconds + 1.0
    assert marks() == frozenset()
    assert marks(elapsed_seconds=late) == {contract.COST_BOUND_AUTHORITY}
    assert marks(physical_attempts=2) == {
        contract.COST_BOUND_ATTEMPTS,
        contract.COST_BOUND_AUTHORITY,
    }
    assert marks(terminal_reason="end_turn") == {
        contract.COST_BOUND_TERMINAL,
        contract.COST_BOUND_AUTHORITY,
    }
    # The policy is the one the routing file names, not whichever is bound.
    renamed = dataclasses.replace(reference, cost_policy="another-cost-policy")
    assert _rejudged(guarded, reference=renamed) - WIDE == {contract.COST_BOUND_TERMINAL}
    # As a test item a result that lost a mark fails: no entry describes it.
    lost = dataclasses.replace(terminal, elapsed_seconds=late)
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation"):
        ledger.compare(
            LEDGER,
            guarded.cell.id,
            _rejudged(guarded, limit_terminal=lost),
            current=guarded.current,
            evaluated=guarded.report.evaluated(),
            unasked=guarded.report.unasked,
        )
    # The selector is the routing file's opt-in and nothing else: held as a
    # route without it, the same result owes "rankable", as on every other
    # column, and the interval is not asked for.
    plain = dataclasses.replace(reference, cost_bounded=False, cost_policy=None)
    assert _rejudged(guarded, reference=plain) == WIDE | {UNRANKED}


def test_a_cache_read_count_in_the_reply_leaves_the_guarded_result_diagnostic(
    tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    # The reply of a request that hit the cache carries the count: the
    # accounting of the result is then complete, and nothing of M06 is left.
    execution = _played("S01", tmp_path, "guarded", driver=_with_dialect(_WithCacheReadCount))
    result, terminal = execution.observation.result, execution.observation.limit_terminal
    assert result is not None and terminal is not None and result.error is None
    assert execution.report.seam_problems == ()
    assert result.completeness.complete

    # The route still opts into the cost upper bound, so its guard still
    # runs under the cost policy and still does not call the result
    # rankable: a resolution of M06 that leaves the opt-in set leaves every
    # result of the route diagnostic. Clause C5 holds the result to the
    # marks of a diagnostic one, as the declared difference of such routes
    # says, and names nothing. So no code of a cell shows this state; this
    # proof does, and the verifier requires it by name.
    assert execution.reference.cost_bounded
    assert execution.differences == (D_COST_BOUND_UNRANKED,)
    assert not terminal.rankable and terminal.diagnostic_cost_authority
    assert terminal.terminal_reason == contract.COST_BOUND_REASON
    assert terminal.resolution.diagnostic_cost_policy == DIAGNOSTIC_COST_POLICY
    assert execution.report.outcomes["C5"].status == "ok"
    # The interval is exact now, and it is priced with the rates of the
    # route's cost policy, which are the shipped price row of its model, not
    # with the rate card of the matrix: the cost is above what the rate card
    # prices the billed usage at.
    bounds, row = result.usage.cost_bounds, execution.reference.bound_pricing
    assert bounds is not None and bounds.covers(result.usage) and row is not None
    assert result.usage.cost_usd == bounds.lower_bound_usd == bounds.upper_bound_usd
    # The count a segment records is the one this reply carried: the provider
    # side read it back from the bytes it sent, as it reads that the reply of
    # the columns carries none.
    assert execution.observation.provider.cache_counts == [CacheCounts(0, 0)]
    assert played("S01", "guarded").observation.provider.cache_counts == [UNREAD]
    for segment in bounds.segments:
        assert segment.cache_read_tokens == 0
        assert (segment.lower_bound_usd, segment.upper_bound_usd) == pytest.approx(
            contract.bound_amounts(row, segment), rel=1e-12
        )
    # A result that left the count unknown although the reply carried it,
    # with its flags cleared as for the reply of the columns and its bounds
    # right for an unknown count, reports the low end of a range for a
    # response whose price is known. The clause names the count.
    unknown = _stating(result, row, read=lambda _item: None, accounted=False)
    assert unknown.usage.cost_usd < result.usage.cost_usd
    assert unknown.usage.budget_cost_usd == pytest.approx(result.usage.cost_usd, rel=1e-12)
    assert _rejudged(execution, result=unknown) == WIDE | {MISCOUNTED}
    billed = execution.observation.provider.billed_total()
    assert result.usage.cost_usd > execution.reference.pricing.compute_cost(
        input_tokens=billed.input_tokens, output_tokens=billed.output_tokens
    )
    assert execution.report.codes == {"usage.cost_usd:gt"}
    # As a test item the cell fails either way: with the ledger as it is the
    # flags of M06 are gone, and without M06 no entry describes the price.
    assert "gap fixed: column-wide code" in _verdict(execution) and "(M06)" in _verdict(execution)
    without_m06 = dataclasses.replace(
        LEDGER, column_wide=tuple(scope for scope in LEDGER.column_wide if scope.gap != "M06")
    )
    assert _verdict(execution, without_m06).endswith(
        "unexpected deviation ['usage.cost_usd:gt']"
    )
    # The honest play of the cell has neither that code nor complete flags.
    honest = played("S01", "guarded")
    assert honest.report.codes == WIDE


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
    # Each segment is priced as the matrix states the bound formula, at the
    # shipped price row of the route's model, and the product's public replay
    # of the formula accepts the same interval: the rule of the cells and
    # the replay of the limit guard cannot drift apart unseen.
    row = execution.reference.bound_pricing
    assert row is not None
    for segment in bounds.segments:
        assert (segment.lower_bound_usd, segment.upper_bound_usd) == pytest.approx(
            contract.bound_amounts(row, segment), rel=1e-12
        )
    assert valid_diagnostic_cost_bounds(usage)
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


# -- what a segment of the interval costs ------------------------------------------------


def _repriced(result: Any, *, cost: Callable[[Any], float] | None = None, **changed: Any) -> Any:
    """``result`` with fields of every segment of its interval recomputed."""

    bounds = result.usage.cost_bounds
    assert bounds is not None
    repriced = type(bounds)(
        tuple(
            dataclasses.replace(
                item, **{name: value(item) for name, value in changed.items()}
            )
            for item in bounds.segments
        ),
        bounds.coverage_complete,
    )
    usage = dataclasses.replace(result.usage, cost_bounds=repriced)
    if cost is not None:
        usage = dataclasses.replace(usage, cost_usd=cost(repriced))
    return dataclasses.replace(result, usage=usage)


# Three ways to state amounts the provider's bill does not give. Every lower
# bound and the reported cost at nothing; every upper bound, which is what a
# budget is charged, lowered to the lower one; and the missing count of cache
# reads read as zero, so that every segment is exact at the price without a
# read and that is the reported cost. Each leaves the accounting flags as the
# adapter set them (cleared), the tokens and the segments as billed and the
# reported cost the lower bound of the interval.
_MISPRICED: Mapping[str, Callable[[Any], Any]] = {
    "priced-at-nothing": lambda result: _repriced(
        result, cost=lambda _bounds: 0.0, lower_bound_usd=lambda _item: 0.0
    ),
    "upper-bound-at-the-lower": lambda result: _repriced(
        result, upper_bound_usd=lambda item: item.lower_bound_usd
    ),
    "missing-count-read-as-zero": lambda result: _repriced(
        result,
        cost=lambda bounds: bounds.lower_bound_usd,
        cache_read_tokens=lambda _item: 0,
        lower_bound_usd=lambda item: item.upper_bound_usd,
    ),
}


@pytest.mark.parametrize(
    ("fault", "named"),
    [
        ("priced-at-nothing", {MISPRICED}),
        ("upper-bound-at-the-lower", {MISPRICED}),
        ("missing-count-read-as-zero", {"usage.cost_usd:gt", MISCOUNTED}),
    ],
    ids=["priced-at-nothing", "upper-bound-at-the-lower", "missing-count-read-as-zero"],
)
def test_an_interval_that_states_other_amounts_is_a_deviation_of_the_cell(
    fault: str,
    named: set[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    played: Callable[..., runner.Execution],
) -> None:
    honest = played("S03")
    built = adapter.KimiBackend._result_from_outcome

    # One fault in the parent adapter, in this process: the worker and its
    # requests are untouched.
    def faulty(self: Any, invocation: Any, policy: Any, outcome: Any, **more: Any) -> Any:
        return _MISPRICED[fault](built(self, invocation, policy, outcome, **more))

    monkeypatch.setattr(adapter.KimiBackend, "_result_from_outcome", faulty)
    execution = _played("S03", tmp_path)

    result = execution.observation.result
    assert result is not None and result.error is None
    assert execution.report.seam_problems == ()
    usage, bounds = result.usage, result.usage.cost_bounds
    billed = execution.observation.provider.billed
    # Everything the clause held before it held the amounts is in order: the
    # flags are cleared as on every result of the route, the tokens are the
    # billed ones, the interval covers the usage and has the billed segments.
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert bounds is not None and bounds.covers(usage)
    assert [(item.input_tokens, item.output_tokens) for item in bounds.segments] == [
        (item.input_tokens, item.output_tokens) for item in billed
    ]
    # The raw cell names it, in clause C6, beside the two flags of M06: as a
    # test item it is no expected failure any more.
    assert execution.report.codes == honest.report.codes | named == WIDE | named
    assert named <= {item.code for item in execution.report.outcomes["C6"].deviations}
    assert _verdict(execution).endswith(f"unexpected deviation {sorted(named)}")
    if named == {MISPRICED}:
        # The limit guard's replay of the formula refuses the same interval,
        # and the counts are the ones the replies carried.
        assert not valid_diagnostic_cost_bounds(usage)
        assert [item.cache_read_tokens for item in bounds.segments] == [None] * len(billed)
        return
    # A count of zero that no reply carried is part of the usage the segment
    # records, and the rule on the bounds prices that record: it names
    # nothing, and the product's replay accepts the interval. Two other
    # rules name it. The count is not the one the replies carried, which the
    # provider side of the cell read back from the bytes it sent. And the
    # reported cost, which the rate card of the matrix holds on every column,
    # is above that card, with the flags left cleared.
    assert valid_diagnostic_cost_bounds(usage)
    assert [item.cache_read_tokens for item in bounds.segments] == [0] * len(billed)
    assert execution.observation.provider.cache_counts == [UNREAD] * len(billed)
    assert usage.cost_usd == bounds.upper_bound_usd > execution.reference.pricing.compute_cost(
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
    )


def test_the_bounds_of_an_honest_interval_are_the_shipped_price_of_what_was_billed(
    played: Callable[..., runner.Execution],
) -> None:
    execution = played("S03")
    result = execution.observation.result
    assert result is not None and result.error is None
    usage, bounds = result.usage, result.usage.cost_bounds
    assert bounds is not None

    # The rates the bounds are held to are the price row the product ships
    # for the model of the route: read where the matrix reads prices, handed
    # to the contract by the runner, typed nowhere in the matrix.
    reference = execution.reference
    row = reference.bound_pricing
    assert row is not None and row == matrix_columns.shipped_price(reference.model)
    assert reference.model == declarations.routed_model(
        PRIMARY, matrix_columns.ROLE_BY_CAPABILITY["read_only"]
    )
    # The interval of the honest play is what the matrix's statement of the
    # formula gives at that row, segment by segment, and the product's
    # public replay of the formula accepts it.
    for segment in bounds.segments:
        assert segment.cache_read_tokens is None and segment.cache_creation_tokens == 0
        assert (segment.lower_bound_usd, segment.upper_bound_usd) == pytest.approx(
            contract.bound_amounts(row, segment), rel=1e-12
        )
    assert valid_diagnostic_cost_bounds(usage)
    assert MISPRICED not in execution.report.codes
    # And as numbers. The provider billed three responses of 120 input and
    # 30 output tokens: with no count of cache reads the interval runs from
    # 972 to 1620 millionths of a dollar. The rule above reads the shipped
    # row, so a change that the adapter's constants and that row make
    # together moves no cell; it moves these two numbers.
    billed = execution.observation.provider.billed
    assert [(item.input_tokens, item.output_tokens) for item in billed] == [(120, 30)] * 3
    assert bounds.lower_bound_usd == pytest.approx(972e-6, rel=1e-12)
    assert bounds.upper_bound_usd == pytest.approx(1620e-6, rel=1e-12)
    assert usage.cost_usd == bounds.lower_bound_usd
    assert usage.budget_cost_usd == bounds.upper_bound_usd


# -- the cache counts of a segment are the ones the reply carried ------------------------


def _stating(
    result: Any,
    row: Any,
    *,
    read: Callable[[Any], int | None],
    write: int = 0,
    accounted: bool | None = None,
) -> Any:
    """``result`` with every segment stating these cache counts, priced for them.

    Each segment records ``read(segment)`` cache reads (``None``: unknown)
    and ``write`` cache writes, and its two bounds are what the price row
    gives for exactly that record, written out here and not taken from the
    contract. The reported cost is the lower bound and the cache totals of
    the usage are the sums of the segments, so the interval covers the usage
    and the rule on the amounts names nothing. ``accounted`` sets the two
    accounting flags; left out, they stay as the adapter set them.
    """

    bounds = result.usage.cost_bounds
    assert bounds is not None

    def priced(item: Any, reads: int) -> float:
        return float(
            row.compute_cost(
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                cache_creation_tokens=write,
                cache_read_tokens=reads,
            )
        )

    def restated(item: Any) -> Any:
        reads = read(item)
        return dataclasses.replace(
            item,
            cache_read_tokens=reads,
            cache_creation_tokens=write,
            lower_bound_usd=priced(item, item.input_tokens - write if reads is None else reads),
            upper_bound_usd=priced(item, 0 if reads is None else reads),
        )

    told = type(bounds)(tuple(restated(item) for item in bounds.segments), True)
    changed: dict[str, Any] = {
        "usage": dataclasses.replace(
            result.usage,
            cost_bounds=told,
            cost_usd=told.lower_bound_usd,
            cache_read_tokens=sum(item.cache_read_tokens or 0 for item in told.segments),
            cache_creation_tokens=write * len(told.segments),
        )
    }
    if accounted is not None:
        changed["completeness"] = dataclasses.replace(
            result.completeness, token_usage_complete=accounted, cost_complete=accounted
        )
        changed["pricing_evidence"] = result.pricing_evidence if accounted else None
    return dataclasses.replace(result, **changed)


def _every_token_read(result: Any, row: Any) -> Any:
    """Every input token of every response is said to have come from the cache."""

    return _stating(result, row, read=lambda item: item.input_tokens)


@pytest.mark.parametrize("lens", ["raw", "guarded"])
def test_a_count_of_cache_reads_that_no_reply_carried_is_a_deviation_of_the_cell(
    lens: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    played: Callable[..., runner.Execution],
) -> None:
    honest = played("S03", lens)
    honest_result = honest.observation.result
    assert honest_result is not None
    row = honest.reference.bound_pricing
    built = adapter.KimiBackend._result_from_outcome

    # One fault in the parent adapter, in this process: its result says of
    # every response that all input tokens were read from the provider's
    # cache, and prices each segment exactly for that. The worker, its
    # requests and the replies are untouched; no reply carried a count.
    def faulty(self: Any, invocation: Any, policy: Any, outcome: Any, **more: Any) -> Any:
        told = built(self, invocation, policy, outcome, **more)
        return told if told.usage.cost_bounds is None else _every_token_read(told, row)

    monkeypatch.setattr(adapter.KimiBackend, "_result_from_outcome", faulty)
    execution = _played("S03", tmp_path, lens)

    result = execution.observation.result
    assert result is not None and result.error is None
    assert execution.report.seam_problems == ()
    usage, bounds = result.usage, result.usage.cost_bounds
    billed = execution.observation.provider.billed
    # Everything else the clause holds is in order. The flags are cleared as
    # on every result of the route, the tokens are the billed ones, the
    # interval covers the usage and has the billed segments, every bound is
    # what the shipped price row gives for the usage its segment records,
    # and the product's replay of the formula accepts the interval.
    assert not result.completeness.token_usage_complete
    assert not result.completeness.cost_complete
    assert bounds is not None and bounds.covers(usage) and row is not None
    assert [
        (item.input_tokens, item.output_tokens, item.cache_read_tokens)
        for item in bounds.segments
    ] == [(item.input_tokens, item.output_tokens, item.input_tokens) for item in billed]
    for segment in bounds.segments:
        assert (segment.lower_bound_usd, segment.upper_bound_usd) == pytest.approx(
            contract.bound_amounts(row, segment), rel=1e-12
        )
    assert valid_diagnostic_cost_bounds(usage)
    # What the count does: the reported cost is the honest lower bound, and
    # what a budget is charged falls from the honest upper bound to it.
    honest_usage = honest_result.usage
    assert usage.cost_usd == pytest.approx(honest_usage.cost_usd, rel=1e-12)
    assert usage.budget_cost_usd == pytest.approx(honest_usage.cost_usd, rel=1e-12)
    assert usage.budget_cost_usd < honest_usage.budget_cost_usd
    if lens == "guarded":
        # The limit guard replays the formula on the usage each segment
        # records, so it lets the result stand and keeps its cost authority:
        # no mark of clause C5 is missing and a second call is answered.
        terminal = execution.observation.limit_terminal
        assert terminal is not None and terminal.terminal_reason == contract.COST_BOUND_REASON
        assert terminal.diagnostic_cost_authority and terminal.physical_attempts == 1
        assert execution.report.outcomes["C5"].status == "ok"
        assert execution.observation.second_call == SecondCall(requests=1)
    # The provider side of the cell read back from the bytes it sent what
    # each reply said of the cache: no count of reads. The cell names the
    # count, in clause C6, beside the two flags of M06: as a test item it is
    # no expected failure, in either lens.
    assert execution.observation.provider.cache_counts == [UNREAD] * len(billed)
    assert execution.report.codes == honest.report.codes | {MISCOUNTED} == WIDE | {MISCOUNTED}
    assert MISCOUNTED in {item.code for item in execution.report.outcomes["C6"].deviations}
    assert _verdict(execution).endswith(f"unexpected deviation ['{MISCOUNTED}']")


def test_the_cache_counts_of_an_interval_are_held_to_what_the_replies_carried(
    tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    honest = played("S03")
    result, provider = honest.observation.result, honest.observation.provider
    assert result is not None and result.usage.cost_bounds is not None
    row, card = honest.reference.bound_pricing, honest.reference.pricing
    assert row is not None
    # The honest play: the replies carried no count of cache reads and a
    # count of no cache writes, and every segment records exactly that.
    assert provider.cache_counts == [UNREAD] * 3
    assert [
        (item.cache_read_tokens, item.cache_creation_tokens)
        for item in result.usage.cost_bounds.segments
    ] == [(None, 0)] * 3
    assert MISCOUNTED not in honest.report.codes

    def stated(read: int | None, write: int = 0) -> frozenset[str]:
        told = _stating(result, row, read=lambda _item: read, write=write)
        assert valid_diagnostic_cost_bounds(told.usage)  # the guard's replay accepts each
        return _rejudged(honest, result=told) - WIDE

    # A count of cache reads that every segment states although no reply
    # carried one, each segment priced exactly for it. The rate card of the
    # matrix names the reported cost only while the count is small: a
    # response of 120 input and 30 output tokens costs less than the card
    # says from a count of ten ninths of the output tokens up, and from
    # there on nothing but the count itself shows.
    dearer, cheaper = (0, 1, 33), (34, 60, 120)
    for read in (*dearer, *cheaper):
        segment = dataclasses.replace(result.usage.cost_bounds.segments[0], cache_read_tokens=read)
        exact, _upper = contract.bound_amounts(row, segment)
        at_card = card.compute_cost(input_tokens=120, output_tokens=30)
        assert (exact > at_card) is (read in dearer), read
        assert stated(read) == ({"usage.cost_usd:gt"} if read in dearer else set()) | {
            MISCOUNTED
        }, read
    # A count of cache writes that no reply carried, the reads left unknown:
    # the lower bound, and with it the reported cost, rises.
    assert stated(None, write=60) == {MISCOUNTED}
    assert stated(None) == frozenset()

    # The other way round: what the replies said is the provider ledger's to
    # know, so with another statement there the honest result is the one
    # that deviates. A reply that carried a count the result calls unknown,
    # and a reply that carried no count of cache writes.
    def carried(counts: CacheCounts) -> frozenset[str]:
        return _rejudged(honest, provider=dataclasses.replace(provider, cache_counts=[counts] * 3))

    assert carried(UNREAD) == WIDE
    assert carried(CacheCounts(0, 0)) == WIDE | {MISCOUNTED}
    assert carried(CacheCounts(None, None)) == WIDE | {MISCOUNTED}
    # A provider side that recorded nothing cannot judge the cell at all.
    with pytest.raises(AssertionError, match="recorded no cache counts"):
        _rejudged(honest, provider=dataclasses.replace(provider, cache_counts=None))

    # On a limit exit too. The call that stops at its turn limit returns an
    # error result with the interval of the responses it was billed for, and
    # the clause holds its counts as on a clean finish: with every input
    # token said to be read, the exit shows the count beside what it shows
    # today, and nothing else changes.
    stopped = _played("S10a", tmp_path)
    exit_result = stopped.observation.result
    assert exit_result is not None and exit_result.error is not None
    assert exit_result.stop_reason == "max_turns" and exit_result.usage.cost_bounds is not None
    assert stopped.observation.provider.cache_counts == [UNREAD] * len(
        stopped.observation.provider.billed
    )
    assert MISCOUNTED not in stopped.report.codes
    told = _every_token_read(exit_result, row)
    assert told.usage.budget_cost_usd < exit_result.usage.budget_cost_usd
    assert _rejudged(stopped, result=told) == stopped.report.codes | {MISCOUNTED}


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
    required = set(cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[KIMI_KIND]])
    assert {
        test_the_strict_route_rejects_the_plain_reply_for_its_missing_cache_read_count.__name__,
        test_the_strict_route_answers_once_the_reply_carries_a_cache_read_count.__name__,
    } <= required
    # Nor does a cell show what the declared difference of the cost-bounded
    # routes rests on: that such a route is called as its routing file asks
    # and refuses to run otherwise, and that its guarded result stays
    # diagnostic once the accounting is complete. And the rule on the bounds
    # of a segment reads the shipped price row, so only the proof with the
    # amounts as numbers shows a change of that row and the adapter together.
    assert required == {
        test_the_strict_route_rejects_the_plain_reply_for_its_missing_cache_read_count.__name__,
        test_the_strict_route_answers_once_the_reply_carries_a_cache_read_count.__name__,
        test_a_route_with_the_cost_bound_gets_its_budget_and_its_policy.__name__,
        test_a_cache_read_count_in_the_reply_leaves_the_guarded_result_diagnostic.__name__,
        test_the_bounds_of_an_honest_interval_are_the_shipped_price_of_what_was_billed.__name__,
        # And no cell shows why the route ended a call after a tool call it
        # will not run: its codes are those of any error of the worker.
        *(
            f"{test_the_call_ends_where_the_route_refuses_a_tool_call_it_will_not_run.__name__}"
            f"[{cell}]"
            for cell in cells.REFUSED_TOOL_CALLS_ON_THE_KIMI_ROUTE
        ),
    }


# -- a tool call the route will not run: why the call ends ------------------------------

# What ends the call on each cell, by scenario: the call the provider sent
# (its slot, and whether its argument text was no JSON object), the text of
# the error, and the tool events of the trace. A tool that is not attached
# is refused by the worker before any tool event. A call whose arguments are
# no JSON object is reported by the worker as a tool call; the vendor's SDK
# answers it itself, the worker refuses that answer because it approved no
# such call, and the adapter then refuses the trace of a call without a
# result.
_WHY_THE_CALL_ENDS: Mapping[str, tuple[tuple[str, bool], str, list[str]]] = {
    "S06": (
        (UNATTACHED, False),
        "WorkerQualificationError: Kimi requested a tool outside the attached ToolSet",
        [],
    ),
    "S05": (
        (READ_A, True),
        "Kimi worker emitted an invalid per-attempt tool event sequence",
        ["tool_call"],
    ),
}


@pytest.mark.parametrize("cell_id", cells.REFUSED_TOOL_CALLS_ON_THE_KIMI_ROUTE)
def test_the_call_ends_where_the_route_refuses_a_tool_call_it_will_not_run(
    cell_id: str, played: Callable[..., runner.Execution], request: pytest.FixtureRequest
) -> None:
    """The cause of the codes that the ledger books under M18.

    The codes say that the call ended in a transport error after its first
    reply and that the script was not played to its end. Any error of the
    worker after that reply shows them. What the record names is one cause
    per cell, and this test holds the cell to it.
    """

    # The verifier requires this test by name on each cell: the record of the
    # ledger leans on it, and a run without it is refused. No entry attributes
    # a code to that record on a cell this test does not play.
    assert request.node.name in cells.REQUIRED_PROOFS[cells.MODULE_BY_KIND[KIMI_KIND]]
    attributed = {entry.cell for entry in LEDGER.entries if "M18" in entry.codes.values()}
    assert attributed == set(cells.REFUSED_TOOL_CALLS_ON_THE_KIMI_ROUTE)
    kind, scenario, role, lens = ledger.parse_cell_id(cell_id)
    assert (kind, role, lens) == (KIMI_KIND, PRIMARY.role, "raw")
    entry = LEDGER.entry(cell_id)
    assert entry is not None
    sent, text, tool_events = _WHY_THE_CALL_ENDS[scenario]

    execution = played(scenario)

    # The expected failure the ledger records for the cell ...
    assert execution.report.seam_problems == ()
    assert execution.report.codes == set(entry.codes) | WIDE
    # ... after exactly one request, which the provider matched, answered
    # with the scripted tool call and billed. Nothing reached the ToolSet,
    # and the model was not asked again.
    provider = execution.observation.provider
    assert provider.request_count == 1 and provider.extra == 0
    assert [item.matched for item in provider.requests] == [True]
    assert [(call.slot, call.malformed) for call in provider.sent_calls] == [sent]
    assert len(provider.billed) == 1 and provider.unconsumed >= 1
    assert execution.observation.toolset_log == ()
    # ... for the cause the record names: the refusal, in the text of an
    # error that is not retriable, and the tool events the trace holds.
    assert _rejection(execution) == text
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    assert [
        event.kind for event in result.execution_trace.events if event.kind.startswith("tool_")
    ] == tool_events

    # The lying counterpart: the same result with the error of a worker
    # that went away on the reply. The contract judges it exactly as it
    # judges the cell, so the entry would stand over it: the codes name no
    # cause, and only the text held above tells the two apart.
    assert result.error is not None
    gone = dataclasses.replace(result.error, message="Kimi worker exited without an outcome")
    assert _rejudged(
        execution, result=dataclasses.replace(result, error=gone)
    ) == execution.report.codes
    # Nor do the codes tell the two cells apart: those of the call for a
    # tool that is not attached are all among those of the other cell.
    first, second = (
        LEDGER.entry(cell) for cell in cells.REFUSED_TOOL_CALLS_ON_THE_KIMI_ROUTE
    )
    assert first is not None and second is not None and set(first.codes) < set(second.codes)


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
    assert not reply.call_malformed
    # The arguments of a call travel as text in a string, so a text that is
    # no JSON object travels too, through the same envelope: the stream is
    # the provider's, and the worker reads it. The reply says that the call
    # names no arguments.
    for text in ('{"file_path": "notes', "null", '["notes.txt"]'):
        cut = dialect.render(Call(READ_A, malformed=text), seen, "call-8")
        carried = [
            call["function"]["arguments"]
            for chunk in cut.body
            if isinstance(chunk, dict)
            for choice in chunk["choices"]
            for call in choice["delta"].get("tool_calls", ())
        ]
        assert carried == [text]
        assert cut.call_malformed and cut.call is not None and cut.call[2] == {}
        assert cut.envelope == reply.envelope
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


def test_the_cache_counts_of_a_reply_are_read_back_from_its_bytes() -> None:
    dialect = driver.MoonshotChatDialect()
    seen = dialect.parse("POST", dialect.path, {}, b"{}", credential="x")
    # The two replies of the columns say that nothing was written to the
    # cache and carry no count of reads; the reply of a request that hit the
    # cache carries the count.
    plain, call = dialect.render(Say("x"), seen, "1"), dialect.render(Call(READ_A), seen, "2")
    counted = _WithCacheReadCount().render(Say("x"), seen, "1")
    assert dialect.cache_counts(plain) == dialect.cache_counts(call) == UNREAD
    assert dialect.cache_counts(counted) == CacheCounts(read_tokens=0, write_tokens=0)
    assert dialect.cache_counts(_CountedWithoutDone().render(Say("x"), seen, "1")) == CacheCounts(
        0, 0
    )
    # Read from the bytes that travel, not from what they were rendered
    # from: a reply is recorded as it was sent, whatever built it.
    assert dialect.cache_counts(dataclasses.replace(plain, raw=counted.raw)) == CacheCounts(0, 0)
    assert dialect.cache_counts(dataclasses.replace(counted, raw=plain.raw)) == UNREAD

    def stream(*blocks: Mapping[str, Any]) -> bytes:
        chunks: list[Any] = [{"choices": [], "usage": block} for block in blocks]
        return driver.server_sent_events([*chunks, "[DONE]"])

    def usage(**details: Any) -> dict[str, Any]:
        return {"prompt_tokens": 120, "prompt_tokens_details": details}

    read = driver.stated_cache_counts
    assert read(stream(usage(cached_tokens=7, cache_write_tokens=3))) == CacheCounts(7, 3)
    # The alias beside the totals states the count of reads as well.
    assert read(stream({**usage(cache_write_tokens=0), "cached_tokens": 5})) == CacheCounts(5, 0)
    assert read(stream({**usage(cached_tokens=5), "cached_tokens": 5})) == CacheCounts(5, None)
    # The usage block inside the finish choice is a statement of the reply too.
    finish = {"index": 0, "usage": usage(cached_tokens=4, cache_write_tokens=0)}
    assert read(driver.server_sent_events([{"choices": [finish]}, "[DONE]"])) == CacheCounts(4, 0)
    # A count that is not there is missing, never zero.
    assert read(stream(usage())) == read(stream({"prompt_tokens": 1})) == CacheCounts(None, None)
    assert read(b"data: [DONE]\n\n") == read(b"") == CacheCounts(None, None)
    # A reply that states two counts states none the ledger could record,
    # and neither does one whose count is no integer.
    for broken, named in (
        (stream({**usage(cached_tokens=1), "cached_tokens": 2}), "two counts of cache reads"),
        (stream(usage(cached_tokens=1), usage(cached_tokens=2)), "different cache counts"),
        (stream(usage(cached_tokens=1), usage()), "different cache counts"),
        (stream(usage(cache_write_tokens=0), usage(cache_write_tokens=1)), "different cache"),
        (stream(usage(cached_tokens=True)), "cannot record"),
        (stream(usage(cache_write_tokens="0")), "cannot record"),
    ):
        with pytest.raises(MatrixSetupError, match=named):
            read(broken)


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
    # item the cell fails: the ledger does not describe it (neither cell has
    # an entry of its own).
    assert report.codes == honest.report.codes | {"request.content:replay"}
    (problem, *_later) = execution.observation.provider.replay_problems
    assert problem.startswith("request 2 carries") and "without their reasoning" in problem
    assert _verdict(execution).endswith("unexpected deviation ['request.content:replay']")
    assert LEDGER.entry(execution.cell.id) is None


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

    # The two replies of the tool loop, the two refusals that are streams,
    # a reply the route ended at the cap on its tokens and a reply in which
    # nothing was produced.
    steps: tuple[Step, ...] = (
        Say("the answer"),
        Call(READ_A),
        Refuse("content_filter"),
        Refuse("refusal_part"),
        Cut("what the cap left"),
        Empty(),
    )
    for step in steps:
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
    ] * len(steps)
    assert locked_worker_launch.violations() == []
    # A filter stop names the filter as its finish reason and carries no
    # text; a refusal part fills the refusal field of its first delta. Both
    # are billed like a plain reply of a request that read nothing from the
    # cache: the provider ledger records "no count of cache reads".
    filtered = dialect.render(Refuse("content_filter"), blank, "call-1")
    declined = dialect.render(Refuse("refusal_part"), blank, "call-1")
    assert [chunk["choices"][0]["finish_reason"] for chunk in filtered.body[:2]] == [
        None,
        "content_filter",
    ]
    assert filtered.body[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert declined.body[0]["choices"][0]["delta"]["refusal"]
    assert "content" not in declined.body[0]["choices"][0]["delta"]
    for reply in (filtered, declined):
        assert reply.usage is not None and reply.served_model == "asked"
        assert dialect.cache_counts(reply) == CacheCounts(None, 0)
    # The mark of a cut is the finish reason of the chunk that ends the
    # choice: the one the pinned SDK types for a reply that reached its cap.
    # The text that was produced travels before it and the usage after it.
    # A reply in which nothing was produced ends on an ordinary finish, with
    # an empty content, no reasoning and no completion token.
    cut = dialect.render(Cut("what the cap left"), blank, "call-1")
    silent = dialect.render(Empty(), blank, "call-1")

    def finishes(reply: Reply) -> list[str | None]:
        return [
            choice["finish_reason"] for chunk in reply.body[:-1] for choice in chunk["choices"]
        ]

    assert finishes(cut) == [None, None, "length"]
    assert cut.body[1]["choices"][0]["delta"] == {"content": "what the cap left"}
    assert cut.envelope == f"moonshot_chat/moonshot.{chat.OUTPUT_CUT}"
    assert cut.usage == Cut("what the cap left").usage
    assert finishes(silent) == [None, "stop"]
    assert silent.body[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert not any("reasoning_content" in str(chunk) for chunk in silent.body)
    assert silent.envelope == f"moonshot_chat/moonshot.{chat.EMPTY_COMPLETION}"
    assert silent.usage == Empty().usage and silent.usage.output_tokens == 0
    assert silent.body[-2]["usage"]["completion_tokens"] == 0
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
    assert rejected.envelope == "moonshot_chat/moonshot.refusal_http_policy"


class _AnswersWhereItShouldRefuse(driver.MoonshotChatDialect):
    """Renders a refusal as a plain answer: the provider never refuses."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        if isinstance(step, Refuse):
            step = Say("an answer where the script refuses", step.usage)
        return super().render(step, seen, call_id)


def test_a_refusal_that_is_never_sent_fails_the_cell_whatever_its_entry(
    tmp_path: Path,
) -> None:
    class Lying(driver.KimiDriver):
        def _dialect(self, journal: launch.Journal) -> Any:
            return _AnswersWhereItShouldRefuse(journal=journal, interpreter=str(self.interpreter))

    # The entry of the cell is strict: a provider side that answers instead
    # of refusing yields other codes, and the cell fails as a different
    # deviation instead of staying an expected failure.
    with pytest.raises(ledger.LedgerMismatch, match="different deviation"):
        runner.run_cell(
            Cell(PRIMARY, "S23c", "raw"),
            tmp_path,
            preflight={KIMI_KIND: None},
            known=LEDGER,
            driver=Lying(),
        )


def test_every_cell_of_the_columns_is_graded_by_the_envelopes_it_used(
    played: Callable[..., runner.Execution],
) -> None:
    text, tools = played("S01"), played("S04")

    loop = {"moonshot_chat/moonshot.completion", "moonshot_chat/moonshot.tool_call"}
    reasoning = {"moonshot_chat/moonshot.reasoning_only"}
    unended = f"moonshot_chat/moonshot.{chat.MALFORMED}"
    cut = f"moonshot_chat/moonshot.{chat.OUTPUT_CUT}"
    silent = f"moonshot_chat/moonshot.{chat.EMPTY_COMPLETION}"
    refusals = set(DRIVER.envelopes) - loop - reasoning - {unended, cut, silent}
    assert set(text.observation.provider.envelopes) == {"moonshot_chat/moonshot.completion"}
    assert set(tools.observation.provider.envelopes) == loop
    assert set(DRIVER.envelopes) == {
        name for name in envelopes.all_envelopes() if name.startswith("moonshot_chat/")
    }
    # The two envelopes of the tool loop rest on the vendor's documentation
    # and on what the live checks of the route recorded; neither is a
    # recording.
    assert {envelopes.load(name).grade for name in loop} == {"documented"}
    assert text.grade == tools.grade == "documented"
    # No refusal of the route was observed or is documented in reach: the
    # three envelopes of a refusal are built to the types of the pinned SDK.
    assert len(refusals) == 3
    assert {envelopes.load(name).grade for name in refusals} == {"constructed"}
    # Nor was a reply that reasons and shows no text: every stream of the
    # live checks held text after its reasoning.
    assert {envelopes.load(name).grade for name in reasoning} == {"constructed"}
    # That a stream of the route ends in order before its terminal is stated
    # by no source either: the envelope of that reply is constructed.
    assert envelopes.load(unended).grade == "constructed"
    # A reply the route ended at the cap on its tokens was seen once through
    # the adapter and no field of its stream was noted; a reply without
    # output was never seen. Both envelopes are constructed.
    assert {envelopes.load(name).grade for name in (cut, silent)} == {"constructed"}
    # The schema answer of the planner shape travels in the prompt, once, and
    # in no provider field: what the route declares.
    assert SCENARIOS["S04"].governing == ("structured_output",)
    assert tools.current == {"structured_output": "prompt_only"}
    assert not any(request.schema_field for request in _matched(tools))
    assert not {code for code in tools.report.codes if code.startswith("request.content")}


# -- what a request carries beside prompt, model turns and tool results ----------------


def test_the_route_names_a_message_of_the_clients_own_as_any_chat_wire_does() -> None:
    opening = [("system", None, False), ("user", None, False)]
    asked, answered = ("assistant", ("call-1",), True), ("tool", "call-1", False)
    first = _request(opening)
    dialect = driver.MoonshotChatDialect()

    assert dialect.foreign_messages([first, _request([*opening, asked, answered])]) == []
    told = [first, _request([*opening, asked, answered, ("user", None, False)])]
    (problem,) = dialect.foreign_messages(told)
    assert problem.startswith("request 2 carries ") and "['user']" in problem
    assert dialect.foreign_messages([first, dataclasses.replace(told[1], matched=False)]) == []
    # The replay rule of the route takes such a message: only this rule names it.
    assert driver.replay_problems(told, [SentCall(READ_A, "call-1", 0, {})]) == []


# -- broken terminals: a stream without its terminal, a stream that is cut ----------


def test_a_stream_without_its_terminal_and_a_cut_stream_are_what_the_wire_carries() -> None:
    dialect = driver.MoonshotChatDialect()
    asked = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")
    text = "an answer that is cut off"

    unended = dialect.render(Malformed(text), asked, "call-1")
    cut = dialect.render(Drop(text), asked, "call-1")
    whole = dialect.render(Say(text), asked, "call-1")

    # A stream that ends without its terminal: the reasoning and the answer
    # of a plain completion, and the body ends in order before the finish
    # chunk, the statistics and the closing message. The usage travels in
    # the chunks that never come, so none is billed; the chunks name the model.
    assert unended.body == whole.body[:2]
    assert unended.envelope == f"moonshot_chat/moonshot.{chat.MALFORMED}"
    assert (unended.status, unended.usage, unended.served_model, unended.cut) == (
        200,
        None,
        "asked",
        None,
    )
    assert unended.raw == driver.server_sent_events(unended.body)
    for chunk in unended.body:
        assert "usage" not in chunk and chunk["choices"][0]["finish_reason"] is None
    assert b"[DONE]" not in (unended.raw or b"")
    # A stream that is cut: the bytes of a plain completion, of which the
    # reasoning chunk and the content chunk arrive whole and the chunk after
    # them is cut. Neither the finish reason nor a usage block arrives.
    assert cut.envelope == "moonshot_chat/moonshot.completion"
    assert (cut.status, cut.usage, cut.served_model) == (200, None, "asked")
    assert cut.raw is not None and cut.cut is not None and 0 < cut.cut < len(cut.raw)
    arrived = cut.raw[: cut.cut].split(b"\n\n")
    chunks = [json.loads(frame.removeprefix(b"data: ")) for frame in arrived[:-1]]
    assert chunks == whole.body[:2] and chunks[-1]["choices"][0]["delta"]["content"] == text
    assert arrived[-1].startswith(b"data: ") and b"[DONE]" not in cut.raw[: cut.cut]
    with pytest.raises(json.JSONDecodeError):
        json.loads(arrived[-1].removeprefix(b"data: "))
    assert driver.stated_cache_counts(cut.raw[: cut.cut]) == CacheCounts(None, None)
    with pytest.raises(MatrixSetupError, match="inside its body only"):
        dialect.render(Drop(text, after="headers"), asked, "call-3")


class _AnswersWhole(driver.MoonshotChatDialect):
    """Renders a broken terminal as a plain answer: the stream ends on its terminal."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        if isinstance(step, (Malformed, Drop)):
            step = Say(step.text)
        return super().render(step, seen, call_id)


# What the worker says of each broken stream, in the text of the error the
# adapter returns: its own check names the closing message that never came,
# and its HTTP stack names the body that ended short of what was announced.
_BROKEN_STREAM = {
    "S26": (chat.MALFORMED, "WorkerQualificationError", "code=missing_done", False),
    "S27": (
        "completion",
        "ChatProviderError",
        "peer closed connection without sending complete message body",
        True,
    ),
}


@pytest.mark.parametrize("scenario", ["S26", "S27"])
def test_a_stream_that_ends_on_its_terminal_fails_the_cell_of_a_broken_terminal(
    scenario: str, tmp_path: Path, played: Callable[..., runner.Execution]
) -> None:
    cell = Cell(PRIMARY, scenario, "raw")
    entry = LEDGER.entry(cell.id)
    honest = played(scenario)
    provider, result = honest.observation.provider, honest.observation.result
    envelope, raised, cause, retriable = _BROKEN_STREAM[scenario]

    # The honest cell shows what the ledger expects of it, its entry or
    # nothing, ...
    assert honest.report.seam_problems == ()
    assert honest.report.codes == (frozenset() if entry is None else frozenset(entry.codes))
    assert (provider.request_count, provider.extra, provider.unconsumed) == (1, 0, 0)
    assert provider.envelopes == [f"moonshot_chat/moonshot.{envelope}"]
    # ... the chunks that arrived named the model and stated no usage ...
    assert (provider.billed, provider.served_models) == ([], [honest.reference.model])
    # ... and it shows it for the cause the scenario scripts. The codes of
    # an entry here are those of any call whose outcome the adapter builds
    # from an exception, whatever raised it: an adapter that gave up on its
    # worker for a reason of its own returns the same codes. The cause is in
    # the text of the error: the worker read the scripted stream to where
    # it broke.
    assert result is not None and result.error is not None
    assert type(result.error).__name__ == "TransportError"
    text = str(result.error.message)
    assert text.startswith(f"{raised}: ") and cause in text, text
    assert result.error.retriable is retriable, text
    if scenario == "S27":
        # The body ended short of the length its headers announced.
        short = re.search(r"received (\d+) bytes, expected (\d+)", text)
        assert short is not None and 0 < int(short[1]) < int(short[2]), text

    # The lying counterpart. The entry of the cell is strict: a provider
    # side whose stream ends on its terminal yields a clean finish where the
    # scenario expects the transport error of a turn without its terminal,
    # and the cell fails as a different deviation instead of staying an
    # expected failure.
    with pytest.raises(ledger.LedgerMismatch, match="different deviation") as refused:
        runner.run_cell(
            cell,
            tmp_path,
            preflight={KIMI_KIND: None},
            known=LEDGER,
            driver=_with_dialect(_AnswersWhole),
        )
    assert "'kind:none!=transport'" in str(refused.value).splitlines()[0]


# -- a route without its credential ---------------------------------------------------


def test_a_play_without_the_credential_keeps_the_key_variable_until_the_call(
    tmp_path: Path,
) -> None:
    role = matrix_columns.ROLE_BY_CAPABILITY["text"]
    path = PRIMARY.routing_path(matrix_columns.COLUMN_PRODUCT)
    variable = declarations.credential_variable(path, role)

    # The route names its key as a variable, which is one of the scripted
    # environment: routing reads it whenever it loads the file.
    assert variable is not None and variable in matrix_columns.SCRIPTED_ENV
    assert variable not in matrix_columns.ENDPOINT_ENV.values()
    with DRIVER.open(
        PRIMARY,
        capability="text",
        lens="raw",
        script=ProviderScript(()),
        tmp_path=tmp_path,
        without_credential=True,
    ) as bound:
        # The backend is built and the variable is still there: the runner
        # asks routing for the route once more, which loads the file.
        assert os.environ[variable] == matrix_columns.CANARY_CREDENTIAL
        assert declarations.routed_endpoint(PRIMARY, role)
        at_the_call(bound)
        # From the call on it is gone. Routing would now refuse the file:
        # that refusal is no state of the scenario, and no cell plays it.
        assert variable not in os.environ
        with pytest.raises(ValueError, match="unset"):
            declarations.routed_endpoint(PRIMARY, role)
    assert os.environ[variable] == matrix_columns.CANARY_CREDENTIAL
    # A play with its credential leaves no such step.
    with DRIVER.open(
        PRIMARY,
        capability="text",
        lens="raw",
        script=ProviderScript(()),
        tmp_path=tmp_path / "with",
    ) as bound:
        assert bound.before_call is None
        at_the_call(bound)
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
        for column in COLUMNS.values()
        for product, _path in column.routing
        for role in matrix_columns.ROLE_BY_CAPABILITY.values()
    }

    assert states and set(states.values()) == {declarations.NOT_ADMITTED}
    assert len(states) == 18


# -- the module ends with the canary check --------------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
