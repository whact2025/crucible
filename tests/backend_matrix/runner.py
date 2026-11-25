# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The shared cell runner.

One path for every cell, honest or lying, toy or real: build the backend
through its driver, attach the fixture ToolSet, invoke, close twice, observe,
evaluate the clauses that apply, then compare the deviation codes with the
ledger. The driver contributes the built backend and the provider side only.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pytest

from assay import Invocation
from crucible.backends.limits import (
    InvocationLimitAdmissionError,
    InvocationLimitContractError,
    InvocationLimitRequest,
    invocation_limit_resolution,
    invocation_limit_terminal,
)
from crucible.backends.tool_authority import (
    APPROVAL_POLICY_PER_CALL_EXACT,
    TOOL_AUTHORITY_BRIDGE_PROTOCOL,
    TOOL_AUTHORITY_BRIDGE_VERSION,
    tool_authority_digest,
)

from . import cells as matrix_cells
from . import columns as matrix_columns
from . import contract, declarations, drivers, envelopes, ledger, netguard, toolsets
from . import report as matrix_report
from . import verify as matrix_verify
from .cells import Cell
from .columns import Column
from .contract import (
    ADMITTED,
    Admitted,
    Deviation,
    Observation,
    Raised,
    Reference,
    Report,
    SecondCall,
)
from .drivers.base import GUARD_POLICY, Bound, Driver
from .provider import ProviderLedger
from .scenarios import BASE_POLICY, SCENARIOS, SYSTEM, USER, Plan, plan_for
from .script import WRITE, Loop, ProviderScript, Say

OUTCOMES: pytest.StashKey[dict[str, str]] = pytest.StashKey()


@dataclass(frozen=True)
class Execution:
    """One played cell: what was planned, observed and concluded."""

    cell: Cell
    plan: Plan
    observation: Observation
    reference: Reference
    report: Report
    current: Mapping[str, str]
    # The declared differences in force, and the grade of the weakest
    # envelope the provider answered from (``None``: no envelope was used).
    differences: tuple[str, ...] = ()
    grade: str | None = None
    # The budget the cell passed because its route asks for one, where the
    # scenario sets none of its own (``None``: the route asks for none).
    route_budget: float | None = None


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - a broken log record is not the cell's fault
            self.lines.append(str(record.msg))


def shared_tool_authority_digest(
    toolset: Any, *, capability: str, context_budget: int | None
) -> str:
    """The shared tool-authority digest of a ToolSet: what clause C7 holds a trace to.

    Computed by the product's public function from the object a cell
    attaches, under the capability of the cell's scenario and the context
    budget of its plan, never taken from an adapter.
    """

    return tool_authority_digest(
        toolset,
        capability=capability,
        approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
        bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
        bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION,
        context_budget=context_budget,
    )


def _required_steps(script: ProviderScript) -> int:
    return sum(1 for step in script.steps if not isinstance(step, Loop) or step.at_least)


def _debug_text(log_dirs: Sequence[Path], capture: _LogCapture) -> str:
    """Everything the cell logged, and every file under the driver's log dirs."""

    parts = list(capture.lines)
    for log_dir in log_dirs:
        if log_dir.is_dir():
            for path in sorted(item for item in log_dir.rglob("*") if item.is_file()):
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _second_call(bound: Bound, probe: Invocation) -> SecondCall:
    """The guarded lens's second plain call: is the backend still usable?"""

    bound.provider.begin_probe()
    try:
        result = bound.backend.invoke(probe)
    except InvocationLimitContractError:
        return SecondCall(refused=True, requests=bound.provider.final_counts()[0])
    except (netguard.NetworkGuardViolation, drivers.MatrixSetupError):
        raise  # the matrix's own failures are not observations
    except Exception as exc:  # noqa: BLE001 - every raise is an observation
        return SecondCall(
            raised=type(exc).__name__, requests=bound.provider.final_counts()[0]
        )
    return SecondCall(
        error_kind=None if result.error is None else str(result.error.kind),
        requests=bound.provider.final_counts()[0],
    )


def _route_endpoint(column: Column, role: str, bound: Bound) -> str | None:
    """The endpoint the cell's route names, read while its backend is open.

    A column that brings its own profile (a toy column) is built by its
    driver, not by routing: its route is the scripted provider it was built
    for. Any other column is asked through its shipped routing file, where a
    loopback route names the provider only for as long as the driver holds
    the backend open. Never taken from the adapter or from its trace.
    """

    if column.features is not None:
        return bound.provider.url
    try:
        return declarations.routed_endpoint(column, role)
    except drivers.MatrixSetupError:
        raise
    except Exception as exc:  # noqa: BLE001 - the matrix's own failure, not an observation
        raise drivers.MatrixSetupError(
            f"column {column.id}: the endpoint of the route of role {role!r} cannot "
            f"be read from its routing file: {exc}"
        ) from exc


def _require_lens(cell: Cell, bound: Bound, terminal: Any, second: SecondCall | None) -> None:
    """A driver that ignores ``lens`` makes a guarded cell a raw one, or the reverse.

    Every call the limit guard lets through leaves a terminal on the backend,
    and only the guard refuses a call with its own two errors.
    """

    if cell.lens != "guarded":
        if terminal is not None:
            raise drivers.MatrixSetupError(
                f"cell {cell.id}: the driver bound the limit guard in the raw lens"
            )
        return
    assert second is not None
    guard_seen = (
        terminal is not None
        or second.refused
        or second.raised == InvocationLimitAdmissionError.__name__
        or invocation_limit_terminal(bound.backend) is not None
    )
    if not guard_seen:
        raise drivers.MatrixSetupError(
            f"cell {cell.id}: the driver bound no limit guard in the guarded lens, "
            "so clause C5 would report ok without having checked anything"
        )


def execute(
    cell: Cell,
    tmp_path: Path,
    *,
    driver: Driver | None = None,
    differences: Sequence[str] = (),
    admission: Mapping[str, str] | None = None,
) -> Execution:
    """Play one cell and evaluate it. Never consults the ledger.

    ``differences`` are the declared differences in force on the cell; the
    caller takes them from the ledger. ``admission`` is what an admission
    cell is held to: the pin of the inventory unless a self-test passes its
    own.
    """

    scenario = SCENARIOS[cell.scenario]
    column = cell.column
    played_by = driver if driver is not None else drivers.get(column.kind)
    if scenario.admission:
        if differences:
            # An admission cell calls nothing and asserts no result, so no
            # alternative of a declared difference can be asserted on it.
            raise drivers.MatrixSetupError(
                f"cell {cell.id}: the declared differences {list(differences)} are "
                "listed for an admission cell, which applies none"
            )
        return _execute_admission(cell, tmp_path, played_by, expected=admission)
    routed = declarations.routed(column)
    # The route the cell runs through is the one of its capability's role.
    role = matrix_columns.ROLE_BY_CAPABILITY[scenario.capability]
    routed_model = declarations.routed_model(column, role)
    plan = plan_for(
        scenario, declarations.select(scenario.differences, routed), differences
    )
    fixture = (
        None
        if scenario.toolset is None
        else toolsets.build(scenario.toolset, tmp_path / "workspace")
    )
    # A route may ask a budget of every invocation (a closed extra of its
    # routing file). The cell is then called as the route asks, unless the
    # scenario is about a budget of its own.
    route_budget = declarations.route_budget(column, role)
    # Whether the results of that route are cost intervals: the same closed
    # extra. Such a route names the cost policy of its limit guard in its
    # routing file, and the product ships the price row of its model, which
    # is the rate table of that policy; the contract holds a result to both.
    cost_bounded = declarations.cost_bounded(column, role)
    cost_policy = declarations.cost_policy(column, role)
    bound_pricing = matrix_columns.shipped_price(routed_model)
    if cost_bounded and column.routing and bound_pricing is None:
        raise drivers.MatrixSetupError(
            f"cell {cell.id}: the route opts into a cost upper bound and the "
            f"product ships no price row for its model {routed_model!r}: the "
            "bounds of its cost interval would be held to nothing"
        )
    invocation = Invocation(
        system=SYSTEM,
        user=USER,
        tools=None,
        max_turns=plan.max_turns,
        budget_usd=route_budget if plan.budget_usd is None else plan.budget_usd,
        policy=scenario.policy,
        response_schema=plan.response_schema,
    )
    probe = replace(invocation, max_turns=1, budget_usd=route_budget, response_schema=None)
    capture = _LogCapture()
    baseline = contract.settlement_baseline()
    started = time.monotonic()
    result = raised = terminal = second = declared = None
    endpoint: str | None = None
    name = model = ""
    log_dirs: tuple[Path, ...] = ()
    secrets: tuple[str, ...] = ()
    provider_ledger = ProviderLedger(unconsumed=_required_steps(plan.script))
    settlement = contract.Settlement()
    # The handler alone sees nothing below the root logger's level, which is
    # WARNING under pytest. A workflow run sets it to INFO and keeps every
    # record in its stderr log, so the cell reads every level while it plays.
    root_logger = logging.getLogger()
    root_level = root_logger.level
    root_logger.addHandler(capture)
    root_logger.setLevel(logging.DEBUG)
    try:
        with contextlib.ExitStack() as stack:
            bound = None
            try:
                bound = stack.enter_context(
                    played_by.open(
                        column,
                        capability=scenario.capability,
                        lens=cell.lens,  # type: ignore[arg-type]
                        script=plan.script,
                        tmp_path=tmp_path,
                    )
                )
                backend = bound.backend
                endpoint = _route_endpoint(column, role, bound)
                declared = getattr(backend, "features", None)
                name, model = str(backend.name), str(backend.model)
                if fixture is not None:
                    backend.attach_toolset(
                        fixture.toolset, context_budget=plan.context_budget
                    )
                result = backend.invoke(invocation)
            except (netguard.NetworkGuardViolation, drivers.MatrixSetupError):
                raise  # the matrix's own failures are not observations
            except Exception as exc:  # noqa: BLE001 - every raise is an observation
                sent = bound is not None and bound.provider.request_count > 0
                raised = Raised.from_exception(
                    exc, phase="after_request" if sent else "before_request"
                )
            if bound is not None:
                log_dirs, secrets = tuple(bound.log_dirs), tuple(bound.secrets)
                provider_ledger = bound.provider.freeze()
                terminal = invocation_limit_terminal(bound.backend)
                if cell.lens == "guarded":
                    second = _second_call(bound, probe)
                _require_lens(cell, bound, terminal, second)
                # From here on every provider request is a late request.
                bound.provider.end_of_cell()
                close_raised = None
                for _ in range(2):  # a second close() must be a no-op
                    try:
                        bound.backend.close()
                    except Exception as exc:  # noqa: BLE001 - observed as C8
                        close_raised = Raised.from_exception(exc, phase="close")
                        break
                settlement = contract.settle(
                    baseline,
                    driver_children=bound.children(),
                    own_ports=(bound.provider.port,),
                    close_raised=close_raised,
                )
                # Read after the wait above, so a request during it is counted.
                settlement = replace(
                    settlement, late_requests=bound.provider.final_counts()[1]
                )
    finally:
        root_logger.setLevel(root_level)
        root_logger.removeHandler(capture)
    observation = Observation(
        result=result,
        raised=raised,
        provider=provider_ledger,
        toolset_log=() if fixture is None else tuple(fixture.log),
        wrapper_snapshot=None if fixture is None else fixture.progress(),
        write_applied=None if fixture is None else fixture.write_applied(),
        declarations=declared,
        backend_name=name,
        backend_model=model,
        limit_terminal=terminal,
        second_call=second,
        settlement=settlement,
        wall_seconds=time.monotonic() - started,
        debug_dump_text=_debug_text(log_dirs, capture),
    )
    reference = Reference(
        routed=routed,
        model=routed_model,
        endpoint=endpoint,
        # The rate of the model the cell's route names, as its wire model is.
        pricing=matrix_columns.matrix_rate(routed_model),
        secrets=(matrix_columns.CANARY_CREDENTIAL, *secrets),
        tool_authority_digest=(
            None
            if fixture is None
            else shared_tool_authority_digest(
                fixture.toolset,
                capability=scenario.capability,
                context_budget=plan.context_budget,
            )
        ),
        bindings=toolsets.BINDINGS,
        write_slot=WRITE,
        cost_bounded=cost_bounded,
        cost_policy=cost_policy,
        bound_pricing=bound_pricing,
    )
    report = contract.evaluate(
        observation,
        plan.expect,
        reference,
        lens=cell.lens,
        toolset=fixture is not None,
        canary=scenario.canary,
        checks=plan.checks,
    )
    current = declarations.governing_values(routed, scenario.governing)
    return Execution(
        cell,
        plan,
        observation,
        reference,
        report,
        current,
        differences=tuple(differences),
        grade=envelopes.weakest(provider_ledger.envelopes),
        route_budget=route_budget if plan.budget_usd is None else None,
    )


def _execute_admission(
    cell: Cell,
    tmp_path: Path,
    played_by: Driver,
    *,
    expected: Mapping[str, str] | None = None,
) -> Execution:
    """Play an admission cell: build a backend for every capability, call none.

    On the routing file of every product the column has, through the driver
    of its kind, so what is built is what a workflow would get. A refusal by
    routing is what the cell is about and is recorded with its type; nothing
    is invoked, so the provider must see no request at all.
    """

    scenario = SCENARIOS[cell.scenario]
    column = cell.column
    plan = plan_for(scenario, {})
    wanted = dict(matrix_cells.expected_admission(cell) if expected is None else expected)
    capture = _LogCapture()
    baseline = contract.settlement_baseline()
    started = time.monotonic()
    seen: list[Admitted] = []
    texts: list[str] = []
    secrets: list[str] = []
    children: list[int] = []
    ports: list[int] = []
    close_raised: Raised | None = None
    provider_ledger = ProviderLedger()
    late = 0
    # A column without a routing file (a toy column) is built once.
    products: list[str | None] = [name for name, _path in column.routing] or [None]
    root_logger = logging.getLogger()
    root_level = root_logger.level
    root_logger.addHandler(capture)
    root_logger.setLevel(logging.DEBUG)
    try:
        for product in products:
            view = column if product is None else column.routed_by(product)
            where = "the column" if product is None else f"the {product} routing file"
            if product is not None:
                seen.append(
                    Admitted(
                        matrix_cells.AVAILABILITY,
                        matrix_columns.catalog_availability(column, product),
                        f"the {product} profile catalog",
                    )
                )
            for capability in matrix_columns.ROLE_BY_CAPABILITY:
                directory = tmp_path / (product or "column") / capability
                directory.mkdir(parents=True)
                got = ADMITTED
                try:
                    with played_by.open(
                        view,
                        capability=capability,
                        lens=cell.lens,  # type: ignore[arg-type]
                        script=plan.script,
                        tmp_path=directory,
                    ) as bound:
                        frozen = bound.provider.freeze()
                        # From here on every provider request is a late request.
                        bound.provider.end_of_cell()
                        for _ in range(2):  # a second close() must be a no-op
                            try:
                                bound.backend.close()
                            except Exception as exc:  # noqa: BLE001 - observed as C8
                                close_raised = Raised.from_exception(exc, phase="close")
                                break
                        provider_ledger.requests.extend(frozen.requests)
                        provider_ledger.extra += frozen.extra
                        provider_ledger.seam_problems.extend(frozen.seam_problems)
                        late += bound.provider.final_counts()[1]
                        children.extend(bound.children())
                        ports.append(bound.provider.port)
                        secrets.extend(bound.secrets)
                        texts.append(_debug_text(bound.log_dirs, _LogCapture()))
                except (netguard.NetworkGuardViolation, drivers.MatrixSetupError):
                    raise  # the matrix's own failures are not observations
                except Exception as exc:  # noqa: BLE001 - a refusal is the observation
                    raised = Raised.from_exception(exc, phase="before_request")
                    got = raised.type
                    # A workflow logs the text of a refusal: clause C9 reads it.
                    texts.append(raised.message)
                seen.append(Admitted(capability, got, where))
        settlement = replace(
            contract.settle(
                baseline,
                driver_children=tuple(children),
                own_ports=tuple(ports),
                close_raised=close_raised,
            ),
            late_requests=late,
        )
    finally:
        root_logger.setLevel(root_level)
        root_logger.removeHandler(capture)
    # A request during a build matched no step: the script of the cell is empty.
    provider_ledger.late_requests = late
    observation = Observation(
        result=None,
        raised=None,
        provider=provider_ledger,
        toolset_log=(),
        wrapper_snapshot=None,
        write_applied=None,
        declarations=None,
        backend_name="",
        backend_model="",
        limit_terminal=None,
        second_call=None,
        settlement=settlement,
        wall_seconds=time.monotonic() - started,
        debug_dump_text="\n".join([*capture.lines, *texts]),
    )
    routed = declarations.routed(column)
    reference = Reference(
        routed=routed,
        model=column.model,
        endpoint=None,
        pricing=matrix_columns.matrix_rate(column.model),
        secrets=(matrix_columns.CANARY_CREDENTIAL, *secrets),
        tool_authority_digest=None,
        bindings=toolsets.BINDINGS,
        write_slot=WRITE,
    )
    report = contract.evaluate_admission(
        observation, seen, wanted, reference, lens=cell.lens
    )
    return Execution(
        cell,
        plan,
        observation,
        reference,
        report,
        declarations.governing_values(routed, scenario.governing),
    )


def pin_problems(
    cell: Cell, report: Report, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """A not-evaluated clause must be pinned, and a pinned clause not evaluated."""

    problems: list[str] = []
    for clause in report.outcomes:
        pinned = matrix_cells.pinned_not_evaluated(rows, cell, clause)
        if clause in report.not_evaluated() and not pinned:
            problems.append(
                f"clause {clause} was not evaluated and inventory.yaml does not pin it"
            )
        if clause in report.evaluated() and pinned:
            problems.append(
                f"clause {clause} is pinned not evaluated in inventory.yaml but was "
                "evaluated; delete the pin"
            )
    return problems


def profile_binding(
    column: Column, tmp_path: Path, *, driver: Driver | None = None
) -> list[Deviation]:
    """Static cell P02: one guarded call, then what it left bound to the backend."""

    played_by = driver if driver is not None else drivers.get(column.kind)
    invocation = Invocation(
        system=SYSTEM,
        user=USER,
        tools=None,
        max_turns=1,
        budget_usd=1.0,
        policy=BASE_POLICY,
    )
    request = InvocationLimitRequest(
        mode=GUARD_POLICY.mode,
        max_turns=invocation.max_turns,
        budget_usd=invocation.budget_usd,
        deadline_seconds=GUARD_POLICY.deadline_seconds,
        diagnostic_cost_policy=GUARD_POLICY.diagnostic_cost_policy,
    )
    with played_by.open(
        column,
        capability="text",
        lens="guarded",
        script=ProviderScript((Say("matrix profile-binding answer"),)),
        tmp_path=tmp_path,
    ) as bound:
        try:
            bound.backend.invoke(invocation)
        except InvocationLimitAdmissionError as exc:
            bound_limits: tuple[str, Any] = (
                declarations.REFUSED_LIMITS,
                type(exc).__name__,
            )
        else:
            resolution = invocation_limit_resolution(bound.backend)
            bound_limits = (
                (declarations.UNBOUND_LIMITS, None)
                if resolution is None
                else (declarations.RESOLVED_LIMITS, resolution)
            )
        problems = declarations.profile_binding_problems(
            bound.backend,
            declarations.routed(column),
            bound_limits=bound_limits,
            request=request,
        )
        bound.backend.close()
    return problems


def run_cell(
    cell: Cell,
    tmp_path: Path,
    *,
    preflight: Mapping[str, str | None],
    known: ledger.Ledger,
    driver: Driver | None = None,
    pins: Sequence[Mapping[str, Any]] | None = None,
) -> Execution:
    """Play a cell as a test item: seam, pins, then the ledger's verdict.

    ``driver`` and ``pins`` default to the registered driver of the cell's
    kind and the pins of ``inventory.yaml``; the self-test passes its own.
    """

    failure = preflight.get(cell.column.kind)
    if failure is not None:
        pytest.fail(f"cell {cell.id}: column cannot be driven here: {failure}", pytrace=False)
    execution = execute(
        cell, tmp_path, driver=driver, differences=known.differences_for(cell.id)
    )
    report = execution.report
    if driver is None and pins is None:
        # The cell as the matrix plays it, not a self-test with its own driver.
        matrix_report.note_cell(execution, known)
    if report.seam_problems:
        pytest.fail(
            f"cell {cell.id}: the driver seam is broken, so nothing this column "
            "shows is evidence: " + "; ".join(report.seam_problems),
            pytrace=False,
        )
    if pins is None:
        pins = matrix_cells.load_inventory().not_evaluated
    problems = pin_problems(cell, report, pins)
    if problems:
        pytest.fail(f"cell {cell.id}: " + "; ".join(problems), pytrace=False)
    ledger.compare(
        known,
        cell.id,
        report.codes,
        current=execution.current,
        evaluated=report.evaluated(),
        unasked=report.unasked,
        report=report.describe(),
    )
    return execution


# -- what the package conftest wires up ------------------------------------


def network_guard() -> Iterator[None]:
    """Fixture body: the guard is on for the test, and its record must be empty."""

    yield from netguard.watch(lambda reason: pytest.fail(reason, pytrace=False))


def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete every credential and configuration variable of the closed list."""

    for name in matrix_columns.CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)


def isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Per-test environment: scripted routes, the matrix rate card, a private home.

    Every credential and configuration variable of the closed list is deleted,
    so a real one on a developer machine never reaches a cell.
    """

    clear_environment(monkeypatch)
    for name, value in matrix_columns.SCRIPTED_ENV.items():
        monkeypatch.setenv(name, value)
    home = tmp_path / "matrix-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(
        "CRUCIBLE_PRICING_CONFIG",
        str(matrix_columns.write_rate_card(tmp_path / "matrix-pricing.yaml")),
    )


def _expected_to_fail(known: ledger.Ledger, cell: str) -> bool:
    """Whether ``cell`` is the id of a cell the ledger expects to fail.

    The verifier's rule, so that an expected failure this process lets stand
    is one the verifier accepts: the cell has an entry of its own, or owes a
    column-wide gap of its kind (never the seam canary, never a static cell).
    """

    try:
        owed, wide = matrix_verify.ledgered(
            known, cell, matrix_cells.load_inventory().column_wide_not_owed
        )
    except drivers.MatrixSetupError:
        return False  # not a cell id at all
    return bool(owed or wide)


def refuse_stray_expected_failure(item: Any, report: Any, known: ledger.Ledger) -> None:
    """Only the test item of a ledgered cell may end as an expected failure.

    A self-test that reaches the comparator's expected failure by mistake
    would otherwise be reported as one itself, and pass unnoticed. The item of
    a cell is parametrised by the cell, so its id is the cell's id; an id that
    merely starts with the kind of a column-wide entry (a proof parametrised
    by a column, the canary) is no such item.
    """

    if report.when != "call" or not hasattr(report, "wasxfail"):
        return
    callspec = getattr(item, "callspec", None)
    if callspec is not None and _expected_to_fail(known, str(callspec.id)):
        return
    report.longrepr = (
        f"{item.nodeid} ended as an expected failure ({report.wasxfail}) but is "
        "not the test item of a cell the ledger expects to fail"
    )
    report.outcome = "failed"
    del report.wasxfail


def record_outcome(item: Any, report: Any) -> None:
    """Remember which cells passed in this process, for the canary check."""

    if report.when != "call":
        return
    callspec = getattr(item, "callspec", None)
    cell = None if callspec is None else callspec.params.get("cell")
    if isinstance(cell, Cell):
        passed = report.passed and not hasattr(report, "wasxfail")
        outcomes = item.config.stash.setdefault(OUTCOMES, {})
        outcomes[cell.id] = "passed" if passed else str(report.outcome)
    # The cell report keeps the outcome of every cell, static ones included.
    path, name = getattr(item, "path", None), getattr(item, "name", None)
    judged = (
        None
        if path is None or name is None
        else matrix_cells.cell_of_item(Path(str(path)).stem, str(name))
    )
    if judged is not None:
        matrix_report.note_outcome(
            judged,
            "expected_failure"
            if hasattr(report, "wasxfail")
            else "passed"
            if report.passed
            else str(report.outcome),
        )


def judge_report(item: Any, report: Any, known: ledger.Ledger) -> None:
    """What the package conftest does with every test report, in this order."""

    refuse_stray_expected_failure(item, report, known)
    record_outcome(item, report)


def write_cell_report(config: Any) -> Path | None:
    """What the package conftest does when the session ends."""

    return matrix_report.write(config.getoption("xmlpath", None))


def preflight_all() -> dict[str, str | None]:
    """Run every registered driver's preflight once; keep the failure text.

    With the variables a cell never sees deleted for the time of the call, and
    put back afterwards. The session fixture that keeps the answer runs before
    the per-test isolation of the environment, and a preflight that read a
    developer's variable there (an override of a worker interpreter, which an
    adapter's pin refuses) would decide every cell of a column by something
    the cells themselves never see.
    """

    failures: dict[str, str | None] = {}
    with pytest.MonkeyPatch.context() as cleared:
        clear_environment(cleared)
        for kind in sorted(drivers.load()):
            try:
                drivers.get(kind).preflight()
                failures[kind] = None
            except drivers.MatrixSetupError as exc:
                failures[kind] = str(exc)
    return failures


def canary_problems(outcomes: Mapping[str, str], cells: Sequence[Cell]) -> list[str]:
    """Columns whose seam canary did not pass in this process."""

    problems: list[str] = []
    for column in sorted({cell.column.id for cell in cells}):
        canaries = [
            cell.id
            for cell in cells
            if cell.column.id == column and SCENARIOS[cell.scenario].canary
        ]
        if not canaries:
            problems.append(f"column {column} has no canary cell")
        for canary in canaries:
            if outcomes.get(canary) != "passed":
                problems.append(
                    f"canary {canary} did not pass in this process "
                    f"({outcomes.get(canary, 'not run')}); no cell of column "
                    f"{column} is evidence without it"
                )
    return problems
