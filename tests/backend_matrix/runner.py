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
from . import contract, declarations, drivers, ledger, netguard, toolsets
from .cells import Cell
from .columns import Column
from .contract import Deviation, Observation, Raised, Reference, Report, SecondCall
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


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - a broken log record is not the cell's fault
            self.lines.append(str(record.msg))


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


def execute(cell: Cell, tmp_path: Path, *, driver: Driver | None = None) -> Execution:
    """Play one cell and evaluate it. Never consults the ledger."""

    scenario = SCENARIOS[cell.scenario]
    column = cell.column
    played_by = driver if driver is not None else drivers.get(column.kind)
    routed = declarations.routed(column)
    # The route the cell runs through is the one of its capability's role.
    routed_model = declarations.routed_model(
        column, matrix_columns.ROLE_BY_CAPABILITY[scenario.capability]
    )
    plan = plan_for(scenario, declarations.select(scenario.differences, routed))
    fixture = (
        None
        if scenario.toolset is None
        else toolsets.build(scenario.toolset, tmp_path / "workspace")
    )
    invocation = Invocation(
        system=SYSTEM,
        user=USER,
        tools=None,
        max_turns=plan.max_turns,
        budget_usd=plan.budget_usd,
        policy=scenario.policy,
        response_schema=plan.response_schema,
    )
    probe = replace(invocation, max_turns=1, budget_usd=None, response_schema=None)
    capture = _LogCapture()
    baseline = contract.settlement_baseline()
    started = time.monotonic()
    result = raised = terminal = second = declared = None
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
        pricing=matrix_columns.rate_card()[column.model],
        secrets=(matrix_columns.CANARY_CREDENTIAL, *secrets),
        tool_authority_digest=(
            None
            if fixture is None
            else tool_authority_digest(
                fixture.toolset,
                capability=scenario.capability,
                approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
                bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
                bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION,
                context_budget=plan.context_budget,
            )
        ),
        bindings=toolsets.BINDINGS,
        write_slot=WRITE,
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
    return Execution(cell, plan, observation, reference, report, current)


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
    execution = execute(cell, tmp_path, driver=driver)
    report = execution.report
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
        report=report.describe(),
    )
    return execution


# -- what the package conftest wires up ------------------------------------


def network_guard() -> Iterator[None]:
    """Fixture body: the guard is on for the test, and its record must be empty."""

    netguard.take_violations()
    netguard.install()
    try:
        yield
    finally:
        netguard.remove()
        violations = netguard.take_violations()
        if violations:
            pytest.fail(
                "the test reached for the network (blocked by the matrix guard): "
                + "; ".join(violations),
                pytrace=False,
            )


def isolate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Per-test environment: scripted routes, the matrix rate card, a private home.

    Every credential and configuration variable of the closed list is deleted,
    so a real one on a developer machine never reaches a cell.
    """

    for name in matrix_columns.CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in matrix_columns.SCRIPTED_ENV.items():
        monkeypatch.setenv(name, value)
    home = tmp_path / "matrix-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(
        "CRUCIBLE_PRICING_CONFIG",
        str(matrix_columns.write_rate_card(tmp_path / "matrix-pricing.yaml")),
    )


def refuse_stray_expected_failure(item: Any, report: Any, known: ledger.Ledger) -> None:
    """Only the test item of a ledgered cell may end as an expected failure.

    A self-test that reaches the comparator's expected failure by mistake
    would otherwise be reported as one itself, and pass unnoticed.
    """

    if report.when != "call" or not hasattr(report, "wasxfail"):
        return
    callspec = getattr(item, "callspec", None)
    cell = "" if callspec is None else str(callspec.id)
    if known.entry(cell) is not None or any(
        cell.startswith(f"{scope.kind}/") for scope in known.column_wide
    ):
        return
    report.longrepr = (
        f"{item.nodeid} ended as an expected failure ({report.wasxfail}) but is "
        "not the test item of a cell with a ledger entry"
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


def judge_report(item: Any, report: Any, known: ledger.Ledger) -> None:
    """What the package conftest does with every test report, in this order."""

    refuse_stray_expected_failure(item, report, known)
    record_outcome(item, report)


def preflight_all() -> dict[str, str | None]:
    """Run every registered driver's preflight once; keep the failure text."""

    failures: dict[str, str | None] = {}
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
