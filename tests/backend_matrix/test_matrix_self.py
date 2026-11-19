# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The matrix proves itself on the toy driver.

Honest cells show that the oracle can pass: ``HonestBackend`` plays every
scenario in every lens on every toy column with an empty deviation set. Liar
cells show that it can fail: each fault yields exactly its recorded codes. The
remaining tests keep the matrix loud: the comparator's three-way rule, the
generator, the collection hook, the network guard, the pins and the H03 and
P04 rules on synthetic ledgers and inventories.
"""

from __future__ import annotations

import copy
import dataclasses
import http.client
import json
import logging
import os
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from assay import (
    CONSERVATIVE_BACKEND_FEATURES,
    BackendExecutionTrace,
    BackendFeatureSet,
    InvocationResult,
    Message,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from . import cells, contract, declarations, ledger, netguard, runner
from . import columns as matrix_columns
from .cells import Cell
from .columns import TOY_KIND, Column
from .contract import (
    RESULT_CLAUSES,
    ClauseOutcome,
    Deviation,
    Expectation,
    Raised,
    Report,
    SecondCall,
)
from .drivers import MatrixSetupError, registered_kinds, toy
from .provider import UNSCRIPTED_STATUS, LoopbackProvider, ProviderLedger, ToyDialect
from .scenarios import (
    BUDGET_USD,
    PACK_A,
    SCENARIOS,
    STARTER,
    budget_in_request,
    governing_fields,
    no_overshoot,
    plan_for,
    tool_events_paired,
)
from .script import READ_A, Call, Loop, ProviderScript, Say
from .toolsets import BINDINGS

LEDGER = ledger.load()
TOY_CELLS = cells.cells_of((TOY_KIND,))
# The nine behavioural high gaps (the other three are closed by the matrix).
BEHAVIOURAL_HIGH_GAPS = ("G02", "G03", "G14", "G15", "G16", "G17", "G18", "G28", "G29")
XFAILED = pytest.xfail.Exception


# -- the oracle can pass: honest cells ------------------------------------------


@pytest.mark.parametrize("cell", TOY_CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    execution = runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)

    assert execution.report.codes == frozenset()
    assert ledger.LEDGER_PATH.is_file() and LEDGER.entry(cell.id) is None


# -- the oracle can fail: liars -------------------------------------------------


def _liar_cell(liar: toy.Liar) -> Cell:
    for column in cells.toy_columns():
        if all(str(getattr(column.features, k)) == v for k, v in liar.needs.items()):
            return Cell(column, liar.scenario, liar.lens)
    raise MatrixSetupError(f"no toy column declares {dict(liar.needs)} for {liar.id}")


@pytest.mark.parametrize("liar", toy.LIARS, ids=lambda liar: liar.id)
def test_liar_yields_exactly_its_codes(liar: toy.Liar, tmp_path: Path) -> None:
    cell = _liar_cell(liar)
    driver = toy.ToyDriver(liar.faults)
    if liar.via == "P02":
        found = runner.profile_binding(cell.column, tmp_path, driver=driver)
        assert {item.code for item in found} == liar.codes
        return

    execution = runner.execute(cell, tmp_path, driver=driver)
    report = execution.report

    assert report.codes == liar.codes, f"liar {liar.id} on {cell.id}:\n{report.describe()}"
    assert set(liar.trips) <= report.tripped(), report.describe()
    # Each code is booked under the clause that reported it: a column-wide
    # entry is owed exactly where that clause was evaluated.
    for clause, outcome in report.outcomes.items():
        assert {contract.clause_of(item.code) for item in outcome.deviations} <= {clause}
    if execution.observation.raised is not None:
        # A raise stands for the result clauses: its code, and no outcome of theirs.
        for clause in RESULT_CLAUSES:
            assert report.outcomes[clause].status == "stood_for"
    if not SCENARIOS[liar.scenario].canary:
        assert report.not_evaluated() == frozenset()
    # As a test item the liar's cell fails: it has no ledger entry.
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation"):
        ledger.compare(
            LEDGER,
            cell.id,
            report.codes,
            current=execution.current,
            evaluated=report.evaluated(),
        )


def _normalise(code: str) -> str:
    """A raise code up to the module segment, where that names an adapter."""

    match = re.fullmatch(r"(raise:\w+@\w+/)(.+)", code)
    if match and (
        match.group(2).startswith("crucible.backends.") or match.group(2) == toy.__name__
    ):
        return match.group(1) + "<adapter>"
    return code


@pytest.mark.parametrize("gap", BEHAVIOURAL_HIGH_GAPS)
def test_each_behavioural_high_gap_is_rehearsed_as_predicted(gap: str) -> None:
    predicted = {
        frozenset(map(_normalise, item.predicted)) for item in LEDGER.registry[gap].cells
    }
    rehearsals = [liar for liar in toy.LIARS if gap in liar.rehearses]

    assert rehearsals, f"no liar rehearses {gap}"
    assert any(
        frozenset(map(_normalise, liar.codes)) in predicted for liar in rehearsals
    ), f"no liar yields the registry's predicted codes of a {gap} cell"
    # G03 as it shows today needs two faults; every other gap one.
    assert all(liar.composite == (gap == "G03") for liar in rehearsals)


# -- the comparator: the three-way rule ----------------------------------------

_CELL = "standin/S13/primary/raw"
_CODES = {"stop:end_turn!=budget_exhausted", "event.missing:limit/budget_usd"}


def _ledger(*, column_wide: bool = False) -> ledger.Ledger:
    record: dict[str, Any] = {
        "title": "a known gap",
        "origin": "gap list",
        "severity": "medium",
        "cells": [
            {
                "cell": _CELL,
                "observed_under": {"usd_limit": "post_hoc"},
                "closes_under": {"usd_limit": "post_hoc"},
            }
        ],
    }
    known: list[dict[str, Any]] = [
        {
            "cell": _CELL,
            "observed_under": {"usd_limit": "post_hoc"},
            "codes": {code: "G99" for code in _CODES},
        }
    ]
    if column_wide:
        record["column_wide"] = [{"kind": "standin", "code": "pricing:none"}]
        known.append(
            {"column_wide": {"kind": "standin", "code": "pricing:none", "gap": "G99"}}
        )
    return ledger.parse(
        {
            "schema_version": ledger.LEDGER_SCHEMA,
            "ceilings": {"max_known_gaps": {"pack_a": 2}, "max_declared_differences": 0},
            "registry": {"G99": record},
            "known_gaps": known,
            "declared_differences": [],
        }
    )


def test_comparator_exact_match_is_an_expected_failure() -> None:
    with pytest.raises(XFAILED, match="G99: a known gap"):
        ledger.compare(_ledger(), _CELL, _CODES, current={"usd_limit": "post_hoc"})


def test_comparator_empty_set_under_an_entry_is_gap_fixed() -> None:
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed, delete the entry"):
        ledger.compare(_ledger(), _CELL, set(), current={"usd_limit": "post_hoc"})


@pytest.mark.parametrize(
    "codes",
    [{"stop:end_turn!=budget_exhausted"}, _CODES | {"kind:none!=budget_exhausted"}],
    ids=["subset", "superset"],
)
def test_comparator_another_set_is_a_different_deviation(codes: set[str]) -> None:
    with pytest.raises(ledger.LedgerMismatch, match="different deviation"):
        ledger.compare(_ledger(), _CELL, codes, current={"usd_limit": "post_hoc"})


@pytest.mark.parametrize("codes", [_CODES, set()], ids=["still-red", "now-green"])
def test_comparator_changed_declaration_is_never_gap_fixed(codes: set[str]) -> None:
    with pytest.raises(ledger.LedgerMismatch, match="decl.changed:usd_limit") as caught:
        ledger.compare(_ledger(), _CELL, codes, current={"usd_limit": "unsupported"})

    assert "gap fixed" not in str(caught.value)


def test_comparator_without_an_entry_passes_clean_and_fails_any_deviation() -> None:
    other = "standin/S01/primary/raw"

    ledger.compare(_ledger(), other, set(), current={})
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation"):
        ledger.compare(_ledger(), other, {"kind:transport!=none"}, current={})


def test_comparator_column_wide_code_is_owed_where_its_clause_ran() -> None:
    known = _ledger(column_wide=True)
    other = "standin/S01/primary/raw"
    evaluated = frozenset({"C6"})

    with pytest.raises(XFAILED, match="G99"):
        ledger.compare(known, other, {"pricing:none"}, current={}, evaluated=evaluated)
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed: column-wide"):
        ledger.compare(known, other, set(), current={}, evaluated=evaluated)
    # Where C6 was not evaluated (a raise stood for it) the code is not owed...
    ledger.compare(known, other, set(), current={}, evaluated=frozenset({"C0"}))
    # ...and it is tolerated on no other kind.
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation"):
        ledger.compare(
            known, "other/S01/primary/raw", {"pricing:none"}, current={}, evaluated=evaluated
        )
    # The cell's own entry is compared after the column-wide code is removed.
    with pytest.raises(XFAILED):
        ledger.compare(
            known,
            _CELL,
            _CODES | {"pricing:none"},
            current={"usd_limit": "post_hoc"},
            evaluated=evaluated,
        )


@pytest.mark.parametrize("entry", LEDGER.entries, ids=lambda entry: entry.cell)
def test_each_shipped_entry_is_strict(entry: ledger.Entry) -> None:
    without = dataclasses.replace(
        LEDGER, entries=tuple(item for item in LEDGER.entries if item is not entry)
    )
    current = dict(entry.observed_under)

    # Without its entry the cell fails; with it, a passing cell fails too.
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation"):
        ledger.compare(without, entry.cell, set(entry.codes), current=current)
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed, delete the entry"):
        ledger.compare(LEDGER, entry.cell, set(), current=current)
    with pytest.raises(XFAILED):
        ledger.compare(LEDGER, entry.cell, set(entry.codes), current=current)


def test_ledger_rejects_a_declared_difference_without_an_alternative() -> None:
    document = yaml.safe_load(ledger.LEDGER_PATH.read_text(encoding="utf-8"))
    document["declared_differences"] = [
        {
            "id": "D-EXAMPLE",
            "citation": "crucible/backends/features.py:79-81",
            "owner": "owner",
            "date": "2026-11-18",
            "alternative": " ",
        }
    ]

    with pytest.raises(MatrixSetupError, match="alternative"):
        ledger.parse(document)


# -- the contract on what part 1 has no honest play for -------------------------


def _observation(**changes: Any) -> contract.Observation:
    base = contract.Observation(
        result=None,
        raised=None,
        provider=ProviderLedger(),
        toolset_log=(),
        wrapper_snapshot=None,
        write_applied=None,
        declarations=None,
        backend_name="toy",
        backend_model=matrix_columns.TOY_MODEL,
        limit_terminal=None,
        second_call=None,
        settlement=contract.Settlement(),
        wall_seconds=0.0,
        debug_dump_text="",
    )
    return dataclasses.replace(base, **changes)


def _reference() -> contract.Reference:
    return contract.Reference(
        routed=CONSERVATIVE_BACKEND_FEATURES,
        model=matrix_columns.TOY_MODEL,
        pricing=matrix_columns.MATRIX_RATE,
        secrets=(matrix_columns.CANARY_CREDENTIAL,),
        tool_authority_digest=None,
        bindings={},
        write_slot="write",
    )


def test_refused_variant_accepts_only_its_refusal_before_transport() -> None:
    expect = Expectation(outcome="refused", refused_type="FeatureAdmissionError")

    def codes(observation: contract.Observation) -> frozenset[str]:
        report = contract.evaluate(
            observation, expect, _reference(), lens="raw", toolset=False
        )
        assert not set(RESULT_CLAUSES) & set(report.outcomes)
        return report.codes

    refusal = Raised("FeatureAdmissionError", "before_request", "crucible.backends.config")
    other = Raised("ValueError", "before_request", "crucible.backends.config")
    late = Raised("FeatureAdmissionError", "after_request", "crucible.backends.config")
    result = InvocationResult("", (), Usage(), "end_turn")

    assert codes(_observation(raised=refusal)) == frozenset()
    assert codes(_observation(raised=other)) == {other.code}
    assert codes(_observation(raised=late)) == {late.code}
    assert codes(_observation(result=result)) == {"kind:none!=refused"}


def test_refused_variant_plans_zero_requests_and_a_table_must_be_total() -> None:
    refused = declarations.REFUSED("FeatureAdmissionError")
    table = {
        "turn_limit": {
            value: refused if value == "unsupported" else declarations.BASE
            for value in ("hard", "provider_stop", "round_boundary", "post_hoc", "unsupported")
        }
    }
    scenario = dataclasses.replace(SCENARIOS["S10a"], differences=table, fallbacks={})
    profile = dataclasses.replace(CONSERVATIVE_BACKEND_FEATURES, turn_limit="unsupported")

    plan = plan_for(scenario, declarations.select(table, profile))

    assert plan.script == ProviderScript(()) and plan.checks == ()
    assert plan.expect == Expectation(outcome="refused", refused_type="FeatureAdmissionError")
    assert declarations.totality_gaps(table) == []
    partial = {"turn_limit": {"hard": declarations.BASE, "soft": declarations.BASE}}
    assert declarations.totality_gaps(partial) == [
        "turn_limit=post_hoc has no variant",
        "turn_limit=provider_stop has no variant",
        "turn_limit=round_boundary has no variant",
        "turn_limit=unsupported has no variant",
        "turn_limit=soft is not a value",
    ]
    with pytest.raises(MatrixSetupError, match="no variant for 'unsupported'"):
        declarations.select(partial, profile)
    with pytest.raises(MatrixSetupError, match="citation"):
        declarations.FALLBACK("undocumented", "")


def test_applicability_follows_the_cell_definition_only() -> None:
    always = ("C0", "C1", "C8", "C9")

    assert contract.applicable(lens="raw", toolset=False, expects_result=False) == always
    assert contract.applicable(lens="guarded", toolset=True, expects_result=True) == (
        contract.CLAUSES
    )
    assert "C5" not in contract.applicable(lens="raw", toolset=True, expects_result=True)
    assert "C10" not in contract.applicable(
        lens="guarded", toolset=False, expects_result=True
    )


def test_a_code_outside_the_vocabulary_is_an_error() -> None:
    assert contract.clause_of("request.unconsumed") == "C0"
    assert contract.clause_of("raise:ValueError@before_request/a.b") == contract.SCENARIO
    with pytest.raises(ValueError, match="closed vocabulary"):
        contract.clause_of("event.order:terminal_not_last")


# Every branch of a clause, shown on an honest observation with one fact changed.


def _honest(tmp_path: Path, scenario: str, lens: str = "raw") -> runner.Execution:
    execution = runner.execute(Cell(cells.toy_columns()[0], scenario, lens), tmp_path)
    assert execution.report.codes == frozenset(), execution.report.describe()
    return execution


def _result_and_trace(
    execution: runner.Execution,
) -> tuple[InvocationResult, BackendExecutionTrace]:
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    return result, result.execution_trace


def _judge(execution: runner.Execution, **changes: Any) -> frozenset[str]:
    """The codes of ``execution`` with fields of its observation or result changed."""

    expect = changes.pop("expect", execution.plan.expect)
    checks = changes.pop("checks", execution.plan.checks)
    observation = execution.observation
    result_changes = {
        name: changes.pop(name)
        for name in list(changes)
        if name in {f.name for f in dataclasses.fields(InvocationResult)}
    }
    if result_changes:
        assert observation.result is not None
        result = dataclasses.replace(observation.result, **result_changes)
        observation = dataclasses.replace(observation, result=result)
    observation = dataclasses.replace(observation, **changes)
    scenario = SCENARIOS[execution.cell.scenario]
    return contract.evaluate(
        observation,
        expect,
        execution.reference,
        lens=execution.cell.lens,
        toolset=scenario.toolset is not None,
        checks=checks,
    ).codes


def test_c7_names_each_identity_field_that_is_missing_or_wrong(tmp_path: Path) -> None:
    execution = _honest(tmp_path, "S03")
    result, trace = _result_and_trace(execution)
    other = "sha256:" + "0" * 64

    def changed(**fields: Any) -> frozenset[str]:
        return _judge(execution, execution_trace=dataclasses.replace(trace, **fields))

    assert changed(backend="other") == {"digest.mismatch:backend"}
    # A requested model that is not the adapter's is not the one the provider
    # was asked for and served either: two statements, both reported.
    assert changed(requested_model="other") == {
        "digest.mismatch:requested_model",
        "served_model:not_requested",
    }
    assert changed(served_model=None) == {"digest.missing:served_model"}
    assert changed(served_model="other") == {"digest.mismatch:served_model"}
    assert changed(feature_profile_digest=other) == {"digest.mismatch:feature_profile"}
    assert changed(tool_authority_digest=None) == {"digest.missing:tool_authority"}
    assert changed(tool_authority_digest=other) == {"digest.mismatch:tool_authority"}
    # A provider that names another model than it was asked for is a scenario
    # of its own; a trace that records that name truthfully does not deviate.
    provider = execution.observation.provider
    renamed = dataclasses.replace(provider, served_models=["other"])
    assert _judge(
        execution,
        provider=renamed,
        execution_trace=dataclasses.replace(trace, served_model="other"),
    ) == frozenset()
    # The wire is judged without the trace: a result that carries none does
    # not hide a substituted model behind its missing trace.
    asked = [dataclasses.replace(item, model="other") for item in provider.requests]
    untraced = dataclasses.replace(result.completeness, trace_complete=False)
    assert _judge(
        execution,
        provider=dataclasses.replace(provider, requests=asked),
        completeness=untraced,
        execution_trace=None,
        checks=(),
    ) == {"flag:trace_complete=False", "digest.missing:trace", "request.content:model"}


def test_c4_and_c6_report_flags_and_accounting(tmp_path: Path) -> None:
    execution = _honest(tmp_path, "S03")
    result, _trace = _result_and_trace(execution)
    flags, usage = result.completeness, result.usage
    assert result.pricing_evidence is not None

    def accounted(**fields: Any) -> frozenset[str]:
        reported = dataclasses.replace(usage, **fields)
        segment = UsagePricingSegment(
            reported.input_tokens, reported.output_tokens, 0, 0, reported.cost_usd
        )
        return _judge(
            execution, usage=reported, pricing_evidence=UsagePricingEvidence((segment,), True)
        )

    assert accounted(input_tokens=usage.input_tokens + 1) == {"usage.input_tokens:gt"}
    assert accounted(output_tokens=usage.output_tokens - 1) == {"usage.output_tokens:lt"}
    assert accounted(cost_usd=usage.cost_usd * 2) == {"usage.cost_usd:gt"}
    partial = UsagePricingEvidence(result.pricing_evidence.segments, False)
    assert _judge(execution, pricing_evidence=partial) == {"pricing:unreconciled"}

    # An incomplete flag excuses a lower bound, never more than was billed.
    unaccounted = dataclasses.replace(flags, token_usage_complete=False, cost_complete=False)
    lower = dataclasses.replace(usage, input_tokens=1, output_tokens=1, cost_usd=0.0)
    higher = dataclasses.replace(usage, input_tokens=usage.input_tokens + 1)
    untold = {"flag:token_usage_complete=False", "flag:cost_complete=False"}
    assert _judge(
        execution, completeness=unaccounted, usage=lower, pricing_evidence=None
    ) == untold
    assert _judge(
        execution, completeness=unaccounted, usage=higher, pricing_evidence=None
    ) == untold | {"usage.input_tokens:gt"}
    costlier = dataclasses.replace(usage, cost_usd=usage.cost_usd * 2)
    assert _judge(
        execution, completeness=unaccounted, usage=costlier, pricing_evidence=None
    ) == untold | {"usage.cost_usd:gt"}

    assert _judge(
        execution, completeness=dataclasses.replace(flags, terminal_complete=False)
    ) == {"flag:terminal_complete=False"}
    # Without a complete trace C4 cannot be passed by clearing the flag.
    assert _judge(
        execution,
        completeness=dataclasses.replace(flags, trace_complete=False),
        execution_trace=None,
        checks=(),
    ) == {"flag:trace_complete=False", "digest.missing:trace"}


def test_c8_c9_and_c10_report_what_the_cell_left_and_dispatched(tmp_path: Path) -> None:
    execution = _honest(tmp_path, "S03")
    log = execution.observation.toolset_log
    closed = Raised("RuntimeError", "close", "crucible.backends.standin")
    loud = TransportError(message="x" * (contract.MAX_ERROR_TEXT + 1), retriable=False)
    failing = dataclasses.replace(
        execution.plan.expect, kind="transport", retriable=False
    )

    def settled(**fields: Any) -> frozenset[str]:
        return _judge(execution, settlement=contract.Settlement(**fields))

    assert settled(close_raised=closed) == {closed.code}
    assert contract.clause_of(closed.code) == "C8"
    assert settled(listeners=(4242,)) == {"child:alive"}
    assert settled(late_requests=1) == {"child:alive"}
    assert settled(children_alive=(4242,)) == {"child:alive"}
    assert _judge(execution, error=loud, expect=failing) == {"text:unbounded"}

    # C9 reads every text a workflow would keep or log, one at a time.
    canary = matrix_columns.CANARY_CREDENTIAL
    result, trace = _result_and_trace(execution)
    unasserted = dataclasses.replace(execution.plan.expect, final_text=None)
    leaked = {"secret:leak"}
    assert _judge(execution, final_message=f"key {canary}", expect=unasserted) == leaked
    assert _judge(execution, transcript=(Message("assistant", canary),)) == leaked
    in_trace = dataclasses.replace(trace, endpoint=f"http://127.0.0.1/?key={canary}")
    assert _judge(execution, execution_trace=in_trace) == leaked
    told = TransportError(message=f"refused {canary}", retriable=False)
    assert _judge(execution, error=told, expect=failing) == leaked
    in_close = dataclasses.replace(closed, message=f"close failed for {canary}")
    assert settled(close_raised=in_close) == {closed.code} | leaked
    assert canary not in in_close.code
    assert result.error is None and _judge(execution) == frozenset()

    def dispatched(*entries: Any, **more: Any) -> frozenset[str]:
        # C10 alone: the scenario's own assertions read the same log.
        return _judge(execution, toolset_log=entries, checks=(), **more)

    assert dispatched(*log, log[0]) == {"dispatch.extra:read_a"}
    assert dispatched(log[0]) == {"dispatch.missing:read_b"}
    wrong = dataclasses.replace(log[0], arguments={"file_path": "elsewhere.txt"})
    assert dispatched(wrong, log[1]) == {"dispatch.missing:read_a", "dispatch.extra:read_a"}
    # A call the scenario forbids (after a terminal condition) must not run.
    forbidding = dataclasses.replace(execution.plan.expect, dispatch=("read_a",))
    assert dispatched(*log, expect=forbidding) == {"dispatch.extra:read_b"}


def test_c5_and_optional_tool_events_report_on_their_own(tmp_path: Path) -> None:
    guarded = _honest(tmp_path / "guarded", "S03", "guarded")
    # The honest second call: answered, without error, from one provider request.
    assert guarded.observation.second_call == SecondCall(requests=1)
    assert _judge(guarded, second_call=SecondCall(refused=True)) == {"guard:revoked"}
    for unusable in (
        None,
        SecondCall(raised="RuntimeError", requests=1),
        SecondCall(error_kind="transport", requests=1),
        SecondCall(requests=0),
        SecondCall(requests=2),
    ):
        assert _judge(guarded, second_call=unusable) == {"guard:second_call_failed"}

    raw = _honest(tmp_path / "raw", "S03")
    _result, trace = _result_and_trace(raw)
    kept = [event for event in trace.events if event.kind != "tool_result"]
    unpaired = dataclasses.replace(
        trace,
        events=tuple(dataclasses.replace(e, sequence=i) for i, e in enumerate(kept)),
    )
    assert _judge(raw, execution_trace=unpaired, checks=(tool_events_paired,)) == {
        "event.missing:tool_result/read_a",
        "event.missing:tool_result/read_b",
    }
    assert _judge(raw, checks=(tool_events_paired,)) == frozenset()


def test_usd_ceiling_variants_assert_the_request_and_the_overshoot(tmp_path: Path) -> None:
    execution = _honest(tmp_path, "S13")
    observation, reference = execution.observation, execution.reference

    assert [item.code for item in budget_in_request(observation, reference)] == [
        "request.content:budget"
    ]
    assert [item.code for item in no_overshoot(observation, reference)] == [
        "usage.cost_usd:gt"
    ]
    carried = dataclasses.replace(observation.provider.requests[0], budget_control=BUDGET_USD)
    provider = dataclasses.replace(observation.provider, requests=[carried])
    assert budget_in_request(dataclasses.replace(observation, provider=provider), reference) == []
    result, _trace = _result_and_trace(execution)
    cheap = dataclasses.replace(result, usage=Usage(), pricing_evidence=None)
    assert no_overshoot(dataclasses.replace(observation, result=cheap), reference) == []


def test_a_guard_overshoot_is_a_rewrite_that_stands_for_kind_and_stop(tmp_path: Path) -> None:
    # S13 has no guarded lens. Played under the guard, its cost above the
    # budget is an overshoot, which the guard rewrites and latches.
    execution = runner.execute(Cell(cells.toy_columns()[0], "S13", "guarded"), tmp_path)
    terminal = execution.observation.limit_terminal
    report = execution.report

    assert terminal is not None and terminal.overshoots == ("usd",)
    assert report.codes == {"guard:budget_exhausted", "guard:revoked"}, report.describe()
    assert {report.outcomes[clause].status for clause in ("C2", "C3")} == {"stood_for"}


class _LensBlindDriver(toy.ToyDriver):
    """A driver that binds one lens whatever the cell asks for."""

    def __init__(self, lens: Any) -> None:
        super().__init__()
        self._lens = lens

    def open(self, column: Column, *, lens: Any, **arguments: Any) -> Any:
        return super().open(column, lens=self._lens, **arguments)


class _GuardOn(toy.ToyDriver):
    """A driver that binds the limit guard to a changed profile."""

    def __init__(self, **changes: str) -> None:
        super().__init__()
        self._changes = changes

    def guarded_profile(self, profile: BackendFeatureSet) -> BackendFeatureSet:
        return dataclasses.replace(profile, **self._changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bound_lens", "cell_lens", "message"),
    [
        ("raw", "guarded", "bound no limit guard in the guarded lens"),
        ("guarded", "raw", "bound the limit guard in the raw lens"),
    ],
)
def test_a_driver_that_ignores_the_lens_is_a_setup_error(
    bound_lens: str, cell_lens: str, message: str, tmp_path: Path
) -> None:
    cell = Cell(cells.toy_columns()[0], "S01", cell_lens)

    # Without this, a guarded cell would be a raw cell with a green C5.
    with pytest.raises(MatrixSetupError, match=message):
        runner.execute(cell, tmp_path, driver=_LensBlindDriver(bound_lens))


def test_a_cell_leaves_the_root_logger_as_it_found_it(tmp_path: Path) -> None:
    # A cell lowers the root logger to read every record (the two liars that
    # log a token at INFO and at DEBUG prove that); the rest of the process
    # must not inherit that level or the cell's handler.
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    quiet = logging.ERROR
    root.setLevel(quiet)
    try:
        _honest(tmp_path / "honest", "S01")
        assert (root.level, root.handlers) == (quiet, handlers)
        # A cell that ends in a setup error restores both as well.
        with pytest.raises(MatrixSetupError):
            runner.execute(
                Cell(cells.toy_columns()[0], "S01", "guarded"),
                tmp_path / "blind",
                driver=_LensBlindDriver("raw"),
            )
        assert (root.level, root.handlers) == (quiet, handlers)
    finally:
        root.setLevel(level)


def test_the_reference_model_is_the_model_of_the_route_of_the_cells_role(
    tmp_path: Path,
) -> None:
    column = cells.toy_columns()[0]
    roles = matrix_columns.ROLE_BY_CAPABILITY
    shipped = matrix_columns.real_columns()[0].routing_path(matrix_columns.COLUMN_PRODUCT)
    route = tmp_path / "routing.yaml"
    route.write_text(
        yaml.safe_dump(
            {
                "schema_version": yaml.safe_load(shipped.read_text(encoding="utf-8"))[
                    "schema_version"
                ],
                "default": {
                    "backend": "local",
                    "model": column.model,
                    "base_url": matrix_columns.SCRIPTED_ENV["CRUCIBLE_LOCAL_URL"],
                },
                # The roles of one routing file are separate entries.
                "roles": {roles["read_only"]: {"model": column.model + "-planner"}},
            }
        ),
        encoding="utf-8",
    )
    routed = dataclasses.replace(
        column, routing=((matrix_columns.COLUMN_PRODUCT, route),)
    )

    assert {
        capability: declarations.routed_model(routed, role)
        for capability, role in roles.items()
    } == {
        "text": column.model,
        "read_only": column.model + "-planner",
        "workspace_write": column.model,
    }
    # A toy column has no routing file: its one model serves every role.
    assert {declarations.routed_model(column, role) for role in roles.values()} == {
        column.model
    }

    def codes(scenario: str) -> frozenset[str]:
        return runner.execute(Cell(routed, scenario, "raw"), tmp_path / scenario).report.codes

    # The toy sends its one model whatever the role. That is the model of the
    # critic's and the implementer's route, and a substitution on the planner's.
    assert SCENARIOS["S03"].capability == "read_only"
    assert codes("S01") == codes("S40") == frozenset()
    assert codes("S03") == {"request.content:model"}


def test_p02_compares_what_the_guard_bound_with_what_the_profile_implies(
    tmp_path: Path,
) -> None:
    column = cells.toy_columns()[0]
    assert column.features is not None

    def details(name: str, column: Column, driver: toy.ToyDriver | None = None) -> list[str]:
        found = runner.profile_binding(column, tmp_path / name, driver=driver)
        assert {item.code for item in found} <= {"static:profile_binding"}
        return [item.detail for item in found]

    assert details("honest", column) == []
    (unbound,) = details("blind", column, _LensBlindDriver("raw"))
    assert "no limit guard is bound" in unbound
    (resolved,) = details("usd", column, _GuardOn(usd_limit="unsupported"))
    assert "differs on" in resolved and "usd_enforcement" in resolved
    (refused,) = details("deadline", column, _GuardOn(deadline="post_hoc"))
    assert "is refused, the profile implies resolved" in refused
    # A profile the diagnostic guard cannot bind implies the refusal itself.
    late = dataclasses.replace(
        column, features=dataclasses.replace(column.features, deadline="post_hoc")
    )
    assert details("late", late) == []
    (admitted,) = details("admitted", late, _GuardOn(deadline="request_only"))
    assert "is resolved, the profile implies refused" in admitted


def test_settlement_sees_a_new_child_a_named_process_and_a_new_listener() -> None:
    def settled(baseline: Any, **named: Any) -> contract.Settlement:
        arguments: dict[str, Any] = {"driver_children": (), "own_ports": ()}
        arguments.update(named)
        return contract.settle(baseline, close_raised=None, grace_seconds=0.0, **arguments)

    before = contract.settlement_baseline()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        after = contract.settlement_baseline()
        # A process started since the baseline is left, named by the driver or not.
        assert settled(before).children_alive == (child.pid,)
        assert settled(after).children_alive == ()
        # One that is no new child of this process is left when the driver names it.
        assert settled(after, driver_children=(child.pid,)).children_alive == (child.pid,)
        # A new listening port is left, unless it is the provider's own.
        assert settled(before).listeners == (port,)
        assert settled(before, own_ports=(port,)).listeners == ()
        assert settled(after).listeners == ()
    finally:
        listener.close()
        child.kill()
        child.wait(timeout=10)
    assert settled(before, driver_children=(child.pid,)) == contract.Settlement()


def _ask(
    provider: LoopbackProvider,
    body: dict[str, Any],
    *,
    method: str = "POST",
    path: str = ToyDialect.path,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=5)
    try:
        connection.request(method, path, body=json.dumps(body).encode("utf-8"))
        response = connection.getresponse()
        payload = response.read()
        return response.status, json.loads(payload) if payload else {}
    finally:
        connection.close()


def test_provider_matches_on_method_path_call_ids_and_advertised_tools() -> None:
    tools = {"tools": [BINDINGS[READ_A].name]}

    def answer(call_id: str) -> dict[str, Any]:
        return {"tool_results": [{"id": call_id, "text": "result"}]}

    # Method and path: only the scripted POST consumes a step.
    with LoopbackProvider(ProviderScript((Say("one"),)), credential="c") as provider:
        for method in ("GET", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS"):
            assert _ask(provider, {}, method=method)[0] == UNSCRIPTED_STATUS
        assert _ask(provider, {}, path="/toy/v1/other")[0] == UNSCRIPTED_STATUS
        assert _ask(provider, {})[1]["text"] == "one"
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed, seen.request_count) == (7, 0, 8)
    assert [item.matched for item in seen.requests] == [False] * 7 + [True]

    # Tool-call ids: a request must answer exactly the calls of the last reply.
    script = ProviderScript((Call(READ_A), Say("two")))
    with LoopbackProvider(script, credential="c") as provider:
        (call,) = _ask(provider, tools)[1]["calls"]
        assert _ask(provider, answer("another-id"))[0] == UNSCRIPTED_STATUS
        assert _ask(provider, {})[0] == UNSCRIPTED_STATUS
        assert _ask(provider, answer(call["id"]))[1]["text"] == "two"
        assert _ask(provider, answer(call["id"]))[0] == UNSCRIPTED_STATUS
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (3, 0)

    # The tail answers one request that advertises no tools, after the steps.
    script = ProviderScript((Say("three"),), tail=Say("tail"))
    with LoopbackProvider(script, credential="c") as provider:
        assert _ask(provider, tools)[1]["text"] == "three"
        assert _ask(provider, tools)[0] == UNSCRIPTED_STATUS
        assert _ask(provider, {})[1]["text"] == "tail"
        assert _ask(provider, {})[0] == UNSCRIPTED_STATUS
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (2, 0)

    # A loop answers requests that advertise tools; one without leaves it.
    script = ProviderScript((Loop(Call(READ_A), limit=3), Say("four")))
    with LoopbackProvider(script, credential="c") as provider:
        (call,) = _ask(provider, tools)[1]["calls"]
        assert _ask(provider, answer(call["id"]))[1]["text"] == "four"
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (0, 0)
    with LoopbackProvider(script, credential="c") as provider:
        assert _ask(provider, {})[0] == UNSCRIPTED_STATUS  # the loop needs one round
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (1, 2)


def test_run_cell_fails_on_a_failed_preflight_a_broken_seam_and_a_wrong_pin(
    tmp_path: Path,
) -> None:
    canary = next(cell for cell in TOY_CELLS if SCENARIOS[cell.scenario].canary)
    clean = Cell(canary.column, "S01", "raw")
    driven: dict[str, str | None] = {TOY_KIND: None}
    failed = pytest.fail.Exception

    with pytest.raises(failed, match="cannot be driven here: no worker runtime"):
        runner.run_cell(
            clean, tmp_path / "a", preflight={TOY_KIND: "no worker runtime"}, known=LEDGER
        )
    foreign = toy.ToyDriver(frozenset({toy.FOREIGN_USER_AGENT}))
    with pytest.raises(failed, match="the driver seam is broken.*user agent"):
        runner.run_cell(clean, tmp_path / "b", preflight=driven, known=LEDGER, driver=foreign)
    with pytest.raises(failed, match="C1 was not evaluated and inventory.yaml does not pin"):
        runner.run_cell(canary, tmp_path / "c", preflight=driven, known=LEDGER, pins=())
    stale = {**cells.load_inventory().not_evaluated[0], "scenario": clean.scenario}
    with pytest.raises(failed, match="C1 is pinned not evaluated"):
        runner.run_cell(clean, tmp_path / "d", preflight=driven, known=LEDGER, pins=(stale,))
    assert runner.run_cell(
        clean, tmp_path / "e", preflight=driven, known=LEDGER
    ).report.codes == frozenset()


def test_no_cell_sees_a_credential_or_configuration_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cleared, scripted = matrix_columns.CLEARED_ENV, matrix_columns.SCRIPTED_ENV

    assert len(set(cleared)) == len(cleared) and not set(cleared) & set(scripted)
    # The package fixture ran before this test.
    assert not set(cleared) & set(os.environ)
    assert {name: os.environ[name] for name in scripted} == dict(scripted)
    # A value a developer machine holds is deleted, not kept.
    for name in cleared:
        monkeypatch.setenv(name, "a-value-from-the-machine")
    again = tmp_path / "again"
    again.mkdir()
    runner.isolate_environment(monkeypatch, again)
    assert not set(cleared) & set(os.environ)
    assert os.environ["HOME"] == str(again / "matrix-home")


def test_not_evaluated_clause_needs_a_pin_and_a_pin_needs_a_clause_not_evaluated() -> None:
    cell = TOY_CELLS[0]
    report = Report(
        {
            "C0": ClauseOutcome("ok"),
            "C6": ClauseOutcome("not_evaluated", reason="the peer bills nothing"),
        }
    )

    def pin(*clauses: str) -> dict[str, Any]:
        return {
            "scenario": cell.scenario,
            "kind": cell.column.kind,
            "role": "*",
            "lens": "*",
            "clauses": list(clauses),
            "reason": "synthetic",
        }

    assert runner.pin_problems(cell, report, [pin("C6")]) == []
    unpinned = runner.pin_problems(cell, report, [])
    assert len(unpinned) == 1 and "C6 was not evaluated" in unpinned[0]
    stale = runner.pin_problems(cell, report, [pin("C0", "C6")])
    assert len(stale) == 1 and "C0 is pinned" in stale[0]


# -- nothing skips, nothing vanishes ----------------------------------------------


def test_inventory_and_snapshot_files_have_closed_shapes() -> None:
    inventory = yaml.safe_load(cells.INVENTORY_PATH.read_text(encoding="utf-8"))
    row = {"scenario": "S05", "kind": "kimi", "step": "Call", "reason": "r", "checklist": "c"}

    assert cells.parse_inventory({**inventory, "not_expressible": [row]}).not_expressible
    # A not-expressible row has no field for a declared value: none selects it.
    with pytest.raises(MatrixSetupError, match="unknown fields"):
        cells.parse_inventory(
            {**inventory, "not_expressible": [{**row, "structured_output": "native"}]}
        )
    with pytest.raises(MatrixSetupError, match="unknown clause"):
        pin = {**inventory["not_evaluated"][0], "clauses": ["C11"]}
        cells.parse_inventory({**inventory, "not_evaluated": [pin]})
    # The H01 pin names test files of a worker kind, and is deleted when empty.
    wrong: tuple[dict[str, Any], ...] = (
        {TOY_KIND: ["test_x.py"]},
        {"kimi": []},
        {"kimi": "test_x.py"},
    )
    for files in wrong:
        with pytest.raises(MatrixSetupError, match="worker_skip_files"):
            cells.parse_inventory({**inventory, "worker_skip_files": files})
    assert cells.parse_inventory({**inventory, "worker_skip_files": None}).worker_skip_files == {}

    snapshot = yaml.safe_load(declarations.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    kind = sorted(snapshot["columns"])[0]
    role = sorted(snapshot["columns"][kind])[0]
    del snapshot["columns"][kind][role]["features"]["usd_limit"]
    with pytest.raises(MatrixSetupError, match="exactly the 14 declared values"):
        declarations.parse_snapshot(snapshot)
    snapshot["columns"][kind][role]["features"]["usd_limit"] = "sometimes"
    with pytest.raises(MatrixSetupError, match="usd_limit"):
        declarations.parse_snapshot(snapshot)


def _lane(key: str, backend: str, transport: str) -> SimpleNamespace:
    return SimpleNamespace(
        key=key,
        backend=backend,
        transport=transport,
        runtime_backend=backend,
        routing_path="route.yaml",
        default_route=SimpleNamespace(model="standin-model"),
    )


_KINDS = {("standin", "wire"): "standin"}


def test_generator_raises_on_an_unmapped_backend_transport_pair() -> None:
    with pytest.raises(MatrixSetupError, match="maps to no driver kind"):
        matrix_columns.build_columns(
            [_lane("one", "standin", "other-wire")],
            product="x",
            kind_table=_KINDS,
            role_table={"one": "primary"},
        )


def test_generator_raises_on_a_duplicate_kind_and_role() -> None:
    lanes = [_lane("one", "standin", "wire"), _lane("two", "standin", "wire")]

    with pytest.raises(MatrixSetupError, match="share the column standin/primary"):
        matrix_columns.build_columns(
            lanes,
            product="x",
            kind_table=_KINDS,
            role_table={"one": "primary", "two": "primary"},
        )
    numbered = matrix_columns.build_columns(
        lanes,
        product="x",
        kind_table=_KINDS,
        role_table={"one": "secondary", "two": "secondary-2"},
    )
    assert [column.id for column in numbered] == ["standin/secondary", "standin/secondary-2"]
    with pytest.raises(MatrixSetupError, match="share the \\(kind, role\\)"):
        cells.generate(
            [numbered[0], numbered[0]],
            registered=frozenset(),
            enablement={"standin": ()},
        )


@pytest.mark.parametrize(
    ("roles", "kinds", "message"),
    [
        ({}, _KINDS, "has no role"),
        ({"one": "primary", "gone": "secondary"}, _KINDS, "does not list"),
        ({"one": "main"}, _KINDS, "malformed"),
        ({"one": "primary"}, {**_KINDS, ("idle", "wire"): "idle"}, "has no catalog column"),
    ],
    ids=["profile-without-role", "role-without-profile", "bad-role", "kind-without-column"],
)
def test_role_and_kind_tables_must_match_the_catalog(roles, kinds, message) -> None:
    with pytest.raises(MatrixSetupError, match=message):
        matrix_columns.build_columns(
            [_lane("one", "standin", "wire")],
            product="x",
            kind_table=kinds,
            role_table=roles,
        )


def _standin(role: str) -> Column:
    return Column("standin", role, f"p-{role}", "standin", "standin", "standin-model")


def test_generator_raises_on_an_empty_cell_set_of_a_driven_kind() -> None:
    with pytest.raises(MatrixSetupError, match="yields no cell"):
        cells.generate(
            [_standin("primary")],
            registered=frozenset({"standin"}),
            enablement={"standin": ()},
        )
    with pytest.raises(MatrixSetupError, match="has no column"):
        cells.generate(
            [_standin("primary")],
            registered=frozenset({"standin", "absent"}),
            enablement={"standin": (STARTER,)},
        )
    with pytest.raises(MatrixSetupError, match="yield no cell"):
        cells.cells_of(("standin",))


def test_kinds_without_a_driver_yield_no_cell_and_no_error() -> None:
    generated = cells.generate(cells.all_columns(), registered=frozenset({TOY_KIND}))

    assert {cell.column.kind for cell in generated} == {TOY_KIND}
    assert generated == cells.behaviour_cells()
    assert set(matrix_columns.REAL_KINDS) & registered_kinds() == set()


def test_role_scope_and_the_not_expressible_list_shape_a_column() -> None:
    columns = [_standin("primary"), _standin("secondary-2"), _standin("blocked")]
    enablement = {"standin": (STARTER, PACK_A)}
    generated = cells.generate(
        columns, registered=frozenset({"standin"}), enablement=enablement
    )
    by_role = {
        role: {cell.scenario for cell in generated if cell.column.role == role}
        for role in ("primary", "secondary-2", "blocked")
    }

    assert by_role["primary"] == set(SCENARIOS) == by_role["secondary-2"]
    assert by_role["blocked"] == {"S00", "S01"}
    row = {"scenario": "S23c", "kind": "standin", "step": "Refuse", "reason": "r", "checklist": "c"}
    pinned = cells.generate(
        columns,
        registered=frozenset({"standin"}),
        enablement=enablement,
        not_expressible=[row],
    )
    assert "S23c" not in {cell.scenario for cell in pinned}
    assert len(generated) - len(pinned) == 2  # primary and secondary, one lens each


def test_canary_seam_names_a_request_the_toy_client_did_not_send() -> None:
    dialect = ToyDialect()
    seen = dialect.parse(
        "POST",
        dialect.path,
        {"User-Agent": dialect.user_agent, "Authorization": "Bearer canary"},
        b"{}",
        credential="canary",
    )

    assert dialect.seam_problems(seen) == []
    stub = dataclasses.replace(seen, user_agent="python-requests", credential_ok=False)
    assert len(dialect.seam_problems(stub)) == 2
    assert len(dialect.seam_problems(dataclasses.replace(seen, path="/v1/other"))) == 1


def test_canary_check_fails_when_s00_did_not_pass_for_a_column() -> None:
    outcomes = {cell.id: "passed" for cell in TOY_CELLS}
    canary = next(cell.id for cell in TOY_CELLS if SCENARIOS[cell.scenario].canary)

    assert runner.canary_problems(outcomes, TOY_CELLS) == []
    outcomes[canary] = "skipped"  # how an expected failure is recorded
    assert any(canary in line for line in runner.canary_problems(outcomes, TOY_CELLS))
    del outcomes[canary]
    assert any("not run" in line for line in runner.canary_problems(outcomes, TOY_CELLS))


def test_guard_blocks_connect_datagram_send_and_name_resolution() -> None:
    assert netguard.installed()
    assert not issubclass(netguard.NetworkGuardViolation, OSError)
    stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    datagram = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(netguard.NetworkGuardViolation):
            stream.connect(("192.0.2.1", 9))
        with pytest.raises(netguard.NetworkGuardViolation):
            stream.connect_ex(("192.0.2.1", 9))
        with pytest.raises(netguard.NetworkGuardViolation):
            datagram.sendto(b"x", ("192.0.2.1", 9))
        with pytest.raises(netguard.NetworkGuardViolation):
            datagram.sendmsg([b"x"], [], 0, ("192.0.2.1", 9))
        with pytest.raises(netguard.NetworkGuardViolation):
            socket.getaddrinfo("provider.invalid", 443)
        with pytest.raises(netguard.NetworkGuardViolation):
            socket.gethostbyname("provider.invalid")
        with pytest.raises(netguard.NetworkGuardViolation):
            socket.create_connection(("provider.invalid", 443), timeout=1)
    finally:
        stream.close()
        datagram.close()

    # Taking the record is what keeps this test's own teardown from failing.
    recorded = netguard.take_violations()
    assert len(recorded) == 7
    assert recorded[0] == "connect -> ('192.0.2.1', 9)"


def test_guard_is_removed_without_residue_and_never_installed_twice() -> None:
    with pytest.raises(RuntimeError, match="already installed"):
        netguard.install()
    netguard.remove()
    try:
        assert not netguard.installed()
        assert not {"connect", "connect_ex", "sendto", "sendmsg"} & set(vars(socket.socket))
        assert socket.getaddrinfo.__module__ == socket.create_connection.__module__ == "socket"
        with pytest.raises(RuntimeError, match="not installed"):
            netguard.remove()
    finally:
        netguard.install()  # the fixture's teardown removes it again


def test_guard_lets_loopback_and_unix_sockets_pass(tmp_path: Path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    local = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    local.bind(str(tmp_path / "guard.sock"))
    local.listen(1)
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        socket.create_connection(("127.0.0.1", port), timeout=5).close()
        assert socket.getaddrinfo("localhost", port)
        sender.sendto(b"x", receiver.getsockname())
        assert receiver.recv(1) == b"x"
        peer.connect(str(tmp_path / "guard.sock"))
    finally:
        for item in (listener, receiver, sender, local, peer):
            item.close()

    assert netguard.take_violations() == ()


_INNER_CONFTEST = """
from pathlib import Path

import pytest
from inner_known import KNOWN

from crucible.tests.backend_matrix import cells, runner

ROOT = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _guard():
    yield from runner.network_guard()


def pytest_collection_modifyitems(config, items):
    cells.refuse_skip_markers(items, ROOT)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    runner.judge_report(item, outcome.get_result(), KNOWN)
"""
_INNER_KNOWN = """
from crucible.tests.backend_matrix import ledger

CELL = "kimi/H01/-/static"
CODES = {"static:worker_skip"}
KNOWN = ledger.parse(
    {
        "schema_version": ledger.LEDGER_SCHEMA,
        "ceilings": {"max_known_gaps": {"pack_a": 1}, "max_declared_differences": 0},
        "registry": {
            "G99": {
                "title": "a known gap",
                "origin": "gap list",
                "severity": "medium",
                "cells": [{"cell": CELL, "observed_under": {}, "closes_under": {}}],
            }
        },
        "known_gaps": [
            {"cell": CELL, "observed_under": {}, "codes": {code: "G99" for code in CODES}}
        ],
        "declared_differences": [],
    }
)
"""
_INNER_CANARY = """
import pytest

from crucible.tests.backend_matrix import cells, ledger, runner
from crucible.tests.backend_matrix.columns import TOY_KIND
from crucible.tests.backend_matrix.scenarios import SCENARIOS

CANARIES = [
    cell for cell in cells.cells_of((TOY_KIND,)) if SCENARIOS[cell.scenario].canary
][:3]


@pytest.mark.parametrize("cell", CANARIES, ids=lambda cell: cell.id)
def test_cell(cell):
    if cell is CANARIES[0]:
        raise ledger.LedgerMismatch("the canary of the first column fails")


def test_canary_passed_for_every_column(request):
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CANARIES) == []
"""
_INNER_GUARD = """
import socket


def test_blocked_call_swallowed_by_the_code_under_test():
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=1)
    except Exception:
        pass


def test_loopback_passes():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    socket.create_connection(server.getsockname(), timeout=5).close()
    server.close()
"""
_INNER_LEDGER = """
import pytest
from inner_known import CELL, CODES, KNOWN

from crucible.tests.backend_matrix import ledger


@pytest.mark.parametrize("cell", [CELL], ids=str)
def test_known_gap(cell):
    ledger.compare(KNOWN, cell, CODES, current={})
"""
_INNER_STRAY = """
import pytest
from inner_known import CELL, CODES, KNOWN

from crucible.tests.backend_matrix import ledger


def test_a_self_test_that_reaches_the_comparators_verdict():
    ledger.compare(KNOWN, CELL, CODES, current={})


@pytest.mark.parametrize("cell", ["kimi/H02/-/static"], ids=str)
def test_a_cell_without_an_entry(cell):
    ledger.compare(KNOWN, CELL, CODES, current={})
"""
_INNER_MARKED = """
import pytest


@pytest.mark.skip(reason="not here")
def test_skipped():
    pass


@pytest.mark.xfail(reason="known")
def test_marked_expected_failure():
    assert False


@pytest.mark.parametrize("value", [])
def test_empty_parameter_set(value):
    pass
"""


def _inner_tree(root: Path) -> Path:
    root.mkdir()
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    for name, source in (
        ("conftest.py", _INNER_CONFTEST),
        ("inner_known.py", _INNER_KNOWN),
        ("test_inner_canary.py", _INNER_CANARY),
        ("test_inner_guard.py", _INNER_GUARD),
        ("test_inner_ledger.py", _INNER_LEDGER),
        ("test_inner_marked.py", _INNER_MARKED),
        ("test_inner_stray.py", _INNER_STRAY),
    ):
        (root / name).write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return root


def _inner_pytest(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *arguments],
        cwd=root,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )


def test_known_gap_is_xfailed_in_junit_and_a_swallowed_violation_fails_teardown(
    tmp_path: Path,
) -> None:
    root = _inner_tree(tmp_path / "inner")

    run = _inner_pytest(
        root, "test_inner_guard.py", "test_inner_ledger.py", "--junitxml=junit.xml"
    )
    junit = (root / "junit.xml").read_text(encoding="utf-8")

    assert run.returncode == 1, run.stdout + run.stderr
    assert "1 xfailed" in run.stdout and "1 error" in run.stdout
    assert "blocked by the matrix guard" in run.stdout
    assert "test_loopback_passes" not in run.stdout.split("short test summary")[-1]
    assert 'type="pytest.xfail"' in junit and 'type="pytest.skip"' not in junit


def test_runxfail_turns_an_expected_failure_into_a_failure(tmp_path: Path) -> None:
    root = _inner_tree(tmp_path / "inner")

    run = _inner_pytest(root, "test_inner_ledger.py", "--runxfail")

    assert run.returncode == 1, run.stdout + run.stderr
    assert "1 failed" in run.stdout and "known gap still open" in run.stdout


def test_canary_check_fails_in_a_run_where_a_canary_did_not_pass(tmp_path: Path) -> None:
    root = _inner_tree(tmp_path / "inner")

    run = _inner_pytest(root, "test_inner_canary.py", "-vv")

    # The recorder keeps what really happened to each cell, so the module's
    # closing check fails when the canary of one column failed.
    assert run.returncode == 1, run.stdout + run.stderr
    assert "2 failed, 2 passed" in run.stdout
    assert run.stdout.count("did not pass in this process (failed)") >= 1
    assert "did not pass in this process (not run)" not in run.stdout


def test_recorder_keeps_an_expected_failure_apart_from_a_pass() -> None:
    cell = TOY_CELLS[0]
    stash: dict[Any, Any] = {}
    item = SimpleNamespace(
        callspec=SimpleNamespace(params={"cell": cell}), config=SimpleNamespace(stash=stash)
    )

    def recorded(**report: Any) -> str:
        runner.record_outcome(item, SimpleNamespace(when="call", **report))
        return stash[runner.OUTCOMES][cell.id]

    assert recorded(passed=True, outcome="passed") == "passed"
    assert recorded(passed=False, outcome="failed") == "failed"
    # How pytest reports an expected failure: skipped, with the reason kept.
    assert recorded(passed=False, outcome="skipped", wasxfail="G99") == "skipped"
    # Set-up and tear-down reports do not overwrite the call's outcome.
    runner.record_outcome(item, SimpleNamespace(when="teardown", passed=True, outcome="passed"))
    assert stash[runner.OUTCOMES][cell.id] == "skipped"


def test_an_expected_failure_outside_a_ledgered_cell_is_a_failure(tmp_path: Path) -> None:
    root = _inner_tree(tmp_path / "inner")

    run = _inner_pytest(root, "test_inner_ledger.py", "test_inner_stray.py")

    # The ledgered cell's own item is the expected failure; a self-test and a
    # cell without an entry that reach the same verdict fail.
    assert run.returncode == 1, run.stdout + run.stderr
    assert "2 failed, 1 xfailed" in run.stdout
    assert run.stdout.count("is not the test item of a cell with a ledger entry") >= 2


def test_skip_and_xfail_markers_are_a_collection_error(tmp_path: Path) -> None:
    root = _inner_tree(tmp_path / "inner")

    run = _inner_pytest(root, "test_inner_marked.py")
    output = run.stdout + run.stderr

    assert run.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "refuses skip and xfail markers" in output
    for offender in ("test_skipped (skip)", "test_marked_expected_failure (xfail)", "NOTSET"):
        assert offender in output
    assert "passed" not in output and "skipped" not in output.split("refuses")[0]


# -- H03 and P04 on synthetic ledgers and inventories ----------------------------


def _document() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(ledger.LEDGER_PATH.read_text(encoding="utf-8")))


def _problem_codes(document: dict[str, Any] | None = None, **world: Any) -> set[str]:
    known = LEDGER if document is None else ledger.parse(document)
    found = ledger.integrity_problems(known, dataclasses.replace(cells.world(), **world))
    return {item.code for item in found}


def _registry_cell(document: dict[str, Any], gap: str, cell: str) -> dict[str, Any]:
    return next(item for item in document["registry"][gap]["cells"] if item["cell"] == cell)


def test_h03_rejects_an_entry_whose_cell_its_gap_does_not_list() -> None:
    document = _document()
    document["known_gaps"][0]["codes"] = {"static:worker_skip": "G02"}

    assert "static:entry_not_in_registry" in _problem_codes(document)


def test_h03_rejects_a_registry_cell_that_is_not_generated_once_its_kind_runs_the_set() -> None:
    enabled = {**cells.ENABLEMENT, "deepseek": (STARTER, PACK_A)}

    assert "static:registry_cell_not_generated" in _problem_codes(enabled_sets=enabled)
    assert "static:registry_cell_not_generated" not in _problem_codes()


def test_h03_rejects_a_not_expressible_row_on_a_registry_cell_without_a_ruling() -> None:
    row = {"scenario": "S12", "kind": "kimi", "step": "Call", "reason": "r", "checklist": "c"}
    ruled = _document()
    ruled["registry"]["G02"]["ruling"] = "owner, 2026-11-18: the cell is withdrawn"

    assert "static:not_expressible_registry_cell" in _problem_codes(not_expressible=(row,))
    assert "static:not_expressible_registry_cell" not in _problem_codes(
        ruled, not_expressible=(row,)
    )
    # S12 has one form, so it cannot be pinned on an open-weight kind at all.
    assert "static:not_expressible_pack_a" in _problem_codes(ruled, not_expressible=(row,))


def test_h03_keeps_one_form_of_a_multi_form_pack_a_scenario_on_open_weight_kinds() -> None:
    def rows(*forms: str) -> tuple[dict[str, str], ...]:
        return tuple(
            {"scenario": form, "kind": "kimi", "step": "Refuse", "reason": "r", "checklist": "c"}
            for form in forms
        )

    assert "static:not_expressible_pack_a" not in _problem_codes(
        not_expressible=rows("S23a", "S23b")
    )
    assert "static:not_expressible_pack_a" in _problem_codes(
        not_expressible=rows("S23a", "S23b", "S23c")
    )
    toy_row = {"scenario": "S23a", "kind": TOY_KIND, "step": "Refuse", "reason": "r", "checklist": "c"}
    assert "static:not_expressible_on_toy" in _problem_codes(not_expressible=(toy_row,))


def test_h03_rejects_closes_under_that_differs_without_a_named_change() -> None:
    cell = "deepseek/S10a/primary/raw"
    document = _document()
    _registry_cell(document, "G14", cell)["closes_under"] = {"turn_limit": "unsupported"}

    assert "static:registry_closes_under" in _problem_codes(document)

    document["registry"]["G14"]["declaration_change"] = {
        "field": "turn_limit",
        "from": "hard",
        "to": "unsupported",
        "citation": "an owner ruling",
    }
    named = _problem_codes(document)
    assert "static:registry_closes_under" not in named
    # The change is named, but the cell does not run under it: still not closed.
    assert "static:registry_closes_mismatch" in named


def test_h03_rejects_a_not_evaluated_pin_on_a_toy_cell() -> None:
    def pin(scenario: str, kind: str) -> dict[str, Any]:
        return {
            "scenario": scenario,
            "kind": kind,
            "role": "*",
            "lens": "*",
            "clauses": ["C6"],
            "reason": "synthetic",
        }

    assert "static:not_evaluated_on_toy" in _problem_codes(not_evaluated=(pin("S01", TOY_KIND),))
    assert "static:not_evaluated_on_toy" in _problem_codes(not_evaluated=(pin("S01", "*"),))
    assert "static:not_evaluated_on_toy" not in _problem_codes(
        not_evaluated=(pin("S00", "*"), pin("S01", "kimi"))
    )


def test_h03_registry_rules() -> None:
    missing = _document()
    del missing["registry"]["G15"]
    assert "static:registry_gap_missing" in _problem_codes(missing)

    undefined = _document()
    _registry_cell(undefined, "G15", "deepseek/S11/primary/raw")["cell"] = (
        "deepseek/S11/primary/guarded"
    )
    assert "static:registry_cell_undefined" in _problem_codes(undefined)

    unpredicted = _document()
    del _registry_cell(unpredicted, "G15", "deepseek/S11/primary/raw")["predicted"]
    assert "static:registry_prediction_missing" in _problem_codes(unpredicted)

    # Without the H02 entry of its kind, a predicted cell would be invisible.
    unreachable = _document()
    unreachable["known_gaps"] = [
        item for item in unreachable["known_gaps"] if item["cell"] != "deepseek/H02/-/static"
    ]
    assert "static:registry_cell_unreachable" in _problem_codes(unreachable)

    # A weakened declaration is seen even before the cell's column is driven.
    weakened = dataclasses.replace(cells.world(), current=lambda _cell: {"turn_limit": "unsupported"})
    assert "static:registry_closes_mismatch" in {
        item.code for item in ledger.integrity_problems(LEDGER, weakened)
    }

    # H02 has no code for a set beyond pack A, so a cell of such a set may be
    # recorded only once its kind runs the set.
    real = cells.world()
    assert "static:registry_cell_early" in _problem_codes(
        scenario_sets={**real.scenario_sets, "S12": "pack_b"}
    )

    # A static registry cell is generated from the start, or the record is wrong.
    ungenerated = _document()
    ungenerated["registry"]["G04"]["cells"].append(
        {
            "cell": "all/H09/-/static",
            "observed_under": {},
            "closes_under": {},
            "predicted": ["static:worker_skip"],
        }
    )
    assert "static:registry_cell_not_generated" in _problem_codes(
        ungenerated, static_scenarios=real.static_scenarios | {"H09"}
    )
    assert _problem_codes() == set()


def test_h03_entry_rules() -> None:
    on_canary = _document()
    canary = next(cell.id for cell in TOY_CELLS if SCENARIOS[cell.scenario].canary)
    on_canary["known_gaps"].append(
        {"cell": canary, "observed_under": {}, "codes": {"request.unconsumed": "G02"}}
    )
    assert {"static:entry_on_canary", "static:ceiling_exceeded"} <= _problem_codes(on_canary)

    unused = _document()
    unused["known_gaps"][0]["cell"] = "all/H09/-/static"
    assert "static:entry_unused" in _problem_codes(unused)

    observed = _document()
    observed["known_gaps"][0]["observed_under"] = {"turn_limit": "hard"}
    assert "static:entry_observed_under" in _problem_codes(observed)

    invalid = _document()
    invalid["known_gaps"][0]["codes"] = {"decl.changed:turn_limit": "G04"}
    assert "static:entry_code_invalid" in _problem_codes(invalid)

    ceiling = _document()
    ceiling["ceilings"]["max_known_gaps"]["pack_a"] = 7
    assert "static:ceiling_exceeded" in _problem_codes(ceiling)

    reference = _document()
    reference["registry"]["M01"]["cells"].append(
        {"cell": "claude/S01/primary/raw", "observed_under": {}, "closes_under": {}}
    )
    reference["known_gaps"].append(
        {"cell": "claude/S01/primary/raw", "observed_under": {}, "codes": {"secret:leak": "M01"}}
    )
    assert "static:entry_ruling_missing" in _problem_codes(reference)

    duplicate = _document()
    duplicate["known_gaps"].append(copy.deepcopy(duplicate["known_gaps"][0]))
    assert "static:entry_duplicate" in _problem_codes(duplicate)

    empty = _document()
    empty["known_gaps"][0]["codes"] = {}
    assert _problem_codes(empty) == {"static:entry_without_codes"}
    # The comparator does not let such an entry excuse a passing cell either.
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed, delete the entry"):
        ledger.compare(ledger.parse(empty), empty["known_gaps"][0]["cell"], set(), current={})

    differing = _document()
    differing["declared_differences"] = [
        {"id": "D-EXAMPLE", "citation": "c", "owner": "o", "date": "2026-11-18", "alternative": "a"}
    ]
    assert _problem_codes(differing) == {"static:ceiling_exceeded"}
    differing["ceilings"]["max_declared_differences"] = 1
    assert _problem_codes(differing) == set()

    # A column-wide entry is attributed like any other: its gap must list it.
    wide = _document()
    wide["ceilings"]["max_known_gaps"]["pack_a"] += 1
    wide["known_gaps"].append(
        {"column_wide": {"kind": "kimi", "code": "pricing:none", "gap": "G02"}}
    )
    assert _problem_codes(wide) == {"static:entry_not_in_registry"}
    wide["registry"]["G02"]["column_wide"] = [{"kind": "kimi", "code": "pricing:none"}]
    assert _problem_codes(wide) == set()


def _single_entry_ledger(cell: str, code: str) -> ledger.Ledger:
    record = {
        "title": "a known gap",
        "origin": "gap list",
        "severity": "medium",
        "cells": [{"cell": cell, "observed_under": {}, "closes_under": {}}],
    }
    return ledger.parse(
        {
            "schema_version": ledger.LEDGER_SCHEMA,
            "ceilings": {"max_known_gaps": {"pack_a": 1}, "max_declared_differences": 0},
            "registry": {"G99": record},
            "known_gaps": [{"cell": cell, "observed_under": {}, "codes": {code: "G99"}}],
            "declared_differences": [],
        }
    )


def test_an_integrity_cell_is_never_an_expected_failure() -> None:
    moved = [Deviation("static:snapshot_mismatch", "a declared value moved")]

    assert ledger.LEDGERABLE_STATIC == {"H01", "H02"}
    for scenario in sorted(set(cells.STATIC_SCENARIOS) - ledger.LEDGERABLE_STATIC):
        cell = ledger.cell_id(ledger.ALL_KINDS, scenario, ledger.NO_ROLE, ledger.STATIC_LENS)
        known = _single_entry_ledger(cell, moved[0].code)
        # With an entry that lists exactly its codes, the cell still fails ...
        with pytest.raises(ledger.LedgerMismatch, match="never an expected failure"):
            ledger.judge_static(known, cell, moved)
        # ... and passes only when it is clean.
        ledger.judge_static(known, cell, [])
    for scenario in sorted(ledger.LEDGERABLE_STATIC):
        cell = ledger.cell_id("kimi", scenario, ledger.NO_ROLE, ledger.STATIC_LENS)
        with pytest.raises(XFAILED):
            ledger.judge_static(_single_entry_ledger(cell, moved[0].code), cell, moved)


def test_h03_rejects_an_entry_on_an_integrity_cell_or_a_toy_cell() -> None:
    column = cells.toy_columns()[0]
    behaviour = next(cell.id for cell in TOY_CELLS if not SCENARIOS[cell.scenario].canary)
    for cell, code in (
        ("all/H03/-/static", "static:registry_closes_mismatch"),
        ("all/P04/-/static", "static:snapshot_mismatch"),
        ("all/H04/-/static", "static:worker_env_unguarded"),
        (ledger.cell_id(TOY_KIND, "P02", column.role, ledger.STATIC_LENS), "static:profile_binding"),
        (behaviour, "kind:transport!=none"),
    ):
        document = _document()
        document["registry"]["G02"]["cells"].append(
            {"cell": cell, "observed_under": {}, "closes_under": {}}
        )
        document["known_gaps"].append(
            {"cell": cell, "observed_under": {}, "codes": {code: "G02"}}
        )
        assert "static:entry_on_integrity_cell" in _problem_codes(document), cell

    wide = _document()
    wide["known_gaps"].append(
        {"column_wide": {"kind": TOY_KIND, "code": "pricing:none", "gap": "G02"}}
    )
    assert "static:entry_on_integrity_cell" in _problem_codes(wide)


def test_h03_inventory_and_oracle_rules() -> None:
    real = cells.world()

    assert "static:inventory_count" in _problem_codes(
        pinned_counts={**real.pinned_counts, TOY_KIND: real.pinned_counts[TOY_KIND] - 1}
    )
    assert "static:scenario_without_liar" in _problem_codes(
        liar_scenarios=real.liar_scenarios - {"S12"}
    )
    assert "static:clause_not_tripped" in _problem_codes(
        liar_clauses=real.liar_clauses - {"C8"}
    )
    declared = {**real.snapshot_values, "usd_limit": frozenset({"post_hoc", "hard"})}
    assert "static:value_without_honest_pass" in _problem_codes(snapshot_values=declared)


def _p04(**changes: Any) -> set[str]:
    arguments: dict[str, Any] = {
        "columns": matrix_columns.real_columns(),
        "snapshot": declarations.load_snapshot(),
        "lanes_by_product": {
            product: matrix_columns.catalog_lanes(product)
            for product in matrix_columns.PRODUCTS
        },
    }
    arguments.update(changes)
    columns = arguments.pop("columns")
    snapshot = arguments.pop("snapshot")
    return {
        item.code for item in declarations.snapshot_problems(columns, snapshot, **arguments)
    }


def test_p04_fails_on_a_column_that_resolves_to_the_conservative_fallback() -> None:
    victim = matrix_columns.real_columns()[0]

    def routed(column: Column, product: str):
        if column == victim:
            return CONSERVATIVE_BACKEND_FEATURES
        return declarations.routed(column, product)

    assert _p04() == set()
    assert _p04(routed_fn=routed) == {"static:conservative_profile"}
    # A snapshot row copied from such a column would pin the loss; it is refused.
    snapshot = dict(declarations.load_snapshot())
    key = (victim.kind, victim.role)
    snapshot[key] = declarations.SnapshotRow(snapshot[key].adapter, CONSERVATIVE_BACKEND_FEATURES)
    assert "static:conservative_profile" in _p04(snapshot=snapshot, routed_fn=routed)


def test_p04_compares_snapshot_and_catalogs_in_both_directions() -> None:
    snapshot = dict(declarations.load_snapshot())
    key = sorted(snapshot)[0]
    row = snapshot[key]
    weakened = dataclasses.replace(row.features, usd_limit="unsupported")
    lanes = {
        product: matrix_columns.catalog_lanes(product)
        for product in matrix_columns.PRODUCTS
    }
    other = next(product for product in lanes if product != matrix_columns.COLUMN_PRODUCT)

    assert _p04(snapshot={**snapshot, key: declarations.SnapshotRow(row.adapter, weakened)}) == {
        "static:snapshot_mismatch"
    }
    assert _p04(snapshot={**snapshot, key: declarations.SnapshotRow("other", row.features)}) == {
        "static:snapshot_mismatch"
    }
    assert _p04(snapshot={k: v for k, v in snapshot.items() if k != key}) == {
        "static:snapshot_mismatch"
    }
    assert _p04(snapshot={**snapshot, ("standin", "primary"): row}) == {
        "static:snapshot_mismatch"
    }
    assert _p04(lanes_by_product={**lanes, other: lanes[other][1:]}) == {
        "static:catalog_mismatch"
    }


def test_p04_computes_the_profile_from_every_products_routing_file() -> None:
    victim = matrix_columns.real_columns()[0]
    products = [name for name, _path in victim.routing]
    assert products[0] == matrix_columns.COLUMN_PRODUCT and len(products) == len(
        matrix_columns.PRODUCTS
    )

    def routed(column: Column, product: str) -> BackendFeatureSet:
        profile = declarations.routed(column, product)
        if column == victim and product == products[-1]:
            other = "unsupported" if profile.usd_limit != "unsupported" else "post_hoc"
            return dataclasses.replace(profile, usd_limit=other)  # type: ignore[arg-type]
        return profile

    # A difference in the second catalog's routing file alone fails P04.
    assert _p04(routed_fn=routed) == {"static:snapshot_mismatch"}


def test_p04_reads_the_routing_file_of_the_product_it_is_asked_for(tmp_path: Path) -> None:
    columns = matrix_columns.real_columns()
    victim = columns[0]
    first = matrix_columns.COLUMN_PRODUCT
    second = next(product for product in matrix_columns.PRODUCTS if product != first)
    donor = next(
        column
        for column in columns
        if declarations.routed(column, second) != declarations.routed(victim, second)
    )
    # The victim's second-product route becomes a file that resolves to another
    # profile; its first-product route is untouched.
    foreign = tmp_path / "routing.yaml"
    foreign.write_text(
        donor.routing_path(second).read_text(encoding="utf-8"), encoding="utf-8"
    )
    moved = dataclasses.replace(
        victim,
        routing=tuple(
            (product, foreign if product == second else path)
            for product, path in victim.routing
        ),
    )

    assert declarations.routed(moved, first) == declarations.routed(victim, first)
    assert declarations.routed(moved, second) == declarations.routed(donor, second)
    assert _p04() == set()
    assert _p04(columns=(moved, *columns[1:])) == {"static:snapshot_mismatch"}


def test_a_route_whose_roles_resolve_to_different_profiles_is_a_setup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    column = matrix_columns.real_columns()[0]
    for_route = declarations.feature_profile_for_route
    asked: list[str] = []

    def last_role_differs(backend: str, **route: Any) -> BackendFeatureSet:
        profile = for_route(backend, **route)
        asked.append(backend)
        if len(asked) < len(declarations.ROUTED_ROLES):
            return profile
        other = "unsupported" if profile.usd_limit != "unsupported" else "post_hoc"
        return dataclasses.replace(profile, usd_limit=other)  # type: ignore[arg-type]

    agreed = declarations.routed(column)
    monkeypatch.setattr(declarations, "feature_profile_for_route", last_role_differs)

    # One variant is selected per column, so its roles must resolve alike.
    with pytest.raises(MatrixSetupError, match="resolve to different feature profiles"):
        declarations.routed(column)
    assert len(asked) == len(declarations.ROUTED_ROLES) == 3
    assert agreed != CONSERVATIVE_BACKEND_FEATURES


def test_toy_columns_are_derived_from_the_snapshot_not_typed() -> None:
    snapshot = declarations.snapshot_profiles(declarations.load_snapshot())
    fields = governing_fields()
    assert fields == ("event_detail", "structured_output", "turn_limit", "usd_limit")
    combinations = {tuple(str(getattr(row, name)) for name in fields) for row in snapshot.values()}

    derived = cells.toy_columns()

    assert {tuple(column.role.split("+")) for column in derived} == combinations
    # Every toy column plays every scenario in each of its lenses.
    per_column = sum(len(scenario.lenses) for scenario in SCENARIOS.values())
    assert len(derived) == len(combinations) and len(TOY_CELLS) == len(derived) * per_column
    assert all(column.features in snapshot.values() for column in derived)
    # A declared value the toy cannot play has no honest pass: refused at collection.
    assert derived[0].features is not None
    unplayable = dataclasses.replace(derived[0].features, usd_limit="hard")
    assert toy.unsupported_values(unplayable) == ["usd_limit=hard"]


def test_inventory_is_exactly_what_this_module_collected(
    request: pytest.FixtureRequest,
) -> None:
    listed = cells.inventory()
    collected = sorted(
        getattr(item, "callspec").id  # noqa: B009 - only parametrised items have one
        for item in request.session.items
        if getattr(item, "originalname", "") == "test_cell"
        and Path(str(item.path)) == Path(__file__)
    )

    assert collected == sorted(listed["behaviour"][TOY_KIND])
    assert {kind: len(ids) for kind, ids in listed["behaviour"].items()} == dict(
        cells.load_inventory().cell_counts
    )
    # P04, H03 and H04 once; H01 per worker; H02 per kind; P01 and P02 per toy column.
    assert len(listed["static"]) == len(set(listed["static"])) == (
        3
        + len(matrix_columns.WORKER_KINDS)
        + len(matrix_columns.REAL_KINDS)
        + 2 * len(cells.toy_columns())
    )


# -- the module ends with the canary check --------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, TOY_CELLS) == []
