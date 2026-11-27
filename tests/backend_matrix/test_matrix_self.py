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

import ast
import contextlib
import copy
import dataclasses
import http.client
import json
import logging
import math
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import psutil
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
    UsageCostBounds,
    UsageCostBoundSegment,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from crucible.backends.config import BackendRouting, RoleConfig, load_routing
from crucible.backends.features import FeatureAdmissionError
from crucible.backends.pricing import ModelPricing

from . import cells, contract, declarations, ledger, mutation, netguard, runner, toolsets
from . import scenarios
from . import report as matrix_report
from . import write_host as host_program
from . import columns as matrix_columns
from . import scenarios as scenarios_module
from . import envelopes as matrix_envelopes
from . import verify as matrix_verify
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
from .provider import (
    UNSCRIPTED_STATUS,
    CacheCountingDialect,
    CacheCounts,
    KeeperDialect,
    LoopbackProvider,
    ProviderLedger,
    Reply,
    ResponseCapDialect,
    SeenRequest,
    StatelessDialect,
    ToyDialect,
    TurnControlDialect,
    cut_after,
    cut_inside,
    whole_and_partial,
)
from .scenarios import (
    BASE_POLICY,
    BUDGET_USD,
    D_CDX_SERVED,
    D_COST_BOUND_UNRANKED,
    D_OAI_SCHEMA_FINAL,
    D_STRUCT_UNQUALIFIED,
    DEADLINE_SLACK_SECONDS,
    GUARDED,
    MALFORMED_ARGUMENT_TEXTS,
    PACK_A,
    PACK_B,
    SCENARIOS,
    SCHEMA_ANSWER,
    SCHEMA_FINAL_ANSWER,
    SCHEMA_TITLE,
    STARTER,
    budget_in_request,
    governing_fields,
    history_replayed,
    no_overshoot,
    plan_for,
    suite_default_max_turns,
    tool_events_paired,
    turn_count_in_request,
)
from .script import (
    READ_A,
    WRITE,
    Call,
    Cut,
    Drop,
    Empty,
    Loop,
    Malformed,
    ProviderScript,
    Refuse,
    Say,
    ServedAs,
    Step,
    TokenUsage,
)
from .toolsets import BINDINGS, NOTES_FILE, WRITER

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
        # What a liar needs of its column: a declared value, or the state of
        # a route extra that the toy column states itself.
        stated = dict(column.route_states)
        if all(
            (stated.get(k) if k in declarations.KEYING_EXTRAS else str(getattr(column.features, k)))
            == v
            for k, v in liar.needs.items()
        ):
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

    execution = runner.execute(cell, tmp_path, driver=driver, differences=liar.differences)
    report = execution.report

    assert report.codes == liar.codes, f"liar {liar.id} on {cell.id}:\n{report.describe()}"
    assert set(liar.trips) <= report.tripped(), report.describe()
    # Each code is booked under the clause that reported it: a column-wide
    # entry is owed exactly where that clause was evaluated.
    for clause, outcome in report.outcomes.items():
        assert {contract.clause_of(item.code) for item in outcome.deviations} <= {clause}
    if execution.observation.raised is not None:
        # A raise stands for the result clauses: its code, and no outcome of
        # theirs. Where the selected variant expects a refusal, those
        # clauses are no part of the cell at all.
        if execution.plan.expect.outcome == "result":
            for clause in RESULT_CLAUSES:
                assert report.outcomes[clause].status == "stood_for"
        else:
            assert not set(RESULT_CLAUSES) & set(report.outcomes)
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


def test_a_rehearsed_gap_of_the_matrix_is_a_registry_record() -> None:
    # A gap that was known before the matrix was built (G.., origin
    # ``gap list``) may be rehearsed before any cell of it is recorded. A
    # gap the matrix found itself exists in the registry only, so
    # a liar that names one names a record, and yields what the ledger
    # attributes to that gap wherever an entry still does.
    own = [
        (liar, gap)
        for liar in toy.LIARS
        for gap in liar.rehearses
        if not gap.startswith("G")
    ]
    assert own, "no liar rehearses a gap of origin matrix"
    for liar, gap in own:
        record = LEDGER.registry.get(gap)
        assert record is not None and record.origin == "matrix", (liar.id, gap)
        for entry in LEDGER.entries:
            attributed = {code for code, owner in entry.codes.items() if owner == gap}
            assert not attributed or attributed == set(liar.codes), (liar.id, entry.cell)


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
    # ...nor where the result's own flags switched its check off (C6 asks for
    # pricing evidence only of a result that flags its cost complete), and it
    # is not tolerated there either.
    unasked = frozenset({"pricing:none"})
    ledger.compare(known, other, set(), current={}, evaluated=evaluated, unasked=unasked)
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation"):
        ledger.compare(
            known, other, {"pricing:none"}, current={}, evaluated=evaluated, unasked=unasked
        )
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


def test_each_shipped_entry_is_strict() -> None:
    # One test over whatever entries the ledger holds today, not one item per
    # entry: the fix that deletes the last entry would otherwise leave an empty
    # parameter set, which the package refuses at collection.
    for entry in LEDGER.entries:
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


# -- the contract on what no scenario plays honestly ----------------------------


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
        endpoint=None,
        pricing=matrix_columns.MATRIX_RATE,
        secrets=(matrix_columns.CANARY_CREDENTIAL,),
        tool_authority_digest=None,
        bindings={},
        write_slot="write",
    )


def test_refused_variant_accepts_only_its_refusal_before_transport() -> None:
    expect = Expectation(outcome="refused", refused_type="FeatureAdmissionError")
    adapter = "crucible.backends.an_adapter"

    def codes(observation: contract.Observation, built: str = adapter) -> frozenset[str]:
        report = contract.evaluate(
            dataclasses.replace(observation, backend_module=built),
            expect,
            _reference(),
            lens="raw",
            toolset=False,
        )
        assert not set(RESULT_CLAUSES) & set(report.outcomes)
        return report.codes

    refusal = Raised("FeatureAdmissionError", "before_request", adapter, left_invoke=True)
    other = Raised("ValueError", "before_request", adapter, left_invoke=True)
    late = Raised("FeatureAdmissionError", "after_request", adapter, left_invoke=True)
    result = InvocationResult("", (), Usage(), "end_turn")

    assert codes(_observation(raised=refusal)) == frozenset()
    assert codes(_observation(raised=other)) == {other.code}
    assert codes(_observation(raised=late)) == {late.code}
    assert codes(_observation(result=result)) == {"kind:none!=refused"}
    # The type alone is no refusal of the route. The agreed type that never
    # left invoke (the driver or routing raised it while the backend was
    # built), the agreed type that left invoke out of another module than
    # the one of the backend, and any raise of a cell that built no backend
    # keep their code: a cell passes on a refusal only where the backend it
    # built raised it out of the call.
    at_build = dataclasses.replace(refusal, left_invoke=False)
    elsewhere = dataclasses.replace(refusal, module="crucible.backends.config")
    assert at_build.code == refusal.code != elsewhere.code
    assert codes(_observation(raised=at_build)) == {at_build.code}
    assert codes(_observation(raised=elsewhere)) == {elsewhere.code}
    assert codes(_observation(raised=refusal), built="") == {refusal.code}
    assert codes(_observation(raised=refusal), built=elsewhere.module) == {refusal.code}
    assert not Raised("FeatureAdmissionError", "before_request", adapter).left_invoke
    # A result where a refusal is expected is what the scenario contract
    # says of the cell: no clause about a result is part of it, so the code
    # is booked under none of them.
    assert contract.clause_of("kind:none!=refused") == contract.SCENARIO
    assert contract.clause_of("kind:none!=transport") == "C2"


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
    reference = changes.pop("reference", execution.reference)
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
        reference,
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
    # The endpoint of the route, where the route names one: it is what tells
    # two routes of one adapter apart. The reference is the provider the toy
    # was built for, not what its trace says.
    addressed = execution.observation.provider.requests[0].host
    assert execution.reference.endpoint == f"http://{addressed}" == trace.endpoint
    assert changed(endpoint="http://127.0.0.1:9/elsewhere") == {"digest.mismatch:endpoint"}
    assert changed(endpoint=None) == {"digest.missing:endpoint"}
    unnamed = dataclasses.replace(execution.reference, endpoint=None)
    for recorded in (None, "http://127.0.0.1:9/elsewhere"):
        assert _judge(
            execution,
            reference=unnamed,
            execution_trace=dataclasses.replace(trace, endpoint=recorded),
        ) == frozenset()
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


def test_c6_holds_the_cost_interval_of_a_route_that_opts_into_a_cost_bound(
    tmp_path: Path,
) -> None:
    execution = _honest(tmp_path, "S03")
    result, _trace = _result_and_trace(execution)
    usage, flags = result.usage, result.completeness
    billed = execution.observation.provider.billed
    rate = execution.reference.pricing
    assert len(billed) == 3 and not execution.reference.cost_bounded
    assert execution.reference.bound_pricing is None
    # A toy column has no shipped price row, so the reference of this test
    # carries rates of its own for the bounds: those of the rate card, and a
    # cache read at a tenth of the input rate, as a route with such a policy
    # has it.
    reads_per_mtok = rate.input_per_mtok / 10
    rates = ModelPricing(
        input_per_mtok=rate.input_per_mtok,
        output_per_mtok=rate.output_per_mtok,
        cache_write_per_mtok=rate.input_per_mtok,
        cache_read_per_mtok=reads_per_mtok,
    )
    bounded = dataclasses.replace(execution.reference, cost_bounded=True, bound_pricing=rates)
    # What a result of such a route flags for a reply without a count of
    # cache reads: token usage and cost incomplete, on purpose.
    untold = {"flag:token_usage_complete=False", "flag:cost_complete=False"}
    unaccounted = dataclasses.replace(flags, token_usage_complete=False, cost_complete=False)
    mispriced, miscounted = "usage.cost_bounds:amounts", "usage.cost_bounds:counts"
    # The toy wire states no cache count and its dialect reads none back, so
    # the ledger of the play holds none. What the replies of such a route
    # say is set here: nothing written to the cache, and no count of reads.
    ledger_of_the_play = execution.observation.provider
    assert ledger_of_the_play.cache_counts is None
    unread = CacheCounts(read_tokens=None, write_tokens=0)

    def replies(*counts: CacheCounts) -> ProviderLedger:
        return dataclasses.replace(ledger_of_the_play, cache_counts=list(counts))

    def price(tokens: Any, read: int, write: int = 0) -> float:
        # The bound formula, written out here and not taken from the contract.
        # A cache write is priced as input at the rates of this test.
        unread_tokens = tokens.input_tokens - read
        assert write <= unread_tokens
        return (
            unread_tokens * rate.input_per_mtok
            + read * reads_per_mtok
            + tokens.output_tokens * rate.output_per_mtok
        ) / 1_000_000

    def segment(
        tokens: Any, *, read: int | None = None, write: int = 0
    ) -> UsageCostBoundSegment:
        # Unknown count: every input token that was not written may have
        # been read, or none.
        return UsageCostBoundSegment(
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            cache_creation_tokens=write,
            cache_read_tokens=read,
            lower_bound_usd=price(
                tokens, tokens.input_tokens - write if read is None else read, write
            ),
            upper_bound_usd=price(tokens, 0 if read is None else read, write),
            policy="a-cost-policy",
        )

    def interval(*segments: UsageCostBoundSegment, coverage: bool = True) -> UsageCostBounds:
        return UsageCostBounds(tuple(segments), coverage)

    def judged(
        bounds: UsageCostBounds | None,
        *,
        reference: Any = bounded,
        stated: ProviderLedger | None = None,
        **fields: Any,
    ) -> frozenset[str]:
        cost = usage.cost_usd if bounds is None else bounds.lower_bound_usd
        counted = {} if bounds is None else {
            "cache_read_tokens": sum(item.cache_read_tokens or 0 for item in bounds.segments),
            "cache_creation_tokens": sum(item.cache_creation_tokens for item in bounds.segments),
        }
        reported = dataclasses.replace(
            usage, **{"cost_usd": cost, "cost_bounds": bounds, **counted, **fields}
        )
        return _judge(
            execution,
            reference=reference,
            provider=replies(*[unread] * len(billed)) if stated is None else stated,
            usage=reported,
            completeness=unaccounted,
            pricing_evidence=None,
        )

    whole = interval(*(segment(item) for item in billed))
    # The interval of an honest result: one segment per billed response with
    # its tokens, each priced by the formula at the rates of the reference,
    # the reported cost its lower bound. Only the flags are named.
    assert judged(whole) == untold
    assert whole.lower_bound_usd < whole.upper_bound_usd
    for item in whole.segments:
        assert (item.lower_bound_usd, item.upper_bound_usd) == pytest.approx(
            contract.bound_amounts(rates, item), rel=1e-12
        )
    # What a segment costs is held. Every lower bound and the reported cost
    # at nothing; every upper bound, which is what a budget is charged,
    # lowered to the lower one; one bound of one segment a little off; a
    # segment whose count is known and whose bounds are those of an unknown
    # one. Each leaves coverage, tokens and segments in order.
    free = interval(
        *(dataclasses.replace(item, lower_bound_usd=0.0) for item in whole.segments)
    )
    narrow = interval(
        *(
            dataclasses.replace(item, upper_bound_usd=item.lower_bound_usd)
            for item in whole.segments
        )
    )
    first, *others = whole.segments

    def nudged(factor: float) -> UsageCostBounds:
        moved_bound = dataclasses.replace(first, upper_bound_usd=first.upper_bound_usd * factor)
        return interval(moved_bound, *others)

    for wrong in (free, narrow, nudged(1 + 1e-6)):
        assert judged(wrong) == untold | {mispriced}
    # The segment with a known count: its reply carried a count of zero, and
    # its bounds are those of an unknown one. Where no reply carried the
    # count, the count is named as well.
    counted = interval(dataclasses.replace(first, cache_read_tokens=0), *others)
    first_counted = replies(CacheCounts(0, 0), unread, unread)
    assert judged(counted, stated=first_counted) == untold | {mispriced}
    assert judged(counted) == untold | {mispriced, miscounted}
    assert contract.clause_of(mispriced) == contract.clause_of(miscounted) == "C6"
    # Within the tolerance of the product's own replay (a millionth of a
    # millionth of a dollar, or of the amount) nothing is named.
    assert judged(nudged(1 + 1e-10)) == untold
    # The reference decides, not the interval: at another price row the same
    # interval is mispriced (the rate card has no cheaper cache read, so its
    # lower bound is its upper one).
    at_rate_card = dataclasses.replace(bounded, bound_pricing=rate)
    assert judged(whole, reference=at_rate_card) == untold | {mispriced}
    # A cost-bounded reference without a price row cannot judge an interval:
    # the runner refuses such a route before the cell is played.
    with pytest.raises(AssertionError, match="carries no price row"):
        judged(whole, reference=dataclasses.replace(bounded, bound_pricing=None))
    # The cache counts a segment records are held to what the reply of its
    # response carried, because the bounds follow from them. Each interval
    # below is priced by the formula for exactly the counts it states, so
    # the rule on the amounts names nothing; no reply carried a count of
    # reads or wrote to the cache. Every input token said to be read, which
    # lowers what a budget is charged to the honest lower bound; one read in
    # one response; a missing count read as zero, which at these rates is
    # the rate-card price and so shows nowhere else; one token said to be
    # written.
    all_read = interval(*(segment(item, read=item.input_tokens) for item in billed))
    assert all_read.upper_bound_usd == pytest.approx(whole.lower_bound_usd, rel=1e-12)
    one_read = interval(segment(billed[0], read=1), *others)
    none_read = interval(*(segment(item, read=0) for item in billed))
    assert math.isclose(none_read.lower_bound_usd, usage.cost_usd, rel_tol=1e-12)
    written = interval(*(segment(item, write=1) for item in billed))
    for told in (all_read, one_read, none_read, written):
        for item in told.segments:
            assert (item.lower_bound_usd, item.upper_bound_usd) == pytest.approx(
                contract.bound_amounts(rates, item), rel=1e-12
            )
        assert judged(told) == untold | {miscounted}
    # The other way round: a count a reply carried and the segment calls
    # unknown, a reply that carried no count of writes, one response of
    # three. What the replies said is the provider ledger's, not the result's.
    assert judged(whole, stated=replies(*[CacheCounts(0, 0)] * 3)) == untold | {miscounted}
    assert judged(whole, stated=replies(*[CacheCounts(None, None)] * 3)) == untold | {
        miscounted
    }
    assert judged(whole, stated=replies(unread, CacheCounts(3, 0), unread)) == untold | {
        miscounted
    }
    assert judged(one_read, stated=replies(CacheCounts(1, 0), unread, unread)) == untold
    # The counts are compared response by response, so only where the
    # interval has a segment for every response: an interval that lost its
    # first response is named for that, and its segments are not held to the
    # replies of other responses.
    first_read = replies(CacheCounts(1, 0), unread, unread)
    assert judged(interval(*others), stated=first_read) == untold | {
        "usage.cost_bounds:uncovered",
        "usage.cost_bounds:segments",
    }
    # A provider side that recorded no counts, or not one statement for
    # every billed response, cannot judge an interval: an error of the
    # matrix, no deviation and no pass.
    with pytest.raises(AssertionError, match="recorded no cache counts"):
        judged(whole, stated=ledger_of_the_play)
    with pytest.raises(AssertionError, match="for 2 of 3 billed"):
        judged(whole, stated=replies(unread, unread))
    # Nothing is reported at all. Without the opt-in the cleared flags excuse
    # it as a lower bound; with it the tokens and the interval are owed.
    silent: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    short = {"usage.input_tokens:lt", "usage.output_tokens:lt"}
    assert judged(None, reference=execution.reference, **silent) == untold
    assert judged(None, **silent) == untold | short | {"usage.cost_bounds:none"}
    assert judged(None) == untold | {"usage.cost_bounds:none"}
    # The last response alone: an interval that covers what is reported, and
    # that is not what the provider billed.
    last = billed[-1]
    assert judged(
        interval(segment(last)),
        input_tokens=last.input_tokens,
        output_tokens=last.output_tokens,
    ) == untold | short | {"usage.cost_bounds:segments"}
    # A response without its segment, and segments that are not the billed
    # responses although they add up to them.
    assert judged(interval(*(segment(item) for item in billed[1:]))) == untold | {
        "usage.cost_bounds:uncovered",
        "usage.cost_bounds:segments",
    }
    moved = [
        dataclasses.replace(billed[0], input_tokens=billed[0].input_tokens - 1),
        dataclasses.replace(billed[1], input_tokens=billed[1].input_tokens + 1),
        billed[2],
    ]
    assert judged(interval(*(segment(item) for item in moved))) == untold | {
        "usage.cost_bounds:segments"
    }
    # Coverage that is not claimed, and a cost that is not the lower bound.
    partial = interval(*(segment(item) for item in billed), coverage=False)
    assert judged(partial) == untold | {"usage.cost_bounds:uncovered"}
    assert judged(whole, cost_usd=whole.upper_bound_usd) == untold | {
        "usage.cost_bounds:uncovered"
    }
    # With every count of cache reads known the interval is exact and the
    # result may flag its accounting complete: nothing is named, where the
    # replies carried those counts. Where they carried none, a result that
    # flags its accounting complete on a count of its own names the count.
    exact = interval(*(segment(item, read=0) for item in billed))
    assert math.isclose(exact.lower_bound_usd, usage.cost_usd, rel_tol=1e-12)
    complete = dataclasses.replace(usage, cost_bounds=exact)
    assert _judge(
        execution,
        reference=bounded,
        provider=replies(*[CacheCounts(0, 0)] * 3),
        usage=complete,
    ) == frozenset()
    assert _judge(
        execution, reference=bounded, provider=replies(*[unread] * 3), usage=complete
    ) == {miscounted}
    assert _judge(execution, reference=bounded) == {"usage.cost_bounds:none"}
    # A scenario that does not owe the accounting flags asks for no interval.
    unowed = dataclasses.replace(execution.plan.expect, flags=())
    assert _judge(
        execution,
        reference=bounded,
        expect=unowed,
        usage=dataclasses.replace(usage, **silent),
        completeness=unaccounted,
        pricing_evidence=None,
    ) == frozenset()


def test_pricing_evidence_is_asked_only_of_a_result_that_flags_its_cost_complete(
    tmp_path: Path,
) -> None:
    execution = _honest(tmp_path, "S01")
    result, _trace = _result_and_trace(execution)

    def judged(**changes: Any) -> Report:
        changed = dataclasses.replace(result, **changes)
        return contract.evaluate(
            dataclasses.replace(execution.observation, result=changed),
            execution.plan.expect,
            execution.reference,
            lens="raw",
            toolset=False,
            checks=execution.plan.checks,
        )

    assert execution.report.unasked == frozenset()
    # Flagged complete without evidence: the check is asked, and it reports.
    bare = judged(pricing_evidence=None)
    assert (bare.codes, bare.unasked) == ({"pricing:none"}, frozenset())
    # Flagged incomplete: a lower bound is owed and no evidence, so a
    # column-wide "pricing:none" is not owed on such a cell.
    untold = dataclasses.replace(
        result.completeness, token_usage_complete=False, cost_complete=False
    )
    flagged = judged(completeness=untold, pricing_evidence=None)
    assert flagged.codes == {"flag:token_usage_complete=False", "flag:cost_complete=False"}
    assert flagged.unasked == contract.PRICING_CODES == {"pricing:none", "pricing:unreconciled"}
    # A raise leaves no result: nothing is switched off, the clause was not run.
    raised = contract.evaluate(
        dataclasses.replace(
            execution.observation,
            result=None,
            raised=Raised("RuntimeError", "after_request", "crucible.backends.standin"),
        ),
        execution.plan.expect,
        execution.reference,
        lens="raw",
        toolset=False,
    )
    assert raised.unasked == frozenset() and "C6" not in raised.evaluated()


def test_a_flag_the_scenario_waives_is_not_asked_of_the_result(tmp_path: Path) -> None:
    """A column-wide flag code is owed only where the scenario owes the flag.

    A refusal the provider never billed carries no usage, so its scenario
    waives the accounting flags and the terminal flag: the clauses run, and
    the checks that would report a false flag are not asked. A kind whose
    every result flags its accounting incomplete owes that column-wide code
    wherever the flag is owed, and not on such a cell.
    """

    waived = "flag:token_usage_complete=False"
    kind = cells.toy_columns()[0].kind
    known = dataclasses.replace(
        LEDGER,
        registry={
            **LEDGER.registry,
            "G99": ledger.GapRecord(
                "G99", "synthetic", "matrix", "low", column_wide=((kind, waived),)
            ),
        },
        column_wide=(ledger.ColumnWide(kind, waived, "G99"),),
    )

    def verdict(execution: runner.Execution, report: Report) -> None:
        ledger.compare(
            known,
            execution.cell.id,
            report.codes,
            current=execution.current,
            evaluated=report.evaluated(),
            unasked=report.unasked,
        )

    def incomplete(execution: runner.Execution) -> Report:
        """The same cell, with a result that flags its accounting incomplete."""

        result, _trace = _result_and_trace(execution)
        untold = dataclasses.replace(
            result.completeness, token_usage_complete=False, cost_complete=False
        )
        return contract.evaluate(
            dataclasses.replace(
                execution.observation,
                result=dataclasses.replace(
                    result, completeness=untold, pricing_evidence=None
                ),
            ),
            execution.plan.expect,
            execution.reference,
            lens="raw",
            toolset=False,
            checks=execution.plan.checks,
        )

    # The unbilled refusal: its scenario owes none of the three flags.
    unbilled = _honest(tmp_path / "unbilled", "S23c")
    assert not set(unbilled.plan.expect.flags) & {"token_usage_complete", "cost_complete"}
    assert "C6" in unbilled.report.evaluated()
    for report in (unbilled.report, incomplete(unbilled)):
        assert report.codes == frozenset()
        assert {waived, "flag:cost_complete=False", "flag:terminal_complete=False"} <= (
            report.unasked
        )
        verdict(unbilled, report)  # the column-wide code is not owed: the cell passes
    # The trace flag is waived by no scenario: its check is always asked.
    assert "flag:trace_complete=False" not in unbilled.report.unasked

    # The lying counterpart: a billed refusal owes the accounting flags. The
    # code is owed there, and an honest result is "gap fixed"; a result that
    # flags its accounting incomplete is the expected failure.
    billed = _honest(tmp_path / "billed", "S23a")
    assert waived not in billed.report.unasked
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed: column-wide"):
        verdict(billed, billed.report)
    untold = incomplete(billed)
    assert waived in untold.codes and waived not in untold.unasked
    with pytest.raises(XFAILED, match="G99"):
        # The one ledgered code of the two the result shows.
        ledger.compare(
            known,
            billed.cell.id,
            {waived},
            current=billed.current,
            evaluated=untold.evaluated(),
            unasked=untold.unasked,
        )


def test_a_flag_the_scenario_expects_false_deviates_when_the_result_claims_it(
    tmp_path: Path,
) -> None:
    """A turn that never reached its terminal: what its result cannot claim.

    The scenario of a reply that is cut inside the turn expects the token,
    the cost and the terminal flag false. A result that flags one of them
    complete deviates with the code of that flag and no other: on such a
    cell clause C6 judges its second arm whatever the result flags, a lower
    bound that is never above what was billed, and asks no pricing evidence.
    """

    dropped = _honest(tmp_path / "dropped", "S27")
    result, _trace = _result_and_trace(dropped)
    flags, expect = result.completeness, dropped.plan.expect
    false = contract.EXPECTED_FALSE_FLAGS

    assert set(expect.flags_false) == set(false) and expect.flags == ("trace_complete",)
    assert [getattr(flags, name) for name in contract.FLAGS] == [False, False, False, True]
    # One code for each flag a result claims, and nothing else.
    for name in false:
        claimed = dataclasses.replace(flags, **{name: True})
        assert _judge(dropped, completeness=claimed) == {f"flag:{name}=True"}
    every = dataclasses.replace(flags, **{name: True for name in false})
    assert _judge(dropped, completeness=every) == {f"flag:{name}=True" for name in false}
    # The trace flag is owed as on every cell.
    untraced = dataclasses.replace(flags, trace_complete=False)
    assert "flag:trace_complete=False" in _judge(
        dropped, completeness=untraced, execution_trace=None, checks=()
    )
    # The second arm of C6. Where the provider billed something, a result
    # that reports less is a lower bound whatever it flags; one that reports
    # more than was billed is above the bill whatever it flags.
    accounting = dataclasses.replace(flags, token_usage_complete=True, cost_complete=True)
    claims = {"flag:token_usage_complete=True", "flag:cost_complete=True"}
    billed = dataclasses.replace(dropped.observation.provider, billed=[TokenUsage(120, 30)])
    assert _judge(dropped, provider=billed) == frozenset()
    assert _judge(dropped, provider=billed, completeness=accounting) == claims
    more = dataclasses.replace(result.usage, input_tokens=1)
    assert _judge(dropped, usage=more) == {"usage.input_tokens:gt"}
    assert _judge(dropped, usage=more, completeness=accounting) == claims | {
        "usage.input_tokens:gt"
    }
    costlier = dataclasses.replace(result.usage, cost_usd=0.5)
    assert _judge(dropped, usage=costlier) == {"usage.cost_usd:gt"}
    # No pricing evidence is asked of a claim the scenario excludes ...
    costed = dataclasses.replace(flags, cost_complete=True)
    assert _judge(dropped, completeness=costed, pricing_evidence=None) == {
        "flag:cost_complete=True"
    }
    # ... where a scenario that only waives the flags judges the claim in the
    # first arm of the clause: evidence, and the numbers exactly.
    waiving = dataclasses.replace(expect, flags_false=("terminal_complete",))
    assert _judge(
        dropped, expect=waiving, completeness=costed, pricing_evidence=None
    ) == {"pricing:none"}
    counted = dataclasses.replace(flags, token_usage_complete=True)
    assert _judge(dropped, expect=waiving, provider=billed, completeness=counted) == {
        "usage.input_tokens:lt",
        "usage.output_tokens:lt",
    }
    assert _judge(dropped, expect=waiving, completeness=every) == {
        "flag:terminal_complete=True",
        "pricing:unreconciled",
    }

    # The form. Only a flag that a scenario may waive can be expected false,
    # only an accounting flag can go by what the provider billed, and a flag
    # is in one list: owed, expected false, or expected false where nothing
    # was billed.
    for wrong in (
        {"flags_false": ("trace_complete",)},
        {"flags": ("terminal_complete",), "flags_false": ("terminal_complete",)},
        {"flags": ("usage_complete",)},
        {"flags": (), "flags_false_if_unbilled": ("terminal_complete",)},
        {"flags_false_if_unbilled": ("cost_complete",)},
        {
            "flags": (),
            "flags_false": ("cost_complete",),
            "flags_false_if_unbilled": ("cost_complete",),
        },
    ):
        with pytest.raises(ValueError, match="an expectation names the flags"):
            Expectation(**wrong)  # type: ignore[arg-type]
    assert Expectation().flags_false == () == Expectation().flags_false_if_unbilled
    # No scenario of the starter set or of pack A expects a flag false in
    # either way: the form changes what none of their cells asserts.
    for scenario in SCENARIOS.values():
        if scenario.set in (STARTER, PACK_A):
            expected = scenario.plan().expect
            assert expected.flags_false == () == expected.flags_false_if_unbilled, scenario.id

    # What is not asked on such a cell: the checks that would report the
    # three flags false, and pricing evidence whatever the result flags. So a
    # column-wide accounting code of a kind is not owed there, and a result
    # that claims its cost is one unexpected code, never "gap fixed".
    unasked = {f"flag:{name}=False" for name in false} | contract.PRICING_CODES
    assert dropped.report.unasked == unasked
    kind = dropped.cell.column.kind
    wide = ("flag:token_usage_complete=False", "flag:cost_complete=False", "pricing:none")
    known = dataclasses.replace(
        LEDGER,
        registry={
            **LEDGER.registry,
            "G99": ledger.GapRecord(
                "G99", "synthetic", "matrix", "low", column_wide=tuple((kind, c) for c in wide)
            ),
        },
        column_wide=tuple(ledger.ColumnWide(kind, code, "G99") for code in wide),
    )

    def verdict(report: Report) -> None:
        ledger.compare(
            known,
            dropped.cell.id,
            report.codes,
            current=dropped.current,
            evaluated=report.evaluated(),
            unasked=report.unasked,
        )

    verdict(dropped.report)  # nothing is owed: the cell passes
    claimed = contract.evaluate(
        dataclasses.replace(
            dropped.observation,
            result=dataclasses.replace(result, completeness=costed, pricing_evidence=None),
        ),
        expect,
        dropped.reference,
        lens="raw",
        toolset=False,
        checks=dropped.plan.checks,
    )
    assert claimed.unasked == unasked
    with pytest.raises(ledger.LedgerMismatch, match="unexpected deviation") as refused:
        verdict(claimed)
    assert "flag:cost_complete=True" in str(refused.value).splitlines()[0]
    # The same kind owes the same codes on a cell whose scenario owes the flags.
    owing = _honest(tmp_path / "owing", "S01")
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed: column-wide"):
        ledger.compare(
            known,
            owing.cell.id,
            owing.report.codes,
            current=owing.current,
            evaluated=owing.report.evaluated(),
            unasked=owing.report.unasked,
        )

    # A reply that is no terminal of its wire: the terminal flag is expected
    # false, and the two accounting flags go by what the provider billed.
    unended = _honest(tmp_path / "unended", "S26")
    plain = unended.plan.expect
    assert plain.flags_false == ("terminal_complete",) and plain.flags == ("trace_complete",)
    assert plain.flags_false_if_unbilled == contract.ACCOUNTING_FLAGS
    told = unended.observation.result
    assert told is not None and not told.completeness.terminal_complete
    ended = dataclasses.replace(told.completeness, terminal_complete=True)
    assert _judge(unended, completeness=ended) == {"flag:terminal_complete=True"}
    # Nothing was billed for the reply of this wire: its usage would have
    # travelled with the terminal that never came. A result that flags a
    # part of its accounting complete then deviates with the code of that
    # flag and no other, as on a reply that is cut, and no evidence is asked.
    assert unended.observation.provider.billed == [] and unended.report.unasked == unasked
    assert _judge(unended, completeness=counted) == {"flag:token_usage_complete=True"}
    assert _judge(unended, completeness=costed, pricing_evidence=None) == {
        "flag:cost_complete=True"
    }
    # The claim that clause C6 alone has nothing to hold against: zero
    # tokens and zero cost are what was billed, and complete evidence of no
    # response reconciles them. An expectation that only waives the two
    # flags lets it pass; this one does not.
    nothing = UsagePricingEvidence((), complete=True)
    assert nothing.reconciles(told.usage)
    assert _judge(unended, completeness=accounting, pricing_evidence=nothing) == claims
    only_waived = dataclasses.replace(plain, flags_false_if_unbilled=())
    assert (
        _judge(unended, expect=only_waived, completeness=accounting, pricing_evidence=nothing)
        == frozenset()
    )
    # Where the frames that travelled stated the usage, the provider billed
    # it: the flags are not asked and the clause judges both arms. A result
    # that reports what was billed and claims it, with evidence, passes; so
    # does one that reports less and flags it incomplete; a claim over less
    # than was billed is short, and a claim of the cost without evidence is
    # asked for it. Only the provider ledger differs between the two halves.
    paid = owing.observation.result
    assert paid is not None and paid.completeness.cost_complete
    stated = dataclasses.replace(
        unended.observation.provider, billed=owing.observation.provider.billed
    )
    assert stated.billed == [TokenUsage(120, 30)]
    assert _judge(unended, provider=stated) == frozenset()
    assert (
        _judge(
            unended,
            provider=stated,
            completeness=accounting,
            usage=paid.usage,
            pricing_evidence=paid.pricing_evidence,
        )
        == frozenset()
    )
    assert _judge(unended, provider=stated, completeness=counted) == {
        "usage.input_tokens:lt",
        "usage.output_tokens:lt",
    }
    assert _judge(
        unended, provider=stated, completeness=accounting, usage=paid.usage, pricing_evidence=None
    ) == {"pricing:none"}
    # What is not asked follows the bill as well: with nothing billed no
    # evidence is asked whatever the result flags, and with a bill it is
    # asked of a result that flags its cost complete.
    for bill, cost_claimed, pricing_asked in (
        ([], False, False),
        ([], True, False),
        (stated.billed, False, False),
        (stated.billed, True, True),
    ):
        report = contract.evaluate(
            dataclasses.replace(
                unended.observation,
                provider=dataclasses.replace(unended.observation.provider, billed=bill),
                result=dataclasses.replace(
                    told, completeness=accounting if cost_claimed else told.completeness
                ),
            ),
            plain,
            unended.reference,
            lens="raw",
            toolset=False,
            checks=unended.plan.checks,
        )
        assert (contract.PRICING_CODES <= report.unasked) is not pricing_asked
        assert {f"flag:{name}=False" for name in false} <= report.unasked


def test_the_text_of_a_reply_that_was_cut_off_is_no_final_message(tmp_path: Path) -> None:
    dropped = _honest(tmp_path, "S27")
    result, trace = _result_and_trace(dropped)
    (step,) = dropped.plan.script.steps

    assert isinstance(step, Drop) and result.final_message == ""
    assert result.error is not None and str(result.error.kind) == "transport"
    # Nothing of the reply was billed and no document named a model.
    provider = dropped.observation.provider
    assert (provider.billed, provider.served_models, provider.request_count) == ([], [], 1)
    # A leading part of the scripted answer in the final message is the text
    # that had arrived, whatever the kind of the result; other words are not.
    arrived = step.text[: len(step.text) // 2]
    assert _judge(dropped, final_message=arrived) == {"text:mismatch"}
    assert _judge(dropped, final_message=f"the provider said: {step.text}") == {"text:mismatch"}
    assert _judge(dropped, final_message="the connection went away") == frozenset()
    # A stop reason that a workflow reads as an outcome of its own is refused.
    for stop in contract.READ_STOP_REASONS:
        named = dataclasses.replace(
            trace,
            events=(*trace.events[:-1], dataclasses.replace(trace.events[-1], name=stop)),
        )
        assert _judge(dropped, stop_reason=stop, execution_trace=named) == {
            f"stop:{stop}!=other"
        }


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
    # An endpoint that carries a credential in its query is a leak, and it is
    # not the endpoint the route names either: two statements, both reported.
    in_trace = dataclasses.replace(trace, endpoint=f"{trace.endpoint}?key={canary}")
    assert _judge(execution, execution_trace=in_trace) == leaked | {"digest.mismatch:endpoint"}
    # The reference carries no user-info part and no query: a trace must not.
    carrying = f"https://user:{canary}@host.example:8443/v1?key=x#f"
    assert declarations.endpoint_identity(carrying) == "https://host.example:8443/v1"
    with pytest.raises(MatrixSetupError, match="not an absolute URL"):
        declarations.endpoint_identity("host.example/v1")
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

    # Rankable is read from the terminal the guard left, never inferred from
    # the absence of a rewrite: under a cost policy the guard lets a result
    # stand that it does not call rankable.
    terminal = guarded.observation.limit_terminal
    assert terminal is not None and terminal.rankable
    unranked = dataclasses.replace(terminal, rankable=False)
    assert _judge(guarded, limit_terminal=unranked) == {"guard:not_rankable"}
    assert contract.clause_of(contract.NOT_RANKABLE) == "C5"
    assert _judge(
        guarded, limit_terminal=unranked, second_call=SecondCall(refused=True)
    ) == {"guard:not_rankable", "guard:revoked"}
    # An error result is not rankable by its kind, which C2 judges: C5 does
    # not say it a second time.
    failed = TransportError(message="refused", retriable=False)
    failing = dataclasses.replace(guarded.plan.expect, kind="transport", retriable=False)
    assert _judge(guarded, limit_terminal=unranked, error=failed, expect=failing) == frozenset()
    assert _judge(guarded, limit_terminal=unranked, error=failed) == {"kind:transport!=none"}

    # Where the route opts into a cost upper bound, C5 does not ask
    # "rankable": the guard never calls a result of such a route rankable,
    # and the result is held to the marks of a diagnostic one instead (a
    # declared difference of the ledger). The selector is the reference's,
    # read from the routing file. The terminal of the plain limit policy,
    # which the toy's is, has none of the marks: it calls the result
    # rankable, it is no terminal of a cost policy and it holds no cost
    # authority. (The toy returns no interval either, which is C6's code.)
    bounded = dataclasses.replace(
        guarded.reference, cost_bounded=True, cost_policy="a-cost-policy"
    )
    no_interval = {"usage.cost_bounds:none"}
    assert _judge(guarded, reference=bounded) == no_interval | {
        contract.COST_BOUND_RANKABLE,
        contract.COST_BOUND_TERMINAL,
        contract.COST_BOUND_AUTHORITY,
    }
    # One that is not rankable for another reason is no diagnostic result.
    assert _judge(guarded, reference=bounded, limit_terminal=unranked) == no_interval | {
        contract.COST_BOUND_TERMINAL,
        contract.COST_BOUND_AUTHORITY,
    }
    # Nor is one that needed a second physical attempt.
    retried = dataclasses.replace(unranked, physical_attempts=2)
    assert _judge(guarded, reference=bounded, limit_terminal=retried) == no_interval | {
        contract.COST_BOUND_TERMINAL,
        contract.COST_BOUND_ATTEMPTS,
        contract.COST_BOUND_AUTHORITY,
    }
    assert contract.COST_BOUND_MARKS == {
        contract.COST_BOUND_TERMINAL,
        contract.COST_BOUND_ATTEMPTS,
        contract.COST_BOUND_AUTHORITY,
        contract.COST_BOUND_RANKABLE,
    }
    assert {contract.clause_of(code) for code in contract.COST_BOUND_MARKS} == {"C5"}
    # What C5 asks of every route it asks there too: a second call answered.
    assert "guard:revoked" in _judge(
        guarded, reference=bounded, second_call=SecondCall(refused=True)
    )
    # An error result is asked for no mark, as it is asked for no "rankable".
    assert _judge(
        guarded, reference=bounded, limit_terminal=unranked, error=failed, expect=failing
    ) == no_interval
    # On a route without the opt-in nothing of this applies: "rankable" is
    # read as before, whatever policy the reference might name.
    named_only = dataclasses.replace(guarded.reference, cost_policy="a-cost-policy")
    assert _judge(guarded, reference=named_only) == frozenset()
    assert _judge(guarded, reference=named_only, limit_terminal=unranked) == {
        "guard:not_rankable"
    }

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


def test_a_cell_is_priced_at_the_model_of_the_route_of_its_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    column = cells.toy_columns()[0]
    roles = matrix_columns.ROLE_BY_CAPABILITY
    shipped = matrix_columns.real_columns()[0].routing_path(matrix_columns.COLUMN_PRODUCT)
    # A model whose built-in price has a long-context tier is priced in another
    # shape than the plain matrix rate, so the two references can be told apart.
    tiered = sorted(
        model
        for model, price in matrix_columns.DEFAULT_PRICING.items()
        if price.long_context_threshold_tokens is not None
    )[0]
    assert matrix_columns.matrix_rate(tiered) != matrix_columns.matrix_rate(column.model)
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
                "roles": {roles["read_only"]: {"model": tiered}},
            }
        ),
        encoding="utf-8",
    )
    routed = dataclasses.replace(column, routing=((matrix_columns.COLUMN_PRODUCT, route),))

    def priced(scenario: str) -> Any:
        return runner.execute(Cell(routed, scenario, "raw"), tmp_path / scenario).reference.pricing

    assert priced("S01") == matrix_columns.matrix_rate(column.model)
    assert priced("S03") == matrix_columns.matrix_rate(tiered)
    # The rate card the adapters price from holds every model a route names,
    # not only the default route of each lane.
    card = matrix_columns.rate_card()
    assert all(
        declarations.routed_model(real, role) in card
        for real in matrix_columns.real_columns()
        for role in declarations.ROUTED_ROLES
    )
    lane = SimpleNamespace(
        default_route=SimpleNamespace(model="default-model"),
        role_routes=(SimpleNamespace(model="planner-model"),),
    )
    monkeypatch.setattr(matrix_columns, "catalog_lanes", lambda _product: (lane,))
    assert set(matrix_columns.rate_card()) == {
        matrix_columns.TOY_MODEL,
        "default-model",
        "planner-model",
    }


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


class _NoReplayRule:
    """The toy wire, read by a dialect that states no replay rule."""

    name, path = ToyDialect.name, ToyDialect.path

    def __init__(self) -> None:
        self._toy = ToyDialect()

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return self._toy.parse(*arguments, **named)

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        return self._toy.seam_problems(seen)

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        return self._toy.render(step, seen, call_id)


def test_a_stateless_dialect_states_what_a_request_replays_and_tool_scenarios_report_it(
    tmp_path: Path,
) -> None:
    tools = {"tools": [BINDINGS[READ_A].name]}
    script = ProviderScript((Call(READ_A), Call(READ_A), Say("three")))

    def exchange(dialect: Any, *, forget: bool) -> ProviderLedger:
        def results(call: dict[str, Any]) -> dict[str, Any]:
            return {**tools, "tool_results": [{"id": call["id"], "text": "result"}]}

        with LoopbackProvider(script, credential="c", dialect=dialect) as provider:
            (first,) = _ask(provider, {**tools, "answered": []})[1]["calls"]
            (second,) = _ask(provider, {**results(first), "answered": []})[1]["calls"]
            # A request that matches no step is not judged: it is clause C0's.
            assert _ask(provider, {**tools, "answered": ["x"]})[0] == UNSCRIPTED_STATUS
            answered = [] if forget else [first["id"]]
            final = _ask(provider, {**results(second), "answered": answered})
            assert final[1]["text"] == "three"
            return provider.freeze()

    # The toy wire keeps no conversation: a request names the calls that
    # earlier requests answered. The rule is the dialect's; the provider
    # applies it when the exchange ends, and never picks a reply by it.
    assert isinstance(ToyDialect(), StatelessDialect)
    kept = exchange(ToyDialect(), forget=False)
    assert (kept.replay_problems, kept.extra, kept.unconsumed) == ([], 1, 0)
    forgotten = exchange(ToyDialect(), forget=True)
    (problem,) = forgotten.replay_problems
    assert problem.startswith("request 4 names [] as answered by earlier requests")
    assert (forgotten.extra, forgotten.unconsumed) == (1, 0)
    # A dialect that states no rule is asked nothing, whatever was sent.
    assert not isinstance(_NoReplayRule(), StatelessDialect)
    assert exchange(_NoReplayRule(), forget=True).replay_problems == []

    # The scenario assertion reports what the ledger holds, under one code.
    reference = _reference()
    assert history_replayed(_observation(), reference) == []
    named = _observation(provider=ProviderLedger(replay_problems=["one", "two"]))
    assert history_replayed(named, reference) == [
        Deviation("request.content:replay", "one; two")
    ]
    assert contract.clause_of("request.content:replay") == contract.SCENARIO
    # Every scenario with a ToolSet asserts it and no other does, whatever
    # variant a column selects and whatever declared difference is in force;
    # a variant that expects a refusal sends nothing and asserts nothing.
    for scenario in SCENARIOS.values():
        assert (history_replayed in scenario.plan().checks) == (scenario.toolset is not None)
    for cell in TOY_CELLS:
        scenario = SCENARIOS[cell.scenario]
        states = declarations.route_states(
            cell.column, matrix_columns.ROLE_BY_CAPABILITY[scenario.capability], scenario.governing
        )
        variants = declarations.select(
            scenario.differences, declarations.routed(cell.column), states
        )
        for differences in ((), tuple(scenario.alternatives)):
            plan = plan_for(scenario, variants, differences)
            asserted = scenario.toolset is not None and plan.expect.outcome == "result"
            assert (history_replayed in plan.checks) == asserted, cell.id
    # On an honest cell the rule ran and named nothing.
    honest = _honest(tmp_path, "S03")
    assert len(honest.observation.provider.requests) == 3
    assert honest.observation.provider.replay_problems == []
    assert [item.evidence["answered"] for item in honest.observation.provider.requests] == [
        [],
        [],
        [honest.observation.provider.sent_calls[0].call_id],
    ]


class _Stream(ToyDialect):
    """Answers with bytes of its own under a content type of its own."""

    raw = b"event: answer\ndata: {}\n\n"

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        return dataclasses.replace(
            super().render(step, seen, call_id),
            raw=self.raw,
            content_type="text/event-stream",
        )


def test_a_reply_travels_as_the_bytes_and_the_content_type_its_dialect_gives() -> None:
    def asked(provider: LoopbackProvider) -> tuple[str | None, bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=5)
        try:
            connection.request("POST", ToyDialect.path, body=b"{}")
            response = connection.getresponse()
            return response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    script = ProviderScript((Say("one"),))
    # One JSON document, unless the dialect renders the bytes itself.
    with LoopbackProvider(script, credential="c") as provider:
        kind, body = asked(provider)
    assert kind == "application/json" and json.loads(body)["text"] == "one"
    with LoopbackProvider(script, credential="c", dialect=_Stream()) as provider:
        kind, body = asked(provider)
        seen = provider.freeze()
    assert (kind, body) == ("text/event-stream", _Stream.raw)
    # What the ledger keeps is the reply's, whatever its bytes are.
    assert (seen.request_count, seen.unconsumed, seen.extra) == (1, 0, 0)
    assert seen.billed == [Say("one").usage]


class _CutsOutsideItsReply(ToyDialect):
    """Cuts a reply after its last byte: nothing of it would be missing."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        return dataclasses.replace(reply, cut=len(reply.raw or b""))


def test_a_reply_that_is_cut_ends_short_of_what_its_headers_announce() -> None:
    text = "an answer that is cut off"

    def asked(provider: LoopbackProvider) -> tuple[int, int, bytes, bool]:
        connection = http.client.HTTPConnection("127.0.0.1", provider.port, timeout=5)
        try:
            connection.request("POST", ToyDialect.path, body=b"{}")
            response = connection.getresponse()
            announced = int(response.getheader("Content-Length") or 0)
            try:
                return response.status, announced, response.read(), False
            except http.client.IncompleteRead as cut:
                return response.status, announced, bytes(cut.partial), True
        finally:
            connection.close()

    # The headers announce the whole reply, the body ends inside the answer
    # text, and the socket is closed: the client is told that the reply is
    # incomplete, and holds no document.
    with LoopbackProvider(ProviderScript((Drop(text),)), credential="c") as provider:
        status, announced, arrived, short = asked(provider)
        seen = provider.freeze()
    assert (status, short) == (200, True) and 0 < len(arrived) < announced
    assert arrived.endswith(text[: len(text) // 2].encode()) and text.encode() not in arrived
    with pytest.raises(json.JSONDecodeError):
        json.loads(arrived)
    # The step is consumed; nothing is billed for it and no model is named.
    assert (seen.request_count, seen.unconsumed, seen.extra) == (1, 0, 0)
    assert (seen.billed, seen.served_models) == ([], [])
    # A reply that is not cut arrives whole, as before.
    with LoopbackProvider(ProviderScript((Say(text),)), credential="c") as provider:
        status, announced, arrived, short = asked(provider)
    assert (status, short, len(arrived)) == (200, False, announced)
    # A reply without a turn: a whole document, no bill, the model named.
    with LoopbackProvider(ProviderScript((Malformed(text),)), credential="c") as provider:
        status, body = _ask(provider, {"model": "asked"})
        seen = provider.freeze()
    assert (status, body) == (200, {"model": "asked"})
    assert (seen.billed, seen.served_models, seen.unconsumed) == ([], ["asked"], 0)

    # Where a reply is cut: inside its answer text, or inside the frame that
    # follows the frame which carries the text.
    assert cut_inside(b'{"text": "abcd"}', "abcd") == 12
    frames = [b"first ", b"the text here ", b"then more"]
    assert cut_after(frames, "text") == len(frames[0]) + len(frames[1]) + len(frames[2]) // 2
    # The same cut as frames: the frame that carries the text is the last
    # that travels whole, and half of the next one follows it. A wire whose
    # frames are lines of a tool's output is cut by this.
    assert whole_and_partial(frames, "text") == (2, b"then")
    for wrong in (
        lambda: cut_inside(b"no such words", "text"),
        lambda: cut_inside(b"x", "x"),
        lambda: cut_after(frames, "missing"),
        lambda: cut_after(frames[:2], "text"),
        lambda: whole_and_partial(frames[:2], "text"),
    ):
        with pytest.raises(MatrixSetupError, match="can be cut"):
            wrong()
    # A dialect drops a reply inside its body only, and a cut that leaves the
    # whole reply is no cut: both are setup errors where the provider is built.
    with pytest.raises(MatrixSetupError, match="inside its body only"):
        LoopbackProvider(ProviderScript((Drop(text, after="headers"),)), credential="c")
    with pytest.raises(MatrixSetupError, match="cannot be cut after"):
        LoopbackProvider(
            ProviderScript((Drop(text),)), credential="c", dialect=_CutsOutsideItsReply()
        )


class _TwoFrames(ToyDialect):
    """Renders one reply from two envelopes, as a wire of several frames does."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        return dataclasses.replace(
            super().render(step, seen, call_id),
            envelopes=("openai_chat/gateway.completion", "openai_chat/local.completion"),
        )


def test_a_reply_of_several_frames_records_every_envelope_it_was_rendered_from() -> None:
    script = ProviderScript((Say("one"),))
    with LoopbackProvider(script, credential="c", dialect=_TwoFrames()) as provider:
        assert _ask(provider, {})[1]["text"] == "one"
        seen = provider.freeze()

    assert seen.envelopes == ["openai_chat/gateway.completion", "openai_chat/local.completion"]
    # The grade of a cell is the weakest of every envelope it used, not of
    # the first of each reply.
    assert matrix_envelopes.weakest(seen.envelopes[:1]) == "documented"
    assert matrix_envelopes.weakest(seen.envelopes) == "constructed"
    # A dialect that knows nothing about how a request was sent says nothing.
    assert seen.requests[0].evidence == {}


_FROM_THE_CACHE = "answered from the cache"


class _CountsTheCache(ToyDialect):
    """Reads back what each reply says of the provider's cache.

    The toy wire states no cache count. This one states a count of cache
    reads in the usage of one reply, as a provider does for a request that
    hit its cache, and says of every reply what its usage carried.
    """

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        if isinstance(step, Say) and step.text == _FROM_THE_CACHE:
            usage = {**reply.body["usage"], "cache_read_tokens": 7}
            reply = dataclasses.replace(reply, body={**reply.body, "usage": usage})
        return reply

    def cache_counts(self, reply: Reply) -> CacheCounts:
        usage = reply.body["usage"]
        return CacheCounts(usage.get("cache_read_tokens"), usage.get("cache_write_tokens"))


class _CannotReadItsReplies(_CountsTheCache):
    def cache_counts(self, reply: Reply) -> CacheCounts:
        raise MatrixSetupError("a reply whose cache counts cannot be read back")


def test_the_ledger_keeps_what_each_billed_reply_said_of_the_cache() -> None:
    script = ProviderScript((Say("one"), Refuse("http_policy"), Say(_FROM_THE_CACHE)))
    # A dialect that reads no counts back leaves the ledger without any: that
    # is not the statement that the replies carried none.
    assert not isinstance(ToyDialect(), CacheCountingDialect)
    with LoopbackProvider(script, credential="c") as provider:
        assert _ask(provider, {})[0] == 200
        assert provider.freeze().cache_counts is None
    assert ProviderLedger().cache_counts is None

    assert isinstance(_CountsTheCache(), CacheCountingDialect)
    with LoopbackProvider(script, credential="c", dialect=_CountsTheCache()) as provider:
        assert [_ask(provider, {})[0] for _ in script.steps] == [200, 400, 200]
        seen = provider.freeze()
        # The second call of the guarded lens is no part of the exchange.
        provider.begin_probe()
        assert _ask(provider, {})[0] == 200
        assert provider.freeze() is seen and provider.final_counts() == (1, 0)
    # One statement for every billed response, in the order billed, read
    # from the reply as it was sent: a count that is not there is missing,
    # not zero. The refusal by HTTP policy bills nothing and states nothing.
    assert seen.billed == [Say("one").usage, Say(_FROM_THE_CACHE).usage]
    assert seen.cache_counts == [CacheCounts(None, None), CacheCounts(7, None)]
    # A reply the dialect cannot read back is a setup error when the provider
    # is built, in the thread of the test.
    with pytest.raises(MatrixSetupError, match="cannot be read back"):
        LoopbackProvider(script, credential="c", dialect=_CannotReadItsReplies())


class _NoTurnField(ToyDialect):
    """The toy wire, read as a wire that has no field for a turn count."""

    turn_control_field = False

    def parse(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        credential: str,
    ) -> SeenRequest:
        seen = super().parse(method, path, headers, body, credential=credential)
        return dataclasses.replace(seen, turn_control=None)


def test_the_turn_count_is_owed_only_on_a_wire_that_has_a_field_for_it() -> None:
    script = ProviderScript((Say("one"),))
    reference = _reference()
    asked = suite_default_max_turns()

    def exchange(dialect: ToyDialect, body: dict[str, Any]) -> ProviderLedger:
        with LoopbackProvider(script, credential="c", dialect=dialect) as provider:
            assert _ask(provider, body)[0] == 200
            return provider.freeze()

    def controls(seen: ProviderLedger) -> list[int | None]:
        return [request.turn_control for request in seen.requests if request.matched]

    # The ledger keeps what the dialect says of its wire. On a wire with a
    # field for the count a request without it is named ...
    assert isinstance(ToyDialect(), TurnControlDialect)
    told, untold = exchange(ToyDialect(), {"max_turns": asked}), exchange(ToyDialect(), {})
    assert told.turn_control_field and untold.turn_control_field
    assert (controls(told), controls(untold)) == ([asked], [None])
    assert turn_count_in_request(_observation(provider=told), reference) == []
    (found,) = turn_count_in_request(_observation(provider=untold), reference)
    assert found.code == "request.content:turn_control"
    assert contract.clause_of(found.code) == contract.SCENARIO
    # ... and on a wire without one no adapter can send it, so the same
    # request says nothing. A ledger of no exchange claims no such field.
    silent = exchange(_NoTurnField(), {"max_turns": asked})
    assert not silent.turn_control_field and controls(silent) == [None]
    assert turn_count_in_request(_observation(provider=silent), reference) == []
    assert not ProviderLedger().turn_control_field
    # The scenario asks it under the two declared values that place the
    # enforcement of a limit where the provider is, and under no other.
    scenario = SCENARIOS["S11"]
    enforced = {"hard", "provider_stop"}
    assert {
        value
        for value, variant in scenario.differences["turn_limit"].items()
        if variant.kind == "FALLBACK"
    } == enforced
    declared = set()
    for cell in TOY_CELLS:
        if cell.scenario != scenario.id:
            continue
        routed = declarations.routed(cell.column)
        plan = plan_for(scenario, declarations.select(scenario.differences, routed))
        assert (turn_count_in_request in plan.checks) == (routed.turn_limit in enforced), cell.id
        declared.add(routed.turn_limit)
    assert enforced <= declared and declared - enforced


def test_an_envelope_key_slot_is_a_whole_key_filled_with_a_string() -> None:
    document = {
        "schema_version": matrix_envelopes.ENVELOPE_SCHEMA,
        "dialect": "standin",
        "provider": "standin",
        "behaviour": "usage",
        "provenance": {"grade": "constructed", "reason": "synthetic"},
        "status": 200,
        "body": {
            "per_model": {"${slot:served_model}": {"tokens": {"slot": "input_tokens"}}},
            "about ${slot:served_model}": "a key that only contains the notation is a key",
        },
    }
    envelope = matrix_envelopes.parse(document, where="x")
    filled = envelope.fill({"served_model": "the-model", "input_tokens": 7})

    assert envelope.slots() == {"served_model", "input_tokens"}
    assert filled == {
        "per_model": {"the-model": {"tokens": 7}},
        "about ${slot:served_model}": "a key that only contains the notation is a key",
    }
    # Filling copies: the envelope keeps its slots.
    assert list(envelope.body["per_model"]) == ["${slot:served_model}"]
    # A key is a string, every slot is supplied, and a key slot is held to
    # the closed slot list like any other.
    with pytest.raises(MatrixSetupError, match="fills a mapping key and must be a string"):
        envelope.fill({"served_model": 3, "input_tokens": 7})
    with pytest.raises(MatrixSetupError, match="has the slots"):
        envelope.fill({"input_tokens": 7})
    with pytest.raises(MatrixSetupError, match=r"slots \['price'\]"):
        matrix_envelopes.parse({**document, "body": {"${slot:price}": 1}}, where="x")


def test_a_routing_overlay_adds_the_closed_extras_of_its_kind_and_nothing_else() -> None:
    kind = matrix_columns.CLAUDE_KIND
    column = next(
        item
        for item in matrix_columns.real_columns()
        if (item.kind, item.role) == (kind, "primary")
    )
    shipped = load_routing(config_path=column.routing_path(matrix_columns.COLUMN_PRODUCT))
    (field,) = matrix_columns.ALLOWED_OVERLAY[kind]
    name = field.removeprefix("extra.")
    overlaid = matrix_columns.overlaid_routing(
        shipped, {field: "/a/peer"}, kind=kind, backend=column.backend
    )

    # Every route of the file gains the one extra and keeps everything else.
    assert shipped.roles and set(overlaid.roles) == set(shipped.roles)
    for role in (None, *shipped.roles):
        before = shipped.default if role is None else shipped.for_role(role)
        after = overlaid.default if role is None else overlaid.for_role(role)
        assert name not in before.extra and after.extra == {**before.extra, name: "/a/peer"}
        assert dataclasses.replace(after, extra=before.extra) == before
        assert (after.base_url, after.api_key) == (before.base_url, before.api_key)
    assert (overlaid.schema_version, overlaid.source_sha256) == (
        shipped.schema_version,
        shipped.source_sha256,
    )
    # Anything outside the closed list is refused: an endpoint, a credential,
    # a model, another extra; and a kind without an overlay may add nothing.
    for refused in ("base_url", "api_key", "model", "extra.credential_source"):
        with pytest.raises(MatrixSetupError, match="may overlay"):
            matrix_columns.overlaid_routing(
                shipped, {refused: "x"}, kind=kind, backend=column.backend
            )
    with pytest.raises(MatrixSetupError, match="may overlay"):
        matrix_columns.overlaid_routing(
            shipped,
            {field: "x"},
            kind=matrix_columns.HOSTED_GATEWAY_KIND,
            backend=column.backend,
        )
    # A route of another backend family is left alone.
    mixed = BackendRouting(
        default=RoleConfig(backend=column.backend, model="one"),
        roles={"implementer": RoleConfig(backend="codex", model="another")},
    )
    partly = matrix_columns.overlaid_routing(
        mixed, {field: "/a/peer"}, kind=kind, backend=column.backend
    )
    assert partly.default.extra == {name: "/a/peer"}
    assert partly.roles == mixed.roles


class _FixedServedId(ToyDialect):
    """Answers every request under one id, whatever model was asked for."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        return dataclasses.replace(
            reply, served_model="fixed-id", body={**reply.body, "model": "fixed-id"}
        )


def test_a_reply_under_another_model_without_a_served_as_step_is_a_seam_problem() -> None:
    script = ProviderScript((Say("one"), ServedAs(Say("two")), Say("three")))
    asked = {"model": "asked"}

    # The toy dialect echoes the model it was asked for; a ServedAs step is
    # the one way a script names another.
    with LoopbackProvider(script, credential="c") as provider:
        assert [_ask(provider, asked)[1]["model"] for _ in range(3)] == [
            "asked",
            "asked-served",
            "asked",
        ]
        seen = provider.freeze()
    assert seen.served_models == ["asked", "asked-served", "asked"]
    assert not any("answered under model" in problem for problem in seen.seam_problems)
    # A dialect that answers under a fixed id would switch off "served equals
    # requested" for its whole kind, with clause C7 still ok.
    with LoopbackProvider(script, credential="c", dialect=_FixedServedId()) as provider:
        for _ in range(3):
            assert _ask(provider, asked)[1]["model"] == "fixed-id"
        seen = provider.freeze()
    named = [problem for problem in seen.seam_problems if "answered under model" in problem]
    assert len(named) == 1 and "'fixed-id'" in named[0] and "'asked'" in named[0]


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
    # A pin that a column-wide gap is not owed states the cell, the code and
    # the reason, and nothing else.
    owed = {"cell": "kimi/S23c/primary/raw", "code": "pricing:none", "reason": "no flag asks"}
    for rows, parsed in (([owed], (owed,)), (None, ())):
        pins = cells.parse_inventory({**inventory, "column_wide_not_owed": rows})
        assert pins.column_wide_not_owed == parsed
    malformed: tuple[Any, ...] = (
        {name: value for name, value in owed.items() if name != "reason"},
        {**owed, "reason": " "},
        {**owed, "code": 6},
        {**owed, "lens": "raw"},
        owed["cell"],
    )
    for row in malformed:
        with pytest.raises(MatrixSetupError, match="must state a cell, a code and the reason"):
            cells.parse_inventory({**inventory, "column_wide_not_owed": [row]})
    with pytest.raises(MatrixSetupError, match="cell id"):
        cells.parse_inventory({**inventory, "column_wide_not_owed": [{**owed, "cell": "S23c"}]})
    with pytest.raises(MatrixSetupError, match="unknown shape"):
        cells.parse_inventory(
            {name: value for name, value in inventory.items() if name != "column_wide_not_owed"}
        )

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
    # The real kinds that are driven today, and no other, have cells.
    driven = {cell.column.kind for cell in cells.behaviour_cells()} - {TOY_KIND}
    assert driven == registered_kinds() - {TOY_KIND} == {
        matrix_columns.HOSTED_GATEWAY_KIND,
        matrix_columns.LOCAL_GATEWAY_KIND,
        matrix_columns.CLAUDE_KIND,
        matrix_columns.DEEPSEEK_KIND,
        matrix_columns.KIMI_KIND,
        matrix_columns.CODEX_LEGACY_KIND,
        matrix_columns.CODEX_OFFICIAL_KIND,
    }
    assert [cell for cell in cells.behaviour_cells() if cell.column.kind == TOY_KIND] == list(
        generated
    )
    assert set(cells.MODULE_BY_KIND) == registered_kinds()


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

    enabled = {name for name, item in SCENARIOS.items() if item.set in enablement["standin"]}
    assert by_role["primary"] == enabled == by_role["secondary-2"]
    # A blocked column runs its three scenarios whatever sets its kind runs:
    # the admission scenario is of a set this stand-in kind does not run.
    assert by_role["blocked"] == set(cells.BLOCKED_SCENARIOS) == {"S00", "S01", "S51"}
    assert SCENARIOS["S51"].set not in enablement["standin"] and "S51" not in enabled
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
# A kind with a column-wide entry, and what a cell of it shows.
WIDE_KIND = "standin"
WIDE_CELL = "standin/S01/primary/raw"
WIDE_CODES = {"pricing:none"}
WIDE_EVALUATED = frozenset({"C6"})
KNOWN = ledger.parse(
    {
        "schema_version": ledger.LEDGER_SCHEMA,
        "ceilings": {"max_known_gaps": {"pack_a": 2}, "max_declared_differences": 0},
        "registry": {
            "G99": {
                "title": "a known gap",
                "origin": "gap list",
                "severity": "medium",
                "cells": [{"cell": CELL, "observed_under": {}, "closes_under": {}}],
                "column_wide": [{"kind": WIDE_KIND, "code": code} for code in WIDE_CODES],
            }
        },
        "known_gaps": [
            {"cell": CELL, "observed_under": {}, "codes": {code: "G99" for code in CODES}},
            *(
                {"column_wide": {"kind": WIDE_KIND, "code": code, "gap": "G99"}}
                for code in WIDE_CODES
            ),
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
from inner_known import CELL, CODES, KNOWN, WIDE_CELL, WIDE_CODES, WIDE_EVALUATED, WIDE_KIND

from crucible.tests.backend_matrix import ledger


def _column_wide_verdict():
    ledger.compare(KNOWN, WIDE_CELL, WIDE_CODES, current={}, evaluated=WIDE_EVALUATED)


def test_a_self_test_that_reaches_the_comparators_verdict():
    ledger.compare(KNOWN, CELL, CODES, current={})


@pytest.mark.parametrize("cell", ["kimi/H02/-/static"], ids=str)
def test_a_cell_without_an_entry(cell):
    ledger.compare(KNOWN, CELL, CODES, current={})


@pytest.mark.parametrize("cell", [WIDE_CELL], ids=str)
def test_a_cell_that_owes_a_column_wide_gap(cell):
    _column_wide_verdict()


# Of a kind with a column-wide entry, and still no cell that owes it: a proof
# parametrised by the column, the seam canary, a static cell.
@pytest.mark.parametrize(
    "name",
    [f"{WIDE_KIND}/primary", f"{WIDE_KIND}/S00/primary/raw", f"{WIDE_KIND}/H02/-/static"],
    ids=str,
)
def test_an_item_of_a_kind_with_a_column_wide_entry_that_is_no_cell_owing_it(name):
    _column_wide_verdict()
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

    # The item of a cell the ledger expects to fail is the expected failure:
    # the cell with an entry, and a cell that owes a column-wide gap. A
    # self-test and a cell without an entry that reach the same verdict fail,
    # and so does an item whose id merely starts with a kind that has a
    # column-wide entry: a proof parametrised by the column, the seam canary,
    # a static cell.
    assert run.returncode == 1, run.stdout + run.stderr
    assert "5 failed, 2 xfailed" in run.stdout
    assert run.stdout.count("is not the test item of a cell the ledger expects to fail") >= 5


def test_skip_and_xfail_markers_are_a_collection_error(tmp_path: Path) -> None:
    root = _inner_tree(tmp_path / "inner")

    run = _inner_pytest(root, "test_inner_marked.py")
    output = run.stdout + run.stderr

    assert run.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert "refuses skip and xfail markers" in output
    for offender in ("test_skipped (skip)", "test_marked_expected_failure (xfail)", "NOTSET"):
        assert offender in output
    assert "passed" not in output and "skipped" not in output.split("refuses")[0]


def test_an_item_of_the_package_without_its_fixtures_refuses_the_run() -> None:
    package = Path(__file__).resolve().parent
    fixtures = ("_matrix_network_guard", "_matrix_environment")

    def item(path: Path, *names: str) -> SimpleNamespace:
        return SimpleNamespace(nodeid=f"{path.name}::test_it", path=path, fixturenames=names)

    whole = item(package / "test_matrix_static.py", "tmp_path", *fixtures)
    # The fixtures are in force for every item of the package: nothing is said.
    cells.refuse_items_without_fixtures([whole], package, fixtures)
    # An item outside the package is not the package's to judge.
    cells.refuse_items_without_fixtures(
        [whole, item(package.parent / "test_other.py", "tmp_path")], package, fixtures
    )
    # The lying run: pytest collected a module of the package after it had
    # left the directory, and the items of that module carry neither fixture
    # of the package, or one of the two. The run is refused before any test,
    # with the count, the first such item and how the package is called.
    for names in ((), fixtures[:1], fixtures[1:]):
        lost = item(package / "test_matrix_launch.py", "tmp_path", *names)
        with pytest.raises(pytest.UsageError, match="side by side") as refused:
            cells.refuse_items_without_fixtures([whole, lost, lost], package, fixtures)
        assert "2 item(s)" in str(refused.value) and lost.nodeid in str(refused.value)
        assert whole.nodeid not in str(refused.value)
    # An item that states no fixtures at all is refused as well.
    bare = SimpleNamespace(nodeid="test_matrix_self.py::bare", path=package / "test_matrix_self.py")
    with pytest.raises(pytest.UsageError, match="side by side"):
        cells.refuse_items_without_fixtures([bare], package, fixtures)
    # The conftest of the package asks for every fixture it applies to all
    # tests: one it gained and did not name here would be lost in silence.
    conftest = ast.parse((package / "conftest.py").read_text(encoding="utf-8"))
    applied = {
        node.name
        for node in conftest.body
        if isinstance(node, ast.FunctionDef)
        and any("autouse=True" in ast.unparse(decorator) for decorator in node.decorator_list)
    }
    (named,) = (
        ast.literal_eval(node.value)
        for node in conftest.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_IN_FORCE"
    )
    assert applied == set(named) == set(fixtures)


# -- H03 and P04 on synthetic ledgers and inventories ----------------------------


def _document() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(ledger.LEDGER_PATH.read_text(encoding="utf-8")))


def _problem_codes(document: dict[str, Any] | None = None, **world: Any) -> set[str]:
    known = LEDGER if document is None else ledger.parse(document)
    found = ledger.integrity_problems(known, dataclasses.replace(cells.world(), **world))
    return {item.code for item in found}


def _registry_cell(document: dict[str, Any], gap: str, cell: str) -> dict[str, Any]:
    return next(item for item in document["registry"][gap]["cells"] if item["cell"] == cell)


# A cell of pack B that has no entry: an admission cell is green by its
# definition, since its pin states what routing does.
_PACK_B_CELL = "deepseek/S51/primary/raw"


def _with_an_entry_of_its_own(cell: str = "deepseek/S13/primary/raw") -> dict[str, Any]:
    """The shipped ledger with one known gap more, its entry last in ``known_gaps``.

    An entry for a test to break. It stands on a cell that has none, under a
    record that is made to list the cell, and the ceiling of its pack is
    raised by the one, so H03 says nothing of it as it stands. It is no entry
    of the shipped ledger, so the fix of a gap edits no test for it. The
    cell is one of pack A unless the test names another.
    """

    assert LEDGER.entry(cell) is None
    world = cells.world()
    current = dict(world.current(cell))
    document = _document()
    document["registry"]["M01"]["cells"].append(
        {"cell": cell, "observed_under": current, "closes_under": current}
    )
    document["known_gaps"].append(
        {"cell": cell, "observed_under": current, "codes": {"secret:leak": "M01"}}
    )
    before = ledger.pack_counts(LEDGER, world)
    after = ledger.pack_counts(ledger.parse(document), world)
    (pack,) = (name for name in after if after[name] != before.get(name, 0))
    document["ceilings"]["max_known_gaps"][pack] += 1
    return document


def test_h03_rejects_an_entry_whose_cell_its_gap_does_not_list() -> None:
    document = _with_an_entry_of_its_own()
    assert _problem_codes(document) == set()
    document["known_gaps"][-1]["codes"] = {"static:worker_skip": "G02"}

    assert "static:entry_not_in_registry" in _problem_codes(document)


def test_h03_rejects_a_registry_cell_that_is_not_generated_once_its_kind_runs_the_set() -> None:
    # Every real kind runs pack A, so a pack A cell of a record that the
    # generator no longer yields is a cell that vanished.
    fewer = cells.world().generated - {"deepseek/S11/primary/raw"}

    assert "static:registry_cell_not_generated" in _problem_codes(generated=fewer)
    assert "static:registry_cell_not_generated" not in _problem_codes()
    # A kind that did not run the set would owe no such cell; it would owe the
    # expected failure of its H02 cell instead (the registry rules, below).
    starter_only = {**cells.ENABLEMENT, "deepseek": (STARTER,)}
    assert "static:registry_cell_not_generated" not in _problem_codes(
        generated=fewer, enabled_sets=starter_only
    )


def test_h03_rejects_a_not_expressible_row_on_a_registry_cell_without_a_ruling() -> None:
    # A form of refusal that one record alone lists on the kind.
    row = {"scenario": "S23c", "kind": "kimi", "step": "Refuse", "reason": "r", "checklist": "c"}
    ruled = _document()
    ruled["registry"]["G10"]["ruling"] = "owner, 2026-11-18: the cells are withdrawn"

    assert "static:not_expressible_registry_cell" in _problem_codes(not_expressible=(row,))
    assert "static:not_expressible_registry_cell" not in _problem_codes(
        ruled, not_expressible=(row,)
    )
    # S12 has one form, so it cannot be pinned on an open-weight kind at all.
    one_form = {**row, "scenario": "S12", "step": "Call"}
    assert "static:not_expressible_pack_a" in _problem_codes(not_expressible=(one_form,))
    assert "static:not_expressible_pack_a" not in _problem_codes(not_expressible=(row,))


def test_h03_refuses_a_ruling_on_a_record_that_withdraws_no_cell() -> None:
    # A ruling on a registry record is the licence to withdraw a cell of the
    # record, whatever its text says. One that keeps the gap withdraws
    # nothing and is refused: on a record with cells of its own, and on one
    # that is column-wide only, where it would license nothing at all.
    accepted = "the gap is accepted as it stands"
    for gap in ("M07", "M08", "R04", "M06"):
        kept = _document()
        kept["registry"][gap]["ruling"] = accepted
        assert _problem_codes(kept) == {"static:registry_ruling_unused"}, gap
    # In the entries of the gap the same words license nothing: that is
    # where the shipped ledger carries every ruling it has.
    assert _problem_codes() == set()
    assert not [record.id for record in LEDGER.registry.values() if record.ruling]
    ruled: list[ledger.Entry | ledger.ColumnWide] = [*LEDGER.entries, *LEDGER.column_wide]
    assert {gap for item in ruled if item.ruling for gap in _gaps_of(item)} == {
        "M06",
        "M07",
        "R04",
        "R05",
        "R06",
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R12",
        "R13",
        "R14",
        "R15",
        "R16",
        "R17",
        "R19",
        "R20",
        "R21",
        "R26",
    }
    # With a cell of the record withdrawn the ruling is what H03 asks for.
    # Either the cell is pinned not expressible ...
    row = {"scenario": "S01", "kind": "kimi", "step": "Say", "reason": "r", "checklist": "c"}
    withdrawing = _document()
    withdrawing["registry"]["M07"]["ruling"] = "the two cells are withdrawn"
    pinned = _problem_codes(withdrawing, not_expressible=(row,))
    assert "static:registry_ruling_unused" not in pinned
    # ... or it is no longer generated although its kind runs the set: with
    # the entries of the two cells deleted, and the frozen ceiling of their
    # pack lowered by the two, nothing is left to object to.
    gone = {item["cell"] for item in withdrawing["registry"]["M07"]["cells"]}
    withdrawing["known_gaps"] = [
        item for item in withdrawing["known_gaps"] if item.get("cell") not in gone
    ]
    _lowered(withdrawing)
    fewer = cells.world().generated - gone
    assert _problem_codes(withdrawing, generated=fewer) == set()
    # Without the ruling the same withdrawal is refused, cell by cell.
    del withdrawing["registry"]["M07"]["ruling"]
    assert _problem_codes(withdrawing, generated=fewer) == {
        "static:registry_cell_not_generated"
    }


def _gaps_of(item: ledger.Entry | ledger.ColumnWide) -> set[str]:
    return {item.gap} if isinstance(item, ledger.ColumnWide) else set(item.codes.values())


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
    # A registry cell that is generated and has no entry of its own.
    cell = "deepseek/S13/primary/raw"
    assert LEDGER.entry(cell) is None
    document = _document()
    _registry_cell(document, "G14", cell)["closes_under"] = {"usd_limit": "unsupported"}

    assert "static:registry_closes_under" in _problem_codes(document)

    document["registry"]["G14"]["declaration_change"] = {
        "field": "usd_limit",
        "from": "post_hoc",
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

    # On a kind that does not run pack A, a recorded pack A cell is visible
    # only through the expected failure of the kind's H02 cell. No kind is in
    # that state today and no H02 cell has an entry, so a kind that stopped
    # running pack A would leave its recorded cells unreachable...
    starter_only = {**cells.ENABLEMENT, "deepseek": (STARTER,)}
    assert LEDGER.entry("deepseek/H02/-/static") is None
    assert "static:registry_cell_unreachable" in _problem_codes(enabled_sets=starter_only)
    # ...unless the H02 cell of the kind is an expected failure again.
    reachable = _document()
    reachable["ceilings"]["max_known_gaps"]["pack_a"] += 1
    reachable["known_gaps"].append(
        {
            "cell": "deepseek/H02/-/static",
            "observed_under": {},
            "codes": {"static:pack_a_not_enabled": "G20"},
        }
    )
    assert "static:registry_cell_unreachable" not in _problem_codes(
        reachable, enabled_sets=starter_only
    )

    # A weakened declaration is seen even before the cell's column is driven.
    weakened = dataclasses.replace(cells.world(), current=lambda _cell: {"turn_limit": "unsupported"})
    assert "static:registry_closes_mismatch" in {
        item.code for item in ledger.integrity_problems(LEDGER, weakened)
    }

    # H02 has no code for a set beyond pack A, so a cell of such a set may be
    # recorded only once its kind runs the set. Every real kind runs pack B,
    # so the state before is made here: S12, of which records list cells, is
    # taken for a scenario of pack B, on kinds that run the two other sets.
    real = cells.world()
    later = {**real.scenario_sets, "S12": PACK_B}
    not_yet = {
        kind: sets if kind == TOY_KIND else tuple(name for name in sets if name != PACK_B)
        for kind, sets in cells.ENABLEMENT.items()
    }
    assert "static:registry_cell_early" in _problem_codes(
        scenario_sets=later, enabled_sets=not_yet
    )
    assert "static:registry_cell_early" not in _problem_codes(scenario_sets=later)

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


def _lowered(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with every frozen ceiling lowered by the entries that left its pack.

    What the change that takes an entry of a frozen pack away does in the
    same change: the shipped ceiling, less the shipped count of the pack,
    plus its count now.
    """

    world = cells.world()
    before = ledger.pack_counts(LEDGER, world)
    after = ledger.pack_counts(ledger.parse(document), world)
    for pack in document["ceilings"].get("frozen") or ():
        gone = before.get(pack, 0) - after.get(pack, 0)
        document["ceilings"]["max_known_gaps"][pack] = LEDGER.ceilings[pack] - gone
    return document


def _without_the_codes_of(gap: str) -> dict[str, Any]:
    """The shipped ledger as a fix of ``gap`` leaves it: no entry names the gap.

    The fix lowers the ceiling of a frozen pack by the entries it deleted.
    """

    document = _document()
    for item in document["known_gaps"]:
        if "codes" in item:
            item["codes"] = {code: owner for code, owner in item["codes"].items() if owner != gap}
    document["known_gaps"] = [
        item for item in document["known_gaps"] if "column_wide" in item or item["codes"]
    ]
    return _lowered(document)


def _without_an_entry_of(pack: str) -> dict[str, Any]:
    """The shipped ledger less the first entry that counts under ``pack``.

    Nothing else is edited, the ceilings least of all: what a fix that
    forgot the ceiling leaves. The entry is taken by its place and not by
    its gap, so the fix of a gap edits no test for it.
    """

    world = cells.world()
    count = ledger.pack_counts(LEDGER, world).get(pack, 0)
    shipped = _document()
    for index in range(len(shipped["known_gaps"])):
        fewer = copy.deepcopy(shipped)
        del fewer["known_gaps"][index]
        if ledger.pack_counts(ledger.parse(fewer), world).get(pack, 0) < count:
            return fewer
    raise MatrixSetupError(f"the ledger holds no entry that counts under {pack}")


def _without_the_entries_of(pack: str, document: dict[str, Any]) -> dict[str, Any]:
    """``document`` less every entry that counts under ``pack``.

    Nothing else is edited, the ceilings least of all. The entries are taken
    by their place, as above: which entries a pack holds, and how many, is
    the ledger's statement.
    """

    world = cells.world()

    def count(known_gaps: list[Any]) -> int:
        held = ledger.parse({**document, "known_gaps": known_gaps})
        return ledger.pack_counts(held, world).get(pack, 0)

    kept = list(document["known_gaps"])
    left = count(kept)
    index = 0
    while left and index < len(kept):
        fewer = kept[:index] + kept[index + 1 :]
        if count(fewer) < left:
            kept, left = fewer, left - 1
        else:
            index += 1
    return {**copy.deepcopy(document), "known_gaps": copy.deepcopy(kept)}


def _constructed(_cell: str) -> str:
    return "constructed"


def _closed_below_grade(document: dict[str, Any], world: ledger.World) -> list[str]:
    """What H03 says of the cells that closed on a reply below their record's grade."""

    return [
        item.detail
        for item in ledger.integrity_problems(ledger.parse(document), world)
        if item.code == "static:registry_closed_below_grade"
    ]


def test_h03_refuses_to_close_a_provider_fact_gap_on_a_reply_below_its_grade() -> None:
    # The lying counterpart: a fix of G18 made its three cells green, the
    # comparator asked for the codes of the gap to go, and they went, while
    # each cell is answered from a reply that was constructed from the types
    # of an SDK, a shape no provider is known to send. The grade is given
    # here and not read from the shipped envelopes, and the ledger is the
    # shipped one without the codes of the gap, so the test says the same
    # once the gap is fixed on better replies.
    closed = _without_the_codes_of("G18")
    listed = {item.cell for item in LEDGER.registry["G18"].cells}
    # Constructed replies for the cells of this record and a recorded one
    # for every other cell: whether another graded record is closed, and on
    # what, is the ledger's statement, so its fix edits nothing here.
    constructed = dataclasses.replace(
        cells.world(), grade=lambda cell: "constructed" if cell in listed else "recorded"
    )
    found = _closed_below_grade(closed, constructed)

    assert _problem_codes(closed, grade=_constructed) == {"static:registry_closed_below_grade"}
    assert len(found) == len(listed) == 3
    assert all(any(cell in detail for detail in found) for cell in listed)
    # The same deletion on replies of the grade the record names is a fix.
    for grade in ("documented", "recorded"):
        assert _problem_codes(closed, grade=lambda _cell, grade=grade: grade) == set()
    # With the grades of the shipped envelopes the rule names exactly the
    # cells of the record that a weaker reply still answers.
    real = cells.world()
    grades = matrix_envelopes.GRADES
    weaker = {
        cell
        for cell in listed
        if real.grade(cell) in grades[grades.index("documented") + 1 :]
    }
    named = _closed_below_grade(closed, real)
    assert {cell for cell in listed if any(cell in detail for detail in named)} == weaker
    assert len(named) == len(weaker)
    # The rule reads the codes an entry attributes to the record, not whether
    # the cell has an entry: one that stands for another gap only does not
    # keep this gap open.
    cell = sorted(listed)[0]
    elsewhere = _without_the_codes_of("G18")
    elsewhere["known_gaps"] = [
        item for item in elsewhere["known_gaps"] if item.get("cell") != cell
    ]
    elsewhere["registry"]["M01"]["cells"].append(
        {"cell": cell, "observed_under": {}, "closes_under": {}}
    )
    elsewhere["known_gaps"].append(
        {"cell": cell, "observed_under": {}, "codes": {"secret:leak": "M01"}}
    )
    # The frozen ceiling follows whatever that leaves of the count of its
    # pack: which cells of the gap keep an entry for another record is the
    # ledger's statement, and no number of this test.
    _lowered(elsewhere)
    assert ledger.parse(elsewhere).entry(cell) is not None
    assert _problem_codes(elsewhere, grade=_constructed) == {
        "static:registry_closed_below_grade"
    }
    assert any(cell in detail for detail in _closed_below_grade(elsewhere, constructed))
    # The shipped ledger states the grade for every gap whose substance is
    # what a provider sends, and on the reference columns for every gap
    # whose substance is what the vendor's tool relays of such a reply or
    # sends when a session is stopped; the rule holds each of them.
    graded = {record.id for record in LEDGER.registry.values() if record.closes_on_grade}
    assert set(ledger.PROVIDER_FACT_GAPS) & set(LEDGER.registry) <= graded
    # The rule holds a cell for as long as its record lists it, and a record
    # is permanent: the graded records are named and their cells counted
    # here, so that no record drops the field and no cell leaves its record
    # together with its code. The design's list holds the field of the
    # open-weight records that are on it; nothing else holds the field of a
    # record of origin ``reference`` or of the matrix.
    counted = {
        "G09": 1,
        "G10": 3,
        "G13": 1,
        "G18": 3,
        "G21": 1,
        "G25": 2,
        "G27": 1,
        "G34": 2,
        "G35": 11,
        "M10": 2,
        "R07": 2,
        "R08": 1,
        "R09": 1,
        "R10": 2,
        "R11": 2,
        "R14": 1,
        "R16": 1,
        "R17": 1,
        "R20": 1,
        "R21": 1,
        "R26": 1,
    }
    assert graded == set(counted)
    assert {gap: len(LEDGER.registry[gap].cells) for gap in counted} == counted
    references = {gap for gap in graded if LEDGER.registry[gap].origin == "reference"}
    assert references == {
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R14",
        "R16",
        "R17",
        "R20",
        "R21",
        "R26",
    }
    # What the record of each says its cells are answered from today. A
    # cell of a reference may be answered from several frames; its grade is
    # that of the weakest, and each of these is constructed. A change that
    # brings another frame into the script of such a cell, or regrades one,
    # rewrites this.
    tool_call = {
        "codex_app_server/app_server.tool_call_started",
        "codex_app_server/app_server.token_usage_updated",
    }
    answered = {
        "R07": {"claude_cli/cli.init", "claude_cli/cli.assistant_tool_use"},
        "R08": tool_call,
        "R09": tool_call,
        "R10": {
            "claude_cli/cli.init",
            "claude_cli/cli.assistant_refusal",
            "claude_cli/cli.result_refusal",
        },
        "R11": {
            "claude_cli/cli.init",
            "claude_cli/cli.assistant_api_error",
            "claude_cli/cli.result_api_error",
        },
        "R14": {"codex_app_server/app_server.turn_failed_policy"},
        "R16": tool_call
        | {
            "codex_app_server/app_server.reasoning",
            "codex_app_server/app_server.turn_completed",
        },
        # The general frames of a tool call and of an answered turn. The
        # failed item of a call for a name no server lists, which the grade
        # of the record is about, is rendered from what the tool server
        # answered and is read here for no cell.
        "R17": tool_call
        | {
            "codex_app_server/app_server.agent_message",
            "codex_app_server/app_server.turn_completed",
        },
        # An empty completion: the usage notification and the completed turn.
        "R20": {
            "codex_app_server/app_server.token_usage_updated",
            "codex_app_server/app_server.turn_completed",
        },
        # A refused resume: the one envelope of the refusal, which no model
        # turn follows.
        "R21": {"codex_app_server/app_server.resume_rejected"},
        "R26": {"claude_cli/cli.init", "claude_cli/cli.assistant_tool_use"},
    }
    assert set(answered) == references
    for gap, frames in answered.items():
        for item in LEDGER.registry[gap].cells:
            assert set(cells.answered_from(item.cell)) == frames, item.cell
            assert cells.grade_of(item.cell) == "constructed", item.cell
    for gap in sorted(graded):
        assert _problem_codes(_without_the_codes_of(gap), grade=_constructed) == {
            "static:registry_closed_below_grade"
        }, gap
    # A record of a gap the closure rule lists cannot drop the field ...
    ungraded = _document()
    del ungraded["registry"]["G18"]["closes_on_grade"]
    assert _problem_codes(ungraded) == {"static:registry_grade_missing"}
    # ... and the field names a grade above the weakest one, or the record
    # says nothing.
    for grade in ("constructed", "guessed", ""):
        unknown = _document()
        unknown["registry"]["G18"]["closes_on_grade"] = grade
        with pytest.raises(MatrixSetupError, match="closes_on_grade"):
            ledger.parse(unknown)
    # A gap that is no provider fact closes on whatever its cells are
    # answered from: a record without the field is not held.
    assert LEDGER.registry["G16"].closes_on_grade is None
    assert _problem_codes(_without_the_codes_of("G16")) == set()
    # The terminal flag that the gateway adapter writes true on every path
    # is no provider fact either, and has a record of its own for that
    # reason: its cells close when they stop showing the code, whatever
    # they are answered from.
    assert LEDGER.registry["M15"].closes_on_grade is None
    assert _problem_codes(_without_the_codes_of("M15"), grade=_constructed) == set()
    # The lying counterpart: the same cells listed in the record of the
    # raise of the same adapter, with a grade on that record, as a gap
    # about a reply no provider is known to send names one. The grade is
    # written here and not read from the shipped record, so the test says
    # the same whatever grade that record names. The fix of the flag
    # deletes the same codes, and the deletion is refused on every cell
    # that a constructed reply answers, for a reason that is none of the
    # flag's: no edit of the ledger lands that fix.
    rebooked = _without_the_codes_of("M15")
    moved = rebooked["registry"].pop("M15")["cells"]
    rebooked["registry"]["G34"]["closes_on_grade"] = "documented"
    rebooked["registry"]["G34"]["cells"].extend(moved)
    assert _problem_codes(rebooked, grade=_constructed) == {
        "static:registry_closed_below_grade"
    }
    refused = _closed_below_grade(
        rebooked, dataclasses.replace(cells.world(), grade=_constructed)
    )
    assert len(refused) == len(moved) and all(
        any(item["cell"] in detail for detail in refused) for item in moved
    )


def test_h03_holds_a_gap_open_while_its_record_waits_for_a_ruling() -> None:
    # The lying counterpart: a change made the one cell of G15 stop raising by
    # cutting the turn count to a ceiling of the adapter's own, the comparator
    # asked for the codes of the gap to go, and they went. The closing state
    # of the gap is a ceiling that is reported, or a refusal at routing. What
    # "reported" asks of a route whose wire carries no turn control is stated
    # by no clause, so the cell cannot show that state: the record says that
    # a ruling is owed first, and the integrity check holds the gap open
    # whatever its cell shows.
    cell = "deepseek/S11/primary/raw"
    closed = _without_the_codes_of("G15")
    assert ledger.parse(closed).entry(cell) is None

    assert _problem_codes(closed) == {"static:registry_closed_before_ruling"}
    record = LEDGER.registry["G15"]
    assert record.awaits_ruling and [item.cell for item in record.cells] == [cell]
    (held,) = ledger.integrity_problems(ledger.parse(closed), cells.world())
    assert cell in held.detail and record.awaits_ruling in held.detail
    # What such a change leaves on the cell is the terminal name of another
    # record. The code is written out here and not read from an entry of
    # that record: those go with its own fix. An entry that stands for that
    # record only does not keep this gap open.
    elsewhere = _document()
    (entry,) = (item for item in elsewhere["known_gaps"] if item.get("cell") == cell)
    current = dict(cells.world().current(cell))
    entry["codes"] = {"event.terminal:response.completed!=end_turn": "M05"}
    elsewhere["registry"]["M05"]["cells"].append(
        {"cell": cell, "observed_under": current, "closes_under": current}
    )
    assert ledger.parse(elsewhere).entry(cell) is not None
    assert _problem_codes(elsewhere) == {"static:registry_closed_before_ruling"}
    # The change that enters the ruling makes the scenario assert it and
    # takes the field away: the same two ledgers are then a fix.
    for document in (closed, elsewhere):
        del document["registry"]["G15"]["awaits_ruling"]
        assert _problem_codes(document) == set()
    # The field is the question that is owed an answer, in words.
    for text in ("", " ", None, True):
        blank = _document()
        blank["registry"]["G15"]["awaits_ruling"] = text
        with pytest.raises(MatrixSetupError, match="awaits_ruling"):
            ledger.parse(blank)
    # A record that waits for no ruling closes when its cells stop deviating,
    # and the shipped ledger, in which the gap is open, is clean.
    assert LEDGER.registry["G16"].awaits_ruling is None
    assert _problem_codes(_without_the_codes_of("G16")) == set()
    assert _problem_codes() == set()


def test_the_grade_h03_reads_is_that_of_the_replies_the_dialect_answers_a_cell_from() -> None:
    world = cells.world()
    grades = matrix_envelopes.GRADES

    def stated(name: str) -> str:
        """The grade an envelope states of itself, read from its file."""

        text = (matrix_envelopes.ROOT / f"{name}.yaml").read_text(encoding="utf-8")
        return str(yaml.safe_load(text)["provenance"]["grade"])

    # Read without playing the cell: the script of the cell's variant, each
    # step rendered by the dialect of its kind against a blank request. The
    # examples are what the envelopes state: on every cell of a real column
    # the grade is that of the weakest envelope the cell is answered from,
    # as the file of that envelope has it.
    answered_kinds: set[str] = set()
    for cell in cells.behaviour_cells():
        if cell.column.kind == TOY_KIND:
            continue
        names = cells.answered_from(cell.id)
        weakest = max((stated(name) for name in names), key=grades.index, default=None)
        assert world.grade(cell.id) == weakest, cell.id
        if names:
            answered_kinds.add(cell.column.kind)
    # The examples are there on every real kind; a cell that is answered
    # from no envelope (an admission cell calls nothing) has no grade.
    assert answered_kinds == set(matrix_columns.REAL_KINDS)
    assert cells.answered_from(_PACK_B_CELL) == () and world.grade(_PACK_B_CELL) is None
    # No grade is written here for a cell of a record that closes on a
    # grade, and no envelope of such a cell is named: the fix of such a gap
    # brings a reply of a better grade first, and edits nothing here. The
    # literals stand on cells that no such record lists. They hold the
    # dialect: one that answered the plain call, the tool round or the
    # content-filter stop of these columns from an envelope of another
    # grade, or from another envelope, fails here.
    graded = {
        item.cell
        for record in LEDGER.registry.values()
        if record.closes_on_grade
        for item in record.cells
    }
    literal = {
        "claude/S01/primary/raw": "constructed",
        "deepseek/S01/primary/raw": "documented",
        "kimi/S23a/secondary/raw": "constructed",
        "openai_compat:hosted/S01/primary/guarded": "documented",
        "openai_compat:hosted/S03/primary/guarded": "constructed",
    }
    answered = {
        "deepseek/S01/primary/raw": ("deepseek_responses/deepseek.completion",),
        "openai_compat:hosted/S04/primary/raw": (
            "openai_chat/gateway.tool_call",
            "openai_chat/gateway.completion",
        ),
    }
    assert graded and not graded & {*literal, *answered}
    for cell_id, grade in literal.items():
        assert world.grade(cell_id) == grade, cell_id
    for cell_id, envelope_names in answered.items():
        assert cells.answered_from(cell_id) == envelope_names, cell_id
    # A toy cell is answered from no envelope, and a static cell from nothing.
    assert world.grade(TOY_CELLS[0].id) is None
    assert world.grade("all/H03/-/static") is None


# The cells that no model answers. Their invocation asks to continue a
# conversation, their route answers the request, and a tool beside the
# adapter keeps the conversations of their wire: each is answered from what
# that tool says of a handle it does not hold, an envelope that says of
# itself that it is emulated and was not observed.
_REFUSED_BY_A_KEEPER = {
    "claude/S50/primary/raw": "claude_cli/cli.resume_rejected",
    "codex:legacy/S50/primary/raw": "codex_app_server/app_server.resume_rejected",
    "codex:official/S50/primary/raw": "codex_app_server/app_server.resume_rejected",
}


def test_h03_reads_what_the_keeper_of_a_conversation_says_as_what_a_cell_is_answered_from() -> (
    None
):
    world = cells.world()
    # The lying counterpart: the verdict of such a cell is held as emulated
    # until the tool is observed, in the form the ledger has for that, a
    # record that closes on a documented reply. Then the codes of the record
    # go from the cell, or were never there because the cell was green. The
    # script of the cell is empty, so a reading that renders the steps of the
    # script finds no envelope, the cell has no grade, and the record closes
    # on the emulation.
    for cell, envelope in _REFUSED_BY_A_KEEPER.items():
        current = dict(world.current(cell))
        closed = _document()
        closed["registry"]["emulated"] = {
            "title": "A verdict that rests on what a vendor tool is emulated to say",
            "origin": "reference",
            "severity": "medium",
            "closes_on_grade": "documented",
            "cells": [{"cell": cell, "observed_under": current, "closes_under": current}],
        }
        assert _problem_codes(closed) == {"static:registry_closed_below_grade"}, cell
        (held,) = _closed_below_grade(closed, world)
        assert cell in held and "of grade constructed" in held
        # The same ledger is a fix once a reply of the named grade answers
        # the cell.
        assert _problem_codes(closed, grade=lambda _cell: "documented") == set(), cell
        # What is read without a play: the one envelope, and the grade it
        # states of itself.
        assert cells.answered_from(cell) == (envelope,)
        assert world.grade(cell) == matrix_envelopes.load(envelope).grade == "constructed"
    # No other cell is answered from such an envelope. A route that refuses
    # the request before transport asks nobody: its cell is answered from
    # nothing and has no grade.
    refusals = set(_REFUSED_BY_A_KEEPER.values())
    asked = {
        cell.id
        for cell in cells.behaviour_cells()
        if cell.column.kind != TOY_KIND and refusals & set(cells.answered_from(cell.id))
    }
    assert asked == set(_REFUSED_BY_A_KEEPER)
    for kind in sorted(matrix_columns.OPEN_WEIGHT_KINDS):
        refused = f"{kind}/S50/primary/raw"
        assert cells.answered_from(refused) == () and world.grade(refused) is None
    # The reading follows the plan and not the kind. On a wire with such a
    # tool a route that refused the request before transport would ask
    # nobody, and neither does a call that asks to continue no conversation;
    # on a wire without one there is nobody but the adapter to ask.
    scenario = SCENARIOS["S50"]
    answering = scenario.plan()
    refusing = plan_for(
        scenario, {"session_continuity": scenario.differences["session_continuity"]["none"]}
    )
    assert answering.conversation is not None and refusing.conversation is not None
    assert (answering.expect.outcome, refusing.expect.outcome) == ("result", "refused")
    for cell, envelope in _REFUSED_BY_A_KEEPER.items():
        kind = ledger.parse_cell_id(cell)[0]
        assert cells.envelopes_of(kind, answering) == (envelope,)
        assert cells.envelopes_of(kind, refusing) == ()
        assert cells.envelopes_of(kind, dataclasses.replace(answering, conversation=None)) == ()
    for kind in sorted(matrix_columns.OPEN_WEIGHT_KINDS):
        assert cells.envelopes_of(kind, answering) == ()


def test_h03_entry_rules() -> None:
    world = cells.world()
    # What the ceiling of pack A limits: the entries of that pack, and not
    # the length of the list, which holds those of pack B as well.
    pack_a = ledger.pack_counts(LEDGER, world)["pack_a"]
    on_canary = _document()
    canary = next(cell.id for cell in TOY_CELLS if SCENARIOS[cell.scenario].canary)
    # At its ceiling, whatever room the shipped ceiling leaves today.
    on_canary["ceilings"]["max_known_gaps"]["pack_a"] = pack_a
    on_canary["known_gaps"].append(
        {"cell": canary, "observed_under": {}, "codes": {"request.unconsumed": "G02"}}
    )
    assert {"static:entry_on_canary", "static:ceiling_exceeded"} <= _problem_codes(on_canary)

    # The entry each of the following breaks is one of the test's own, last
    # in the list: which entries the shipped ledger holds, and in which
    # order, is the ledger's statement.
    assert _problem_codes(_with_an_entry_of_its_own()) == set()

    unused = _with_an_entry_of_its_own()
    unused["known_gaps"][-1]["cell"] = "all/H09/-/static"
    assert "static:entry_unused" in _problem_codes(unused)

    observed = _with_an_entry_of_its_own()
    assert "turn_limit" not in observed["known_gaps"][-1]["observed_under"]
    observed["known_gaps"][-1]["observed_under"] = {"turn_limit": "hard"}
    assert "static:entry_observed_under" in _problem_codes(observed)

    invalid = _with_an_entry_of_its_own()
    invalid["known_gaps"][-1]["codes"] = {"decl.changed:turn_limit": "G04"}
    assert "static:entry_code_invalid" in _problem_codes(invalid)

    ceiling = _document()
    ceiling["ceilings"]["max_known_gaps"]["pack_a"] = pack_a - 1
    assert "static:ceiling_exceeded" in _problem_codes(ceiling)
    exact = _document()
    exact["ceilings"]["max_known_gaps"]["pack_a"] = pack_a
    assert _problem_codes(exact) == set()
    # An entry of pack B counts under the ceiling of pack B and under no
    # other: with the entry there, each pack is still held to its own count.
    other = _with_an_entry_of_its_own(_PACK_B_CELL)
    counts = ledger.pack_counts(ledger.parse(other), world)
    assert counts["pack_a"] == pack_a and counts["pack_b"] == len(other["known_gaps"]) - pack_a
    assert _problem_codes(other) == set()
    other["ceilings"]["max_known_gaps"]["pack_a"] = pack_a - 1
    assert _problem_codes(other) == {"static:ceiling_exceeded"}
    other["ceilings"]["max_known_gaps"]["pack_a"] = pack_a
    other["ceilings"]["max_known_gaps"]["pack_b"] = counts["pack_b"] - 1
    assert _problem_codes(other) == {"static:ceiling_exceeded"}

    reference = _document()
    reference["registry"]["M01"]["cells"].append(
        {"cell": "claude/S01/primary/raw", "observed_under": {}, "closes_under": {}}
    )
    reference["known_gaps"].append(
        {"cell": "claude/S01/primary/raw", "observed_under": {}, "codes": {"secret:leak": "M01"}}
    )
    assert "static:entry_ruling_missing" in _problem_codes(reference)
    # A column-wide entry on a reference kind needs the owner's ruling too.
    wide = _document()
    wide["ceilings"]["max_known_gaps"]["pack_a"] += 1
    wide["registry"]["M01"]["column_wide"] = [{"kind": "claude", "code": "pricing:none"}]
    wide["known_gaps"].append(
        {"column_wide": {"kind": "claude", "code": "pricing:none", "gap": "M01"}}
    )
    assert _problem_codes(wide) == {"static:entry_ruling_missing"}
    wide["known_gaps"][-1]["ruling"] = "the owner accepts it as a gap of the reference"
    assert _problem_codes(wide) == set()

    duplicate = _with_an_entry_of_its_own()
    duplicate["known_gaps"].append(copy.deepcopy(duplicate["known_gaps"][-1]))
    assert "static:entry_duplicate" in _problem_codes(duplicate)

    empty = _with_an_entry_of_its_own()
    empty["known_gaps"][-1]["codes"] = {}
    assert _problem_codes(empty) == {"static:entry_without_codes"}
    # The comparator does not let such an entry excuse a passing cell either.
    with pytest.raises(ledger.LedgerMismatch, match="gap fixed, delete the entry"):
        ledger.compare(
            ledger.parse(empty),
            empty["known_gaps"][-1]["cell"],
            set(),
            current=empty["known_gaps"][-1]["observed_under"],
        )

    differing = _document()
    differing["ceilings"]["max_declared_differences"] = len(differing["declared_differences"]) - 1
    assert _problem_codes(differing) == {"static:ceiling_exceeded"}

    # A column-wide entry is attributed like any other: its gap must list it.
    wide = _document()
    wide["ceilings"]["max_known_gaps"]["pack_a"] += 1
    wide["known_gaps"].append(
        {"column_wide": {"kind": "kimi", "code": "pricing:none", "gap": "G02"}}
    )
    assert _problem_codes(wide) == {"static:entry_not_in_registry"}
    wide["registry"]["G02"]["column_wide"] = [{"kind": "kimi", "code": "pricing:none"}]
    assert _problem_codes(wide) == set()


def test_h03_holds_a_frozen_ceiling_to_the_count_of_its_pack() -> None:
    world = cells.world()
    counts = ledger.pack_counts(LEDGER, world)
    # The shipped ledger states the ceiling of pack A frozen. A frozen
    # ceiling only falls, and no run has the history to see a number rise.
    # So it is held to the count of its pack, and never above the number it
    # was frozen at or has fallen to since, by the fix of a gap: an entry
    # that a ruling admits after the freeze raises that number here, in the
    # change that writes the entry.
    assert LEDGER.frozen == {"pack_a"}
    assert LEDGER.ceilings["pack_a"] == counts["pack_a"] <= 74
    assert _problem_codes() == set()

    # The lying counterpart: a fix deleted its entry and left the ceiling.
    # That opens room under a number that says it only falls, and a later
    # entry would fit into it with no edit of the ceiling at all. The entry
    # is the first the ledger holds in the pack, whichever gap it stands
    # for: the fix of a gap deletes entries and edits nothing here.
    refused = "static:ceiling_not_the_count"
    forgot = _without_an_entry_of("pack_a")
    assert forgot["ceilings"] == _document()["ceilings"]
    # The same fix with the ceiling lowered by the entry it deleted.
    fixed = _lowered(copy.deepcopy(forgot))
    assert fixed["ceilings"]["max_known_gaps"]["pack_a"] == LEDGER.ceilings["pack_a"] - 1
    assert fixed["known_gaps"] == forgot["known_gaps"]
    # Of the fix that forgot H03 says the code of the ceiling, beside what
    # it says of the same deletion once the ceiling is lowered: the entry
    # may be one that H03 does not let go for a reason of its own.
    lowered = _problem_codes(fixed)
    assert refused not in lowered and _problem_codes(forgot) == lowered | {refused}
    (found,) = (
        item
        for item in ledger.integrity_problems(ledger.parse(forgot), world)
        if item.code == refused
    )
    assert "pack_a" in found.detail and "frozen" in found.detail
    # A frozen ceiling above the count is refused however it came about, and
    # one below the count is exceeded, as any ceiling is.
    for moved, code in ((1, "static:ceiling_not_the_count"), (-1, "static:ceiling_exceeded")):
        other = _document()
        other["ceilings"]["max_known_gaps"]["pack_a"] += moved
        assert _problem_codes(other) == {code}, moved
    # While the entries of a pack still land its ceiling may leave room: the
    # rule binds a pack once the ledger states it frozen, and not before.
    rising = _document()
    rising["ceilings"]["frozen"] = []
    rising["ceilings"]["max_known_gaps"]["pack_a"] += 36
    assert _problem_codes(rising) == set()
    del rising["ceilings"]["frozen"]
    assert ledger.parse(rising).frozen == frozenset() and _problem_codes(rising) == set()
    # An entry of another pack does not count under a frozen ceiling: with
    # an entry of pack B there, the ceiling of pack A is still the count of
    # pack A, and stays where it is.
    other = _with_an_entry_of_its_own(_PACK_B_CELL)
    assert ledger.pack_counts(ledger.parse(other), world) == {
        **counts,
        "pack_b": counts.get("pack_b", 0) + 1,
    }
    assert other["ceilings"]["max_known_gaps"]["pack_a"] == LEDGER.ceilings["pack_a"]
    assert _problem_codes(other) == set()
    # A frozen pack that holds no entry has a ceiling of nothing. Pack B is
    # emptied for it, whatever the ledger holds there: every entry of the
    # pack goes, by its place, the one of this test among them. What H03
    # says of the entries that went is read first, with the pack not frozen:
    # an entry may be one that H03 does not let go for a reason of its own.
    empty = _without_the_entries_of("pack_b", other)
    assert len(empty["known_gaps"]) == counts["pack_a"]
    assert not ledger.pack_counts(ledger.parse(empty), world).get("pack_b")
    empty["ceilings"]["max_known_gaps"]["pack_b"] = 0
    empty["ceilings"]["frozen"] = sorted(LEDGER.frozen - {"pack_b"})
    unfrozen = _problem_codes(empty)
    assert refused not in unfrozen
    empty["ceilings"]["frozen"] = sorted(LEDGER.frozen | {"pack_b"})
    assert _problem_codes(empty) == unfrozen
    empty["ceilings"]["max_known_gaps"]["pack_b"] = 1
    assert _problem_codes(empty) == unfrozen | {refused}
    (found,) = (
        item
        for item in ledger.integrity_problems(ledger.parse(empty), world)
        if item.code == refused
    )
    assert "pack_b" in found.detail and "frozen" in found.detail
    # What the rule cannot see is one change that deletes an entry and writes
    # another: the count and the number stand as they were. The new entry is
    # then in the diff of that change, on a cell that had none.
    # The field names packs that have a ceiling, once each.
    for value in (["pack_c"], ["pack_a", "pack_a"], "pack_a", [1], {"pack_a": True}):
        unknown = _document()
        unknown["ceilings"]["frozen"] = value
        with pytest.raises(MatrixSetupError, match="ceilings.frozen"):
            ledger.parse(unknown)


def _leaned_on(known: ledger.Ledger, gaps: Any = None) -> set[tuple[str, str]]:
    """The proofs the entries of ``known`` lean on, for every record or for ``gaps``."""

    def wanted(gap: str) -> bool:
        return gaps is None or gap in gaps

    return {
        item
        for entry in known.entries
        for gap in set(entry.codes.values())
        if wanted(gap)
        for item in cells.proofs_leaned_on(gap, entry.cell)
    } | {
        item
        for scope in known.column_wide
        if wanted(scope.gap)
        for item in cells.proofs_leaned_on(scope.gap, None)
    }


def test_h03_refuses_an_entry_that_outlives_a_proof_it_leans_on() -> None:
    world = cells.world()
    required = cells.required_proof_items()
    owed = _leaned_on(LEDGER)

    # A proof holds what the codes of an entry do not say. The records whose
    # entries lean on one are named here, so that a row of the table cannot
    # go with its proof while an entry still names the record.
    assert set(cells.RECORD_PROOFS) == {
        "G25",
        "M18",
        "M28",
        "R04",
        "R05",
        "R06",
        "R08",
        "R09",
        "R13",
        "R16",
        "R17",
        "R20",
        "R21",
        "R26",
    }
    # Where a record has a proof for each kind of its cells, an entry leans
    # on the proof of its own kind and on no other.
    legacy, official = cells.RECORD_PROOFS["R06"]
    assert (legacy.kind, official.kind) == (
        matrix_columns.CODEX_LEGACY_KIND,
        matrix_columns.CODEX_OFFICIAL_KIND,
    )
    assert legacy.item("codex:legacy/S28/primary/raw") == legacy.test
    assert legacy.item("codex:official/S28/primary/raw") is None
    assert official.item("codex:legacy/S28/primary/raw") is None
    # Where a record has cells of more than one scenario in the gap list and
    # its proof plays one, an entry of the record on another scenario leans
    # on nothing, and so does a column-wide entry.
    (refused,) = cells.RECORD_PROOFS["G25"]
    assert refused.scenario == cells.SERVED_UNDER_ANOTHER_MODEL
    assert refused.item("deepseek/S24/primary/raw") == refused.test
    assert refused.item("deepseek/S01/primary/raw") is None and refused.item(None) is None
    # A proof with one item for every column plays the cell of the entry's
    # own column.
    (on_column,) = cells.RECORD_PROOFS["M28"]
    assert on_column.item("openai_compat:local/S24/primary/raw") == (
        f"{on_column.test}[openai_compat:local/primary]"
    )
    # A proof with one item for each order of the scripted tool's request:
    # the entry leans on the item of the order its cell plays, and the item
    # of the other order is required beside it.
    (in_order,) = cells.RECORD_PROOFS["R26"]
    assert in_order.item("claude/S24/primary/raw") == (
        f"{cells.REFUSED_TURN_IN_EACH_ORDER}[joined]"
    )
    assert in_order.item("claude/S04/primary/raw") is None and in_order.item(None) is None
    assert {
        (in_order.module, f"{cells.REFUSED_TURN_IN_EACH_ORDER}[{order}]")
        for order in ("joined", "held")
    } <= required
    assert set(cells.RECORD_PROOFS) <= set(LEDGER.registry)
    # Each of them is named by an entry today, and every proof those entries
    # lean on is one the verifier requires by name.
    for gap in cells.RECORD_PROOFS:
        assert _leaned_on(LEDGER, {gap}), gap
    assert owed and owed <= required == world.required_proofs
    assert _problem_codes() == set()

    # The lying counterpart: a change deleted a proof and its name from the
    # list the verifier requires and left the entries that lean on it. Every
    # test of the package is green without the proof, and the verifier calls
    # a run without it complete. Shown for every proof an entry leans on.
    for module, name in sorted(owed):
        found = ledger.integrity_problems(
            LEDGER, dataclasses.replace(world, required_proofs=required - {(module, name)})
        )
        assert found and {item.code for item in found} == {
            "static:entry_proof_not_required"
        }, name
        assert all(f"{module}::{name}" in item.detail for item in found), name
    # An entry leans on the item of the proof that plays its own cell, and
    # on no other: without the twin of one guarded cell, the entry of that
    # cell is named and no entry of another scenario or lens.
    raw, twin = cells.RECORD_PROOFS["R05"]
    cell = "codex:official/S10a/primary/guarded"
    assert twin.item(cell.replace("guarded", "raw")) is None and twin.item(None) is None
    assert raw.item(None) is None and raw.item(cell) == f"{raw.test}[S10a]"
    without = required - {(twin.module, str(twin.item(cell)))}
    (named,) = ledger.integrity_problems(LEDGER, dataclasses.replace(world, required_proofs=without))
    assert f"entry {cell} names R05" in named.detail
    # A column-wide entry names no cell: it leans on a test of which one
    # item stands for the record. Without that item both column-wide
    # entries of the record are named, and each of its cell entries.
    (one,) = cells.RECORD_PROOFS["R04"]
    wide = [scope for scope in LEDGER.column_wide if scope.gap == "R04"]
    on_cells = [entry for entry in LEDGER.entries if "R04" in entry.codes.values()]
    found = ledger.integrity_problems(
        LEDGER, dataclasses.replace(world, required_proofs=required - {(one.module, one.test)})
    )
    assert wide and on_cells and len(found) == len(wide) + len(on_cells)
    assert sum(item.detail.startswith("column-wide ") for item in found) == len(wide)
    # The same deletion with the entries of the record gone is what the fix
    # of the record may do: a record that no entry names leans on nothing.
    for gap in sorted(cells.RECORD_PROOFS):
        fixed = _without_the_codes_of(gap)
        fixed["known_gaps"] = [
            item
            for item in fixed["known_gaps"]
            if "column_wide" not in item or item["column_wide"]["gap"] != gap
        ]
        fewer = required - _leaned_on(LEDGER, {gap})
        assert fewer != required
        assert "static:entry_proof_not_required" not in _problem_codes(
            _lowered(fixed), required_proofs=fewer
        ), gap
    # A record that is not in the table leans on nothing: with no proof
    # required at all, the entries H03 names are those of the table's records.
    assert cells.proofs_leaned_on("G17", "deepseek/S40/primary/raw") == ()
    found = ledger.integrity_problems(
        LEDGER, dataclasses.replace(world, required_proofs=frozenset())
    )
    assert {item.code for item in found} == {"static:entry_proof_not_required"}
    assert {item.detail.split(" names ")[1].split(",")[0] for item in found} == set(
        cells.RECORD_PROOFS
    )


def _difference(document: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in document["declared_differences"] if item["id"] == name)


def test_h03_declared_difference_rules() -> None:
    shipped = {item.id: item for item in LEDGER.differences}
    hosted = matrix_columns.HOSTED_GATEWAY_KIND
    in_force = f"{hosted}/S04/primary/raw"

    # Those of today: one a declared value selects, three in force on listed
    # cells.
    assert set(shipped) == {
        D_STRUCT_UNQUALIFIED,
        D_OAI_SCHEMA_FINAL,
        D_CDX_SERVED,
        D_COST_BOUND_UNRANKED,
    }
    assert len(shipped) == LEDGER.max_declared_differences
    assert shipped[D_STRUCT_UNQUALIFIED].cells == ()
    assert LEDGER.differences_for(in_force) == (D_OAI_SCHEMA_FINAL,)
    assert LEDGER.differences_for(f"{hosted}/S03/primary/raw") == ()
    assert cells.world().difference_variants == {D_STRUCT_UNQUALIFIED}
    # One is a scenario's own; one belongs to a wire and so to every
    # scenario that asserts a result: not the canary, and not the admission
    # scenario, which calls nothing. One belongs to the routes that opt into
    # a cost upper bound and so to every such scenario that runs guarded.
    results = {
        name for name, item in SCENARIOS.items() if not item.canary and not item.admission
    }
    assert cells.world().difference_alternatives == {
        D_OAI_SCHEMA_FINAL: {"S04"},
        D_CDX_SERVED: results,
        D_COST_BOUND_UNRANKED: {name for name in results if GUARDED in SCENARIOS[name].lenses},
    }
    # An admission cell is played on a path of its own, which applies no
    # declared difference, so its scenario defines no alternative and no
    # list may name such a cell (below); the runner refuses one that does.
    for name, item in SCENARIOS.items():
        if item.canary or item.admission:
            assert item.alternatives == {}, name
    admission = next(cell for cell in TOY_CELLS if SCENARIOS[cell.scenario].admission)
    with pytest.raises(MatrixSetupError, match="listed for an admission cell"):
        runner.execute(admission, Path("unused"), differences=(D_CDX_SERVED,))

    # A scenario may assert an alternative only when the ledger declares it...
    for name in shipped:
        undeclared = _document()
        undeclared["declared_differences"].remove(_difference(undeclared, name))
        assert _problem_codes(undeclared) == {"static:difference_undeclared"}
    # ...and the ledger may declare only what a scenario asserts.
    unused = _document()
    unused["ceilings"]["max_declared_differences"] += 1
    unused["declared_differences"].append(
        {"id": "D-EXAMPLE", "citation": "c", "owner": "o", "date": "2026-11-18", "alternative": "a"}
    )
    assert _problem_codes(unused) == {"static:difference_unused"}
    twice = _document()
    twice["ceilings"]["max_declared_differences"] += 1
    twice["declared_differences"].append(copy.deepcopy(_difference(twice, D_OAI_SCHEMA_FINAL)))
    assert _problem_codes(twice) == {"static:difference_duplicate"}

    def scoped(name: str, listed: list[str]) -> set[str]:
        document = _document()
        _difference(document, name)["cells"] = listed
        return _problem_codes(document)

    # In force on no cell, it asserts nothing.
    assert scoped(D_OAI_SCHEMA_FINAL, []) == {"static:difference_unused"}
    # A listed cell is a generated cell of a real kind whose scenario defines
    # the alternative; never a toy cell, never the canary, never listed twice.
    canary = next(cell.id for cell in TOY_CELLS if SCENARIOS[cell.scenario].canary)
    toy_cell = next(cell.id for cell in TOY_CELLS if cell.scenario == "S04")
    for wrong in (
        toy_cell,
        canary,
        f"{hosted}/S00/primary/raw",
        f"{hosted}/S03/primary/raw",
        "kimi/S04/blocked/raw",
        f"{hosted}/S04/tertiary/raw",
    ):
        assert scoped(D_OAI_SCHEMA_FINAL, [in_force, wrong]) == {
            "static:difference_cell_invalid"
        }, wrong
    assert scoped(D_OAI_SCHEMA_FINAL, [in_force, in_force]) == {"static:difference_cell_invalid"}
    # One that a declared value selects has no cell list of its own.
    assert scoped(D_STRUCT_UNQUALIFIED, [in_force]) == {"static:difference_cell_invalid"}
    # One that belongs to the wire of a kind is in force on every result cell
    # of that kind: where the wire names no served model the identity clause
    # compares none, so a cell the list left out would have the field checked
    # by nothing. It is in force on no cell of another kind.
    wire = cells.world().wire_differences
    served = list(shipped[D_CDX_SERVED].cells)
    assert wire == {D_CDX_SERVED: matrix_columns.UNNAMED_SERVED_MODEL_KINDS}
    assert {ledger.parse_cell_id(cell)[0] for cell in served} == set(wire[D_CDX_SERVED])
    for left_out in served:
        assert scoped(D_CDX_SERVED, [cell for cell in served if cell != left_out]) == {
            "static:difference_cell_missing"
        }, left_out
    assert scoped(D_CDX_SERVED, [*served, f"{hosted}/S01/primary/raw"]) == {
        "static:difference_cell_invalid"
    }
    # Without the rule that list is taken: the cell is generated, of a real
    # kind, not the canary, and its scenario defines the alternative.
    assert _problem_codes(wire_differences={}) == set()
    document = _document()
    _difference(document, D_CDX_SERVED)["cells"] = served[1:]
    assert _problem_codes(document, wire_differences={}) == set()
    # No list may name an admission cell: nothing could assert it there.
    assert scoped(D_CDX_SERVED, [*served, "kimi/S51/blocked/raw"]) == {
        "static:difference_cell_invalid"
    }
    # One that a clause asserts itself, for the routes a closed extra of the
    # routing file selects, is in force on exactly the cells of those routes:
    # the clause decides by the routing file alone, so the list has to say
    # where that is. Today those are the guarded result cells of the two
    # routes that opt into the cost upper bound.
    selected = cells.world().route_differences
    unranked = list(shipped[D_COST_BOUND_UNRANKED].cells)
    assert set(selected) == {D_COST_BOUND_UNRANKED}
    assert set(unranked) == selected[D_COST_BOUND_UNRANKED] == cells.cost_bounded_guarded_cells()
    # S01, S03, S04 and the turn limit, on the two diagnostic columns.
    assert len(unranked) == 8 and all(
        ledger.parse_cell_id(cell)[2:] in (("primary", GUARDED), ("secondary", GUARDED))
        and declarations.cost_bounded(
            column,
            matrix_columns.ROLE_BY_CAPABILITY[SCENARIOS[ledger.parse_cell_id(cell)[1]].capability],
        )
        for cell in unranked
        for column in matrix_columns.real_columns()
        if column.id == "/".join(ledger.parse_cell_id(cell)[::2])
    )
    for left_out in unranked:
        assert scoped(D_COST_BOUND_UNRANKED, [cell for cell in unranked if cell != left_out]) == {
            "static:difference_cell_missing"
        }, left_out
    # A guarded cell of a route without the opt-in, a raw cell of one with it.
    for elsewhere in ("kimi/S01/blocked/guarded", f"{hosted}/S01/primary/guarded"):
        assert scoped(D_COST_BOUND_UNRANKED, [*unranked, elsewhere]) == {
            "static:difference_cell_invalid"
        }, elsewhere
    assert scoped(D_COST_BOUND_UNRANKED, [*unranked, "kimi/S01/primary/raw"]) == {
        "static:difference_cell_invalid"
    }
    # A new guarded cell of such a route is owed in the list from the day it
    # is generated: it would be held to the alternative without the ledger
    # saying so.
    wider = {D_COST_BOUND_UNRANKED: selected[D_COST_BOUND_UNRANKED] | {"kimi/S01/blocked/guarded"}}
    assert _problem_codes(route_differences=wider) == {"static:difference_cell_missing"}
    # A route that stops opting in leaves the difference in force on no cell.
    assert _problem_codes(route_differences={D_COST_BOUND_UNRANKED: frozenset()}) == {
        "static:difference_cell_invalid"
    }
    # Without the rule a shorter list is taken.
    document = _document()
    _difference(document, D_COST_BOUND_UNRANKED)["cells"] = unranked[1:]
    assert _problem_codes(document, route_differences={}) == set()
    # A scenario's own difference stays where the ledger lists it.
    assert D_OAI_SCHEMA_FINAL not in wire and D_OAI_SCHEMA_FINAL not in selected
    malformed = _document()
    _difference(malformed, D_OAI_SCHEMA_FINAL)["cells"] = ["S04"]
    with pytest.raises(MatrixSetupError, match="cell id"):
        ledger.parse(malformed)


def test_an_alternative_is_asserted_positively_and_replaces_no_clause(tmp_path: Path) -> None:
    column = next(
        item
        for item in cells.toy_columns()
        if item.features is not None and item.features.structured_output == "prompt_only"
    )
    cell = Cell(column, "S04", "raw")
    in_force = (D_OAI_SCHEMA_FINAL,)
    untold = {"flag:token_usage_complete=False", "flag:cost_complete=False"}

    def codes(name: str, *faults: str, differences: tuple[str, ...] = in_force) -> frozenset[str]:
        return runner.execute(
            cell,
            tmp_path / name,
            driver=toy.ToyDriver(frozenset(faults)),
            differences=differences,
        ).report.codes

    # The alternative holds: one further request without tools carries the
    # schema, its answer is the final message, its usage is counted.
    assert codes("honest", toy.FINAL_SCHEMA_REQUEST) == frozenset()
    # It is asserted, not waived: an adapter that does not behave so fails it,
    # and so does one that carries the schema on its tool-bearing requests
    # (and, with that fault, no schema text in the prompt: the other code).
    assert codes("absent") == {"request.content:schema"}
    # Where no finalisation was sent, the answer is still the draft's: an
    # adapter that returns another valid object instead is told that too.
    assert codes("absent and replaced", toy.SCHEMA_ANSWER_REPLACED) == {
        "request.content:schema",
        "text:mismatch",
    }
    assert codes("misplaced", toy.FINAL_SCHEMA_REQUEST, toy.SCHEMA_MISPLACED) == {
        "request.content:schema",
        "request.content:schema_prompt",
    }
    # It replaces no clause: the accounting of the further request is owed.
    assert codes("untold", toy.FINAL_SCHEMA_REQUEST, toy.CLEAR_ACCOUNTING_FLAGS) == untold
    # Where the ledger does not put it in force, the same request is extra.
    assert "request.extra" in codes("elsewhere", toy.FINAL_SCHEMA_REQUEST, differences=())
    # It restates where the schema field travels and whose answer is returned,
    # nothing else: the prompt half of the declared value stays (this column
    # declares the schema in the prompt), and the answer is the finalisation's.
    assert codes("no schema text", toy.FINAL_SCHEMA_REQUEST, toy.SCHEMA_NOT_IN_PROMPT) == {
        "request.content:schema_prompt"
    }
    assert codes("the draft", toy.FINAL_SCHEMA_REQUEST, toy.RETURN_SCHEMA_DRAFT) == {
        "text:mismatch"
    }
    plan = plan_for(
        SCENARIOS["S04"],
        declarations.select(SCENARIOS["S04"].differences, declarations.routed(column)),
        in_force,
    )
    assert isinstance(plan.script.tail, Say) and plan.script.tail.text == SCHEMA_FINAL_ANSWER
    assert json.loads(SCHEMA_FINAL_ANSWER) != json.loads(SCHEMA_ANSWER)
    # A route that declares no schema text in the prompt owes none there: on
    # those columns the honest play of the alternative sends none and passes.
    for value in ("native", "unsupported"):
        other = next(
            item
            for item in cells.toy_columns()
            if item.features is not None and item.features.structured_output == value
        )
        played = runner.execute(
            Cell(other, "S04", "raw"),
            tmp_path / value,
            driver=toy.ToyDriver(frozenset({toy.FINAL_SCHEMA_REQUEST})),
            differences=in_force,
        )
        assert played.report.codes == frozenset(), value
        assert SCHEMA_TITLE not in played.observation.provider.requests[0].system

    scenario = SCENARIOS["S04"]
    with pytest.raises(MatrixSetupError, match="defines no alternative"):
        plan_for(SCENARIOS["S03"], {}, in_force)
    changes: tuple[dict[str, Any], ...] = (
        {"expect": Expectation(kind="transport")},
        {"max_turns": 1},
        {"script": ProviderScript((Say("another exchange"),))},
    )
    for change in changes:

        def widened(plan: Any, change: dict[str, Any] = change) -> Any:
            return dataclasses.replace(plan, **change)

        wider = dataclasses.replace(scenario, alternatives={"D-WIDER": widened})
        with pytest.raises(MatrixSetupError, match="never stands in for a contract clause"):
            plan_for(wider, {}, ("D-WIDER",))


def test_a_known_gap_on_the_prompt_half_does_not_hide_a_wrong_finalisation_request(
    tmp_path: Path,
) -> None:
    column = next(
        item
        for item in cells.toy_columns()
        if item.features is not None and item.features.structured_output == "prompt_only"
    )
    cell = Cell(column, "S04", "raw")

    def played(name: str, fault: str) -> runner.Execution:
        return runner.execute(
            cell,
            tmp_path / name,
            driver=toy.ToyDriver(frozenset({toy.FINAL_SCHEMA_REQUEST, fault})),
            differences=(D_OAI_SCHEMA_FINAL,),
        )

    def shape(execution: runner.Execution) -> list[tuple[bool, bool]]:
        requests = execution.observation.provider.requests
        assert SCHEMA_TITLE not in requests[0].system
        return [(bool(item.tools), item.schema_field) for item in requests]

    # The gap as a column shows it: the finalisation request is right, and the
    # schema text the route declares is missing from the prompt. A ledger that
    # records exactly that makes the cell an expected failure ...
    gap = played("the gap", toy.SCHEMA_NOT_IN_PROMPT)
    assert shape(gap) == [(True, False), (True, False), (False, True)]
    assert gap.report.codes
    known = _single_entry_ledger(cell.id, *sorted(gap.report.codes), observed=gap.current)
    with pytest.raises(XFAILED):
        ledger.compare(known, cell.id, gap.report.codes, current=gap.current)
    # ... and covers nothing else. With the same gap, an adapter that carries
    # the schema field on its tool-bearing requests and not on the final one
    # shows another set of codes: the two halves of the placement have a code
    # each, so an entry for the one cannot stand in for the other.
    wrong = played("the gap and a wrong request", toy.SCHEMA_MISPLACED)
    assert shape(wrong) == [(True, True), (True, True), (False, False)]
    assert wrong.report.codes > gap.report.codes
    with pytest.raises(ledger.LedgerMismatch, match="different deviation"):
        ledger.compare(known, cell.id, wrong.report.codes, current=wrong.current)


def _single_entry_ledger(
    cell: str, *codes: str, observed: Mapping[str, str] | None = None
) -> ledger.Ledger:
    under = dict(observed or {})
    record = {
        "title": "a known gap",
        "origin": "gap list",
        "severity": "medium",
        "cells": [{"cell": cell, "observed_under": under, "closes_under": under}],
    }
    return ledger.parse(
        {
            "schema_version": ledger.LEDGER_SCHEMA,
            "ceilings": {"max_known_gaps": {"pack_a": 1}, "max_declared_differences": 0},
            "registry": {"G99": record},
            "known_gaps": [
                {
                    "cell": cell,
                    "observed_under": under,
                    "codes": {code: "G99" for code in codes},
                }
            ],
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


def test_h03_a_column_wide_pin_names_a_cell_that_could_owe_the_gap() -> None:
    # A column-wide entry of this test's own, on a kind that is driven today,
    # whatever the shipped ledger holds column-wide.
    driven = [cell for cell in cells.behaviour_cells() if cell.column.kind != TOY_KIND]
    kind = driven[0].column.kind
    of_kind = [cell for cell in driven if cell.column.kind == kind]
    plain = next(cell.id for cell in of_kind if not SCENARIOS[cell.scenario].canary)
    canary = next(cell.id for cell in of_kind if SCENARIOS[cell.scenario].canary)
    code = "secret:leak"
    document = _document()
    document["ceilings"]["max_known_gaps"]["pack_a"] += 1
    document["registry"]["G02"]["column_wide"] = [{"kind": kind, "code": code}]
    document["known_gaps"].append(
        {
            "column_wide": {"kind": kind, "code": code, "gap": "G02"},
            # Whichever kind is driven first: on a reference kind an entry
            # needs the owner's ruling.
            "ruling": "synthetic",
        }
    )

    def pinned(*rows: tuple[str, str]) -> set[str]:
        return _problem_codes(
            document,
            column_wide_not_owed=tuple(
                {"cell": cell, "code": pin, "reason": "synthetic"} for cell, pin in rows
            ),
        )

    assert pinned() == pinned((plain, code)) == set()
    invalid = {"static:column_wide_pin_invalid"}
    # Listed once; a generated behaviour cell; not the canary; of a kind and
    # for a code that a column-wide entry names.
    assert pinned((plain, code), (plain, code)) == invalid
    assert pinned((plain.replace("/S", "/S9"), code)) == invalid
    static = ledger.cell_id(kind, "H02", ledger.NO_ROLE, ledger.STATIC_LENS)
    assert pinned((static, code)) == invalid
    assert pinned((canary, code)) == invalid
    assert pinned((plain, "pricing:unreconciled")) == invalid
    assert pinned((next(cell.id for cell in TOY_CELLS if cell.scenario == "S01"), code)) == invalid
    # Without the column-wide entry the same pin has nothing to exempt.
    assert _problem_codes(
        column_wide_not_owed=({"cell": plain, "code": code, "reason": "synthetic"},)
    ) == invalid


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
    assert fields == (
        "event_detail",
        "structured_output",
        "turn_limit",
        "usd_limit",
        "tool_loop",
        "response_token_limit",
        "session_continuity",
    )
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


# -- the admission scenario, and what a route asks of an invocation ---------------


def test_an_admission_cell_calls_nothing_and_is_held_to_its_subjects(tmp_path: Path) -> None:
    cell = next(cell for cell in TOY_CELLS if SCENARIOS[cell.scenario].admission)
    scenario = SCENARIOS[cell.scenario]
    admitted = contract.ADMITTED

    # Its clauses follow from its definition: backends are built, none is called.
    assert contract.applicable(
        lens="raw", toolset=False, expects_result=False, call=False
    ) == ("C0", "C8", "C9")
    assert (scenario.lenses, scenario.toolset, scenario.governing) == (("raw",), None, ())
    # A toy column stands behind no routing gate, and no catalog names it.
    expected = cells.expected_admission(cell)
    assert expected == dict.fromkeys(matrix_columns.ROLE_BY_CAPABILITY, admitted)

    honest = runner.execute(cell, tmp_path / "honest")
    assert honest.report.codes == frozenset()
    assert set(honest.report.outcomes) == {"C0", "C8", "C9", contract.SCENARIO}
    assert honest.observation.result is None and honest.observation.raised is None
    assert honest.observation.provider.request_count == 0
    # Held to another pin, a subject is named with what was seen and what is
    # pinned; so is a pinned subject the cell never saw, and one it saw that
    # the pin does not name.
    refusal = "FeatureAdmissionError"

    def held_to(name: str, pin: Mapping[str, str]) -> frozenset[str]:
        return runner.execute(cell, tmp_path / name, admission=pin).report.codes

    assert held_to("refused", {**expected, "read_only": refusal}) == {
        f"admission:read_only:{admitted}!={refusal}"
    }
    assert held_to("unseen", {**expected, cells.AVAILABILITY: "blocked"}) == {
        "admission:availability:unseen!=blocked"
    }
    assert held_to("unpinned", {"text": admitted}) == {
        f"admission:read_only:{admitted}!=unpinned",
        f"admission:workspace_write:{admitted}!=unpinned",
    }
    for code in ("admission:text:admitted!=FeatureAdmissionError", "admission:a:b!=c"):
        assert contract.clause_of(code) == contract.SCENARIO


class _LogsAtBuild(toy.ToyDriver):
    """A toy driver that logs one warning while it builds the text role."""

    def __init__(self, logger: str, words: str, *, then_refuses: bool = False) -> None:
        super().__init__()
        self._logger, self._words, self._refuses = logger, words, then_refuses

    @contextlib.contextmanager
    def open(self, column: Column, *, capability: str, **arguments: Any) -> Iterator[Any]:
        with super().open(column, capability=capability, **arguments) as bound:
            if capability == "text":
                logging.getLogger(self._logger).warning(
                    "routing: %s a route: a reason", self._words
                )
                if self._refuses:
                    raise FeatureAdmissionError("refused after the warning")
            yield bound


def test_an_admission_cell_tells_a_route_that_routing_retained_from_one_it_admitted(
    tmp_path: Path,
) -> None:
    cell = next(cell for cell in TOY_CELLS if SCENARIOS[cell.scenario].admission)
    expected = cells.expected_admission(cell)
    admitted, retained = contract.ADMITTED, contract.RETAINED
    logger, words = matrix_columns.ROUTING_LOGGER, matrix_columns.RETAINED_ROUTE

    def played(
        name: str, driver: toy.ToyDriver | None = None, pin: Mapping[str, str] = expected
    ) -> frozenset[str]:
        return runner.execute(cell, tmp_path / name, driver=driver, admission=pin).report.codes

    # The warning of the routing module during the build of a role says that
    # the role was built although the profile of its route does not meet it.
    # The build that logged it is retained, and no build after it.
    assert played("retained", _LogsAtBuild(logger, words)) == {
        f"admission:text:{retained}!={admitted}"
    }
    # Held to a pin that says so the same play is clean, and an honest build
    # is not: a profile that comes to meet its role changes the pin.
    pinned = {**expected, "text": retained}
    assert played("pinned", _LogsAtBuild(logger, words), pinned) == frozenset()
    assert played("met", None, pinned) == {f"admission:text:{admitted}!={retained}"}
    # The same words under another logger are not that warning, and neither
    # is another warning of the routing module.
    assert played("other-logger", _LogsAtBuild(f"{logger}.other", words)) == frozenset()
    assert played("other-words", _LogsAtBuild(logger, "ignored a setting of")) == frozenset()
    # A role that is refused after the warning is refused.
    assert played("refused", _LogsAtBuild(logger, words, then_refuses=True)) == {
        f"admission:text:FeatureAdmissionError!={admitted}"
    }
    # A cell that cannot hear the routing module could not tell the two
    # apart, and says so: where the warning would not reach the root logger
    # the play is a setup error, never a role read as admitted.
    routing = logging.getLogger(logger)
    was = (routing.level, routing.propagate, logging.root.manager.disable)
    mutes = {
        "level": lambda: routing.setLevel(logging.ERROR),
        "disabled": lambda: logging.disable(logging.WARNING),
        "no-propagation": lambda: setattr(routing, "propagate", False),
    }
    for name, mute in mutes.items():
        try:
            mute()
            with pytest.raises(MatrixSetupError, match="would not reach the cell"):
                played(name, _LogsAtBuild(logger, words))
        finally:
            routing.setLevel(was[0])
            routing.propagate = was[1]
            logging.disable(was[2])
    assert played("heard", _LogsAtBuild(logger, words), pinned) == frozenset()
    # The logger is that of the module routing lives in. That the words are
    # those of its warning is held where the product logs it: by an
    # admission cell of a real column that is pinned retained. Without one,
    # nothing would hold the words to the product.
    assert logger == load_routing.__module__
    assert any(
        row[capability] == retained
        for row in cells.load_inventory().admission
        for capability in matrix_columns.ROLE_BY_CAPABILITY
    )


class _BoundByItsDriver(toy.ToyDriver):
    """A toy route that is built only with what its driver binds to it."""

    def __init__(
        self, field: str, *, admits_as_shipped: bool = False, keeps_its_overlay: bool = False
    ) -> None:
        super().__init__()
        self._field, self._admits, self._keeps = field, admits_as_shipped, keeps_its_overlay

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        if column.shipped and not self._keeps:
            return {}
        return {self._field: "bound"}

    @contextlib.contextmanager
    def open(self, column: Column, **arguments: Any) -> Iterator[Any]:
        if not self.overlay(column, {}) and not self._admits:
            raise FeatureAdmissionError(f"the toy route of column {column.id} is not bound")
        with super().open(column, **arguments) as bound:
            yield bound


def test_an_admission_cell_asks_again_on_the_file_as_shipped_where_its_driver_binds_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = next(cell for cell in TOY_CELLS if SCENARIOS[cell.scenario].admission)
    (field,) = matrix_columns.ADMISSION_OVERLAY
    refusal = "FeatureAdmissionError"
    bound = cells.expected_admission(cell)
    shipped = {cells.as_shipped(name): refusal for name in matrix_columns.ROLE_BY_CAPABILITY}
    both = {**bound, **shipped}

    def played(name: str, driver: toy.ToyDriver, pin: Mapping[str, str]) -> frozenset[str]:
        return runner.execute(cell, tmp_path / name, driver=driver, admission=pin).report.codes

    # The driver of this kind may add nothing that routing reads to admit a
    # route, so every capability is asked for once: a pin that states the
    # file as shipped names subjects the cell never saw.
    assert matrix_columns.admission_inputs(TOY_KIND) == ()
    assert cells.admission_subjects(TOY_KIND) == cells.ADMISSION_SUBJECTS
    assert played("once", _BoundByItsDriver(field), both) == {
        f"admission:{subject}:unseen!={refusal}" for subject in shipped
    }

    # A kind whose driver binds such a field (and the path of a peer, which
    # is none) is asked a second time on the file as shipped.
    peer = "extra.cli_path"
    assert peer in matrix_columns.ALLOWED_OVERLAY[matrix_columns.CLAUDE_KIND]
    monkeypatch.setitem(matrix_columns.ALLOWED_OVERLAY, TOY_KIND, (peer, field))
    assert matrix_columns.admission_inputs(TOY_KIND) == (field,)
    assert cells.admission_subjects(TOY_KIND) == (*cells.ADMISSION_SUBJECTS, *shipped)
    honest = runner.execute(
        cell, tmp_path / "honest", driver=_BoundByItsDriver(field), admission=both
    )
    assert honest.report.codes == frozenset()
    assert set(honest.report.outcomes) == {"C0", "C8", "C9", contract.SCENARIO}
    assert honest.observation.provider.request_count == 0
    assert honest.observation.debug_dump_text.count("is not bound") == len(shipped)
    # A file that routing admits as shipped, where the pin says it is
    # refused: the gate opened, and the pin changes with it.
    assert played("opened", _BoundByItsDriver(field, admits_as_shipped=True), both) == {
        f"admission:{subject}:{contract.ADMITTED}!={refusal}" for subject in shipped
    }
    # A pin that does not state the second play leaves it unpinned.
    assert played("unpinned", _BoundByItsDriver(field), bound) == {
        f"admission:{subject}:{refusal}!=unpinned" for subject in shipped
    }
    # A driver that binds its input to the file it is asked to read as
    # shipped would play the first build twice under another name.
    with pytest.raises(MatrixSetupError, match="asked to build from as shipped"):
        played("kept", _BoundByItsDriver(field, keeps_its_overlay=True), both)

    # The view names the same column and the same routing file.
    column = matrix_columns.real_columns()[0]
    view = column.routed_by(matrix_columns.COLUMN_PRODUCT).as_shipped()
    assert view.shipped and not column.shipped
    assert dataclasses.replace(view, shipped=False) == column.routed_by(
        matrix_columns.COLUMN_PRODUCT
    )


def test_an_admission_cell_of_a_real_column_needs_its_pin_and_a_pin_its_cell() -> None:
    blocked = _standin("blocked")
    generated = cells.generate(
        [_standin("primary"), blocked],
        registered=frozenset({"standin"}),
        enablement={"standin": (STARTER,)},
    )
    (cell,) = (item for item in generated if SCENARIOS[item.scenario].admission)
    assert cell.column is blocked
    row = {
        "cell": cell.id,
        "availability": "blocked",
        "text": "admitted",
        "read_only": "admitted",
        "workspace_write": "FeatureAdmissionError",
        "reason": "synthetic",
    }

    assert cells.admission_pin_problems(generated, [row]) == []
    (missing,) = cells.admission_pin_problems(generated, [])
    assert "has no pin" in missing
    (stale,) = cells.admission_pin_problems(
        generated, [row, {**row, "cell": "standin/S51/primary/raw"}]
    )
    assert "no generated admission cell" in stale
    # A toy cell is never pinned: it is held to the rule, not to a row.
    toy_cell = next(item for item in TOY_CELLS if SCENARIOS[item.scenario].admission)
    (on_toy,) = cells.admission_pin_problems(TOY_CELLS, [{**row, "cell": toy_cell.id}])
    assert "no generated admission cell of a real column" in on_toy
    assert cells.admission_pin_problems(TOY_CELLS, []) == []
    # What the cell is held to is its row, whole.
    assert cells.expected_admission(cell, [row]) == {
        name: row[name] for name in cells.ADMISSION_SUBJECTS
    }
    with pytest.raises(MatrixSetupError, match="has no pin"):
        cells.expected_admission(cell, [])
    # The shipped inventory pins every admission cell that is generated
    # today, and without a pin the cells of the matrix are not generated at
    # all: a collection error, never a cell that passes on nothing.
    shipped = cells.load_inventory()
    assert cells.admission_pin_problems(cells.behaviour_cells(), shipped.admission) == []
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(
            cells, "load_inventory", lambda: dataclasses.replace(shipped, admission=())
        )
        with pytest.raises(MatrixSetupError, match="has no pin in inventory.yaml"):
            cells.behaviour_cells.__wrapped__()
    # A column-wide gap of a result clause is not owed on an admission cell:
    # the cell calls nothing, by its definition, so it needs no pin for that.
    known = _ledger(column_wide=True)
    (scope,) = known.column_wide
    assert contract.clause_of(scope.code) in RESULT_CLAUSES
    plain_cell = ledger.cell_id(scope.kind, "S01", "primary", "raw")
    admission_cell = ledger.cell_id(scope.kind, cell.scenario, "primary", "raw")
    assert matrix_verify.ledgered(known, plain_cell)[1] == {scope.gap}
    assert matrix_verify.ledgered(known, admission_cell) == (set(), set())

    # The cell runs on the routing file of every product: a column as one
    # product routes it names that product's file where a driver builds from.
    column = matrix_columns.real_columns()[0]
    files = {product: column.routing_path(product) for product in matrix_columns.PRODUCTS}
    assert len(set(files.values())) == len(matrix_columns.PRODUCTS) > 1
    for product, path in files.items():
        view = column.routed_by(product)
        assert view.routing == ((matrix_columns.COLUMN_PRODUCT, path),)
        assert (view.id, view.profile, view.model) == (column.id, column.profile, column.model)
    # The role table and the catalogs agree on which columns are blocked.
    for real in matrix_columns.real_columns():
        words = {
            matrix_columns.catalog_availability(real, product)
            for product in matrix_columns.PRODUCTS
        }
        assert len(words) == 1, real.id
        assert (words == {"blocked"}) == real.role.startswith("blocked"), real.id

    inventory = yaml.safe_load(cells.INVENTORY_PATH.read_text(encoding="utf-8"))
    assert cells.parse_inventory({**inventory, "admission": [row]}).admission == (row,)
    malformed: tuple[Any, ...] = (
        {name: value for name, value in row.items() if name != "reason"},
        {name: value for name, value in row.items() if name != "text"},
        {**row, "read_only": " "},
        {**row, "lens": "raw"},
        row["cell"],
    )
    for wrong in malformed:
        with pytest.raises(MatrixSetupError, match="admission row"):
            cells.parse_inventory({**inventory, "admission": [wrong]})
    with pytest.raises(MatrixSetupError, match="cell id"):
        cells.parse_inventory({**inventory, "admission": [{**row, "cell": "S51"}]})
    with pytest.raises(MatrixSetupError, match="pinned twice"):
        cells.parse_inventory({**inventory, "admission": [row, dict(row)]})
    # A row of a kind whose driver binds an admission input states each
    # capability on the file as shipped too; a row of another kind may not,
    # and a cell of such a kind is held to the whole row.
    bound_kind = next(
        kind for kind in matrix_columns.ALLOWED_OVERLAY if matrix_columns.admission_inputs(kind)
    )
    as_shipped = {
        cells.as_shipped(name): "FeatureAdmissionError"
        for name in matrix_columns.ROLE_BY_CAPABILITY
    }
    bound_cell = Cell(dataclasses.replace(_standin("primary"), kind=bound_kind), "S51", "raw")
    bound_row = {**row, "cell": bound_cell.id, **as_shipped}
    assert set(as_shipped).isdisjoint(cells.ADMISSION_SUBJECTS)
    assert cells.parse_inventory({**inventory, "admission": [bound_row]}).admission == (
        bound_row,
    )
    assert cells.expected_admission(bound_cell, [bound_row]) == {
        name: value for name, value in bound_row.items() if name not in ("cell", "reason")
    }
    for wrong in (
        {**row, "cell": bound_cell.id},
        {**row, **as_shipped},
        {name: value for name, value in bound_row.items() if name != cells.as_shipped("text")},
    ):
        with pytest.raises(MatrixSetupError, match="admission row"):
            cells.parse_inventory({**inventory, "admission": [wrong]})
    with pytest.raises(MatrixSetupError, match="unknown shape"):
        cells.parse_inventory(
            {name: value for name, value in inventory.items() if name != "admission"}
        )


def test_a_route_budget_comes_from_the_routing_file_and_a_scenario_budget_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = matrix_columns.COST_BOUND_EXTRA
    text_role = matrix_columns.ROLE_BY_CAPABILITY["text"]
    plain = next(cell for cell in TOY_CELLS if cell.scenario == "S01" and cell.lens == "raw")
    overrun = next(
        cell
        for cell in TOY_CELLS
        if cell.scenario == "S13"
        and declarations.routed(cell.column).usd_limit == "post_hoc"
    )

    # A column without a routing file asks for nothing.
    assert declarations.route_extra(plain.column, text_role, extra) is None
    assert declarations.route_budget(plain.column, text_role) is None
    assert not declarations.cost_bounded(plain.column, text_role)
    unbounded = runner.execute(plain, tmp_path / "none")
    assert unbounded.route_budget is None and not unbounded.reference.cost_bounded
    # The routes that opt in, read from the shipped routing files, are the
    # ones that get the matrix's route budget, on every role a cell builds.
    asked = {
        column.id: {
            declarations.route_budget(column, role)
            for role in matrix_columns.ROLE_BY_CAPABILITY.values()
        }
        for column in matrix_columns.real_columns()
    }
    assert set(map(frozenset, asked.values())) == {
        frozenset({None}),
        frozenset({matrix_columns.ROUTE_BUDGET_USD}),
    }
    for column in matrix_columns.real_columns():
        opted = {
            declarations.route_extra(column, role, extra)
            for role in matrix_columns.ROLE_BY_CAPABILITY.values()
        }
        assert asked[column.id] == (
            {matrix_columns.ROUTE_BUDGET_USD} if opted == {True} else {None}
        ), column.id
        # The same extra says that the results of the route are cost intervals.
        assert {
            declarations.cost_bounded(column, role)
            for role in matrix_columns.ROLE_BY_CAPABILITY.values()
        } == {opted == {True}}, column.id
        # A file that opts in names one cost policy for its callers, the same
        # on every role, and the product ships a price row for its model.
        named = {
            declarations.cost_policy(column, role)
            for role in matrix_columns.ROLE_BY_CAPABILITY.values()
        }
        assert len(named) == 1 and (named != {None}) == (opted == {True}), column.id
        if opted == {True}:
            assert matrix_columns.shipped_price(
                declarations.routed_model(column, text_role)
            ) not in (None, matrix_columns.MATRIX_RATE), column.id

    # The runner passes it: a budget so small that the plain reply overruns
    # it ends the plain cell as a budget stop, which the scenario does not
    # expect ...
    monkeypatch.setattr(declarations, "route_budget", lambda *_args, **_more: 1e-9)
    budgeted = runner.execute(plain, tmp_path / "route")
    assert budgeted.route_budget == 1e-9
    assert "kind:budget_exhausted!=none" in budgeted.report.codes
    # ... and a scenario that sets a budget of its own keeps it: with the
    # route budget the overrun of that scenario would be no overrun at all.
    monkeypatch.setattr(declarations, "route_budget", lambda *_args, **_more: 1000.0)
    own = runner.execute(overrun, tmp_path / "own")
    assert own.route_budget is None and own.plan.budget_usd == BUDGET_USD
    assert own.report.codes == frozenset()
    # The runner also tells the contract that the route's results are cost
    # intervals: the toy returns none, and its honest cell says so.
    with monkeypatch.context() as patched:
        patched.setattr(declarations, "cost_bounded", lambda *_args, **_more: True)
        interval_owed = runner.execute(plain, tmp_path / "bounded")
    assert interval_owed.reference.cost_bounded
    assert interval_owed.report.codes == {"usage.cost_bounds:none"}
    # A column without a routing file names no policy and has no shipped row.
    assert interval_owed.reference.cost_policy is None
    assert interval_owed.reference.bound_pricing is None
    assert matrix_columns.shipped_price(matrix_columns.TOY_MODEL) is None
    # A routing file that opts in and does not say which policy its callers
    # pass is an error, and so is a route that opts in for a model without a
    # shipped price row: the cell would be held to nothing.
    opted_in = next(
        column
        for column in matrix_columns.real_columns()
        if declarations.cost_bounded(column, text_role)
    )
    shipped_file = opted_in.routing_path(matrix_columns.COLUMN_PRODUCT)
    silent = tmp_path / shipped_file.name
    silent.write_text(
        shipped_file.read_text(encoding="utf-8").replace("--diagnostic-cost-policy", "--a-policy"),
        encoding="utf-8",
        newline="\n",
    )
    unnamed = dataclasses.replace(
        opted_in, routing=((matrix_columns.COLUMN_PRODUCT, silent),)
    )
    assert declarations.cost_bounded(unnamed, text_role)
    with pytest.raises(MatrixSetupError, match="names no cost policy"):
        declarations.cost_policy(unnamed, text_role)
    with monkeypatch.context() as patched:
        patched.setattr(matrix_columns, "shipped_price", lambda _model: None)
        with pytest.raises(MatrixSetupError, match="ships no price row"):
            runner.execute(Cell(opted_in, "S01", "raw"), tmp_path / "no-row")


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


# -- where a write ran, and what a writer is offered and told -------------------


def test_where_a_write_ran_is_judged_on_the_recorder_and_on_the_hosts(tmp_path: Path) -> None:
    honest = _honest(tmp_path, "S41")
    observation = honest.observation
    seen = observation.mutation
    name = BINDINGS[WRITE].name
    assert seen is not None and observation.write_applied is True
    (host,) = seen.hosts

    # The honest toy: the write is on disk and the recorder of this process
    # never saw it; a process other than this one recorded it as started, was
    # gone when the call had returned, and was bound to the call's deadline.
    assert all(item.name != name for item in observation.toolset_log)
    assert host.pid not in (None, os.getpid()) and host.started_writes == (name,)
    assert not host.alive_at_return and host.left_behind == ()
    own = seen.invoked_at + seen.timeout_seconds
    assert 0 <= host.deadline - own <= DEADLINE_SLACK_SECONDS
    # One fact changed at a time, each with its code and no other.
    here = (*observation.toolset_log, toolsets.Dispatch(name, {}, "OK"))
    assert _judge(honest, toolset_log=here) == {"mutation:in_process"}
    left = dataclasses.replace(seen, hosts=(dataclasses.replace(host, alive_at_return=True),))
    assert _judge(honest, mutation=left) == {"mutation:host_alive"}
    # The deadline of the host is the invocation's: the moment of the call
    # plus the policy's timeout, give or take the slack between two readings
    # of the clock. A host that is built into the call with a whole timeout
    # of its own is beyond it, however soon the call returned afterwards.
    def bound_to(deadline: float) -> frozenset[str]:
        hosts = (dataclasses.replace(host, deadline=deadline),)
        return _judge(honest, mutation=dataclasses.replace(seen, hosts=hosts))

    assert bound_to(own + DEADLINE_SLACK_SECONDS - 0.01) == frozenset()
    assert bound_to(own + DEADLINE_SLACK_SECONDS + 0.01) == {"mutation:deadline_unbound"}
    assert bound_to(own - 5.0) == frozenset()
    # The next case states its own times, a call of ten seconds whose host
    # was built two seconds into it; how long the honest call took on this
    # machine is no part of it.
    assert DEADLINE_SLACK_SECONDS < 2.0
    late_call = dataclasses.replace(seen, returned_at=seen.invoked_at + 10.0)
    fresh = dataclasses.replace(host, deadline=seen.invoked_at + 2.0 + seen.timeout_seconds)
    assert _judge(honest, mutation=dataclasses.replace(late_call, hosts=(fresh,))) == {
        "mutation:deadline_unbound"
    }
    # A host that is gone and left a process behind.
    orphaned = dataclasses.replace(seen, hosts=(dataclasses.replace(host, left_behind=(4242,)),))
    assert _judge(honest, mutation=orphaned) == {"mutation:left_behind"}
    # A host that ran no write of the call counts as well: left running, or
    # gone with something left behind.
    idle = dataclasses.replace(host, started_writes=(), alive_at_return=True)
    both = dataclasses.replace(seen, hosts=(host, idle))
    assert _judge(honest, mutation=both) == {"mutation:host_alive"}
    littering = dataclasses.replace(host, started_writes=(), left_behind=(4242, 4243))
    assert _judge(honest, mutation=dataclasses.replace(seen, hosts=(host, littering))) == {
        "mutation:left_behind"
    }
    # A write that is on disk and that nothing saw cannot be judged: that is
    # an error of the matrix, never a deviation and never a pass.
    nowhere = dataclasses.replace(seen, hosts=())
    silent = dataclasses.replace(seen, hosts=(dataclasses.replace(host, started_writes=()),))
    for unseen in (None, nowhere, silent):
        with pytest.raises(MatrixSetupError, match="cannot say where it ran"):
            _judge(honest, mutation=unseen)
    # A write that is not on disk is the missing dispatch of the clause about
    # tools, and nothing is said of its host.
    assert _judge(honest, write_applied=False, mutation=None) == {f"dispatch.missing:{WRITE}"}
    # A cell that attaches no write tool watches nothing.
    assert _honest(tmp_path / "read", "S03").observation.mutation is None


def test_the_watch_holds_every_host_built_in_the_process_and_changes_none(
    tmp_path: Path,
) -> None:
    classes = mutation.host_classes()
    product, own = classes
    name, arguments = BINDINGS[WRITE].name, dict(BINDINGS[WRITE].arguments)
    fixture = toolsets.build(WRITER(2), tmp_path / toolsets.WORKSPACE)
    before = {item: (item.__init__, item.start, item.stop) for item in classes}

    # The two classes: the product's shared host, a private name the module
    # lists, and the toy's own.
    assert own is toy.WriteHost
    assert (f"{product.__module__}.{product.__qualname__}",) == mutation.PRIVATE_SEAMS
    with mutation.watch() as seen:
        assert all(
            wrapped is not original
            for item in classes
            for wrapped, original in zip(
                (item.__init__, item.start, item.stop), before[item], strict=True
            )
        )
        host = toy.WriteHost(fixture.workspace, invocation_deadline=time.monotonic() + 30.0)
        # Built and not started: no process, nothing recorded.
        (built,) = seen.at_return()
        assert built.pid is None and not built.alive_at_return and built.started_writes == ()
        host.start()
        try:
            assert host.dispatch(name, arguments).startswith("OK")
            # Still running after the grace: reported as alive, with what it
            # recorded and the pid of a process that is not this one.
            (running,) = seen.at_return()
            assert running.alive_at_return and running.started_writes == (name,)
            assert running.pid == host.pid and running.pid not in (None, os.getpid())
        finally:
            host.stop()
        (stopped,) = seen.at_return()
        assert not stopped.alive_at_return and stopped.pid == running.pid
        assert running.left_behind == stopped.left_behind == ()
        # A host that is not built with the keyword of its deadline cannot be
        # held to one: an error of the matrix.
        with pytest.raises(MatrixSetupError, match="invocation_deadline"):
            seen.built(host, None)
    # The write ran through the real tool, in no process of the test.
    assert fixture.write_applied() and fixture.log == []
    # Outside the block the classes are as they were, and nothing is held.
    assert {item: (item.__init__, item.start, item.stop) for item in classes} == before
    with mutation.watch() as later:
        assert later.at_return() == ()


def _gone(pid: int, seconds: float = 10.0) -> bool:
    until = time.monotonic() + seconds
    while mutation._exists(pid) and time.monotonic() < until:
        time.sleep(0.02)
    return not mutation._exists(pid)


def test_the_watch_sees_what_a_stopped_host_left_behind(tmp_path: Path) -> None:
    fixture = toolsets.build(WRITER(2), tmp_path / toolsets.WORKSPACE)
    sleeper = [sys.executable, "-I", "-S", "-c", "import time; time.sleep(30)"]
    with mutation.watch() as seen:
        # Below the host when it was stopped. The lying host of the toy
        # starts a process in a session of its own; the kill of the host
        # does not reach it, and once the host is gone nothing names the
        # host as its parent, its process group or its session.
        host = toy.WriteHost(
            fixture.workspace,
            invocation_deadline=time.monotonic() + 30.0,
            leaves_a_process=True,
        )
        host.start()
        below = host.left
        try:
            assert below is not None and mutation._exists(below)
            assert os.getsid(below) == below != os.getsid(host.pid or 0)
            (running,) = seen.at_return()
            # While the host lives, what it started is its child.
            assert running.alive_at_return and running.left_behind == (below,)
            host.stop()
            (stopped,) = seen.at_return()
            assert not stopped.alive_at_return and stopped.left_behind == (below,)
            assert psutil.Process(below).ppid() != stopped.pid
        finally:
            host.stop()
            if below is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(below, signal.SIGKILL)
        assert below is not None and _gone(below)
        (cleared,) = seen.at_return()
        assert cleared.left_behind == ()

        # Of the host's process group and session. A host that is ended
        # otherwise than through its ``stop`` leaves no reading of what was
        # below it; what is still in the group and the session it led is
        # found by its pid. The stand-in is no host class: the watch is told
        # of it as the wrapped constructor and ``start`` tell it.
        leader = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                "import subprocess, sys, time\n"
                f"member = subprocess.Popen({sleeper!r})\n"
                "print(member.pid, flush=True)\n"
                "time.sleep(30)\n",
            ],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            assert leader.stdout is not None
            member = int(leader.stdout.readline())
            standin = SimpleNamespace(pid=leader.pid, started_write_names=())
            seen.built(standin, time.monotonic() + 30.0)
            seen.started(standin)
            leader.kill()
            leader.wait(timeout=10)
            _first, ended = seen.at_return()
            assert not ended.alive_at_return and ended.left_behind == (member,)
            assert os.getpgid(member) == os.getsid(member) == leader.pid
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(leader.pid, signal.SIGKILL)
            leader.wait(timeout=10)
            if leader.stdout is not None:
                leader.stdout.close()
        assert _gone(member)
        assert [item.left_behind for item in seen.at_return()] == [(), ()]


def test_the_write_host_of_the_toy_runs_under_the_network_guard(tmp_path: Path) -> None:
    fixture = toolsets.build(WRITER(2), tmp_path / toolsets.WORKSPACE)
    command = [sys.executable, "-B", "-m", host_program.__name__, str(fixture.workspace)]

    # The process says that it is ready, and that the guard is installed in
    # it, before it takes a call; the toy's host requires both.
    process = subprocess.run(
        [*command, host_program.SCOPED], input=b"", capture_output=True, timeout=60, check=False
    )
    (line,) = process.stdout.decode("utf-8").splitlines()
    assert process.returncode == 0, process.stderr.decode("utf-8", "replace")
    assert json.loads(line) == {
        host_program.READY: {host_program.GUARDED: True, host_program.LEFT: None}
    }
    # The guard of that process is the matrix's, and a reach beyond loopback
    # ends the process whoever swallows the exception: the toy's dispatch
    # then finds its host gone, which is an error of the matrix.
    reach = (
        "import importlib, socket\n"
        f"host = importlib.import_module({host_program.__name__!r})\n"
        "host.guard()\n"
        "try:\n"
        "    socket.create_connection(('192.0.2.1', 9), timeout=1)\n"
        "except BaseException:\n"
        "    pass\n"
        "print('still here')\n"
    )
    reached = subprocess.run(
        [sys.executable, "-B", "-c", reach], capture_output=True, timeout=60, check=False
    )
    assert reached.returncode == host_program.GUARD_EXIT
    assert reached.stdout == b"" and b"192.0.2.1" in reached.stderr
    # A host that does not say so is refused before it is handed a write.
    silent = toy.WriteHost(fixture.workspace, invocation_deadline=time.monotonic() + 30.0)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(host_program, "GUARDED", "not said")
        with pytest.raises(MatrixSetupError, match="under the network guard"):
            silent.start()
    assert silent.pid is not None and _gone(silent.pid)


def test_the_cell_report_names_the_private_name_the_watch_wraps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The report of a process says, per kind, which private names of the
    # product were wrapped while its cells played: none for a kind of which
    # only cells without write tools played, the product's host class once a
    # cell that attaches write tools did.
    monkeypatch.setattr(matrix_report, "_CELLS", {})
    monkeypatch.setattr(matrix_report, "_WATCHED", set())
    column = cells.toy_columns()[0]

    reading = runner.execute(Cell(column, "S03", "raw"), tmp_path / "read")
    matrix_report.note_cell(reading, LEDGER)
    assert not reading.watched
    assert matrix_report.document()["kinds"][TOY_KIND]["watch_seams"] == []
    writing = runner.execute(Cell(column, "S41", "raw"), tmp_path / "write")
    matrix_report.note_cell(writing, LEDGER)
    assert writing.watched
    described = matrix_report.document()["kinds"][TOY_KIND]
    assert described["watch_seams"] == list(mutation.PRIVATE_SEAMS) == [mutation.PRODUCT_HOST]
    assert described["private_seams"] == []
    matrix_report.reset()
    assert matrix_report.describe_kind(TOY_KIND)["watch_seams"] == []


def test_a_write_is_judged_by_the_last_text_its_script_puts_on_disk(tmp_path: Path) -> None:
    honest = _honest(tmp_path, "S45")
    observation = honest.observation
    writes = [call for call in observation.provider.sent_calls if call.slot == WRITE]
    texts = [str(call.arguments["content"]) for call in writes]

    # Six writes, each a longer text than the one before, the last on disk.
    assert len(texts) == 6 and all(b.startswith(a) and a != b for a, b in zip(texts, texts[1:]))
    assert observation.write_applied is True
    assert (tmp_path / toolsets.WORKSPACE / toolsets.WRITE_FILE).read_text(encoding="utf-8") == (
        texts[-1]
    )
    # A script that states no text writes the one of the write slot.
    assert runner._last_written(ProviderLedger()) == toolsets.WRITE_CONTENT
    assert runner._last_written(observation.provider) == texts[-1]
    # A file that holds another text than the last write's: the dispatch of
    # the writes is missing, whatever earlier write is still on disk.
    assert _judge(honest, write_applied=False) == {f"dispatch.missing:{WRITE}"}
    # The toy wire names what the adapter says to the model on its own.
    dialect = ToyDialect()
    blank = dialect.parse("POST", dialect.path, {}, b"{}", credential="")
    noted = dataclasses.replace(blank, matched=True, evidence={"notes": ["a word"]})
    assert dialect.foreign_messages([dataclasses.replace(blank, matched=True)]) == []
    assert dialect.foreign_messages([dataclasses.replace(noted, matched=False)]) == []
    (problem,) = dialect.foreign_messages([dataclasses.replace(blank, matched=True), noted])
    assert problem.startswith("request 2 carries 1 note(s)")
    # The scenario's two assertions, each on one changed fact.
    told = dataclasses.replace(observation.provider, foreign_messages=[problem])
    assert _judge(honest, provider=told) == {"request.content:messages"}
    fewer = dataclasses.replace(
        honest.reference, attached_tools=honest.reference.attached_tools[1:]
    )
    assert _judge(honest, reference=fewer) == {"request.content:tools"}
    # The scenario plays every turn of its budget: as many requests as turns.
    assert honest.plan.max_turns == observation.provider.request_count == 13

    # What a message says, beyond which kind it is. The system and the user
    # text of every scripted request are those of the first.
    def with_request(index: int, **changed: Any) -> frozenset[str]:
        requests = list(observation.provider.requests)
        requests[index] = dataclasses.replace(requests[index], **changed)
        return _judge(honest, provider=dataclasses.replace(observation.provider, requests=requests))

    last = observation.provider.requests[-1]
    assert with_request(12, system=last.system + " One more word.") == {
        "request.content:messages"
    }
    assert with_request(12, user=last.user + " One more word.") == {"request.content:messages"}
    assert with_request(0, system="Another prompt.") == {"request.content:messages"}
    # A request that matched no scripted step is clause C0's and is not read.
    assert with_request(12, system="Another prompt.", matched=False) == frozenset()
    # The result a request carries for a call is a text the attached ToolSet
    # handed back for that tool: the six reads of the script, each answered
    # in this process.
    read = BINDINGS[READ_A].name
    returned = observation.toolset_returned
    assert [item.name for item in returned] == [read] * 6
    assert [item.result for item in returned] == [item.result for item in observation.toolset_log]
    second = observation.provider.requests[1]
    ((call_id, text),) = second.tool_results
    assert text == returned[0].result
    assert with_request(1, tool_results=((call_id, text + "\n\nOne more word."),)) == {
        "request.content:messages"
    }
    assert with_request(1, tool_results=((call_id, "ERROR the tool is not offered now"),)) == {
        "request.content:messages"
    }
    # What a wrapper of the workflow adds to a result is the ToolSet's: the
    # reference is what the attached object handed back, not the innermost.
    warned = text + "\n\n[a word of the workflow's wrapper]"
    handed = (dataclasses.replace(returned[0], result=warned), *returned[1:])
    requests = list(observation.provider.requests)
    requests[1] = dataclasses.replace(second, tool_results=((call_id, warned),))
    carried = dataclasses.replace(observation.provider, requests=requests)
    assert _judge(honest, provider=carried, toolset_returned=handed) == frozenset()
    assert _judge(honest, provider=carried) == {"request.content:messages"}
    # The result of a write that ran in a host was handed back by no ToolSet
    # object of this process, and nothing is said of it.
    third = observation.provider.requests[2]
    ((write_id, written),) = third.tool_results
    assert written.startswith("OK") and all(item.name != BINDINGS[WRITE].name for item in returned)
    assert with_request(2, tool_results=((write_id, written + " And a word."),)) == frozenset()


# -- limits inside a tool loop, and tool calls that must not be dispatched ----------


def test_a_loop_that_requires_nothing_is_never_owed_and_counts_its_own_requests() -> None:
    tools = {"tools": [BINDINGS[READ_A].name]}

    def answer(call_id: str) -> dict[str, Any]:
        return {**tools, "tool_results": [{"id": call_id, "text": "result"}]}

    script = ProviderScript(
        (Loop(Call(READ_A), limit=3, at_least=2), Loop(Say("answer"), limit=1, at_least=0))
    )
    assert runner._required_steps(script) == 1

    # The call ends after the rounds the first loop requires: nothing is owed.
    with LoopbackProvider(script, credential="c") as provider:
        (call,) = _ask(provider, tools)[1]["calls"]
        assert _ask(provider, answer(call["id"]))[1]["type"] == "tool_calls"
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed, seen.request_count) == (0, 0, 2)
    # One round short of them: the first loop is owed, the second never.
    with LoopbackProvider(script, credential="c") as provider:
        _ask(provider, tools)
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (0, 1)
    with LoopbackProvider(script, credential="c") as provider:
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (0, 1)
    # Asked on: the rounds up to the limit of the first loop, then the answer
    # of the second, which counts its own requests, once; then nothing.
    with LoopbackProvider(script, credential="c") as provider:
        (call,) = _ask(provider, tools)[1]["calls"]
        for _ in range(2):
            (call,) = _ask(provider, answer(call["id"]))[1]["calls"]
        assert _ask(provider, answer(call["id"]))[1]["text"] == "answer"
        assert _ask(provider, tools)[0] == UNSCRIPTED_STATUS
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed, seen.request_count) == (1, 0, 5)
    # A loop that requires a request is owed where it was never reached.
    owed = ProviderScript((Loop(Call(READ_A), limit=1), Loop(Say("answer"), limit=1)))
    with LoopbackProvider(owed, credential="c") as provider:
        _ask(provider, tools)
        seen = provider.freeze()
    assert (seen.extra, seen.unconsumed) == (0, 1)


def test_a_call_that_names_no_arguments_or_no_attached_tool_allows_no_dispatch(
    tmp_path: Path,
) -> None:
    read = BINDINGS[READ_A]
    malformed = _honest(tmp_path / "malformed", "S05")
    # Three calls, none with a JSON object for its arguments: a text that is
    # no JSON document, a document that says "nothing", a list that is not
    # empty. Each was answered, and none reached the ToolSet.
    texts = MALFORMED_ARGUMENT_TEXTS
    decoded = [json.loads(text) for text in texts[1:]]
    with pytest.raises(json.JSONDecodeError):
        json.loads(texts[0])
    assert decoded == [None, [NOTES_FILE]] and len(texts) == 3
    calls = malformed.observation.provider.sent_calls
    assert len(calls) == len(texts) and malformed.observation.provider.request_count == 4
    assert all(call.malformed and call.arguments == {} for call in calls)
    assert malformed.observation.toolset_log == ()
    sent = calls[0]
    # Whatever reaches the ToolSet for the call is a dispatch the script never
    # asked for: the tool with an empty object, with the arguments of its
    # binding, or with anything else.
    for arguments in ({}, dict(read.arguments), {"file_path": "another.txt"}):
        reached = (toolsets.Dispatch(read.name, arguments, "a result"),)
        assert _judge(malformed, toolset_log=reached) == {f"dispatch.extra:{READ_A}"}
    # So is what a ToolSet was handed in the place of an argument object:
    # the text as the model wrote it, or the value a text decodes to.
    for handed in (*texts, *decoded, 0):
        reached = (toolsets.Dispatch(read.name, {}, "an answer", not_a_mapping=(handed,)),)
        assert _judge(malformed, toolset_log=reached) == {f"dispatch.extra:{READ_A}"}
    # The same call with arguments is one the script asked for; where the
    # scenario allows no dispatch of it, it still must not run.
    plain = dataclasses.replace(sent, malformed=False, arguments=dict(read.arguments))
    asked = dataclasses.replace(malformed.observation.provider, sent_calls=[plain])
    reached = (toolsets.Dispatch(read.name, dict(read.arguments), "a result"),)
    assert _judge(malformed, provider=asked, toolset_log=reached) == {
        f"dispatch.extra:{READ_A}"
    }
    allowed = dataclasses.replace(malformed.plan.expect, dispatch=(READ_A,))
    assert _judge(malformed, provider=asked, toolset_log=reached, expect=allowed) == frozenset()
    # A call that names no arguments is the dispatch of nothing, whatever a
    # scenario requires of it.
    assert _judge(malformed, toolset_log=reached, expect=allowed) == {
        f"dispatch.extra:{READ_A}",
        f"dispatch.missing:{READ_A}",
    }
    # And a ToolSet that was handed nothing is not the dispatch of a call
    # whose arguments are an empty object: "no arguments" is not "{}".
    empty = dataclasses.replace(sent, malformed=False, arguments={})
    asked_empty = dataclasses.replace(malformed.observation.provider, sent_calls=[empty])
    as_object = (toolsets.Dispatch(read.name, {}, "a result"),)
    assert _judge(malformed, provider=asked_empty, toolset_log=as_object, expect=allowed) == (
        frozenset()
    )
    for handed in (None, []):
        nothing = (toolsets.Dispatch(read.name, {}, "a result", not_a_mapping=(handed,)),)
        assert _judge(
            malformed, provider=asked_empty, toolset_log=nothing, expect=allowed
        ) == {f"dispatch.extra:{READ_A}", f"dispatch.missing:{READ_A}"}

    unattached = _honest(tmp_path / "unattached", "S06")
    (sent,) = unattached.observation.provider.sent_calls
    slot = sent.slot
    name = BINDINGS[slot].name
    assert name == toolsets.UNATTACHED_TOOL and not sent.malformed
    assert name not in unattached.reference.attached_tools
    assert unattached.observation.toolset_log == ()
    # The name itself must not reach the ToolSet, and no attached tool in its
    # place.
    handed_on = (toolsets.Dispatch(name, dict(sent.arguments), "ERROR: unknown tool"),)
    assert _judge(unattached, toolset_log=handed_on) == {f"dispatch.extra:{slot}"}
    in_its_place = (toolsets.Dispatch(read.name, dict(sent.arguments), "a result"),)
    assert _judge(unattached, toolset_log=in_its_place) == {f"dispatch.extra:{READ_A}"}
    # No fixture of the matrix attaches the name.
    for spec in (toolsets.READ, toolsets.READ_BOUNDED(1), WRITER(2)):
        fixture = toolsets.build(spec, tmp_path / f"fixture-{spec.label}")
        assert name not in {str(item["name"]) for item in fixture.toolset.to_anthropic_tools()}


def test_the_recorder_keeps_what_a_toolset_was_handed_and_passes_its_answer_on(
    tmp_path: Path,
) -> None:
    read = BINDINGS[READ_A]
    # What an adapter can hand a ToolSet in the place of an argument object:
    # a list, the text the model wrote, any other text, nothing, an empty
    # list, a number. The ToolSet of the product answers each of them; the
    # recorder must not fail where it answers, and must not lose the call.
    for number, handed in enumerate(
        ([NOTES_FILE], MALFORMED_ARGUMENT_TEXTS[0], NOTES_FILE, None, [], 0)
    ):
        fixture = toolsets.build(toolsets.READ, tmp_path / f"fixture-{number}")
        # The method of the product's class, past the recorder of the instance.
        unrecorded = type(fixture.toolset).dispatch(fixture.toolset, read.name, handed)
        assert fixture.log == []
        answer = fixture.toolset.dispatch(read.name, handed)
        # The caller gets the ToolSet's own answer, and the call is on record
        # with what was handed over, as it came.
        assert answer == unrecorded and answer.startswith("ERROR")
        assert fixture.log == [toolsets.Dispatch(read.name, {}, answer, (handed,))]
        assert fixture.returned is fixture.log
        (logged,) = fixture.log
        assert logged.not_a_mapping[0] is handed or logged.not_a_mapping[0] == handed

    # A mapping is recorded as the arguments of the call, as before.
    fixture = toolsets.build(toolsets.READ, tmp_path / "fixture-mapping")
    for arguments in (dict(read.arguments), {}):
        answer = fixture.toolset.dispatch(read.name, arguments)
        assert fixture.log[-1] == toolsets.Dispatch(read.name, arguments, answer)
        assert fixture.log[-1].not_a_mapping == ()
    assert [item.result.startswith("ERROR") for item in fixture.log] == [False, True]


def test_only_a_wire_of_function_calls_carries_a_call_without_arguments() -> None:
    script = SCENARIOS["S05"].plan().script
    pinned = {
        row["kind"]: row["step"]
        for row in cells.load_inventory().not_expressible
        if row["scenario"] == "S05"
    }
    carried: set[str] = set()
    for kind in sorted(registered_kinds() - {TOY_KIND}):
        try:
            answered = cells.drivers.get(kind).answers(script)
        except MatrixSetupError as exc:
            assert "decoded object" in str(exc) and "not expressible" in str(exc)
        else:
            assert answered
            carried.add(kind)

    # The rows are exactly the kinds whose dialect refuses the step, and the
    # scenario has a cell on every other kind. The three are the kinds whose
    # scripted wire is a session of the vendor's tool, all references: every
    # open-weight kind plays the scenario.
    assert set(pinned.values()) == {"Call(malformed)"} and len(pinned) == 3
    assert set(pinned) <= set(matrix_columns.REFERENCE_KINDS)
    assert carried == set(matrix_columns.REAL_KINDS) - set(pinned)
    assert set(matrix_columns.OPEN_WEIGHT_KINDS) <= carried
    generated = {
        cell.column.kind for cell in cells.behaviour_cells() if cell.scenario == "S05"
    }
    assert generated == carried | {TOY_KIND}
    # The call for a tool that is not attached travels on every wire.
    unattached = SCENARIOS["S06"].plan().script
    for kind in sorted(registered_kinds() - {TOY_KIND}):
        assert cells.drivers.get(kind).answers(unattached)


def test_the_budget_inside_a_tool_loop_goes_by_whose_the_loop_is() -> None:
    scenario = SCENARIOS["S14"]
    assert scenario.governing == ("tool_loop",)
    assert declarations.totality_gaps(scenario.differences) == []

    def plan(value: str) -> Any:
        profile = dataclasses.replace(CONSERVATIVE_BACKEND_FEATURES, tool_loop=value)  # type: ignore[arg-type]
        return plan_for(scenario, declarations.select(scenario.differences, profile))

    own, vendors = plan("adapter_owned"), plan("sdk_owned")
    # A loop of the adapter's own: exactly the two rounds, no third request.
    (rounds,) = own.script.steps
    assert isinstance(rounds, Loop) and (rounds.limit, rounds.at_least) == (2, 2)
    # A loop of a vendor's tool: two to four rounds, and an answer that is
    # never required.
    rounds, answer = vendors.script.steps
    assert isinstance(rounds, Loop) and (rounds.limit, rounds.at_least) == (4, 2)
    assert isinstance(answer, Loop) and isinstance(answer.step, Say) and answer.at_least == 0
    assert runner._required_steps(vendors.script) == runner._required_steps(own.script) == 1
    assert len(vendors.expect.dispatch_optional) == 3 and len(own.expect.dispatch_optional) == 1
    for played in (own, vendors):
        assert played.expect.kind == "budget_exhausted" and played.expect.retriable is False
        assert played.expect.dispatch == (READ_A,) and played.budget_usd == BUDGET_USD
        # One round costs less than the budget and two cost more.
        assert isinstance(rounds.step, Call)
        usage = rounds.step.usage
        cost = matrix_columns.MATRIX_RATE.compute_cost(
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
        )
        assert cost < BUDGET_USD < 2 * cost
    # Nothing is stated for a route that runs no tool loop: such a column
    # cannot play the cell.
    with pytest.raises(MatrixSetupError, match="states nothing for tool_loop='none'"):
        plan("none")


def test_the_turn_limit_without_an_error_keeps_what_the_model_had_written(
    tmp_path: Path,
) -> None:
    scenario = SCENARIOS["S10b"]
    assert scenario.policy.on_max_turns == "continue_with_warning"
    assert scenario.governing == ("turn_limit",) and scenario.lenses == ("raw",)
    assert declarations.totality_gaps(scenario.differences) == []
    kept = scenario.plan().checks[0]

    def column(value: str) -> Column:
        return next(
            item
            for item in cells.toy_columns()
            if item.features is not None and item.features.turn_limit == value
        )

    # Where the model is never asked again it had written nothing.
    at_limit = runner.execute(Cell(column("hard"), "S10b", "raw"), tmp_path / "hard")
    assert at_limit.report.codes == frozenset(), at_limit.report.describe()
    result = at_limit.observation.result
    assert result is not None and result.error is None and result.stop_reason == "max_turns"
    assert result.final_message == "" and not at_limit.observation.provider.tail_served
    assert _judge(at_limit, final_message="a warning") == {"text:mismatch"}
    # Where one further request without tools was answered, that answer is
    # the partial answer.
    asked_on = runner.execute(Cell(column("round_boundary"), "S10b", "raw"), tmp_path / "rb")
    assert asked_on.report.codes == frozenset(), asked_on.report.describe()
    tail = asked_on.plan.script.tail
    assert tail is not None and asked_on.observation.provider.tail_served
    result = asked_on.observation.result
    assert result is not None and result.final_message == tail.text != ""
    assert _judge(asked_on, final_message="") == {"text:mismatch"}
    # An error result promises no answer: the kind is what deviates.
    failed = TransportError(message="x", retriable=False)
    assert _judge(asked_on, error=failed, final_message="") == {"kind:transport!=none"}

    def plan(value: str) -> Any:
        profile = dataclasses.replace(CONSERVATIVE_BACKEND_FEATURES, turn_limit=value)  # type: ignore[arg-type]
        return plan_for(scenario, declarations.select(scenario.differences, profile))

    # The limit noted after more rounds: the answer the model gave is kept.
    late = plan("post_hoc")
    answer = late.script.steps[-1]
    assert isinstance(answer, Say) and late.expect.final_text == answer.text
    assert late.expect.stop == ("max_turns",) and late.expect.kind == "none"
    assert kept not in late.checks and len(late.expect.dispatch) == 3
    # No turn limit claimed: the finite script ends with its answer.
    unlimited = plan("unsupported")
    assert unlimited.expect.stop == ("end_turn",) and kept not in unlimited.checks
    assert unlimited.expect.final_text and len(unlimited.expect.dispatch) == 3


# -- schema without tools, a reply cut at the cap, a reply without output, a
# -- conversation request ---------------------------------------------------------


def _toy_column(**declared: str) -> Column:
    for column in cells.toy_columns():
        if all(str(getattr(column.features, name)) == value for name, value in declared.items()):
            return column
    raise MatrixSetupError(f"no toy column declares {declared}")


def test_the_schema_scenario_without_tools_has_the_table_of_the_planner_shape() -> None:
    plain, planner = SCENARIOS["S02"], SCENARIOS["S04"]

    # One table for both: where the schema travels follows the declaration,
    # with or without tools.
    assert plain.differences == planner.differences
    assert set(plain.fallbacks) == set(planner.fallbacks)
    assert (plain.toolset, plain.lenses, plain.policy) == (None, ("raw",), scenarios.BASE_POLICY)
    placed = {
        value: plan_for(plain, {"structured_output": variant}).checks
        for value, variant in plain.differences["structured_output"].items()
    }
    assert placed == {
        "native": (scenarios.schema_answer_returned, scenarios.schema_in_field),
        "prompt_only": (scenarios.schema_answer_returned, scenarios.schema_in_prompt),
        "unsupported": (scenarios.schema_answer_returned,),
    }
    # Without a ToolSet there is no tool loop to finalise: the declared
    # difference of a schema finalisation has no alternative here, and a
    # cell that listed it would be a setup error.
    assert D_OAI_SCHEMA_FINAL in planner.alternatives
    assert D_OAI_SCHEMA_FINAL not in plain.alternatives
    with pytest.raises(MatrixSetupError, match="defines no alternative"):
        plan_for(plain, {}, [D_OAI_SCHEMA_FINAL])


def test_a_reply_cut_at_the_cap_and_a_reply_without_output_on_the_toy_wire(
    tmp_path: Path,
) -> None:
    dialect = ToyDialect()
    asked = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")
    cut = dialect.render(Cut("what was produced"), asked, "call-1")
    silent = dialect.render(Empty(), asked, "call-2")

    # Both are complete replies of the wire that state their usage: the mark
    # of a cut, or an ordinary finish over nothing at all.
    assert (cut.body["finish"], cut.body["text"], cut.usage) == (
        "length",
        "what was produced",
        Cut("what was produced").usage,
    )
    assert (silent.body["finish"], silent.body["text"]) == ("stop", "")
    assert silent.usage == Empty().usage and silent.usage.output_tokens == 0
    # A reply that produced nothing bills no output token, by its definition.
    with pytest.raises(ValueError, match="bills no output token"):
        Empty(TokenUsage(120, 1))

    # The honest toy: the cut is the general kind, not retriable, with its
    # own stop reason and limit event, and its usage is the bill ...
    capped = runner.execute(
        Cell(_toy_column(response_token_limit="provider_stop"), "S17", "raw"), tmp_path / "cut"
    )
    result, trace = _result_and_trace(capped)
    assert capped.report.codes == frozenset() and result.error is not None
    assert (result.error.kind, result.error.retriable, result.stop_reason) == (
        "transport",
        False,
        "max_tokens",
    )
    assert [event.name for event in trace.events if event.kind == "limit"] == ["response_tokens"]
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert result.final_message == "" and all(dataclasses.asdict(result.completeness).values())
    # ... and where the route claims no cap the reply is complete, the
    # finish the clean one of the plain text scenario, and no cap is sent.
    uncapped = runner.execute(
        Cell(_toy_column(response_token_limit="unsupported"), "S17", "raw"), tmp_path / "whole"
    )
    assert uncapped.report.codes == frozenset()
    assert uncapped.plan.script == SCENARIOS["S01"].plan().script
    assert uncapped.plan.expect == SCENARIOS["S01"].plan().expect
    assert set(SCENARIOS["S01"].plan().checks) < set(uncapped.plan.checks)
    assert [request.response_cap for request in uncapped.observation.provider.requests] == [None]
    # A reply without output: a transport error a caller may retry, named
    # after no stop reason a workflow reads, with the input tokens billed.
    nothing = _honest(tmp_path / "silent", "S25")
    result, _trace = _result_and_trace(nothing)
    assert result.error is not None and (result.error.kind, result.error.retriable) == (
        "transport",
        True,
    )
    assert result.stop_reason not in contract.READ_STOP_REASONS
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 0)
    assert all(dataclasses.asdict(result.completeness).values())


def test_the_cap_of_a_response_is_asked_only_where_the_wire_has_a_field_for_it(
    tmp_path: Path,
) -> None:
    capped = runner.execute(
        Cell(_toy_column(response_token_limit="provider_stop"), "S17", "raw"), tmp_path
    )
    provider = capped.observation.provider
    (request,) = provider.requests

    # The toy wire has the field, the dialect says so, and the request of a
    # route that declares an enforced cap carried one.
    assert isinstance(ToyDialect(), ResponseCapDialect) and provider.response_cap_field
    assert request.response_cap is not None and request.response_cap >= 1
    assert scenarios.response_cap_in_request in capped.plan.checks
    # Read without the cap, the request is one code on a wire that has the
    # field, and none on a wire that has no field for it: there no adapter
    # could have sent one.
    untold = dataclasses.replace(
        provider, requests=[dataclasses.replace(request, response_cap=None)]
    )
    assert _judge(capped, provider=untold) == {"request.content:response_cap"}
    assert _judge(capped, provider=dataclasses.replace(untold, response_cap_field=False)) == (
        frozenset()
    )
    # A cap that is no positive count is none.
    zero = dataclasses.replace(provider, requests=[dataclasses.replace(request, response_cap=0)])
    assert _judge(capped, provider=zero) == {"request.content:response_cap"}
    # Every value of the field has its variant: an enforced cap is asked in
    # the request, a route that counts between rounds or afterwards is not
    # asked, and a route that claims none plays the complete reply.
    table = SCENARIOS["S17"].differences["response_token_limit"]
    assert {value: variant.name for value, variant in table.items()} == {
        "hard": "cap_in_request",
        "provider_stop": "cap_in_request",
        "round_boundary": "",
        "post_hoc": "",
        "unsupported": "complete_reply",
    }
    assert declarations.totality_gaps(SCENARIOS["S17"].differences) == []


def test_a_conversation_request_is_answered_by_declaration(tmp_path: Path) -> None:
    scenario = SCENARIOS["S50"]
    table = scenario.differences["session_continuity"]
    request = scenarios.CONVERSATION

    # The request of the scenario asks to continue a conversation, and the
    # table is total: a route that declares continuity for a workflow
    # answers, every other one refuses with the one agreed type.
    assert (request.mode, request.action, request.opaque_id) == (
        "workflow_bound",
        "reuse",
        scenarios.STALE_HANDLE,
    )
    assert declarations.totality_gaps(scenario.differences) == []
    assert {value: (variant.kind, variant.name) for value, variant in table.items()} == {
        "workflow_bound": ("BASE", ""),
        "retry_only": ("REFUSED", "NotImplementedError"),
        "none": ("REFUSED", "NotImplementedError"),
    }
    # Where it is refused nothing is sent, and the plan still carries the
    # request: the refusal is of the conversation, not of a plain call.
    refused = runner.execute(
        Cell(_toy_column(session_continuity="none"), "S50", "raw"), tmp_path / "refused"
    )
    raised = refused.observation.raised
    assert refused.report.codes == frozenset() and refused.observation.result is None
    assert raised is not None and (raised.type, raised.phase) == (
        "NotImplementedError",
        "before_request",
    )
    # The refusal is the route's own: it left the call, and the frame that
    # raised it is in the module of the backend the cell built.
    assert raised.left_invoke
    assert raised.module == refused.observation.backend_module == toy.__name__
    assert refused.plan.conversation is scenarios.CONVERSATION
    assert refused.plan.script == ProviderScript(())
    assert refused.observation.provider.request_count == 0
    assert not set(RESULT_CLAUSES) & set(refused.report.outcomes)
    # Where it is answered and nobody holds the handle: the typed stale
    # conversation, with an outcome that names no handle and is bound to
    # the request; no request was made and no usage is claimed.
    stale = runner.execute(
        Cell(_toy_column(session_continuity="workflow_bound"), "S50", "raw"), tmp_path / "stale"
    )
    result, _trace = _result_and_trace(stale)
    outcome = result.conversation
    assert stale.report.codes == frozenset() and result.error is not None
    assert (result.error.kind, result.stop_reason) == ("stale_conversation", "stale_conversation")
    # The same handle is refused again: no error a caller may try again.
    assert stale.plan.expect.retriable is False and result.error.retriable is False
    again = dataclasses.replace(result.error, retriable=True)
    assert _judge(stale, error=again) == {"retriable:True"}
    assert outcome is not None
    assert (outcome.state, outcome.opaque_id) == ("stale", None)
    assert (outcome.identity_fingerprint, outcome.operation_digest) == (
        request.identity_fingerprint,
        request.operation_digest,
    )
    assert stale.observation.provider.request_count == 0
    # The three flags a scenario may waive are not asked of such a result.
    assert stale.plan.expect.flags == ("trace_complete",)
    assert {"flag:token_usage_complete=False", "flag:cost_complete=False"} <= set(
        stale.report.unasked
    )
    # Each fact of the outcome has its code.
    untyped = TransportError(message="no such conversation", retriable=False)
    assert _judge(stale, conversation=None, error=untyped) == {
        "kind:transport!=stale_conversation",
        "conversation:missing",
    }
    other = dataclasses.replace(outcome, operation_digest=request.identity_fingerprint)
    assert _judge(stale, conversation=other) == {"conversation:unbound"}
    # A plain call carries no conversation request: only the scenario about
    # continuity plans one, and the guarded lens's second call never does.
    assert [name for name, item in SCENARIOS.items() if item.plan().conversation] == ["S50"]


class _KeepsConversations(ToyDialect):
    """The toy wire with a tool beside the adapter that keeps the conversations."""

    resume_path = "/toy/v1/resume"

    def refuse_resume(self, body: bytes) -> tuple[str, Reply]:
        handle = str(json.loads(body)["handle"])
        return handle, Reply(200, {"type": "no_conversation", "handle": handle})


def test_a_handle_is_asked_of_the_tool_that_keeps_the_conversations(tmp_path: Path) -> None:
    kept_by_a_tool = _KeepsConversations()
    asked_for = {"handle": "a-handle"}

    # On a wire where a tool beside the adapter keeps the conversations the
    # provider side answers for that tool. The exchange is no model request:
    # the ledger keeps the handle apart from the requests of the script.
    assert isinstance(kept_by_a_tool, KeeperDialect)
    assert not isinstance(ToyDialect(), KeeperDialect)
    with LoopbackProvider(
        ProviderScript(()), credential="c", dialect=kept_by_a_tool
    ) as provider:
        answered = _ask(provider, asked_for, path=kept_by_a_tool.resume_path)
        seen = provider.freeze()
    assert answered == (200, {"type": "no_conversation", "handle": "a-handle"})
    assert seen.conversation_keeper and seen.refused_handles == ["a-handle"]
    assert (seen.request_count, seen.extra, seen.unconsumed) == (0, 0, 0)
    # Where the adapter keeps them itself there is nobody to ask: the same
    # request is one the script does not have.
    with LoopbackProvider(ProviderScript(()), credential="c") as provider:
        assert _ask(provider, asked_for, path=kept_by_a_tool.resume_path)[0] == UNSCRIPTED_STATUS
        plain = provider.freeze()
    assert not plain.conversation_keeper and plain.refused_handles == []
    assert (plain.request_count, plain.extra) == (1, 1)

    # The toy keeps its conversations itself, so its stale result asks
    # nobody and the cell asks nothing of that.
    stale = runner.execute(
        Cell(_toy_column(session_continuity="workflow_bound"), "S50", "raw"), tmp_path
    )
    own = stale.observation.provider
    assert stale.report.codes == frozenset()
    assert not own.conversation_keeper and own.refused_handles == []
    # The same result on a wire where a tool keeps them is an adapter that
    # called the handle stale without asking whoever holds the conversations ...
    kept = dataclasses.replace(own, conversation_keeper=True)
    assert _judge(stale, provider=kept) == {"conversation:unasked"}
    assert contract.clause_of("conversation:unasked") == contract.SCENARIO
    # ... unless the tool was asked for the handle of the request, and for
    # no other one.
    handle = scenarios.STALE_HANDLE
    assert _judge(stale, provider=dataclasses.replace(kept, refused_handles=[handle])) == (
        frozenset()
    )
    assert _judge(stale, provider=dataclasses.replace(kept, refused_handles=[handle] * 2)) == (
        frozenset()
    )
    for others in (["another-handle"], [handle, "another-handle"]):
        assert _judge(stale, provider=dataclasses.replace(kept, refused_handles=others)) == {
            "conversation:unasked"
        }


# -- a table keyed by the state of a route extra ------------------------------------

_PIN = declarations.SERVED_MODEL_EXTRA
_CARRIED, _LEFT_OUT, _REFUSED = declarations.EXTRA_STATES


def _real(kind: str, role: str = "primary") -> Column:
    return next(
        item for item in matrix_columns.real_columns() if (item.kind, item.role) == (kind, role)
    )


def _routing_document(kind: str) -> dict[str, Any]:
    path = _real(kind).routing_path(matrix_columns.COLUMN_PRODUCT)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _state(document: dict[str, Any], role: str) -> str:
    return declarations.extra_state(
        yaml.safe_dump(document, sort_keys=False).encode("utf-8"), role, _PIN
    )


def test_the_state_of_a_route_extra_is_what_routing_states_and_is_never_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text, reader = (matrix_columns.ROLE_BY_CAPABILITY[name] for name in ("text", "read_only"))
    hosted = _routing_document(matrix_columns.HOSTED_GATEWAY_KIND)
    claude = _routing_document(matrix_columns.CLAUDE_KIND)

    # The three states, on documents held in memory. The shipped gateway
    # file carries the pin, set to the route's own model. Without the pin
    # routing still admits it there. A backend whose routes do not take the
    # extra is refused for exactly that extra.
    assert _state(hosted, reader) == _CARRIED
    without = copy.deepcopy(hosted)
    del without["default"]["extra"][_PIN]
    assert _state(without, reader) == _LEFT_OUT
    assert _state(claude, reader) == _REFUSED
    # A role of another backend family than its default reads the state of
    # its own route.
    mixed = copy.deepcopy(hosted)
    mixed["roles"][text] = {**claude["default"], **claude["roles"].get(text, {})}
    assert (_state(mixed, text), _state(mixed, reader)) == (_REFUSED, _CARRIED)
    # A route pinned to another id is not played; a document routing refuses
    # as it stands is no document.
    elsewhere = copy.deepcopy(hosted)
    elsewhere["default"]["extra"][_PIN] = "another-model-id"
    with pytest.raises(MatrixSetupError, match="pinned to another id"):
        _state(elsewhere, reader)
    broken = copy.deepcopy(hosted)
    broken["default"]["extra"]["no_such_extra"] = True
    with pytest.raises(MatrixSetupError, match="routing refuses the routing document as it stands"):
        _state(broken, reader)

    # An answer of routing that is neither an acceptance nor its refusal of
    # exactly the one extra on that route is an error, never a state.
    lint, calls = declarations.lint_routing_config, []

    def answering(sentence: str) -> Any:
        def linted(document: bytes) -> Any:
            calls.append(document)
            if len(calls) % 2 == 0:  # the second validation: the extra is added
                raise ValueError(sentence)
            return lint(document)

        return linted

    refusal = f"{reader} config has unsupported 'claude' extra fields ['{_PIN}']; allowed fields are []"
    for sentence, state in (
        (refusal, _REFUSED),
        (refusal.replace(reader, "default"), None),
        (refusal.replace(f"['{_PIN}']", f"['{_PIN}', 'another']"), None),
        ("the route config must declare a model", None),
    ):
        calls.clear()
        monkeypatch.setattr(declarations, "lint_routing_config", answering(sentence))
        if state is None:
            with pytest.raises(MatrixSetupError, match="neither with an acceptance nor"):
                _state(without, reader)
        else:
            assert _state(without, reader) == state
    monkeypatch.setattr(declarations, "lint_routing_config", lint)

    # Selection cannot be typed: a column with a routing file that states a
    # state itself is an error, and so is a column without one that states
    # none. A real column on which routing admits the pin and which leaves
    # it out is not played: no ruling names one.
    column = _real(matrix_columns.CLAUDE_KIND)
    assert declarations.route_state(column, reader, _PIN) == _REFUSED
    typed = dataclasses.replace(column, route_states=((_PIN, _LEFT_OUT),))
    with pytest.raises(MatrixSetupError, match="never typed"):
        declarations.route_state(typed, reader, _PIN)
    toy_column = cells.toy_columns()[0]
    with pytest.raises(MatrixSetupError, match="states no state"):
        declarations.route_state(dataclasses.replace(toy_column, route_states=()), reader, _PIN)
    with pytest.raises(MatrixSetupError, match="no route extra that keys"):
        declarations.route_state(column, reader, matrix_columns.COST_BOUND_EXTRA)
    monkeypatch.setattr(declarations, "extra_state", lambda *_arguments: _LEFT_OUT)
    with pytest.raises(MatrixSetupError, match="No ruling names a real column"):
        declarations.route_state(column, reader, _PIN)


def test_a_table_keyed_by_a_route_extra_is_total_and_selects_from_the_state() -> None:
    table = SCENARIOS["S24"].differences
    profile = cells.toy_columns()[0].features
    assert profile is not None

    assert SCENARIOS["S24"].governing == (_PIN,) and declarations.totality_gaps(table) == []
    # A state left out, and a state that is none.
    short = {_PIN: {_CARRIED: declarations.BASE, _REFUSED: declarations.BASE}}
    assert declarations.totality_gaps(short) == [f"{_PIN}={_LEFT_OUT} has no variant"]
    invented = {_PIN: {**table[_PIN], "pinned_elsewhere": declarations.BASE}}
    assert declarations.totality_gaps(invented) == [f"{_PIN}=pinned_elsewhere is not a value"]
    with pytest.raises(MatrixSetupError, match="has no variant for 'pinned_elsewhere'"):
        declarations.variants(table, {_PIN: "pinned_elsewhere"})
    # The variant follows the state and nothing of the profile; without the
    # state nothing is selected.
    kinds = {
        state: declarations.select(table, profile, {_PIN: state})[_PIN].kind
        for state in declarations.EXTRA_STATES
    }
    assert kinds == {_CARRIED: "BASE", _LEFT_OUT: "FALLBACK", _REFUSED: "BASE"}
    with pytest.raises(MatrixSetupError, match="its state on the route was not read"):
        declarations.select(table, profile)
    # The key is no declared field: the toy columns stay derived from the
    # declared fields, and the snapshot has no column for it.
    assert _PIN not in governing_fields() and _PIN not in declarations.FEATURE_FIELDS
    assert scenarios_module.keying_extras() == (_PIN,)


def test_every_state_of_the_route_extra_has_an_honest_toy_pass(tmp_path: Path) -> None:
    columns = cells.toy_columns()
    stated = [dict(column.route_states) for column in columns]

    # Every toy column states its state itself, and every state is stated.
    assert all(set(item) == {_PIN} for item in stated)
    assert {item[_PIN] for item in stated} == set(declarations.EXTRA_STATES)
    world = cells.world()
    for state in declarations.EXTRA_STATES:
        column = next(item for item in columns if dict(item.route_states)[_PIN] == state)
        cell = Cell(column, "S24", "raw")
        execution = runner.execute(cell, tmp_path / state)
        result, trace = _result_and_trace(execution)
        provider = execution.observation.provider

        assert execution.report.codes == frozenset(), (state, execution.report.describe())
        # The current values of the cell are its state, and of a cell of
        # another scenario on the same column they hold no such key.
        assert execution.current == {_PIN: state} == dict(world.current(cell.id))
        assert _PIN not in world.current(Cell(column, "S01", "raw").id)
        assert trace.served_model == provider.served_models[-1] != trace.requested_model
        if state == _LEFT_OUT:
            # The fallback: the reply is taken, its call dispatched, and a
            # second reply under the same id ends the turn.
            assert result.error is None and provider.request_count == 2
            assert [item.name for item in execution.observation.toolset_log] == [
                BINDINGS[READ_A].name
            ]
        else:
            # The base contract: incomplete evidence that names the field,
            # nothing dispatched, the cost not complete and the tokens kept.
            assert result.error is not None and result.error.kind == "incomplete_evidence"
            assert result.stop_reason == "model_mismatch" and provider.request_count == 1
            assert execution.observation.toolset_log == ()
            flags = result.completeness
            assert (flags.token_usage_complete, flags.cost_complete) == (True, False)
            billed = provider.billed_total()
            assert (result.usage.input_tokens, result.usage.output_tokens) == (
                billed.input_tokens,
                billed.output_tokens,
            )
            # The terminal flag is not asked: either value passes.
            for value in (True, False):
                assert _judge(
                    execution,
                    completeness=dataclasses.replace(flags, terminal_complete=value),
                ) == frozenset()


def test_an_entry_and_a_record_of_a_served_model_cell_are_locked_to_the_state() -> None:
    world = cells.world()
    listed = [
        (record.id, item)
        for record in LEDGER.registry.values()
        for item in record.cells
        if ledger.parse_cell_id(item.cell)[1] == "S24"
    ]

    # A record is permanent, so the cells it lists stay to be read here.
    assert listed
    for _gap, item in listed:
        current = dict(world.current(item.cell))
        assert set(current) == {_PIN} and current[_PIN] in declarations.EXTRA_STATES
        assert dict(item.observed_under) == dict(item.closes_under) == current
    gap, item = listed[0]
    state = dict(world.current(item.cell))[_PIN]
    other = next(name for name in declarations.EXTRA_STATES if name != state)
    # The comparator holds an entry to the state it was observed under: a
    # cell whose route changed state fails with the name of the key, never
    # with "gap fixed", whatever it then shows.
    known = ledger.parse(
        {
            "schema_version": ledger.LEDGER_SCHEMA,
            "ceilings": {"max_known_gaps": {"pack_b": 1}, "max_declared_differences": 0},
            "registry": {
                "M99": {
                    "title": "a known gap",
                    "origin": "matrix",
                    "severity": "medium",
                    "cells": [
                        {
                            "cell": item.cell,
                            "observed_under": {_PIN: state},
                            "closes_under": {_PIN: state},
                        }
                    ],
                }
            },
            "known_gaps": [
                {
                    "cell": item.cell,
                    "observed_under": {_PIN: state},
                    "codes": {"kind:transport!=incomplete_evidence": "M99"},
                }
            ],
            "declared_differences": [],
        }
    )
    codes = {"kind:transport!=incomplete_evidence"}
    with pytest.raises(XFAILED):
        ledger.compare(known, item.cell, codes, current={_PIN: state})
    for shown in (codes, set()):
        with pytest.raises(ledger.LedgerMismatch, match=f"decl.changed:{_PIN}"):
            ledger.compare(known, item.cell, shown, current={_PIN: other})
    # H03 holds a listed cell that has no entry to its closes_under: as the
    # fix of the gap leaves the ledger, the record is clean while it names
    # the state the cell runs under, and not once it names another.
    fixed = _document()
    fixed["known_gaps"] = [entry for entry in fixed["known_gaps"] if entry.get("cell") != item.cell]
    locks = {"static:registry_closes_under", "static:registry_closes_mismatch"}
    assert not locks & _problem_codes(fixed)
    _registry_cell(fixed, gap, item.cell)["closes_under"] = {_PIN: other}
    assert "static:registry_closes_under" in _problem_codes(fixed)
    fixed["registry"][gap]["declaration_change"] = {
        "field": _PIN,
        "from": state,
        "to": other,
        "citation": "a ruling that names the route",
    }
    named = _problem_codes(fixed)
    assert "static:registry_closes_mismatch" in named
    assert "static:registry_closes_under" not in named


# -- a route without its credential --------------------------------------------------


def test_a_call_without_a_credential_asks_one_typed_error_no_request_and_the_trace_flag(
    tmp_path: Path,
) -> None:
    cell = Cell(cells.toy_columns()[0], "S28", "raw")
    scenario = SCENARIOS[cell.scenario]

    def played(name: str, *faults: str) -> runner.Execution:
        return runner.execute(cell, tmp_path / name, driver=toy.ToyDriver(frozenset(faults)))

    honest = _honest(tmp_path / "honest", "S28")
    result, trace = _result_and_trace(honest)

    # The state is the scenario's, said once and without a kind: nothing is
    # scripted, no check of its own, no governing field, the base policy.
    assert scenario.credential_unset and scenario.plan().script == ProviderScript(())
    assert scenario.plan().checks == () and scenario.governing == ()
    assert scenario.policy == BASE_POLICY and scenario.lenses == ("raw",)
    assert [item.id for item in SCENARIOS.values() if item.credential_unset] == ["S28"]
    # The honest toy: a transport error that is not retried, a stop reason
    # no workflow reads as an outcome, nothing sent, and a complete trace
    # that ends in its one terminal event.
    assert result.error is not None and result.error.kind == "transport"
    assert result.error.retriable is False and result.stop_reason == "error"
    assert honest.observation.provider.request_count == 0
    assert [(event.kind, event.name) for event in trace.events] == [
        ("request", None),
        ("terminal", "error"),
    ]
    flags = result.completeness
    assert (flags.token_usage_complete, flags.cost_complete, flags.terminal_complete) == (
        False,
        False,
        False,
    )
    # The three flags are not asked, and none is expected false: a result
    # that accounts for the nothing it sent and claims its terminal passes
    # as the one that flags all three false does.
    claimed = played("claimed", toy.ACCOUNT_FOR_NOTHING, toy.CLAIM_TERMINAL_OF_A_BROKEN_TURN)
    claims = _result_and_trace(claimed)[0].completeness
    assert all(getattr(claims, name) for name in contract.FLAGS)
    assert claimed.report.codes == frozenset()
    # The request event of the attempt is neither asked nor refused.
    silent = played("silent", toy.NO_REQUEST_EVENT)
    assert [event.kind for event in _result_and_trace(silent)[1].events] == ["terminal"]
    assert silent.report.codes == frozenset()
    # The trace flag is owed, and what such a result reports is still held
    # to what the provider billed, which is nothing.
    assert _judge(
        honest, completeness=dataclasses.replace(flags, trace_complete=False)
    ) == {"flag:trace_complete=False"}
    assert _judge(
        honest, usage=dataclasses.replace(result.usage, input_tokens=7)
    ) == {"usage.input_tokens:gt"}
    # A stop reason a workflow reads as an outcome of its own deviates; no
    # spelling is asked beyond that.
    assert _judge(honest, stop_reason="end_turn") == {
        "stop:end_turn!=other",
        "event.terminal:error!=end_turn",
    }
    assert _judge(honest, stop_reason="missing_credential") == {
        "event.terminal:error!=missing_credential"
    }


class _BuiltWithTheCredentialAllTheSame(toy.ToyDriver):
    """A driver that builds the backend with the credential, whatever it is asked for."""

    def route_credential(self, without_credential: bool) -> str:
        return matrix_columns.CANARY_CREDENTIAL


def test_a_play_without_the_credential_that_sent_the_scripted_one_is_an_error_of_the_matrix(
    tmp_path: Path,
) -> None:
    cell = Cell(cells.toy_columns()[0], "S28", "raw")

    # The toy reads its credential when it is built. A backend that was
    # built with it sends its one request under the scripted credential:
    # the codes of the cell would be those of an adapter that sends a
    # placeholder, so the provider's own record of the bearer decides.
    with pytest.raises(MatrixSetupError, match="carried the scripted credential"):
        runner.execute(cell, tmp_path / "built", driver=_BuiltWithTheCredentialAllTheSame())
    # The placeholder of the lying toy is no scripted credential: that play
    # is an observation, with its code.
    liar = runner.execute(
        cell, tmp_path / "liar", driver=toy.ToyDriver(frozenset({toy.SEND_PLACEHOLDER_KEY}))
    )
    (sent,) = liar.observation.provider.requests
    assert liar.report.codes == {"request.extra"}
    assert not sent.scripted_credential and not sent.credential_ok
    # On every other cell the request carries it, and the provider says so
    # whatever the dialect accepts.
    (plain,) = _honest(tmp_path / "plain", "S01").observation.provider.requests
    assert plain.scripted_credential and plain.credential_ok


def test_a_driver_may_leave_one_step_for_the_last_moment_before_the_call(tmp_path: Path) -> None:
    cell = Cell(cells.toy_columns()[0], "S01", "raw")
    moments: list[int] = []

    class _Late(toy.ToyDriver):
        @contextlib.contextmanager
        def open(self, column: Column, **arguments: Any) -> Iterator[Any]:
            with super().open(column, **arguments) as bound:
                # The step sees the backend built and nothing sent yet.
                bound.before_call = lambda: moments.append(bound.provider.request_count)  # type: ignore[attr-defined]
                yield bound

    execution = runner.execute(cell, tmp_path, driver=_Late())

    assert moments == [0] and execution.observation.provider.request_count == 1
    assert execution.report.codes == frozenset()


# -- the module ends with the canary check --------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, TOY_CELLS) == []
