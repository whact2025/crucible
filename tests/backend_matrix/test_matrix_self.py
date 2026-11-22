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
import math
import os
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

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
from crucible.backends.pricing import ModelPricing

from . import cells, contract, declarations, ledger, netguard, runner
from . import columns as matrix_columns
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
    LoopbackProvider,
    ProviderLedger,
    Reply,
    SeenRequest,
    StatelessDialect,
    ToyDialect,
)
from .scenarios import (
    BUDGET_USD,
    D_CDX_SERVED,
    D_COST_BOUND_UNRANKED,
    D_OAI_SCHEMA_FINAL,
    D_STRUCT_UNQUALIFIED,
    GUARDED,
    PACK_A,
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
    tool_events_paired,
)
from .script import READ_A, Call, Loop, ProviderScript, Say, ServedAs, Step
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

    execution = runner.execute(cell, tmp_path, driver=driver, differences=liar.differences)
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


def test_a_rehearsed_gap_of_the_matrix_is_a_registry_record() -> None:
    # A gap of the gap list (G..) may be rehearsed before any cell of it is
    # recorded. A gap the matrix found itself exists in the registry only, so
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
        endpoint=None,
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
    mispriced = "usage.cost_bounds:amounts"

    def price(tokens: Any, read: int) -> float:
        # The bound formula, written out here and not taken from the contract.
        unread = tokens.input_tokens - read
        return (
            unread * rate.input_per_mtok
            + read * reads_per_mtok
            + tokens.output_tokens * rate.output_per_mtok
        ) / 1_000_000

    def segment(tokens: Any, *, read: int | None = None) -> UsageCostBoundSegment:
        # Unknown count: every input token may have been read, or none.
        return UsageCostBoundSegment(
            input_tokens=tokens.input_tokens,
            output_tokens=tokens.output_tokens,
            cache_creation_tokens=0,
            cache_read_tokens=read,
            lower_bound_usd=price(tokens, tokens.input_tokens if read is None else read),
            upper_bound_usd=price(tokens, 0 if read is None else read),
            policy="a-cost-policy",
        )

    def interval(*segments: UsageCostBoundSegment, coverage: bool = True) -> UsageCostBounds:
        return UsageCostBounds(tuple(segments), coverage)

    def judged(
        bounds: UsageCostBounds | None, *, reference: Any = bounded, **fields: Any
    ) -> frozenset[str]:
        cost = usage.cost_usd if bounds is None else bounds.lower_bound_usd
        reported = dataclasses.replace(
            usage, **{"cost_usd": cost, "cost_bounds": bounds, **fields}
        )
        return _judge(
            execution,
            reference=reference,
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

    counted = interval(dataclasses.replace(first, cache_read_tokens=0), *others)
    for wrong in (free, narrow, nudged(1 + 1e-6), counted):
        assert judged(wrong) == untold | {mispriced}
    assert contract.clause_of(mispriced) == "C6"
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
    # Nothing is reported at all. Without the opt-in the cleared flags excuse
    # it as a lower bound; with it the tokens and the interval are owed.
    silent = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
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
    # result may flag its accounting complete: nothing is named.
    exact = interval(*(segment(item, read=0) for item in billed))
    assert math.isclose(exact.lower_bound_usd, usage.cost_usd, rel_tol=1e-12)
    assert _judge(
        execution, reference=bounded, usage=dataclasses.replace(usage, cost_bounds=exact)
    ) == frozenset()
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
        variants = declarations.select(scenario.differences, declarations.routed(cell.column))
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
    }
    # With a cell of the record withdrawn the ruling is what H03 asks for.
    # Either the cell is pinned not expressible ...
    row = {"scenario": "S01", "kind": "kimi", "step": "Say", "reason": "r", "checklist": "c"}
    withdrawing = _document()
    withdrawing["registry"]["M07"]["ruling"] = "the two cells are withdrawn"
    pinned = _problem_codes(withdrawing, not_expressible=(row,))
    assert "static:registry_ruling_unused" not in pinned
    # ... or it is no longer generated although its kind runs the set: with
    # the entries of the two cells deleted, nothing is left to object to.
    gone = {item["cell"] for item in withdrawing["registry"]["M07"]["cells"]}
    withdrawing["known_gaps"] = [
        item for item in withdrawing["known_gaps"] if item.get("cell") not in gone
    ]
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
        item
        for item in unreachable["known_gaps"]
        if item.get("cell") != "deepseek/H02/-/static"
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
    # At its ceiling, whatever room the shipped ceiling leaves today.
    on_canary["ceilings"]["max_known_gaps"]["pack_a"] = len(on_canary["known_gaps"])
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
    ceiling["ceilings"]["max_known_gaps"]["pack_a"] = len(ceiling["known_gaps"]) - 1
    assert "static:ceiling_exceeded" in _problem_codes(ceiling)
    exact = _document()
    exact["ceilings"]["max_known_gaps"]["pack_a"] = len(exact["known_gaps"])
    assert _problem_codes(exact) == set()

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
    assert len(unranked) == 6 and all(
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


# -- the module ends with the canary check --------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, TOY_CELLS) == []
