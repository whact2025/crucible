# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The per-process cell report: what each cell showed, beyond pass or fail.

JUnit says whether a cell passed, failed or was an expected failure. This
report says why: the deviation codes, the gaps the ledger attributes them to,
the variant each governing declared value selected, the declared differences
in force, the outcome of every clause that applies, the grade of the weakest
envelope the provider answered from, and what the driver of the kind bypasses,
with the private names of the product that it and the required proofs of the
kind's module replace, and the private names that the watch of write hosts
wraps while a cell of the kind that attaches write tools plays.

One file per pytest process, written beside the JUnit file when the run has
one (``test-results-3.xml`` gives ``matrix-report-3.json``). JUnit properties
are not used: ``record_property`` warns under pytest's default JUnit family.
A process that judges no cell writes no file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from . import cells, declarations, drivers, ledger, mutation
from .contract import Deviation
from .scenarios import SCENARIOS

if TYPE_CHECKING:
    from .runner import Execution

REPORT_SCHEMA = "crucible.backend-matrix.report.v1"
_JUNIT_PREFIX = "test-results-"
_REPORT_PREFIX = "matrix-report-"
NOT_JUDGED = "not_judged"

# What this process has seen, by cell id.
_CELLS: dict[str, dict[str, Any]] = {}
# The kinds of which a cell played under the watch of write hosts.
_WATCHED: set[str] = set()


def reset() -> None:
    _CELLS.clear()
    _WATCHED.clear()


def _gaps(known: ledger.Ledger, cell: str, codes: Sequence[str]) -> list[str]:
    """The gaps the ledger attributes the codes of ``cell`` to."""

    kind = ledger.parse_cell_id(cell)[0]
    gaps = {
        scope.gap
        for scope in known.column_wide
        if scope.kind == kind and scope.code in codes
    }
    entry = known.entry(cell)
    if entry is not None:
        gaps.update(gap for code, gap in entry.codes.items() if code in codes)
    return sorted(gaps)


def note_cell(execution: Execution, known: ledger.Ledger) -> None:
    """Record what a played behaviour cell showed."""

    cell, report = execution.cell, execution.report
    if execution.watched:
        _WATCHED.add(cell.column.kind)
    if _CELLS.get(cell.id, {}).get("outcome", NOT_JUDGED) != NOT_JUDGED:
        return  # the cell's own test item was judged; a later replay is not it
    scenario = SCENARIOS[cell.scenario]
    selected = declarations.select(scenario.differences, execution.reference.routed)
    codes = sorted(report.codes)
    _CELLS[cell.id] = {
        "kind": cell.column.kind,
        "scenario": cell.scenario,
        "role": cell.column.role,
        "lens": cell.lens,
        "outcome": NOT_JUDGED,
        "codes": codes,
        "gaps": _gaps(known, cell.id, codes),
        "declared": dict(execution.current),
        "variants": {
            name: variant.kind if not variant.name else f"{variant.kind}:{variant.name}"
            for name, variant in selected.items()
        },
        "differences": [
            *sorted(v.difference for v in selected.values() if v.difference),
            *execution.differences,
        ],
        "clauses": {
            clause: outcome.status for clause, outcome in report.outcomes.items()
        },
        "grade": execution.grade,
        "envelopes": sorted(set(execution.observation.provider.envelopes)),
        "seam_problems": list(report.seam_problems),
    }


def note_static(cell: str, problems: Sequence[Deviation], known: ledger.Ledger) -> None:
    """Record what a static cell found."""

    kind, scenario, role, lens = ledger.parse_cell_id(cell)
    codes = sorted({item.code for item in problems})
    _CELLS[cell] = {
        "kind": kind,
        "scenario": scenario,
        "role": role,
        "lens": lens,
        "outcome": NOT_JUDGED,
        "codes": codes,
        "gaps": _gaps(known, cell, codes),
    }


def note_outcome(cell: str, outcome: str) -> None:
    """Record how the test item of a noted cell ended."""

    if cell in _CELLS:
        _CELLS[cell]["outcome"] = outcome


def path_beside(junit: str | Path) -> Path:
    """The report file that belongs to a JUnit file."""

    junit = Path(junit)
    stem = junit.stem
    name = (
        _REPORT_PREFIX + stem[len(_JUNIT_PREFIX) :]
        if stem.startswith(_JUNIT_PREFIX)
        else _REPORT_PREFIX + stem
    )
    return junit.with_name(name + ".json")


def describe_kind(kind: str) -> dict[str, Any]:
    """What the evidence of a driven kind leans on, said once per kind.

    ``watch_seams`` are the private names of the product that the watch of
    write hosts wrapped in the test process: named where a cell of the kind
    that this process played attaches write tools, and empty otherwise.
    """

    driver = drivers.get(kind)
    return {
        "dialect": driver.dialect,
        "fidelity": driver.fidelity,
        "bypassed_layers": list(driver.bypassed_layers),
        "private_seams": list(driver.private_seams),
        "proof_seams": list(cells.PROOF_SEAMS.get(cells.module_of(kind), ())),
        "watch_seams": list(mutation.PRIVATE_SEAMS) if kind in _WATCHED else [],
        "envelopes": list(driver.envelopes),
    }


def document() -> dict[str, Any]:
    """The report of this process."""

    kinds = sorted({record["kind"] for record in _CELLS.values()} & drivers.registered_kinds())
    described: dict[str, Mapping[str, Any]] = {kind: describe_kind(kind) for kind in kinds}
    return {
        "schema_version": REPORT_SCHEMA,
        "kinds": described,
        "cells": {cell: _CELLS[cell] for cell in sorted(_CELLS)},
    }


def write(junit: str | Path | None) -> Path | None:
    """Write the report beside the JUnit file; nothing without one or without cells."""

    if junit is None or not _CELLS:
        return None
    path = path_beside(junit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
