# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cells: columns x enabled scenarios x lenses, and the inventory of them.

The generator never hands pytest an empty parameter set and never drops a
cell quietly. It raises when a column's kind is unknown, when a kind has no
column, when two columns share a (kind, role) and when a kind that has a
registered driver yields no cell. A kind *without* a driver yields no cell by
definition; static cell H02 reports that, as a ledgered expected failure.

Enablement is by scenario *set*, never by single scenario, so one scenario
cannot be withdrawn from one kind. The only per-scenario exception is the
not-expressible list pinned in ``inventory.yaml``, applied before any declared
value is read.

The *item inventory* names the pytest item that judges each cell: the test
module of its driver family and the test's name. The verifier compares it
with the JUnit files of a run, and the outcome recorder reads it back.

Everything here is computable without a driver's ``preflight()`` and without a
worker environment.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

from . import columns as matrix_columns
from . import declarations, drivers, ledger
from .columns import REAL_KINDS, TOY_KIND, Column
from .contract import CLAUSES
from .drivers import toy
from .drivers.base import MatrixSetupError
from .scenarios import (
    PACK_A,
    SCENARIOS,
    STARTER,
    Scenario,
    difference_alternatives,
    difference_variants,
    governing_fields,
)

INVENTORY_PATH = Path(__file__).with_name("inventory.yaml")
INVENTORY_SCHEMA = "crucible.backend-matrix.inventory.v1"

# Which scenario sets run on which kind. A real kind gets the starter set with
# its driver, pack A at the gate and pack B after it.
ENABLEMENT: Mapping[str, tuple[str, ...]] = {
    TOY_KIND: (STARTER, PACK_A),
    **{kind: () for kind in REAL_KINDS},
    matrix_columns.HOSTED_GATEWAY_KIND: (STARTER,),
    matrix_columns.LOCAL_GATEWAY_KIND: (STARTER,),
    matrix_columns.CLAUDE_KIND: (STARTER,),
    matrix_columns.DEEPSEEK_KIND: (STARTER,),
}
# A secondary column runs these sets; a blocked column these scenarios.
SECONDARY_SETS: tuple[str, ...] = (STARTER, PACK_A)
BLOCKED_SCENARIOS: tuple[str, ...] = ("S00", "S01", "S51")
STATIC_SCENARIOS: tuple[str, ...] = ("P01", "P02", "P04", "H01", "H02", "H03", "H04")

# Where each cell is judged. One test module per driver family, because the
# shard planner places whole files; a module may hold two kinds.
PACKAGE_MODULE = "tests.backend_matrix"
BEHAVIOUR_TEST = "test_cell"
MODULE_BY_KIND: Mapping[str, str] = {
    TOY_KIND: "test_matrix_self",
    matrix_columns.HOSTED_GATEWAY_KIND: "test_matrix_openai_compat",
    matrix_columns.LOCAL_GATEWAY_KIND: "test_matrix_openai_compat",
    matrix_columns.CLAUDE_KIND: "test_matrix_claude",
    matrix_columns.DEEPSEEK_KIND: "test_matrix_deepseek",
}
STATIC_MODULE = "test_matrix_static"
# Static scenario -> its test, and whether the test is parametrised by cell id.
STATIC_TESTS: Mapping[str, tuple[str, bool]] = {
    "P01": ("test_p01_protocol_surface", True),
    "P02": ("test_p02_profile_binding", True),
    "P04": ("test_p04_declared_profile_snapshot", False),
    "H01": ("test_h01_no_test_skips_on_a_missing_worker_runtime", True),
    "H02": ("test_h02_every_backend_is_driven", True),
    "H03": ("test_h03_ledger_and_oracle_integrity", False),
    "H04": ("test_h04_worker_environments_only_behind_the_guard", False),
}
# The last test of each module: it fails when a cell the module relies on
# did not pass, so it must have run too.
CANARY_CHECK = "test_canary_passed_for_every_column"
STATIC_CHECK = "test_every_static_cell_of_the_inventory_was_judged"
# Proofs a module holds beside its cells, which the verifier requires by name:
# product behaviour that no clause states, so no cell's verdict shows when it
# stops. The DeepSeek adapter revalidates the pin of its worker before every
# launch; only the launch journal of a tool round shows it.
REQUIRED_PROOFS: Mapping[str, tuple[str, ...]] = {
    MODULE_BY_KIND[matrix_columns.DEEPSEEK_KIND]: (
        "test_the_probe_and_the_request_launch_run_under_the_harness",
    ),
}


@dataclass(frozen=True)
class Cell:
    """One scenario on one column in one lens."""

    column: Column
    scenario: str
    lens: str

    @property
    def id(self) -> str:
        return ledger.cell_id(self.column.kind, self.scenario, self.column.role, self.lens)


@dataclass(frozen=True)
class Item:
    """One expected pytest item: where a cell is judged, a required proof or a closing check."""

    module: str
    name: str
    cell: str | None = None


@dataclass(frozen=True)
class Inventory:
    cell_counts: Mapping[str, int]
    not_evaluated: tuple[Mapping[str, Any], ...]
    not_expressible: tuple[Mapping[str, str], ...]
    worker_skip_files: Mapping[str, frozenset[str]]
    # (cell, code) pairs on which a column-wide known gap is not owed.
    column_wide_not_owed: tuple[Mapping[str, str], ...] = ()


def parse_inventory(document: Any) -> Inventory:
    expected = {
        "schema_version",
        "cell_counts",
        "not_evaluated",
        "not_expressible",
        "worker_skip_files",
        "column_wide_not_owed",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected
        or document["schema_version"] != INVENTORY_SCHEMA
    ):
        raise MatrixSetupError("inventory.yaml has an unknown shape")
    not_evaluated = tuple(document["not_evaluated"] or ())
    for row in not_evaluated:
        if set(row) != {"scenario", "kind", "role", "lens", "clauses", "reason"}:
            raise MatrixSetupError(f"not_evaluated row {row!r} has unknown fields")
        if not set(row["clauses"]) <= set(CLAUSES):
            raise MatrixSetupError(f"not_evaluated row {row!r} names an unknown clause")
    not_expressible = tuple(document["not_expressible"] or ())
    for row in not_expressible:
        # No field for a declared value: a declaration cannot select this list.
        if set(row) != {"scenario", "kind", "step", "reason", "checklist"}:
            raise MatrixSetupError(f"not_expressible row {row!r} has unknown fields")
    not_owed = tuple(document["column_wide_not_owed"] or ())
    for row in not_owed:
        if (
            not isinstance(row, dict)
            or set(row) != {"cell", "code", "reason"}
            or not all(isinstance(value, str) and value.strip() for value in row.values())
        ):
            raise MatrixSetupError(
                f"column_wide_not_owed row {row!r} must state a cell, a code and the reason"
            )
        ledger.parse_cell_id(row["cell"])
    skip_files = document["worker_skip_files"] or {}
    worker_kinds = set(matrix_columns.WORKER_KINDS.values())
    if not isinstance(skip_files, dict) or not all(
        kind in worker_kinds
        and isinstance(files, list)
        and files
        and all(isinstance(name, str) and name for name in files)
        for kind, files in skip_files.items()
    ):
        raise MatrixSetupError(
            "worker_skip_files must map a worker kind "
            f"({sorted(worker_kinds)}) to a non-empty list of test files"
        )
    return Inventory(
        cell_counts={str(k): int(v) for k, v in document["cell_counts"].items()},
        not_evaluated=not_evaluated,
        not_expressible=not_expressible,
        worker_skip_files={
            str(kind): frozenset(files) for kind, files in skip_files.items()
        },
        column_wide_not_owed=not_owed,
    )


@functools.cache
def load_inventory() -> Inventory:
    return parse_inventory(yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8")))


def pinned_not_evaluated(
    rows: Sequence[Mapping[str, Any]], cell: Cell, clause: str
) -> bool:
    """Whether the inventory pins ``clause`` as not evaluated on ``cell``."""

    values = {
        "scenario": cell.scenario,
        "kind": cell.column.kind,
        "role": cell.column.role,
        "lens": cell.lens,
    }
    return any(
        clause in row["clauses"]
        and all(row[name] in ("*", value) for name, value in values.items())
        for row in rows
    )


def _in_role_scope(column: Column, scenario: Scenario) -> bool:
    if column.kind == TOY_KIND:
        return True
    role = column.role.split("-", 1)[0]
    if role == "blocked":
        return scenario.id in BLOCKED_SCENARIOS
    return role == "primary" or scenario.set in SECONDARY_SETS


def generate(
    columns: Sequence[Column],
    *,
    registered: frozenset[str],
    enablement: Mapping[str, tuple[str, ...]] = ENABLEMENT,
    scenarios: Mapping[str, Scenario] = SCENARIOS,
    not_expressible: Sequence[Mapping[str, str]] = (),
) -> tuple[Cell, ...]:
    """Generate every behaviour cell, or raise ``MatrixSetupError``."""

    seen: set[str] = set()
    for column in columns:
        if column.id in seen:
            raise MatrixSetupError(f"two columns share the (kind, role) {column.id}")
        seen.add(column.id)
        if column.kind not in enablement:
            raise MatrixSetupError(
                f"column {column.id}: kind {column.kind!r} is not in the enablement table"
            )
    kinds = {column.kind for column in columns}
    for kind in sorted(set(enablement) | registered):
        if kind not in kinds:
            raise MatrixSetupError(f"kind {kind!r} has no column")
    pinned = {(row["kind"], row["scenario"]) for row in not_expressible}
    cells: list[Cell] = []
    for column in columns:
        if column.kind not in registered:
            continue  # no driver yet: H02 reports static:no_driven_column
        for scenario in scenarios.values():
            if scenario.set not in enablement[column.kind]:
                continue
            if not _in_role_scope(column, scenario):
                continue
            if (column.kind, scenario.id) in pinned:
                continue  # a dialect property; no declared value is consulted
            cells.extend(Cell(column, scenario.id, lens) for lens in scenario.lenses)
    for kind in sorted(registered):
        if not any(cell.column.kind == kind for cell in cells):
            raise MatrixSetupError(
                f"kind {kind!r} has a registered driver and yields no cell; an "
                "empty parameter set would become a skipped item"
            )
    return tuple(cells)


@functools.cache
def toy_columns() -> tuple[Column, ...]:
    """The toy columns, derived from the declaration snapshot."""

    snapshot = declarations.snapshot_profiles(declarations.load_snapshot())
    derived = matrix_columns.toy_columns(snapshot, governing_fields())
    for column in derived:
        assert column.features is not None
        unsupported = toy.unsupported_values(column.features)
        if unsupported:
            raise MatrixSetupError(
                f"the snapshot declares {unsupported}, which the toy cannot play: "
                "no honest pass is possible for that variant"
            )
    return derived


@functools.cache
def all_columns() -> tuple[Column, ...]:
    return (*matrix_columns.real_columns(), *toy_columns())


@functools.cache
def behaviour_cells() -> tuple[Cell, ...]:
    """Every behaviour cell of the matrix today."""

    return generate(
        all_columns(),
        registered=drivers.load(),
        not_expressible=load_inventory().not_expressible,
    )


def cells_of(kinds: Sequence[str]) -> tuple[Cell, ...]:
    """The cells a test module of these kinds parametrises over; never empty."""

    selected = tuple(cell for cell in behaviour_cells() if cell.column.kind in kinds)
    if not selected:
        raise MatrixSetupError(f"kinds {list(kinds)} yield no cell")
    return selected


def static_cell_ids() -> tuple[str, ...]:
    """The static cells: always generated, whatever is driven."""

    ids = [
        ledger.cell_id(ledger.ALL_KINDS, name, ledger.NO_ROLE, ledger.STATIC_LENS)
        for name in ("P04", "H03", "H04")
    ]
    ids.extend(
        ledger.cell_id(kind, "H01", ledger.NO_ROLE, ledger.STATIC_LENS)
        for kind in sorted(matrix_columns.WORKER_KINDS.values())
    )
    ids.extend(
        ledger.cell_id(kind, "H02", ledger.NO_ROLE, ledger.STATIC_LENS)
        for kind in REAL_KINDS
    )
    for column in toy_columns():
        ids.extend(
            ledger.cell_id(TOY_KIND, name, column.role, ledger.STATIC_LENS)
            for name in ("P01", "P02")
        )
    return tuple(ids)


def inventory() -> dict[str, Any]:
    """Every expected test item: behaviour cells by kind, and the static cells."""

    by_kind: dict[str, list[str]] = {kind: [] for kind in sorted(ENABLEMENT)}
    for cell in behaviour_cells():
        by_kind[cell.column.kind].append(cell.id)
    return {"behaviour": by_kind, "static": list(static_cell_ids())}


def module_of(kind: str) -> str:
    """The test module that holds the cells of a driven kind."""

    try:
        return MODULE_BY_KIND[kind]
    except KeyError as exc:
        raise MatrixSetupError(
            f"kind {kind!r} is driven but no test module is named for it"
        ) from exc


@functools.cache
def expected_items() -> tuple[Item, ...]:
    """Every pytest item a complete run of the package holds a verdict in."""

    items: list[Item] = []
    modules: list[str] = []
    for cell in behaviour_cells():
        module = module_of(cell.column.kind)
        if module not in modules:
            modules.append(module)
        items.append(Item(module, f"{BEHAVIOUR_TEST}[{cell.id}]", cell.id))
    for cell_id in static_cell_ids():
        test, parametrised = STATIC_TESTS[ledger.parse_cell_id(cell_id)[1]]
        items.append(
            Item(STATIC_MODULE, f"{test}[{cell_id}]" if parametrised else test, cell_id)
        )
    for module in modules:
        items.extend(Item(module, proof) for proof in REQUIRED_PROOFS.get(module, ()))
    items.extend(Item(module, CANARY_CHECK) for module in modules)
    items.append(Item(STATIC_MODULE, STATIC_CHECK))
    return tuple(items)


@functools.cache
def _cell_by_item() -> Mapping[tuple[str, str], str]:
    return {
        (item.module, item.name): item.cell
        for item in expected_items()
        if item.cell is not None
    }


def cell_of_item(module: str, name: str) -> str | None:
    """The cell a pytest item judges, by its module and test name."""

    return _cell_by_item().get((module, name))


def refuse_skip_markers(items: Sequence[Any], root: Path) -> None:
    """Refuse every item under ``root`` that carries a skip or xfail marker.

    An empty parameter set (``[NOTSET]``) carries a skip marker too, so it is
    refused here as well. Only the ledger's comparator may report an expected
    failure, and nothing in the matrix may skip.
    """

    offenders = [
        f"{item.nodeid} ({name})"
        for item in items
        if root in Path(str(item.path)).parents
        for name in ("skip", "skipif", "xfail")
        if item.get_closest_marker(name) is not None
    ]
    if offenders:
        raise pytest.UsageError(
            "the backend matrix refuses skip and xfail markers; a cell that "
            "cannot run is an error and a known gap is a ledger entry:\n  "
            + "\n  ".join(offenders)
        )


def world() -> ledger.World:
    """The real world static cell H03 checks the ledger against."""

    inventory_ = load_inventory()
    cells = behaviour_cells()
    columns = {(column.kind, column.role): column for column in all_columns()}
    snapshot = declarations.snapshot_profiles(declarations.load_snapshot())
    fields = governing_fields()

    def current(cell: str) -> Mapping[str, str]:
        kind, scenario, role, _lens = ledger.parse_cell_id(cell)
        profile = declarations.routed(columns[(kind, role)])
        return declarations.governing_values(profile, SCENARIOS[scenario].governing)

    counts = {kind: 0 for kind in ENABLEMENT}
    for cell in cells:
        counts[cell.column.kind] += 1
    single = [liar for liar in toy.LIARS if not liar.composite]
    return ledger.World(
        generated=frozenset(cell.id for cell in cells) | frozenset(static_cell_ids()),
        columns=frozenset(columns),
        kinds=frozenset(ENABLEMENT),
        scenario_sets={name: item.set for name, item in SCENARIOS.items()},
        scenario_lenses={name: item.lenses for name, item in SCENARIOS.items()},
        static_scenarios=frozenset(STATIC_SCENARIOS),
        canary_scenarios=frozenset(n for n, item in SCENARIOS.items() if item.canary),
        enabled_sets=ENABLEMENT,
        current=current,
        cell_counts=counts,
        pinned_counts=inventory_.cell_counts,
        toy_kind=TOY_KIND,
        reference_kinds=matrix_columns.REFERENCE_KINDS,
        open_weight_kinds=matrix_columns.OPEN_WEIGHT_KINDS,
        not_expressible=inventory_.not_expressible,
        not_evaluated=inventory_.not_evaluated,
        column_wide_not_owed=inventory_.column_wide_not_owed,
        liar_scenarios=frozenset(liar.scenario for liar in single),
        liar_clauses=frozenset(clause for liar in single for clause in liar.trips),
        snapshot_values={
            name: frozenset(str(getattr(row, name)) for row in snapshot.values())
            for name in fields
        },
        toy_values={
            name: frozenset(str(getattr(c.features, name)) for c in toy_columns())
            for name in fields
        },
        difference_variants=difference_variants(),
        difference_alternatives=difference_alternatives(),
    )
