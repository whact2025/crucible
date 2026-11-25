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
from typing import Any, Literal, Mapping, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

from . import columns as matrix_columns
from . import declarations, drivers, envelopes, ledger
from .columns import REAL_KINDS, TOY_KIND, Column
from .contract import ADMITTED, CLAUSES
from .drivers import toy
from .drivers.base import MatrixSetupError
from .scenarios import (
    D_CDX_SERVED,
    D_COST_BOUND_UNRANKED,
    GUARDED,
    PACK_A,
    PACK_B,
    SCENARIOS,
    STARTER,
    Scenario,
    difference_alternatives,
    difference_variants,
    governing_fields,
    plan_for,
)

INVENTORY_PATH = Path(__file__).with_name("inventory.yaml")
INVENTORY_SCHEMA = "crucible.backend-matrix.inventory.v1"

# Which scenario sets run on which kind. A real kind gets the starter set with
# its driver, then pack A, and pack B after it. Every real kind runs all
# three: a scenario of pack B runs on the primary column of each kind from
# the change that defines it.
ENABLEMENT: Mapping[str, tuple[str, ...]] = {
    TOY_KIND: (STARTER, PACK_A, PACK_B),
    **{kind: () for kind in REAL_KINDS},
    matrix_columns.HOSTED_GATEWAY_KIND: (STARTER, PACK_A, PACK_B),
    matrix_columns.LOCAL_GATEWAY_KIND: (STARTER, PACK_A, PACK_B),
    matrix_columns.CLAUDE_KIND: (STARTER, PACK_A, PACK_B),
    matrix_columns.DEEPSEEK_KIND: (STARTER, PACK_A, PACK_B),
    matrix_columns.KIMI_KIND: (STARTER, PACK_A, PACK_B),
    matrix_columns.CODEX_LEGACY_KIND: (STARTER, PACK_A, PACK_B),
    matrix_columns.CODEX_OFFICIAL_KIND: (STARTER, PACK_A, PACK_B),
}
# A secondary column runs these sets: pack B runs on the primary column of
# a kind only. A blocked column runs these scenarios as soon as its kind is
# driven, whatever sets the kind runs: what a blocked profile shows is its
# seam, one plain call and what routing admits.
SECONDARY_SETS: tuple[str, ...] = (STARTER, PACK_A)
BLOCKED_SCENARIOS: tuple[str, ...] = ("S00", "S01", "S51")
# What an admission cell reads from the profile catalog and asks routing for:
# the subjects of its pin. Where the driver of a kind adds an admission input
# to the routing file (``columns.ADMISSION_OVERLAY``), the cell asks for each
# capability again on the file as shipped, and the pin states that too
# (``admission_subjects``).
AVAILABILITY = "availability"
ADMISSION_SUBJECTS: tuple[str, ...] = (AVAILABILITY, *matrix_columns.ROLE_BY_CAPABILITY)
AS_SHIPPED = "_as_shipped"
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
    matrix_columns.KIMI_KIND: "test_matrix_kimi",
    matrix_columns.CODEX_LEGACY_KIND: "test_matrix_codex",
    matrix_columns.CODEX_OFFICIAL_KIND: "test_matrix_codex_official",
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
# The scenarios of the official Codex column whose turn has several model
# responses and ends in a result: where known gap R05 shows.
SEVERAL_RESPONSES_ON_THE_OFFICIAL_CODEX_ROUTE: tuple[str, ...] = (
    "S03",
    "S04",
    "S10a",
    "S18",
    "S40",
)
# Those of them that run in the guarded lens too: where the limit guard
# replaces the result (known gap R04) of a turn that shows R05.
GUARDED_SEVERAL_RESPONSES_ON_THE_OFFICIAL_CODEX_ROUTE: tuple[str, ...] = (
    "S03",
    "S04",
    "S10a",
)
# The cell of the official Codex column whose result is that of a
# dispatch-budget denial: where known gap R09 shows.
DISPATCH_DENIAL_ON_THE_OFFICIAL_CODEX_ROUTE = ledger.cell_id(
    matrix_columns.CODEX_OFFICIAL_KIND, "S12", "primary", "raw"
)
# The cell of the app-server Codex column whose result is that of the same
# denial: where known gap R08 shows.
DISPATCH_DENIAL_ON_THE_APP_SERVER_CODEX_ROUTE = ledger.cell_id(
    matrix_columns.CODEX_LEGACY_KIND, "S12", "primary", "raw"
)
# The cells of the Claude kind that attach a ToolSet with write names and end
# in a result of the attempt: where known gap R13 shows.
WRITES_ON_THE_CLAUDE_COLUMNS: tuple[str, ...] = tuple(
    ledger.cell_id(matrix_columns.CLAUDE_KIND, "S40", role, "raw")
    for role in ("primary", "secondary")
)
# Proofs a module holds beside its cells, which the verifier requires by name:
# product behaviour that no clause states, so no cell's verdict shows when it
# stops. The DeepSeek adapter revalidates the pin of its worker before every
# launch; only the launch journal of a tool round shows it. The strict Kimi
# route rejects the provider's plain reply for its missing count of cache
# reads; the codes of its cells are those of any rejection after the request,
# so only the text of the error shows the cause, and only the same cells
# answered with the count show that nothing else is behind their entries.
# The two diagnostic Kimi routes run under a cost policy: that they are
# called as their routing files ask and refuse to run otherwise, and that a
# guarded result stays diagnostic (not rankable) once the reply carries the
# count, is what the declared difference of those routes rests on, and no
# cell's verdict shows either. What a segment of their cost interval costs is
# held by a rule that reads the shipped price row; only the proof that pins
# the amounts of one cell as numbers shows a change that the adapter and that
# row make together.
# The official Codex route never flags its token usage complete on a reply
# that reads nothing from a cache, and the accounting clause holds a result
# that is flagged incomplete to no number: only this proof compares what the
# route reports for a response with what the provider billed. What the route
# reports for a turn of several responses is known gap R05 of the ledger,
# unconfirmed: its one code shows only while the cost is flagged complete, so
# the record leans on the test that plays both readings of the usage
# notification on the raw cell of every scenario whose turn has several
# responses, and that test is held here too, cell by cell. In the guarded lens
# the limit guard replaces the result of such a turn, and its two codes say
# neither which evidence the guard missed nor what the turn it replaced
# reported: a twin of that test plays both readings on every guarded cell
# whose entry names R05. The result of a dispatch-budget denial on the route
# reports no usage for the responses the provider billed (known gap R09), and
# it flags its accounting incomplete, so no clause compares it with anything:
# only a proof shows it, and shows when it stops.
# On the app-server route the result of the same denial flags its accounting
# incomplete too (known gap R08) and reports every response the provider
# billed. While its entry stands no clause compares what it reports either,
# and a result that had lost its usage would show the same two codes: only a
# proof holds the numbers.
# On a write the trace of the Claude adapter binds a tool-authority digest
# that is not the shared one (known gap R13). The one code of its cells says
# that the digest differs, not why: only the proof that plays the write of
# each Claude column shows that the value is the digest of the same tools
# without a path scope, and that nothing else stands behind the code.
REQUIRED_PROOFS: Mapping[str, tuple[str, ...]] = {
    MODULE_BY_KIND[matrix_columns.CLAUDE_KIND]: tuple(
        "test_the_trace_of_a_write_binds_the_digest_of_the_same_tools_without_a_path_scope"
        f"[{kind}/{role}]"
        for kind, _scenario, role, _lens in map(ledger.parse_cell_id, WRITES_ON_THE_CLAUDE_COLUMNS)
    ),
    MODULE_BY_KIND[matrix_columns.DEEPSEEK_KIND]: (
        "test_the_probe_and_the_request_launch_run_under_the_harness",
    ),
    MODULE_BY_KIND[matrix_columns.KIMI_KIND]: (
        "test_the_strict_route_rejects_the_plain_reply_for_its_missing_cache_read_count",
        "test_the_strict_route_answers_once_the_reply_carries_a_cache_read_count",
        "test_a_route_with_the_cost_bound_gets_its_budget_and_its_policy",
        "test_a_cache_read_count_in_the_reply_leaves_the_guarded_result_diagnostic",
        "test_the_bounds_of_an_honest_interval_are_the_shipped_price_of_what_was_billed",
    ),
    MODULE_BY_KIND[matrix_columns.CODEX_LEGACY_KIND]: (
        "test_the_result_of_a_dispatch_budget_denial_reports_every_response_the_provider_billed",
    ),
    MODULE_BY_KIND[matrix_columns.CODEX_OFFICIAL_KIND]: (
        "test_a_turn_of_one_response_reports_the_tokens_the_provider_billed",
        *(
            f"test_what_a_turn_of_several_responses_reports_rests_on_the_reading_of_last[{name}]"
            for name in SEVERAL_RESPONSES_ON_THE_OFFICIAL_CODEX_ROUTE
        ),
        *(
            "test_what_a_guarded_turn_of_several_responses_shows_rests_on_the_reading_of_last"
            f"[{name}]"
            for name in GUARDED_SEVERAL_RESPONSES_ON_THE_OFFICIAL_CODEX_ROUTE
        ),
        "test_the_result_of_a_dispatch_budget_denial_reports_no_usage_for_what_the_provider_billed",
    ),
}
# The private names of the product that a required proof replaces in the test
# process, by module: what the proof leans on beside the seams of its driver.
# The report of a run prints them with the driver's own list. The proof of the
# Claude module binds the adapter's hash to the ToolSet that was attached, to
# show what its cell is left with once that one cause is removed.
PROOF_SEAMS: Mapping[str, tuple[str, ...]] = {
    MODULE_BY_KIND[matrix_columns.CLAUDE_KIND]: (
        "crucible.backends.claude.ClaudeBackend._tool_authority_digest",
    ),
}


@dataclass(frozen=True)
class RecordProof:
    """A required proof that the known-gap entries of a registry record lean on."""

    module: str
    test: str
    # How an item of the test names the cell it plays: one item whatever the
    # cell (""), one for every scenario, or one for every column.
    per: Literal["", "scenario", "column"] = ""
    # The lens of the entries that lean on it; every lens where none is named.
    lens: str | None = None

    def item(self, cell: str | None) -> str | None:
        """The pytest item an entry on ``cell`` leans on, if it leans on this test.

        ``None`` for ``cell`` is a column-wide entry: it names no cell, so it
        leans on a test only where one item of the test stands for the record.
        """

        if cell is None:
            return None if self.per else self.test
        kind, scenario, role, lens = ledger.parse_cell_id(cell)
        if self.lens is not None and lens != self.lens:
            return None
        if self.per == "scenario":
            return f"{self.test}[{scenario}]"
        if self.per == "column":
            return f"{self.test}[{kind}/{role}]"
        return self.test


# The proofs that the entries of a registry record lean on. A proof holds what
# the codes of an entry do not say: the cause of a code, or a number that no
# clause compares while the entry stands. So an entry must not outlive it:
# static cell H03 fails when an entry attributes a code to a record of this
# table and the verifier does not require the item of the proof that plays the
# entry's cell (static:entry_proof_not_required). A change that deletes a
# proof and its name from the list above therefore deletes the entries that
# lean on it, or fails. Each proof refuses, for its part, an entry that
# attributes a code to its record on a cell it does not play.
# The official Codex route: while the column-wide flag code of R04 stands,
# clause C6 compares no reported token with the provider's ledger on any cell
# of the kind, and the test of a turn of one response does. The usage code of
# R05 rests on the reading of `last`: the test that plays both readings on
# the raw cell of the entry's scenario and, for an entry on a guarded cell,
# its twin. The result of a dispatch-budget denial is compared by no clause
# on either Codex route while its entry stands (R08, R09). The digest code
# of a write on the Claude columns says that the digest differs, not why
# (R13).
_CLAUDE_MODULE = MODULE_BY_KIND[matrix_columns.CLAUDE_KIND]
_APP_SERVER_MODULE = MODULE_BY_KIND[matrix_columns.CODEX_LEGACY_KIND]
_OFFICIAL_MODULE = MODULE_BY_KIND[matrix_columns.CODEX_OFFICIAL_KIND]
RECORD_PROOFS: Mapping[str, tuple[RecordProof, ...]] = {
    "R04": (
        RecordProof(
            _OFFICIAL_MODULE,
            "test_a_turn_of_one_response_reports_the_tokens_the_provider_billed",
        ),
    ),
    "R05": (
        RecordProof(
            _OFFICIAL_MODULE,
            "test_what_a_turn_of_several_responses_reports_rests_on_the_reading_of_last",
            per="scenario",
        ),
        RecordProof(
            _OFFICIAL_MODULE,
            "test_what_a_guarded_turn_of_several_responses_shows_rests_on_the_reading_of_last",
            per="scenario",
            lens=GUARDED,
        ),
    ),
    "R08": (
        RecordProof(
            _APP_SERVER_MODULE,
            "test_the_result_of_a_dispatch_budget_denial_reports_every_response_the_provider_billed",
        ),
    ),
    "R09": (
        RecordProof(
            _OFFICIAL_MODULE,
            "test_the_result_of_a_dispatch_budget_denial_reports_no_usage_for_what_the_provider_billed",
        ),
    ),
    "R13": (
        RecordProof(
            _CLAUDE_MODULE,
            "test_the_trace_of_a_write_binds_the_digest_of_the_same_tools_without_a_path_scope",
            per="column",
        ),
    ),
}


def proofs_leaned_on(gap: str, cell: str | None) -> tuple[tuple[str, str], ...]:
    """The pytest items (module, name) an entry leans on that names ``gap`` on ``cell``.

    ``cell`` is the cell of a known-gap entry, or ``None`` for a column-wide
    entry. Empty for a record that leans on no proof.
    """

    return tuple(
        (proof.module, name)
        for proof in RECORD_PROOFS.get(gap, ())
        for name in (proof.item(cell),)
        if name is not None
    )


def required_proof_items() -> frozenset[tuple[str, str]]:
    """The pytest items (module, name) the verifier requires by name beside the cells."""

    return frozenset(
        (module, name) for module, names in REQUIRED_PROOFS.items() for name in names
    )


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
    # What routing does today on each admission cell of a real column.
    admission: tuple[Mapping[str, str], ...] = ()


def parse_inventory(document: Any) -> Inventory:
    expected = {
        "schema_version",
        "cell_counts",
        "not_evaluated",
        "not_expressible",
        "worker_skip_files",
        "column_wide_not_owed",
        "admission",
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
    admission = tuple(document["admission"] or ())
    for row in admission:
        if not isinstance(row, dict) or not isinstance(row.get("cell"), str):
            raise MatrixSetupError(f"admission row {row!r} must state a cell")
        subjects = admission_subjects(ledger.parse_cell_id(row["cell"])[0])
        if set(row) != {"cell", *subjects, "reason"} or not all(
            isinstance(value, str) and value.strip() for value in row.values()
        ):
            raise MatrixSetupError(
                f"admission row {row!r} must state a cell, {list(subjects)} "
                "and the reason"
            )
    if len({row["cell"] for row in admission}) != len(admission):
        raise MatrixSetupError("an admission cell is pinned twice")
    return Inventory(
        cell_counts={str(k): int(v) for k, v in document["cell_counts"].items()},
        not_evaluated=not_evaluated,
        not_expressible=not_expressible,
        worker_skip_files={
            str(kind): frozenset(files) for kind, files in skip_files.items()
        },
        column_wide_not_owed=not_owed,
        admission=admission,
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


def _is_blocked(column: Column) -> bool:
    return column.kind != TOY_KIND and column.role.split("-", 1)[0] == "blocked"


def _in_role_scope(column: Column, scenario: Scenario) -> bool:
    if column.kind == TOY_KIND:
        return True
    if _is_blocked(column):
        return scenario.id in BLOCKED_SCENARIOS
    return column.role.split("-", 1)[0] == "primary" or scenario.set in SECONDARY_SETS


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
            # A blocked column runs its three scenarios whatever sets its
            # kind runs; any other column a scenario of an enabled set.
            if not _is_blocked(column) and scenario.set not in enablement[column.kind]:
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


def admission_pin_problems(
    cells: Sequence[Cell],
    pins: Sequence[Mapping[str, str]],
    scenarios: Mapping[str, Scenario] = SCENARIOS,
) -> list[str]:
    """An admission cell of a real column has its pin, and a pin its cell."""

    owed = {
        cell.id
        for cell in cells
        if scenarios[cell.scenario].admission and cell.column.kind != TOY_KIND
    }
    pinned = {row["cell"] for row in pins}
    return [
        *(
            f"admission cell {cell} has no pin in inventory.yaml: what routing "
            "admits there today is not stated"
            for cell in sorted(owed - pinned)
        ),
        *(
            f"inventory.yaml pins the admission of {cell}, which is no generated "
            "admission cell of a real column"
            for cell in sorted(pinned - owed)
        ),
    ]


@functools.cache
def behaviour_cells() -> tuple[Cell, ...]:
    """Every behaviour cell of the matrix today."""

    generated = generate(
        all_columns(),
        registered=drivers.load(),
        not_expressible=load_inventory().not_expressible,
    )
    problems = admission_pin_problems(generated, load_inventory().admission)
    if problems:
        raise MatrixSetupError("; ".join(problems))
    return generated


def as_shipped(capability: str) -> str:
    """The subject of ``capability`` asked for on the routing file as shipped."""

    return f"{capability}{AS_SHIPPED}"


def admission_subjects(kind: str) -> tuple[str, ...]:
    """The subjects the pin of an admission cell of ``kind`` states.

    The catalog's word and each capability; and each capability on the
    routing file as shipped, where the driver of the kind adds an admission
    input to that file for every other play.
    """

    if not matrix_columns.admission_inputs(kind):
        return ADMISSION_SUBJECTS
    return (
        *ADMISSION_SUBJECTS,
        *(as_shipped(capability) for capability in matrix_columns.ROLE_BY_CAPABILITY),
    )


def expected_admission(
    cell: Cell, pins: Sequence[Mapping[str, str]] | None = None
) -> dict[str, str]:
    """What an admission cell must see of each of its subjects.

    A column that brings its own profile (a toy column) is built by its
    driver and stands behind no routing gate: every capability is admitted,
    and no catalog names it. A real column is held to its pin.
    """

    if cell.column.kind == TOY_KIND:
        return {name: ADMITTED for name in matrix_columns.ROLE_BY_CAPABILITY}
    rows = load_inventory().admission if pins is None else pins
    for row in rows:
        if row["cell"] == cell.id:
            return {name: row[name] for name in admission_subjects(cell.column.kind)}
    raise MatrixSetupError(f"admission cell {cell.id} has no pin in inventory.yaml")


@functools.cache
def cost_bounded_guarded_cells() -> frozenset[str]:
    """The guarded result cells whose route opts into a cost upper bound.

    Where clause C5 holds a result to the marks of a diagnostic one: read
    from the routing file of each cell's role, as the runner reads it.
    """

    return frozenset(
        cell.id
        for cell in behaviour_cells()
        if cell.lens == GUARDED
        and not SCENARIOS[cell.scenario].canary
        and not SCENARIOS[cell.scenario].admission
        and declarations.cost_bounded(
            cell.column,
            matrix_columns.ROLE_BY_CAPABILITY[SCENARIOS[cell.scenario].capability],
        )
    )


@functools.cache
def answered_from(cell: str) -> tuple[str, ...]:
    """The envelopes the dialect of a cell's kind answers the cell from, by name.

    Read without playing the cell: the script of the variant that the routed
    profile of the cell's column selects, each step rendered by the driver of
    the kind against a blank request. The optional tail that a declared
    difference adds is not in it. A static cell, an admission cell and a
    cell of a kind without a driver are answered from nothing.
    """

    kind, scenario, role, _lens = ledger.parse_cell_id(cell)
    defined = SCENARIOS.get(scenario)
    if defined is None or defined.admission or kind not in drivers.load():
        return ()
    columns = [item for item in all_columns() if (item.kind, item.role) == (kind, role)]
    if not columns:
        raise MatrixSetupError(f"cell {cell} names no column")
    variants = declarations.select(defined.differences, declarations.routed(columns[0]))
    return drivers.get(kind).answers(plan_for(defined, variants).script)


def grade_of(cell: str) -> str | None:
    """The grade of the weakest envelope a cell is answered from, if any.

    What static cell H03 holds a gap about a provider's replies to: such a
    gap cannot close on a cell while this is weaker than its record allows.
    """

    return envelopes.weakest(answered_from(cell))


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
        wire_differences={D_CDX_SERVED: matrix_columns.UNNAMED_SERVED_MODEL_KINDS},
        route_differences={D_COST_BOUND_UNRANKED: cost_bounded_guarded_cells()},
        grade=grade_of,
        proofs_leaned_on=proofs_leaned_on,
        required_proofs=required_proof_items(),
    )
