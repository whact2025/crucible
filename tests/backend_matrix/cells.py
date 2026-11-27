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
import os
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
from .drivers.base import KeeperDriver, MatrixSetupError
from .scenarios import (
    D_CDX_SERVED,
    D_COST_BOUND_UNRANKED,
    GUARDED,
    PACK_A,
    PACK_B,
    SCENARIOS,
    STARTER,
    Plan,
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
    "S41",
    "S44",
    "S45",
    # The turn limit without an error (on this route a finite script that
    # ends with its answer) and the USD budget inside a tool loop. On the
    # second more than the usage code moves with the reading: the route
    # compares the budget with the cost it reports, so the reading also
    # decides whether the budget is typed. The proof holds exactly the
    # three codes of a budget that is not typed beside the usage code where
    # the plan has a budget, and nothing beside it where it has none; the
    # entry of that cell names the record for all four, by a ruling.
    "S10b",
    "S14",
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
# The cell of the official Codex column whose model calls a tool that no
# server lists: where known gap R17 shows.
UNLISTED_TOOL_CALL_ON_THE_OFFICIAL_CODEX_ROUTE = ledger.cell_id(
    matrix_columns.CODEX_OFFICIAL_KIND, "S06", "primary", "raw"
)
# The cell of the app-server Codex column whose result is that of the same
# denial: where known gap R08 shows.
DISPATCH_DENIAL_ON_THE_APP_SERVER_CODEX_ROUTE = ledger.cell_id(
    matrix_columns.CODEX_LEGACY_KIND, "S12", "primary", "raw"
)
# The cell of the app-server Codex column whose billed final turn shows no
# message: where known gap R16 shows.
EMPTY_FINAL_TURN_ON_THE_APP_SERVER_CODEX_ROUTE = ledger.cell_id(
    matrix_columns.CODEX_LEGACY_KIND, "S44", "primary", "raw"
)
# The cell of the same column whose one response completes with nothing: where
# known gap R20 shows.
EMPTY_COMPLETION_ON_THE_APP_SERVER_CODEX_ROUTE = ledger.cell_id(
    matrix_columns.CODEX_LEGACY_KIND, "S25", "primary", "raw"
)
# The cells of the Claude kind that attach a ToolSet with write names and end
# in a result of the attempt: where known gap R13 shows. A scenario of the
# set that runs on the primary column only has its one cell here.
WRITES_ON_THE_CLAUDE_COLUMNS: tuple[str, ...] = (
    *(
        ledger.cell_id(matrix_columns.CLAUDE_KIND, "S40", role, "raw")
        for role in ("primary", "secondary")
    ),
    *(
        ledger.cell_id(matrix_columns.CLAUDE_KIND, scenario, "primary", "raw")
        for scenario in ("S41", "S44", "S45")
    ),
)
# The cells of the Kimi kind on which the route ends the invocation because
# of a tool call it will not run: where known gap M18 shows. A tool that is
# not attached, and a call whose argument text is no JSON object.
REFUSED_TOOL_CALLS_ON_THE_KIMI_ROUTE: tuple[str, ...] = tuple(
    ledger.cell_id(matrix_columns.KIMI_KIND, scenario, "primary", "raw")
    for scenario in ("S06", "S05")
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
# unconfirmed: its usage code shows only while the cost is flagged complete, so
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
# On the same route a billed turn whose last model response shows no message
# ends as a transport error (known gap R16). The two codes of its cell are
# those of any error that ends the turn after its accounting, a protocol
# error of another text and a retriable error among them: only a proof
# shows that the error is not retriable and carries the sentence the client
# library of the route raises for a turn without a final assistant message.
# The same raise ends an empty completion on that route as a transport error
# that is not retried (known gap R20). The one code of its cell says that the
# error is not retriable and not why: a protocol error of another text shows
# the same code, so a proof of its own holds the sentence, and that the one
# billed response is reported.
# On the official route a refused resume is typed by the code of the error
# object the tool answers with, and that code is emulated, not observed
# (known gap R21, unconfirmed). The three codes of its cell do not say that
# they turn on it: the test that plays the cell under the emulated code and
# under the code of invalid parameters is held here too, as it was written
# before the record.
# Where the model calls a tool that no server lists, the official Codex
# route ends the call with a transport error and no usage (known gap R17).
# The codes of its cell are those of any error of the route that reports no
# usage, a turn the tool reports as failed and a tool outside the attached
# set that ran among them: only a proof shows that the worker handed over a
# completed turn with one failed item of the unlisted name and that the
# adapter's check of the item names ended the call.
# On a write the trace of the Claude adapter binds a tool-authority digest
# that is not the shared one (known gap R13). The one code of its cells says
# that the digest differs, not why: only the proof that plays every write
# cell of the kind shows that the value is the digest of the same tools
# without a path scope, and that nothing else stands behind the code.
# Where the Kimi route ends a call because of a tool call it will not run
# (known gap M18), the codes of the cell are those of any call that ends in
# an error of the worker after its first billed reply: a worker that dies on
# the reply or a bridge that is broken would show them too. Only the proof
# holds each cell to its cause: one matched request, nothing dispatched, and
# the text of the refusal.
REQUIRED_PROOFS: Mapping[str, tuple[str, ...]] = {
    MODULE_BY_KIND[matrix_columns.CLAUDE_KIND]: tuple(
        "test_the_trace_of_a_write_binds_the_digest_of_the_same_tools_without_a_path_scope"
        f"[{cell}]"
        for cell in WRITES_ON_THE_CLAUDE_COLUMNS
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
        *(
            f"test_the_call_ends_where_the_route_refuses_a_tool_call_it_will_not_run[{cell}]"
            for cell in REFUSED_TOOL_CALLS_ON_THE_KIMI_ROUTE
        ),
    ),
    MODULE_BY_KIND[matrix_columns.CODEX_LEGACY_KIND]: (
        "test_the_result_of_a_dispatch_budget_denial_reports_every_response_the_provider_billed",
        "test_a_billed_turn_that_shows_no_message_ends_in_the_protocol_error_of_the_client"
        "_and_is_not_retried",
        "test_an_empty_completion_ends_in_the_protocol_error_of_the_client_and_is_not_retried",
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
        "test_a_call_for_a_tool_no_server_lists_ends_in_the_adapters_check_of_its_failed_item",
        "test_the_cells_of_a_refused_thread_rest_on_the_code_the_tool_is_emulated_to_answer_with",
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
    # cell (""), one for every scenario, one for every column, or one for
    # every cell.
    per: Literal["", "scenario", "column", "cell"] = ""
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
        if self.per == "cell":
            return f"{self.test}[{cell}]"
        return self.test


# The proofs that the entries of a registry record lean on. A proof holds what
# the codes of an entry do not say: the cause of a code, or a number that no
# clause compares while the entry stands. So an entry must not outlive it:
# static cell H03 fails when an entry attributes a code to a record of this
# table and the verifier does not require the item of the proof that plays the
# entry's cell (static:entry_proof_not_required). A change that deletes a
# proof and its name from the list above therefore deletes the entries that
# lean on it, or fails. Each proof refuses, for its part, an entry that
# attributes a code to its record on a cell it does not play; the one
# exception is the test R21 leans on, which was written before its record
# and is taken as it stands, so there the words of the entry say what the
# record covers.
# The official Codex route: while the column-wide flag code of R04 stands,
# clause C6 compares no reported token with the provider's ledger on any cell
# of the kind, and the test of a turn of one response does. The usage code of
# R05 rests on the reading of `last`: the test that plays both readings on
# the raw cell of the entry's scenario and, for an entry on a guarded cell,
# its twin. The result of a dispatch-budget denial is compared by no clause
# on either Codex route while its entry stands (R08, R09). The digest code
# of a write on the Claude columns says that the digest differs, not why
# (R13). The two codes of a billed turn without a message on the app-server
# route say neither that its error is not retriable nor what its text is
# (R16). The one code of an empty completion on that route does not say why
# its error is not retriable (R20). The three codes of a refused resume on
# the official route do not say that they rest on the code the tool is
# emulated to answer with (R21). The codes of a call the Kimi route ends
# because of a tool call it will not run say that the call ended in an
# error, not which (M18). Nor do the codes of the call the official Codex
# route ends for the failed item of a tool that no server lists say what
# the worker handed over (R17).
_CLAUDE_MODULE = MODULE_BY_KIND[matrix_columns.CLAUDE_KIND]
_KIMI_MODULE = MODULE_BY_KIND[matrix_columns.KIMI_KIND]
_APP_SERVER_MODULE = MODULE_BY_KIND[matrix_columns.CODEX_LEGACY_KIND]
_OFFICIAL_MODULE = MODULE_BY_KIND[matrix_columns.CODEX_OFFICIAL_KIND]
RECORD_PROOFS: Mapping[str, tuple[RecordProof, ...]] = {
    "M18": (
        RecordProof(
            _KIMI_MODULE,
            "test_the_call_ends_where_the_route_refuses_a_tool_call_it_will_not_run",
            per="cell",
        ),
    ),
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
            per="cell",
        ),
    ),
    "R16": (
        RecordProof(
            _APP_SERVER_MODULE,
            "test_a_billed_turn_that_shows_no_message_ends_in_the_protocol_error_of_the_client"
            "_and_is_not_retried",
        ),
    ),
    "R17": (
        RecordProof(
            _OFFICIAL_MODULE,
            "test_a_call_for_a_tool_no_server_lists_ends_in_the_adapters_check_of_its_failed_item",
        ),
    ),
    "R20": (
        RecordProof(
            _APP_SERVER_MODULE,
            "test_an_empty_completion_ends_in_the_protocol_error_of_the_client_and_is_not_retried",
        ),
    ),
    "R21": (
        RecordProof(
            _OFFICIAL_MODULE,
            "test_the_cells_of_a_refused_thread_rest_on_the_code_the_tool_is_emulated_to_answer_with",
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


def envelopes_of(kind: str, plan: Plan) -> tuple[str, ...]:
    """The envelopes the dialect of ``kind`` answers a play of ``plan`` from, by name.

    The steps of the script, each rendered by the driver of the kind against
    a blank request. Where the invocation of the plan asks to continue a
    conversation, the route answers the request and a tool beside the
    adapter keeps the conversations of the wire, that tool is asked before
    any model is. It holds no conversation of an earlier call, so what it
    says of a handle it does not hold is what the cell is answered from,
    whatever the script holds. A route that refuses the request before
    transport asks nobody.
    """

    driver = drivers.get(kind)
    names = driver.answers(plan.script)
    if (
        plan.conversation is not None
        and plan.expect.outcome != "refused"
        and isinstance(driver, KeeperDriver)
    ):
        names = (*driver.refuses(), *names)
    return names


@functools.cache
def answered_from(cell: str) -> tuple[str, ...]:
    """The envelopes the dialect of a cell's kind answers the cell from, by name.

    Read without playing the cell: the plan of the variant that the routed
    profile of the cell's column selects, read as :func:`envelopes_of` reads
    a plan. The optional tail that a declared difference adds is not in it.
    A static cell, an admission cell and a cell of a kind without a driver
    are answered from nothing.
    """

    kind, scenario, role, _lens = ledger.parse_cell_id(cell)
    defined = SCENARIOS.get(scenario)
    if defined is None or defined.admission or kind not in drivers.load():
        return ()
    columns = [item for item in all_columns() if (item.kind, item.role) == (kind, role)]
    if not columns:
        raise MatrixSetupError(f"cell {cell} names no column")
    variants = declarations.select(defined.differences, declarations.routed(columns[0]))
    return envelopes_of(kind, plan_for(defined, variants))


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


def refuse_items_without_fixtures(
    items: Sequence[Any], root: Path, fixtures: Sequence[str]
) -> None:
    """Refuse the run when an item under ``root`` would run without ``fixtures``.

    The fixtures a ``conftest.py`` applies to every test are in force for an
    item by the directory pytest collected it under. A module that pytest
    collects after it has left that directory (a call that names files of the
    directory on both sides of a file outside it) gets none of them, and
    nothing says so: its tests would run without the network guard and
    outside the isolated environment. An item states the fixtures that are
    in force for it, so such a run is refused here, before any test.
    """

    wanted = set(fixtures)
    unguarded = [
        item.nodeid
        for item in items
        if root in Path(str(item.path)).parents
        and not wanted <= set(getattr(item, "fixturenames", ()))
    ]
    if unguarded:
        raise pytest.UsageError(
            f"the fixtures of the backend matrix ({', '.join(fixtures)}) are not in "
            f"force for {len(unguarded)} item(s) of the package, the first "
            f"{unguarded[0]}: pytest collected them after it had left the "
            "directory of the package, and they would run without its network "
            "guard and outside its isolated environment. Call the package as "
            "one directory, or name its modules side by side in one call, never "
            "on both sides of a file outside the package"
        )


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


# How much longer than the invocation deadline of its scenario the test item
# of a cell may run, where that deadline is not below the suite's limit.
LIMIT_MARGIN_SECONDS = 120


def suite_limit(config: Any) -> float | None:
    """The time after which the suite ends a test, as the timeout plug-in reads it.

    The command line, then the environment, then the configuration file;
    ``None`` where none of them states a limit or the plug-in is not loaded.
    """

    sources = (
        lambda: config.getoption("timeout", None),
        lambda: os.environ.get("PYTEST_TIMEOUT"),
        lambda: config.getini("timeout"),
    )
    for read in sources:
        try:
            value = read()
        except ValueError:
            continue  # the option is not registered: no plug-in, no limit
        if value not in (None, ""):
            return float(value) or None
    return None


def limit_long_cells(items: Sequence[Any], root: Path, limit: float | None) -> None:
    """Give the test item of a cell with a long invocation deadline a limit above it.

    The suite ends every test after ``limit`` seconds. A cell whose scenario
    passes an invocation deadline that is not below that limit could never be
    ended by its own deadline: the suite would end the item first, as a
    plain failure, and the verifier refuses a plain failure of a cell the
    ledger expects to fail. The item of such a cell is marked with the
    deadline plus :data:`LIMIT_MARGIN_SECONDS`. No other item is touched.
    """

    if not limit:
        return
    for item in items:
        path = Path(str(item.path))
        if root not in path.parents:
            continue
        cell = cell_of_item(path.stem, str(item.name))
        scenario = None if cell is None else SCENARIOS.get(ledger.parse_cell_id(cell)[1])
        deadline = None if scenario is None else scenario.policy.timeout_seconds
        if deadline and deadline >= limit:
            item.add_marker(pytest.mark.timeout(deadline + LIMIT_MARGIN_SECONDS))


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
