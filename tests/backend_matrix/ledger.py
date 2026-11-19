# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The ledger: known gaps as strict expected failures, and the gap registry.

A cell computes its deviation codes; ``ledger.yaml`` says what is expected.

- The *comparator* is the only place that may report an expected failure. A
  cell with an entry is an expected failure only when its codes are exactly the
  entry's, under the declared values the entry was observed under. It fails
  with ``decl.changed`` when the declaration moved, with "gap fixed" when it no
  longer deviates and with "different deviation" for any other set.
- The *gap registry* is permanent. It lists every cell of a gap with the
  declared values it was observed under and must close under, so a fix that
  deletes its entry leaves the passing cell pinned, and a declaration cannot
  be weakened to close a gap unless the record names that change.
- A *declared difference* is a named alternative contract with a product
  citation, an owner and a date. Either a variant of a difference table
  asserts it (a declared value selects it), or a scenario defines it and the
  entry lists the cells on which it is in force. It is counted apart from
  the known gaps and replaces no contract clause.
- ``integrity_problems`` is static cell H03: registry, entries, declared
  differences, inventory and oracle rules, over a :class:`World` that tests
  may build synthetically.
- Only real-kind behaviour cells and the static cells H01 and H02 may be known
  gaps. The other static cells are the matrix's own integrity checks:
  ``judge_static`` never consults the ledger for them, and H03 rejects an
  entry on one or on a toy cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Callable, Mapping, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

from .contract import CLAUSES, Deviation, clause_of
from .drivers.base import MatrixSetupError

LEDGER_PATH = Path(__file__).with_name("ledger.yaml")
LEDGER_SCHEMA = "crucible.backend-matrix.ledger.v1"
STATIC_LENS = "static"
ALL_KINDS = "all"
NO_ROLE = "-"
# The code-fixable high-severity gaps of the gap list (design section 9.2).
CODE_FIXABLE_HIGH_GAPS: tuple[str, ...] = (
    "G02", "G03", "G04", "G14", "G15", "G16",
    "G17", "G18", "G20", "G28", "G29", "G30",
)  # fmt: skip
ORIGINS = frozenset({"gap list", "reference", "matrix"})
SEVERITIES = frozenset({"high", "medium", "low"})
# Sets whose absence on a kind the H02 cell reports; pack B has no such code.
H02_SETS = frozenset({"starter", "pack_a"})
# The only static cells that report a product gap and so may be known gaps.
# Every other static cell is one of the matrix's own integrity checks.
LEDGERABLE_STATIC = frozenset({"H01", "H02"})


def cell_id(kind: str, scenario: str, role: str, lens: str) -> str:
    return f"{kind}/{scenario}/{role}/{lens}"


def parse_cell_id(value: str) -> tuple[str, str, str, str]:
    parts = value.split("/")
    if len(parts) != 4 or not all(parts):
        raise MatrixSetupError(f"cell id {value!r} is not <kind>/<scenario>/<role>/<lens>")
    return parts[0], parts[1], parts[2], parts[3]


@dataclass(frozen=True)
class RegistryCell:
    cell: str
    observed_under: Mapping[str, str]
    closes_under: Mapping[str, str]
    predicted: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeclarationChange:
    field: str
    before: str
    after: str
    citation: str


@dataclass(frozen=True)
class GapRecord:
    id: str
    title: str
    origin: str
    severity: str
    cells: tuple[RegistryCell, ...] = ()
    column_wide: tuple[tuple[str, str], ...] = ()
    declaration_change: DeclarationChange | None = None
    ruling: str | None = None


@dataclass(frozen=True)
class Entry:
    """A known gap on one cell: each code attributed to one registry gap."""

    cell: str
    observed_under: Mapping[str, str]
    codes: Mapping[str, str]
    ruling: str | None = None


@dataclass(frozen=True)
class ColumnWide:
    """A known gap that shows as one clause code on every cell of a kind."""

    kind: str
    code: str
    gap: str
    ruling: str | None = None


@dataclass(frozen=True)
class DeclaredDifference:
    id: str
    citation: str
    owner: str
    date: str
    alternative: str
    # The cells it is in force on, when no declared value selects it.
    cells: tuple[str, ...] = ()
    ruling: str | None = None


@dataclass(frozen=True)
class Ledger:
    ceilings: Mapping[str, int]
    max_declared_differences: int
    registry: Mapping[str, GapRecord]
    entries: tuple[Entry, ...] = ()
    column_wide: tuple[ColumnWide, ...] = ()
    differences: tuple[DeclaredDifference, ...] = ()

    def entry(self, cell: str) -> Entry | None:
        for item in self.entries:
            if item.cell == cell:
                return item
        return None

    def differences_for(self, cell: str) -> tuple[str, ...]:
        """The declared differences in force on ``cell``, in ledger order."""

        return tuple(item.id for item in self.differences if cell in item.cells)


# -- loading ---------------------------------------------------------------


def _fields(
    value: Any, where: str, required: set[str], optional: AbstractSet[str] = frozenset()
) -> dict:
    if not isinstance(value, dict):
        raise MatrixSetupError(f"ledger: {where} must be a mapping")
    missing, unknown = required - set(value), set(value) - required - optional
    if missing or unknown:
        raise MatrixSetupError(
            f"ledger: {where} has missing fields {sorted(missing)} and unknown "
            f"fields {sorted(unknown)}"
        )
    return value


def _values(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise MatrixSetupError(f"ledger: {where} must map declared fields to values")
    return dict(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatrixSetupError(f"ledger: {where} must be non-empty text")
    return value


def parse(document: Any) -> Ledger:
    """Build a ledger from the YAML document; any unknown shape is an error."""

    root = _fields(
        document,
        "the document",
        {"schema_version", "ceilings", "registry", "known_gaps", "declared_differences"},
    )
    if root["schema_version"] != LEDGER_SCHEMA:
        raise MatrixSetupError(f"ledger: schema_version must be {LEDGER_SCHEMA!r}")
    ceilings = _fields(
        root["ceilings"], "ceilings", {"max_known_gaps", "max_declared_differences"}
    )
    registry: dict[str, GapRecord] = {}
    for gap, raw in (root["registry"] or {}).items():
        where = f"registry.{gap}"
        record = _fields(
            raw,
            where,
            {"title", "origin", "severity"},
            {"cells", "column_wide", "declaration_change", "ruling"},
        )
        if record["origin"] not in ORIGINS or record["severity"] not in SEVERITIES:
            raise MatrixSetupError(f"ledger: {where} has an unknown origin or severity")
        cells = []
        for raw_cell in record.get("cells") or ():
            item = _fields(
                raw_cell,
                f"{where}.cells[]",
                {"cell", "observed_under", "closes_under"},
                {"predicted"},
            )
            parse_cell_id(_text(item["cell"], f"{where}.cells[].cell"))
            cells.append(
                RegistryCell(
                    cell=item["cell"],
                    observed_under=_values(item["observed_under"], f"{where} observed_under"),
                    closes_under=_values(item["closes_under"], f"{where} closes_under"),
                    predicted=tuple(item.get("predicted") or ()),
                )
            )
        change = None
        if record.get("declaration_change") is not None:
            raw_change = _fields(
                record["declaration_change"],
                f"{where}.declaration_change",
                {"field", "from", "to", "citation"},
            )
            change = DeclarationChange(
                field=raw_change["field"],
                before=raw_change["from"],
                after=raw_change["to"],
                citation=_text(raw_change["citation"], f"{where} citation"),
            )
        registry[str(gap)] = GapRecord(
            id=str(gap),
            title=_text(record["title"], f"{where}.title"),
            origin=record["origin"],
            severity=record["severity"],
            cells=tuple(cells),
            column_wide=tuple(
                (
                    _fields(item, f"{where}.column_wide[]", {"kind", "code"})["kind"],
                    item["code"],
                )
                for item in record.get("column_wide") or ()
            ),
            declaration_change=change,
            ruling=record.get("ruling"),
        )
    entries: list[Entry] = []
    column_wide: list[ColumnWide] = []
    for raw in root["known_gaps"] or ():
        if isinstance(raw, dict) and "column_wide" in raw:
            item = _fields(raw, "known_gaps[]", {"column_wide"}, {"ruling"})
            scope = _fields(item["column_wide"], "column_wide", {"kind", "code", "gap"})
            column_wide.append(
                ColumnWide(scope["kind"], scope["code"], scope["gap"], item.get("ruling"))
            )
            continue
        item = _fields(raw, "known_gaps[]", {"cell", "observed_under", "codes"}, {"ruling"})
        parse_cell_id(_text(item["cell"], "known_gaps[].cell"))
        entries.append(
            Entry(
                cell=item["cell"],
                observed_under=_values(item["observed_under"], "known_gaps[] observed_under"),
                codes=_values(item["codes"], f"known_gaps[{item['cell']}].codes"),
                ruling=item.get("ruling"),
            )
        )
    differences = []
    for raw in root["declared_differences"] or ():
        item = _fields(
            raw,
            "declared_differences[]",
            {"id", "citation", "owner", "date", "alternative"},
            {"cells", "ruling"},
        )
        scoped = item.get("cells") or ()
        if not isinstance(scoped, (list, tuple)):
            raise MatrixSetupError("ledger: declared difference cells must be a list")
        for scoped_cell in scoped:
            parse_cell_id(_text(scoped_cell, "declared difference cell"))
        differences.append(
            DeclaredDifference(
                id=_text(item["id"], "declared difference id"),
                citation=_text(item["citation"], "declared difference citation"),
                owner=_text(item["owner"], "declared difference owner"),
                date=_text(str(item["date"]), "declared difference date"),
                # A difference without an alternative contract asserts nothing.
                alternative=_text(item["alternative"], "declared difference alternative"),
                cells=tuple(scoped),
                ruling=item.get("ruling"),
            )
        )
    return Ledger(
        ceilings={str(k): int(v) for k, v in (ceilings["max_known_gaps"] or {}).items()},
        max_declared_differences=int(ceilings["max_declared_differences"]),
        registry=registry,
        entries=tuple(entries),
        column_wide=tuple(column_wide),
        differences=tuple(differences),
    )


def load(path: Path = LEDGER_PATH) -> Ledger:
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")))


# -- the comparator ----------------------------------------------------------


class LedgerMismatch(AssertionError):
    """A cell's deviation codes are not what the ledger expects."""


def _fail(cell: str, verdict: str, report: str) -> None:
    raise LedgerMismatch(f"cell {cell}: {verdict}\n{report}".rstrip())


def _expected_failure(ledger: Ledger, cell: str, gaps: Sequence[str]) -> None:
    titles = "; ".join(
        f"{gap}: {ledger.registry[gap].title if gap in ledger.registry else '?'}"
        for gap in sorted(set(gaps))
    )
    pytest.xfail(titles)
    # Reached only under --runxfail, which turns the expected failure into one.
    raise LedgerMismatch(f"cell {cell}: known gap still open ({titles})")


def compare(
    ledger: Ledger,
    cell: str,
    codes: frozenset[str] | set[str],
    *,
    current: Mapping[str, str],
    evaluated: frozenset[str] = frozenset(),
    unasked: frozenset[str] = frozenset(),
    report: str = "",
) -> None:
    """Pass, report an expected failure, or fail: the three-way rule.

    ``current`` holds the cell's governing declared values now; ``evaluated``
    the clauses the cell evaluated and ``unasked`` the codes whose check the
    result's own flags switched off there. A column-wide code is owed where
    its clause was evaluated and its check was asked, and nowhere else.
    """

    kind = parse_cell_id(cell)[0]
    remaining = set(codes)
    gaps: list[str] = []
    for scope in ledger.column_wide:
        if (
            scope.kind != kind
            or clause_of(scope.code) not in evaluated
            or scope.code in unasked
        ):
            continue
        if scope.code not in remaining:
            _fail(
                cell,
                f"gap fixed: column-wide code {scope.code} ({scope.gap}) is absent; "
                "delete the column-wide entry",
                report,
            )
        remaining.discard(scope.code)
        gaps.append(scope.gap)
    entry = ledger.entry(cell)
    if entry is None:
        if remaining:
            _fail(cell, f"unexpected deviation {sorted(remaining)}", report)
        if gaps:
            _expected_failure(ledger, cell, gaps)
        return
    changed = sorted(
        name
        for name in set(entry.observed_under) | set(current)
        if entry.observed_under.get(name) != current.get(name)
    )
    if changed:
        codes_text = ", ".join(f"decl.changed:{name}" for name in changed)
        _fail(
            cell,
            f"{codes_text}: the entry was observed under {dict(entry.observed_under)}, "
            f"the cell now runs under {dict(current)}. A changed declaration never "
            "closes a gap; see the registry's declaration_change",
            report,
        )
    # A passing cell is never an expected failure, whatever its entry lists.
    if not remaining:
        _fail(cell, f"gap fixed, delete the entry ({sorted(entry.codes)})", report)
    if remaining == set(entry.codes):
        _expected_failure(ledger, cell, [*gaps, *entry.codes.values()])
    _fail(
        cell,
        "different deviation: "
        f"missing {sorted(set(entry.codes) - remaining)}, "
        f"unexpected {sorted(remaining - set(entry.codes))}",
        report,
    )


def judge_static(ledger: Ledger, cell: str, problems: Sequence[Deviation]) -> None:
    """The verdict of a static cell.

    H01 and H02 report product gaps and go through the comparator. Every other
    static cell is an integrity check of the matrix itself: it must be clean,
    and the ledger is not consulted, so no entry can make it an expected
    failure.
    """

    scenario = parse_cell_id(cell)[1]
    codes = {item.code for item in problems}
    report = "\n".join(f"  {item.code}: {item.detail}" for item in problems)
    if scenario in LEDGERABLE_STATIC:
        compare(ledger, cell, codes, current={}, report=report)
    elif codes:
        _fail(
            cell,
            f"integrity check failed with {sorted(codes)}; an integrity cell is "
            "never an expected failure",
            report,
        )


# -- integrity (static cell H03) -------------------------------------------


@dataclass(frozen=True)
class World:
    """What the ledger is checked against. ``cells.world()`` builds the real one."""

    generated: frozenset[str]
    columns: frozenset[tuple[str, str]]
    kinds: frozenset[str]
    scenario_sets: Mapping[str, str]
    scenario_lenses: Mapping[str, tuple[str, ...]]
    static_scenarios: frozenset[str]
    canary_scenarios: frozenset[str]
    enabled_sets: Mapping[str, tuple[str, ...]]
    current: Callable[[str], Mapping[str, str]]
    cell_counts: Mapping[str, int]
    pinned_counts: Mapping[str, int]
    toy_kind: str
    reference_kinds: frozenset[str]
    open_weight_kinds: frozenset[str]
    not_expressible: tuple[Mapping[str, str], ...] = ()
    not_evaluated: tuple[Mapping[str, str], ...] = ()
    liar_scenarios: frozenset[str] = frozenset()
    liar_clauses: frozenset[str] = frozenset()
    snapshot_values: Mapping[str, frozenset[str]] = field(default_factory=dict)
    toy_values: Mapping[str, frozenset[str]] = field(default_factory=dict)
    # Declared differences the scenarios assert: through a variant of a
    # difference table, or as an alternative in force on listed cells
    # (id -> the scenarios that define it).
    difference_variants: frozenset[str] = frozenset()
    difference_alternatives: Mapping[str, frozenset[str]] = field(default_factory=dict)


def _pack(world: World, scenario: str) -> str:
    if scenario in world.static_scenarios:
        return "pack_a"
    return "pack_a" if world.scenario_sets.get(scenario) in H02_SETS else "pack_b"


def _registry_problems(ledger: Ledger, world: World) -> list[Deviation]:
    out: list[Deviation] = []
    for gap in CODE_FIXABLE_HIGH_GAPS:
        record = ledger.registry.get(gap)
        if record is None or not record.cells:
            out.append(
                Deviation(
                    "static:registry_gap_missing",
                    f"high gap {gap} has no registry record with a defined cell",
                )
            )
    pinned = {
        (row.get("kind"), row.get("scenario")) for row in world.not_expressible
    }
    for record in ledger.registry.values():
        for item in record.cells:
            kind, scenario, role, lens = parse_cell_id(item.cell)
            where = f"{record.id} cell {item.cell}"
            static = scenario in world.static_scenarios
            if static:
                defined = lens == STATIC_LENS and (kind == ALL_KINDS or kind in world.kinds)
            else:
                defined = (
                    scenario in world.scenario_sets
                    and (kind, role) in world.columns
                    and lens in world.scenario_lenses.get(scenario, ())
                )
            if not defined:
                out.append(
                    Deviation(
                        "static:registry_cell_undefined",
                        f"{where} names no defined scenario, column and lens",
                    )
                )
                continue
            if record.severity == "high" and not item.predicted:
                out.append(
                    Deviation(
                        "static:registry_prediction_missing",
                        f"{where} of a high gap records no predicted codes",
                    )
                )
            if dict(item.closes_under) != dict(item.observed_under):
                change = record.declaration_change
                differing = {
                    name
                    for name in set(item.observed_under) | set(item.closes_under)
                    if item.observed_under.get(name) != item.closes_under.get(name)
                }
                named = (
                    change is not None
                    and differing == {change.field}
                    and item.observed_under.get(change.field) == change.before
                    and item.closes_under.get(change.field) == change.after
                )
                if not named:
                    out.append(
                        Deviation(
                            "static:registry_closes_under",
                            f"{where}: closes_under differs from observed_under on "
                            f"{sorted(differing)} and the record names no "
                            "declaration_change that accounts for it",
                        )
                    )
            if (kind, scenario) in pinned and record.ruling is None:
                out.append(
                    Deviation(
                        "static:not_expressible_registry_cell",
                        f"{where} is pinned not expressible without a ruling",
                    )
                )
            if static:
                if item.cell not in world.generated:
                    out.append(
                        Deviation(
                            "static:registry_cell_not_generated",
                            f"{where} is a static cell that no test generates",
                        )
                    )
                continue
            scenario_set = world.scenario_sets[scenario]
            if scenario_set in world.enabled_sets.get(kind, ()):
                if item.cell not in world.generated and record.ruling is None:
                    out.append(
                        Deviation(
                            "static:registry_cell_not_generated",
                            f"{where}: kind {kind} runs {scenario_set} but the cell "
                            "is not generated, and the record has no ruling",
                        )
                    )
            elif scenario_set not in H02_SETS:
                out.append(
                    Deviation(
                        "static:registry_cell_early",
                        f"{where}: a {scenario_set} cell can be recorded only once "
                        f"kind {kind} runs that set",
                    )
                )
            elif ledger.entry(cell_id(kind, "H02", NO_ROLE, STATIC_LENS)) is None:
                out.append(
                    Deviation(
                        "static:registry_cell_unreachable",
                        f"{where}: kind {kind} does not run {scenario_set} and its "
                        "H02 cell is not an expected failure",
                    )
                )
            if ledger.entry(item.cell) is None and dict(world.current(item.cell)) != dict(
                item.closes_under
            ):
                out.append(
                    Deviation(
                        "static:registry_closes_mismatch",
                        f"{where} has no entry and runs under "
                        f"{dict(world.current(item.cell))}, not under its "
                        f"closes_under {dict(item.closes_under)}",
                    )
                )
    return out


def _entry_problems(ledger: Ledger, world: World) -> list[Deviation]:
    out: list[Deviation] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for entry in ledger.entries:
        kind, scenario, _role, lens = parse_cell_id(entry.cell)
        where = f"entry {entry.cell}"
        counts[_pack(world, scenario)] = counts.get(_pack(world, scenario), 0) + 1
        if entry.cell in seen:
            out.append(Deviation("static:entry_duplicate", f"{where} appears twice"))
        seen.add(entry.cell)
        if entry.cell not in world.generated:
            out.append(
                Deviation("static:entry_unused", f"{where} names no generated cell")
            )
        if scenario in world.canary_scenarios:
            out.append(
                Deviation("static:entry_on_canary", f"{where}: the canary is never ledgered")
            )
        if kind == world.toy_kind or (
            scenario in world.static_scenarios and scenario not in LEDGERABLE_STATIC
        ):
            out.append(
                Deviation(
                    "static:entry_on_integrity_cell",
                    f"{where}: a toy cell and an integrity check of the matrix "
                    f"(every static cell but {sorted(LEDGERABLE_STATIC)}) are "
                    "never ledgered",
                )
            )
        if not entry.codes:
            out.append(Deviation("static:entry_without_codes", f"{where} expects nothing"))
        if kind in world.reference_kinds and lens != STATIC_LENS and not entry.ruling:
            out.append(
                Deviation(
                    "static:entry_ruling_missing",
                    f"{where} is on a reference column and records no owner ruling",
                )
            )
        for code, gap in entry.codes.items():
            try:
                clause = clause_of(code)
            except ValueError:
                clause = None
            if clause is None or clause == "D":
                out.append(
                    Deviation(
                        "static:entry_code_invalid",
                        f"{where}: {code!r} is not a code a known gap may expect",
                    )
                )
            record = ledger.registry.get(gap)
            listed = None if record is None else [
                item for item in record.cells if item.cell == entry.cell
            ]
            if not listed:
                out.append(
                    Deviation(
                        "static:entry_not_in_registry",
                        f"{where} attributes {code} to {gap}, whose registry record "
                        "does not list this cell",
                    )
                )
            elif dict(listed[0].observed_under) != dict(entry.observed_under):
                out.append(
                    Deviation(
                        "static:entry_observed_under",
                        f"{where}: observed_under differs from the registry's ({gap})",
                    )
                )
    for scope in ledger.column_wide:
        counts["pack_a"] = counts.get("pack_a", 0) + 1
        if scope.kind == world.toy_kind:
            out.append(
                Deviation(
                    "static:entry_on_integrity_cell",
                    f"column-wide {scope.code}: the toy kind is never ledgered",
                )
            )
        record = ledger.registry.get(scope.gap)
        if record is None or (scope.kind, scope.code) not in record.column_wide:
            out.append(
                Deviation(
                    "static:entry_not_in_registry",
                    f"column-wide {scope.code} on {scope.kind} is not listed under "
                    f"{scope.gap}'s column-wide scope",
                )
            )
    for pack, count in sorted(counts.items()):
        if count > ledger.ceilings.get(pack, 0):
            out.append(
                Deviation(
                    "static:ceiling_exceeded",
                    f"{count} known gaps in {pack}, ceiling {ledger.ceilings.get(pack, 0)}",
                )
            )
    if len(ledger.differences) > ledger.max_declared_differences:
        out.append(
            Deviation(
                "static:ceiling_exceeded",
                f"{len(ledger.differences)} declared differences, ceiling "
                f"{ledger.max_declared_differences}",
            )
        )
    return out


def _difference_problems(ledger: Ledger, world: World) -> list[Deviation]:
    """A declared difference is asserted by a scenario, and only where listed."""

    out: list[Deviation] = []
    declared = [item.id for item in ledger.differences]
    asserted = world.difference_variants | set(world.difference_alternatives)
    for name in sorted(asserted - set(declared)):
        out.append(
            Deviation(
                "static:difference_undeclared",
                f"a scenario asserts the alternative of {name}, which the ledger "
                "does not declare",
            )
        )
    for name in sorted({name for name in declared if declared.count(name) > 1}):
        out.append(Deviation("static:difference_duplicate", f"{name} is declared twice"))
    for item in ledger.differences:
        where = f"declared difference {item.id}"
        if item.id not in asserted:
            out.append(
                Deviation(
                    "static:difference_unused",
                    f"{where}: no scenario asserts its alternative contract",
                )
            )
            continue
        scenarios = world.difference_alternatives.get(item.id)
        if scenarios is None:
            # A declared value selects it; it has no cell list of its own.
            if item.cells:
                out.append(
                    Deviation(
                        "static:difference_cell_invalid",
                        f"{where} is selected by a declared value and lists cells",
                    )
                )
            continue
        if not item.cells:
            out.append(
                Deviation("static:difference_unused", f"{where} is in force on no cell")
            )
        for cell in item.cells:
            kind, scenario, _role, _lens = parse_cell_id(cell)
            if (
                item.cells.count(cell) > 1
                or cell not in world.generated
                or kind == world.toy_kind
                or scenario in world.canary_scenarios
                or scenario not in scenarios
            ):
                out.append(
                    Deviation(
                        "static:difference_cell_invalid",
                        f"{where} lists {cell}: it must be listed once, be a "
                        "generated cell of a real kind, not the canary, and its "
                        "scenario must define the alternative",
                    )
                )
    return out


def _inventory_problems(world: World) -> list[Deviation]:
    out: list[Deviation] = []
    for kind in sorted(set(world.cell_counts) | set(world.pinned_counts)):
        got, pinned = world.cell_counts.get(kind, 0), world.pinned_counts.get(kind)
        if got != pinned:
            out.append(
                Deviation(
                    "static:inventory_count",
                    f"kind {kind} generates {got} cell(s), inventory.yaml pins {pinned}",
                )
            )
    for row in world.not_expressible:
        if row.get("kind") == world.toy_kind:
            out.append(
                Deviation(
                    "static:not_expressible_on_toy",
                    f"{row.get('scenario')} cannot be pinned on the toy kind",
                )
            )
    pack_a = [name for name, item in world.scenario_sets.items() if item in H02_SETS]
    groups: dict[str, set[str]] = {}
    for name in pack_a:
        groups.setdefault(re.sub(r"[a-z]\Z", "", name), set()).add(name)
    for kind in sorted(world.open_weight_kinds):
        pinned_here = {
            row.get("scenario") for row in world.not_expressible if row.get("kind") == kind
        }
        for group, forms in sorted(groups.items()):
            if forms and forms <= pinned_here:
                out.append(
                    Deviation(
                        "static:not_expressible_pack_a",
                        f"kind {kind} would run no cell of pack A scenario {group}",
                    )
                )
    for row in world.not_evaluated:
        on_toy = row.get("kind") in (world.toy_kind, "*")
        if on_toy and row.get("scenario") not in world.canary_scenarios:
            out.append(
                Deviation(
                    "static:not_evaluated_on_toy",
                    f"clauses {row.get('clauses')} of {row.get('scenario')} may not "
                    "be pinned not evaluated on the toy kind",
                )
            )
    return out


def _oracle_problems(world: World) -> list[Deviation]:
    out: list[Deviation] = []
    enabled = {
        name
        for name, item in world.scenario_sets.items()
        if any(item in sets for sets in world.enabled_sets.values())
    }
    for name in sorted(enabled - world.liar_scenarios):
        out.append(
            Deviation("static:scenario_without_liar", f"scenario {name} has no liar")
        )
    for clause in CLAUSES:
        if clause not in world.liar_clauses:
            out.append(
                Deviation("static:clause_not_tripped", f"no liar trips clause {clause}")
            )
    for name, values in sorted(world.snapshot_values.items()):
        for value in sorted(values - world.toy_values.get(name, frozenset())):
            out.append(
                Deviation(
                    "static:value_without_honest_pass",
                    f"{name}={value} is declared by a column and no toy column plays it",
                )
            )
    return out


def integrity_problems(ledger: Ledger, world: World) -> list[Deviation]:
    """H03: every rule that keeps the ledger, the inventory and the oracle honest."""

    return [
        *_registry_problems(ledger, world),
        *_entry_problems(ledger, world),
        *_difference_problems(ledger, world),
        *_inventory_problems(world),
        *_oracle_problems(world),
    ]
