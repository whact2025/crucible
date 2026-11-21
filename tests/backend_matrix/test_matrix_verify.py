# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The cell-inventory verifier, shown on synthetic and on real test reports.

Its rules are shown on a synthetic inventory and ledger, so that no test here
depends on which gaps the shipped ledger holds today. A complete run is
accepted. Each way a cell can fail to run, or to mean what the ledger says, is
rejected: a missing cell, a deselected module, a skip, the item of an empty
parameter set, an expected failure without an entry, an entry whose cell
passed, a column-wide gap whose cell passed or that is named where it is not
owed, a duplicate item, a count that differs from the pinned one.

The shipped inventory and ledger are read only where the real program runs:
the item names the verifier expects are compared with what pytest collects,
the reports of two real runs are read back, and the command runs as the CI
step runs it. Those tests assert the shape of what they read and that it
agrees with the shipped ledger, never a gap id or a code.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

from . import cells, contract, envelopes, ledger, verify
from . import columns as matrix_columns
from . import report as matrix_report
from .cells import Item
from .columns import HOSTED_GATEWAY_KIND, TOY_KIND
from .scenarios import SCENARIOS
from .verify import EXPECTED_FAILURE, PASSED, Seen

REPOSITORY = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"
COMMAND = "python -m tests.backend_matrix.verify"

# -- a synthetic inventory and ledger ---------------------------------------------
#
# One kind with a column-wide gap, one without, and two static cells. The ids
# of the scenarios are real (the canary is read from the catalogue); the
# kinds, modules and gaps are stand-ins.

_WIDE, _PLAIN = "standin:wide", "standin:plain"
_MODULE, _STATIC = "test_matrix_standin", "test_matrix_standin_static"
_WIDE_CODE = "pricing:none"
_CANARY = ledger.cell_id(_WIDE, "S00", "primary", "raw")
_WIDE_ONLY = ledger.cell_id(_WIDE, "S01", "primary", "raw")
_ENTRY = ledger.cell_id(_WIDE, "S03", "primary", "guarded")
_UNASKED = ledger.cell_id(_WIDE, "S04", "primary", "raw")
_RAISED = ledger.cell_id(_WIDE, "S10a", "primary", "raw")
_PINNED = ledger.cell_id(_WIDE, "S23c", "primary", "raw")
_PASSING = ledger.cell_id(_PLAIN, "S01", "primary", "raw")
_STATIC_GAP = ledger.cell_id(_PLAIN, "H02", ledger.NO_ROLE, ledger.STATIC_LENS)
_STATIC_OK = ledger.cell_id(ledger.ALL_KINDS, "H03", ledger.NO_ROLE, ledger.STATIC_LENS)

# What an honest run records for each cell: the gaps its expected failure
# names, none where the cell passes. Written down here, not computed by the
# verifier, so that the verifier is compared with something.
_HONEST: Mapping[str, tuple[str, ...]] = {
    _CANARY: (),  # checks the seam and C0 only
    _WIDE_ONLY: ("G92",),  # no entry of its own
    _ENTRY: ("G91", "G92"),  # its own entry, and the column-wide gap beside it
    _UNASKED: ("G93",),  # its entry shows the cost flagged incomplete
    _RAISED: ("G94",),  # its entry is a raise: no result clause ran
    _PINNED: (),  # the inventory pins the column-wide gap as not owed
    _PASSING: (),
    _STATIC_GAP: ("G95",),
    _STATIC_OK: (),
}


def _record(title: str, *listed: str, column_wide: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"title": title, "origin": "gap list", "severity": "medium"}
    if listed:
        record["cells"] = [
            {"cell": cell, "observed_under": {}, "closes_under": {}} for cell in listed
        ]
    if column_wide:
        record["column_wide"] = [{"kind": _WIDE, "code": _WIDE_CODE}]
    return record


def _entry(cell: str, gap: str, *codes: str) -> dict[str, Any]:
    return {"cell": cell, "observed_under": {}, "codes": {code: gap for code in codes}}


KNOWN = ledger.parse(
    {
        "schema_version": ledger.LEDGER_SCHEMA,
        "ceilings": {"max_known_gaps": {"pack_a": 5}, "max_declared_differences": 0},
        "registry": {
            "G91": _record("a gap of one cell", _ENTRY),
            "G92": _record("a gap on every cell of a kind", column_wide=True),
            "G93": _record("a gap that clears the accounting flags", _UNASKED),
            "G94": _record("a gap that raises", _RAISED),
            "G95": _record("a gap of a static cell", _STATIC_GAP),
        },
        "known_gaps": [
            _entry(_ENTRY, "G91", "guard:revoked"),
            _entry(
                _UNASKED, "G93", "flag:token_usage_complete=False", "flag:cost_complete=False"
            ),
            _entry(_RAISED, "G94", "raise:IndexError@after_request/standin.adapter"),
            _entry(_STATIC_GAP, "G95", "static:no_driven_column"),
            {"column_wide": {"kind": _WIDE, "code": _WIDE_CODE, "gap": "G92"}},
        ],
        "declared_differences": [],
    }
)
NOT_OWED: tuple[Mapping[str, str], ...] = (
    {"cell": _PINNED, "code": _WIDE_CODE, "reason": "the scenario waives the cost flag"},
)
EXPECTED: tuple[Item, ...] = (
    *(
        Item(_MODULE, f"{cells.BEHAVIOUR_TEST}[{cell}]", cell)
        for cell in _HONEST
        if not cell.endswith(ledger.STATIC_LENS)
    ),
    Item(_STATIC, f"test_h02_every_backend_is_driven[{_STATIC_GAP}]", _STATIC_GAP),
    Item(_STATIC, "test_h03_ledger_and_oracle_integrity", _STATIC_OK),
    Item(_MODULE, cells.CANARY_CHECK),
    Item(_STATIC, cells.STATIC_CHECK),
)
COUNTS: Mapping[str, int] = {_WIDE: 6, _PLAIN: 1}


def _named(known: ledger.Ledger, gaps: Sequence[str]) -> str:
    """The reason of an expected failure for ``gaps``, as the comparator words it."""

    return "; ".join(f"{gap}: {known.registry[gap].title}" for gap in gaps)


def _complete_run() -> list[Seen]:
    """The records of a complete, honest run of the synthetic inventory."""

    modules = sorted({item.module for item in EXPECTED})
    cases: list[Seen] = []
    for item in EXPECTED:
        gaps = () if item.cell is None else _HONEST[item.cell]
        cases.append(
            Seen(
                item.module,
                item.name,
                EXPECTED_FAILURE if gaps else PASSED,
                _named(KNOWN, gaps),
                # One file per module, as a shard holds whole files.
                f"test-results-{modules.index(item.module)}.xml",
            )
        )
    return cases


def _write(root: Path, cases: Sequence[Seen]) -> Path:
    """Write ``cases`` as JUnit files, one per source, beside a foreign test."""

    root.mkdir(parents=True, exist_ok=True)
    suites: dict[str, ElementTree.Element] = {}
    for case in cases:
        suite = suites.get(case.source)
        if suite is None:
            suite = suites[case.source] = ElementTree.Element("testsuite", name="pytest")
            # A test outside the package is none of the verifier's business.
            other = ElementTree.SubElement(
                suite, "testcase", classname="tests.test_other", name="test_cell[x/S00/y/raw]"
            )
            ElementTree.SubElement(other, "skipped", type=verify.SKIP_TYPE, message="elsewhere")
        element = ElementTree.SubElement(
            suite,
            "testcase",
            classname=f"{cells.PACKAGE_MODULE}.{case.module}",
            name=case.name,
            time="0.01",
        )
        if case.outcome == EXPECTED_FAILURE:
            ElementTree.SubElement(
                element, "skipped", type=verify.XFAIL_TYPE, message=case.message
            )
        elif case.outcome == "skipped":
            ElementTree.SubElement(
                element, "skipped", type=verify.SKIP_TYPE, message=case.message
            )
        elif case.outcome != PASSED:
            ElementTree.SubElement(
                element, case.outcome.replace("failed", "failure"), message=case.message
            )
    for source, suite in suites.items():
        document = ElementTree.Element("testsuites")
        document.append(suite)
        ElementTree.ElementTree(document).write(root / source, encoding="utf-8")
    return root


def _found(root: Path, cases: Sequence[Seen], **changed: object) -> list[str]:
    """What the verifier finds in ``cases``, read back from the files it reads."""

    arguments: dict[str, object] = {
        "expected": EXPECTED,
        "known": KNOWN,
        "pinned_counts": COUNTS,
        "not_owed": NOT_OWED,
    }
    arguments.update(changed)
    paths = sorted(_write(root, cases).glob(verify.JUNIT_GLOB))
    return verify.problems(verify.read_junit(paths), **arguments)  # type: ignore[arg-type]


def _index(cases: Sequence[Seen], cell: str) -> int:
    name = next(item.name for item in EXPECTED if item.cell == cell)
    return next(index for index, case in enumerate(cases) if case.name == name)


def test_a_complete_run_is_accepted(tmp_path: Path) -> None:
    cases = _complete_run()

    assert _found(tmp_path, cases) == []
    assert len(list(tmp_path.glob(verify.JUNIT_GLOB))) == len({item.module for item in EXPECTED})
    # What a complete run holds: every item once; canary and closing checks passed.
    assert len({(case.module, case.name) for case in cases}) == len(cases) == len(EXPECTED)
    assert {case.outcome for case in cases} == {PASSED, EXPECTED_FAILURE}
    # What the verifier takes each cell to owe is what was written down above.
    for cell, gaps in _HONEST.items():
        owed, wide = verify.ledgered(KNOWN, cell, NOT_OWED)
        assert owed | wide == set(gaps), cell
        assert verify.named_gaps(_named(KNOWN, gaps)) == set(gaps)


Mutation = Callable[[list[Seen]], list[Seen]]


def _without(module: str) -> Mutation:
    return lambda cases: [case for case in cases if case.module != module]


def _changed(cell: str, **fields: str) -> Mutation:
    def change(cases: list[Seen]) -> list[Seen]:
        index = _index(cases, cell)
        return [*cases[:index], dataclasses.replace(cases[index], **fields), *cases[index + 1 :]]

    return change


def _expecting(cell: str, *gaps: str) -> Mutation:
    """``cell`` recorded as an expected failure that names ``gaps``."""

    return _changed(cell, outcome=EXPECTED_FAILURE, message=_named(KNOWN, gaps))


def _dropped(cell: str) -> Mutation:
    def drop(cases: list[Seen]) -> list[Seen]:
        index = _index(cases, cell)
        return [*cases[:index], *cases[index + 1 :]]

    return drop


def _twice(cell: str) -> Mutation:
    def repeat(cases: list[Seen]) -> list[Seen]:
        again = dataclasses.replace(cases[_index(cases, cell)], source="test-results-7.xml")
        return [*cases, again]

    return repeat


def _empty_parameter_set(cases: list[Seen]) -> list[Seen]:
    kept = [case for case in cases if case.module != _MODULE or "[" not in case.name]
    return [
        *kept,
        Seen(_MODULE, "test_cell[NOTSET]", "skipped", "got empty parameter set", "test-results-7.xml"),
    ]


@pytest.mark.parametrize(
    ("mutation", "named"),
    [
        (
            _dropped(_PASSING),
            [f"[{_PASSING}] did not run", f"kind {_PLAIN}: 0 cell item(s) ran, inventory.yaml pins 1"],
        ),
        (
            _without(_MODULE),
            [
                f"{_MODULE}::{cells.CANARY_CHECK} did not run",
                f"[{_CANARY}] did not run",
                f"kind {_WIDE}: 0 cell item(s) ran, inventory.yaml pins {COUNTS[_WIDE]}",
            ],
        ),
        (
            _changed(_PASSING, outcome="skipped", message="no runtime"),
            ["was skipped (no runtime)"],
        ),
        (_empty_parameter_set, ["is the item of an empty parameter set", "did not run"]),
        (_expecting(_PASSING, "G91"), [f"the ledger has no entry for cell {_PASSING}"]),
        (_expecting(_CANARY, "G92"), [f"the ledger has no entry for cell {_CANARY}"]),
        (
            _changed(_ENTRY, outcome=PASSED, message=""),
            ["passed although the ledger has an entry"],
        ),
        (
            _changed(_STATIC_GAP, outcome=PASSED, message=""),
            ["passed although the ledger has an entry"],
        ),
        (
            _changed(_WIDE_ONLY, outcome=PASSED, message=""),
            [f"passed although the ledger owes ['G92'] column-wide on cell {_WIDE_ONLY}"],
        ),
        (_expecting(_ENTRY, "G91"), ["is an expected failure for ['G91']", "owes ['G92']"]),
        (_expecting(_ENTRY, "G92"), ["is an expected failure for ['G92']"]),
        (
            _expecting(_UNASKED, "G92", "G93"),
            ["is an expected failure for ['G92', 'G93']", f"attributes ['G93'] to cell {_UNASKED}"],
        ),
        (_expecting(_RAISED, "G92", "G94"), ["is an expected failure for ['G92', 'G94']"]),
        (
            _expecting(_PINNED, "G92"),
            [f"the ledger has no entry for cell {_PINNED}", "pins ['pricing:none'] as not owed"],
        ),
        (_twice(_PASSING), ["is recorded 2 times (test-results-"]),
        (_expecting(_ENTRY, "G95"), ["is an expected failure for ['G95']"]),
        (_changed(_ENTRY, message=""), ["is an expected failure for []"]),
        (_changed(_PASSING, outcome="failed", message="boom"), [f"[{_PASSING}] failed"]),
        (_changed(_PASSING, outcome="error", message="boom"), [f"[{_PASSING}] error"]),
    ],
    ids=[
        "a-missing-cell",
        "a-deselected-module",
        "a-skip-record",
        "an-empty-parameter-set",
        "an-expected-failure-without-an-entry",
        "an-expected-failure-on-a-canary",
        "an-entry-whose-cell-passed",
        "a-static-entry-whose-cell-passed",
        "a-column-wide-gap-whose-cell-passed",
        "an-expected-failure-that-leaves-out-the-column-wide-gap",
        "an-expected-failure-for-the-column-wide-gap-alone-on-a-cell-with-an-entry",
        "a-column-wide-gap-named-where-the-entry-shows-its-check-switched-off",
        "a-column-wide-gap-named-where-the-entry-is-a-raise",
        "a-column-wide-gap-named-on-a-pinned-cell",
        "a-duplicate-item",
        "an-expected-failure-for-another-gap",
        "an-expected-failure-that-names-no-gap",
        "a-failed-cell",
        "an-errored-cell",
    ],
)
def test_each_incomplete_or_dishonest_run_is_rejected(
    mutation: Mutation, named: list[str], tmp_path: Path
) -> None:
    found = _found(tmp_path, mutation(_complete_run()))

    for fragment in named:
        assert any(fragment in line for line in found), (fragment, found)


def test_counts_pins_and_strangers_are_compared_with_the_inventory(tmp_path: Path) -> None:
    cases = _complete_run()
    wide = COUNTS[_WIDE]

    # A count that differs from the pinned one, in either direction.
    for pinned in (wide - 1, wide + 1):
        (line,) = _found(
            tmp_path / str(pinned), cases, pinned_counts={**COUNTS, _WIDE: pinned}
        )
        assert line == f"kind {_WIDE}: {wide} cell item(s) ran, inventory.yaml pins {pinned}"
    # A pinned not-expressible cell must have no item.
    scenario = ledger.parse_cell_id(_PASSING)[1]
    row = {"kind": _PLAIN, "scenario": scenario, "step": "Say", "reason": "r", "checklist": "c"}
    pinned_ran = _found(tmp_path / "pin", cases, not_expressible=(row,))
    assert pinned_ran and all("pins it not expressible" in line for line in pinned_ran)
    # A cell item the inventory does not hold, and an expected failure that is no cell.
    stranger = Seen(_MODULE, f"test_cell[{_PLAIN}/S99/x/raw]", PASSED, "", "test-results-0.xml")
    stray = Seen(_MODULE, "test_a_self_test", EXPECTED_FAILURE, "G91: x", "test-results-0.xml")
    found = _found(tmp_path / "strange", [*cases, stranger, stray])
    assert any("is a cell the inventory does not hold" in line for line in found)
    assert any("is an expected failure and is no cell of the matrix" in line for line in found)
    # A closing check may not be an expected failure, and must have run.
    closing = next(index for index, case in enumerate(cases) if case.name == cells.STATIC_CHECK)
    soft = dataclasses.replace(cases[closing], outcome=EXPECTED_FAILURE, message="G91: x")
    assert any(
        "closing check and ended as an expected failure" in line
        for line in _found(tmp_path / "soft", [*cases[:closing], soft, *cases[closing + 1 :]])
    )
    # Without a report nothing shows that a cell ran.
    empty = tmp_path / "empty"
    empty.mkdir()
    (missing,), count = verify.verify(empty)
    assert "nothing shows that a cell ran" in missing and count == 0


def test_a_column_wide_gap_is_owed_exactly_where_its_check_ran() -> None:
    def owed(cell: str, pins: Sequence[Mapping[str, str]] = NOT_OWED) -> tuple[set[str], set[str]]:
        return verify.ledgered(KNOWN, cell, pins)

    # (the gaps of the cell's own entry, the column-wide gaps owed on it)
    assert owed(_WIDE_ONLY) == (set(), {"G92"})
    assert owed(_ENTRY) == ({"G91"}, {"G92"})
    # The seam canary checks C0 only; a static cell evaluates no clause; a
    # cell of another kind has nothing to do with the entry.
    assert owed(_CANARY) == (set(), set())
    assert owed(ledger.cell_id(_WIDE, "H02", ledger.NO_ROLE, ledger.STATIC_LENS)) == (set(), set())
    assert owed(_PASSING) == (set(), set())
    # An entry of the cell itself shows that the check did not run: the result
    # flags its cost incomplete, or a raise stands for the result clauses.
    assert owed(_UNASKED) == ({"G93"}, set())
    assert owed(_RAISED) == ({"G94"}, set())
    assert contract.not_owed_given(_WIDE_CODE, KNOWN.entry(_UNASKED).codes)  # type: ignore[union-attr]
    assert contract.not_owed_given(_WIDE_CODE, KNOWN.entry(_RAISED).codes)  # type: ignore[union-attr]
    assert not contract.not_owed_given(_WIDE_CODE, KNOWN.entry(_ENTRY).codes)  # type: ignore[union-attr]
    # A raise stands for the result clauses only: a code of another clause is
    # still owed, and so is one after a raise from close().
    leak = dataclasses.replace(
        KNOWN, column_wide=(ledger.ColumnWide(_WIDE, "secret:leak", "G92"),)
    )
    assert verify.ledgered(leak, _RAISED, ()) == ({"G94"}, {"G92"})
    assert not contract.not_owed_given(_WIDE_CODE, ["raise:RuntimeError@close/standin.adapter"])
    # A guard rewrite stands for the kind and stop reason it replaced, not for
    # the accounting; a refused second call rewrites nothing.
    rewritten = ["guard:limit_contract_violated", "guard:revoked"]
    assert contract.not_owed_given("kind:transport!=none", rewritten)
    assert contract.not_owed_given("stop:error!=end_turn", rewritten)
    assert not contract.not_owed_given(_WIDE_CODE, rewritten)
    assert not contract.not_owed_given("kind:transport!=none", ["guard:revoked"])
    # Neither does a result the guard let stand without calling it rankable.
    assert not contract.not_owed_given("kind:transport!=none", [contract.NOT_RANKABLE])
    assert not contract.not_owed_given("stop:error!=end_turn", [contract.NOT_RANKABLE])
    # A clause the cell's definition does not have is owed nowhere: the guard
    # clause in the raw lens, the dispatch clause of a scenario without a ToolSet.
    for code in ("guard:revoked", "dispatch.extra:read_a"):
        other = dataclasses.replace(KNOWN, column_wide=(ledger.ColumnWide(_WIDE, code, "G92"),))
        assert verify.ledgered(other, _WIDE_ONLY, ()) == (set(), set()), code
        assert verify.ledgered(other, _ENTRY, ()) == ({"G91"}, {"G92"}), code
    # The inventory's pin: for the pinned cell and the pinned code, nothing else.
    assert owed(_PINNED) == (set(), set())
    assert owed(_PINNED, ()) == (set(), {"G92"})
    assert owed(_PINNED, [{**NOT_OWED[0], "code": "pricing:unreconciled"}]) == (set(), {"G92"})
    assert owed(_WIDE_ONLY, [{**NOT_OWED[0], "cell": _PINNED}]) == (set(), {"G92"})


# -- the shipped inventory and ledger, where the real program runs -----------------

REAL = cells.expected_items()
REAL_LEDGER = ledger.load()
REAL_INVENTORY = cells.load_inventory()
_GATEWAY_MODULE = cells.MODULE_BY_KIND[HOSTED_GATEWAY_KIND]


def _real_owed(cell: str) -> set[str]:
    """The gaps the shipped ledger expects of ``cell`` today; none: it passes."""

    owed, wide = verify.ledgered(REAL_LEDGER, cell, REAL_INVENTORY.column_wide_not_owed)
    return owed | wide


def _real_complete_run() -> list[Seen]:
    """A complete run of the shipped inventory, as the shipped ledger expects it."""

    modules = sorted({item.module for item in REAL})
    cases: list[Seen] = []
    for item in REAL:
        gaps = [] if item.cell is None else sorted(_real_owed(item.cell))
        cases.append(
            Seen(
                item.module,
                item.name,
                EXPECTED_FAILURE if gaps else PASSED,
                _named(REAL_LEDGER, gaps),
                f"test-results-{modules.index(item.module)}.xml",
            )
        )
    return cases


def _pytest(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *arguments],
        cwd=REPOSITORY,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )


def test_expected_items_are_the_items_pytest_collects() -> None:
    run = _pytest("--collect-only", "-q", "tests/backend_matrix")
    collected = {
        (Path(path).stem, name)
        for path, _, name in (
            line.partition("::") for line in run.stdout.splitlines() if "::" in line
        )
        if path.startswith("tests/backend_matrix/")
    }

    assert run.returncode == 0, run.stdout + run.stderr
    # Every expected item exists under the module and name the verifier looks for...
    assert {(item.module, item.name) for item in REAL} <= collected
    # ...and no cell is collected that the inventory does not hold.
    cell_items = {key for key in collected if key[1].startswith(f"{cells.BEHAVIOUR_TEST}[")}
    assert cell_items == {
        (item.module, item.name)
        for item in REAL
        if item.cell is not None and item.module != cells.STATIC_MODULE
    }
    assert len(REAL) == len({(item.module, item.name) for item in REAL})


def test_the_reports_of_a_real_run_are_read_back(tmp_path: Path) -> None:
    junit = tmp_path / "test-results-0.xml"
    run = _pytest(f"tests/backend_matrix/{cells.STATIC_MODULE}.py", f"--junitxml={junit}", "-q")
    found, _count = verify.verify(tmp_path)
    static = {
        item.name: item.cell
        for item in REAL
        if item.module == cells.STATIC_MODULE and item.cell is not None
    }

    assert run.returncode == 0, run.stdout + run.stderr
    # The module that ran is complete and honest; the others are missing.
    assert found and not any(cells.STATIC_MODULE in line for line in found)
    recorded = {case.name: case for case in verify.read_junit([junit])}
    assert {*static, cells.STATIC_CHECK} <= set(recorded)
    # Each static cell ended as the shipped ledger says, whatever it says today.
    for name, cell in static.items():
        gaps = _real_owed(cell)
        assert recorded[name].outcome == (EXPECTED_FAILURE if gaps else PASSED), cell
        assert verify.named_gaps(recorded[name].message) == gaps, cell
    # The cell report of the process is written beside its JUnit file.
    written = matrix_report.path_beside(junit)
    assert written == tmp_path / "matrix-report-0.json"
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["schema_version"] == matrix_report.REPORT_SCHEMA
    assert set(document["cells"]) == set(static.values())
    for cell, record in document["cells"].items():
        gaps = _real_owed(cell)
        assert set(record) == {"kind", "scenario", "role", "lens", "outcome", "codes", "gaps"}
        assert record["outcome"] == ("expected_failure" if gaps else "passed"), cell
        assert record["gaps"] == sorted(gaps) and bool(record["codes"]) == bool(gaps), cell
    assert matrix_report.path_beside("out/junit.xml") == Path("out/matrix-report-junit.json")


def test_the_cell_report_says_what_a_behaviour_cell_showed(tmp_path: Path) -> None:
    junit = tmp_path / "test-results-3.xml"
    column = f"{HOSTED_GATEWAY_KIND}/"
    starter = ("S00", "S01", "S03", "S04")
    run = _pytest(
        f"tests/backend_matrix/{_GATEWAY_MODULE}.py",
        "-q",
        "-k",
        "test_cell and primary and ("
        + " or ".join(f"{column}{scenario}" for scenario in starter)
        + ")",
        f"--junitxml={junit}",
    )
    selected = {
        item.cell
        for item in REAL
        if item.cell is not None
        and item.cell.startswith(column)
        and ledger.parse_cell_id(item.cell)[1:3] in {(scenario, "primary") for scenario in starter}
    }
    document = json.loads(matrix_report.path_beside(junit).read_text(encoding="utf-8"))
    shown = document["cells"]
    kind = document["kinds"][HOSTED_GATEWAY_KIND]

    assert run.returncode == 0, run.stdout + run.stderr
    assert set(shown) == selected and len(selected) > len(starter)
    # Per cell: outcome and gaps as the shipped ledger says (whatever it says
    # today), the declared values and differences, each clause, the grade.
    statuses = {"ok", "deviation", "not_evaluated", "stood_for"}
    for cell, record in shown.items():
        gaps = _real_owed(cell)
        scenario = SCENARIOS[record["scenario"]]
        assert record["outcome"] == ("expected_failure" if gaps else "passed"), cell
        assert record["gaps"] == sorted(gaps) and bool(record["codes"]) == bool(gaps), cell
        assert all(contract.clause_of(code) for code in record["codes"])
        assert set(record["declared"]) == set(record["variants"]) == set(scenario.governing)
        assert set(REAL_LEDGER.differences_for(cell)) <= set(record["differences"])
        assert set(record["clauses"]) <= {*contract.CLAUSES, contract.SCENARIO}
        assert "C0" in record["clauses"] and set(record["clauses"].values()) <= statuses
        assert record["grade"].split(" (")[0] in envelopes.GRADES
        assert record["envelopes"] and set(record["envelopes"]) <= set(kind["envelopes"])
        assert record["seam_problems"] == []
    # The canary checks the seam and C0 only and is never ledgered.
    canary = shown[ledger.cell_id(HOSTED_GATEWAY_KIND, "S00", "primary", "raw")]
    assert (canary["outcome"], canary["codes"], canary["gaps"]) == ("passed", [], [])
    assert canary["clauses"]["C0"] == "ok"
    assert set(canary["clauses"].values()) == {"ok", "not_evaluated"}
    # What the driver of the kind bypasses is said once per kind.
    assert kind["fidelity"] == "wire" and kind["dialect"] == "openai_chat"
    assert any("DNS and TLS" in layer for layer in kind["bypassed_layers"])
    assert kind["private_seams"] == [] and len(kind["envelopes"]) == 2


def test_the_verifier_runs_as_the_ci_step_runs_it(tmp_path: Path) -> None:
    def step(root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *COMMAND.split()[1:], "--root", str(root)],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=100,
            check=False,
        )

    complete = _real_complete_run()
    accepted = step(_write(tmp_path / "complete", complete))
    rejected = step(_write(tmp_path / "partial", _without(_GATEWAY_MODULE)(complete)))

    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert f"{len(REAL)} expected items ran exactly once" in accepted.stdout
    assert rejected.returncode == 1
    assert "do not match the cell inventory" in rejected.stderr
    assert f"{_GATEWAY_MODULE}::{cells.CANARY_CHECK} did not run" in rejected.stderr
    # In process, the same two sets and the same two answers.
    assert verify.verify(tmp_path / "complete") == ([], len(REAL))
    assert verify.main(["--root", str(tmp_path / "partial")]) == 1


def test_a_required_proof_is_held_by_name_like_a_closing_check() -> None:
    closing = {cells.CANARY_CHECK, cells.STATIC_CHECK}
    proofs = [item for item in REAL if item.cell is None and item.name not in closing]
    complete = _real_complete_run()

    def found(cases: Sequence[Seen]) -> list[str]:
        return verify.problems(
            cases,
            expected=REAL,
            known=REAL_LEDGER,
            pinned_counts=REAL_INVENTORY.cell_counts,
            not_expressible=REAL_INVENTORY.not_expressible,
            not_owed=REAL_INVENTORY.column_wide_not_owed,
        )

    # What no cell's verdict shows is required by name, in a module of cells.
    assert {(item.module, item.name) for item in proofs} == {
        (module, name) for module, names in cells.REQUIRED_PROOFS.items() for name in names
    }
    assert proofs and set(cells.REQUIRED_PROOFS) <= set(cells.MODULE_BY_KIND.values())
    assert found(complete) == []

    def ended(index: int, outcome: str, message: str = "") -> list[str]:
        changed = dataclasses.replace(complete[index], outcome=outcome, message=message)
        return found([*complete[:index], changed, *complete[index + 1 :]])

    for proof in proofs:
        label = f"{proof.module}::{proof.name}"
        at = next(
            index
            for index, case in enumerate(complete)
            if (case.module, case.name) == (proof.module, proof.name)
        )
        # A run in which it was deselected, failed, was skipped or was let
        # stand as an expected failure is not a complete run.
        assert found([*complete[:at], *complete[at + 1 :]]) == [
            f"{label} did not run: no shard recorded it"
        ]
        assert ended(at, "failed") == [f"{label} failed"]
        assert ended(at, "skipped", "no runtime") == [f"{label} was skipped (no runtime)"]
        soft = ended(at, EXPECTED_FAILURE, "G91: x")
        assert len(soft) == 2 and all(line.startswith(label) for line in soft)
        assert any("ended as an expected failure" in line for line in soft)


def test_a_reference_column_and_the_open_weight_columns_are_held_to_the_same_scenarios(
    tmp_path: Path,
) -> None:
    def behaviour(kinds: frozenset[str]) -> dict[str, tuple[str, str]]:
        """Scenario and lens of every behaviour cell of these kinds, by cell."""

        found: dict[str, tuple[str, str]] = {}
        for item in REAL:
            if item.cell is None:
                continue
            kind, scenario, _role, lens = ledger.parse_cell_id(item.cell)
            if kind in kinds and lens != ledger.STATIC_LENS:
                found[item.cell] = (scenario, lens)
        return found

    def found(name: str, cases: Sequence[Seen]) -> list[str]:
        paths = sorted(_write(tmp_path / name, cases).glob(verify.JUNIT_GLOB))
        return verify.problems(
            verify.read_junit(paths),
            expected=REAL,
            known=REAL_LEDGER,
            pinned_counts=REAL_INVENTORY.cell_counts,
            not_expressible=REAL_INVENTORY.not_expressible,
            not_owed=REAL_INVENTORY.column_wide_not_owed,
        )

    def item_of(cell: str) -> str:
        return next(item.name for item in REAL if item.cell == cell)

    reference = behaviour(matrix_columns.REFERENCE_KINDS)
    open_weight = behaviour(matrix_columns.OPEN_WEIGHT_KINDS)
    complete = _real_complete_run()

    # A reference column is driven, and whatever it plays, in whichever lens,
    # an open-weight column plays too: the same scenario judged on both.
    assert reference and set(reference.values()) <= set(open_weight.values())
    # A reference cell is expected to pass unless the owner ruled on its gap:
    # an entry on one records the ruling, cell by cell or column-wide.
    for cell in reference:
        entry = REAL_LEDGER.entry(cell)
        assert entry is None or entry.ruling, cell
    assert all(
        scope.ruling
        for scope in REAL_LEDGER.column_wide
        if scope.kind in matrix_columns.REFERENCE_KINDS
    )
    # The verifier holds a run to both sets. Complete, it is accepted ...
    assert found("complete", complete) == []
    # ... a cell of either side that did not run is missed ...
    same = [cell for cell, played in open_weight.items() if played in set(reference.values())]
    for side, cell in (("reference", min(reference)), ("open-weight", min(same))):
        missing = [case for case in complete if case.name != item_of(cell)]
        assert any(
            f"{item_of(cell)} did not run" in line for line in found(f"no-{side}", missing)
        ), cell
    # ... and a reference cell cannot be red on the quiet: an expected failure
    # that no entry stands for is rejected, whatever gap it names.
    quiet = min(cell for cell in reference if not _real_owed(cell))
    red = [
        dataclasses.replace(case, outcome=EXPECTED_FAILURE, message="G00: a gap")
        if case.name == item_of(quiet)
        else case
        for case in complete
    ]
    assert any(
        f"the ledger has no entry for cell {quiet}" in line for line in found("red", red)
    )


def test_ci_runs_the_verifier_after_the_shard_evidence_is_complete() -> None:
    jobs = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    steps = jobs["coverage"]["steps"]
    runs = [str(step.get("run", "")) for step in steps]
    evidence = next(index for index, run in enumerate(runs) if "verify_shard_evidence" in run)
    verifier = [index for index, run in enumerate(runs) if COMMAND in run]

    # Once, in the job that holds every shard's JUnit file, after the step that
    # requires all of them, from the repository root, with the locked tools.
    assert len(verifier) == 1 and verifier[0] == evidence + 1
    step = steps[verifier[0]]
    assert step["run"].strip() == f"uv run --frozen {COMMAND}"
    assert "working-directory" not in step and "if" not in step
    assert "continue-on-error" not in step
    assert jobs["coverage"]["defaults"]["run"]["working-directory"] == "crucible"
    assert "coverage" in jobs["gate"]["needs"]
    # The shards write the files the verifier reads, where it reads them.
    shard = (REPOSITORY / "scripts" / "test_shard.py").read_text(encoding="utf-8")
    assert "--junitxml=test-results-{args.index}.xml" in shard
    assert verify.JUNIT_GLOB == "test-results-*.xml"
    assert {kind for kind, count in REAL_INVENTORY.cell_counts.items() if count} == set(
        cells.MODULE_BY_KIND
    )
    assert set(cells.MODULE_BY_KIND) - {TOY_KIND} <= set(matrix_columns.REAL_KINDS)
