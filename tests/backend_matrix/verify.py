# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The cell-inventory verifier: every cell of the matrix ran, exactly once.

A test that does not run cannot fail. Inside one pytest process the matrix
already refuses skips and requires each column's canary; across the shards of
a run only the JUnit files can show that every cell was there. So after the
shards::

    python -m tests.backend_matrix.verify

computes the inventory (the catalogs, the enablement table, ``inventory.yaml``
and the ledger; no driver ``preflight()``, no worker environment) and compares
it with ``test-results-*.xml`` in the repository root:

- every expected item ran exactly once and did not fail: each behaviour and
  static cell, and the closing check of every matrix module;
- no item of the package has a ``pytest.skip`` record, and none is the
  ``[NOTSET]`` item pytest makes of an empty parameter set;
- the items recorded as ``pytest.xfail`` are exactly the ledgered cells, each
  naming exactly the gaps the ledger attributes to it: those of the cell's
  own entry and every column-wide gap owed on the cell. A column-wide gap is
  owed on every behaviour cell of its kind whose definition has the clause of
  its code, except the seam canary, a cell whose own entry shows that the
  check did not run (a raise, a guard rewrite, the cost flagged incomplete)
  and a cell ``inventory.yaml`` pins as not owing it. A ledgered cell that
  passed is rejected, and so is an expected failure on any other cell;
- no pinned not-expressible cell has an item, and no cell item exists that
  the inventory does not hold;
- the cell count of each kind equals the count pinned in ``inventory.yaml``.

It lives in the test package, not in ``scripts/``: it imports the matrix, and
``scripts/`` ships in the wheel while ``tests`` may not.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import cells, contract, ledger
from .cells import Item
from .drivers.base import MatrixSetupError
from .scenarios import SCENARIOS

JUNIT_GLOB = "test-results-*.xml"
XFAIL_TYPE = "pytest.xfail"
SKIP_TYPE = "pytest.skip"
PASSED = "passed"
EXPECTED_FAILURE = "expected_failure"
_PREFIX = cells.PACKAGE_MODULE + "."
_GAP = re.compile(r"(?:^|; )([A-Z]+\d+): ")
_CELL_ITEM = re.compile(re.escape(cells.BEHAVIOUR_TEST) + r"\[(.+)\]\Z")


@dataclass(frozen=True)
class Seen:
    """One test case of the package as a JUnit file records it."""

    module: str
    name: str
    outcome: str  # passed, expected_failure, skipped, failed or error
    message: str
    source: str

    @property
    def label(self) -> str:
        return f"{self.module}::{self.name}"


def read_junit(paths: Sequence[Path]) -> list[Seen]:
    """Every test case of the matrix package in ``paths``."""

    seen: list[Seen] = []
    for path in paths:
        for case in ElementTree.parse(path).getroot().iter("testcase"):
            classname = case.get("classname", "")
            if not classname.startswith(_PREFIX):
                continue
            outcome, message = PASSED, ""
            for child in case:
                if child.tag in ("failure", "error"):
                    outcome = "failed" if child.tag == "failure" else "error"
                    message = child.get("message", "")
                    break
                if child.tag == "skipped":
                    xfailed = child.get("type") == XFAIL_TYPE
                    outcome = EXPECTED_FAILURE if xfailed else "skipped"
                    message = child.get("message", "")
            seen.append(
                Seen(
                    module=classname[len(_PREFIX) :].split(".", 1)[0],
                    name=case.get("name", ""),
                    outcome=outcome,
                    message=message,
                    source=path.name,
                )
            )
    return seen


def named_gaps(message: str) -> set[str]:
    """The gap ids an expected failure names in its reason."""

    return set(_GAP.findall(message))


def _pinned_not_owed(not_owed: Sequence[Mapping[str, str]], cell: str) -> set[str]:
    """The codes ``inventory.yaml`` pins as not owed on ``cell``."""

    return {row["code"] for row in not_owed if row["cell"] == cell}


def _has_clause(code: str, scenario: str, lens: str) -> bool:
    """Whether a cell of this scenario and lens has the clause ``code`` is booked under.

    From the cell's definition, as the contract decides it. Whether the
    selected variant expects a result at all depends on a declared value; a
    cell whose variant expects none is pinned in the inventory.
    """

    clause = contract.clause_of(code)
    defined = SCENARIOS.get(scenario)
    if clause not in contract.CLAUSES or defined is None:
        return True
    return clause in contract.applicable(
        lens=lens, toolset=defined.toolset is not None, expects_result=True
    )


def ledgered(
    known: ledger.Ledger, cell: str, not_owed: Sequence[Mapping[str, str]] = ()
) -> tuple[set[str], set[str]]:
    """``(gaps the cell's own entry names, column-wide gaps owed on the cell)``.

    Exact, and computed without playing the cell: the comparator owes a
    column-wide code where its clause was evaluated and its check was asked,
    and the cell's definition, its own entry and the inventory's pins say
    where that is.
    """

    kind, scenario, _role, lens = ledger.parse_cell_id(cell)
    entry = known.entry(cell)
    own = {} if entry is None else entry.codes
    owed = set(own.values())
    canary = scenario in SCENARIOS and SCENARIOS[scenario].canary
    if lens == ledger.STATIC_LENS or canary:
        # A static cell evaluates no clause; the canary checks the seam and C0.
        return owed, set()
    pinned = _pinned_not_owed(not_owed, cell)
    wide = {
        scope.gap
        for scope in known.column_wide
        if scope.kind == kind
        and scope.code not in pinned
        and _has_clause(scope.code, scenario, lens)
        and not contract.not_owed_given(scope.code, own)
    }
    return owed, wide


def problems(
    seen: Sequence[Seen],
    *,
    expected: Sequence[Item],
    known: ledger.Ledger,
    pinned_counts: Mapping[str, int],
    not_expressible: Sequence[Mapping[str, str]] = (),
    not_owed: Sequence[Mapping[str, str]] = (),
) -> list[str]:
    """Why the JUnit records are not a complete, honest run of the inventory."""

    out: list[str] = []
    by_item: dict[tuple[str, str], list[Seen]] = {}
    for case in seen:
        by_item.setdefault((case.module, case.name), []).append(case)
    for (module, name), cases in sorted(by_item.items()):
        if len(cases) > 1:
            out.append(
                f"{module}::{name} is recorded {len(cases)} times "
                f"({', '.join(sorted(case.source for case in cases))})"
            )
    for case in seen:
        if "[NOTSET" in case.name:
            out.append(
                f"{case.label} is the item of an empty parameter set: its cells "
                "did not run"
            )
        elif case.outcome == "skipped":
            out.append(f"{case.label} was skipped ({case.message or 'no reason'})")

    cells_expected = {
        (item.module, item.name): item.cell for item in expected if item.cell is not None
    }
    for item in expected:
        cases = by_item.get((item.module, item.name), [])
        label = f"{item.module}::{item.name}"
        if not cases:
            out.append(f"{label} did not run: no shard recorded it")
            continue
        case = cases[0]
        if case.outcome in ("failed", "error"):
            out.append(f"{label} {case.outcome}")
        if item.cell is None:
            if case.outcome == EXPECTED_FAILURE:
                out.append(f"{label} is a closing check and ended as an expected failure")
            continue
        owed, wide = ledgered(known, item.cell, not_owed)
        if case.outcome == EXPECTED_FAILURE:
            named = named_gaps(case.message)
            if not owed and not wide:
                pins = sorted(_pinned_not_owed(not_owed, item.cell))
                out.append(
                    f"{label} is an expected failure and the ledger has no entry "
                    f"for cell {item.cell}"
                    + (f"; inventory.yaml pins {pins} as not owed on it" if pins else "")
                )
            elif named != owed | wide:
                out.append(
                    f"{label} is an expected failure for {sorted(named)}; the "
                    f"ledger attributes {sorted(owed)} to cell {item.cell}"
                    + (f" and owes {sorted(wide)} column-wide on it" if wide else "")
                )
        elif case.outcome == PASSED and owed:
            out.append(
                f"{label} passed although the ledger has an entry for cell "
                f"{item.cell} ({sorted(owed)}): delete the entry"
            )
        elif case.outcome == PASSED and wide:
            out.append(
                f"{label} passed although the ledger owes {sorted(wide)} "
                f"column-wide on cell {item.cell}: delete the column-wide entry, "
                "or pin the cell in inventory.yaml with the reason of its "
                "definition for which the gap cannot show there"
            )

    counts: Counter[str] = Counter()
    pinned = {(row["kind"], row["scenario"]) for row in not_expressible}
    for key, cases in sorted(by_item.items()):
        case = cases[0]  # a duplicate is reported above and counted once
        match = _CELL_ITEM.match(case.name)
        if match is not None and "[NOTSET" not in case.name:
            parts = match.group(1).split("/")
            counts[parts[0]] += 1
            if key not in cells_expected:
                out.append(f"{case.label} is a cell the inventory does not hold")
            if len(parts) > 1 and (parts[0], parts[1]) in pinned:
                out.append(
                    f"{case.label} ran although inventory.yaml pins it not expressible"
                )
        elif case.outcome == EXPECTED_FAILURE and key not in cells_expected:
            out.append(
                f"{case.label} is an expected failure and is no cell of the matrix"
            )
    for kind in sorted(set(counts) | set(pinned_counts)):
        if counts.get(kind, 0) != pinned_counts.get(kind):
            out.append(
                f"kind {kind}: {counts.get(kind, 0)} cell item(s) ran, inventory.yaml "
                f"pins {pinned_counts.get(kind)}"
            )
    return out


def verify(root: Path) -> tuple[list[str], int]:
    """Compare the JUnit files under ``root`` with the inventory: problems, items."""

    paths = sorted(root.glob(JUNIT_GLOB))
    if not paths:
        return [f"no {JUNIT_GLOB} file in {root}: nothing shows that a cell ran"], 0
    expected = cells.expected_items()
    inventory = cells.load_inventory()
    found = problems(
        read_junit(paths),
        expected=expected,
        known=ledger.load(),
        pinned_counts=inventory.cell_counts,
        not_expressible=inventory.not_expressible,
        not_owed=inventory.column_wide_not_owed,
    )
    return found, len(expected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=f"directory that holds the {JUNIT_GLOB} of the shards (default: the "
        "repository root)",
    )
    arguments = parser.parse_args(argv)
    try:
        found, count = verify(arguments.root.resolve())
    except MatrixSetupError as exc:
        print(f"backend matrix inventory cannot be computed: {exc}", file=sys.stderr)
        return 2
    if found:
        print(
            f"backend matrix: the test reports do not match the cell inventory "
            f"({len(found)} problem(s)):",
            file=sys.stderr,
        )
        for line in found:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(
        f"backend matrix: {count} expected items ran exactly once, as the ledger says"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
