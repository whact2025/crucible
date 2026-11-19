# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The cell-inventory verifier, shown on synthetic and on real test reports.

A complete run is accepted. Each way a cell can fail to run, or to mean what
the ledger says, is rejected: a missing cell, a deselected module, a skip, the
item of an empty parameter set, an expected failure without an entry, an entry
whose cell passed, a duplicate item, a count that differs from the pinned one.
The item names the verifier expects are compared with what pytest collects,
and the report of a real run of one matrix module is read back.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Callable, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

from . import cells, ledger, verify
from . import columns as matrix_columns
from . import report as matrix_report
from .columns import HOSTED_GATEWAY_KIND, TOY_KIND
from .verify import EXPECTED_FAILURE, PASSED, Seen

LEDGER = ledger.load()
EXPECTED = cells.expected_items()
INVENTORY = cells.load_inventory()
REPOSITORY = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"
COMMAND = "python -m tests.backend_matrix.verify"
_GATEWAY_MODULE = cells.MODULE_BY_KIND[HOSTED_GATEWAY_KIND]
_TOY_MODULE = cells.MODULE_BY_KIND[TOY_KIND]


def _complete_run() -> list[Seen]:
    """The records of a complete run in which every cell does what the ledger says."""

    modules = sorted({item.module for item in EXPECTED})
    cases: list[Seen] = []
    for item in EXPECTED:
        outcome, message = PASSED, ""
        if item.cell is not None:
            owed, wide = verify.ledgered(LEDGER, item.cell)
            gaps = sorted(owed or wide)
            if gaps:
                outcome = EXPECTED_FAILURE
                message = "; ".join(f"{gap}: {LEDGER.registry[gap].title}" for gap in gaps)
        # One file per module, as a shard holds whole files.
        source = f"test-results-{modules.index(item.module)}.xml"
        cases.append(Seen(item.module, item.name, outcome, message, source))
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
    arguments: dict[str, object] = {
        "expected": EXPECTED,
        "known": LEDGER,
        "pinned_counts": INVENTORY.cell_counts,
        "not_expressible": INVENTORY.not_expressible,
    }
    arguments.update(changed)
    paths = sorted(_write(root, cases).glob(verify.JUNIT_GLOB))
    return verify.problems(verify.read_junit(paths), **arguments)  # type: ignore[arg-type]


def _index(cases: Sequence[Seen], cell: str) -> int:
    name = next(item.name for item in EXPECTED if item.cell == cell)
    return next(index for index, case in enumerate(cases) if case.name == name)


def _first(module: str, outcome: str) -> str:
    """The first cell of ``module`` whose honest record has ``outcome``."""

    for item, case in zip(EXPECTED, _complete_run(), strict=True):
        if item.module == module and item.cell is not None and case.outcome == outcome:
            return item.cell
    raise AssertionError(f"{module} has no {outcome} cell")


def test_a_complete_run_is_accepted(tmp_path: Path) -> None:
    cases = _complete_run()
    root = _write(tmp_path, cases)

    assert len(list(root.glob(verify.JUNIT_GLOB))) == len({item.module for item in EXPECTED}) >= 3
    assert verify.verify(root) == ([], len(EXPECTED))
    assert verify.main(["--root", str(root)]) == 0
    # What a complete run holds: every cell once, canaries passed, gaps expected.
    outcomes = {(case.module, case.name): case.outcome for case in cases}
    assert len(outcomes) == len(cases) == len(EXPECTED)
    assert {case.outcome for case in cases} == {PASSED, EXPECTED_FAILURE}
    assert all(
        outcomes[(item.module, item.name)] == PASSED
        for item in EXPECTED
        if item.cell is None or "/S00/" in item.cell
    )


Mutation = Callable[[list[Seen]], list[Seen]]


def _without(module: str) -> Mutation:
    return lambda cases: [case for case in cases if case.module != module]


def _changed(cell: str, **fields: str) -> Mutation:
    def change(cases: list[Seen]) -> list[Seen]:
        index = _index(cases, cell)
        return [*cases[:index], dataclasses.replace(cases[index], **fields), *cases[index + 1 :]]

    return change


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
    kept = [case for case in cases if case.module != _GATEWAY_MODULE or "[" not in case.name]
    return [
        *kept,
        Seen(
            _GATEWAY_MODULE,
            "test_cell[NOTSET]",
            "skipped",
            "got empty parameter set",
            "test-results-7.xml",
        ),
    ]


_PASSING_TOY = _first(_TOY_MODULE, PASSED)
# A behaviour cell with an entry of its own, on a kind with a column-wide entry.
_LEDGERED = next(
    item.cell
    for item in EXPECTED
    if item.module == _GATEWAY_MODULE and item.cell and LEDGER.entry(item.cell)
)
_STATIC_GAP = _first(cells.STATIC_MODULE, EXPECTED_FAILURE)
_CANARY = _first(_GATEWAY_MODULE, PASSED)
_TOY_COUNT = INVENTORY.cell_counts[TOY_KIND]
_HOSTED_COUNT = INVENTORY.cell_counts[HOSTED_GATEWAY_KIND]


@pytest.mark.parametrize(
    ("mutation", "named"),
    [
        (
            _dropped(_PASSING_TOY),
            [f"[{_PASSING_TOY}] did not run", f"kind toy: {_TOY_COUNT - 1} cell item(s)"],
        ),
        (
            _without(_GATEWAY_MODULE),
            [
                f"{_GATEWAY_MODULE}::{cells.CANARY_CHECK} did not run",
                f"[{_CANARY}] did not run",
                f"kind {HOSTED_GATEWAY_KIND}: 0 cell item(s) ran, inventory.yaml "
                f"pins {_HOSTED_COUNT}",
            ],
        ),
        (
            _changed(_PASSING_TOY, outcome="skipped", message="no runtime"),
            ["was skipped (no runtime)"],
        ),
        (_empty_parameter_set, ["is the item of an empty parameter set", "did not run"]),
        (
            _changed(_PASSING_TOY, outcome=EXPECTED_FAILURE, message="G02: a gap"),
            [f"the ledger has no entry for cell {_PASSING_TOY}"],
        ),
        (
            _changed(_CANARY, outcome=EXPECTED_FAILURE, message="G32: no pricing evidence"),
            [f"the ledger has no entry for cell {_CANARY}"],
        ),
        (
            _changed(_LEDGERED, outcome=PASSED, message=""),
            ["passed although the ledger has an entry"],
        ),
        (
            _changed(_STATIC_GAP, outcome=PASSED, message=""),
            ["passed although the ledger has an entry"],
        ),
        (_twice(_PASSING_TOY), ["is recorded 2 times (test-results-"]),
        (_changed(_LEDGERED, message="G02: another gap"), ["is an expected failure for ['G02']"]),
        (_changed(_LEDGERED, message=""), ["is an expected failure for []"]),
        (_changed(_PASSING_TOY, outcome="failed", message="boom"), [f"[{_PASSING_TOY}] failed"]),
        (_changed(_PASSING_TOY, outcome="error", message="boom"), [f"[{_PASSING_TOY}] error"]),
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
    assert verify.main(["--root", str(tmp_path)]) == 1


def test_counts_pins_and_strangers_are_compared_with_the_inventory(tmp_path: Path) -> None:
    cases = _complete_run()
    toy = _TOY_COUNT

    # A count that differs from the pinned one, in either direction.
    for pinned in (toy - 1, toy + 1):
        (line,) = _found(
            tmp_path / str(pinned), cases, pinned_counts={**INVENTORY.cell_counts, TOY_KIND: pinned}
        )
        assert line == f"kind toy: {toy} cell item(s) ran, inventory.yaml pins {pinned}"
    # A pinned not-expressible cell must have no item.
    scenario = ledger.parse_cell_id(_PASSING_TOY)[1]
    row = {"kind": TOY_KIND, "scenario": scenario, "step": "Say", "reason": "r", "checklist": "c"}
    pinned_ran = _found(tmp_path / "pin", cases, not_expressible=(row,))
    assert pinned_ran and all("pins it not expressible" in line for line in pinned_ran)
    # A cell item the inventory does not hold, and an expected failure that is no cell.
    stranger = Seen(_TOY_MODULE, "test_cell[toy/S99/x/raw]", PASSED, "", "test-results-0.xml")
    stray = Seen(_TOY_MODULE, "test_a_self_test", EXPECTED_FAILURE, "G02: x", "test-results-0.xml")
    found = _found(tmp_path / "strange", [*cases, stranger, stray])
    assert any("is a cell the inventory does not hold" in line for line in found)
    assert any("is an expected failure and is no cell of the matrix" in line for line in found)
    # A closing check may not be an expected failure, and must have run.
    closing = next(index for index, case in enumerate(cases) if case.name == cells.STATIC_CHECK)
    soft = dataclasses.replace(cases[closing], outcome=EXPECTED_FAILURE, message="G02: x")
    assert any(
        "closing check and ended as an expected failure" in line
        for line in _found(tmp_path / "soft", [*cases[:closing], soft, *cases[closing + 1 :]])
    )
    # Without a report nothing shows that a cell ran.
    empty = tmp_path / "empty"
    empty.mkdir()
    (missing,), count = verify.verify(empty)
    assert "nothing shows that a cell ran" in missing and count == 0


def test_a_column_wide_gap_may_be_named_beside_the_entry_but_never_alone_on_it(
    tmp_path: Path,
) -> None:
    cases = _complete_run()
    owed, wide = verify.ledgered(LEDGER, _LEDGERED)
    titles = {gap: LEDGER.registry[gap].title for gap in owed | wide}

    def message(gaps: set[str]) -> str:
        return "; ".join(f"{gap}: {titles[gap]}" for gap in sorted(gaps))

    assert owed and wide and not owed & wide
    # Both the entry's gaps and the column-wide gap: what the comparator reports.
    assert _found(tmp_path / "both", _changed(_LEDGERED, message=message(owed | wide))(cases)) == []
    # The column-wide gap alone does not account for a cell that has an entry.
    assert any(
        "the ledger attributes" in line
        for line in _found(tmp_path / "wide", _changed(_LEDGERED, message=message(wide))(cases))
    )
    # A canary and a static cell are never expected failures of a column-wide entry.
    assert verify.ledgered(LEDGER, _CANARY) == (set(), set())
    assert verify.ledgered(LEDGER, f"{HOSTED_GATEWAY_KIND}/H02/-/static")[1] == set()


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
    assert {(item.module, item.name) for item in EXPECTED} <= collected
    # ...and no cell is collected that the inventory does not hold.
    cell_items = {key for key in collected if key[1].startswith(f"{cells.BEHAVIOUR_TEST}[")}
    assert cell_items == {
        (item.module, item.name)
        for item in EXPECTED
        if item.cell is not None and item.module != cells.STATIC_MODULE
    }
    assert len(EXPECTED) == len({(item.module, item.name) for item in EXPECTED})


def test_the_reports_of_a_real_run_are_read_back(tmp_path: Path) -> None:
    junit = tmp_path / "test-results-0.xml"
    run = _pytest(f"tests/backend_matrix/{cells.STATIC_MODULE}.py", f"--junitxml={junit}", "-q")
    found, _count = verify.verify(tmp_path)
    static = {item.name: item.cell for item in EXPECTED if item.module == cells.STATIC_MODULE}

    assert run.returncode == 0, run.stdout + run.stderr
    # The module that ran is complete and honest; the others are missing.
    assert found and not any(cells.STATIC_MODULE in line for line in found)
    recorded = {case.name: case for case in verify.read_junit([junit])}
    assert set(static) <= set(recorded)
    gap = next(item.name for item in EXPECTED if item.cell == _STATIC_GAP)
    assert recorded[gap].outcome == EXPECTED_FAILURE
    assert verify.named_gaps(recorded[gap].message) == verify.ledgered(LEDGER, _STATIC_GAP)[0]
    # The cell report of the process is written beside its JUnit file.
    written = matrix_report.path_beside(junit)
    assert written == tmp_path / "matrix-report-0.json"
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["schema_version"] == matrix_report.REPORT_SCHEMA
    assert set(document["cells"]) == {cell for cell in static.values() if cell is not None}
    known = document["cells"][_STATIC_GAP]
    assert (known["outcome"], known["gaps"]) == ("expected_failure", sorted(
        verify.ledgered(LEDGER, _STATIC_GAP)[0]
    ))
    assert {record["outcome"] for record in document["cells"].values()} == {
        "passed",
        "expected_failure",
    }
    assert matrix_report.path_beside("out/junit.xml") == Path("out/matrix-report-junit.json")


def test_the_cell_report_says_what_a_behaviour_cell_showed(tmp_path: Path) -> None:
    junit = tmp_path / "test-results-3.xml"
    wanted = f"{HOSTED_GATEWAY_KIND}/S0"
    run = _pytest(
        f"tests/backend_matrix/{_GATEWAY_MODULE}.py",
        "-q",
        "-k",
        f"test_cell and primary and ({wanted}1 or {wanted}3 or {wanted}4 or {wanted}0)",
        f"--junitxml={junit}",
    )
    document = json.loads(matrix_report.path_beside(junit).read_text(encoding="utf-8"))
    shown = document["cells"]
    kind = document["kinds"][HOSTED_GATEWAY_KIND]

    assert run.returncode == 0, run.stdout + run.stderr
    assert len(shown) == 7
    # Grade, gaps, declared values and differences, per cell.
    canary = shown[f"{HOSTED_GATEWAY_KIND}/S00/primary/raw"]
    text = shown[f"{HOSTED_GATEWAY_KIND}/S01/primary/raw"]
    schema = shown[f"{HOSTED_GATEWAY_KIND}/S04/primary/guarded"]
    assert (canary["outcome"], canary["codes"], canary["grade"]) == ("passed", [], "documented")
    assert (text["outcome"], text["codes"], text["gaps"]) == (
        "expected_failure",
        ["pricing:none"],
        ["G32"],
    )
    assert schema["gaps"] == ["G28", "G29"] and schema["grade"] == "constructed"
    assert set(schema["declared"]) == {"structured_output"}
    assert "D-OAI-SCHEMA-FINAL" in schema["differences"]
    assert schema["clauses"]["C2"] == "stood_for" and schema["clauses"]["C6"] == "deviation"
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

    accepted = step(_write(tmp_path / "complete", _complete_run()))
    rejected = step(_write(tmp_path / "partial", _without(_GATEWAY_MODULE)(_complete_run())))

    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert f"{len(EXPECTED)} expected items ran exactly once" in accepted.stdout
    assert rejected.returncode == 1
    assert "do not match the cell inventory" in rejected.stderr
    assert f"{_GATEWAY_MODULE}::{cells.CANARY_CHECK} did not run" in rejected.stderr


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
    assert {kind for kind, count in INVENTORY.cell_counts.items() if count} == set(
        cells.MODULE_BY_KIND
    )
    assert set(cells.MODULE_BY_KIND) - {TOY_KIND} <= set(matrix_columns.REAL_KINDS)
