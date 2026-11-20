# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Static and coverage cells of the matrix, and the package's hygiene checks.

A static cell needs no driver and computes deviation codes like any cell. H01
and H02 report product gaps and go through the comparator, so the ones that
are red today (H01 for the Kimi worker, H02 for every kind that has no driver
yet) are strict expected failures tied to their gap ids. The other static
cells check the matrix itself and must be clean whatever the ledger says.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Iterator, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

import assay
from assay import BackendFeatureSet
from crucible.backends import BACKEND_NAMES

from . import cells, contract, declarations, drivers, envelopes, ledger, runner
from . import columns as matrix_columns
from . import report as matrix_report
from .columns import REAL_KINDS, TOY_KIND, WORKER_KINDS, Column
from .contract import Deviation
from .drivers import toy
from .peers import launch, worker_harness
from .scenarios import PACK_A, SCENARIOS
from .script import ProviderScript

LEDGER = ledger.load()
PACKAGE = Path(__file__).resolve().parent
TESTS = PACKAGE.parent
REPOSITORY = TESTS.parent
CI_WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPOSITORY / "pyproject.toml"
BASELINE = TESTS / "TEST_BASELINE.md"
TOY_COLUMNS = cells.toy_columns()
# Static cells that were judged in this process.
_JUDGED: set[str] = set()


def _static_id(kind: str, scenario: str, role: str = ledger.NO_ROLE) -> str:
    return ledger.cell_id(kind, scenario, role, ledger.STATIC_LENS)


def _verdict(cell: str, problems: Sequence[Deviation]) -> None:
    """Judge a static cell: H01 and H02 by the ledger, the others must be clean."""

    assert cell in cells.static_cell_ids(), f"{cell} is not in the inventory"
    _JUDGED.add(cell)
    matrix_report.note_static(cell, problems, LEDGER)
    ledger.judge_static(LEDGER, cell, problems)


@contextlib.contextmanager
def _toy_backend(column: Column, tmp_path: Path) -> Iterator[Any]:
    with drivers.get(TOY_KIND).open(
        column, capability="text", lens="raw", script=ProviderScript(()), tmp_path=tmp_path
    ) as bound:
        yield bound.backend


# -- P01, P02, P04 -------------------------------------------------------------


@pytest.mark.parametrize(
    "column", TOY_COLUMNS, ids=lambda column: _static_id(TOY_KIND, "P01", column.role)
)
def test_p01_protocol_surface(column: Column, tmp_path: Path) -> None:
    problems: list[Deviation] = []
    with _toy_backend(column, tmp_path) as backend:
        checks = {
            "is an assay.Backend": isinstance(backend, assay.Backend),
            "is feature aware": isinstance(backend, assay.FeatureAwareBackend),
            "name": isinstance(backend.name, str) and bool(backend.name),
            "model": isinstance(backend.model, str) and bool(backend.model),
            "capability": backend.capability == "text",
            "features": type(backend.features) is BackendFeatureSet,
            "attach_toolset and close": callable(backend.attach_toolset)
            and callable(backend.close),
            # What the limit guard reads before it admits a call.
            "limit guard attributes": callable(backend.invoke)
            and getattr(backend, "diagnostic_cost_upper_bound", False) is False,
        }
        problems.extend(
            Deviation("static:protocol_surface", name)
            for name, held in checks.items()
            if not held
        )

    _verdict(_static_id(TOY_KIND, "P01", column.role), problems)


@pytest.mark.parametrize(
    "column", TOY_COLUMNS, ids=lambda column: _static_id(TOY_KIND, "P02", column.role)
)
def test_p02_profile_binding(column: Column, tmp_path: Path) -> None:
    problems = runner.profile_binding(column, tmp_path)

    _verdict(_static_id(TOY_KIND, "P02", column.role), problems)


def test_p04_declared_profile_snapshot() -> None:
    snapshot = declarations.load_snapshot()
    problems = declarations.snapshot_problems(
        matrix_columns.real_columns(),
        snapshot,
        lanes_by_product={
            product: matrix_columns.catalog_lanes(product)
            for product in matrix_columns.PRODUCTS
        },
    )

    _verdict(_static_id(ledger.ALL_KINDS, "P04"), problems)


# -- H01 to H04 ----------------------------------------------------------------


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not {".venv", "__pycache__"} & set(path.relative_to(root).parts)
    )


@dataclass(frozen=True)
class WorkerSkip:
    """One ``skipif`` marker whose condition is a worker interpreter path."""

    file: str
    line: int
    condition: str


def worker_skip_markers(tests_root: Path) -> list[WorkerSkip]:
    """Every ``skipif`` on a worker interpreter under ``tests_root``."""

    found: list[WorkerSkip] = []
    for path in _python_files(tests_root):
        if PACKAGE in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "skipif"
                and node.args
                and "WORKER_PYTHON" in ast.unparse(node.args[0]).upper()
            ):
                found.append(
                    WorkerSkip(
                        path.relative_to(tests_root).as_posix(),
                        node.lineno,
                        ast.unparse(node.args[0]),
                    )
                )
    return found


@functools.cache
def suite_worker_skip_markers() -> tuple[WorkerSkip, ...]:
    """The markers of the suite's own tests directory, scanned once per process.

    Every H01 cell reads the same files; parsing each test module of the
    suite once per cell is most of what the static module costs.
    """

    return tuple(worker_skip_markers(TESTS))


def worker_skip_problems(
    markers: Sequence[WorkerSkip],
    worker: str,
    *,
    workers: AbstractSet[str],
    pinned: AbstractSet[str],
) -> list[Deviation]:
    """H01 for one worker: the known skip in its pinned files, and any other.

    A marker belongs to the worker its file name or its condition names, and
    to every worker when it names none. A marker in a file ``inventory.yaml``
    does not list for the worker is a deviation of its own, so the known gap
    cannot stand in for a new skip.
    """

    def named(marker: WorkerSkip) -> set[str]:
        text = f"{Path(marker.file).name} {marker.condition}".lower()
        return {name for name in workers if name in text}

    mine = [item for item in markers if worker in named(item) or not named(item)]
    known = sorted({item.file for item in mine if item.file in pinned})
    other = sorted({item.file for item in mine if item.file not in pinned})
    problems: list[Deviation] = []
    if known:
        count = sum(item.file in pinned for item in mine)
        problems.append(
            Deviation(
                "static:worker_skip",
                f"{count} skipif marker(s) on the {worker} worker interpreter in {known}",
            )
        )
    if other:
        problems.append(
            Deviation(
                "static:worker_skip_unpinned",
                f"skipif marker(s) on the {worker} worker interpreter in {other}, "
                "which inventory.yaml does not list for this worker: a new skip",
            )
        )
    return problems


@pytest.mark.parametrize(
    "worker", sorted(WORKER_KINDS), ids=lambda worker: _static_id(WORKER_KINDS[worker], "H01")
)
def test_h01_no_test_skips_on_a_missing_worker_runtime(worker: str) -> None:
    kind = WORKER_KINDS[worker]
    problems = worker_skip_problems(
        suite_worker_skip_markers(),
        worker,
        workers=set(WORKER_KINDS),
        pinned=cells.load_inventory().worker_skip_files.get(kind, frozenset()),
    )

    _verdict(_static_id(kind, "H01"), problems)


def test_h01_a_skip_outside_the_pinned_files_is_a_different_deviation(
    tmp_path: Path,
) -> None:
    marker = (
        "import pytest\n\n\n"
        "@pytest.mark.skipif(not {}.is_file(), reason='absent')\n"
        "def test_x():\n    pass\n"
    )
    pinned = {WORKER_KINDS["kimi"]: {"test_backends_kimi.py"}}
    (tmp_path / "test_backends_kimi.py").write_text(
        marker.format("parent._DEFAULT_WORKER_PYTHON"), encoding="utf-8"
    )

    def codes(worker: str) -> set[str]:
        problems = worker_skip_problems(
            worker_skip_markers(tmp_path),
            worker,
            workers=set(WORKER_KINDS),
            pinned=pinned.get(WORKER_KINDS[worker], set()),
        )
        return {item.code for item in problems}

    # The known skip, as its entry lists it; the other workers are clean.
    known = {"static:worker_skip"}
    entry = LEDGER.entry(_static_id(WORKER_KINDS["kimi"], "H01"))
    assert entry is None or set(entry.codes) == known
    assert codes("kimi") == known and codes("deepseek") == codes("codex") == set()
    # A skip on another worker's interpreter is that worker's own deviation.
    (tmp_path / "test_worker_live.py").write_text(
        marker.format("deepseek_mod._DEFAULT_WORKER_PYTHON"), encoding="utf-8"
    )
    assert codes("deepseek") == {"static:worker_skip_unpinned"}
    assert codes("kimi") == known and codes("codex") == set()
    # A Kimi skip in a file the inventory does not list changes the Kimi cell's
    # code set, so the comparator calls it a different deviation ...
    (tmp_path / "test_backends_kimi_extra.py").write_text(
        marker.format("parent._DEFAULT_WORKER_PYTHON"), encoding="utf-8"
    )
    assert codes("kimi") == known | {"static:worker_skip_unpinned"}
    # ... and a marker that names no worker at all is reported by every cell.
    (tmp_path / "test_backends_kimi_extra.py").unlink()
    (tmp_path / "test_other.py").write_text(
        marker.format("module._DEFAULT_WORKER_PYTHON"), encoding="utf-8"
    )
    assert codes("codex") == {"static:worker_skip_unpinned"}
    assert codes("kimi") == known | {"static:worker_skip_unpinned"}


def test_every_worker_project_has_an_h01_cell() -> None:
    projects = {
        path.name.removesuffix("_worker")
        for path in (REPOSITORY / "backends").glob("*_worker")
        if (path / "pyproject.toml").is_file()
    }

    assert projects == set(WORKER_KINDS)
    assert set(WORKER_KINDS.values()) <= set(REAL_KINDS)
    assert len(set(WORKER_KINDS.values())) == len(WORKER_KINDS)


@pytest.mark.parametrize("kind", REAL_KINDS, ids=lambda kind: _static_id(kind, "H02"))
def test_h02_every_backend_is_driven(kind: str) -> None:
    problems: list[Deviation] = []
    if kind not in drivers.load():
        problems.append(
            Deviation("static:no_driven_column", f"kind {kind} has no registered driver")
        )
    elif PACK_A not in cells.ENABLEMENT[kind]:
        problems.append(
            Deviation("static:pack_a_not_enabled", f"kind {kind} does not run pack A yet")
        )

    _verdict(_static_id(kind, "H02"), problems)


def test_every_backend_name_and_catalog_pair_has_a_kind() -> None:
    by_backend = {backend for backend, _transport in matrix_columns.KIND_TABLE}
    pairs = {
        (lane.backend, lane.transport)
        for product in matrix_columns.PRODUCTS
        for lane in matrix_columns.catalog_lanes(product)
    }

    assert set(BACKEND_NAMES) == by_backend
    assert pairs == set(matrix_columns.KIND_TABLE)
    assert set(cells.ENABLEMENT) == {*REAL_KINDS, TOY_KIND}


def test_h03_ledger_and_oracle_integrity() -> None:
    _verdict(
        _static_id(ledger.ALL_KINDS, "H03"),
        ledger.integrity_problems(LEDGER, cells.world()),
    )
    # Rules about the shipped ledger, never a table of today's gaps. Which
    # cells are ledgered and with which codes is the ledger's own statement:
    # the comparator holds each cell to it and the verifier holds a run to
    # it, so a gap fix deletes its entries and edits no test.
    for entry in LEDGER.entries:
        _kind, scenario, _role, lens = ledger.parse_cell_id(entry.cell)
        if lens == ledger.STATIC_LENS:
            # A static cell has no governing declared value.
            assert dict(entry.observed_under) == {}, entry.cell
        else:
            # A behaviour entry is locked to the value of every declared field
            # that governs its scenario, and to nothing else.
            assert set(entry.observed_under) == set(SCENARIOS[scenario].governing), entry.cell
    # A column-wide entry names a code of a contract clause on a real kind.
    for scope in LEDGER.column_wide:
        assert scope.kind in REAL_KINDS, scope
        assert contract.clause_of(scope.code) in contract.CLAUSES, scope


_LAUNCHERS = frozenset({"run", "Popen", "call", "check_call", "check_output"})


def installed_worker_environments(ci_text: str) -> list[str]:
    """The worker projects the tests job of the CI workflow names: ``<name>_worker``."""

    steps = yaml.safe_load(ci_text)["jobs"]["tests"]["steps"]
    return sorted(
        {
            match.group(1)
            for step in steps
            for match in re.finditer(
                r"--project[=\s]+(?:\./)?backends/(\w+)_worker", str(step.get("run", ""))
            )
        }
    )


def _autouse_fixtures(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Call)
            and ast.unparse(decorator.func).endswith("fixture")
            and any(
                keyword.arg == "autouse"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in decorator.keywords
            )
            for decorator in node.decorator_list
        )
    ]


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Call) and ast.unparse(item.func).endswith(name)
        for item in ast.walk(node)
    )


def _handed_interpreters(tests_root: Path, worker: str, workers: AbstractSet[str]) -> list[str]:
    """Test modules that hand a worker interpreter to ``subprocess`` themselves.

    A launch belongs to the worker that its file name or its argument names,
    and to every worker when it names none.
    """

    found: set[str] = set()
    for path in _python_files(tests_root):
        if tests_root / "backend_matrix" in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LAUNCHERS
                and node.args
                and "WORKER_PYTHON" in ast.unparse(node.args[0]).upper()
            ):
                continue
            text = f"{path.name} {ast.unparse(node.args[0])}".lower()
            named = {name for name in workers if name in text}
            if worker in named or not named:
                found.add(path.relative_to(tests_root).as_posix())
    return sorted(found)


def worker_environment_problems(ci_text: str, tests_root: Path) -> list[Deviation]:
    """H04: a worker environment may be installed in CI only behind the guard.

    For each worker environment the tests job installs: the worker harness and
    the launch interposer exist, ``tests/conftest.py`` holds the suite-wide
    interposer fixture and the file-name rule names the worker, and no test
    module hands that environment's interpreter to ``subprocess`` itself.
    """

    installed = installed_worker_environments(ci_text)
    if not installed:
        return []
    peers = tests_root / "backend_matrix" / "peers"
    tree = ast.parse((tests_root / "conftest.py").read_text(encoding="utf-8"))
    fixtures = _autouse_fixtures(tree)
    adapters: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_WORKER_ADAPTERS"
            for target in node.targets
        ):
            adapters = set(ast.literal_eval(node.value))
    rule = any(
        isinstance(node, ast.FunctionDef)
        and node.name == "names_a_worker_adapter"
        and "_WORKER_ADAPTERS" in ast.unparse(node)
        for node in tree.body
    ) and any(
        _calls(fixture, "netguard.watch") and _calls(fixture, "names_a_worker_adapter")
        for fixture in fixtures
    )
    guard = {
        "the worker harness": (peers / "worker_harness.py").is_file(),
        "the launch interposer": (peers / "launch.py").is_file(),
        "the suite-wide interposer fixture": any(
            _calls(fixture, "launch.suite_guard") for fixture in fixtures
        ),
    }
    problems: list[Deviation] = []
    for worker in installed:
        missing = [name for name, present in guard.items() if not present]
        if not rule or worker not in adapters:
            missing.append(f"the file-name rule for {worker!r}")
        handed = _handed_interpreters(tests_root, worker, set(WORKER_KINDS))
        if missing or handed:
            problems.append(
                Deviation(
                    "static:worker_env_unguarded",
                    f"CI installs the {worker} worker environment but {missing} "
                    f"is missing and {handed} start its interpreter themselves",
                )
            )
    return problems


def test_h04_worker_environments_only_behind_the_guard() -> None:
    ci_text = CI_WORKFLOW.read_text(encoding="utf-8")
    problems = worker_environment_problems(ci_text, TESTS)

    _verdict(_static_id(ledger.ALL_KINDS, "H04"), problems)
    # The cell is about something. CI installs exactly the worker environments
    # that a registered driver runs its column in (such a driver names the
    # interpreter of its environment), in every shard: a column that cannot
    # run is an error, and an environment no driver needs is not installed.
    driven = drivers.load()
    needed = sorted(
        worker
        for worker, kind in WORKER_KINDS.items()
        if kind in driven and hasattr(drivers.get(kind), "interpreter")
    )
    assert installed_worker_environments(ci_text) == needed != []


def _guarded_tests_root(root: Path, conftest: str) -> Path:
    """A tests directory with the guard's files and the given suite conftest."""

    peers = root / "backend_matrix" / "peers"
    peers.mkdir(parents=True)
    for name in ("worker_harness.py", "launch.py"):
        (peers / name).write_text("", encoding="utf-8")
    (root / "conftest.py").write_text(conftest, encoding="utf-8")
    return root


def test_h04_fails_when_ci_installs_a_worker_environment_before_the_guard(
    tmp_path: Path,
) -> None:
    ci_text = CI_WORKFLOW.read_text(encoding="utf-8")
    conftest = (TESTS / "conftest.py").read_text(encoding="utf-8")

    def problems(name: str, text: str = conftest, ci: str = ci_text) -> list[str]:
        root = _guarded_tests_root(tmp_path / name, text)
        found = worker_environment_problems(ci, root)
        assert {item.code for item in found} <= {"static:worker_env_unguarded"}
        return [item.detail for item in found]

    # The guard as it stands, on a tree with no test that starts a worker.
    assert problems("guarded") == []
    # Without the suite-wide fixture.
    assert conftest.count("launch.suite_guard(") == 1
    (missing,) = problems("no-fixture", conftest.replace("launch.suite_guard(", "launch.other("))
    assert "the suite-wide interposer fixture" in missing
    # With the fixture defined and not autouse.
    assert conftest.count("@pytest.fixture(autouse=True)\ndef locked_worker_launch") == 1
    (opt_in,) = problems(
        "not-autouse",
        conftest.replace(
            "@pytest.fixture(autouse=True)\ndef locked_worker_launch",
            "@pytest.fixture()\ndef locked_worker_launch",
        ),
    )
    assert "the suite-wide interposer fixture" in opt_in
    # Without the file-name rule, or with one that does not name the worker.
    (unruled,) = problems("no-rule", conftest.replace("netguard.watch(", "netguard.other("))
    assert "the file-name rule for 'deepseek'" in unruled
    (unnamed,) = problems("other-rule", conftest.replace('"kimi", "deepseek"', '"kimi",'))
    assert "the file-name rule for 'deepseek'" in unnamed
    # Without the harness.
    root = _guarded_tests_root(tmp_path / "no-harness", conftest)
    (root / "backend_matrix" / "peers" / "worker_harness.py").unlink()
    (no_harness,) = worker_environment_problems(ci_text, root)
    assert "the worker harness" in no_harness.detail
    # With a test module that hands the interpreter to subprocess itself. The
    # one of another worker is that worker's business.
    root = _guarded_tests_root(tmp_path / "handed", conftest)
    source = "import subprocess\n\nsubprocess.run([str({}._DEFAULT_WORKER_PYTHON), '-c', ''])\n"
    (root / "test_backends_kimi_own.py").write_text(source.format("parent"), encoding="utf-8")
    assert worker_environment_problems(ci_text, root) == []
    (root / "test_other.py").write_text(source.format("deepseek_mod"), encoding="utf-8")
    (handed,) = worker_environment_problems(ci_text, root)
    assert "['test_other.py'] start its interpreter themselves" in handed.detail
    # An install step for the Kimi worker on the real tree: its test modules
    # still start the interpreter themselves, so the step cannot merge yet.
    ci = yaml.safe_load(ci_text)
    ci["jobs"]["tests"]["steps"].append(
        {"run": "uv sync --frozen --project backends/kimi_worker"}
    )
    (kimi,) = worker_environment_problems(yaml.safe_dump(ci), TESTS)
    assert kimi.code == "static:worker_env_unguarded" and "the kimi worker" in kimi.detail
    assert "test_backends_kimi.py" in kimi.detail and "test_backends_kimi_k3.py" in kimi.detail
    # No install step, no demand.
    ci = yaml.safe_load(ci_text)
    ci["jobs"]["tests"]["steps"] = [
        step for step in ci["jobs"]["tests"]["steps"] if "_worker" not in str(step.get("run"))
    ]
    assert worker_environment_problems(yaml.safe_dump(ci), tmp_path / "absent") == []


def deselection_problems(pyproject: str, baseline: str) -> list[str]:
    """The documented way to leave the worker tests out keeps the configured expression.

    A ``-m`` on the command line replaces the one in ``addopts``. So the
    command the test baseline documents must be the configured expression
    with ``not matrix_worker and`` in front, or it would select the live
    tests.
    """

    options = tomllib.loads(pyproject)["tool"]["pytest"]["ini_options"]
    problems: list[str] = []
    if not any(str(marker).startswith("matrix_worker:") for marker in options["markers"]):
        problems.append("pyproject.toml does not register the matrix_worker marker")
    configured = re.search(r"-m\s+(['\"])(.+?)\1", str(options.get("addopts", "")))
    if configured is None:
        return [*problems, "addopts carries no -m expression to keep"]
    expected = f"not matrix_worker and {configured.group(2)}"
    documented = re.findall(r'-m\s+"([^"\n]*)"', baseline)
    commands = [expression for expression in documented if "matrix_worker" in expression]
    if expected not in commands:
        problems.append(
            f'tests/TEST_BASELINE.md does not document -m "{expected}", the configured '
            "expression with the worker tests left out"
        )
    for expression in commands:
        if expression != expected and f'shorter `-m "{expression}"`' not in baseline:
            problems.append(
                f'tests/TEST_BASELINE.md documents -m "{expression}", which replaces the '
                f"configured expression {configured.group(2)!r}"
            )
    return problems


def test_the_documented_deselection_is_the_configured_expression_with_the_marker() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    baseline = BASELINE.read_text(encoding="utf-8")
    full = "not matrix_worker and not live_docker and not llm_live"

    assert deselection_problems(pyproject, baseline) == []
    assert f'-m "{full}"' in baseline
    # The short form, documented as the command to run.
    (short,) = deselection_problems(pyproject, baseline.replace(full, "not matrix_worker"))
    assert "does not document" in short
    # A form that keeps only part of the configured expression.
    partial = "not matrix_worker and not llm_live"
    kept, replaced = deselection_problems(pyproject, baseline.replace(full, partial))
    assert "does not document" in kept and "replaces the configured expression" in replaced
    # The configured expression moved and the document did not.
    assert pyproject.count("not live_docker and not llm_live") == 1
    missing, stale = deselection_problems(
        pyproject.replace("not live_docker and not llm_live", "not live_docker"), baseline
    )
    assert '-m "not matrix_worker and not live_docker"' in missing
    assert "replaces the configured expression" in stale
    # The marker is registered where the expression is configured.
    (unregistered,) = deselection_problems(pyproject.replace("matrix_worker:", "worker:"), baseline)
    assert "does not register" in unregistered


# -- hygiene -------------------------------------------------------------------

_RULE_MODULES = ("contract.py", "scenarios.py", "ledger.py", "declarations.py")
_PUBLIC_BACKEND_MODULES = {
    "crucible.backends.config",
    "crucible.backends.features",
    "crucible.backends.limits",
    "crucible.backends.pricing",
}
_TEST_LIBRARIES = {"pytest", "yaml", "psutil"}


def _tree(name: str) -> ast.Module:
    return ast.parse((PACKAGE / name).read_text(encoding="utf-8"))


def test_package_has_no_skip_or_xfail_marker() -> None:
    uses: dict[tuple[str, str], int] = {}
    for path in _python_files(PACKAGE):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Attribute)
                and node.attr in {"skip", "skipif", "xfail", "importorskip"}
            ):
                continue
            owner = ast.unparse(node.value)
            if owner in {"pytest", "pytest.mark", "mark"}:
                key = (path.relative_to(PACKAGE).as_posix(), f"{owner}.{node.attr}")
                uses[key] = uses.get(key, 0) + 1

    # Only the comparator reports an expected failure; the self-test names the
    # exception type it raises. Nothing skips.
    assert uses == {
        ("ledger.py", "pytest.xfail"): 1,
        ("test_matrix_self.py", "pytest.xfail"): 1,
    }


@pytest.mark.parametrize("module", ["scenarios.py", "contract.py"])
def test_scenarios_and_contract_name_no_backend(module: str) -> None:
    forbidden = {*BACKEND_NAMES, *REAL_KINDS}
    tree = _tree(module)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }

    assert literals & forbidden == set()


def _import_problems(tree: ast.Module, *, private_ok: bool = False) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(tree):
        targets: list[tuple[str, str | None]]
        if isinstance(node, ast.Import):
            targets = [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names = [alias.name for alias in node.names]
                if not private_ok and any(name.startswith("_") for name in names):
                    problems.append(f"private matrix name in {ast.unparse(node)}")
                continue
            targets = [(node.module or "", alias.name) for alias in node.names]
        else:
            continue
        for module, name in targets:
            root = module.split(".")[0]
            private = any(part.startswith("_") for part in module.split(".")) or (
                name is not None and name.startswith("_") and name != "__future__"
            )
            if root == "__future__" or root in sys.stdlib_module_names:
                continue
            if root in _TEST_LIBRARIES or module == "assay":
                allowed = not private
            elif root == "crucible":
                allowed = not private and (private_ok or module in _PUBLIC_BACKEND_MODULES)
            else:
                allowed = False
            if not allowed:
                problems.append(f"{module}{'' if name is None else '.' + name}")
    return problems


@pytest.mark.parametrize("module", _RULE_MODULES)
def test_rule_modules_import_only_the_public_contract_surface(module: str) -> None:
    assert _import_problems(_tree(module)) == []


def test_gateway_drivers_patch_no_product_name() -> None:
    tree = _tree("drivers/openai_compat.py")
    imported = [
        (node.module or "", alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
        for alias in node.names
        if (node.module or "").startswith(("crucible", "assay"))
    ]

    # Routing is the boundary: the driver calls what a workflow calls.
    assert imported == [
        ("crucible.backends.config", "load_routing"),
        ("crucible.backends.config", "resolve_for_role"),
    ]
    for kind in (matrix_columns.HOSTED_GATEWAY_KIND, matrix_columns.LOCAL_GATEWAY_KIND):
        driver = drivers.get(kind)
        assert driver.private_seams == () and driver.fidelity == "wire"
        assert driver.bypassed_layers and driver.inexpressible == {}
    # The redirect and the guard are loadable by path inside a worker later.
    for name, allowed in (("tap.py", {"httpx"}), ("netguard.py", set())):
        assert _import_problems(_tree(name)) == sorted(allowed)


def test_claude_driver_reads_one_private_name_and_patches_none() -> None:
    tree = _tree("drivers/claude.py")
    imported = sorted(
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
        for alias in node.names
        if (node.module or "").startswith(("crucible", "assay"))
    )
    driver = drivers.get(matrix_columns.CLAUDE_KIND)

    # Routing is the boundary: the driver calls what a workflow calls. The one
    # private name is the adapter's table of reviewed tool versions, read for
    # the answer of the peer to the version probe.
    assert imported == [
        "crucible.backends._claude_session_transport."
        "CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL",
        "crucible.backends.config.load_routing",
        "crucible.backends.config.resolve_for_role",
    ]
    assert list(driver.private_seams) == [
        name for name in imported if any(part.startswith("_") for part in name.split("."))
    ]
    assert driver.fidelity == "wire" and driver.dialect == "claude_cli"
    assert len(driver.bypassed_layers) == 2 and driver.inexpressible == {}
    # Nothing of the product is replaced: the driver sets no attribute and
    # uses no patching helper.
    called = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert not {name for name in called if "setattr" in name or "patch" in name}
    # The peer runs outside the test environment, on the standard library alone.
    assert _import_problems(_tree("peers/claude_cli.py")) == []
    assert _import_problems(_tree("peers/__init__.py")) == []


def test_deepseek_driver_patches_one_private_name_through_the_launch_interposer() -> None:
    tree = _tree("drivers/deepseek.py")
    imported = sorted(
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
        for alias in node.names
        if (node.module or "").startswith(("crucible", "assay"))
    )
    driver = drivers.get(matrix_columns.DEEPSEEK_KIND)

    # Routing is the boundary: the driver calls what a workflow calls, and
    # takes the adapter module for its public capture function, its two
    # public error types and the launch seam.
    assert imported == [
        "crucible.backends.config.load_routing",
        "crucible.backends.config.resolve_for_role",
        "crucible.backends.deepseek",
    ]
    from crucible.backends import deepseek as adapter  # noqa: PLC0415

    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "adapter"
    }
    assert used - {"__name__"} <= set(adapter.__all__)
    # The one private name is the supervisor function, which the driver hands
    # to the launch interposer by name and never sets itself.
    (seam,) = driver.private_seams
    module, _, name = seam.rpartition(".")
    assert (module, name) in launch.SUPERVISOR_SEAMS and module == adapter.__name__
    assert name.startswith("_") and callable(getattr(adapter, name))
    called = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert not {item for item in called if "setattr" in item or "patch" in item}
    assert "launch.at_supervisor" in called
    assert driver.fidelity == "wire" and driver.dialect == "deepseek_responses"
    assert len(driver.bypassed_layers) == 3 and driver.inexpressible == {}
    # The harness runs in a locked interpreter, on the standard library alone,
    # and the interposer is imported by the suite's conftest for every test.
    assert _import_problems(_tree("peers/worker_harness.py")) == []
    assert _import_problems(_tree("peers/launch.py")) == []
    assert worker_harness.SETTINGS_ENV.startswith("CRUCIBLE_MATRIX_")
    # The seams the suite fixture interposes at exist as the adapters name them.
    for module, name in (*launch.SUPERVISOR_SEAMS, *launch.SUBPROCESS_SEAMS):
        assert hasattr(importlib.import_module(module), name), (module, name)


def test_every_envelope_is_rendered_by_a_dialect_and_exists() -> None:
    rendered = {
        name for kind in sorted(drivers.load()) for name in drivers.get(kind).envelopes
    }
    stored = envelopes.all_envelopes()

    # An envelope no dialect refers to is dead weight with a grade; a dialect
    # that refers to a missing one cannot answer.
    assert rendered == set(stored) != set()
    assert {path.suffix for path in envelopes.ROOT.rglob("*") if path.is_file()} <= {
        ".py",
        ".yaml",
        ".pyc",
    }
    assert drivers.get(TOY_KIND).envelopes == ()
    # Every stored envelope states its grade and the evidence the grade needs
    # (parsing refuses anything else), and stores nothing credential-shaped.
    for name, envelope in stored.items():
        path = envelopes.ROOT / f"{name}.yaml"
        assert envelope.grade in envelopes.GRADES and envelope.slots() <= envelopes.SLOTS
        assert envelopes.hygiene_problems(path.read_text(encoding="utf-8")) == []


def test_every_driven_kind_names_its_test_module() -> None:
    driven = drivers.load()

    assert set(cells.MODULE_BY_KIND) == driven
    for module in {*cells.MODULE_BY_KIND.values(), cells.STATIC_MODULE}:
        source = (PACKAGE / f"{module}.py").read_text(encoding="utf-8")
        closing = cells.STATIC_CHECK if module == cells.STATIC_MODULE else cells.CANARY_CHECK
        # Each module ends with its closing check.
        last = [
            node.name
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ][-1]
        assert last == closing, module
    assert set(cells.STATIC_TESTS) == set(cells.STATIC_SCENARIOS)
    static_source = ast.parse((PACKAGE / f"{cells.STATIC_MODULE}.py").read_text(encoding="utf-8"))
    defined = {node.name for node in static_source.body if isinstance(node, ast.FunctionDef)}
    assert {test for test, _parametrised in cells.STATIC_TESTS.values()} <= defined


def test_toy_driver_touches_no_private_product_name() -> None:
    tree = _tree("drivers/toy.py")
    crucible_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and not node.level
        and (node.module or "").startswith(("crucible", "assay"))
    ]

    assert toy.ToyDriver.private_seams == ()
    assert crucible_imports
    for node in crucible_imports:
        assert not any(part.startswith("_") for part in (node.module or "").split("."))
        assert not any(alias.name.startswith("_") for alias in node.names)


def test_import_rule_sees_what_it_forbids() -> None:
    source = [
        "import httpx",
        "from crucible.backends.codex import CodexBackend",
        "from crucible.backends.limits import _TERMINAL_ATTRIBUTE",
        "from crucible.tools._dispatch_budget import ToolDispatchBudgetExceeded",
        "from assay.types.invocation import Usage",
        "from .drivers.toy import _Run",
        "from assay import Usage",
        "from crucible.backends.limits import resolve_invocation_limits",
    ]

    assert _import_problems(ast.parse("\n".join(source))) == [
        "httpx",
        "crucible.backends.codex.CodexBackend",
        "crucible.backends.limits._TERMINAL_ATTRIBUTE",
        "crucible.tools._dispatch_budget.ToolDispatchBudgetExceeded",
        "assay.types.invocation.Usage",
        "private matrix name in from .drivers.toy import _Run",
    ]


def test_model_ids_and_profile_names_live_in_columns_only() -> None:
    literals = matrix_columns.catalog_literals()
    offenders = sorted(
        f"{path.relative_to(PACKAGE).as_posix()}: {literal}"
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".md", ".json"}
        and path.name != "columns.py"
        for literal in literals
        if literal in path.read_text(encoding="utf-8")
    )

    assert len(literals) >= 11
    assert offenders == []
    # Model ids are read from the catalogs: even columns.py types none.
    columns_source = (PACKAGE / "columns.py").read_text(encoding="utf-8")
    models = {
        lane.default_route.model
        for lane in matrix_columns.catalog_lanes(matrix_columns.COLUMN_PRODUCT)
    }
    assert [model for model in sorted(models) if model in columns_source] == []


def test_routing_overlay_is_a_closed_list() -> None:
    references: set[str] = set()
    for product in matrix_columns.PRODUCTS:
        for lane in matrix_columns.catalog_lanes(product):
            text = Path(lane.routing_path).read_text(encoding="utf-8")
            code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
            references.update(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", code))

    # Every ${VAR} a shipped route expands has a scripted value, and no other.
    assert references == set(matrix_columns.SCRIPTED_ENV)
    assert set(matrix_columns.ALLOWED_OVERLAY) <= set(REAL_KINDS)
    for fields in matrix_columns.ALLOWED_OVERLAY.values():
        assert all(re.fullmatch(r"extra\.[a-z_]+", name) for name in fields)
    for kind in sorted(drivers.load()):
        allowed = set(matrix_columns.ALLOWED_OVERLAY.get(kind, ()))
        for column in (item for item in cells.all_columns() if item.kind == kind):
            overlay = drivers.get(kind).overlay(column, matrix_columns.SCRIPTED_ENV)
            assert set(overlay) <= allowed


def test_difference_tables_are_total_and_cite_their_fallbacks() -> None:
    for scenario in SCENARIOS.values():
        assert declarations.totality_gaps(scenario.differences) == [], scenario.id
        for table in scenario.differences.values():
            for variant in table.values():
                assert variant.kind in {"BASE", "FALLBACK", "REFUSED"}
                assert variant.kind != "FALLBACK" or variant.citation


def test_every_scenario_passes_the_complete_base_policy() -> None:
    for scenario in SCENARIOS.values():
        policy = scenario.policy
        assert (
            policy.retry_budget,
            policy.timeout_seconds,
            policy.on_cancellation,
            policy.on_max_turns,
        ) == (1, 60, "convert_to_backend_error", "raise"), scenario.id


def test_every_recorded_code_is_in_the_closed_vocabulary() -> None:
    recorded = {code for liar in toy.LIARS for code in liar.codes}
    recorded.update(code for entry in LEDGER.entries for code in entry.codes)
    recorded.update(scope.code for scope in LEDGER.column_wide)
    recorded.update(
        code
        for record in LEDGER.registry.values()
        for item in record.cells
        for code in item.predicted
    )

    assert {code: contract.clause_of(code) for code in recorded}
    assert all(liar.scenario in SCENARIOS for liar in toy.LIARS)


def test_data_files_carry_the_license_header() -> None:
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            head = path.read_text(encoding="utf-8").splitlines()[:2]
            assert head == [
                "# SPDX-FileCopyrightText: 2026 Whact Inc.",
                "# SPDX-License-Identifier: MIT",
            ], path.name


# -- the module ends with its own inventory check ------------------------------


def test_every_static_cell_of_the_inventory_was_judged() -> None:
    assert set(cells.static_cell_ids()) == _JUDGED
