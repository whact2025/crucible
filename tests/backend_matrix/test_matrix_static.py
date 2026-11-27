# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Static and coverage cells of the matrix, and the package's hygiene checks.

A static cell needs no driver and computes deviation codes like any cell. H01
and H02 report product gaps and go through the comparator, so the ones that
are red today (H02 for every kind that has no driver yet or does not run pack
A yet) are strict expected failures tied to their gap ids. The other static
cells check the matrix itself and must be clean whatever the ledger says.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib
import importlib.metadata
import inspect
import re
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import AbstractSet, Any, Iterator, Mapping, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

import assay
from assay import BackendFeatureSet
from crucible.backends import BACKEND_NAMES

from . import cells, contract, declarations, drivers, envelopes, ledger, mutation, runner
from . import columns as matrix_columns
from . import conftest as matrix_conftest
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
            # A behaviour entry is locked to the value of every key that
            # governs its scenario (a declared field, or the route extra
            # that keys its table), and to nothing else.
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


def _launcher_fixture(conftest: ast.Module) -> str | None:
    """The suite fixture whose ``run`` and ``popen`` start a worker under the harness."""

    for fixture in _autouse_fixtures(conftest):
        if _calls(fixture, "launch.suite_guard"):
            return fixture.name
    return None


def _through_the_launcher(tree: ast.Module, launcher: str | None) -> set[int]:
    """The calls of ``tree`` that are a method of the suite's launcher fixture.

    Only where the function the call is in takes the fixture by its own name
    as a parameter and never rebinds that name: anything else is a name the
    test chose, which may be ``subprocess`` itself.
    """

    through: set[int] = set()
    if launcher is None:
        return through
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        taken = {item.arg for item in (*arguments.args, *arguments.kwonlyargs)}
        rebound = any(
            isinstance(node, ast.Name) and node.id == launcher and isinstance(node.ctx, ast.Store)
            for node in ast.walk(function)
        )
        if launcher not in taken or rebound:
            continue
        through.update(
            id(node)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == launcher
        )
    return through


def _handed_interpreters(
    tests_root: Path,
    worker: str,
    workers: AbstractSet[str],
    launcher: str | None = None,
) -> list[str]:
    """Test modules that hand a worker interpreter to ``subprocess`` themselves.

    A launch belongs to the worker that its file name or its argument names,
    and to every worker when it names none. A launch through the launcher of
    the suite fixture (``launcher`` is the fixture's name) runs under the
    harness and is not one.
    """

    found: set[str] = set()
    for path in _python_files(tests_root):
        if tests_root / "backend_matrix" in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = _through_the_launcher(tree, launcher)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LAUNCHERS
                # ``asyncio.run`` runs a coroutine and starts no process.
                and ast.unparse(node.func.value) != "asyncio"
                and node.args
                and "WORKER_PYTHON" in ast.unparse(node.args[0]).upper()
                and id(node) not in guarded
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
        handed = _handed_interpreters(
            tests_root, worker, set(WORKER_KINDS), _launcher_fixture(tree)
        )
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


def _without_install_step(ci_text: str, worker: str) -> str:
    """The CI workflow without the step that installs one worker environment."""

    ci = yaml.safe_load(ci_text)
    ci["jobs"]["tests"]["steps"] = [
        step
        for step in ci["jobs"]["tests"]["steps"]
        if f"backends/{worker}_worker" not in str(step.get("run"))
    ]
    return str(yaml.safe_dump(ci))


def _only_install_step(ci_text: str, worker: str) -> str:
    """The CI workflow with the install step of one worker environment and of no other."""

    for other in installed_worker_environments(ci_text):
        if other != worker:
            ci_text = _without_install_step(ci_text, other)
    return ci_text


def test_h04_fails_when_ci_installs_a_worker_environment_before_the_guard(
    tmp_path: Path,
) -> None:
    every = CI_WORKFLOW.read_text(encoding="utf-8")
    # The cases up to those of one worker's own are about one environment,
    # the DeepSeek one, whatever else the workflow installs.
    ci_text = _only_install_step(every, "deepseek")
    assert installed_worker_environments(ci_text) == ["deepseek"]
    assert len(installed_worker_environments(every)) > 1
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
    assert conftest.count('"deepseek", ') == 1
    (unnamed,) = problems("other-rule", conftest.replace('"deepseek", ', ""))
    assert "the file-name rule for 'deepseek'" in unnamed
    # Without the harness: one deviation, and one per environment the
    # workflow installs.
    root = _guarded_tests_root(tmp_path / "no-harness", conftest)
    (root / "backend_matrix" / "peers" / "worker_harness.py").unlink()
    (no_harness,) = worker_environment_problems(ci_text, root)
    assert "the worker harness" in no_harness.detail
    unguarded = worker_environment_problems(every, root)
    assert len(unguarded) == len(installed_worker_environments(every))
    assert all("the worker harness" in item.detail for item in unguarded)
    # With a test module that hands the interpreter to subprocess itself. The
    # one of another worker is that worker's business.
    root = _guarded_tests_root(tmp_path / "handed", conftest)
    source = "import subprocess\n\nsubprocess.run([str({}._DEFAULT_WORKER_PYTHON), '-c', ''])\n"
    (root / "test_backends_kimi_own.py").write_text(source.format("parent"), encoding="utf-8")
    assert worker_environment_problems(ci_text, root) == []
    (root / "test_other.py").write_text(source.format("deepseek_mod"), encoding="utf-8")
    (handed,) = worker_environment_problems(ci_text, root)
    assert "['test_other.py'] start its interpreter themselves" in handed.detail
    # With every environment the workflow installs. A coroutine runner starts
    # no process, whatever its coroutine is given; a launcher in a module
    # named after another worker is that worker's.
    assert "codex" in installed_worker_environments(every)
    root = _guarded_tests_root(tmp_path / "coroutine", conftest)
    module = root / "test_codex_official_own.py"
    module.write_text(
        "import asyncio\n\nasyncio.run(start(worker_python=WORKER_PYTHON))\n", encoding="utf-8"
    )
    assert worker_environment_problems(every, root) == []
    module.write_text(
        "import subprocess\n\nsubprocess.run([WORKER_PYTHON, '-c', ''])\n", encoding="utf-8"
    )
    (codex,) = worker_environment_problems(every, root)
    assert "the codex worker" in codex.detail and module.name in codex.detail
    # With the install step of the Kimi worker: on the real tree its test
    # modules start that interpreter through the launcher of the suite
    # fixture, which runs it under the harness, so the step is behind the
    # guard.
    assert installed_worker_environments(every) == ["codex", "deepseek", "kimi"]
    assert worker_environment_problems(every, TESTS) == []
    launch_line = "{}.run([str(parent._DEFAULT_WORKER_PYTHON), '-c', ''])\n"
    through = "def test_x(locked_worker_launch):\n    " + launch_line.format(
        "locked_worker_launch"
    )
    root = _guarded_tests_root(tmp_path / "launcher", conftest)
    (root / "test_backends_kimi_own.py").write_text(through, encoding="utf-8")
    assert worker_environment_problems(every, root) == []
    # Not behind it: the same line under a name the fixture is not (a helper
    # that takes it under another name), under the fixture's name where no
    # function takes the fixture, and where the function rebinds the name.
    for name, source in (
        ("alias", through.replace("locked_worker_launch", "starter")),
        ("no-fixture", launch_line.format("locked_worker_launch")),
        (
            "rebound",
            "import subprocess\n\n\ndef test_x(locked_worker_launch):\n"
            "    locked_worker_launch = subprocess\n    "
            + launch_line.format("locked_worker_launch"),
        ),
        ("subprocess", "import subprocess\n\n" + launch_line.format("subprocess")),
    ):
        root = _guarded_tests_root(tmp_path / f"handed-{name}", conftest)
        (root / "test_backends_kimi_own.py").write_text(source, encoding="utf-8")
        (kimi,) = worker_environment_problems(every, root)
        assert kimi.code == "static:worker_env_unguarded" and "the kimi worker" in kimi.detail
        assert "['test_backends_kimi_own.py'] start its interpreter themselves" in kimi.detail
    # Without the fixture in the suite conftest nothing is its launcher.
    root = _guarded_tests_root(
        tmp_path / "launcher-without-fixture",
        conftest.replace("launch.suite_guard(", "launch.other("),
    )
    (root / "test_backends_kimi_own.py").write_text(through, encoding="utf-8")
    assert any(
        "['test_backends_kimi_own.py'] start its interpreter" in item.detail
        for item in worker_environment_problems(every, root)
    )
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


_HOSTED, _LOCAL = matrix_columns.HOSTED_GATEWAY_KIND, matrix_columns.LOCAL_GATEWAY_KIND
# The proof the row of the hosted kind rests on, as its module names it.
_KEY_BOUND_AT_LOAD = "test_the_hosted_route_binds_its_key_while_routing_loads_the_file"


def _gateway_pin_problems(
    rows: Sequence[Mapping[str, str]], required: Mapping[str, Sequence[str]]
) -> list[str]:
    """Why the rows of the two gateway kinds are not the one row they may have.

    The hosted kind pins the missing-credential scenario and nothing else,
    the local kind nothing, and the proof that row rests on is required by
    name for the routing file of every product. So a row put on the local
    kind, and the row left behind once its proof is no longer required, are
    both found here: no integrity rule knows the ground of a row.
    """

    problems: list[str] = []
    pinned = sorted(
        (row["kind"], row["scenario"]) for row in rows if row["kind"] in (_HOSTED, _LOCAL)
    )
    if pinned != [(_HOSTED, "S28")]:
        problems.append(
            f"the gateway kinds pin {pinned} not expressible; exactly {(_HOSTED, 'S28')} "
            "may be pinned"
        )
    held = required.get(cells.MODULE_BY_KIND[_HOSTED], ())
    missing = sorted(
        product
        for product in matrix_columns.PRODUCTS
        if f"{_KEY_BOUND_AT_LOAD}[{product}]" not in held
    )
    if pinned and missing:
        problems.append(
            f"the verifier does not require {_KEY_BOUND_AT_LOAD} for the routing file of "
            f"{missing}: a row of the gateway kinds rests on it"
        )
    return problems


def test_a_row_on_the_local_gateway_or_one_without_its_proof_is_found() -> None:
    # The two lying counterparts of the statement in the test below, each on
    # an inventory held in memory: the row of the hosted kind copied to the
    # local kind, and the shipped rows with the proof taken off the list of
    # what the verifier requires.
    rows = cells.load_inventory().not_expressible
    (row,) = (item for item in rows if item["kind"] == _HOSTED)
    module = cells.MODULE_BY_KIND[_HOSTED]
    without_the_proof = {
        name: tuple(item for item in items if not item.startswith(_KEY_BOUND_AT_LOAD))
        for name, items in cells.REQUIRED_PROOFS.items()
    }

    assert _gateway_pin_problems(rows, cells.REQUIRED_PROOFS) == []
    on_local = _gateway_pin_problems([*rows, {**row, "kind": _LOCAL}], cells.REQUIRED_PROOFS)
    assert len(on_local) == 1 and "exactly" in on_local[0]
    unproven = _gateway_pin_problems(rows, without_the_proof)
    assert len(unproven) == 1 and "does not require" in unproven[0]
    # That proof alone was taken off the list, for every product: the other
    # proofs of the module are still required.
    assert len(without_the_proof[module]) == len(cells.REQUIRED_PROOFS[module]) - len(
        matrix_columns.PRODUCTS
    )
    # With no row at all the proof is not asked for; the row alone is.
    no_row = [item for item in rows if item["kind"] != _HOSTED]
    assert _gateway_pin_problems(no_row, without_the_proof) == [
        f"the gateway kinds pin [] not expressible; exactly {(_HOSTED, 'S28')} may be pinned"
    ]


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
        assert driver.bypassed_layers
    # What the two kinds pin not expressible, stated exactly: the hosted
    # kind the one scenario whose subject is a state of its route, the local
    # kind nothing; and the proof that row rests on is required by name.
    inventory = cells.load_inventory()
    assert _gateway_pin_problems(inventory.not_expressible, cells.REQUIRED_PROOFS) == []
    assert [name.split("/", 1)[0] for name in drivers.get(_HOSTED).inexpressible] == ["S28"]
    assert drivers.get(_LOCAL).inexpressible == {}
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
    assert len(driver.bypassed_layers) == 2
    # The one step the frames of the tool cannot carry, a refusal part; and
    # the one a session of the tool cannot, a call whose arguments are no
    # object.
    assert list(driver.inexpressible) == ["S23b/Refuse(refusal_part)", "S05/Call(malformed)"]
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


def test_kimi_driver_touches_two_private_names_and_patches_through_the_interposer() -> None:
    tree = _tree("drivers/kimi.py")
    imported = sorted(
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
        for alias in node.names
        if (node.module or "").startswith(("crucible", "assay"))
    )
    driver = drivers.get(matrix_columns.KIMI_KIND)

    # Routing is the boundary: the driver calls what a workflow calls. It
    # takes the adapter module for its public capture function, its two
    # public error types and its two seams, and the public names of the limit
    # policy that a route with the cost upper bound must be bound with.
    assert imported == [
        "crucible.backends.config.load_routing",
        "crucible.backends.config.resolve_for_role",
        "crucible.backends.cost_bounds.DIAGNOSTIC_COST_POLICY",
        "crucible.backends.kimi",
        "crucible.backends.limits.INVOCATION_LIMIT_LEDGER_LEAF",
        "crucible.backends.limits.InvocationLimitLedger",
    ]
    from crucible.backends import kimi as adapter  # noqa: PLC0415

    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "adapter"
    }
    assert not {name for name in used - {"__name__"} if name.startswith("_")}
    assert all(hasattr(adapter, name) for name in used)
    # The two private names are reached by name only: the launch seam, which
    # the driver hands to the launch interposer and never sets itself, and the
    # adapter's memory of a passed probe, which it clears and never replaces.
    seam, probe_cache = driver.private_seams
    module, _, name = seam.rpartition(".")
    assert (module, name) in launch.SUBPROCESS_SEAMS and module == adapter.__name__
    module, _, name = probe_cache.rpartition(".")
    assert module == adapter.__name__ and name.startswith("_")
    assert callable(getattr(adapter, name)) and callable(getattr(adapter, name).cache_clear)
    called = {
        ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert not {item for item in called if "setattr" in item or "patch" in item}
    assert "launch.at_subprocess" in called
    assert driver.fidelity == "wire" and driver.dialect == "moonshot_chat"
    # The scripted wire is the provider's chat stream, which the worker
    # reads: no step of a scenario is pinned on it.
    assert len(driver.bypassed_layers) == 3 and driver.inexpressible == {}
    # The driver overlays nothing on the shipped routes.
    assert matrix_columns.KIMI_KIND not in matrix_columns.ALLOWED_OVERLAY


def test_a_test_that_requires_a_worker_environment_carries_the_marker() -> None:
    # A test outside the package that needs a locked worker environment takes
    # the fixture that fails without it, and the marker the documented
    # deselection leaves out: both or neither, so that a machine without the
    # environment gets an error or a counted deselection, never a quiet pass.
    fixture, marker = "require_kimi_worker", "matrix_worker"
    conftest = _tree("../conftest.py")
    assert fixture in {
        node.name for node in conftest.body if isinstance(node, ast.FunctionDef)
    }
    paired = 0
    for path in _python_files(TESTS):
        if PACKAGE in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            decorators = [ast.unparse(item) for item in node.decorator_list]
            requires = fixture in {item.arg for item in node.args.args} or any(
                "usefixtures" in item and fixture in item for item in decorators
            )
            marked = any(item.endswith(f"mark.{marker}") for item in decorators)
            assert requires == marked, f"{path.name}::{node.name}"
            paired += requires
    assert paired > 0


_APP_SERVER_INEXPRESSIBLE = [
    "S23a/Refuse(content_filter)",
    "S23b/Refuse(refusal_part)",
    # On the session wire of both routes a tool call is a decoded object.
    "S05/Call(malformed)",
    "S24/ServedAs",
]


def test_codex_drivers_touch_only_the_seams_they_list() -> None:
    def imported(tree: ast.Module) -> list[str]:
        return sorted(
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and not node.level
            for alias in node.names
            if (node.module or "").startswith(("crucible", "assay"))
        )

    def used(tree: ast.Module, name: str) -> set[str]:
        return {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        }

    def called(tree: ast.Module) -> set[str]:
        return {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}

    # The app-server column. Routing is the boundary; of the adapter module
    # the driver touches two private names and lists both: the native-binary
    # predicate, which it replaces for as long as a backend is open, and the
    # table of reviewed tool versions, which it reads.
    legacy = _tree("drivers/codex_legacy.py")
    driver = drivers.get(matrix_columns.CODEX_LEGACY_KIND)
    assert imported(legacy) == [
        "crucible.backends.codex",
        "crucible.backends.config.load_routing",
        "crucible.backends.config.resolve_for_role",
    ]
    from crucible.backends import codex as adapter  # noqa: PLC0415

    private = {name for name in used(legacy, "adapter") if name.startswith("_")} - {"__name__"}
    assert {f"{adapter.__name__}.{name}" for name in private} == set(driver.private_seams)
    assigned = {
        ast.unparse(target)
        for node in ast.walk(legacy)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "adapter"
    }
    assert assigned == {"adapter._is_native_codex_binary"}
    assert all(hasattr(adapter, name) for name in private)
    assert not {name for name in called(legacy) if "setattr" in name or "patch" in name}
    assert driver.fidelity == "wire" and driver.dialect == "codex_app_server"
    assert len(driver.bypassed_layers) == 3
    # What the notifications of the app-server cannot carry: a finish reason
    # of a model response and a refusal part of a message.
    assert list(driver.inexpressible) == _APP_SERVER_INEXPRESSIBLE

    # The official-SDK column. Of the worker client the driver uses public
    # names only; its seams are the two module globals it hands to the launch
    # interposer by name and never sets itself.
    official = _tree("drivers/codex_official.py")
    driver = drivers.get(matrix_columns.CODEX_OFFICIAL_KIND)
    assert imported(official) == [
        "crucible.backends.codex_official_worker",
        "crucible.backends.config.load_routing",
        "crucible.backends.config.resolve_for_role",
    ]
    from crucible.backends import codex_official_worker as worker_client  # noqa: PLC0415

    assert used(official, "worker_client") - {"__name__"} <= set(worker_client.__all__)
    seams = {(worker_client.__name__, name.rpartition(".")[2]) for name in driver.private_seams}
    assert seams == {
        seam
        for seam in (*launch.SUBPROCESS_SEAMS, *launch.ASYNCIO_SEAMS)
        if seam[0] == worker_client.__name__
    }
    assert len(seams) == 2 and all(hasattr(worker_client, name) for _module, name in seams)
    assert not {name for name in called(official) if "setattr" in name or "patch" in name}
    assert {"launch.at_subprocess", "launch.at_asyncio"} <= called(official)
    assert driver.fidelity == "wire" and driver.dialect == "codex_app_server"
    assert len(driver.bypassed_layers) == 3
    # What the notifications of the app-server cannot carry: a finish reason
    # of a model response and a refusal part of a message.
    assert list(driver.inexpressible) == _APP_SERVER_INEXPRESSIBLE
    # Both columns answer from the same envelopes, through the same peer.
    assert driver.envelopes == drivers.get(matrix_columns.CODEX_LEGACY_KIND).envelopes
    # That dialect names no served model (it refuses the step that would),
    # and the kinds that speak it are the ones the difference of that wire is
    # owed on (H03).
    assert matrix_columns.UNNAMED_SERVED_MODEL_KINDS == {
        kind for kind in drivers.load() if drivers.get(kind).dialect == driver.dialect
    }
    # The peer and the stage run outside the test environment, on the
    # standard library alone.
    assert _import_problems(_tree("peers/codex_app_server.py")) == []
    assert _import_problems(_tree("peers/codex_worker.py")) == []
    for module, name in launch.ASYNCIO_SEAMS:
        assert hasattr(importlib.import_module(module), name), (module, name)


def _private_product_names_replaced(tree: ast.Module, functions: AbstractSet[str]) -> set[str]:
    """The private names of the product that the named functions of a module replace.

    A replacement is a call of a ``setattr`` method (the patching helper of
    pytest) on a name the module imports from the product, with an attribute
    that begins with an underscore.
    """

    imported = {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and not node.level
        and (node.module or "").split(".")[0] in {"crucible", "assay"}
        for alias in node.names
    }
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in functions:
            continue
        for call in ast.walk(node):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "setattr"
                and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and str(call.args[1].value).startswith("_")
            ):
                continue
            parts: list[str] = []
            target = call.args[0]
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name) and target.id in imported:
                name = str(call.args[1].value)
                found.add(".".join([imported[target.id], *reversed(parts), name]))
    return found


def test_the_private_names_a_required_proof_replaces_are_listed_beside_the_proofs() -> None:
    # A required proof is evidence an entry of the ledger leans on, as a
    # cell is. What a driver replaces of the product is in its list of
    # private seams; what a required proof replaces is in the list beside the
    # proofs, module by module, and the report of a run prints both.
    assert set(cells.PROOF_SEAMS) <= set(cells.REQUIRED_PROOFS)
    for module, proofs in cells.REQUIRED_PROOFS.items():
        names = {proof.split("[", 1)[0] for proof in proofs}
        replaced = _private_product_names_replaced(_tree(f"{module}.py"), names)
        assert replaced == set(cells.PROOF_SEAMS.get(module, ())), module
    for seams in cells.PROOF_SEAMS.values():
        for seam in seams:
            owner, _, name = seam.rpartition(".")
            module_name, _, class_name = owner.rpartition(".")
            assert name.startswith("_")
            assert hasattr(getattr(importlib.import_module(module_name), class_name), name)
    for kind in sorted(drivers.load()):
        described = matrix_report.describe_kind(kind)
        assert described["proof_seams"] == list(
            cells.PROOF_SEAMS.get(cells.module_of(kind), ())
        )
        assert described["private_seams"] == list(drivers.get(kind).private_seams)
        # A third list: what the watch of write hosts wraps, named for a
        # kind once a cell of it that attaches write tools has played.
        assert set(described["watch_seams"]) <= set(mutation.PRIVATE_SEAMS)
    assert matrix_report.describe_kind(matrix_columns.CLAUDE_KIND)["proof_seams"]
    # The rule sees what it is there for: a proof that replaces private names
    # nobody listed, on a class and through a module alias. A public name and
    # a test that is no required proof are not its subject.
    lying = ast.parse(
        "from crucible.backends import kimi as adapter\n"
        "from crucible.backends.claude import ClaudeBackend\n"
        "def test_proof(monkeypatch):\n"
        "    monkeypatch.setattr(ClaudeBackend, '_invoke_once', None)\n"
        "    monkeypatch.setattr(adapter.KimiBackend, '_result_from_outcome', None)\n"
        "    monkeypatch.setattr(ClaudeBackend, 'attach_toolset', None)\n"
        "def test_other(monkeypatch):\n"
        "    monkeypatch.setattr(ClaudeBackend, '_tool_authority_digest', None)\n"
    )
    assert _private_product_names_replaced(lying, {"test_proof"}) == {
        "crucible.backends.claude.ClaudeBackend._invoke_once",
        "crucible.backends.kimi.KimiBackend._result_from_outcome",
    }
    assert _private_product_names_replaced(lying, {"test_missing"}) == set()


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


# -- a mapping key is written once ------------------------------------------------


def keys_written_twice(text: str) -> list[str]:
    """The mapping keys a YAML text writes more than once, at any depth.

    The loader of the package keeps the last of two equal keys and says
    nothing, so a record, a code or a whole block that is written under a
    key a second time takes the place of the first.
    """

    found: list[tuple[int, str]] = []

    class Strict(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any, deep: bool = False) -> Any:
        first: dict[Any, int] = {}
        for key_node, _value in node.value:
            key = loader.construct_object(key_node, deep=True)
            line = key_node.start_mark.line + 1
            if key in first:
                found.append((line, f"{key!r} is already a key at line {first[key]}"))
            first.setdefault(key, line)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    loader = Strict(text)
    try:
        loader.get_single_data()
    finally:
        loader.dispose()
    # A mapping is read when the loader comes to it, the outer one first.
    return [f"line {line}: {problem}" for line, problem in sorted(found)]


def test_no_yaml_file_of_the_package_writes_a_mapping_key_twice() -> None:
    files = sorted(PACKAGE.rglob("*.yaml"))
    problems = [
        f"{path.relative_to(PACKAGE).as_posix()} {problem}"
        for path in files
        for problem in keys_written_twice(path.read_text(encoding="utf-8"))
    ]

    # The ledger, the inventory, the snapshot of the declarations and every
    # envelope: each key of each mapping stands once.
    data = (ledger.LEDGER_PATH, cells.INVENTORY_PATH, declarations.SNAPSHOT_PATH)
    assert {path.resolve() for path in data} <= set(files)
    assert len(files) == len(data) + len(envelopes.all_envelopes())
    assert problems == []
    # The check is loud where the loader is not: two records under one id,
    # and a top-level block that is opened a second time.
    twice = (
        "registry:\n"
        "  G01: {title: the first}\n"
        "  G01: {title: the second}\n"
        "known_gaps: [one]\n"
        "known_gaps:\n"
    )
    assert yaml.safe_load(twice) == {
        "registry": {"G01": {"title": "the second"}},
        "known_gaps": None,
    }
    assert keys_written_twice(twice) == [
        "line 3: 'G01' is already a key at line 2",
        "line 5: 'known_gaps' is already a key at line 4",
    ]
    # The same key in two mappings is no repetition.
    assert keys_written_twice("a: {title: one}\nb: {title: one}\n") == []


# -- stated versions -----------------------------------------------------------

TEST_PROCESS = "the test process"
# A release typed in words: three dotted numbers (an address has four).
_TYPED_RELEASE = re.compile(r"(?<![\d.])\d+(?:\.\d+){2}(?!\.?\d)")
_NO_RELEASE = "0.0.0+not.the.pinned.version"


def sdk_readers(dialect: str) -> tuple[str, ...]:
    """Where an SDK reads the frames of a dialect: read from the drivers.

    A kind whose adapter hands the wire to a locked worker reads it there, with
    what that worker's lock pins; every other kind reads it in the test
    process. A dialect that two kinds share has both readers.
    """

    locks = {kind: f"backends/{worker}_worker/uv.lock" for worker, kind in WORKER_KINDS.items()}
    kinds = [kind for kind in sorted(drivers.load()) if drivers.get(kind).dialect == dialect]
    return tuple(sorted({locks.get(kind, TEST_PROCESS) for kind in kinds}))


@functools.cache
def _locked(lock: str) -> dict[str, tuple[str, ...]]:
    packages = tomllib.loads((REPOSITORY / lock).read_text(encoding="utf-8")).get("package", ())
    pinned: dict[str, list[str]] = {}
    for package in packages:
        pinned.setdefault(str(package["name"]), []).append(str(package.get("version")))
    return {name: tuple(sorted(versions)) for name, versions in pinned.items()}


def held_versions(reader: str, distribution: str) -> tuple[str, ...]:
    """The releases of a distribution a reader holds: installed, or locked."""

    if reader != TEST_PROCESS:
        return _locked(reader).get(distribution, ())
    try:
        return (importlib.metadata.version(distribution),)
    except importlib.metadata.PackageNotFoundError:
        return ()


def stated_version_problems(envelope: envelopes.Envelope) -> list[str]:
    """Why a release an envelope states is not the one that reads its dialect.

    A distribution is locked more than once here (the root environment and the
    workers each pin their own), so "some lock pins it" proves nothing. The
    readers of the dialect decide: one of them holds the distribution at the
    stated release.
    """

    readers = sdk_readers(envelope.dialect)
    problems: list[str] = []
    for distribution, version in envelope.sdk_versions:
        held = {reader: held_versions(reader, distribution) for reader in readers}
        if any(version in versions for versions in held.values()):
            continue
        holders = [f"{reader} holds {' and '.join(at)}" for reader, at in held.items() if at]
        problems.append(
            f"envelope {envelope.name} states {distribution} {version}; "
            + (", ".join(holders) or f"no reader of the dialect holds it: {list(readers)}")
        )
    return problems


def _words(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, (dict, list)):
        for item in node.values() if isinstance(node, dict) else node:
            yield from _words(item)


def stated_in_words(package: Path) -> dict[str, str]:
    """The prose of a package that can name a release, by file.

    An envelope's comments and its provenance outside ``sdk_version``, and the
    docstring of every module: where a protocol source is cited.
    """

    prose: dict[str, str] = {}
    for path in sorted((package / "envelopes").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        provenance = dict(yaml.safe_load(text)["provenance"])
        provenance.pop("sdk_version", None)
        comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
        prose[path.relative_to(package).as_posix()] = "\n".join([*comments, *_words(provenance)])
    for path in _python_files(package):
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        prose[path.relative_to(package).as_posix()] = docstring or ""
    return prose


def typed_release_problems(prose: Mapping[str, str]) -> list[str]:
    """A release typed where no test reads it: it goes stale with the next lock."""

    return [
        f"{where} types the release {found}; only the sdk_version of an envelope states one"
        for where, text in prose.items()
        for found in _TYPED_RELEASE.findall(text)
    ]


def test_a_stated_sdk_version_is_the_one_the_reader_of_its_dialect_holds(
    tmp_path: Path,
) -> None:
    stored = envelopes.all_envelopes()
    typed = [envelope for envelope in stored.values() if envelope.sdk_versions]
    truths = sorted({pair for envelope in typed for pair in envelope.sdk_versions})

    def restated(envelope: envelopes.Envelope, *pairs: tuple[str, str]) -> envelopes.Envelope:
        stated = ", ".join(f"{distribution} {version}" for distribution, version in pairs)
        return replace(envelope, provenance={**envelope.provenance, "sdk_version": stated})

    assert typed and truths
    assert [item for envelope in typed for item in stated_version_problems(envelope)] == []
    prose = stated_in_words(PACKAGE)
    assert len(prose) == len(stored) + len(_python_files(PACKAGE))
    assert typed_release_problems(prose) == []

    # A number the lock moved away from, in each of the stated places: the
    # problem names what the reader holds, which is what the tree states.
    for envelope in typed:
        for index, (distribution, version) in enumerate(envelope.sdk_versions):
            pairs = list(envelope.sdk_versions)
            pairs[index] = (distribution, _NO_RELEASE)
            (stale,) = stated_version_problems(restated(envelope, *pairs))
            assert f"{envelope.name} states {distribution} {_NO_RELEASE}; " in stale
            assert stale.endswith(f" holds {version}")
    # Pinned somewhere is not enough. What only another environment holds, a
    # distribution or another release of one, is not what reads this dialect.
    environments = {reader for item in stored.values() for reader in sdk_readers(item.dialect)}
    elsewhere = [
        (envelope, (distribution, other))
        for envelope in typed
        for distribution in sorted({name for name, _version in truths})
        for reader in sorted(environments - set(sdk_readers(envelope.dialect)))
        for other in held_versions(reader, distribution)
        if not any(
            other in held_versions(mine, distribution) for mine in sdk_readers(envelope.dialect)
        )
    ]
    assert elsewhere
    for envelope, (distribution, other) in elsewhere:
        (foreign,) = stated_version_problems(restated(envelope, (distribution, other)))
        assert f"{envelope.name} states {distribution} {other}; " in foreign
    # The form. A bare number names no distribution, and a type path without
    # the release it is read in is half of the evidence.
    first = typed[0]
    document = yaml.safe_load((envelopes.ROOT / f"{first.name}.yaml").read_text(encoding="utf-8"))
    typed_only = {key: value for key, value in first.provenance.items() if key != "sdk_version"}
    number = first.sdk_versions[0][1]
    for match, provenance in (
        ("'<distribution> <version>' pairs", {**typed_only, "sdk_version": number}),
        ("'<distribution> <version>' pairs", {**typed_only, "sdk_version": f"a {number},b 1"}),
        ("are stated together", typed_only),
    ):
        with pytest.raises(drivers.MatrixSetupError, match=match):
            envelopes.parse({**document, "provenance": provenance}, where="x")
    # A release in words is held by nothing, so it is refused wherever prose
    # cites a source: a comment or the provenance of an envelope, a module
    # docstring. What sdk_version states is held above; a body and a constant
    # are not prose. An address is not a release.
    lying = tmp_path / "package"
    (lying / "envelopes" / "dialect").mkdir(parents=True)
    words = {**first.provenance, "reason": f"Built for the SDK the worker pins ({number})."}
    (lying / "envelopes" / "dialect" / "frame.yaml").write_text(
        f"# Read against {number}.\n"
        + yaml.safe_dump({**document, "provenance": words, "body": {"version": number}}),
        encoding="utf-8",
    )
    (lying / "peer.py").write_text(
        f'"""Protocol sources, all in the pinned SDK ({number})."""\n\nRELEASE = "{number}"\n',
        encoding="utf-8",
    )
    told = f"types the release {number}; only the sdk_version of an envelope states one"
    assert typed_release_problems(stated_in_words(lying)) == [
        f"envelopes/dialect/frame.yaml {told}",
        f"envelopes/dialect/frame.yaml {told}",
        f"peer.py {told}",
    ]
    for form in (f"``sdk`` ({number}):", f"(``sdk.client``, {number}: the same)"):
        assert typed_release_problems({"x": form}) == [f"x {told}"]
    assert typed_release_problems({"x": "loopback (``127.0.0.0/8``, ``::1``)"}) == []


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


def test_the_watch_of_write_hosts_wraps_one_private_name_and_lists_it() -> None:
    # Beside the drivers and the required proofs, one module of the package
    # touches a private name of the product: the watch that holds the
    # processes a cell's writes run in. It imports that one name, lists it,
    # wraps the constructor, the start and the stop of the class and reads
    # nothing of a host but two public properties; the host of the toy
    # carries all of them too. The toy's wire and adapter stay free of it.
    tree = _tree("mutation.py")
    private = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level
        for alias in node.names
        if (node.module or "").startswith("crucible")
    ]
    product, own = mutation.host_classes()

    assert private == list(mutation.PRIVATE_SEAMS) == [mutation.PRODUCT_HOST]
    assert f"{product.__module__}.{product.__qualname__}" == mutation.PRODUCT_HOST
    assert own is toy.WriteHost
    for host_class in (product, own):
        deadline = inspect.signature(host_class.__init__).parameters["invocation_deadline"]
        assert deadline.kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("pid", "started_write_names"):
            assert isinstance(inspect.getattr_static(host_class, name), property), name
        for name in ("start", "stop"):
            assert inspect.isfunction(inspect.getattr_static(host_class, name)), name
    # No rule module reaches the watch: the contract holds its data types.
    for module in _RULE_MODULES:
        assert "mutation" not in {
            alias.name
            for node in ast.walk(_tree(module))
            if isinstance(node, ast.ImportFrom) and node.level
            for alias in node.names
        }, module


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
    # An admission input is one of those fields. The driver of a kind that
    # has one adds it to the shipped routes of every column, and on the view
    # of the file as shipped it adds none of them and everything else.
    listed = {name for fields in matrix_columns.ALLOWED_OVERLAY.values() for name in fields}
    assert set(matrix_columns.ADMISSION_OVERLAY) <= listed
    for kind in sorted(drivers.load()):
        allowed = set(matrix_columns.ALLOWED_OVERLAY.get(kind, ()))
        inputs = set(matrix_columns.admission_inputs(kind))
        assert inputs == allowed & set(matrix_columns.ADMISSION_OVERLAY)
        for column in (item for item in cells.all_columns() if item.kind == kind):
            overlay = drivers.get(kind).overlay(column, matrix_columns.SCRIPTED_ENV)
            assert inputs <= set(overlay) <= allowed
            as_shipped = drivers.get(kind).overlay(
                column.as_shipped(), matrix_columns.SCRIPTED_ENV
            )
            assert set(as_shipped) == set(overlay) - inputs


def test_difference_tables_are_total_and_cite_their_fallbacks() -> None:
    for scenario in SCENARIOS.values():
        assert declarations.totality_gaps(scenario.differences) == [], scenario.id
        for table in scenario.differences.values():
            for variant in table.values():
                assert variant.kind in {"BASE", "FALLBACK", "REFUSED"}
                assert variant.kind != "FALLBACK" or variant.citation
    # A table keyed by the state of a route extra is total over the three
    # states routing can state, named here; one table is keyed so.
    assert declarations.EXTRA_STATES == ("carried", "admitted_not_carried", "not_admitted")
    keyed = {
        scenario.id: table
        for scenario in SCENARIOS.values()
        for name, table in scenario.differences.items()
        if name in declarations.KEYING_EXTRAS
    }
    assert sorted(keyed) == ["S24"]
    for table in keyed.values():
        assert set(table) == set(declarations.EXTRA_STATES)
        for state in declarations.EXTRA_STATES:
            short = {name: variant for name, variant in table.items() if name != state}
            assert declarations.totality_gaps({declarations.SERVED_MODEL_EXTRA: short}) == [
                f"{declarations.SERVED_MODEL_EXTRA}={state} has no variant"
            ]
    # A governing key of every other kind is a declared field.
    for scenario in SCENARIOS.values():
        for name in scenario.governing:
            assert (name in declarations.KEYING_EXTRAS) != (name in declarations.FEATURE_FIELDS)


def test_every_scenario_passes_the_complete_base_policy() -> None:
    # The scenarios that pass another policy, each with the one field it
    # changes and why. The scenario of the stable writer catalogue is
    # thirteen provider requests long; a backend that starts a worker for
    # every request needs most of the base policy's minute for them, and
    # three times as long where the test shares one hardware thread with
    # another busy process. The scenario is not about a deadline, so its
    # deadline is several times the longer figure. The scenario of the
    # malformed tool arguments is four requests long: a third of that
    # minute for such a backend, and more than the minute where two other
    # busy processes share the thread. It is not about a deadline either
    # and passes the same one. The second form of the turn limit is the
    # limit under the policy that asks for no error.
    exceptions = {
        "S45": ("timeout_seconds", 600),
        "S05": ("timeout_seconds", 600),
        "S10b": ("on_max_turns", "continue_with_warning"),
    }
    for scenario in SCENARIOS.values():
        policy = scenario.policy
        expected = {
            "retry_budget": 1,
            "timeout_seconds": 60,
            "on_cancellation": "convert_to_backend_error",
            "on_max_turns": "raise",
        }
        if scenario.id in exceptions:
            name, value = exceptions[scenario.id]
            assert expected[name] != value
            expected[name] = value
        assert {name: getattr(policy, name) for name in expected} == expected, scenario.id
    assert set(exceptions) <= set(SCENARIOS)


class _CollectedItem:
    """What the collection hook of the package reads and writes of a test item."""

    def __init__(self, module: str, name: str) -> None:
        self.path = PACKAGE / f"{module}.py"
        self.name = name
        self.nodeid = f"{self.path.name}::{name}"
        self.limits: list[float] = []
        # The fixtures of the package are in force for it, as for every item
        # pytest collects under the directory of the package.
        self.fixturenames = matrix_conftest._IN_FORCE

    def get_closest_marker(self, _name: str) -> None:
        return None

    def add_marker(self, marker: Any) -> None:
        assert marker.name == "timeout" and not marker.kwargs
        (seconds,) = marker.args
        self.limits.append(seconds)


def test_the_item_of_a_cell_with_a_long_deadline_has_a_limit_above_it(
    request: pytest.FixtureRequest,
) -> None:
    # The suite ends a test after a fixed time. A cell whose scenario passes
    # an invocation deadline that is not below that time could never be
    # ended by its deadline: the suite would end the item first, a plain
    # failure that the verifier refuses for a cell the ledger expects to
    # fail. The item of such a cell gets a limit of its own at collection,
    # the deadline and a margin; today those are the cells of the two
    # scenarios that pass another deadline than the base policy's.
    options = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]
    configured = float(options["ini_options"]["timeout"])
    long = {
        name: scenario.policy.timeout_seconds
        for name, scenario in SCENARIOS.items()
        if (scenario.policy.timeout_seconds or 0) >= configured
    }

    def limits(limit: float | None, hook: bool = False) -> dict[str, list[float]]:
        items = [_CollectedItem(item.module, item.name) for item in cells.expected_items()]
        items.append(_CollectedItem(cells.STATIC_MODULE, "test_cell[toy/S45/none/raw]"))
        if hook:
            matrix_conftest.pytest_collection_modifyitems(request.config, items)
        else:
            cells.limit_long_cells(items, PACKAGE, limit)
        return {item.name: item.limits for item in items if item.limits}

    expected = {
        f"{cells.BEHAVIOUR_TEST}[{cell.id}]": [long[cell.scenario] + cells.LIMIT_MARGIN_SECONDS]
        for cell in cells.behaviour_cells()
        if cell.scenario in long
    }
    assert long == {"S05": 600, "S45": 600} and cells.LIMIT_MARGIN_SECONDS >= configured
    # The catalogue scenario has a cell on every kind, the scenario of the
    # malformed arguments on the kinds whose wire carries the step, the
    # open-weight ones; both have one on every toy column.
    toys = len(cells.toy_columns())
    carried = {name: sum(1 for item in expected if f"/{name}/" in item) for name in long}
    assert carried == {
        "S05": len(matrix_columns.OPEN_WEIGHT_KINDS) + toys,
        "S45": len(REAL_KINDS) + toys,
    }
    assert len(expected) == sum(carried.values())
    assert limits(configured) == expected
    # A deadline below the limit can end its call, and without a limit there
    # is nothing to exceed: no item is touched. Nor is an item that judges
    # no cell, or one outside the package.
    assert limits(600.5) == limits(None) == limits(0.0) == {}
    outside = _CollectedItem("test_matrix_claude", "test_cell[claude/S45/primary/raw]")
    outside.path = REPOSITORY / "tests" / outside.path.name
    cells.limit_long_cells([outside], PACKAGE, configured)
    assert outside.limits == []
    # Under a suite limit of a minute the base policy's deadline is not
    # below it either: then every cell is such a cell, each with its own
    # deadline and the margin.
    minute = limits(60.0)
    assert set(minute) == {f"{cells.BEHAVIOUR_TEST}[{cell.id}]" for cell in cells.behaviour_cells()}
    assert {name: value for name, value in minute.items() if name in expected} == expected
    assert {tuple(value) for name, value in minute.items() if name not in expected} == {
        (60 + cells.LIMIT_MARGIN_SECONDS,)
    }
    # The hook of the package applies it with the limit this run is under.
    assert limits(None, hook=True) == limits(cells.suite_limit(request.config))
    # The limit is read as the timeout plug-in reads it: the command line,
    # then the environment, then the configuration file; zero is no limit.

    def configured_with(option: Any, ini: Any) -> SimpleNamespace:
        def getini(_name: str) -> Any:
            if isinstance(ini, Exception):
                raise ini
            return ini

        return SimpleNamespace(getoption=lambda _name, _default=None: option, getini=getini)

    with pytest.MonkeyPatch.context() as patched:
        patched.delenv("PYTEST_TIMEOUT", raising=False)
        assert cells.suite_limit(configured_with(300, "120")) == 300.0
        assert cells.suite_limit(configured_with(None, "120")) == 120.0
        assert cells.suite_limit(configured_with(0, "120")) is None
        assert cells.suite_limit(configured_with(None, "")) is None
        assert cells.suite_limit(configured_with(None, ValueError("unknown"))) is None
        patched.setenv("PYTEST_TIMEOUT", "45")
        assert cells.suite_limit(configured_with(None, "120")) == 45.0


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
