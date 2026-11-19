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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Iterator, Sequence

import pytest
import yaml  # type: ignore[import-untyped]

import assay
from assay import BackendFeatureSet
from crucible.backends import BACKEND_NAMES

from . import cells, contract, declarations, drivers, ledger, runner
from . import columns as matrix_columns
from .columns import REAL_KINDS, TOY_KIND, WORKER_KINDS, Column
from .contract import Deviation
from .drivers import toy
from .scenarios import PACK_A, SCENARIOS
from .script import ProviderScript

LEDGER = ledger.load()
PACKAGE = Path(__file__).resolve().parent
TESTS = PACKAGE.parent
REPOSITORY = TESTS.parent
CI_WORKFLOW = REPOSITORY / ".github" / "workflows" / "ci.yml"
TOY_COLUMNS = cells.toy_columns()
# Static cells that were judged in this process.
_JUDGED: set[str] = set()


def _static_id(kind: str, scenario: str, role: str = ledger.NO_ROLE) -> str:
    return ledger.cell_id(kind, scenario, role, ledger.STATIC_LENS)


def _verdict(cell: str, problems: Sequence[Deviation]) -> None:
    """Judge a static cell: H01 and H02 by the ledger, the others must be clean."""

    assert cell in cells.static_cell_ids(), f"{cell} is not in the inventory"
    _JUDGED.add(cell)
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
        worker_skip_markers(TESTS),
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
    # Part 1: exactly eight expected failures, each on a cell its record lists.
    assert sorted(entry.cell for entry in LEDGER.entries) == sorted(
        [
            _static_id(WORKER_KINDS["kimi"], "H01"),
            *(_static_id(kind, "H02") for kind in REAL_KINDS),
        ]
    )
    assert not LEDGER.column_wide and not LEDGER.differences


def worker_environment_problems(ci_text: str, tests_root: Path) -> list[Deviation]:
    """H04: a worker environment may be installed in CI only behind the guard."""

    steps = yaml.safe_load(ci_text)["jobs"]["tests"]["steps"]
    installed = sorted(
        {
            match.group(1)
            for step in steps
            for match in re.finditer(
                r"--project\s+backends/(\w+)_worker", str(step.get("run", ""))
            )
        }
    )
    problems: list[Deviation] = []
    suite_conftest = (tests_root / "conftest.py").read_text(encoding="utf-8")
    peers = tests_root / "backend_matrix" / "peers"
    guard = {
        "the worker harness": (peers / "worker_harness.py").is_file(),
        "the launch interposer": (peers / "launch.py").is_file(),
        "the suite-wide interposer fixture": "backend_matrix.peers" in suite_conftest,
    }
    for worker in installed:
        missing = [name for name, present in guard.items() if not present]
        handed = [
            path.relative_to(tests_root).as_posix()
            for path in _python_files(tests_root)
            if PACKAGE not in path.parents
            and worker in path.name
            and re.search(
                r"subprocess\.\w+\(\s*\[[^\]]*WORKER_PYTHON",
                path.read_text(encoding="utf-8"),
            )
        ]
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
    problems = worker_environment_problems(
        CI_WORKFLOW.read_text(encoding="utf-8"), TESTS
    )

    _verdict(_static_id(ledger.ALL_KINDS, "H04"), problems)


def test_h04_fails_when_ci_installs_a_worker_environment_before_the_guard() -> None:
    ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    ci["jobs"]["tests"]["steps"].append(
        {"run": "uv sync --frozen --project backends/kimi_worker"}
    )

    problems = worker_environment_problems(yaml.safe_dump(ci), TESTS)

    assert [item.code for item in problems] == ["static:worker_env_unguarded"]


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
