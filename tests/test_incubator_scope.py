# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Guard the executable and packaged boundaries of the incubator."""

from __future__ import annotations

import ast
from pathlib import Path

from crucible.workflows.__main__ import _discover_workflows


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    "architect", "backend_config_bind_deepseek", "breach", "deepswe", "goal",
    "handoff", "implement", "journal_recover", "plan", "reconcile", "review",
    "swebench", "tier", "validate",
}
RUNTIME_TREES = (
    "backends", "common", "products", "rubrics", "rules", "scripts", "tools",
    "workflows",
)
REMOVED_MODULES = (
    "crucible.analysis", "crucible.bench", "crucible.dashboard",
    "crucible.products.assay", "crucible.products.isaac_sim",
    "crucible.products.isaac_lab", "crucible.products.kit",
    "crucible.products.spec_program", "crucible.tools.frontier_eval",
)


def test_public_commands_are_solution_building_and_two_benchmark_suites() -> None:
    assert set(_discover_workflows()) == WORKFLOWS


def test_product_adapters_are_swebench_and_deepswe() -> None:
    adapters = {
        path.name
        for path in (ROOT / "products").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert adapters == {"swebench", "deepswe"}


def test_retained_runtime_imports_resolve_without_removed_packages() -> None:
    failures: list[str] = []
    paths = [ROOT / "__init__.py", ROOT / "__main__.py", ROOT / "orchestration.py"]
    for tree in RUNTIME_TREES:
        paths.extend(
            path for path in (ROOT / tree).rglob("*.py")
            if not {".venv", "__pycache__", "tests"}.intersection(path.parts)
        )
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        package = ["crucible", *relative.parent.parts]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[:len(package) - node.level + 1]
                    modules.append(".".join([*prefix, *(node.module or "").split(".")]).rstrip("."))
                elif node.module:
                    modules.append(node.module)
            elif isinstance(node, ast.Call) and node.args:
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    modules.append(node.args[0].value)
            for module in modules:
                if not module.startswith("crucible."):
                    continue
                location = ROOT.joinpath(*module.split(".")[1:])
                if (
                    any(module == removed or module.startswith(removed + ".") for removed in REMOVED_MODULES)
                    or not (location.with_suffix(".py").is_file() or (location / "__init__.py").is_file())
                ):
                    failures.append(f"{relative.as_posix()}:{node.lineno}: {module}")
    assert not failures, "unsupported runtime dependencies:\n" + "\n".join(failures)
