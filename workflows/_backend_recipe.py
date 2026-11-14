# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministic source recipe for every role-routed backend implementation.

Benchmark checkpoints use this inventory to prevent a result produced by one
adapter, transport helper, or isolated worker recipe from being reused after
those bytes change.  Runtime credentials and installed environment state are
deliberately excluded; adapters attest those at their own dispatch boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

_SHARED_RUNTIME_PATHS = (
    "common/authority_io.py",
    "common/process_settlement.py",
    "tools/__init__.py",
    "tools/_dispatch_budget.py",
    "tools/_isolated_mutation_host.py",
    "tools/_mcp_bridge.py",
    "tools/_mcp_http_server.py",
    "workflows/_backends.py",
    "workflows/_backend_recipe.py",
    "workflows/_process_supervisor_contract.py",
    "workflows/_process_supervisor.py",
)

_ISOLATED_WORKER_PROJECTS = (
    "codex_worker",
    "deepseek_worker",
    "kimi_worker",
)

_WORKER_RECIPE_ROOT_PATHS = (
    ".python-version",
    "pyproject.toml",
    "uv.lock",
)


def backend_recipe_identity(
    crucible_root: Path | str,
    *,
    digest_file: Callable[[Path], Any],
) -> dict[str, Any]:
    """Return exact source and lock identities for every registered adapter.

    The top-level backend package contains the registration/factory, routing,
    feature admission, capability, pricing, evidence, limit, and transport
    modules for Claude, Codex (legacy and official), DeepSeek, Kimi, and the
    OpenAI-compatible implementation shared by ``openai_compat`` and ``local``.
    Each project-locked worker additionally binds its Python version recipe,
    project metadata, lock, and complete importable source tree.
    """

    root = Path(crucible_root)
    backend_root = root / "backends"
    backend_modules = tuple(sorted(backend_root.glob("*.py")))
    if not backend_modules:
        raise FileNotFoundError("backend source recipe is empty")

    worker_projects: dict[str, dict[str, Any]] = {}
    for project_name in _ISOLATED_WORKER_PROJECTS:
        project_root = backend_root / project_name
        relative_paths = [Path(name) for name in _WORKER_RECIPE_ROOT_PATHS]
        source_root = project_root / "src"
        source_paths = tuple(
            path.relative_to(project_root)
            for path in sorted(source_root.rglob("*.py"))
            if path.is_file()
        )
        if not source_paths:
            raise FileNotFoundError(
                f"isolated backend worker source recipe is empty: {project_name}"
            )
        relative_paths.extend(source_paths)
        worker_projects[project_name] = {
            relative.as_posix(): digest_file(project_root / relative)
            for relative in relative_paths
        }

    return {
        "backend_modules": {path.name: digest_file(path) for path in backend_modules},
        "shared_runtime": {
            relative: digest_file(root / relative) for relative in _SHARED_RUNTIME_PATHS
        },
        "isolated_worker_projects": worker_projects,
    }


__all__ = ["backend_recipe_identity"]
