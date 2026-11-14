# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Build-side tool handlers — ``build_project`` + ``create_rust_crate``.

``build_project`` invokes the configured build runner (resolved via
``ctx.runners``) under a cross-process lock, then returns the runner's status
and bounded output. Results begin with ``BUILD_STATUS: <status>`` and
``BUILD_ROOT: <path>`` for stable machine and model parsing.

``create_rust_crate`` scaffolds a new Rust crate at
``crates/<name>/Cargo.toml`` + ``src/{lib,main}.rs`` and adds the
crate to the workspace ``Cargo.toml`` members list. An optional
``prefix`` lets a workspace request its house-style crate naming
(e.g. ``prefix="fabric-"`` yields ``fabric-<name>``). Uses stdlib
``tomllib`` for reading and the third-party ``toml`` package for
writing (falls back to regex if ``toml`` isn't installed).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from crucible.common.utils import elide_middle

from .._buildlock import build_lock
from .._builder import ToolSetBuilder
from .._context import ToolContext
from .._paths import resolve_for_write, resolve_within_repo

logger = logging.getLogger(__name__)

# Stable prefixes consumed by downstream verifiers.
_BUILD_STATUS_PREFIX = "BUILD_STATUS:"
_MAX_OUTPUT_CHARS = 15_000


def _resolve_worktree_root(ctx: ToolContext, project: str) -> Path:
    """Resolve the worktree root for *project*.

    Worktree-aware via ``ctx.path_resolver``; otherwise falls back to
    ``ctx.repo_root``.
    """
    resolver = getattr(ctx, "path_resolver", None)
    if resolver is not None:
        try:
            wt = resolver.resolve_worktree_root(project)
            if wt is not None:
                return Path(wt)
        except (AttributeError, ValueError, OSError):
            pass
    return ctx.repo_root


def _make_runner_context(
    ctx: ToolContext,
    project: str = ".",
    files_to_modify: list[str] | None = None,
):
    """Construct a ``crucible.common.runner_protocol.RunnerContext``."""
    from crucible.common.runner_protocol import RunnerContext

    worktree_root = _resolve_worktree_root(ctx, project)
    return RunnerContext(
        workspace_root=ctx.workspace_root,
        repo_root=ctx.repo_root,
        project=project,
        project_root=ctx.repo_root,
        worktree=worktree_root if worktree_root != ctx.repo_root else None,
        cwd=worktree_root,
        build_artifact_path="",
        files_to_modify=tuple(files_to_modify) if files_to_modify else (),
    )


builder = ToolSetBuilder("build")


@builder.tool(
    description=(
        "Build a project in the worktree using the configured build_command. "
        "Returns build output (truncated to 15K chars). The first line of "
        "output is 'BUILD_ROOT: <path>' for callers that need the resolved "
        "root. The build_command is defined in the roadmap spec or "
        ".crucible/config.yaml. Returns 'SKIP' if no build_command is set."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": (
                    "Project to build (default: '.'). Use '.' for standalone "
                    "projects or when build_command is configured in the spec."
                ),
                "default": ".",
            },
            "build_args": {
                "type": "string",
                "description": "Additional build flags",
                "default": "",
            },
            "package": {
                "type": "string",
                "description": (
                    "Rust crate name to scope cargo test to (e.g. 'core' "
                    "for fabric-core). Empty runs all workspace tests."
                ),
                "default": "",
            },
        },
        "required": [],
    },
    write=True,
)
def build_project(
    ctx: ToolContext,
    *,
    project: str = ".",
    build_args: str = "",
    package: str = "",
    **_: Any,
) -> str:
    """Invoke ``ctx.runners``'s build runner under a cross-process flock."""
    if ctx.runners is None:
        return f"{_BUILD_STATUS_PREFIX} skipped\nBUILD_ROOT: {ctx.repo_root}\nNo runners configured"

    runner_ctx = _make_runner_context(ctx, project)
    runner = ctx.runners.resolve_build_runner(runner_ctx)
    if runner is None:
        return (
            f"{_BUILD_STATUS_PREFIX} skipped\n"
            f"BUILD_ROOT: {_resolve_worktree_root(ctx, project)}\n"
            f"No build_runner configured"
        )

    worktree_root = _resolve_worktree_root(ctx, project)
    with build_lock(worktree_root):
        result = runner.build()

    header = (
        f"{_BUILD_STATUS_PREFIX} {result.status}\n"
        f"BUILD_ROOT: {worktree_root}\n"
    )
    # Compile errors in a full build are often in the middle of the output
    # (after early
    # "what's being built" lines, before the linker / postbuild tail),
    # so pure tail truncation hides them. With visible elision the
    # agent can pivot to `head -c` / `tail -c` / `sed -n` on the next
    # turn to retrieve the gap by byte offset.
    body_budget = _MAX_OUTPUT_CHARS - len(header)
    body = elide_middle(
        result.output,
        head_chars=body_budget // 3,
        tail_chars=(body_budget * 2) // 3,
    )
    return header + body


@builder.tool(
    description=(
        "Create a new Rust crate in the Cargo workspace. Sets up "
        "Cargo.toml, src/lib.rs, and adds the crate to workspace members. "
        "Optional 'prefix' applies a workspace-house naming convention "
        "(e.g. prefix='fabric-' yields a package named 'fabric-<name>')."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Crate directory name under crates/ (e.g. 'core')",
            },
            "prefix": {
                "type": "string",
                "description": (
                    "Optional workspace-house prefix prepended to the "
                    "package name (NOT the directory). Empty by default."
                ),
                "default": "",
            },
            "crate_type": {
                "type": "string",
                "description": "lib or bin (default: lib)",
                "default": "lib",
            },
        },
        "required": ["name"],
    },
    write=True,
)
def create_rust_crate(
    ctx: ToolContext,
    *,
    name: str,
    prefix: str = "",
    crate_type: str = "lib",
    **_: Any,
) -> str:
    """Scaffold a new crate at ``crates/<name>/`` + register it in workspace.toml."""
    def _resolve_write(relative: str) -> Path:
        wp = resolve_for_write(ctx, relative)
        if wp is None:
            raise ValueError(f"Cannot resolve path: {relative}")
        return wp

    package_name = f"{prefix}{name}"

    crate_dir = _resolve_write(f"crates/{name}/Cargo.toml").parent
    crate_dir.mkdir(parents=True, exist_ok=True)

    crate_toml = crate_dir / "Cargo.toml"
    crate_toml.write_text(
        f'[package]\nname = "{package_name}"\nversion = "0.1.0"\n'
        f'edition = "2021"\n\n[dependencies]\n',
        encoding="utf-8",
    )

    src_dir = crate_dir / "src"
    src_dir.mkdir(exist_ok=True)
    entry_file = "main.rs" if crate_type == "bin" else "lib.rs"
    (src_dir / entry_file).write_text(
        f"//! {package_name} crate\n", encoding="utf-8",
    )

    workspace_toml_path = _resolve_write("Cargo.toml")
    if workspace_toml_path.exists():
        raw = workspace_toml_path.read_text(encoding="utf-8")
        entry = f"crates/{name}"

        try:
            import tomllib
            data = tomllib.loads(raw)
        except ImportError:  # pragma: no cover — 3.11+ guaranteed
            import toml as _toml
            data = _toml.loads(raw)

        members = data.get("workspace", {}).get("members", [])
        if entry not in members:
            try:
                import toml as _toml
                members.append(entry)
                data.setdefault("workspace", {})["members"] = members
                workspace_toml_path.write_text(_toml.dumps(data), encoding="utf-8")
            except ImportError:
                if entry not in raw:
                    raw = raw.replace(
                        "members = [",
                        f'members = [\n    "{entry}",',
                    )
                    workspace_toml_path.write_text(raw, encoding="utf-8")

    return f"Created crate {package_name} at crates/{name}/"


def build_build_tools(context: ToolContext):
    """Build the build ToolSet bound to *context*."""
    return builder.build(context=context)


__all__ = [
    "build_build_tools",
    "builder",
    "build_project",
    "create_rust_crate",
]
