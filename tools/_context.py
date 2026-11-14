# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``ToolContext`` — shared state passed as the first positional arg
to every tool handler.

Why a context object instead of free functions:
- Most handlers need ``workspace_root`` (where to look up files) plus
  workflow-owned runner and mutation settings. A context object lets
  each domain take what it needs
  without forcing all handlers to share the same signature.
- It keeps the assay.ToolSet protocol shape (``dispatch(name, args)``
  takes args only) — the context is captured by the ToolSet at build
  time, not passed through dispatch.

Fields are optional with sensible defaults so a domain that doesn't
need (e.g.) a runner config can construct a ToolContext with just
``workspace_root``.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PathResolver(Protocol):
    """Protocol for worktree-aware path resolution.

    ``tools/_paths.py``'s ``resolve_for_read`` / ``resolve_for_write``
    / ``resolve_build_root`` duck-typed against these three methods;
    declaring the Protocol here lets type checkers catch missing
    methods on future custom resolvers (e.g. test fixtures) before
    they silently fall through to ``None`` at call time.

    Reference implementation: ``workflows/_worktree.WorktreeResolver``.
    """

    def resolve_read(self, relative: str) -> Path | None:
        """Resolve ``relative`` for reading. Return None on path-
        traversal escape or unresolvable input."""

    def resolve_write(self, relative: str) -> Path | None:
        """Resolve ``relative`` for writing, creating any backing
        worktree lazily. Return None on path-traversal escape."""

    def resolve_worktree_root(self, project: str = ".") -> Path | None:
        """Return the worktree root path (or a project-scoped subpath
        within it). Returns None when no worktree exists yet."""


@dataclass
class ToolContext:
    """Container for per-call shared state.

    Scoped to one ToolSet instance and its workflow-owned mutation authority.
    """

    # Required: where to resolve paths against.
    workspace_root: Path
    repo_root: Path

    # Optional domain-specific resources. Each domain reads only what
    # it cares about; unused fields stay None / default.
    runners: Any = None  # RunnerConfig | None — typed lazily to avoid cycle
    context_budget: int | None = None  # Tool-result allowance in UTF-8 bytes.
    budget_tracker: Any = None
    workspace_relative_paths: bool = True
    # Exact model-authorized write surface, expressed as repository-relative
    # files or directory roots.  The empty tuple is deliberately fail-closed:
    # attaching a write-capable tool profile without a workflow directive does
    # not grant repository-wide mutation authority.  Workflows must set this
    # immediately before each model invocation (and may change it between
    # serial invocations because ToolContext is invocation-local mutable
    # state).
    write_scope: tuple[str, ...] = ()
    # Optional exact-file restriction applied in addition to ``write_scope``.
    # ``write_scope`` entries are roots by design and therefore authorize
    # descendants; focused recovery flows use this field when authority must
    # be limited to named files themselves.  Empty means no extra exactness
    # constraint (the ordinary invocation behavior).
    write_exact_scope: tuple[str, ...] = ()
    # Exact repository-relative files or directory roots that remain denied
    # even when an enclosing write_scope root would otherwise authorize them.
    # Breach uses this to keep generated attack tests immutable to the
    # production-only Defender.
    write_deny_scope: tuple[str, ...] = ()
    # Durable owner for crash-recoverable exact-root deletion. These fields
    # are descriptive identity only: the mutation host resolves them back to
    # the already process-live journal object and re-proves its captured
    # directory identity before it can arm a delete receipt. Ordinary profile
    # construction may leave both unset; delete dispatch then fails closed.
    mutation_journal_dir: Path | None = None
    mutation_journal_transaction_id: str | None = None
    # Optional worktree-aware path resolver. When set, write tools
    # delegate to ``path_resolver.resolve_write(...)`` / ``resolve_read``;
    # otherwise paths resolve strictly within ``repo_root`` via
    # ``tools._paths.resolve_within_repo``.
    path_resolver: Any = None

    # Optional per-task container identity. Benchmark workflows bind it to
    # their isolated authoring environment; deterministic build and test
    # execution remains workflow-owned. DeepSWE and SWE-bench populate it.
    container_id: str | None = None
    # Project root inside that container. ``docker exec --workdir`` uses
    # this for every command so agents do not need product-specific ``cd``
    # prefixes (/app for DeepSWE, /testbed for SWE-bench).
    container_workdir: str = "/app"


__all__ = ["ToolContext"]
