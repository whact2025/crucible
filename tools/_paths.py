# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Shared path-resolution helpers for tool handlers.

Two layered resolvers:

- ``resolve_within_repo(ctx, relative)`` — strict: resolves a repository-
  relative path against ``ctx.repo_root`` and rejects ``..`` escapes. In a
  container-backed context, the exact trusted ``ctx.container_workdir`` prefix
  is also accepted as a lexical alias for the repository root. Returns
  ``None`` on rejection or invalid path. Used by source/navigation handlers
  and apply_patch.

- ``resolve_for_write(ctx, relative)`` — write-side: if
  ``ctx.path_resolver`` is set (worktree-aware), delegates to it;
  otherwise falls back to ``resolve_within_repo``. The worktree
  resolver creates a per-project worktree on first write.

- ``resolve_for_read(ctx, relative)`` — read-side: same as
  write-side but uses ``resolve_read`` on the path_resolver.

Leaf module: stdlib only.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

from ._context import ToolContext

logger = logging.getLogger(__name__)

_VCS_METADATA_COMPONENTS = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".bzr",
    "_darcs",
})


def _normalized_relative(relative: str) -> str | None:
    """Return one safe POSIX repository-relative path.

    This is a lexical check by design.  Files need not exist yet, and the
    path-resolver layer may map the same declared path into an isolated
    worktree.  Filesystem containment is checked separately after mapping.
    VCS administrative directories are categorically unavailable to model
    tools, including nested repositories and case variants.
    """

    if not isinstance(relative, str) or not relative or "\x00" in relative:
        return None
    rendered = relative.replace("\\", "/")
    candidate = PurePosixPath(rendered)
    if candidate.is_absolute() or candidate.as_posix() in {"", "."}:
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    if any(
        part.casefold() in _VCS_METADATA_COMPONENTS
        for part in candidate.parts
    ):
        return None
    return candidate.as_posix()


def _container_workdir_relative(
    ctx: ToolContext,
    absolute: str,
) -> str | None:
    """Map one trusted container-project absolute path to repo-relative form.

    Model roles in container-backed workflows see the repository mounted at a
    workflow-owned path such as ``ctx.container_workdir``. The source tools
    execute on the host-side bind source, so that exact container prefix is a
    safe alias for ``ctx.repo_root``. No other absolute path is meaningful to
    the host tool layer.

    This helper is deliberately lexical. It grants no authority itself: the
    normalized result still passes through write scopes and the existing
    worktree/symlink/hardlink containment checks. Traversal is retained by
    ``PurePosixPath`` and rejected by ``_normalized_relative``.
    """

    if (
        not isinstance(absolute, str)
        or not absolute
        or "\x00" in absolute
        or "\\" in absolute
        or not getattr(ctx, "container_id", None)
    ):
        return None
    workdir = getattr(ctx, "container_workdir", None)
    if (
        not isinstance(workdir, str)
        or not workdir
        or "\x00" in workdir
        or "\\" in workdir
    ):
        return None
    container_root = PurePosixPath(workdir)
    if (
        not container_root.is_absolute()
        or any(part in {"", ".", ".."} for part in container_root.parts[1:])
    ):
        return None

    candidate = PurePosixPath(absolute)
    if not candidate.is_absolute():
        return None
    try:
        relative = candidate.relative_to(container_root)
    except ValueError:
        return None
    return _normalized_relative(relative.as_posix())


def _write_is_authorized(ctx: ToolContext, relative: str) -> str | None:
    normalized = _context_relative(ctx, relative)
    if normalized is None:
        return None
    roots: list[str] = []
    for rendered in getattr(ctx, "write_scope", ()):
        root = _context_relative(ctx, rendered)
        if root is None:
            # A malformed workflow-supplied scope is a wiring error.  Deny the
            # request instead of silently dropping the bad root and widening
            # authority through another entry.
            logger.error("invalid ToolContext write_scope entry: %r", rendered)
            return None
        roots.append(root)
    if not roots:
        return None
    if not any(
        normalized == root or normalized.startswith(root.rstrip("/") + "/")
        for root in roots
    ):
        return None
    exact_paths: list[str] = []
    for rendered in getattr(ctx, "write_exact_scope", ()):
        exact = _context_relative(ctx, rendered)
        if exact is None:
            logger.error(
                "invalid ToolContext write_exact_scope entry: %r",
                rendered,
            )
            return None
        exact_paths.append(exact)
    if exact_paths and normalized not in exact_paths:
        return None
    denied_roots: list[str] = []
    for rendered in getattr(ctx, "write_deny_scope", ()):
        denied = _context_relative(ctx, rendered)
        if denied is None:
            logger.error(
                "invalid ToolContext write_deny_scope entry: %r",
                rendered,
            )
            return None
        denied_roots.append(denied)
    if any(
        normalized == denied
        or normalized.startswith(denied.rstrip("/") + "/")
        for denied in denied_roots
    ):
        return None
    return normalized


def _context_relative(ctx: ToolContext, relative: str) -> str | None:
    """Normalize trusted root aliases and a redundant workspace prefix."""

    if not isinstance(relative, str) or not relative or "\x00" in relative:
        return None
    candidate = PurePosixPath(relative)
    normalized = (
        _container_workdir_relative(ctx, relative)
        if candidate.is_absolute()
        else _normalized_relative(relative)
    )
    if normalized is None:
        return None
    candidate = PurePosixPath(normalized)
    parts = candidate.parts
    repo_root = ctx.repo_root.resolve()
    if parts and parts[0] == repo_root.name and len(parts) > 1:
        unstripped = repo_root.joinpath(*parts)
        if not unstripped.exists() and not unstripped.parent.is_dir():
            candidate = PurePosixPath(*parts[1:])
    return candidate.as_posix()


def resolve_within_repo(ctx: ToolContext, relative: str) -> Path | None:
    """Resolve ``relative`` against ``ctx.repo_root``. Reject escapes."""
    normalized = _context_relative(ctx, relative)
    if normalized is None:
        return None
    root = ctx.repo_root.resolve()
    try:
        resolved = (root / normalized).resolve()
    except (ValueError, OSError):
        return None
    try:
        resolved_relative = resolved.relative_to(root)
    except ValueError:
        return None
    if any(
        part.casefold() in _VCS_METADATA_COMPONENTS
        for part in resolved_relative.parts
    ):
        return None
    return resolved


def _mapped_write_is_safe(
    mapped: Path | None,
    *,
    root: Path,
    normalized: str,
) -> Path | None:
    """Reject resolver aliases and symlinks that widen declared authority."""

    if mapped is None:
        return None
    try:
        resolved_root = root.resolve()
        resolved = Path(mapped).resolve()
        relative = resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if relative.as_posix() != normalized:
        # A symlinked parent/final component can otherwise turn authority for
        # ``declared/link`` into authority over an unrelated repository file.
        return None
    if any(
        part.casefold() in _VCS_METADATA_COMPONENTS
        for part in relative.parts
    ):
        return None
    try:
        if resolved.is_file() and resolved.stat().st_nlink > 1:
            # Writing one declared hardlink mutates every alias to the same
            # inode, including aliases outside the directive surface.
            return None
    except OSError:
        return None
    return resolved


def resolve_for_read(ctx: ToolContext, relative: str) -> Path | None:
    """Resolve a path for reading.

    Delegates to ``ctx.path_resolver`` (worktree-aware) if present;
    otherwise falls back to strict in-repo resolution.
    """
    normalized = _context_relative(ctx, relative)
    if normalized is None:
        return None
    resolver = getattr(ctx, "path_resolver", None)
    if resolver is not None:
        try:
            return resolver.resolve_read(normalized)
        except ValueError as exc:
            logger.warning("resolve_read(%s) failed: %s", relative, exc)
            return None
        except OSError:
            return None
    return resolve_within_repo(ctx, normalized)


def resolve_for_write(ctx: ToolContext, relative: str) -> Path | None:
    """Resolve a path for writing.

    Worktree-aware resolvers create a per-project worktree on first
    write. Without a resolver, falls back to strict in-repo resolution.
    """
    normalized = _write_is_authorized(ctx, relative)
    if normalized is None:
        logger.warning("rejected out-of-scope write path: %r", relative)
        return None
    resolver = getattr(ctx, "path_resolver", None)
    if resolver is not None:
        try:
            mapped = resolver.resolve_write(normalized)
            mapped_root = resolver.resolve_worktree_root(".")
            if mapped_root is None:
                return None
            return _mapped_write_is_safe(
                mapped,
                root=Path(mapped_root),
                normalized=normalized,
            )
        except (ValueError, OSError) as exc:
            logger.warning("resolve_write(%s) failed: %s", relative, exc)
            return None
    return _mapped_write_is_safe(
        resolve_within_repo(ctx, normalized),
        root=ctx.repo_root,
        normalized=normalized,
    )


def resolve_build_root(ctx: ToolContext, project: str = ".") -> Path:
    """Resolve the build root directory for a project.

    Worktree-aware via ``ctx.path_resolver.resolve_worktree_root``; else
    falls back to ``ctx.repo_root / project`` if that exists, else
    ``ctx.repo_root``.
    """
    resolver = getattr(ctx, "path_resolver", None)
    if resolver is not None:
        try:
            wt = resolver.resolve_worktree_root(project)
            if wt is not None and Path(wt) != ctx.repo_root:
                return Path(wt)
        except (AttributeError, ValueError, OSError):
            pass
    if project in (".", ""):
        return ctx.repo_root
    candidate = ctx.repo_root / project
    return candidate if candidate.is_dir() else ctx.repo_root


__all__ = [
    "resolve_within_repo",
    "resolve_for_read",
    "resolve_for_write",
    "resolve_build_root",
]
