# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Locate the solution workspace root and the embedded ``crucible/`` checkout."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_WORKSPACE = "CRUCIBLE_WORKSPACE_ROOT"


def _validate_workspace(candidate: Path) -> Path:
    """Ensure *candidate* contains a ``crucible/`` directory."""
    resolved = candidate.resolve()
    crucible_dir = resolved / "crucible"
    if not crucible_dir.is_dir():
        raise RuntimeError(
            f"CRUCIBLE_WORKSPACE_ROOT={candidate} does not contain a "
            f"crucible/ directory — check the env var"
        )
    return resolved


def detect_workspace_root(start: Path | None = None) -> Path | None:
    """Return the directory that contains ``crucible/.crucible/`` or ``crucible/`` as project.

    Walks upward from *start* (or :func:`os.getcwd`) until a candidate is found.
    """
    p = (start or Path.cwd()).resolve()
    for _ in range(40):
        crucible_dir = p / "crucible"
        dot_crucible = crucible_dir / ".crucible"
        if dot_crucible.is_dir():
            return p
        if crucible_dir.is_dir() and (crucible_dir / "pyproject.toml").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def workspace_root() -> Path:
    """Resolved workspace root (parent of ``crucible/`` project)."""
    raw = os.environ.get(_ENV_WORKSPACE)
    if raw:
        return _validate_workspace(Path(raw))
    w = detect_workspace_root()
    if w is not None:
        return w
    # Development fallback: crucible package lives at <workspace>/crucible
    return Path(__file__).resolve().parent.parent.parent




