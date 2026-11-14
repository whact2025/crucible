# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Build ToolSet — ``build_project`` + ``create_rust_crate``.

Pairs naturally with ``write/`` in the implementer profile: read source,
modify, then compile.

Schemas + handlers live in ``handlers.py``. The cross-process build
lock lives in ``crucible.tools._buildlock``.
"""

from __future__ import annotations

from .handlers import build_build_tools, build_project, builder, create_rust_crate

__all__ = [
    "build_build_tools",
    "builder",
    "build_project",
    "create_rust_crate",
]
