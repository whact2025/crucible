# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Source-navigation ToolSet for crucible.

Four tools: ``read_source``, ``grep_symbol``, ``find_callers``,
``find_definition``.

These tools are read-only. They inspect the workspace filesystem and use
ripgrep or grep for search operations.

Usage::

    from crucible.tools import ToolContext
    from crucible.tools.source import build_source_tools

    ctx = ToolContext(workspace_root=ws, repo_root=ws)
    toolset = build_source_tools(ctx)
    print(toolset.dispatch("read_source", {"file_path": "README.md"}))
"""

from __future__ import annotations

from .handlers import build_source_tools, builder

__all__ = ["build_source_tools", "builder"]
