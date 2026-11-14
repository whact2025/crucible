# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Exact-bound source creation for focused workflow recovery.

The ordinary ``write_source`` tool accepts a model-controlled path because a
normal implementation invocation may edit several declared files.  A focused
missing-output recovery has a stronger contract: workflow code already knows
the one absent path, so advertising another free-form path only creates an
avoidable failure mode.

``build_exact_create_tools`` keeps the canonical public tool name
``write_source`` (so backend write guards recognize it) while binding the path
inside the handler.  The model supplies content only.  Unexpected extra
arguments are ignored rather than becoming mutation authority; regardless of
what a provider emits, the handler can write only the workflow-bound target.
"""

from __future__ import annotations

from typing import Any

from .._builder import BuiltToolSet, ToolSetBuilder
from .._context import ToolContext
from .handlers import write_source as _write_source


def build_exact_create_tools(
    context: ToolContext,
    *,
    target_path: str,
) -> BuiltToolSet:
    """Build a singleton content-only writer bound to ``target_path``."""

    builder = ToolSetBuilder("exact_create")

    @builder.tool(
        name="write_source",
        description=(
            "Create the single exact source file bound by the workflow. "
            "Provide its complete content; the destination path is fixed and "
            "cannot be selected or changed by this call."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete content for the bound file",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        write=True,
    )
    def exact_write_source(
        ctx: ToolContext,
        *,
        content: str,
        **_ignored: Any,
    ) -> str:
        return _write_source(
            ctx,
            file_path=target_path,
            content=content,
        )

    toolset = builder.build(context)
    # The process-isolated mutation host cannot serialize this closure. Expose
    # only its inert, workflow-owned target as typed reconstruction metadata;
    # the host still requires the canonical descriptor and matching exact
    # ToolContext scope before it accepts the binding.
    toolset.isolated_exact_write_target = target_path  # type: ignore[attr-defined]
    return toolset


__all__ = ["build_exact_create_tools"]
