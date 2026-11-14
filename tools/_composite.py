# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``CompositeToolSet`` — merge N ``ToolSet``s into one.

Workflows that want source navigation and scoped writes without execution
surface compose two domain ToolSets into one:

    from crucible.tools.source import build_source_tools
    from crucible.tools.write import build_write_tools
    from crucible.tools import composite, ToolContext

    ctx = ToolContext(workspace_root=..., repo_root=...)
    toolset = composite(
        build_source_tools(ctx),
        build_write_tools(ctx),
    )

``dispatch`` routes by tool name to the originating ToolSet.
Duplicate names across composed ToolSets raise at construction time
(first registration wins is too easy to ship by accident).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from assay.types.tools import ToolSet


class CompositeToolSet:
    """A ``ToolSet`` that merges several others into one dispatch surface."""

    def __init__(self, *toolsets: ToolSet) -> None:
        self._members: tuple[ToolSet, ...] = tuple(toolsets)
        self._dispatch_by_name: dict[str, ToolSet] = {}
        for ts in toolsets:
            for entry in ts.to_anthropic_tools():
                name = entry["name"]
                if name in self._dispatch_by_name:
                    raise ValueError(
                        f"duplicate tool name {name!r} across composed "
                        f"ToolSets; first owner has it"
                    )
                self._dispatch_by_name[name] = ts

    # ------------------------------------------------------------------
    # Public attributes
    # ------------------------------------------------------------------

    @property
    def members(self) -> tuple[ToolSet, ...]:
        return self._members

    @property
    def tool_names(self) -> set[str]:
        return set(self._dispatch_by_name)

    @property
    def write_names(self) -> set[str]:
        """Names explicitly classified as mutating by member toolsets.

        Capability enforcement treats missing metadata as an error, so a
        composite must retain the classification rather than erasing it.
        """
        out: set[str] = set()
        for member in self._members:
            names = getattr(member, "write_names", None)
            if names is None:
                raise ValueError(
                    "composite member lacks write_names capability metadata"
                )
            out.update(str(name) for name in names)
        return out

    @property
    def read_only_names(self) -> set[str]:
        return self.tool_names - self.write_names

    # ------------------------------------------------------------------
    # assay.ToolSet protocol
    # ------------------------------------------------------------------

    def to_anthropic_tools(self) -> list[dict]:
        out: list[dict] = []
        for ts in self._members:
            out.extend(ts.to_anthropic_tools())
        return out

    def to_openai_tools(self) -> list[dict]:
        out: list[dict] = []
        for ts in self._members:
            out.extend(ts.to_openai_tools())
        return out

    def dispatch(self, name: str, arguments: dict) -> str:
        ts = self._dispatch_by_name.get(name)
        if ts is None:
            return (
                f"ERROR: unknown tool {name!r} in composite. "
                f"Available: {sorted(self._dispatch_by_name)}"
            )
        return ts.dispatch(name, arguments)

    def mcp_config(self) -> Any | None:
        # If any member exposes an MCP config, prefer the first one.
        # Composing multiple MCP-bridged toolsets is out-of-scope for
        # this layer — recipes do that wiring.
        for ts in self._members:
            cfg = ts.mcp_config()
            if cfg is not None:
                return cfg
        return None


def composite(*toolsets: ToolSet) -> CompositeToolSet:
    """Convenience constructor — ``composite(a, b, c)`` over the class."""
    return CompositeToolSet(*toolsets)


__all__ = ["CompositeToolSet", "composite"]
