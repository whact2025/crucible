# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``ToolSetBuilder`` — decorator-based registry for ``assay.ToolSet``.

Each crucible/tools/<domain>/ subpackage owns one builder. Handlers
register themselves via ``@builder.tool(...)`` carrying their schema
inline; ``builder.build(context)`` then returns a ``BuiltToolSet``
that satisfies the ``assay.ToolSet`` runtime-checkable protocol.

Handler signature convention:

    @builder.tool(description="...", input_schema={...})
    def read_source(ctx: ToolContext, *, file_path: str, ...) -> str:
        ...

- First positional arg is ``ctx: ToolContext`` (captured at build time).
- The rest are kwargs matching the input_schema's properties.
- Return must be a ``str`` (or coercible) — the tool-use loop wraps
  it in a tool_use_result block.

Ordinary handler errors are returned as strings so the model can adjust.
Terminal dispatch signals propagate to the adapter and close the provider
invocation before it can observe a partially restored workspace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ._context import ToolContext

logger = logging.getLogger(__name__)

_HandlerFn = Callable[..., Any]


@dataclass(frozen=True)
class RegisteredTool:
    """One tool's registration entry — schema + callable + metadata."""

    name: str
    description: str
    input_schema: dict
    handler: _HandlerFn
    write: bool = False


class ToolSetBuilder:
    """Per-domain registry. Use the ``@tool(...)`` decorator to register.

    Usage::

        builder = ToolSetBuilder("source")

        @builder.tool(
            description="Read a file's contents.",
            input_schema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        )
        def read_source(ctx, *, file_path: str) -> str:
            return Path(file_path).read_text()

        TOOLS: ToolSet = builder.build(context=ctx)
    """

    def __init__(self, domain: str) -> None:
        self._domain = domain
        self._tools: dict[str, RegisteredTool] = {}

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def tool_names(self) -> set[str]:
        return set(self._tools)

    def tool(
        self,
        *,
        description: str,
        input_schema: dict,
        name: str | None = None,
        write: bool = False,
    ):
        """Register a handler. Decorator form.

        ``name`` defaults to the handler's ``__name__``. ``write=True``
        marks the tool as a write surface (used by recipe layers that
        want to restrict to read-only tools).
        """
        def decorator(fn: _HandlerFn) -> _HandlerFn:
            tool_name = name or fn.__name__
            if tool_name in self._tools:
                raise ValueError(
                    f"tool {tool_name!r} already registered in domain "
                    f"{self._domain!r}"
                )
            self._tools[tool_name] = RegisteredTool(
                name=tool_name,
                description=description,
                input_schema=input_schema,
                handler=fn,
                write=write,
            )
            return fn
        return decorator

    def build(self, context: ToolContext | None = None) -> "BuiltToolSet":
        """Return a ``BuiltToolSet`` bound to ``context`` for dispatch."""
        return BuiltToolSet(
            domain=self._domain,
            tools=dict(self._tools),
            context=context,
        )


class BuiltToolSet:
    """Concrete ``assay.ToolSet`` implementation produced by a builder.

    Handlers receive ``context`` as their first positional argument;
    the remaining arguments come from ``dispatch(name, arguments)``
    via keyword splat.
    """

    def __init__(
        self,
        *,
        domain: str,
        tools: dict[str, RegisteredTool],
        context: ToolContext | None,
    ) -> None:
        self._domain = domain
        self._tools = dict(tools)
        self._context = context

    # ------------------------------------------------------------------
    # Public attributes
    # ------------------------------------------------------------------

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def context(self) -> ToolContext | None:
        return self._context

    @property
    def tool_names(self) -> set[str]:
        return set(self._tools)

    @property
    def read_only_names(self) -> set[str]:
        return {t.name for t in self._tools.values() if not t.write}

    @property
    def write_names(self) -> set[str]:
        return {t.name for t in self._tools.values() if t.write}

    # ------------------------------------------------------------------
    # assay.ToolSet protocol
    # ------------------------------------------------------------------

    def to_anthropic_tools(self) -> list[dict]:
        """Emit the Anthropic Messages-API tool list shape."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def to_openai_tools(self) -> list[dict]:
        """Emit the OpenAI Chat Completions function-tools shape."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, arguments: dict) -> str:
        """Execute one tool call in-process. Returns the result string.

        Ordinary handler exceptions are caught and returned as error strings
        so the model can read + adapt. The full traceback is logged at
        WARNING; the returned string carries the exception type + message
        only. TerminalToolDispatchError propagates so the adapter can end the
        provider invocation without retry.
        """
        tool = self._tools.get(name)
        if tool is None:
            return (
                f"ERROR: unknown tool {name!r} in domain "
                f"{self._domain!r}. Available: {sorted(self._tools)}"
            )
        try:
            result = tool.handler(self._context, **(arguments or {}))
        except Exception as exc:  # noqa: BLE001 — see docstring
            from ._dispatch_budget import (  # noqa: PLC0415
                TerminalToolDispatchError,
            )

            if isinstance(exc, TerminalToolDispatchError):
                raise
            logger.warning(
                "tool %s.%s dispatch failed: %s",
                self._domain, name, exc, exc_info=True,
            )
            return f"ERROR: {type(exc).__name__}: {exc}"
        if isinstance(result, str):
            return result
        if result is None:
            return ""
        # Best-effort string coercion.
        return str(result)

    def mcp_config(self) -> Any | None:
        """No MCP bridge by default; subclasses override if needed."""
        return None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_to(self, names: Iterable[str]) -> "BuiltToolSet":
        """Return a new ToolSet exposing only the requested tool names.

        Unknown names are silently dropped. Use to give a workflow a
        narrower surface, such as a read-only review toolset.
        """
        keep = set(names) & set(self._tools)
        return BuiltToolSet(
            domain=self._domain,
            tools={k: self._tools[k] for k in keep},
            context=self._context,
        )

    def filter_read_only(self) -> "BuiltToolSet":
        """Return a new ToolSet excluding write tools."""
        return self.filter_to(self.read_only_names)


__all__ = ["ToolSetBuilder", "BuiltToolSet", "RegisteredTool"]
