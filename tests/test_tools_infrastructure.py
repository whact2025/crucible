# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.tools._builder`` + ``_composite`` + ``_context``."""

from __future__ import annotations

from pathlib import Path

import pytest

from assay import ToolSet
from crucible.tools import (
    BuiltToolSet,
    CompositeToolSet,
    RegisteredTool,
    ToolContext,
    ToolSetBuilder,
    composite,
)


# ----------------------------------------------------------------------
# ToolContext
# ----------------------------------------------------------------------


def test_tool_context_minimal_construction(tmp_path: Path):
    ctx = ToolContext(workspace_root=tmp_path, repo_root=tmp_path)
    assert ctx.workspace_root == tmp_path
    assert ctx.repo_root == tmp_path
    assert ctx.runners is None


def test_tool_context_holds_optional_fields(tmp_path: Path):
    ctx = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("src",),
        context_budget=64000,
    )
    assert ctx.write_scope == ("src",)
    assert ctx.context_budget == 64000


# ----------------------------------------------------------------------
# ToolSetBuilder
# ----------------------------------------------------------------------


def test_builder_registers_and_builds(tmp_path: Path):
    builder = ToolSetBuilder("demo")

    @builder.tool(
        description="Echo back the message",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )
    def echo(ctx, *, message: str) -> str:
        return f"echo: {message}"

    assert builder.tool_names == {"echo"}
    ctx = ToolContext(workspace_root=tmp_path, repo_root=tmp_path)
    ts = builder.build(context=ctx)
    assert isinstance(ts, BuiltToolSet)
    assert isinstance(ts, ToolSet)


def test_builder_rejects_duplicate_registration():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="x", input_schema={})
    def t(ctx) -> str:
        return ""

    with pytest.raises(ValueError, match="already registered"):
        @builder.tool(description="dup", input_schema={}, name="t")
        def _other(ctx) -> str:
            return ""


def test_builder_supports_explicit_name():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="x", input_schema={}, name="custom_name")
    def handler(ctx) -> str:
        return ""

    assert builder.tool_names == {"custom_name"}


def test_builder_marks_write_tools():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="r", input_schema={})
    def reader(ctx) -> str:
        return ""

    @builder.tool(description="w", input_schema={}, write=True)
    def writer(ctx) -> str:
        return ""

    ts = builder.build()
    assert ts.read_only_names == {"reader"}
    assert ts.write_names == {"writer"}


# ----------------------------------------------------------------------
# BuiltToolSet — ToolSet protocol surface
# ----------------------------------------------------------------------


def _toy_builder() -> ToolSetBuilder:
    builder = ToolSetBuilder("toy")

    @builder.tool(
        description="Greet someone",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    def greet(ctx, *, name: str) -> str:
        return f"hello, {name}"

    @builder.tool(
        description="Add two ints",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    def add(ctx, *, a: int, b: int) -> str:
        return str(a + b)

    return builder


def test_to_anthropic_tools_shape():
    ts = _toy_builder().build()
    out = ts.to_anthropic_tools()
    assert len(out) == 2
    names = {entry["name"] for entry in out}
    assert names == {"greet", "add"}
    for entry in out:
        assert "description" in entry
        assert "input_schema" in entry


def test_to_openai_tools_shape():
    ts = _toy_builder().build()
    out = ts.to_openai_tools()
    assert len(out) == 2
    for entry in out:
        assert entry["type"] == "function"
        assert "parameters" in entry["function"]


def test_dispatch_routes_to_handler():
    ts = _toy_builder().build()
    assert ts.dispatch("greet", {"name": "world"}) == "hello, world"
    assert ts.dispatch("add", {"a": 2, "b": 3}) == "5"


def test_dispatch_unknown_tool_returns_error_string():
    ts = _toy_builder().build()
    out = ts.dispatch("nope", {})
    assert out.startswith("ERROR: unknown tool 'nope'")


def test_dispatch_catches_handler_exception():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="boom", input_schema={})
    def boom(ctx) -> str:
        raise RuntimeError("kaboom")

    ts = builder.build()
    out = ts.dispatch("boom", {})
    assert "RuntimeError" in out
    assert "kaboom" in out


def test_dispatch_coerces_non_string_return():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="d", input_schema={})
    def n(ctx) -> int:  # type: ignore[return-value]
        return 42

    ts = builder.build()
    assert ts.dispatch("n", {}) == "42"


def test_dispatch_handles_none_return():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="d", input_schema={})
    def n(ctx) -> None:
        return None

    ts = builder.build()
    assert ts.dispatch("n", {}) == ""


def test_dispatch_arguments_default_to_empty_dict():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="d", input_schema={})
    def n(ctx) -> str:
        return "ok"

    ts = builder.build()
    assert ts.dispatch("n", None) == "ok"
    assert ts.dispatch("n", {}) == "ok"


# ----------------------------------------------------------------------
# BuiltToolSet — context passing
# ----------------------------------------------------------------------


def test_dispatch_passes_context_to_handler(tmp_path: Path):
    seen: list[ToolContext] = []
    builder = ToolSetBuilder("demo")

    @builder.tool(description="d", input_schema={})
    def grab(ctx) -> str:
        seen.append(ctx)
        return "ok"

    ctx = ToolContext(workspace_root=tmp_path, repo_root=tmp_path)
    ts = builder.build(context=ctx)
    ts.dispatch("grab", {})
    assert seen == [ctx]


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------


def test_filter_to_drops_unselected_tools():
    ts = _toy_builder().build()
    narrow = ts.filter_to({"greet"})
    assert narrow.tool_names == {"greet"}


def test_filter_read_only_drops_write_tools():
    builder = ToolSetBuilder("demo")

    @builder.tool(description="r", input_schema={})
    def reader(ctx) -> str:
        return "r"

    @builder.tool(description="w", input_schema={}, write=True)
    def writer(ctx) -> str:
        return "w"

    ts = builder.build()
    narrow = ts.filter_read_only()
    assert narrow.tool_names == {"reader"}


def test_filter_to_unknown_name_silently_dropped():
    ts = _toy_builder().build()
    narrow = ts.filter_to({"greet", "nope"})
    assert narrow.tool_names == {"greet"}


# ----------------------------------------------------------------------
# CompositeToolSet
# ----------------------------------------------------------------------


def test_composite_merges_two_toolsets():
    src = ToolSetBuilder("src")

    @src.tool(description="d", input_schema={})
    def alpha(ctx) -> str:
        return "alpha"

    sink = ToolSetBuilder("sink")

    @sink.tool(description="d", input_schema={})
    def beta(ctx) -> str:
        return "beta"

    comp = composite(src.build(), sink.build())
    assert comp.tool_names == {"alpha", "beta"}
    assert comp.dispatch("alpha", {}) == "alpha"
    assert comp.dispatch("beta", {}) == "beta"


def test_composite_routes_dispatch_to_originating_toolset():
    src = ToolSetBuilder("src")

    @src.tool(description="d", input_schema={})
    def fn(ctx) -> str:
        return "from_src"

    sink = ToolSetBuilder("sink")

    @sink.tool(description="d", input_schema={})
    def other(ctx) -> str:
        return "from_sink"

    comp = composite(src.build(), sink.build())
    assert comp.dispatch("fn", {}) == "from_src"
    assert comp.dispatch("other", {}) == "from_sink"


def test_composite_rejects_duplicate_names():
    a = ToolSetBuilder("a")

    @a.tool(description="d", input_schema={})
    def shared(ctx) -> str:
        return "a"

    b = ToolSetBuilder("b")

    @b.tool(description="d", input_schema={})
    def shared(ctx) -> str:  # noqa: F811
        return "b"

    with pytest.raises(ValueError, match="duplicate tool name"):
        composite(a.build(), b.build())


def test_composite_emits_concatenated_anthropic_tools():
    a = ToolSetBuilder("a")
    @a.tool(description="d", input_schema={})
    def x(ctx) -> str: return ""

    b = ToolSetBuilder("b")
    @b.tool(description="d", input_schema={})
    def y(ctx) -> str: return ""

    comp = composite(a.build(), b.build())
    out = comp.to_anthropic_tools()
    assert {e["name"] for e in out} == {"x", "y"}


def test_composite_satisfies_toolset_protocol():
    a = ToolSetBuilder("a")
    @a.tool(description="d", input_schema={})
    def x(ctx) -> str: return ""

    comp = composite(a.build())
    assert isinstance(comp, ToolSet)


def test_composite_dispatch_unknown_name_returns_error_string():
    a = ToolSetBuilder("a")
    @a.tool(description="d", input_schema={})
    def x(ctx) -> str: return ""

    comp = composite(a.build())
    out = comp.dispatch("nope", {})
    assert out.startswith("ERROR: unknown tool 'nope'")
