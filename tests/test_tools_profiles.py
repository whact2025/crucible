# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the workflow-profile factories.

Each profile exposes the navigation and bounded mutation authority its
workflow role requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay import ToolSet
from crucible.tools import ToolContext
from crucible.tools.profiles import (
    build_breach_defender_profile,
    build_autonomous_implementer_profile,
    build_exact_create_recovery_profile,
    build_implementer_profile,
)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, repo_root=tmp_path)


def _container_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path, repo_root=tmp_path, container_id="cid_abc",
    )


# ----------------------------------------------------------------------
# All profiles satisfy ToolSet
# ----------------------------------------------------------------------


@pytest.mark.parametrize("factory", [
    build_implementer_profile,
    build_breach_defender_profile,
])
def test_profile_satisfies_toolset_protocol(tmp_path, factory):
    ts = factory(_ctx(tmp_path))
    assert isinstance(ts, ToolSet)


# ----------------------------------------------------------------------
# IMPLEMENTER profile = source + write + build
# ----------------------------------------------------------------------


def test_implementer_profile_includes_source_write_build(tmp_path):
    ts = build_implementer_profile(_ctx(tmp_path))
    # 4 source + 4 write + 2 build = 10 tools
    assert ts.tool_names == {
        "read_source", "grep_symbol", "find_callers", "find_definition",
        "write_source", "apply_patch", "delete_source",
        "materialize_artifact",
        "build_project", "create_rust_crate",
    }


def test_autonomous_implementer_profile_is_process_hostable_source_only(
    tmp_path,
):
    ts = build_autonomous_implementer_profile(_ctx(tmp_path))

    assert ts.tool_names == {
        "read_source",
        "grep_symbol",
        "find_callers",
        "find_definition",
        "write_source",
        "apply_patch",
        "delete_source",
    }
    assert ts.write_names == {
        "write_source",
        "apply_patch",
        "delete_source",
    }


def test_implementer_profile_container_has_no_shell_or_build(tmp_path):
    """Container authoring exposes only navigation and scoped file writes."""
    ts = build_implementer_profile(_container_ctx(tmp_path))
    assert "container_bash" not in ts.tool_names
    assert "build_project" not in ts.tool_names
    assert "create_rust_crate" not in ts.tool_names
    # source + write still present
    assert {
        "read_source", "write_source", "apply_patch", "delete_source",
        "materialize_artifact",
    } <= ts.tool_names


# ----------------------------------------------------------------------
# EXACT CREATE RECOVERY = source + one bound content-only writer
# ----------------------------------------------------------------------


def test_exact_create_recovery_profile_has_content_only_writer(tmp_path):
    profile = build_exact_create_recovery_profile(
        _ctx(tmp_path),
        target_path="tests/generated_test.py",
    )

    assert isinstance(profile, ToolSet)
    assert profile.tool_names == {
        "read_source",
        "grep_symbol",
        "find_callers",
        "find_definition",
        "write_source",
    }
    assert profile.write_names == {"write_source"}
    assert "apply_patch" not in profile.tool_names
    assert "materialize_artifact" not in profile.tool_names

    anthropic_writer = next(
        tool for tool in profile.to_anthropic_tools()
        if tool["name"] == "write_source"
    )
    schema = anthropic_writer["input_schema"]
    assert schema == {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Complete content for the bound file",
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    }
    openai_writer = next(
        tool for tool in profile.to_openai_tools()
        if tool["function"]["name"] == "write_source"
    )
    assert openai_writer["function"]["parameters"] == schema
    assert "file_path" not in schema["properties"]


def test_exact_create_writer_ignores_model_path_and_writes_bound_target(
    tmp_path,
):
    production = tmp_path / "component.py"
    production.write_text("PRODUCTION = True\n", encoding="utf-8")
    context = _ctx(tmp_path)
    profile = build_exact_create_recovery_profile(
        context,
        target_path="tests/generated_test.py",
    )

    result = profile.dispatch("write_source", {
        "file_path": "component.py",
        "content": "GENERATED = True\n",
    })

    assert result == "OK: wrote 1 lines to tests/generated_test.py"
    assert production.read_text(encoding="utf-8") == "PRODUCTION = True\n"
    assert tmp_path.joinpath("tests/generated_test.py").read_text(
        encoding="utf-8",
    ) == "GENERATED = True\n"
    assert context.write_scope == ()
    assert context.write_exact_scope == ()


# ----------------------------------------------------------------------
# BREACH_DEFENDER profile = source + write only
# ----------------------------------------------------------------------


def test_breach_defender_profile_surface(tmp_path):
    ts = build_breach_defender_profile(_ctx(tmp_path))
    assert len(ts.tool_names) == 6
    assert "materialize_artifact" not in ts.tool_names
    assert "write_breach_test" not in ts.tool_names
    assert "run_breach_test" not in ts.tool_names
    assert "materialize_artifact" not in ts.tool_names
    assert "build_project" not in ts.tool_names


def test_breach_defender_container_has_no_execution_capability(tmp_path):
    """Model-authored attack prose cannot trigger shell or test execution."""
    ts = build_breach_defender_profile(_container_ctx(tmp_path))
    assert "container_bash" not in ts.tool_names
    assert "build_project" not in ts.tool_names
    assert "create_rust_crate" not in ts.tool_names
    assert "write_breach_test" not in ts.tool_names
    assert "run_breach_test" not in ts.tool_names
