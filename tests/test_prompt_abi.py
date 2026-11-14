# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crucible ToolSet integration contracts for Assay prompt ABI v2."""

from __future__ import annotations

from pathlib import Path

from assay.backend.features import BackendFeatureSet
from assay.primitives.implement import _implementer_system_prompt
from assay.primitives.plan import _planner_system_prompt

from crucible.tools import ToolContext
from crucible.tools.profiles import build_implementer_profile
from crucible.tools.source import build_source_tools


class _Backend:
    def __init__(self, name: str, features: BackendFeatureSet) -> None:
        self.name = name
        self.features = features


def _context(root: Path, *, container: bool = False) -> ToolContext:
    return ToolContext(
        workspace_root=root,
        repo_root=root,
        write_scope=("src/example.py",),
        container_id="test-container" if container else None,
        container_workdir="/workspace" if container else None,
    )


def test_codex_mcp_prompt_renders_real_crucible_catalogue(
    tmp_path: Path,
) -> None:
    toolset = build_source_tools(_context(tmp_path))
    prompt = _planner_system_prompt(
        _Backend(
            "codex",
            BackendFeatureSet(
                structured_output="native",
                tool_loop="sdk_owned",
                project_instructions="suppressible",
            ),
        ),
        prompt_toolset=toolset,
    )
    names = {item["name"] for item in toolset.to_anthropic_tools()}

    assert names == {
        "find_callers",
        "find_definition",
        "grep_symbol",
        "read_source",
    }
    for name in names:
        assert f"`mcp__crucible__{name}`" in prompt
    assert "`Read`" not in prompt
    assert "`Grep`" not in prompt
    assert "25K" not in prompt
    assert prompt.count("## Required output contract") == 1
    assert '"additionalProperties"' not in prompt
    assert "serial within one Tier checkout" in " ".join(prompt.split())


def test_kimi_planner_gets_path_and_exhaustion_overlay_only(
    tmp_path: Path,
) -> None:
    toolset = build_source_tools(_context(tmp_path))
    backend = _Backend(
        "kimi",
        BackendFeatureSet(
            structured_output="prompt_only",
            tool_loop="sdk_owned",
            project_instructions="suppressible",
        ),
    )

    planner_prompt = _planner_system_prompt(
        backend,
        prompt_toolset=toolset,
    )
    implementer_prompt = _implementer_system_prompt(
        backend,
        prompt_toolset=toolset,
    )
    normalized_planner_prompt = " ".join(planner_prompt.split())
    names = {item["name"] for item in toolset.to_anthropic_tools()}

    assert "Kimi planner path and exhaustion protocol" in planner_prompt
    assert (
        "exact spelling returned by the attached source tools"
        in normalized_planner_prompt
    )
    assert "repository-root-relative" in normalized_planner_prompt
    assert "already exists" in normalized_planner_prompt
    assert "genuinely new" in normalized_planner_prompt
    assert (
        "make no further tool calls in that invocation"
        in normalized_planner_prompt
    )
    assert (
        "one complete closed TierDAG JSON object"
        in normalized_planner_prompt
    )
    assert "Kimi planner path and exhaustion protocol" not in implementer_prompt
    for name in names:
        assert f"`{name}`" in planner_prompt
    assert {
        line
        for line in planner_prompt.splitlines()
        if line.startswith("- `")
    } == {
        line
        for line in _planner_system_prompt(
            _Backend("generic", backend.features),
            prompt_toolset=toolset,
        ).splitlines()
        if line.startswith("- `")
    }


def test_host_implementer_advertises_only_real_build_tool(
    tmp_path: Path,
) -> None:
    toolset = build_implementer_profile(_context(tmp_path))
    prompt = _implementer_system_prompt(
        _Backend(
            "claude",
            BackendFeatureSet(
                structured_output="native",
                tool_loop="sdk_owned",
                project_instructions="suppressible",
            ),
        ),
        prompt_toolset=toolset,
    )

    assert "`mcp__crucible__build_project`" in prompt
    assert "at most one targeted self-check/repair loop" in prompt
    assert "`Read`" not in prompt
    assert "`Grep`" not in prompt
    assert "25,000" not in prompt


def test_container_implementer_keeps_testing_workflow_owned(
    tmp_path: Path,
) -> None:
    toolset = build_implementer_profile(_context(tmp_path, container=True))
    prompt = _implementer_system_prompt(
        _Backend(
            "kimi",
            BackendFeatureSet(
                structured_output="prompt_only",
                tool_loop="sdk_owned",
                project_instructions="suppressible",
            ),
        ),
        prompt_toolset=toolset,
    )

    assert "`build_project`" not in prompt
    assert "`mcp__crucible__build_project`" not in prompt
    assert "workflow owns authoritative testing" in prompt.lower()
    assert prompt.count("## Required output contract") == 1
