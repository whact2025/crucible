# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused tests for the canonical tool-authority identity."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.backends.tool_authority import (
    APPROVAL_POLICY_PER_CALL_EXACT,
    TOOL_AUTHORITY_SCHEMA_VERSION,
    canonical_tool_authority,
    tool_authority_digest,
)
from crucible.tools import ToolContext, ToolSetBuilder, composite
from crucible.tools._dispatch_budget import ProgressBoundedMutatingToolSet
from crucible.tools.profiles import build_autonomous_implementer_profile


_COMMON = {
    "capability": "workspace_write",
    "approval_policy": APPROVAL_POLICY_PER_CALL_EXACT,
    "bridge_protocol": "crucible-mcp-authority",
    "bridge_version": 1,
    "context_budget": 64_000,
}


def _build_tools(
    *,
    context: ToolContext | None = None,
    reverse: bool = False,
    write_echo: bool = True,
    description: str = "Echo a value.",
):
    builder = ToolSetBuilder("test")
    echo_schema = (
        {
            "required": ["value"],
            "properties": {
                "count": {"minimum": 0, "type": "integer"},
                "value": {"type": "string"},
            },
            "type": "object",
        }
        if reverse
        else {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "count": {"type": "integer", "minimum": 0},
            },
            "required": ["value"],
        }
    )
    registrations = [
        (
            "echo",
            description,
            echo_schema,
            write_echo,
        ),
        (
            "inspect",
            "Inspect a value.",
            {
                "properties": {"path": {"type": "string"}},
                "type": "object",
            },
            False,
        ),
    ]
    if reverse:
        registrations.reverse()
    for name, tool_description, schema, write in registrations:

        @builder.tool(
            name=name,
            description=tool_description,
            input_schema=schema,
            write=write,
        )
        def _handler(_ctx, **arguments: Any) -> str:
            return str(arguments)

    return builder.build(context)


def test_canonical_payload_covers_complete_authority_and_excludes_secrets(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "absolute-private-workspace"
    secret = "runtime-token-never-digest"
    endpoint = "https://127.0.0.1:49152/mcp"
    context = ToolContext(
        workspace_root=workspace,
        repo_root=workspace,
        runners=SimpleNamespace(token=secret, endpoint=endpoint),
        write_scope=("src/z.py", "src/a.py"),
        write_exact_scope=("src/z.py",),
        write_deny_scope=("src/generated",),
    )
    toolset = _build_tools(context=context)

    payload = canonical_tool_authority(toolset, **_COMMON)

    assert payload["schema_version"] == TOOL_AUTHORITY_SCHEMA_VERSION
    assert payload["bridge"] == {
        "protocol": "crucible-mcp-authority",
        "version": "1",
    }
    assert payload["capability"] == "workspace_write"
    assert payload["approval_policy"] == "per_call_exact"
    assert payload["context_budget"] == 64_000
    assert payload["classification"] == {
        "read_only": ["inspect"],
        "write": ["echo"],
    }
    assert [tool["name"] for tool in payload["tools"]] == ["echo", "inspect"]
    assert payload["tools"][0]["classification"] == "write"
    assert payload["tools"][1]["classification"] == "read_only"
    assert payload["tools"][0]["write_scope"] == ["src/a.py", "src/z.py"]
    assert payload["tools"][0]["write_exact_scope"] == ["src/z.py"]
    assert payload["tools"][0]["write_deny_scope"] == ["src/generated"]

    rendered = json.dumps(payload, sort_keys=True)
    assert str(workspace) not in rendered
    assert secret not in rendered
    assert endpoint not in rendered
    assert "_handler" not in rendered
    assert "workspace_root" not in rendered
    assert "repo_root" not in rendered


def test_composite_binds_each_tool_to_its_own_context(tmp_path: Path) -> None:
    left_builder = ToolSetBuilder("left")

    @left_builder.tool(description="left", input_schema={})
    def left(_ctx) -> str:
        return ""

    right_builder = ToolSetBuilder("right")

    @right_builder.tool(description="right", input_schema={}, write=True)
    def right(_ctx) -> str:
        return ""

    left_context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("left-only",),
    )
    right_context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("right-only",),
        write_deny_scope=("right-only/immutable",),
    )
    toolset = composite(
        right_builder.build(right_context),
        left_builder.build(left_context),
    )

    payload = canonical_tool_authority(toolset, **_COMMON)
    by_name = {tool["name"]: tool for tool in payload["tools"]}

    assert by_name["left"]["write_scope"] == ["left-only"]
    assert by_name["left"]["write_deny_scope"] == []
    assert by_name["right"]["write_scope"] == ["right-only"]
    assert by_name["right"]["write_deny_scope"] == [
        "right-only/immutable"
    ]


def test_progress_bounded_profile_preserves_leaf_scope_bindings(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("src/decision.py",),
        write_exact_scope=("src/decision.py",),
        write_deny_scope=("src/generated",),
    )
    inner = build_autonomous_implementer_profile(context)
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=4,
    )

    payload = canonical_tool_authority(bounded, **_COMMON)
    by_name = {tool["name"]: tool for tool in payload["tools"]}

    for name in bounded.tool_names:
        assert by_name[name]["write_scope"] == ["src/decision.py"]
        assert by_name[name]["write_exact_scope"] == ["src/decision.py"]
        assert by_name[name]["write_deny_scope"] == ["src/generated"]


def test_digest_is_stable_across_registration_mapping_and_scope_order(
    tmp_path: Path,
) -> None:
    first_context = ToolContext(
        workspace_root=tmp_path / "one",
        repo_root=tmp_path / "one",
        write_scope=("z", "a"),
        write_deny_scope=("z/generated", "a/generated"),
    )
    second_context = ToolContext(
        workspace_root=tmp_path / "entirely-different-root",
        repo_root=tmp_path / "entirely-different-root",
        write_scope=("a", "z"),
        write_deny_scope=("a/generated", "z/generated"),
    )
    first = _build_tools(context=first_context)
    second = _build_tools(context=second_context, reverse=True)

    assert tool_authority_digest(first, **_COMMON) == tool_authority_digest(
        second,
        **_COMMON,
    )


def test_digest_has_stable_sha256_wire_shape() -> None:
    digest = tool_authority_digest(_build_tools(), **_COMMON)

    assert digest == (
        "sha256:0482b1e576623a0205df6bfafce2ae16"
        "09c2f85390fe99f9d8964f61d4dfc699"
    )


def test_every_semantic_parameter_changes_the_digest(tmp_path: Path) -> None:
    baseline = _build_tools(
        context=ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("src",),
        ),
        write_echo=False,
    )
    original = tool_authority_digest(
        baseline,
        **{**_COMMON, "capability": "read_only"},
    )

    variants = [
        (
            baseline,
            {**_COMMON, "capability": "workspace_write"},
        ),
        (
            baseline,
            {
                **_COMMON,
                "capability": "read_only",
                "bridge_protocol": "other-authority-contract",
            },
        ),
        (
            baseline,
            {
                **_COMMON,
                "capability": "read_only",
                "bridge_version": 2,
            },
        ),
        (
            baseline,
            {
                **_COMMON,
                "capability": "read_only",
                "context_budget": 32_000,
            },
        ),
        (
            _build_tools(
                context=ToolContext(
                    workspace_root=tmp_path,
                    repo_root=tmp_path,
                    write_scope=("other",),
                ),
                write_echo=False,
            ),
            {**_COMMON, "capability": "read_only"},
        ),
        (
            _build_tools(
                context=ToolContext(
                    workspace_root=tmp_path,
                    repo_root=tmp_path,
                    write_scope=("src",),
                ),
                write_echo=False,
                description="Changed descriptor.",
            ),
            {**_COMMON, "capability": "read_only"},
        ),
    ]

    for toolset, kwargs in variants:
        assert tool_authority_digest(toolset, **kwargs) != original


class _MetadataToolSet:
    def __init__(
        self,
        *,
        descriptors: list[dict[str, Any]] | None = None,
        read_only_names: Any = ("tool",),
        write_names: Any = (),
    ) -> None:
        self._descriptors = descriptors or [
            {
                "name": "tool",
                "description": "test",
                "input_schema": {"type": "object"},
            }
        ]
        if read_only_names is not _MISSING:
            self.read_only_names = read_only_names
        if write_names is not _MISSING:
            self.write_names = write_names
        # These deliberately resemble connection/credential state. The
        # canonical helper must never inspect them.
        self.token = "not-for-digest"
        self.url = "https://127.0.0.1:65535/mcp"

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return self._descriptors


_MISSING = object()


def test_duplicate_descriptors_are_rejected() -> None:
    descriptor = {
        "name": "tool",
        "description": "test",
        "input_schema": {},
    }
    toolset = _MetadataToolSet(descriptors=[descriptor, dict(descriptor)])

    with pytest.raises(ValueError, match="duplicate tool descriptor"):
        canonical_tool_authority(toolset, **_COMMON)


@pytest.mark.parametrize(
    ("read_names", "write_names", "match"),
    [
        (_MISSING, (), "no read_only_names"),
        (("tool", "tool"), (), "duplicate 'tool'"),
        (("tool",), ("tool",), "both read-only and mutating"),
        ((), (), r"unclassified=\['tool'\]"),
        (("tool", "ghost"), (), r"unknown=\['ghost'\]"),
    ],
)
def test_incomplete_or_ambiguous_classification_is_rejected(
    read_names: Any,
    write_names: Any,
    match: str,
) -> None:
    toolset = _MetadataToolSet(
        read_only_names=read_names,
        write_names=write_names,
    )

    with pytest.raises(ValueError, match=match):
        canonical_tool_authority(toolset, **_COMMON)


@pytest.mark.parametrize(
    "scope",
    [
        "/absolute/path",
        "../escape",
        "src/../../escape",
        r"C:\absolute\path",
        "https://example.test/path",
        "",
        "\x00bad",
    ],
)
def test_non_relative_or_traversing_scopes_are_rejected(
    tmp_path: Path,
    scope: str,
) -> None:
    toolset = _build_tools(
        context=ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=(scope,),
        )
    )

    with pytest.raises(ValueError, match="write_scope"):
        canonical_tool_authority(toolset, **_COMMON)


def test_duplicate_scope_aliases_are_rejected(tmp_path: Path) -> None:
    toolset = _build_tools(
        context=ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("src/code.py", r"src\code.py"),
        )
    )

    with pytest.raises(ValueError, match="duplicate scope"):
        canonical_tool_authority(toolset, **_COMMON)


@pytest.mark.parametrize("context_budget", [0, -1, True, 1.5, "100"])
def test_invalid_context_budget_is_rejected(context_budget: Any) -> None:
    with pytest.raises(ValueError, match="context_budget"):
        canonical_tool_authority(
            _build_tools(),
            **{**_COMMON, "context_budget": context_budget},
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("approval_policy", "always", "per_call_exact"),
        ("bridge_protocol", "https://host:123/mcp", "not a URL"),
        ("bridge_protocol", "", "non-empty"),
        ("bridge_version", True, "non-empty"),
    ],
)
def test_invalid_policy_or_bridge_identity_is_rejected(
    field: str,
    value: Any,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        canonical_tool_authority(
            _build_tools(),
            **{**_COMMON, field: value},
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        {
            "name": "tool",
            "description": "test",
            "input_schema": {},
            "endpoint": "https://should-not-be-accepted.test",
        },
        {
            "name": "tool",
            "description": "test",
            "input_schema": {"default": object()},
        },
        {
            "name": "tool",
            "description": "test",
            "input_schema": {"default": float("nan")},
        },
    ],
)
def test_unknown_or_non_json_descriptor_metadata_is_rejected(
    descriptor: dict[str, Any],
) -> None:
    toolset = _MetadataToolSet(descriptors=[descriptor])

    with pytest.raises(ValueError):
        canonical_tool_authority(toolset, **_COMMON)
