# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Backend-enforced capability profiles.

Workflow roles declare the strongest surface they need.  Adapters use the
profile to configure their native sandbox/tool catalogue, while this module
rejects accidental attachment of mutating MCP tools to analytical roles.
"""

from __future__ import annotations

from typing import Any, Literal, cast


CapabilityProfile = Literal["text", "read_only", "workspace_write"]
CAPABILITY_PROFILES: frozenset[str] = frozenset(
    {"text", "read_only", "workspace_write"}
)


def normalize_capability(value: str) -> CapabilityProfile:
    if value not in CAPABILITY_PROFILES:
        raise ValueError(
            f"unknown capability profile {value!r}; expected one of "
            f"{sorted(CAPABILITY_PROFILES)}"
        )
    return cast(CapabilityProfile, value)


def toolset_write_names(toolset: Any) -> set[str] | None:
    """Return known mutating tool names, or ``None`` when unclassified."""

    if toolset is None:
        return set()
    names = getattr(toolset, "write_names", None)
    if names is None:
        return None
    return {str(name) for name in names}


def validate_toolset_capability(
    profile: CapabilityProfile,
    toolset: Any,
) -> None:
    """Fail closed when a toolset exceeds a backend's declared profile."""

    if toolset is None:
        return
    if profile == "text":
        raise ValueError("text capability cannot attach an MCP toolset")
    writes = toolset_write_names(toolset)
    if writes is None:
        raise ValueError(
            "toolset has no write_names metadata; classify every tool before "
            f"attaching it to a {profile!r} backend"
        )
    if profile == "read_only" and writes:
        raise ValueError(
            "read_only capability cannot attach mutating tools: "
            + ", ".join(sorted(writes))
        )


__all__ = [
    "CAPABILITY_PROFILES",
    "CapabilityProfile",
    "normalize_capability",
    "toolset_write_names",
    "validate_toolset_capability",
]
