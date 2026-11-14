# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Closed registry for every workflow backend role.

Routing names are control-plane identifiers, not free-form labels.  Keeping a
single registry prevents a typo from silently selecting the expensive default
backend and lets the backend pool reject capability escalation before an
adapter is constructed.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from crucible.backends.capabilities import CapabilityProfile, normalize_capability

if TYPE_CHECKING:
    from .features import FeatureRequirement


RoleKind = Literal["reviewer", "planner", "implementer", "orchestrator"]


@dataclasses.dataclass(frozen=True)
class RoleSpec:
    name: str
    kind: RoleKind
    max_capability: CapabilityProfile
    independence_group: str | None = None


_SPECIAL_ROLES: tuple[RoleSpec, ...] = (
    RoleSpec("plan_planner", "planner", "read_only", "plan"),
    RoleSpec("architect_planner", "planner", "read_only", "architect"),
    RoleSpec("architect_critic", "reviewer", "text", "architect"),
    RoleSpec("architect_judge", "reviewer", "text", "architect"),
    RoleSpec("architect_reviewer", "reviewer", "text", "architect"),
    RoleSpec("implementer", "implementer", "workspace_write", "implementation"),
    RoleSpec("breach_defender", "implementer", "workspace_write", "breach"),
    RoleSpec("tier", "orchestrator", "workspace_write", "tier"),
)


@lru_cache(maxsize=1)
def registry() -> dict[str, RoleSpec]:
    """Return the complete role registry, including every rubric role."""

    # Imported lazily so low-level adapter imports do not eagerly parse the
    # rubric manifest.  The manifest itself is the authority for reviewer
    # names and tool ceilings.
    from crucible.rubrics import contracts

    result = {item.name: item for item in _SPECIAL_ROLES}
    for contract in contracts():
        capability: CapabilityProfile = (
            "read_only" if contract.tools == "read_only" else "text"
        )
        spec = RoleSpec(
            contract.name,
            "reviewer",
            capability,
            contract.family,
        )
        existing = result.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"role {spec.name!r} is defined inconsistently")
        result[spec.name] = spec
    return result


def role_spec(name: str) -> RoleSpec:
    """Resolve a registered role or fail closed with a useful inventory."""

    try:
        return registry()[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend role {name!r}; registered roles are: "
            + ", ".join(sorted(registry()))
        ) from exc


def validate_role_capability(name: str, capability: str) -> CapabilityProfile:
    """Reject a caller asking for more authority than its role permits."""

    requested = normalize_capability(capability)
    allowed = role_spec(name).max_capability
    rank = {"text": 0, "read_only": 1, "workspace_write": 2}
    if rank[requested] > rank[allowed]:
        raise ValueError(
            f"role {name!r} permits at most {allowed!r}, not {requested!r}"
        )
    return requested


def role_feature_requirements(
    name: str,
) -> tuple["FeatureRequirement", ...]:
    """Return feature requirements inherent to a workflow role's protocol.

    Planner and reviewer primitives always send a response schema and retain
    deterministic parsing. They may therefore accept either native constrained
    decoding or an explicitly qualified prompt-only fallback, but never an
    unqualified/unsupported structured-output route.

    The local import keeps the low-level role registry independent from
    feature-profile initialization.
    """

    from .features import FeatureRequirement

    if role_spec(name).kind in {"planner", "reviewer"}:
        return (
            FeatureRequirement(
                "structured_output",
                frozenset({"native", "prompt_only"}),
            ),
        )
    return ()


__all__ = [
    "RoleKind",
    "RoleSpec",
    "registry",
    "role_feature_requirements",
    "role_spec",
    "validate_role_capability",
]
