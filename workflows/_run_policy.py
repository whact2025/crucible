# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Versioned workflow execution policy for backend-neutral orchestration.

The semantic :class:`assay.Spec` describes what should be built.  This module
describes how a workflow may distribute read-only analysis without granting
additional task authority.  Keeping the two contracts separate lets a run
bind scheduler behavior into provenance without rewriting the Spec.

The implemented rollout remains deliberately narrow:

* orchestration is off by default;
* enabled children are read-only, depth one, and parent-synthesized;
* each child receives a bounded share of the remaining workflow budget; and
* cancellation always drains and joins the complete child set.
* writer lanes are a separate, off-by-default policy with one disposable
  worktree per child and a serial, declaration-ordered integration lane; and
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from assay.canonical import hash_record


ORCHESTRATION_POLICY_VERSION = "crucible.orchestration-policy.v1"
WRITER_LANE_POLICY_VERSION = "crucible.writer-lane-policy.v1"
OrchestrationMode = Literal["off", "read_only", "isolated_writer"]
WriterLaneMode = Literal["off", "isolated_writer"]



@dataclass(frozen=True, slots=True)
class OrchestrationPolicy:
    """Fail-closed limits for one workflow-owned orchestration lane.

    ``child_roles`` contains logical child names such as
    ``repository_mapper``.  Routing still uses an existing qualified parent
    role; a child name is attribution, never authority to invent a new route.
    """

    mode: OrchestrationMode = "off"
    parent_roles: tuple[str, ...] = ("architect_planner",)
    child_roles: tuple[str, ...] = ()
    max_concurrency: int = 1
    max_children: int = 1
    max_depth: int = 1
    max_child_turns: int = 8
    max_total_cost_share: float = 0.25
    max_child_cost_share: float = 0.10
    max_child_context_bytes: int = 262_144
    max_child_output_bytes: int = 32_768
    max_joined_output_bytes: int = 65_536
    child_capability: Literal["read_only"] = "read_only"
    project_instruction_policy: Literal["suppress"] = "suppress"
    cancellation_policy: Literal["drain"] = "drain"
    approval_policy: Literal["deny"] = "deny"
    delivery_mode: Literal["parent_synthesis"] = "parent_synthesis"
    fail_fast: bool = True
    schema_version: str = ORCHESTRATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ORCHESTRATION_POLICY_VERSION:
            raise ValueError(
                "unsupported orchestration policy schema version "
                f"{self.schema_version!r}"
            )
        if (
            type(self.mode) is not str
            or self.mode not in {"off", "read_only", "isolated_writer"}
        ):
            raise ValueError(f"unsupported orchestration mode {self.mode!r}")
        closed_fields = {
            "child_capability": "read_only",
            "project_instruction_policy": "suppress",
            "cancellation_policy": "drain",
            "approval_policy": "deny",
            "delivery_mode": "parent_synthesis",
        }
        for field_name, expected in closed_fields.items():
            value = getattr(self, field_name)
            if type(value) is not str or value != expected:
                raise ValueError(
                    f"OrchestrationPolicy.{field_name} must be {expected!r}"
                )
        for field_name in (
            "parent_roles",
            "child_roles",
        ):
            values = getattr(self, field_name)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(
                    f"OrchestrationPolicy.{field_name} entries must be "
                    "non-empty strings"
                )
            if len(values) != len(set(values)):
                raise ValueError(
                    f"OrchestrationPolicy.{field_name} must not contain duplicates"
                )
        if not self.parent_roles:
            raise ValueError("OrchestrationPolicy.parent_roles must not be empty")
        for field_name in (
            "max_concurrency",
            "max_children",
            "max_depth",
            "max_child_turns",
            "max_child_context_bytes",
            "max_child_output_bytes",
            "max_joined_output_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"OrchestrationPolicy.{field_name} must be an integer >= 1"
                )
        if self.max_concurrency > self.max_children:
            raise ValueError(
                "OrchestrationPolicy.max_concurrency cannot exceed max_children"
            )
        if self.max_depth != 1:
            raise ValueError(
                "ORC-01 supports only max_depth=1; native/nested delegation "
                "is not qualified"
            )
        for field_name in (
            "max_total_cost_share",
            "max_child_cost_share",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
                or value > 1.0
            ):
                raise ValueError(
                    f"OrchestrationPolicy.{field_name} must be finite in (0, 1]"
                )
        if self.max_child_cost_share > self.max_total_cost_share:
            raise ValueError(
                "OrchestrationPolicy.max_child_cost_share cannot exceed "
                "max_total_cost_share"
            )
        if type(self.fail_fast) is not bool:
            raise TypeError("OrchestrationPolicy.fail_fast must be bool")
        if self.mode == "read_only" and not self.child_roles:
            raise ValueError(
                "read_only orchestration requires at least one named child role"
            )
        if self.mode == "off" and self.child_roles:
            raise ValueError("off orchestration cannot declare active child roles")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def digest(self) -> str:
        """Return the secret-free immutable policy identity."""

        return hash_record(asdict(self))


@dataclass(frozen=True, slots=True)
class WriterLanePolicy:
    """Fail-closed limits for backend-neutral isolated writer lanes.

    The policy grants no landing authority.  A successful scheduler run
    returns one gate-qualified aggregate patch while every child and
    integration worktree has already been restored and removed.  The owning
    workflow remains responsible for applying that patch under its normal
    mutation journal, validation, budget, and Corpus commit boundary.
    """

    mode: WriterLaneMode = "off"
    eligible_roles: tuple[str, ...] = ()
    max_concurrency: int = 1
    max_lanes: int = 1
    max_turns: int = 16
    max_total_cost_share: float = 0.50
    max_lane_cost_share: float = 0.25
    max_context_bytes: int = 262_144
    max_lane_output_bytes: int = 32_768
    max_lane_patch_bytes: int = 1_048_576
    max_integrated_patch_bytes: int = 2_097_152
    require_disjoint_scopes: bool = True
    require_nonempty_patch: bool = True
    require_union_gate: bool = True
    native_shell: Literal[False] = False
    native_delegation: Literal["off"] = "off"
    project_instruction_policy: Literal["suppress"] = "suppress"
    approval_policy: Literal["per_call_exact"] = "per_call_exact"
    cancellation_policy: Literal["cancel_drain_join"] = "cancel_drain_join"
    integration_order: Literal["declaration"] = "declaration"
    base_policy: Literal["exact_clean_commit"] = "exact_clean_commit"
    landing_mode: Literal["return_patch"] = "return_patch"
    fail_fast: bool = True
    schema_version: str = WRITER_LANE_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WRITER_LANE_POLICY_VERSION:
            raise ValueError(
                "unsupported writer lane policy schema version "
                f"{self.schema_version!r}"
            )
        if (
            type(self.mode) is not str
            or self.mode not in {"off", "isolated_writer"}
        ):
            raise ValueError(f"unsupported writer lane mode {self.mode!r}")
        if any(
            not isinstance(role, str) or not role.strip()
            for role in self.eligible_roles
        ):
            raise ValueError(
                "WriterLanePolicy.eligible_roles entries must be non-empty strings"
            )
        if len(self.eligible_roles) != len(set(self.eligible_roles)):
            raise ValueError(
                "WriterLanePolicy.eligible_roles must not contain duplicates"
            )
        for field_name in (
            "max_concurrency",
            "max_lanes",
            "max_turns",
            "max_context_bytes",
            "max_lane_output_bytes",
            "max_lane_patch_bytes",
            "max_integrated_patch_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"WriterLanePolicy.{field_name} must be an integer >= 1"
                )
        if self.max_concurrency > self.max_lanes:
            raise ValueError(
                "WriterLanePolicy.max_concurrency cannot exceed max_lanes"
            )
        for field_name in ("max_total_cost_share", "max_lane_cost_share"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
                or value > 1.0
            ):
                raise ValueError(
                    f"WriterLanePolicy.{field_name} must be finite in (0, 1]"
                )
        if self.max_lane_cost_share > self.max_total_cost_share:
            raise ValueError(
                "WriterLanePolicy.max_lane_cost_share cannot exceed "
                "max_total_cost_share"
            )
        for field_name in (
            "require_disjoint_scopes",
            "require_nonempty_patch",
            "require_union_gate",
            "fail_fast",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"WriterLanePolicy.{field_name} must be bool")
        if self.native_shell is not False:
            raise ValueError("writer lanes require native_shell=False")
        closed_fields = {
            "native_delegation": "off",
            "project_instruction_policy": "suppress",
            "approval_policy": "per_call_exact",
            "cancellation_policy": "cancel_drain_join",
            "integration_order": "declaration",
            "base_policy": "exact_clean_commit",
            "landing_mode": "return_patch",
        }
        for field_name, expected in closed_fields.items():
            value = getattr(self, field_name)
            if type(value) is not str or value != expected:
                raise ValueError(
                    f"WriterLanePolicy.{field_name} must be {expected!r}"
                )
        if self.mode == "isolated_writer" and not self.eligible_roles:
            raise ValueError(
                "isolated writer mode requires at least one eligible route role"
            )
        if self.mode == "off" and self.eligible_roles:
            raise ValueError("off writer mode cannot declare eligible roles")

    @property
    def enabled(self) -> bool:
        return self.mode == "isolated_writer"

    @property
    def digest(self) -> str:
        return hash_record(asdict(self))


__all__ = [
    "ORCHESTRATION_POLICY_VERSION",
    "OrchestrationMode",
    "OrchestrationPolicy",
    "WRITER_LANE_POLICY_VERSION",
    "WriterLaneMode",
    "WriterLanePolicy",
]
