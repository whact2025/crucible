# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crucible-specific Constraint kinds registered into assay's discriminator map.

Per ``assay/docs/ASSAY_ARCHITECTURE.md`` §16.11, domain Constraint
subtypes are owned by crucible and register themselves into
``assay.canonical.registry.CONSTRAINT_TYPES`` at import time.

This module currently ships:

- ``LanguagesConstraint`` (kind ``"languages"``) — declares the
  language list a Spec touches. ``crucible.workflows._build_verify``
  reads it when wiring the polyglot runner, and per-slice override
  via ``ImplementationDirective.blast_radius.languages`` takes
  precedence when present.
- ``AuthoringInstructionConstraint`` (kind ``"authoring_instruction"``) —
  carries a product-owned, planner-visible boundary instruction without
  contaminating the issue intent.
- ``AuthoringFinalizationConstraint`` (kind ``"authoring_finalization"``) —
  binds one exact plan slice and exact output paths to a workflow-owned
  pre-capture runner. It is absent by default and grants no model authority.

Authoring example::

    constraints:
      - kind: languages
        languages: [python, rust]

Importing this module has the side effect of registering the kind.
``crucible.workflows`` does this transitively so any workflow
invocation sees the registered subtype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from assay.canonical.registry import CONSTRAINT_TYPES


@dataclass(frozen=True)
class LanguagesConstraint:
    """The set of programming languages a Spec's slices touch.

    Authored at the Spec level; per-slice overrides live in
    ``ImplementationDirective.blast_radius.languages``. The polyglot
    runner reads this list as the default when no slice override is
    set, and dispatches verify/build in declared order.

    ``kind`` is the discriminator; ``languages`` is the ordered tuple
    of language identifiers (e.g., ``"python"``, ``"rust"``, ``"cpp"``).
    Order matters: dispatch happens in this order, and the polyglot
    runner short-circuits on the first non-zero exit.
    """

    languages: tuple[str, ...] = ()
    kind: str = "languages"



@dataclass(frozen=True)
class AuthoringInstructionConstraint:
    """One planner-visible product authoring rule.

    The free-form domain content remains in Crucible; Assay only consumes the
    object through its generic ``Constraint`` protocol and renders ``str()``
    in the planner envelope.
    """

    instruction: str
    kind: str = "authoring_instruction"
    # Product-owned authoring policies need stable rule identity so a product
    # can upgrade only its own instructions while preserving operator-authored
    # constraints.  These fields are persistence-optional and canonically
    # omitted when empty: decoding and hashing legacy unowned constraints must
    # remain byte-for-byte compatible.
    policy_owner: str | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    policy_rule_id: str | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    policy_revision: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("authoring instruction must be non-empty")
        owner_present = self.policy_owner is not None
        rule_present = self.policy_rule_id is not None
        if owner_present != rule_present:
            raise ValueError(
                "authoring policy owner and rule id must be set together"
            )
        if owner_present:
            assert self.policy_owner is not None
            assert self.policy_rule_id is not None
            if not self.policy_owner.strip() or not self.policy_rule_id.strip():
                raise ValueError(
                    "authoring policy owner and rule id must be non-empty"
                )
            if self.policy_revision < 1:
                raise ValueError(
                    "managed authoring policy revision must be positive"
                )
        elif self.policy_revision != 0:
            raise ValueError(
                "unowned authoring instruction cannot carry a policy revision"
            )

    def __str__(self) -> str:
        return self.instruction



@dataclass(frozen=True)
class AuthoringFinalizationConstraint:
    """Opt one exact slice/output set into workflow-owned finalization.

    ``materializes_paths`` is the complete runner-write authority. Tier proves
    every entry belongs to the selected directive and rejects any runner-time
    mutation outside this tuple relative to the post-model workspace.
    """

    slice_id: str
    materializes_paths: tuple[str, ...]
    gate: str = "verify"
    kind: str = "authoring_finalization"

    def __post_init__(self) -> None:
        if not isinstance(self.slice_id, str) or not self.slice_id.strip():
            raise ValueError(
                "authoring finalization slice_id must be non-empty"
            )
        if (
            not isinstance(self.materializes_paths, (list, tuple))
            or not self.materializes_paths
            or any(
                not isinstance(path, str) or not path.strip()
                for path in self.materializes_paths
            )
        ):
            raise ValueError(
                "authoring finalization materializes_paths must be a "
                "non-empty tuple of non-empty paths"
            )
        object.__setattr__(
            self,
            "materializes_paths",
            tuple(self.materializes_paths),
        )
        if len(set(self.materializes_paths)) != len(self.materializes_paths):
            raise ValueError(
                "authoring finalization materializes_paths must be unique"
            )
        malformed = tuple(
            path
            for path in self.materializes_paths
            if (
                PurePosixPath(path).is_absolute()
                or "\\" in path
                or ".." in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path
                or path == "."
            )
        )
        if malformed:
            raise ValueError(
                "authoring finalization materializes_paths must be normalized "
                f"repository-relative paths: {list(malformed)!r}"
            )
        if self.gate not in {"verify", "dev_verify"}:
            raise ValueError(
                "authoring finalization gate must be 'verify' or "
                f"'dev_verify', got {self.gate!r}"
            )



# Register the current product-neutral kinds at import time.
CONSTRAINT_TYPES.setdefault("languages", LanguagesConstraint)
CONSTRAINT_TYPES.setdefault("authoring_instruction", AuthoringInstructionConstraint)
CONSTRAINT_TYPES.setdefault("authoring_finalization", AuthoringFinalizationConstraint)


__all__ = [
    "AuthoringFinalizationConstraint",
    "AuthoringInstructionConstraint",
    "LanguagesConstraint",
]
