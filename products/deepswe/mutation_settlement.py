# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE-owned Architect mutation-settlement checkpoint extension.

The generic Architect workflow exposes lifecycle coordinates through an opaque
checkpoint hook.  This module owns every DeepSWE-specific interpretation of
those coordinates: task identity, patch bounds, mutation-journal attestation,
and the stronger clean-workspace proof required when planning exits before any
tier runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.workflows.tier import attest_mutation_journal_settlement

if TYPE_CHECKING:
    from crucible.workflows.architect import ArchitectCheckpoint


DEEPSWE_CHECKPOINT_EXTENSION_KEY = (
    "crucible.products.deepswe.mutation-settlement.v1"
)
DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA = (
    "crucible.deepswe.architect-checkpoint.v1"
)
DEEPSWE_MUTATION_SETTLEMENT_SCHEMA = (
    "crucible.deepswe.mutation-settlement.v1"
)
TIER_MUTATION_SETTLEMENT_SCHEMA = (
    "crucible.tier-mutation-settlement.v1"
)
_EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()


def _failed_attestation(
    *,
    tier_id: str,
    operation_id: str,
    task_id: str,
    ext_id: str,
    exc: Exception,
) -> dict[str, Any]:
    """Serialize one explicit non-proof without leaking into Architect."""

    return {
        # This object occupies an individual Tier-attestation slot inside the
        # DeepSWE aggregate.  Keep its schema aligned with successful proofs
        # emitted by ``attest_mutation_journal_settlement``; ``settled=False``
        # remains the explicit fail-closed discriminator.
        "schema_version": TIER_MUTATION_SETTLEMENT_SCHEMA,
        "settled": False,
        "tier_id": tier_id,
        "operation_id": operation_id,
        "task_id": task_id,
        "ext_id": ext_id,
        "error_kind": type(exc).__name__,
        "error_message": str(exc)[:2000],
    }


@dataclass(slots=True)
class DeepSWEArchitectCheckpointHook:
    """Collect Tier attestations and publish one namespaced final proof."""

    task_id: str
    ext_id: str
    expected_base_commit: str
    max_patch_bytes: int
    _tier_attestations: list[dict[str, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _finalized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("task_id", "ext_id", "expected_base_commit"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"DeepSWE checkpoint hook requires non-empty {field_name}"
                )
        if (
            isinstance(self.max_patch_bytes, bool)
            or not isinstance(self.max_patch_bytes, int)
            or self.max_patch_bytes <= 0
        ):
            raise ValueError(
                "DeepSWE checkpoint hook requires max_patch_bytes > 0"
            )

    def _attest(
        self,
        checkpoint: ArchitectCheckpoint,
    ) -> dict[str, Any]:
        try:
            return attest_mutation_journal_settlement(
                corpus=checkpoint.corpus,
                cwd=checkpoint.cwd,
                tier_id=checkpoint.tier_id,
                operation_id=checkpoint.operation_id,
                task_id=self.task_id,
                ext_id=self.ext_id,
                expected_base_commit=self.expected_base_commit,
                tier_run_dir=checkpoint.child_run_dir,
                max_patch_bytes=self.max_patch_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - persist an explicit non-proof
            return _failed_attestation(
                tier_id=checkpoint.tier_id,
                operation_id=checkpoint.operation_id,
                task_id=self.task_id,
                ext_id=self.ext_id,
                exc=exc,
            )

    @staticmethod
    def _require_clean_zero_tier(
        attestation: dict[str, Any],
    ) -> None:
        patch_bytes = attestation.get("workspace_patch_bytes")
        if (
            attestation.get("settled") is not True
            or attestation.get("workspace_patch_sha256")
            != _EMPTY_PATCH_SHA256
            or isinstance(patch_bytes, bool)
            or patch_bytes != 0
            or attestation.get("workspace_files_modified") != []
        ):
            raise ValueError(
                "zero-tier DeepSWE settlement requires an unchanged workspace"
            )

    def __call__(
        self,
        checkpoint: ArchitectCheckpoint,
    ) -> dict[str, Any] | None:
        if checkpoint.kind == "tier":
            if self._finalized:
                raise RuntimeError(
                    "DeepSWE checkpoint hook received a tier after finalization"
                )
            self._tier_attestations.append(self._attest(checkpoint))
            return None

        if checkpoint.kind != "final":
            raise ValueError(
                f"unsupported Architect checkpoint kind {checkpoint.kind!r}"
            )
        if self._finalized:
            raise RuntimeError("DeepSWE checkpoint hook finalized more than once")
        self._finalized = True

        final_attestation = self._attest(checkpoint)
        if (
            checkpoint.tiers_executed == 0
            and final_attestation.get("settled") is True
        ):
            try:
                self._require_clean_zero_tier(final_attestation)
            except Exception as exc:  # noqa: BLE001 - explicit product non-proof
                final_attestation = _failed_attestation(
                    tier_id=checkpoint.tier_id,
                    operation_id=checkpoint.operation_id,
                    task_id=self.task_id,
                    ext_id=self.ext_id,
                    exc=exc,
                )

        return {
            DEEPSWE_CHECKPOINT_EXTENSION_KEY: {
                "schema_version": DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA,
                "mutation_settlement": {
                    "schema_version": DEEPSWE_MUTATION_SETTLEMENT_SCHEMA,
                    "tiers_executed": checkpoint.tiers_executed,
                    "tier_attestations": list(self._tier_attestations),
                    "final_attestation": final_attestation,
                },
            },
        }


__all__ = [
    "DEEPSWE_CHECKPOINT_EXTENSION_KEY",
    "DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA",
    "DEEPSWE_MUTATION_SETTLEMENT_SCHEMA",
    "TIER_MUTATION_SETTLEMENT_SCHEMA",
    "DeepSWEArchitectCheckpointHook",
]
