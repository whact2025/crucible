# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Closed, provider-free analysis gates for the DeepSWE quality pilot.

The quality-pilot matrix binds these exact source bytes.  This module does no
I/O and deliberately exposes no ranking or promotion decision.  It only
validates a scheduled-result bijection whose records carry closed terminal
statuses, plus the evidence closure required before descriptive diagnostic
summaries may be published. Cell identifiers without their terminal status
evidence can never satisfy the publication gate.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence


ANALYSIS_SCHEMA = "crucible.deepswe-backend-convergence-analysis.v1"
RANKING_ADMISSION = False
PROMOTION_ADMISSION = False

TERMINAL_TASK_STATUSES = (
    "candidate_failed",
    "cascade_error",
    "pristine_replay_failed",
    "resolved",
    "setup_failed",
    "storage_quota_exceeded",
    "toolchain_unavailable",
    "unresolved",
    "verification_failed",
    "verification_incomplete",
)


class DeepSWEConvergenceAnalysisError(ValueError):
    """Raised when result evidence is outside the closed diagnostic contract."""


def validate_terminal_status(value: object) -> str:
    """Return one exact DeepSWE lifecycle terminal status or reject it."""

    if type(value) is not str or value not in TERMINAL_TASK_STATUSES:
        raise DeepSWEConvergenceAnalysisError(
            f"unknown terminal DeepSWE task status: {value!r}"
        )
    return value


@dataclasses.dataclass(frozen=True)
class TerminalCellRecord:
    """One cell identity coupled to one closed terminal task status."""

    cell_id: str
    status: str

    def __post_init__(self) -> None:
        if type(self.cell_id) is not str or not self.cell_id:
            raise DeepSWEConvergenceAnalysisError(
                "terminal record cell_id must be non-empty text"
            )
        validate_terminal_status(self.status)


@dataclasses.dataclass(frozen=True)
class ResultSetValidation:
    """Validated terminal records and their exact scheduled-cell findings."""

    scheduled_cell_ids: tuple[str, ...]
    terminal_records: tuple[TerminalCellRecord, ...]
    missing_cell_ids: tuple[str, ...]
    duplicate_cell_ids: tuple[str, ...]
    unplanned_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        scheduled = self.scheduled_cell_ids
        records = self.terminal_records
        if (
            type(scheduled) is not tuple
            or not scheduled
            or any(type(cell_id) is not str or not cell_id for cell_id in scheduled)
            or len(set(scheduled)) != len(scheduled)
        ):
            raise DeepSWEConvergenceAnalysisError(
                "scheduled cell ids must be a non-empty unique string tuple"
            )
        if type(records) is not tuple or any(
            type(record) is not TerminalCellRecord for record in records
        ):
            raise DeepSWEConvergenceAnalysisError(
                "terminal records must be an exact TerminalCellRecord tuple"
            )
        for record in records:
            validate_terminal_status(record.status)
        groups = (
            self.missing_cell_ids,
            self.duplicate_cell_ids,
            self.unplanned_cell_ids,
        )
        if any(
            type(group) is not tuple
            or any(type(cell_id) is not str or not cell_id for cell_id in group)
            or len(set(group)) != len(group)
            for group in groups
        ):
            raise DeepSWEConvergenceAnalysisError(
                "result-set findings must be unique non-empty string tuples"
            )
        flattened = tuple(item for group in groups for item in group)
        if len(flattened) != len(set(flattened)):
            raise DeepSWEConvergenceAnalysisError(
                "result-set finding categories must be disjoint"
            )

        scheduled_set = set(scheduled)
        observed_ids = tuple(record.cell_id for record in records)
        observed_counts = Counter(observed_ids)
        expected_missing = tuple(
            cell_id for cell_id in scheduled if cell_id not in observed_counts
        )
        expected_duplicates = tuple(
            cell_id for cell_id in scheduled if observed_counts[cell_id] > 1
        )
        expected_unplanned = tuple(
            dict.fromkeys(
                cell_id for cell_id in observed_ids if cell_id not in scheduled_set
            )
        )
        if (
            self.missing_cell_ids != expected_missing
            or self.duplicate_cell_ids != expected_duplicates
            or self.unplanned_cell_ids != expected_unplanned
        ):
            raise DeepSWEConvergenceAnalysisError(
                "result-set findings contradict their structured terminal records"
            )

    @property
    def phase_valid(self) -> bool:
        return not self.duplicate_cell_ids and not self.unplanned_cell_ids

    @property
    def phase_complete(self) -> bool:
        return self.phase_valid and not self.missing_cell_ids


def resolved_indicator(status: str) -> int:
    """Return one only for the exact hidden-verifier ``resolved`` terminal."""

    return int(validate_terminal_status(status) == "resolved")


def validate_terminal_result_set(
    scheduled_cell_ids: Sequence[str],
    terminal_records: Sequence[TerminalCellRecord],
) -> ResultSetValidation:
    """Validate exactly one closed terminal-status record per planned cell."""

    scheduled = tuple(scheduled_cell_ids)
    records = tuple(terminal_records)
    if not scheduled or any(
        type(cell_id) is not str or not cell_id for cell_id in scheduled
    ):
        raise DeepSWEConvergenceAnalysisError(
            "scheduled cell ids must be non-empty strings"
        )
    if len(set(scheduled)) != len(scheduled):
        raise DeepSWEConvergenceAnalysisError("scheduled cell ids must be unique")
    if any(type(record) is not TerminalCellRecord for record in records):
        raise DeepSWEConvergenceAnalysisError(
            "terminal results must be structured TerminalCellRecord values"
        )

    scheduled_set = set(scheduled)
    observed_ids = tuple(record.cell_id for record in records)
    observed_counts = Counter(observed_ids)
    missing = tuple(cell_id for cell_id in scheduled if cell_id not in observed_counts)
    duplicates = tuple(cell_id for cell_id in scheduled if observed_counts[cell_id] > 1)
    unplanned = tuple(
        dict.fromkeys(
            cell_id for cell_id in observed_ids if cell_id not in scheduled_set
        )
    )
    return ResultSetValidation(
        scheduled_cell_ids=scheduled,
        terminal_records=records,
        missing_cell_ids=missing,
        duplicate_cell_ids=duplicates,
        unplanned_cell_ids=unplanned,
    )


def descriptive_publication_allowed(
    result_set: ResultSetValidation,
    *,
    identity_valid: bool,
    safety_valid: bool,
    cost_complete: bool,
    trace_complete: bool,
    settlement_complete: bool,
    ranking_requested: bool = False,
    promotion_requested: bool = False,
) -> bool:
    """Apply the complete-evidence, diagnostic-only publication gate."""

    if type(result_set) is not ResultSetValidation:
        raise DeepSWEConvergenceAnalysisError(
            "result set must use the exact ResultSetValidation contract"
        )
    rebuilt = validate_terminal_result_set(
        result_set.scheduled_cell_ids,
        result_set.terminal_records,
    )
    if rebuilt != result_set:
        raise DeepSWEConvergenceAnalysisError(
            "result set differs from its structured terminal evidence"
        )
    flags = (
        identity_valid,
        safety_valid,
        cost_complete,
        trace_complete,
        settlement_complete,
        ranking_requested,
        promotion_requested,
    )
    if any(type(flag) is not bool for flag in flags):
        raise DeepSWEConvergenceAnalysisError(
            "analysis evidence flags must be exact booleans"
        )
    if ranking_requested or promotion_requested:
        raise DeepSWEConvergenceAnalysisError(
            "the v1 quality pilot has no ranking or promotion authority"
        )
    return bool(
        result_set.phase_complete
        and result_set.phase_valid
        and identity_valid
        and safety_valid
        and cost_complete
        and trace_complete
        and settlement_complete
    )


__all__ = [
    "ANALYSIS_SCHEMA",
    "PROMOTION_ADMISSION",
    "RANKING_ADMISSION",
    "TERMINAL_TASK_STATUSES",
    "DeepSWEConvergenceAnalysisError",
    "ResultSetValidation",
    "TerminalCellRecord",
    "descriptive_publication_allowed",
    "resolved_indicator",
    "validate_terminal_result_set",
    "validate_terminal_status",
]
