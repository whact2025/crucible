# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Exact, provider-free analysis primitives for the v1 backend matrix.

The planner binds these source bytes into every plan.  This module performs
no I/O and accepts only the closed terminal status vocabulary emitted by the
SWE-bench product.  A future result consumer can therefore share the primary,
failure-funnel, stability, and bootstrap semantics without reinterpreting
free-form labels.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from types import MappingProxyType


ANALYSIS_SCHEMA = "crucible.swebench-backend-convergence-analysis.v1"
ANALYSIS_SEED = "swebench-astra-fable51-analysis-20260912-v1"
BOOTSTRAP_RESAMPLES = 10_000

TERMINAL_EVAL_STATUSES = (
    "resolved",
    "unresolved",
    "patch_apply_failed",
    "harness_failed",
    "skipped_pristine_patch_apply_failed",
    "skipped_pristine_test_failed",
    "skipped_pristine_infra_failed",
    "cascade_aborted_partial",
    "skipped_empty",
    "skipped_disabled",
    "skipped_setup_error",
    "skipped_cascade_error",
)

FAILURE_FUNNEL_BY_STATUS: Mapping[str, str] = MappingProxyType(
    {
        "resolved": "resolved",
        "unresolved": "unresolved",
        "patch_apply_failed": "patch_apply_failed",
        "harness_failed": "harness_failed",
        "skipped_pristine_patch_apply_failed": "pristine_patch_apply_failed",
        "skipped_pristine_test_failed": "pristine_test_failed",
        "skipped_pristine_infra_failed": "pristine_infra_failed",
        "cascade_aborted_partial": "cascade_aborted_partial",
        "skipped_empty": "no_eligible_patch",
        "skipped_disabled": "evaluation_disabled",
        "skipped_setup_error": "setup_error",
        "skipped_cascade_error": "cascade_error",
    }
)


class BackendConvergenceAnalysisError(ValueError):
    """Raised when result evidence is outside the closed v1 contract."""


@dataclasses.dataclass(frozen=True)
class ResultSetValidation:
    """Bijection check between scheduled cells and observed terminal records."""

    phase_complete: bool
    phase_valid: bool
    missing_cell_ids: tuple[str, ...]
    duplicate_cell_ids: tuple[str, ...]
    unplanned_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.phase_complete) is not bool or type(self.phase_valid) is not bool:
            raise BackendConvergenceAnalysisError(
                "result-set disposition fields must be exact booleans"
            )
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
            raise BackendConvergenceAnalysisError(
                "result-set finding ids must be unique non-empty string tuples"
            )
        all_findings = tuple(cell_id for group in groups for cell_id in group)
        if len(set(all_findings)) != len(all_findings):
            raise BackendConvergenceAnalysisError(
                "result-set finding categories must be disjoint"
            )
        expected_valid = not self.duplicate_cell_ids and not self.unplanned_cell_ids
        expected_complete = expected_valid and not self.missing_cell_ids
        if (
            self.phase_valid != expected_valid
            or self.phase_complete != expected_complete
        ):
            raise BackendConvergenceAnalysisError(
                "result-set disposition contradicts its exact findings"
            )


def _terminal_status(value: object) -> str:
    if type(value) is not str or value not in TERMINAL_EVAL_STATUSES:
        raise BackendConvergenceAnalysisError(
            f"unknown terminal SWE-bench eval_status: {value!r}"
        )
    return value


def resolved_indicator(eval_status: str) -> int:
    """Return one only for exact official-harness ``resolved`` status."""

    return int(_terminal_status(eval_status) == "resolved")


def paired_resolution_delta(arm_a_status: str, arm_b_status: str) -> int:
    """Return the primary paired difference, Arm A minus Arm B."""

    return resolved_indicator(arm_a_status) - resolved_indicator(arm_b_status)


def primary_paired_resolution_delta(
    status_pairs: Iterable[tuple[str, str]],
) -> Fraction:
    """Return the equal-task-weight mean of replicate-1 A-minus-B pairs."""

    pairs = tuple(status_pairs)
    if not pairs:
        raise BackendConvergenceAnalysisError(
            "primary analysis requires at least one paired task"
        )
    return Fraction(
        sum(paired_resolution_delta(arm_a, arm_b) for arm_a, arm_b in pairs),
        len(pairs),
    )


def failure_funnel_category(eval_status: str) -> str:
    """Map each accepted terminal status to exactly one funnel category."""

    return FAILURE_FUNNEL_BY_STATUS[_terminal_status(eval_status)]


def within_task_replicate_agreement(eval_statuses: Sequence[str]) -> Fraction:
    """Return binary-resolution agreement across all replicate pairs of one task."""

    statuses = tuple(eval_statuses)
    if len(statuses) < 2:
        raise BackendConvergenceAnalysisError(
            "replicate agreement requires at least two terminal statuses"
        )
    indicators = tuple(resolved_indicator(status) for status in statuses)
    agreements = 0
    comparisons = 0
    for left_index, left in enumerate(indicators):
        for right in indicators[left_index + 1 :]:
            agreements += int(left == right)
            comparisons += 1
    return Fraction(agreements, comparisons)


def validate_result_cell_ids(
    scheduled_cell_ids: Sequence[str],
    observed_cell_ids: Sequence[str],
) -> ResultSetValidation:
    """Validate the exact-one terminal-record policy for scheduled cells."""

    scheduled = tuple(scheduled_cell_ids)
    observed = tuple(observed_cell_ids)
    if not scheduled or any(
        type(cell_id) is not str or not cell_id for cell_id in scheduled
    ):
        raise BackendConvergenceAnalysisError(
            "scheduled cell ids must be non-empty strings"
        )
    if len(set(scheduled)) != len(scheduled):
        raise BackendConvergenceAnalysisError("scheduled cell ids must be unique")
    if any(type(cell_id) is not str or not cell_id for cell_id in observed):
        raise BackendConvergenceAnalysisError(
            "observed cell ids must be non-empty strings"
        )

    scheduled_set = set(scheduled)
    observed_counts = Counter(observed)
    missing = tuple(cell_id for cell_id in scheduled if cell_id not in observed_counts)
    duplicates = tuple(cell_id for cell_id in scheduled if observed_counts[cell_id] > 1)
    unplanned = tuple(
        dict.fromkeys(cell_id for cell_id in observed if cell_id not in scheduled_set)
    )
    phase_valid = not duplicates and not unplanned
    return ResultSetValidation(
        phase_complete=phase_valid and not missing,
        phase_valid=phase_valid,
        missing_cell_ids=missing,
        duplicate_cell_ids=duplicates,
        unplanned_cell_ids=unplanned,
    )


def primary_publication_allowed(
    result_set: ResultSetValidation,
    *,
    identity_valid: bool,
    safety_valid: bool,
    cost_complete: bool,
    trace_complete: bool,
) -> bool:
    """Apply the v1 complete-and-valid phase publication gate."""

    if type(result_set) is not ResultSetValidation:
        raise BackendConvergenceAnalysisError(
            "result set must use the exact ResultSetValidation contract"
        )
    flags = (identity_valid, safety_valid, cost_complete, trace_complete)
    if any(type(flag) is not bool for flag in flags):
        raise BackendConvergenceAnalysisError(
            "primary publication evidence flags must be exact booleans"
        )
    return bool(
        result_set.phase_complete
        and result_set.phase_valid
        and identity_valid
        and safety_valid
        and cost_complete
        and trace_complete
    )


def bootstrap_task_index(
    *,
    seed: str,
    sample_index: int,
    stratum: str,
    draw_index: int,
    stratum_size: int,
) -> int:
    """Derive one deterministic within-stratum bootstrap task index."""

    if type(seed) is not str or not seed:
        raise BackendConvergenceAnalysisError("bootstrap seed must be non-empty text")
    if type(stratum) is not str or not stratum:
        raise BackendConvergenceAnalysisError(
            "bootstrap stratum must be non-empty text"
        )
    if type(sample_index) is not int or sample_index < 0:
        raise BackendConvergenceAnalysisError(
            "bootstrap sample index must be a non-negative integer"
        )
    if type(draw_index) is not int or draw_index < 0:
        raise BackendConvergenceAnalysisError(
            "bootstrap draw index must be a non-negative integer"
        )
    if type(stratum_size) is not int or stratum_size <= 0:
        raise BackendConvergenceAnalysisError(
            "bootstrap stratum size must be a positive integer"
        )
    material = f"{seed}\0{sample_index}\0{stratum}\0{draw_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big") % stratum_size


def paired_bootstrap_interval(
    stratum_differences: Mapping[str, Sequence[int]],
    *,
    seed: str = ANALYSIS_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[Fraction, Fraction]:
    """Return the exact v1 95% paired percentile interval.

    Values within each stratum must follow selected-task order and be Arm A
    minus Arm B replicate-1 resolution differences. Each resample retains the
    original stratum sizes and therefore the original selected-task weights.
    """

    if seed != ANALYSIS_SEED or resamples != BOOTSTRAP_RESAMPLES:
        raise BackendConvergenceAnalysisError(
            "v1 bootstrap requires the exact analysis seed and 10,000 resamples"
        )
    if type(stratum_differences) is not dict or not stratum_differences:
        raise BackendConvergenceAnalysisError(
            "bootstrap requires a non-empty exact stratum mapping"
        )

    strata: list[tuple[str, tuple[int, ...]]] = []
    for stratum, raw_values in stratum_differences.items():
        if type(stratum) is not str or not stratum:
            raise BackendConvergenceAnalysisError(
                "bootstrap stratum names must be non-empty text"
            )
        if type(raw_values) not in {list, tuple}:
            raise BackendConvergenceAnalysisError(
                "bootstrap differences must preserve selected-task sequence order"
            )
        values = tuple(raw_values)
        if not values or any(
            type(value) is not int or value not in {-1, 0, 1} for value in values
        ):
            raise BackendConvergenceAnalysisError(
                "bootstrap differences must be non-empty sequences of -1, 0, or 1"
            )
        strata.append((stratum, values))

    task_count = sum(len(values) for _, values in strata)
    estimates: list[Fraction] = []
    for sample_index in range(resamples):
        sampled_total = 0
        for stratum, values in strata:
            for draw_index in range(len(values)):
                sampled_total += values[
                    bootstrap_task_index(
                        seed=seed,
                        sample_index=sample_index,
                        stratum=stratum,
                        draw_index=draw_index,
                        stratum_size=len(values),
                    )
                ]
        estimates.append(Fraction(sampled_total, task_count))
    estimates.sort()
    return estimates[249], estimates[9_749]


__all__ = [
    "ANALYSIS_SCHEMA",
    "ANALYSIS_SEED",
    "BOOTSTRAP_RESAMPLES",
    "FAILURE_FUNNEL_BY_STATUS",
    "TERMINAL_EVAL_STATUSES",
    "BackendConvergenceAnalysisError",
    "ResultSetValidation",
    "bootstrap_task_index",
    "failure_funnel_category",
    "paired_bootstrap_interval",
    "paired_resolution_delta",
    "primary_paired_resolution_delta",
    "primary_publication_allowed",
    "resolved_indicator",
    "validate_result_cell_ids",
    "within_task_replicate_agreement",
]
