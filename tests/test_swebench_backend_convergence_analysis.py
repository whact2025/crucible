# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from fractions import Fraction
from types import MappingProxyType

import pytest

from crucible.products.swebench.backend_convergence_analysis import (
    ANALYSIS_SEED,
    BOOTSTRAP_RESAMPLES,
    FAILURE_FUNNEL_BY_STATUS,
    TERMINAL_EVAL_STATUSES,
    BackendConvergenceAnalysisError,
    ResultSetValidation,
    bootstrap_task_index,
    failure_funnel_category,
    paired_bootstrap_interval,
    paired_resolution_delta,
    primary_paired_resolution_delta,
    primary_publication_allowed,
    resolved_indicator,
    validate_result_cell_ids,
    within_task_replicate_agreement,
)


_V1_ANALYSIS_SEED = "swebench-astra-fable51-analysis-20260912-v1"
_EXPECTED_FAILURE_FUNNEL = {
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


@pytest.mark.parametrize("status", tuple(_EXPECTED_FAILURE_FUNNEL))
def test_resolved_indicator_is_one_only_for_exact_resolved_status(status: str) -> None:
    assert resolved_indicator(status) == int(status == "resolved")


@pytest.mark.parametrize(
    ("arm_a_status", "arm_b_status", "expected"),
    (
        ("resolved", "unresolved", 1),
        ("unresolved", "resolved", -1),
        ("resolved", "resolved", 0),
        ("harness_failed", "skipped_empty", 0),
    ),
)
def test_paired_resolution_delta_is_arm_a_minus_arm_b(
    arm_a_status: str,
    arm_b_status: str,
    expected: int,
) -> None:
    assert paired_resolution_delta(arm_a_status, arm_b_status) == expected


def test_failure_funnel_is_an_exhaustive_closed_terminal_status_partition() -> None:
    assert TERMINAL_EVAL_STATUSES == tuple(_EXPECTED_FAILURE_FUNNEL)
    assert dict(FAILURE_FUNNEL_BY_STATUS) == _EXPECTED_FAILURE_FUNNEL
    assert {
        status: failure_funnel_category(status) for status in TERMINAL_EVAL_STATUSES
    } == _EXPECTED_FAILURE_FUNNEL


@pytest.mark.parametrize("unknown", ("success", "RESOLVED", "", None, 1, True))
def test_unknown_terminal_status_is_rejected_by_every_status_consumer(
    unknown: object,
) -> None:
    with pytest.raises(BackendConvergenceAnalysisError, match="unknown terminal"):
        resolved_indicator(unknown)  # type: ignore[arg-type]
    with pytest.raises(BackendConvergenceAnalysisError, match="unknown terminal"):
        failure_funnel_category(unknown)  # type: ignore[arg-type]
    with pytest.raises(BackendConvergenceAnalysisError, match="unknown terminal"):
        paired_resolution_delta("resolved", unknown)  # type: ignore[arg-type]
    with pytest.raises(BackendConvergenceAnalysisError, match="unknown terminal"):
        within_task_replicate_agreement(
            ("resolved", unknown)  # type: ignore[arg-type]
        )


def test_primary_delta_is_an_exact_equal_task_weight_fraction() -> None:
    status_pairs = iter(
        (
            ("resolved", "unresolved"),
            ("resolved", "skipped_empty"),
            ("harness_failed", "resolved"),
            ("unresolved", "patch_apply_failed"),
        )
    )

    result = primary_paired_resolution_delta(status_pairs)

    assert type(result) is Fraction
    assert result == Fraction(1, 4)


def test_primary_delta_rejects_an_empty_task_cohort() -> None:
    with pytest.raises(
        BackendConvergenceAnalysisError, match="at least one paired task"
    ):
        primary_paired_resolution_delta(())


def test_replicate_agreement_uses_every_binary_resolution_pair() -> None:
    # Indicators are (1, 1, 0, 0): two of the six unordered pairs agree.
    result = within_task_replicate_agreement(
        ("resolved", "resolved", "unresolved", "harness_failed")
    )

    assert type(result) is Fraction
    assert result == Fraction(2, 6)
    # Distinct non-resolved statuses still agree under the binary estimand.
    assert within_task_replicate_agreement(
        ("unresolved", "harness_failed", "skipped_empty")
    ) == Fraction(1, 1)


@pytest.mark.parametrize("statuses", ((), ("resolved",)))
def test_replicate_agreement_requires_at_least_two_replicates(
    statuses: tuple[str, ...],
) -> None:
    with pytest.raises(BackendConvergenceAnalysisError, match="at least two"):
        within_task_replicate_agreement(statuses)


@pytest.mark.parametrize(
    ("scheduled", "observed", "expected"),
    (
        (
            ("cell-a", "cell-b"),
            ("cell-b", "cell-a"),
            ResultSetValidation(True, True, (), (), ()),
        ),
        (
            ("cell-a", "cell-b", "cell-c"),
            ("cell-c",),
            ResultSetValidation(False, True, ("cell-a", "cell-b"), (), ()),
        ),
        (
            ("cell-a", "cell-b"),
            ("cell-b", "cell-b", "cell-a"),
            ResultSetValidation(False, False, (), ("cell-b",), ()),
        ),
        (
            ("cell-a", "cell-b"),
            ("cell-b", "stray", "stray", "cell-a"),
            ResultSetValidation(False, False, (), (), ("stray",)),
        ),
        (
            ("cell-b", "cell-a", "cell-c"),
            ("cell-a", "cell-a", "stray-z", "stray-y", "stray-z"),
            ResultSetValidation(
                False,
                False,
                ("cell-b", "cell-c"),
                ("cell-a",),
                ("stray-z", "stray-y"),
            ),
        ),
    ),
)
def test_result_cell_validation_reports_bijection_failures_deterministically(
    scheduled: tuple[str, ...],
    observed: tuple[str, ...],
    expected: ResultSetValidation,
) -> None:
    assert validate_result_cell_ids(scheduled, observed) == expected


@pytest.mark.parametrize(
    ("scheduled", "observed"),
    (
        ((), ()),
        (("cell-a", "cell-a"), ()),
        (("",), ()),
        (("cell-a",), ("",)),
    ),
)
def test_result_cell_validation_rejects_invalid_id_inputs(
    scheduled: tuple[str, ...],
    observed: tuple[str, ...],
) -> None:
    with pytest.raises(BackendConvergenceAnalysisError):
        validate_result_cell_ids(scheduled, observed)


@pytest.mark.parametrize(
    "arguments",
    (
        (True, True, ("missing",), (), ()),
        (False, True, (), (), ()),
        (False, False, (), (), ()),
        (False, True, (), ("duplicate",), ()),
    ),
)
def test_result_set_validation_rejects_contradictory_direct_construction(
    arguments: tuple[bool, bool, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> None:
    with pytest.raises(BackendConvergenceAnalysisError, match="contradicts"):
        ResultSetValidation(*arguments)


@pytest.mark.parametrize(
    "arguments",
    (
        (1, True, (), (), ()),
        (True, True, [], (), ()),
        (False, True, ("",), (), ()),
        (False, False, (), ("cell-a", "cell-a"), ()),
    ),
)
def test_result_set_validation_rejects_malformed_direct_construction(
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises(BackendConvergenceAnalysisError):
        ResultSetValidation(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("result_set", "evidence", "expected"),
    (
        (
            ResultSetValidation(True, True, (), (), ()),
            (True, True, True, True),
            True,
        ),
        (
            ResultSetValidation(False, True, ("cell-a",), (), ()),
            (True, True, True, True),
            False,
        ),
        (
            ResultSetValidation(False, False, (), ("cell-a",), ()),
            (True, True, True, True),
            False,
        ),
        (
            ResultSetValidation(True, True, (), (), ()),
            (False, True, True, True),
            False,
        ),
        (
            ResultSetValidation(True, True, (), (), ()),
            (True, False, True, True),
            False,
        ),
        (
            ResultSetValidation(True, True, (), (), ()),
            (True, True, False, True),
            False,
        ),
        (
            ResultSetValidation(True, True, (), (), ()),
            (True, True, True, False),
            False,
        ),
    ),
)
def test_primary_publication_requires_complete_valid_result_and_all_evidence(
    result_set: ResultSetValidation,
    evidence: tuple[bool, bool, bool, bool],
    expected: bool,
) -> None:
    identity_valid, safety_valid, cost_complete, trace_complete = evidence
    assert (
        primary_publication_allowed(
            result_set,
            identity_valid=identity_valid,
            safety_valid=safety_valid,
            cost_complete=cost_complete,
            trace_complete=trace_complete,
        )
        is expected
    )


@pytest.mark.parametrize(
    "flag_name",
    ("identity_valid", "safety_valid", "cost_complete", "trace_complete"),
)
def test_primary_publication_requires_exact_boolean_evidence(flag_name: str) -> None:
    evidence: dict[str, object] = {
        "identity_valid": True,
        "safety_valid": True,
        "cost_complete": True,
        "trace_complete": True,
    }
    evidence[flag_name] = 1

    with pytest.raises(BackendConvergenceAnalysisError, match="exact booleans"):
        primary_publication_allowed(
            ResultSetValidation(True, True, (), (), ()),
            **evidence,  # type: ignore[arg-type]
        )


def test_primary_publication_rejects_a_non_contract_result_set() -> None:
    with pytest.raises(
        BackendConvergenceAnalysisError, match="exact ResultSetValidation"
    ):
        primary_publication_allowed(
            object(),  # type: ignore[arg-type]
            identity_valid=True,
            safety_valid=True,
            cost_complete=True,
            trace_complete=True,
        )


@pytest.mark.parametrize(
    ("seed", "sample_index", "stratum", "draw_index", "stratum_size", "expected"),
    (
        (_V1_ANALYSIS_SEED, 0, "astropy", 0, 11, 9),
        (_V1_ANALYSIS_SEED, 0, "astropy", 10, 11, 10),
        (_V1_ANALYSIS_SEED, 249, "django", 3, 11, 9),
        (_V1_ANALYSIS_SEED, 9_749, "requests", 5, 6, 5),
        (_V1_ANALYSIS_SEED, 9_999, "sympy", 9, 10, 3),
        ("alternate-golden-seed", 42, "tiny", 1, 3, 2),
    ),
)
def test_bootstrap_task_index_matches_fixed_sha256_golden_vectors(
    seed: str,
    sample_index: int,
    stratum: str,
    draw_index: int,
    stratum_size: int,
    expected: int,
) -> None:
    assert (
        bootstrap_task_index(
            seed=seed,
            sample_index=sample_index,
            stratum=stratum,
            draw_index=draw_index,
            stratum_size=stratum_size,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("field", "invalid", "match"),
    (
        ("seed", "", "seed"),
        ("seed", None, "seed"),
        ("stratum", "", "stratum"),
        ("stratum", 1, "stratum"),
        ("sample_index", -1, "sample index"),
        ("sample_index", True, "sample index"),
        ("draw_index", -1, "draw index"),
        ("draw_index", 1.0, "draw index"),
        ("stratum_size", 0, "stratum size"),
        ("stratum_size", -1, "stratum size"),
        ("stratum_size", True, "stratum size"),
    ),
)
def test_bootstrap_task_index_rejects_invalid_inputs(
    field: str,
    invalid: object,
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "seed": _V1_ANALYSIS_SEED,
        "sample_index": 0,
        "stratum": "astropy",
        "draw_index": 0,
        "stratum_size": 11,
    }
    arguments[field] = invalid

    with pytest.raises(BackendConvergenceAnalysisError, match=match):
        bootstrap_task_index(**arguments)  # type: ignore[arg-type]


def test_paired_bootstrap_interval_requires_exact_v1_configuration() -> None:
    differences = {"tiny": (-1, 0, 1)}

    with pytest.raises(BackendConvergenceAnalysisError, match="exact analysis seed"):
        paired_bootstrap_interval(differences, seed="changed-seed")
    with pytest.raises(BackendConvergenceAnalysisError, match="10,000 resamples"):
        paired_bootstrap_interval(differences, resamples=9_999)


@pytest.mark.parametrize(
    "differences",
    (
        {},
        MappingProxyType({"tiny": (0,)}),
        {"": (0,)},
        {1: (0,)},
        {"tiny": ()},
        {"tiny": (2,)},
        {"tiny": (True,)},
        {"tiny": (0.0,)},
        {"tiny": {-1, 0, 1}},
        {"tiny": iter((-1, 0, 1))},
    ),
)
def test_paired_bootstrap_interval_rejects_invalid_strata_and_differences(
    differences: object,
) -> None:
    with pytest.raises(BackendConvergenceAnalysisError):
        paired_bootstrap_interval(differences)  # type: ignore[arg-type]


def test_paired_bootstrap_interval_matches_small_deterministic_golden_result() -> None:
    assert ANALYSIS_SEED == _V1_ANALYSIS_SEED
    assert BOOTSTRAP_RESAMPLES == 10_000
    differences = {
        "alpha": (-1, 1),
        "beta": (0, 1, -1),
    }

    assert paired_bootstrap_interval(differences) == (
        Fraction(-4, 5),
        Fraction(4, 5),
    )
