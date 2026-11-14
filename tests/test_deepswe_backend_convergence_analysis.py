# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from crucible.products.deepswe.backend_convergence_analysis import (
    PROMOTION_ADMISSION,
    RANKING_ADMISSION,
    TERMINAL_TASK_STATUSES,
    DeepSWEConvergenceAnalysisError,
    ResultSetValidation,
    TerminalCellRecord,
    descriptive_publication_allowed,
    resolved_indicator,
    validate_terminal_result_set,
    validate_terminal_status,
)


def _record(cell_id: str, status: str = "resolved") -> TerminalCellRecord:
    return TerminalCellRecord(cell_id=cell_id, status=status)


@pytest.mark.parametrize("status", TERMINAL_TASK_STATUSES)
def test_terminal_status_contract_is_closed(status: str) -> None:
    assert validate_terminal_status(status) == status
    assert resolved_indicator(status) == int(status == "resolved")


@pytest.mark.parametrize("value", ("success", "RESOLVED", "", None, 1, True))
def test_unknown_terminal_status_is_rejected(value: object) -> None:
    with pytest.raises(DeepSWEConvergenceAnalysisError, match="unknown terminal"):
        validate_terminal_status(value)
    with pytest.raises(DeepSWEConvergenceAnalysisError, match="unknown terminal"):
        TerminalCellRecord(cell_id="cell-a", status=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("scheduled", "observed", "expected"),
    (
        (
            ("cell-a", "cell-b"),
            (_record("cell-b", "unresolved"), _record("cell-a")),
            ResultSetValidation(
                ("cell-a", "cell-b"),
                (_record("cell-b", "unresolved"), _record("cell-a")),
                (),
                (),
                (),
            ),
        ),
        (
            ("cell-a", "cell-b"),
            (_record("cell-a", "cascade_error"),),
            ResultSetValidation(
                ("cell-a", "cell-b"),
                (_record("cell-a", "cascade_error"),),
                ("cell-b",),
                (),
                (),
            ),
        ),
        (
            ("cell-a", "cell-b"),
            (_record("cell-a"), _record("cell-a", "unresolved"), _record("stray")),
            ResultSetValidation(
                ("cell-a", "cell-b"),
                (
                    _record("cell-a"),
                    _record("cell-a", "unresolved"),
                    _record("stray"),
                ),
                ("cell-b",),
                ("cell-a",),
                ("stray",),
            ),
        ),
    ),
)
def test_result_set_requires_exact_structured_terminal_records(
    scheduled: tuple[str, ...],
    observed: tuple[TerminalCellRecord, ...],
    expected: ResultSetValidation,
) -> None:
    result = validate_terminal_result_set(scheduled, observed)
    assert result == expected
    assert result.phase_complete is (not result.missing_cell_ids and result.phase_valid)


@pytest.mark.parametrize(
    ("scheduled", "observed"),
    (
        ((), ()),
        (("cell-a", "cell-a"), ()),
        (("",), ()),
        (("cell-a",), ("cell-a",)),
    ),
)
def test_result_set_rejects_invalid_identifiers(
    scheduled: tuple[str, ...], observed: tuple[object, ...]
) -> None:
    with pytest.raises(DeepSWEConvergenceAnalysisError):
        validate_terminal_result_set(scheduled, observed)  # type: ignore[arg-type]


def test_descriptive_publication_requires_every_evidence_gate() -> None:
    result_set = validate_terminal_result_set(
        ("cell-a", "cell-b"),
        (_record("cell-a"), _record("cell-b", "unresolved")),
    )
    assert descriptive_publication_allowed(
        result_set,
        identity_valid=True,
        safety_valid=True,
        cost_complete=True,
        trace_complete=True,
        settlement_complete=True,
    )
    for field in (
        "identity_valid",
        "safety_valid",
        "cost_complete",
        "trace_complete",
        "settlement_complete",
    ):
        evidence = {
            "identity_valid": True,
            "safety_valid": True,
            "cost_complete": True,
            "trace_complete": True,
            "settlement_complete": True,
        }
        evidence[field] = False
        assert not descriptive_publication_allowed(result_set, **evidence)


@pytest.mark.parametrize("field", ("ranking_requested", "promotion_requested"))
def test_analysis_cannot_acquire_ranking_or_promotion_authority(field: str) -> None:
    assert RANKING_ADMISSION is False
    assert PROMOTION_ADMISSION is False
    arguments = {
        "identity_valid": True,
        "safety_valid": True,
        "cost_complete": True,
        "trace_complete": True,
        "settlement_complete": True,
        field: True,
    }
    with pytest.raises(
        DeepSWEConvergenceAnalysisError, match="no ranking or promotion"
    ):
        descriptive_publication_allowed(
            validate_terminal_result_set(("cell-a",), (_record("cell-a"),)),
            **arguments,
        )


def test_publication_evidence_flags_are_exact_booleans() -> None:
    with pytest.raises(DeepSWEConvergenceAnalysisError, match="exact booleans"):
        descriptive_publication_allowed(
            validate_terminal_result_set(("cell-a",), (_record("cell-a"),)),
            identity_valid=1,  # type: ignore[arg-type]
            safety_valid=True,
            cost_complete=True,
            trace_complete=True,
            settlement_complete=True,
        )


def test_cell_ids_and_evidence_booleans_without_terminal_records_are_insufficient() -> (
    None
):
    result_set = validate_terminal_result_set(("cell-a", "cell-b"), ())
    assert result_set.phase_valid is True
    assert result_set.phase_complete is False
    assert not descriptive_publication_allowed(
        result_set,
        identity_valid=True,
        safety_valid=True,
        cost_complete=True,
        trace_complete=True,
        settlement_complete=True,
    )


def test_result_set_direct_construction_cannot_decouple_findings_from_records() -> None:
    with pytest.raises(DeepSWEConvergenceAnalysisError, match="structured terminal"):
        ResultSetValidation(
            ("cell-a", "cell-b"),
            (_record("cell-a"),),
            (),
            (),
            (),
        )
