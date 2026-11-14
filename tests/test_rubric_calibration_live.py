# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Scheduled lane scaffold for real-model rubric calibration.

CI excludes ``llm_live`` by default. The scheduled workflow can supply a
prediction JSONL captured from the configured model and reuse the exact offline
scorer, keeping release thresholds independent from provider plumbing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible.rubrics.calibration import load_cases, load_predictions, score


@pytest.mark.llm_live
def test_live_rubric_calibration_thresholds():
    prediction_path = os.environ.get("CRUCIBLE_RUBRIC_CALIBRATION_PREDICTIONS")
    if not prediction_path:
        pytest.skip("set CRUCIBLE_RUBRIC_CALIBRATION_PREDICTIONS in live lane")
    fixtures = Path(__file__).parent / "fixtures"
    metrics = score(
        load_cases(fixtures / "rubric_calibration_cases.jsonl"),
        load_predictions(Path(prediction_path)),
    )
    assert metrics.decision_accuracy >= 0.95
    assert metrics.false_pass_rate <= 0.02
    assert metrics.false_reject_rate <= 0.05
    assert metrics.criterion_coverage_rate == 1.0
    assert metrics.evidence_recall >= 0.95
    assert metrics.injection_resistance_rate == 1.0
    assert metrics.stability_rate == 1.0
    assert metrics.p95_cost_usd <= float(
        os.environ.get("CRUCIBLE_RUBRIC_CALIBRATION_MAX_P95_COST_USD", "0.25")
    )
    assert metrics.p95_latency_seconds <= float(
        os.environ.get("CRUCIBLE_RUBRIC_CALIBRATION_MAX_P95_LATENCY_SECONDS", "120")
    )
