# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.rubrics import available
from crucible.rubrics.calibration import (
    CalibrationRequest,
    generate_with_backend,
    load_cases,
    load_predictions,
    score,
    validate_inventory_coverage,
)


FIXTURES = Path(__file__).parent / "fixtures"
CASES = FIXTURES / "rubric_calibration_cases.jsonl"
PREDICTIONS = FIXTURES / "rubric_calibration_predictions.jsonl"


def test_offline_calibration_golden_corpus_is_inventory_wide_and_perfect():
    cases = load_cases(CASES)
    predictions = load_predictions(PREDICTIONS)

    assert {case.rubric for case in cases} == set(available(include_dormant=False))
    assert len(cases) == len(available(include_dormant=False)) + 8
    assert all(case.artifact_input.text for case in cases)
    assert all(case.input_digest.startswith("sha256:") for case in cases)
    assert all(case.prior_inputs for case in cases)

    metrics = score(cases, predictions)

    assert metrics.total == len(cases)
    assert metrics.pass_cases == len(cases) - 6
    assert metrics.nonpass_cases == 6
    assert metrics.decision_accuracy == 1.0
    assert metrics.false_pass_rate == 0.0
    assert metrics.false_reject_rate == 0.0
    assert metrics.criterion_coverage_rate == 1.0
    assert metrics.evidence_recall == 1.0
    assert metrics.injection_resistance_rate == 1.0
    assert metrics.stability_rate == 1.0
    assert metrics.mean_cost_usd > 0
    assert metrics.p95_cost_usd == pytest.approx(0.005)
    assert metrics.mean_latency_seconds > 0
    assert metrics.p95_latency_seconds == pytest.approx(0.16)


def test_case_loader_rejects_tampered_retained_source(tmp_path):
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines()]
    rows[0]["artifact_input"]["text"] += " tampered"
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sha256 does not match retained text"):
        load_cases(path)


def test_inventory_validator_rejects_an_unrepresented_active_rubric():
    cases = load_cases(CASES)
    omitted = cases[0].rubric
    subset = tuple(case for case in cases if case.rubric != omitted)

    with pytest.raises(ValueError, match="missing active rubrics"):
        validate_inventory_coverage(subset)


def test_calibration_detects_false_results_and_quality_regressions():
    cases = load_cases(CASES)
    predictions = list(load_predictions(PREDICTIONS))
    contradiction_index = next(
        index
        for index, item in enumerate(predictions)
        if item.case_id == "goal_judge-contradiction"
    )
    contradiction_case = next(
        item for item in cases if item.case_id == "goal_judge-contradiction"
    )
    predictions[contradiction_index] = dataclasses.replace(
        predictions[contradiction_index],
        decision="pass",
        criterion_ids=predictions[contradiction_index].criterion_ids[:-1],
        evidence_excerpts=(),
        rendered_output=contradiction_case.forbidden_output_tokens[0],
    )
    positive_index = next(
        index
        for index, item in enumerate(predictions)
        if item.case_id == "api_contract_fidelity-baseline"
    )
    predictions[positive_index] = dataclasses.replace(
        predictions[positive_index],
        decision="reject",
    )

    metrics = score(cases, predictions)

    assert metrics.false_pass_rate == pytest.approx(1 / metrics.nonpass_cases)
    assert metrics.false_reject_rate == pytest.approx(1 / metrics.pass_cases)
    assert metrics.criterion_coverage_rate < 1.0
    assert metrics.evidence_recall < 1.0
    assert metrics.injection_resistance_rate == pytest.approx(
        (len(cases) - 1) / len(cases)
    )


def test_calibration_contains_product_neutral_semantic_invention_cases():
    cases = {case.case_id: case for case in load_cases(CASES)}

    assert cases["plan_critic-core-ui-semantic-invention"].expected_decisions == (
        "revise",
    )
    assert cases["plan_critic-v12-out-of-domain-invention"].expected_decisions == (
        "revise",
    )
    assert "semantic-fidelity" in cases[
        "plan_critic-core-ui-semantic-invention"
    ].tags
    assert "semantic-fidelity" in cases[
        "plan_critic-v12-out-of-domain-invention"
    ].tags


def test_calibration_detects_unstable_replicates():
    cases = load_cases(CASES)
    predictions = [
        dataclasses.replace(item, decision="revise")
        if item.case_id == "goal_judge-stability-replicate"
        else item
        for item in load_predictions(PREDICTIONS)
    ]

    assert score(cases, predictions).stability_rate == 0.0


def test_score_rejects_stale_prediction_input_digest():
    cases = load_cases(CASES)
    predictions = list(load_predictions(PREDICTIONS))
    predictions[0] = dataclasses.replace(
        predictions[0],
        input_digest="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="generated for different inputs"):
        score(cases, predictions)


def test_calibration_rejects_partial_prediction_sets(tmp_path):
    path = tmp_path / "predictions.jsonl"
    first = PREDICTIONS.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cover the calibration cases exactly"):
        score(load_cases(CASES), load_predictions(path))


def test_provider_neutral_backend_generation_is_identity_checked(tmp_path):
    cases = load_cases(CASES)[:2]

    class FakeBackend:
        calls = 0

    backend = FakeBackend()

    def adapter(
        supplied_backend: object,
        request: CalibrationRequest,
    ) -> dict[str, Any]:
        assert supplied_backend is backend
        assert request.artifact
        assert request.prior_inputs
        backend.calls += 1
        case = next(item for item in cases if item.case_id == request.case_id)
        return {
            "schema_version": "crucible.rubric-calibration-prediction.v2",
            "case_id": request.case_id,
            "rubric_digest": request.rubric_digest,
            "input_digest": request.input_digest,
            "decision": case.expected_decisions[0],
            "criterion_ids": list(case.required_criteria),
            "evidence_excerpts": list(case.expected_evidence_tokens),
            "rendered_output": "fake provider result",
            "provider": "fake",
            "model": "fake-v1",
            "cost_usd": 0.0,
            "latency_seconds": 0.01,
        }

    output = tmp_path / "generated.jsonl"
    generated = generate_with_backend(
        cases,
        backend=backend,
        adapter=adapter,
        output_path=output,
    )

    assert backend.calls == len(cases)
    assert load_predictions(output) == generated
    assert score(cases, generated).decision_accuracy == 1.0
