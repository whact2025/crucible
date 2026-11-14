# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from crucible.products.swebench import backend_convergence as convergence
from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.products.swebench.backend_convergence import (
    BackendConvergenceMatrix,
    BackendConvergenceMatrixError,
    ConvergenceCell,
    build_backend_convergence_plan,
    load_backend_convergence_matrix,
    select_matrix_tasks,
)
from crucible.scripts import swebench_backend_convergence as cli


_ROOT = Path(__file__).parents[1]
_PRODUCT_ROOT = _ROOT / "products" / "swebench"
_MATRIX_PATH = _PRODUCT_ROOT / "backend_convergence_matrix.yaml"
_ROUTE_NAMES = (
    "backends_codex_astra.yaml",
    "backends_claude_fable51_org.yaml",
)
_ANALYSIS_MODULE_NAME = "backend_convergence_analysis.py"
_ANALYSIS_MODULE_DIGEST = (
    "sha256:4973d24d7537305350e9930fda269573a083030a6a77ed8b03b4d6b6d656e14d"
)
_EXPECTED_ANALYSIS_IDENTITY: dict[str, object] = {
    "comparison_unit": "paired-task-at-primary-replicate",
    "provider_sampling_seed": "unavailable",
    "primary_endpoint": (
        "paired-intention-to-treat-resolved-pass-at-1-arm-a-minus-arm-b"
    ),
    "primary_success_predicate": "eval-status-exactly-resolved",
    "primary_delta_direction": "arm-a-minus-arm-b",
    "primary_publication_rule": "complete-valid-phase-only",
    "primary_replicate": 1,
    "primary_task_weighting": "equal-per-selected-task",
    "primary_stratum_aggregation": "selected-task-count-weighted",
    "primary_cell_denominator": "all-replicate-1-cells",
    "secondary_attempt_denominator": "all-scheduled-cells",
    "result_set_policy": "exact-scheduled-cell-id-bijection",
    "duplicate_result_treatment": "phase-invalid-primary-suppressed",
    "unplanned_result_treatment": "phase-invalid-primary-suppressed",
    "unknown_terminal_status_treatment": "phase-invalid-primary-suppressed",
    "failure_funnel_policy": "exact-one-category-per-terminal-eval-status-v1",
    "stability_estimand": (
        "equal-task-mean-within-task-all-replicate-pair-binary-resolution-agreement"
    ),
    "interval_method": "sha256-task-within-stratum-paired-percentile-bootstrap-v1",
    "interval_quantile_method": "nearest-rank-one-based",
    "bootstrap_task_order": "selected-task-order-filtered-by-stratum",
    "bootstrap_index_derivation": (
        "sha256-utf8-seed-nul-zero-based-sample-nul-stratum-nul-zero-based-"
        "draw-big-endian-modulo-stratum-size-v1"
    ),
    "bootstrap_stratum_sample_size": "original-selected-task-count",
    "bootstrap_stratum_weighting": "original-selected-task-count",
    "analysis_seed": "swebench-astra-fable51-analysis-20260912-v1",
    "bootstrap_resamples": 10_000,
    "confidence_level": "0.95",
    "terminal_cell_requirement": "exactly-one-per-scheduled-cell",
    "infrastructure_failure_treatment": (
        "accepted-terminal-status-scores-zero-unless-resolved"
    ),
    "cost_or_trace_incomplete_treatment": "phase-incomplete-primary-suppressed",
    "missing_result_treatment": "phase-incomplete-primary-suppressed",
    "identity_drift_treatment": "phase-invalid-primary-suppressed",
    "safety_violation_treatment": "phase-invalid-primary-suppressed",
    "phase_completion_requirement": (
        "all-cells-terminal-identity-valid-cost-trace-complete"
    ),
    "complete_cost_evidence_required": True,
    "complete_trace_evidence_required": True,
    "safety_gate": "zero-authority-or-unsettled-writer-violations",
    "latency_interpretation": "diagnostic-under-concurrency",
    "metrics": [
        "primary_resolved_pass_at_1_by_arm",
        "primary_paired_resolution_delta",
        "primary_paired_both_resolve_rate",
        "primary_paired_either_resolves_rate",
        "primary_paired_disagreement_rate",
        "all_attempt_failure_funnel_by_arm",
        "all_attempt_submitted_patch_rate_by_arm",
        "all_attempt_safety_gate_block_rate_by_arm",
        "all_attempt_cost_usd_by_arm",
        "all_attempt_wall_time_seconds_by_arm",
        "within_task_replicate_agreement_by_arm",
    ],
}
_CALIBRATION_IDS = {
    "astropy/astropy": (
        "astropy__astropy-12907",
        "astropy__astropy-13033",
    ),
    "django/django": (
        "django__django-10097",
        "django__django-10554",
    ),
    "pytest-dev/pytest": (
        "pytest-dev__pytest-10051",
        "pytest-dev__pytest-10081",
    ),
    "scikit-learn/scikit-learn": (
        "scikit-learn__scikit-learn-10297",
        "scikit-learn__scikit-learn-10844",
    ),
    "sympy/sympy": (
        "sympy__sympy-11618",
        "sympy__sympy-12096",
    ),
}


@dataclasses.dataclass(frozen=True)
class MatrixBundle:
    matrix_path: Path
    dataset_path: Path
    output_root: Path
    repo_cache: Path


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_matrix(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _matrix_document(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert type(document) is dict
    return document


def _synthetic_record(instance_id: str, repo: str, ordinal: int) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": f"{ordinal:040x}",
        "patch": "",
        "test_patch": "",
        "problem_statement": f"Synthetic issue {instance_id}",
        "FAIL_TO_PASS": [f"test_{ordinal}"],
        "PASS_TO_PASS": [],
    }


@pytest.fixture
def matrix_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MatrixBundle:
    bundle = tmp_path / "matrix"
    bundle.mkdir()
    for name in _ROUTE_NAMES:
        shutil.copyfile(_PRODUCT_ROOT / name, bundle / name)
    shutil.copyfile(
        _PRODUCT_ROOT / _ANALYSIS_MODULE_NAME,
        bundle / _ANALYSIS_MODULE_NAME,
    )

    document = _matrix_document(_MATRIX_PATH)
    selection = document["selection"]
    assert type(selection) is dict
    strata = selection["strata"]
    assert type(strata) is dict

    records: list[dict[str, object]] = []
    ordinal = 900_000
    for repo_value in strata.values():
        assert type(repo_value) is str
        instance_ids = list(_CALIBRATION_IDS.get(repo_value, ()))
        prefix = repo_value.replace("/", "__")
        while len(instance_ids) < 15:
            candidate = f"{prefix}-{ordinal}"
            ordinal += 1
            if candidate not in instance_ids:
                instance_ids.append(candidate)
        for instance_id in instance_ids:
            records.append(_synthetic_record(instance_id, repo_value, ordinal))
            ordinal += 1
    while len(records) < 500:
        instance_id = f"other__repository-{ordinal}"
        records.append(_synthetic_record(instance_id, "other/repository", ordinal))
        ordinal += 1

    dataset_bytes = (
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    dataset_path = bundle / "verified.json"
    dataset_path.write_bytes(dataset_bytes)
    dataset = document["dataset"]
    assert type(dataset) is dict
    synthetic_digest = _digest(dataset_bytes)
    dataset["sha256"] = synthetic_digest
    monkeypatch.setattr(convergence, "_EXPECTED_DATASET_DIGEST", synthetic_digest)
    matrix_path = bundle / "backend_convergence_matrix.yaml"
    _write_matrix(matrix_path, document)
    return MatrixBundle(
        matrix_path=matrix_path,
        dataset_path=dataset_path,
        output_root=tmp_path / "runs",
        repo_cache=tmp_path / "repo-cache",
    )


def _load(bundle: MatrixBundle) -> BackendConvergenceMatrix:
    return load_backend_convergence_matrix(bundle.matrix_path)


def _argument(argv: tuple[str, ...], flag: str) -> str:
    assert argv.count(flag) == 1
    return argv[argv.index(flag) + 1]


def test_checked_in_matrix_pins_exact_routes_and_analysis_contract() -> None:
    matrix = load_backend_convergence_matrix(_MATRIX_PATH)

    assert tuple(matrix.arms) == ("A", "B")
    assert matrix.dataset.dataset_id == "princeton-nlp/SWE-bench_Verified:test"
    assert matrix.dataset.split == "test"
    assert matrix.dataset.revision == "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
    assert matrix.dataset.source_file == "data/test-00000-of-00001.parquet"
    assert matrix.dataset.source_digest == (
        "sha256:a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
    )
    assert matrix.dataset.export_algorithm == (
        "pyarrow-read-table-to-pylist-json-dumps-indent2-v1"
    )
    assert matrix.dataset.digest == (
        "sha256:1001dcfee9f23df1c084b17ec5f2440d4608fb6972c60303ed1d0c60e3c9493a"
    )
    assert matrix.dataset.row_count == 500
    assert (
        matrix.arms["A"].backend,
        matrix.arms["A"].model,
        matrix.arms["A"].effort,
    ) == ("codex", "gpt-6-astra", "max")
    assert (
        matrix.arms["B"].backend,
        matrix.arms["B"].model,
        matrix.arms["B"].effort,
    ) == ("claude", "claude-fable-5-1", "high")
    for arm in matrix.arms.values():
        assert arm.routing_digest == _digest(arm.routing_path.read_bytes())
        assert arm.routing_fingerprint == routing_fingerprint(
            lint_routing_config(arm.routing_path)
        )

    assert matrix.execution.max_parallel_cells == 2
    assert matrix.execution.max_parallel_per_backend == 1
    assert matrix.execution.forbidden_environment_variables == (
        "CRUCIBLE_BACKEND_CONFIG",
    )
    assert matrix.execution.dependency_condition == "any-terminal"
    assert matrix.execution.ready_queue_policy == "paired-successor-first"
    assert matrix.execution.pair_order_algorithm == (
        "within-stratum-alternating-balanced-v1"
    )
    assert matrix.analysis.provider_sampling_seed == "unavailable"
    assert matrix.analysis.comparison_unit == "paired-task-at-primary-replicate"
    assert matrix.analysis.primary_endpoint == (
        "paired-intention-to-treat-resolved-pass-at-1-arm-a-minus-arm-b"
    )
    assert matrix.analysis.primary_success_predicate == "eval-status-exactly-resolved"
    assert matrix.analysis.primary_delta_direction == "arm-a-minus-arm-b"
    assert matrix.analysis.primary_publication_rule == "complete-valid-phase-only"
    assert matrix.analysis.primary_replicate == 1
    assert matrix.analysis.primary_task_weighting == "equal-per-selected-task"
    assert matrix.analysis.primary_stratum_aggregation == (
        "selected-task-count-weighted"
    )
    assert matrix.analysis.primary_cell_denominator == "all-replicate-1-cells"
    assert matrix.analysis.secondary_attempt_denominator == "all-scheduled-cells"
    assert matrix.analysis.stability_estimand == (
        "equal-task-mean-within-task-all-replicate-pair-binary-resolution-agreement"
    )
    assert matrix.analysis.interval_method == (
        "sha256-task-within-stratum-paired-percentile-bootstrap-v1"
    )
    assert matrix.analysis.interval_quantile_method == "nearest-rank-one-based"
    assert matrix.analysis.analysis_seed == (
        "swebench-astra-fable51-analysis-20260912-v1"
    )
    assert matrix.analysis.bootstrap_resamples == 10_000
    assert matrix.analysis.confidence_level == "0.95"
    assert matrix.analysis.terminal_cell_requirement == (
        "exactly-one-per-scheduled-cell"
    )
    assert matrix.analysis.infrastructure_failure_treatment == (
        "accepted-terminal-status-scores-zero-unless-resolved"
    )
    assert matrix.analysis.cost_or_trace_incomplete_treatment == (
        "phase-incomplete-primary-suppressed"
    )
    assert matrix.analysis.missing_result_treatment == (
        "phase-incomplete-primary-suppressed"
    )
    assert matrix.analysis.identity_drift_treatment == (
        "phase-invalid-primary-suppressed"
    )
    assert matrix.analysis.safety_violation_treatment == (
        "phase-invalid-primary-suppressed"
    )
    assert matrix.analysis.phase_completion_requirement == (
        "all-cells-terminal-identity-valid-cost-trace-complete"
    )
    assert matrix.analysis.complete_cost_evidence_required
    assert matrix.analysis.complete_trace_evidence_required
    assert matrix.analysis.safety_gate == (
        "zero-authority-or-unsettled-writer-violations"
    )
    assert matrix.analysis.latency_interpretation == "diagnostic-under-concurrency"
    assert "primary_paired_disagreement_rate" in matrix.analysis.metrics
    assert matrix.analysis.implementation_digest == _ANALYSIS_MODULE_DIGEST
    assert matrix.analysis.implementation_digest == _digest(
        matrix.analysis.implementation_path.read_bytes()
    )
    assert matrix.analysis.identity() == {
        "implementation": {
            "schema": "crucible.swebench-backend-convergence-analysis.v1",
            "path": str((_PRODUCT_ROOT / _ANALYSIS_MODULE_NAME).resolve()),
            "sha256": _ANALYSIS_MODULE_DIGEST,
        },
        **_EXPECTED_ANALYSIS_IDENTITY,
    }
    assert not any(phase.promotion_evidence for phase in matrix.phases.values())
    assert matrix.phases["pilot"].adaptive_feedback == "forbidden"
    assert matrix.phases["heldout"].adaptive_feedback == "forbidden"
    assert matrix.phases["calibration"].prerequisite_phase is None
    assert matrix.phases["pilot"].prerequisite_phase == "calibration"
    assert matrix.phases["heldout"].prerequisite_phase == "pilot"


def test_exact_ab_matrix_uses_diagnostic_catalog_without_payload_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    real_validator = convergence.validate_backend_lane_catalog

    def _capture(contract, **kwargs):
        captured["contract"] = contract
        captured["lanes"] = kwargs["lanes"]
        return real_validator(contract, **kwargs)

    monkeypatch.setattr(convergence, "validate_backend_lane_catalog", _capture)

    matrix = load_backend_convergence_matrix(_MATRIX_PATH)

    assert captured["contract"] == convergence.SWEBENCH_BACKEND_CATALOG_CONTRACT
    lanes = captured["lanes"]
    assert type(lanes) is tuple
    assert [
        (
            lane.key,
            lane.backend,
            lane.runtime_backend,
            lane.transport,
            lane.evidence_status,
        )
        for lane in lanes
    ] == [
        ("A", "codex", "codex", "legacy_app_server", "diagnostic"),
        ("B", "claude", "claude", "claude_agent_sdk", "diagnostic"),
    ]
    assert set(matrix.arms["A"].identity()) == {
        "key",
        "id",
        "backend",
        "model",
        "effort",
        "routing_path",
        "routing_digest",
        "routing_fingerprint",
    }
    assert not any(phase.promotion_evidence for phase in matrix.phases.values())


def test_selection_is_deterministic_disjoint_and_stratified(
    matrix_bundle: MatrixBundle,
) -> None:
    matrix = _load(matrix_bundle)

    first = select_matrix_tasks(matrix, matrix_bundle.dataset_path)
    second = select_matrix_tasks(matrix, matrix_bundle.dataset_path)

    assert first == second
    calibration, calibration_stability = first["calibration"]
    pilot, pilot_stability = first["pilot"]
    heldout, heldout_stability = first["heldout"]
    assert [task.instance_id for task in calibration] == [
        instance_id
        for repo_ids in _CALIBRATION_IDS.values()
        for instance_id in repo_ids
    ]
    assert calibration_stability == ()
    assert pilot_stability == ()
    assert len(pilot) == 20
    assert Counter(task.stratum for task in pilot) == {
        stratum: 2 for stratum in matrix.strata
    }
    for offset in (0, 10):
        assert {task.stratum for task in pilot[offset : offset + 10]} == set(
            matrix.strata
        )
    assert len(heldout) == 100
    assert Counter(task.stratum for task in heldout) == dict(
        matrix.phases["heldout"].per_stratum
    )
    assert len(heldout_stability) == 20
    for offset in range(0, 60, 10):
        assert {task.stratum for task in heldout[offset : offset + 10]} == set(
            matrix.strata
        )
    stability_tasks = {
        task.instance_id: task
        for task in heldout
        if task.instance_id in heldout_stability
    }
    assert Counter(task.stratum for task in stability_tasks.values()) == {
        stratum: 2 for stratum in matrix.strata
    }
    assert {task.replicates for task in stability_tasks.values()} == {3}
    assert {
        task.replicates for task in heldout if task.instance_id not in heldout_stability
    } == {1}

    cohorts = [
        {task.instance_id for task in tasks} for tasks in (calibration, pilot, heldout)
    ]
    assert cohorts[0].isdisjoint(cohorts[1])
    assert cohorts[0].isdisjoint(cohorts[2])
    assert cohorts[1].isdisjoint(cohorts[2])


@pytest.mark.parametrize(
    (
        "phase",
        "task_count",
        "cell_count",
        "primary_cell_count",
        "replicate_cell_counts",
        "cost",
        "pair_count",
    ),
    (
        ("calibration", 10, 20, 20, {1: 20}, "240", 10),
        ("pilot", 20, 120, 40, {1: 40, 2: 40, 3: 40}, "1440", 60),
        ("heldout", 100, 280, 200, {1: 200, 2: 40, 3: 40}, "3360", 140),
    ),
)
def test_implement_plan_expands_isolated_balanced_pairs(
    matrix_bundle: MatrixBundle,
    phase: str,
    task_count: int,
    cell_count: int,
    primary_cell_count: int,
    replicate_cell_counts: dict[int, int],
    cost: str,
    pair_count: int,
) -> None:
    matrix = _load(matrix_bundle)
    plan = build_backend_convergence_plan(
        matrix,
        phase=phase,
        dataset_path=matrix_bundle.dataset_path,
        output_root=matrix_bundle.output_root,
        repo_cache_dir=matrix_bundle.repo_cache,
        mode="implement",
    )

    assert len(plan.selected_tasks) == task_count
    assert len(plan.cells) == cell_count
    primary_cells = [
        cell
        for cell in plan.cells
        if cell.replicate == matrix.analysis.primary_replicate
    ]
    assert len(primary_cells) == primary_cell_count == task_count * 2
    assert len(plan.cells) - len(primary_cells) == cell_count - primary_cell_count
    assert Counter(cell.replicate for cell in plan.cells) == replicate_cell_counts
    assert set(
        Counter((cell.task_id, cell.arm) for cell in primary_cells).values()
    ) == {1}
    for task in plan.selected_tasks:
        for arm in ("A", "B"):
            assert {
                cell.replicate
                for cell in plan.cells
                if cell.task_id == task.instance_id and cell.arm == arm
            } == set(range(1, task.replicates + 1))
    assert str(plan.nominal_configured_budget_usd) == cost
    assert len({cell.cell_id for cell in plan.cells}) == cell_count
    assert len({cell.root for cell in plan.cells}) == cell_count
    assert Counter(cell.assignment for cell in plan.cells if cell.order_index == 1) == {
        "AB": pair_count // 2,
        "BA": pair_count // 2,
    }
    first_cells = [cell for cell in plan.cells if cell.order_index == 1]
    for stratum in {cell.stratum for cell in first_cells}:
        stratum_orders = Counter(
            cell.assignment for cell in first_cells if cell.stratum == stratum
        )
        assert abs(stratum_orders["AB"] - stratum_orders["BA"]) <= 1
    assert Counter(cell.arm for cell in plan.cells) == {
        "A": cell_count // 2,
        "B": cell_count // 2,
    }
    assert plan.payload()["parallelism"] == {
        "max_parallel_cells": 2,
        "max_parallel_per_backend": 1,
        "dependency_condition": "any-terminal",
        "ready_queue_policy": "paired-successor-first",
        "pair_order_algorithm": "within-stratum-alternating-balanced-v1",
    }
    assert plan.payload()["dispatch_environment"] == {
        "must_be_absent": ["CRUCIBLE_BACKEND_CONFIG"]
    }
    assert plan.payload()["budget_usd_per_instance"] == "12"
    assert plan.payload()["planner_count"] == 3
    assert plan.payload()["planning_budget_share"] == "0.40"
    assert plan.payload()["analysis"] == {
        "implementation": {
            "schema": "crucible.swebench-backend-convergence-analysis.v1",
            "path": str(
                (matrix_bundle.matrix_path.parent / _ANALYSIS_MODULE_NAME).resolve()
            ),
            "sha256": _ANALYSIS_MODULE_DIGEST,
        },
        **_EXPECTED_ANALYSIS_IDENTITY,
    }
    assert plan.payload()["dataset"] == {
        "id": "princeton-nlp/SWE-bench_Verified:test",
        "split": "test",
        "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        "source_file": "data/test-00000-of-00001.parquet",
        "source_sha256": (
            "sha256:a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
        ),
        "export_algorithm": "pyarrow-read-table-to-pylist-json-dumps-indent2-v1",
        "row_count": 500,
        "sha256": _digest(matrix_bundle.dataset_path.read_bytes()),
        "path": str(matrix_bundle.dataset_path.resolve()),
    }

    pairs: dict[tuple[str, int], list[ConvergenceCell]] = defaultdict(list)
    for cell in plan.cells:
        pairs[(cell.task_id, cell.replicate)].append(cell)
        assert cell.root.is_relative_to(matrix_bundle.output_root.resolve())
        assert _argument(cell.argv, "--spec") == str(cell.root / "spec")
        assert _argument(cell.argv, "--dataset") == str(
            matrix_bundle.dataset_path.resolve()
        )
        assert _argument(cell.argv, "--instance-id") == cell.task_id
        assert _argument(cell.argv, "--predictions-out") == str(
            cell.root / "predictions.jsonl"
        )
        assert _argument(cell.argv, "--backend-config") == str(
            matrix.arms[cell.arm].routing_path
        )
        assert cell.backend == matrix.arms[cell.arm].backend
        assert cell.model == matrix.arms[cell.arm].model
        assert _argument(cell.argv, "--max-turns") == "90"
        assert _argument(cell.argv, "--planner-count") == "3"
        assert _argument(cell.argv, "--budget-usd-per-instance") == "12"
        assert _argument(cell.argv, "--budget-usd") == "12"
        assert _argument(cell.argv, "--eval-timeout") == "7200"
        assert _argument(cell.argv, "--repo-cache-dir") == str(
            matrix_bundle.repo_cache.resolve()
        )
        assert _argument(cell.argv, "--max-concurrent-instances") == "1"
        assert "--implement" in cell.argv
        assert "--submit-any-nonempty-best-effort" not in cell.argv
        assert "--no-evaluate-inline" not in cell.argv
        assert "--no-include-breach" not in cell.argv
        assert "--no-include-review" not in cell.argv
        assert "--no-build-verify" not in cell.argv

    assert len(pairs) == pair_count
    for pair in pairs.values():
        first, second = pair
        assert first.order_index == 1
        assert first.depends_on is None
        assert second.order_index == 2
        assert second.depends_on == first.cell_id
        assert {first.arm, second.arm} == {"A", "B"}
        assert first.assignment == second.assignment
        assert first.arm + second.arm == first.assignment


def test_dry_run_plan_is_zero_cost_and_digest_stable(
    matrix_bundle: MatrixBundle,
) -> None:
    matrix = _load(matrix_bundle)
    first = build_backend_convergence_plan(
        matrix,
        phase="calibration",
        dataset_path=matrix_bundle.dataset_path,
        output_root=matrix_bundle.output_root,
        repo_cache_dir=matrix_bundle.repo_cache,
        mode="dry-run",
    )
    second = build_backend_convergence_plan(
        matrix,
        phase="calibration",
        dataset_path=matrix_bundle.dataset_path,
        output_root=matrix_bundle.output_root,
        repo_cache_dir=matrix_bundle.repo_cache,
        mode="dry-run",
    )

    assert first.to_json_bytes() == second.to_json_bytes()
    assert first.nominal_configured_budget_usd == 0
    assert all("--dry-run" in cell.argv for cell in first.cells)
    assert all("--budget-usd" not in cell.argv for cell in first.cells)
    payload = json.loads(first.to_json_bytes())
    assert payload["execution_authority"] == "absent-provider-free-plan-only"
    digest = payload.pop("plan_digest")
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert digest == _digest(canonical)


def test_dataset_byte_drift_is_rejected(matrix_bundle: MatrixBundle) -> None:
    matrix = _load(matrix_bundle)
    matrix_bundle.dataset_path.write_bytes(
        matrix_bundle.dataset_path.read_bytes() + b"\n"
    )

    with pytest.raises(BackendConvergenceMatrixError, match="dataset bytes"):
        select_matrix_tasks(matrix, matrix_bundle.dataset_path)


def test_selection_parses_the_authenticated_bytes_without_reopening(
    matrix_bundle: MatrixBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _load(matrix_bundle)
    measured = matrix_bundle.dataset_path.read_bytes()
    real_parser = convergence.load_dataset_bytes

    def replace_path_after_measurement(
        payload: bytes,
        *,
        source: Path | str,
    ) -> tuple[object, ...]:
        matrix_bundle.dataset_path.write_bytes(b"[]")
        return real_parser(payload, source=source)

    monkeypatch.setattr(
        convergence, "load_dataset_bytes", replace_path_after_measurement
    )

    selections = select_matrix_tasks(matrix, matrix_bundle.dataset_path)

    assert len(selections["heldout"][0]) == 100
    assert matrix_bundle.dataset_path.read_bytes() == b"[]"
    assert _digest(measured) == matrix.dataset.digest


def test_route_byte_drift_is_rejected(matrix_bundle: MatrixBundle) -> None:
    route = matrix_bundle.matrix_path.parent / _ROUTE_NAMES[0]
    route.write_bytes(route.read_bytes() + b"\n# unbound drift\n")

    with pytest.raises(BackendConvergenceMatrixError, match="routing bytes"):
        _load(matrix_bundle)


def test_analysis_implementation_byte_drift_is_rejected(
    matrix_bundle: MatrixBundle,
) -> None:
    implementation = matrix_bundle.matrix_path.parent / _ANALYSIS_MODULE_NAME
    implementation.write_bytes(implementation.read_bytes() + b"\n# drift\n")

    with pytest.raises(
        BackendConvergenceMatrixError,
        match="analysis implementation bytes",
    ):
        _load(matrix_bundle)


def test_semantic_route_drift_is_rejected_even_when_rehashed(
    matrix_bundle: MatrixBundle,
) -> None:
    document = _matrix_document(matrix_bundle.matrix_path)
    arms = document["arms"]
    assert type(arms) is dict
    arm = arms["A"]
    assert type(arm) is dict
    route = matrix_bundle.matrix_path.parent / str(arm["routing"])
    route_document = yaml.safe_load(route.read_text(encoding="utf-8"))
    route_document["default"]["model"] = "gpt-5.6-sol"
    for role in route_document["roles"].values():
        role["model"] = "gpt-5.6-sol"
    route_bytes = yaml.safe_dump(route_document, sort_keys=False).encode("utf-8")
    route.write_bytes(route_bytes)
    arm["routing_sha256"] = _digest(route_bytes)
    arm["routing_fingerprint"] = routing_fingerprint(lint_routing_config(route_bytes))
    _write_matrix(matrix_bundle.matrix_path, document)

    with pytest.raises(
        BackendConvergenceMatrixError, match="drifts from its exact lane"
    ):
        _load(matrix_bundle)


def test_unknown_matrix_fields_and_route_escape_are_rejected(
    matrix_bundle: MatrixBundle,
) -> None:
    document = _matrix_document(matrix_bundle.matrix_path)
    document["surprise"] = True
    _write_matrix(matrix_bundle.matrix_path, document)
    with pytest.raises(BackendConvergenceMatrixError, match="fields drifted"):
        _load(matrix_bundle)

    document.pop("surprise")
    arms = document["arms"]
    assert type(arms) is dict
    arm = arms["A"]
    assert type(arm) is dict
    arm["routing"] = "../backends_codex_astra.yaml"
    _write_matrix(matrix_bundle.matrix_path, document)
    with pytest.raises(BackendConvergenceMatrixError, match="confined path"):
        _load(matrix_bundle)


def test_caller_matrix_tamper_is_rejected(
    matrix_bundle: MatrixBundle,
) -> None:
    matrix = _load(matrix_bundle)
    tampered = dataclasses.replace(matrix, selection_seed="caller-changed")
    with pytest.raises(BackendConvergenceMatrixError, match="differs from source"):
        build_backend_convergence_plan(
            tampered,
            phase="pilot",
            dataset_path=matrix_bundle.dataset_path,
            output_root=matrix_bundle.output_root,
            repo_cache_dir=matrix_bundle.repo_cache,
        )
    with pytest.raises(BackendConvergenceMatrixError, match="differs from source"):
        select_matrix_tasks(tampered, matrix_bundle.dataset_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "omit_role",
        "pilot_feedback",
        "pilot_promotion",
        "pilot_replicates",
        "heldout_quota",
        "disable_review",
        "single_planner",
        "forbidden_environment",
        "incomplete_cost_evidence",
        "analysis_implementation_path",
        "analysis_implementation_digest",
        "primary_success_predicate",
        "primary_delta_direction",
        "primary_publication_rule",
        "primary_replicate",
        "primary_cell_denominator",
        "secondary_attempt_denominator",
        "result_set_policy",
        "failure_funnel_policy",
        "stability_estimand",
        "analysis_seed",
        "interval_method",
        "bootstrap_index_derivation",
        "bootstrap_resamples",
        "confidence_level",
        "metrics",
        "analysis_extra_field",
        "analysis_missing_field",
        "selection_algorithm",
        "dataset_id",
        "dataset_split",
        "dataset_revision",
        "dataset_source_file",
        "dataset_source_sha256",
        "dataset_export_algorithm",
        "dataset_row_count",
        "dataset_digest",
    ),
)
def test_v1_scientific_and_execution_contract_is_closed(
    matrix_bundle: MatrixBundle,
    mutation: str,
) -> None:
    document = _matrix_document(matrix_bundle.matrix_path)
    if mutation == "omit_role":
        roles = document["roles"]
        assert type(roles) is list
        roles.pop()
    elif mutation.startswith("pilot_"):
        phases = document["phases"]
        assert type(phases) is dict
        pilot = phases["pilot"]
        assert type(pilot) is dict
        if mutation == "pilot_feedback":
            pilot["adaptive_feedback"] = "allowed"
        elif mutation == "pilot_promotion":
            pilot["promotion_evidence"] = True
        else:
            pilot["replicates"] = 2
    elif mutation == "heldout_quota":
        phases = document["phases"]
        assert type(phases) is dict
        heldout = phases["heldout"]
        assert type(heldout) is dict
        per_stratum = heldout["per_stratum"]
        assert type(per_stratum) is dict
        per_stratum["django"] = 10
    elif mutation == "single_planner":
        execution = document["execution"]
        assert type(execution) is dict
        execution["planner_count"] = 1
    elif mutation == "disable_review":
        execution = document["execution"]
        assert type(execution) is dict
        execution["include_review"] = False
    elif mutation == "forbidden_environment":
        execution = document["execution"]
        assert type(execution) is dict
        execution["forbidden_environment_variables"] = []
    elif mutation == "incomplete_cost_evidence":
        analysis = document["analysis"]
        assert type(analysis) is dict
        analysis["complete_cost_evidence_required"] = False
    elif mutation in {
        "analysis_implementation_path",
        "analysis_implementation_digest",
    }:
        analysis = document["analysis"]
        assert type(analysis) is dict
        implementation = analysis["implementation"]
        assert type(implementation) is dict
        if mutation == "analysis_implementation_path":
            implementation["path"] = "alternate_analysis.py"
        else:
            implementation["sha256"] = "sha256:" + "0" * 64
    elif mutation in {
        "primary_success_predicate",
        "primary_delta_direction",
        "primary_publication_rule",
        "primary_replicate",
        "primary_cell_denominator",
        "secondary_attempt_denominator",
        "result_set_policy",
        "failure_funnel_policy",
        "stability_estimand",
        "analysis_seed",
        "interval_method",
        "bootstrap_index_derivation",
        "bootstrap_resamples",
        "confidence_level",
        "metrics",
        "analysis_extra_field",
        "analysis_missing_field",
    }:
        analysis = document["analysis"]
        assert type(analysis) is dict
        if mutation == "primary_success_predicate":
            analysis["primary_success_predicate"] = "classification-starts-success"
        elif mutation == "primary_delta_direction":
            analysis["primary_delta_direction"] = "arm-b-minus-arm-a"
        elif mutation == "primary_publication_rule":
            analysis["primary_publication_rule"] = "publish-incomplete"
        elif mutation == "primary_replicate":
            analysis["primary_replicate"] = 2
        elif mutation == "primary_cell_denominator":
            analysis["primary_cell_denominator"] = "all-scheduled-cells"
        elif mutation == "secondary_attempt_denominator":
            analysis["secondary_attempt_denominator"] = "replicate-1-only"
        elif mutation == "result_set_policy":
            analysis["result_set_policy"] = "last-result-wins"
        elif mutation == "failure_funnel_policy":
            analysis["failure_funnel_policy"] = "catch-all"
        elif mutation == "stability_estimand":
            analysis["stability_estimand"] = "all-three-identical"
        elif mutation == "analysis_seed":
            analysis["analysis_seed"] = "changed-seed"
        elif mutation == "interval_method":
            analysis["interval_method"] = "ordinary-bootstrap"
        elif mutation == "bootstrap_index_derivation":
            analysis["bootstrap_index_derivation"] = "ambient-random"
        elif mutation == "bootstrap_resamples":
            analysis["bootstrap_resamples"] = 9_999
        elif mutation == "confidence_level":
            analysis["confidence_level"] = "0.90"
        elif mutation == "metrics":
            metrics = analysis["metrics"]
            assert type(metrics) is list
            metrics.pop()
        elif mutation == "analysis_extra_field":
            analysis["surprise"] = True
        else:
            analysis.pop("terminal_cell_requirement")
    elif mutation == "selection_algorithm":
        selection = document["selection"]
        assert type(selection) is dict
        selection["algorithm"] = "sha256-ranked-by-stratum-v1"
    else:
        dataset = document["dataset"]
        assert type(dataset) is dict
        if mutation == "dataset_id":
            dataset["id"] = "somewhere/else:test"
        elif mutation == "dataset_split":
            dataset["split"] = "train"
        elif mutation == "dataset_revision":
            dataset["revision"] = "0" * 40
        elif mutation == "dataset_source_file":
            dataset["source_file"] = "data/alternate.parquet"
        elif mutation == "dataset_source_sha256":
            dataset["source_sha256"] = "sha256:" + "0" * 64
        elif mutation == "dataset_export_algorithm":
            dataset["export_algorithm"] = "alternate-export"
        elif mutation == "dataset_row_count":
            dataset["row_count"] = 499
        else:
            dataset["sha256"] = "sha256:" + "0" * 64
    _write_matrix(matrix_bundle.matrix_path, document)

    with pytest.raises(BackendConvergenceMatrixError):
        _load(matrix_bundle)


def test_cli_creates_plan_exclusively_without_launching(
    matrix_bundle: MatrixBundle,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = matrix_bundle.output_root / "plans" / "pilot.json"
    argv = [
        "--matrix",
        str(matrix_bundle.matrix_path),
        "--dataset",
        str(matrix_bundle.dataset_path),
        "--output-root",
        str(matrix_bundle.output_root),
        "--repo-cache-dir",
        str(matrix_bundle.repo_cache),
        "--phase",
        "pilot",
        "--mode",
        "implement",
        "--plan-out",
        str(plan_path),
    ]

    assert cli.main(argv) == 0
    first_bytes = plan_path.read_bytes()
    payload = json.loads(first_bytes)
    assert payload["phase"] == "pilot"
    assert len(payload["selected_tasks"]) == 20
    assert len(payload["cells"]) == 120
    assert payload["nominal_configured_budget_usd"] == "1440"
    first_capture = capsys.readouterr()
    assert first_capture.out == ""
    assert "phase=pilot tasks=20 cells=120 nominal_budget_usd=1440" in (
        first_capture.err
    )

    assert cli.main(argv) == 2
    assert plan_path.read_bytes() == first_bytes
    second_capture = capsys.readouterr()
    assert "refusing overwrite" in second_capture.err
