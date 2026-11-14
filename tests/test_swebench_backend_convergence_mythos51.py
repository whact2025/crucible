# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.products.swebench import backend_convergence as convergence
from crucible.products.swebench import backend_convergence_analysis as v1_analysis
from crucible.products.swebench import (
    backend_convergence_analysis_astra_mythos51 as mythos_analysis,
)
from crucible.products.swebench.backend_convergence import (
    BackendConvergenceMatrixError,
    ConvergenceCell,
    build_backend_convergence_plan,
    load_backend_convergence_matrix,
    select_matrix_tasks,
)
from crucible.scripts import swebench_backend_convergence as cli


_ROOT = Path(__file__).parents[1]
_PRODUCT_ROOT = _ROOT / "products" / "swebench"
_V1_MATRIX = _PRODUCT_ROOT / "backend_convergence_matrix.yaml"
_V2_MATRIX = _PRODUCT_ROOT / "backend_convergence_matrix_astra_mythos51.yaml"
_V1_ANALYZER = _PRODUCT_ROOT / "backend_convergence_analysis.py"
_V2_ANALYZER = _PRODUCT_ROOT / "backend_convergence_analysis_astra_mythos51.py"
_V1_MATRIX_DIGEST = (
    "sha256:7d9dd1815876569111e5b02ae48603392634e96005ae7a747cb2880da8e6bc55"
)
_V1_ANALYZER_DIGEST = (
    "sha256:4973d24d7537305350e9930fda269573a083030a6a77ed8b03b4d6b6d656e14d"
)
_V2_MATRIX_DIGEST = (
    "sha256:f17e597da7b034a232cd9c29cd3cb9c982227f791aeccb52212fb9899947ee8e"
)
_V2_ANALYZER_DIGEST = (
    "sha256:4209cee86a7676014e6ad5e0a05cfef23364edabd7cce71c202ccdd80427bd70"
)
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


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _document(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert type(result) is dict
    return result


def _write_document(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _record(instance_id: str, repo: str, ordinal: int) -> dict[str, object]:
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
def mythos_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    bundle = tmp_path / "matrix"
    bundle.mkdir()
    for name in (
        "backends_codex_astra.yaml",
        "backends_claude_fable51_org.yaml",
        "backends_claude_mythos51_org.yaml",
        _V1_ANALYZER.name,
        _V2_ANALYZER.name,
    ):
        shutil.copyfile(_PRODUCT_ROOT / name, bundle / name)

    document = _document(_V2_MATRIX)
    selection = document["selection"]
    assert type(selection) is dict
    strata = selection["strata"]
    assert type(strata) is dict

    records: list[dict[str, object]] = []
    ordinal = 910_000
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
            records.append(_record(instance_id, repo_value, ordinal))
            ordinal += 1
    while len(records) < 500:
        instance_id = f"other__repository-{ordinal}"
        records.append(_record(instance_id, "other/repository", ordinal))
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
    matrix_path = bundle / _V2_MATRIX.name
    _write_document(matrix_path, document)

    v1_document = _document(_V1_MATRIX)
    v1_dataset = v1_document["dataset"]
    assert type(v1_dataset) is dict
    v1_dataset["sha256"] = synthetic_digest
    v1_matrix_path = bundle / _V1_MATRIX.name
    _write_document(v1_matrix_path, v1_document)
    return {
        "matrix": matrix_path,
        "v1_matrix": v1_matrix_path,
        "dataset": dataset_path,
        "output": tmp_path / "astra-mythos51-v2-runs",
        "cache": tmp_path / "repo-cache",
    }


def test_v1_current_matrix_and_analysis_identity() -> None:
    assert _digest(_V1_MATRIX.read_bytes()) == _V1_MATRIX_DIGEST
    assert _digest(_V1_ANALYZER.read_bytes()) == _V1_ANALYZER_DIGEST
    assert cli._DEFAULT_MATRIX == _V1_MATRIX

    matrix = load_backend_convergence_matrix(_V1_MATRIX)
    assert matrix.schema_version == convergence.MATRIX_SCHEMA
    assert matrix.plan_schema == convergence.PLAN_SCHEMA
    assert matrix.arms["B"].model == "claude-fable-5-1"
    assert matrix.analysis.implementation_path == _V1_ANALYZER.resolve()
    assert matrix.analysis.analysis_seed == v1_analysis.ANALYSIS_SEED


def test_checked_in_v2_matrix_binds_exact_astra_mythos_treatment() -> None:
    assert _digest(_V2_MATRIX.read_bytes()) == _V2_MATRIX_DIGEST
    assert _digest(_V2_ANALYZER.read_bytes()) == _V2_ANALYZER_DIGEST

    matrix = load_backend_convergence_matrix(_V2_MATRIX)

    assert matrix.source_digest == _V2_MATRIX_DIGEST
    assert matrix.schema_version == convergence.MATRIX_SCHEMA_V2
    assert matrix.plan_schema == convergence.PLAN_SCHEMA_V2
    assert matrix.selection_seed == ("swebench-astra-mythos51-convergence-20260914-v2")
    assert tuple(matrix.arms) == ("A", "B")
    assert (
        matrix.arms["A"].arm_id,
        matrix.arms["A"].backend,
        matrix.arms["A"].model,
        matrix.arms["A"].effort,
    ) == ("codex-astra", "codex", "gpt-6-astra", "max")
    assert (
        matrix.arms["B"].arm_id,
        matrix.arms["B"].backend,
        matrix.arms["B"].model,
        matrix.arms["B"].effort,
    ) == ("claude-mythos51", "claude", "claude-mythos-5-1", "high")
    for arm in matrix.arms.values():
        assert arm.routing_digest == _digest(arm.routing_path.read_bytes())
        assert arm.routing_fingerprint == routing_fingerprint(
            lint_routing_config(arm.routing_path)
        )
    assert matrix.arms["B"].routing_digest == (
        "sha256:5a19919c4d7cdd8ce90bbb060d31165026032780654daae5e1e579e85354c877"
    )
    assert matrix.arms["B"].routing_fingerprint == (
        "sha256:ee99b26f50ba7d9a7d5f155d3ccf2e7d35f0968226e11eda31306f4c20a48c12"
    )
    assert matrix.analysis.implementation_schema == (
        "crucible.swebench-backend-convergence-analysis.v2"
    )
    assert matrix.analysis.implementation_path == _V2_ANALYZER.resolve()
    assert matrix.analysis.implementation_digest == _V2_ANALYZER_DIGEST
    assert matrix.analysis.analysis_seed == mythos_analysis.ANALYSIS_SEED
    assert _document(_V2_MATRIX)["execution"]["output_namespace"] == (
        "swebench-astra-mythos51-v2"
    )
    assert all(not phase.promotion_evidence for phase in matrix.phases.values())
    assert matrix.phases["pilot"].adaptive_feedback == "forbidden"
    assert matrix.phases["heldout"].adaptive_feedback == "forbidden"


def test_v2_analyzer_is_a_distinct_seed_bound_copy_of_v1_semantics() -> None:
    v2_text = _V2_ANALYZER.read_text(encoding="utf-8")
    normalized = v2_text.replace(
        mythos_analysis.ANALYSIS_SEED,
        v1_analysis.ANALYSIS_SEED,
    ).replace("v2", "v1")

    assert normalized == _V1_ANALYZER.read_text(encoding="utf-8")
    assert mythos_analysis.ANALYSIS_SCHEMA == (
        "crucible.swebench-backend-convergence-analysis.v2"
    )
    assert mythos_analysis.ANALYSIS_SEED != v1_analysis.ANALYSIS_SEED
    assert mythos_analysis.TERMINAL_EVAL_STATUSES == v1_analysis.TERMINAL_EVAL_STATUSES
    assert dict(mythos_analysis.FAILURE_FUNNEL_BY_STATUS) == dict(
        v1_analysis.FAILURE_FUNNEL_BY_STATUS
    )
    assert mythos_analysis.primary_paired_resolution_delta(
        (("resolved", "unresolved"), ("unresolved", "resolved"))
    ) == Fraction(0, 1)
    assert mythos_analysis.paired_bootstrap_interval(
        {"one": [1, 0, -1], "two": [1, 1]}
    ) == (Fraction(-1, 5), Fraction(1, 1))
    with pytest.raises(
        mythos_analysis.BackendConvergenceAnalysisError,
        match="exact analysis seed",
    ):
        mythos_analysis.paired_bootstrap_interval(
            {"one": [1]},
            seed=v1_analysis.ANALYSIS_SEED,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("unknown_schema", "unsupported matrix schema"),
        ("v1_schema", "exact convergence lane"),
        ("fable_treatment", "exact convergence lane"),
        ("selection_seed", "selection.seed"),
        ("output_namespace", "execution.output_namespace"),
        ("v1_analyzer", "schema and path must remain exact"),
        ("v1_analysis_seed", "analysis.analysis_seed"),
        ("promotion", "pilot must be a sealed"),
    ),
)
def test_v2_matrix_rejects_cross_generation_or_promotional_drift(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    for name in (
        "backends_codex_astra.yaml",
        "backends_claude_mythos51_org.yaml",
        _V2_ANALYZER.name,
    ):
        shutil.copyfile(_PRODUCT_ROOT / name, tmp_path / name)
    matrix_path = tmp_path / _V2_MATRIX.name
    document = _document(_V2_MATRIX)
    if mutation == "unknown_schema":
        document["schema_version"] = "crucible.swebench-backend-convergence-matrix.v3"
    elif mutation == "v1_schema":
        document["schema_version"] = convergence.MATRIX_SCHEMA
    elif mutation == "fable_treatment":
        arm = document["arms"]["B"]
        arm.update(
            {
                "id": "claude-fable51",
                "model": "claude-fable-5-1",
                "routing": "backends_claude_fable51_org.yaml",
                "routing_sha256": (
                    "sha256:855a8e856a2f83cdc96f707962d797d050fe4d123bc9a0cac29b3777432e3c6b"
                ),
                "routing_fingerprint": (
                    "sha256:275a5c7e5f1336341fa59cad35824e5ffb19cfc4a5dbc5b67d11ecc0f4552934"
                ),
            }
        )
    elif mutation == "selection_seed":
        document["selection"]["seed"] = "swebench-astra-fable51-convergence-20260912-v1"
    elif mutation == "output_namespace":
        document["execution"]["output_namespace"] = "swebench-astra-fable51-v1"
    elif mutation == "v1_analyzer":
        implementation = document["analysis"]["implementation"]
        implementation.update(
            {
                "schema": v1_analysis.ANALYSIS_SCHEMA,
                "path": _V1_ANALYZER.name,
                "sha256": _V1_ANALYZER_DIGEST,
            }
        )
    elif mutation == "v1_analysis_seed":
        document["analysis"]["analysis_seed"] = v1_analysis.ANALYSIS_SEED
    else:
        document["phases"]["pilot"]["promotion_evidence"] = True
    _write_document(matrix_path, document)

    with pytest.raises(BackendConvergenceMatrixError, match=match):
        load_backend_convergence_matrix(matrix_path)


def test_v2_matrix_rejects_bound_analyzer_byte_drift(tmp_path: Path) -> None:
    for name in (
        "backends_codex_astra.yaml",
        "backends_claude_mythos51_org.yaml",
        _V2_ANALYZER.name,
    ):
        shutil.copyfile(_PRODUCT_ROOT / name, tmp_path / name)
    matrix_path = tmp_path / _V2_MATRIX.name
    shutil.copyfile(_V2_MATRIX, matrix_path)
    analyzer = tmp_path / _V2_ANALYZER.name
    analyzer.write_bytes(analyzer.read_bytes() + b"\n# drift\n")

    with pytest.raises(
        BackendConvergenceMatrixError,
        match="analysis implementation bytes",
    ):
        load_backend_convergence_matrix(matrix_path)


def test_v2_selection_and_plan_are_deterministic_isolated_and_balanced(
    mythos_bundle: dict[str, Path],
) -> None:
    matrix = load_backend_convergence_matrix(mythos_bundle["matrix"])
    first = select_matrix_tasks(matrix, mythos_bundle["dataset"])
    second = select_matrix_tasks(matrix, mythos_bundle["dataset"])

    assert first == second
    calibration = first["calibration"][0]
    pilot = first["pilot"][0]
    heldout, stability = first["heldout"]
    assert len(calibration) == 10
    assert len(pilot) == 20
    assert Counter(task.stratum for task in pilot) == {
        stratum: 2 for stratum in matrix.strata
    }
    assert len(heldout) == 100
    assert Counter(task.stratum for task in heldout) == dict(
        matrix.phases["heldout"].per_stratum
    )
    assert len(stability) == 20
    cohorts = tuple(
        {task.instance_id for task in first[phase][0]}
        for phase in ("calibration", "pilot", "heldout")
    )
    assert cohorts[0].isdisjoint(cohorts[1])
    assert cohorts[0].isdisjoint(cohorts[2])
    assert cohorts[1].isdisjoint(cohorts[2])

    plan = build_backend_convergence_plan(
        matrix,
        phase="pilot",
        dataset_path=mythos_bundle["dataset"],
        output_root=mythos_bundle["output"],
        repo_cache_dir=mythos_bundle["cache"],
        mode="implement",
    )
    payload = plan.payload()
    assert payload["schema_version"] == convergence.PLAN_SCHEMA_V2
    assert payload["matrix_digest"] == matrix.source_digest
    assert payload["output_namespace"] == "swebench-astra-mythos51-v2"
    assert payload["output_root"] == str(mythos_bundle["output"].resolve())
    assert payload["nominal_configured_budget_usd"] == "1440"
    assert payload["promotion_evidence"] is False
    assert payload["dispatch_environment"] == {
        "must_be_absent": ["CRUCIBLE_BACKEND_CONFIG"]
    }
    assert len(plan.cells) == 120
    assert len({cell.cell_id for cell in plan.cells}) == 120
    assert len({cell.root for cell in plan.cells}) == 120
    assert Counter(cell.arm for cell in plan.cells) == {"A": 60, "B": 60}
    assert Counter(cell.assignment for cell in plan.cells if cell.order_index == 1) == {
        "AB": 30,
        "BA": 30,
    }
    namespaced_root = mythos_bundle["output"].resolve() / "swebench-astra-mythos51-v2"
    assert all(cell.root.is_relative_to(namespaced_root) for cell in plan.cells)
    assert all(
        str(matrix.arms[cell.arm].routing_path)
        == cell.argv[cell.argv.index("--backend-config") + 1]
        for cell in plan.cells
    )
    assert {cell.model for cell in plan.cells if cell.arm == "B"} == {
        "claude-mythos-5-1"
    }

    pairs: dict[tuple[str, int], list[ConvergenceCell]] = defaultdict(list)
    for cell in plan.cells:
        pairs[(cell.task_id, cell.replicate)].append(cell)
    assert len(pairs) == 60
    for pair in pairs.values():
        first_cell, second_cell = pair
        assert first_cell.depends_on is None
        assert second_cell.depends_on == first_cell.cell_id

    v1_plan = build_backend_convergence_plan(
        load_backend_convergence_matrix(mythos_bundle["v1_matrix"]),
        phase="calibration",
        dataset_path=mythos_bundle["dataset"],
        output_root=mythos_bundle["output"],
        repo_cache_dir=mythos_bundle["cache"],
        mode="implement",
    )
    v2_plan = build_backend_convergence_plan(
        matrix,
        phase="calibration",
        dataset_path=mythos_bundle["dataset"],
        output_root=mythos_bundle["output"],
        repo_cache_dir=mythos_bundle["cache"],
        mode="implement",
    )
    assert "output_namespace" not in v1_plan.payload()
    assert all(
        cell.root.relative_to(mythos_bundle["output"].resolve()).parts[0]
        == "calibration"
        for cell in v1_plan.cells
    )
    assert all(cell.root.is_relative_to(namespaced_root) for cell in v2_plan.cells)
    assert {cell.root for cell in v1_plan.cells}.isdisjoint(
        cell.root for cell in v2_plan.cells
    )


def test_cli_accepts_v2_only_when_explicitly_selected(
    mythos_bundle: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = mythos_bundle["output"] / "plans" / "pilot.json"
    argv = [
        "--matrix",
        str(mythos_bundle["matrix"]),
        "--dataset",
        str(mythos_bundle["dataset"]),
        "--output-root",
        str(mythos_bundle["output"]),
        "--repo-cache-dir",
        str(mythos_bundle["cache"]),
        "--phase",
        "pilot",
        "--mode",
        "implement",
        "--plan-out",
        str(plan_path),
    ]

    assert cli.main(argv) == 0
    payload = json.loads(plan_path.read_bytes())
    assert payload["schema_version"] == convergence.PLAN_SCHEMA_V2
    assert payload["arms"]["B"]["model"] == "claude-mythos-5-1"
    assert len(payload["cells"]) == 120
    first_capture = capsys.readouterr()
    assert first_capture.out == ""
    assert "phase=pilot tasks=20 cells=120 nominal_budget_usd=1440" in (
        first_capture.err
    )

    assert cli.main(argv) == 2
    assert "refusing overwrite" in capsys.readouterr().err
