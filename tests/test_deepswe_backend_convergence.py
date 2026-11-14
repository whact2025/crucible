# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import stat
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml  # type: ignore[import-untyped]

from crucible.products.deepswe import quality_cohort as convergence
from crucible.products.deepswe.dataset import DeepSWETestsTreeIdentity
from crucible.products.deepswe.quality_cohort import (
    ARMS,
    EXCLUDED_REPOS,
    EXCLUDED_TASK_IDS,
    QUALITY_PILOT_TASK_IDS,
    BoundTask,
    CorpusBinding,
    DeepSWEQualityCohortError,
    QualityCohortMatrix,
    QualityCohortPlan,
    build_quality_cohort_plan,
    load_quality_cohort_matrix,
    validate_frozen_plan,
)
from crucible.scripts import deepswe_backend_convergence as cli
from crucible.workflows import deepswe

_ROOT = Path(__file__).parents[1]
_PRODUCT_ROOT = _ROOT / "products" / "deepswe"
_MATRIX_PATH = _PRODUCT_ROOT / "backend_convergence_matrix.yaml"
_CONTROL_FILES = (
    "backend_convergence_analysis.py",
    "backends_codex_astra.yaml",
    "backends_claude_mythos51_org.yaml",
    "backends_kimi_sdk_diagnostic.yaml",
)


@pytest.fixture
def matrix_bundle(tmp_path: Path) -> Path:
    product = tmp_path / "products" / "deepswe"
    product.mkdir(parents=True)
    shutil.copy2(_MATRIX_PATH, product / _MATRIX_PATH.name)
    for name in _CONTROL_FILES:
        shutil.copy2(_PRODUCT_ROOT / name, product / name)
    return product / _MATRIX_PATH.name


def _fake_corpus(matrix: QualityCohortMatrix, tmp_path: Path) -> CorpusBinding:
    root = tmp_path / "frozen-deep-swe"
    root.mkdir(exist_ok=True)
    manifest = root / "tasks" / "manifest.json"
    tasks = tuple(
        BoundTask(
            task_id=task.task_id,
            repo=task.repo,
            language=task.language,
            category=task.category,
            base_commit=f"{index:040x}",
            tests_identity=DeepSWETestsTreeIdentity(
                sha256=f"sha256:{index:064x}",
                file_count=2,
                directory_count=1,
                total_bytes=100 + index,
            ),
            replicate_orders=task.replicate_orders,
        )
        for index, task in enumerate(matrix.tasks, start=1)
    )
    return CorpusBinding(
        root=root,
        manifest_path=manifest,
        manifest_digest=matrix.corpus.manifest_digest,
        git_commit=matrix.corpus.git_commit,
        git_tree=matrix.corpus.git_tree,
        clean_status_digest=matrix.corpus.clean_status_digest,
        tasks=tasks,
    )


@pytest.fixture
def planned(
    matrix_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> QualityCohortPlan:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)
    return build_quality_cohort_plan(
        matrix,
        dataset_path=corpus.manifest_path,
        output_root=tmp_path / "proposed-output",
    )


def _mutate_matrix(path: Path, mutation: str) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if mutation == "provider_calls":
        document["authority"]["provider_calls"] = True
    elif mutation == "boolean_type_smuggling":
        document["authority"]["provider_calls"] = 0
    elif mutation == "live_dispatch":
        document["authority"]["live_dispatch"] = True
    elif mutation == "ranking":
        document["authority"]["ranking_admission"] = True
    elif mutation == "formal_e12":
        document["authority"]["formal_e12_execution_admission"] = True
    elif mutation == "corpus_count":
        document["corpus"]["task_count"] = 112
    elif mutation == "manifest_digest":
        document["corpus"]["manifest_sha256"] = "sha256:" + "0" * 64
    elif mutation == "fable_model":
        document["arms"]["B"]["model"] = "claude-fable-5-1"
    elif mutation == "sol_model":
        document["arms"]["A"]["model"] = "gpt-5.6-sol"
    elif mutation == "kimi_enforcing_route":
        document["arms"]["C"]["routing"] = "backends_kimi_sdk.yaml"
    elif mutation == "route_fingerprint":
        document["arms"]["A"]["static_routing_fingerprint"] = "sha256:" + "0" * 64
    elif mutation == "exclusions":
        document["selection"]["exclusions"]["repos"] = []
    elif mutation == "task":
        document["selection"]["tasks"][0]["task_id"] = "prometheus-typed-label-sorting"
    elif mutation == "order":
        document["selection"]["tasks"][0]["replicate_orders"] = ["ABC"] * 3
    elif mutation == "parallel_backend":
        document["execution"]["max_parallel_per_backend"] = 2
    elif mutation == "first_block":
        document["execution"]["first_block_serial"] = False
    elif mutation == "budget":
        document["execution"]["budget"]["nominal_usd"] = "1079"
    elif mutation == "planner_count":
        document["execution"]["planner_count"] = 1
    elif mutation == "spend_authority":
        document["execution"]["budget"]["spend_authorized"] = True
    elif mutation == "analysis_ranking":
        document["analysis"]["ranking_admission"] = True
    else:  # pragma: no cover - closed test parametrization
        raise AssertionError(mutation)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_checked_in_matrix_binds_exact_diagnostic_pilot() -> None:
    matrix = load_quality_cohort_matrix(_MATRIX_PATH)

    assert tuple(matrix.arms) == ARMS
    assert tuple(task.task_id for task in matrix.tasks) == QUALITY_PILOT_TASK_IDS
    assert Counter(task.language for task in matrix.tasks) == {
        "go": 2,
        "javascript": 2,
        "python": 2,
        "rust": 2,
        "typescript": 2,
    }
    assert matrix.execution.per_cell_soft_budget_usd == 12
    assert matrix.execution.nominal_budget_usd == 1080
    assert matrix.execution.planner_count == 3
    assert matrix.execution.max_parallel_cells == 3
    assert matrix.execution.max_parallel_per_backend == 1
    assert matrix.execution.first_block_serial is True
    assert matrix.arms["A"].model == "gpt-6-astra"
    assert matrix.arms["B"].model == "claude-mythos-5-1"
    assert matrix.arms["C"].model == "kimi-k2.7-code-highspeed"
    assert matrix.arms["C"].feature_admission == "diagnostic"
    assert matrix.arms["C"].static_routing_fingerprint == (
        "sha256:1f8d11259eceefe784a77ac87a1b7287dc474c92bc1c3a861c248170f40914b4"
    )
    assert matrix.analysis.implementation_digest == (
        "sha256:1629b326ada7c7f375b148a9254ae979636a4eacc1264bc6f191a2bd503feeef"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "provider_calls",
        "boolean_type_smuggling",
        "live_dispatch",
        "ranking",
        "formal_e12",
        "corpus_count",
        "manifest_digest",
        "fable_model",
        "sol_model",
        "kimi_enforcing_route",
        "route_fingerprint",
        "exclusions",
        "task",
        "order",
        "parallel_backend",
        "first_block",
        "budget",
        "planner_count",
        "spend_authority",
        "analysis_ranking",
    ),
)
def test_matrix_authority_identity_and_shape_mutations_fail_closed(
    matrix_bundle: Path, mutation: str
) -> None:
    _mutate_matrix(matrix_bundle, mutation)
    with pytest.raises(DeepSWEQualityCohortError):
        load_quality_cohort_matrix(matrix_bundle)


@pytest.mark.parametrize(
    "filename",
    (
        "backend_convergence_analysis.py",
        "backends_codex_astra.yaml",
        "backends_claude_mythos51_org.yaml",
        "backends_kimi_sdk_diagnostic.yaml",
    ),
)
def test_byte_bound_analysis_and_routes_reject_tamper(
    matrix_bundle: Path, filename: str
) -> None:
    target = matrix_bundle.parent / filename
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(DeepSWEQualityCohortError, match="bytes|digest|identity"):
        load_quality_cohort_matrix(matrix_bundle)


def test_matrix_rejects_duplicate_yaml_authority(matrix_bundle: Path) -> None:
    matrix_bundle.write_bytes(matrix_bundle.read_bytes() + b"\nauthority: {}\n")
    with pytest.raises(DeepSWEQualityCohortError, match="repeats mapping key"):
        load_quality_cohort_matrix(matrix_bundle)


def test_plan_expands_exact_counterbalanced_90_cell_graph(
    planned: QualityCohortPlan,
) -> None:
    assert len(planned.corpus.tasks) == 10
    assert len(planned.cells) == 90
    assert len({cell.cell_id for cell in planned.cells}) == 90
    assert len({cell.root for cell in planned.cells}) == 90
    assert Counter(cell.arm for cell in planned.cells) == {"A": 30, "B": 30, "C": 30}
    assert Counter(cell.assignment for cell in planned.cells) == {
        "ABC": 30,
        "BCA": 30,
        "CAB": 30,
    }
    assert Counter(
        cell.assignment for cell in planned.cells if cell.order_index == 1
    ) == {
        "ABC": 10,
        "BCA": 10,
        "CAB": 10,
    }
    first = planned.cells[:3]
    assert [cell.arm for cell in first] == ["A", "B", "C"]
    assert first[0].depends_on == ()
    assert first[1].depends_on == (first[0].cell_id,)
    assert first[2].depends_on == (first[1].cell_id,)
    first_terminal = first[2].cell_id
    for index in range(3, len(planned.cells), 3):
        block = planned.cells[index : index + 3]
        assert block[0].depends_on == (first_terminal,)
        assert block[1].depends_on == (block[0].cell_id,)
        assert block[2].depends_on == (block[1].cell_id,)


def test_plan_is_explicitly_non_executable_and_zero_authority(
    planned: QualityCohortPlan,
) -> None:
    payload = planned.payload()
    assert payload["planner_mode"] == "provider-free-dry-run"
    assert payload["execution_authority"] == "absent-provider-free-plan-only"
    assert payload["dispatch"] == {
        "enabled": False,
        "entrypoint": None,
        "provider_calls": False,
    }
    assert payload["spend"]["authorized"] is False
    assert payload["spend"]["authorized_usd"] == "0"
    assert payload["spend"]["per_cell_soft_usd"] == "12"
    assert payload["spend"]["nominal_usd"] == "1080"
    assert payload["spend"]["hard_provider_spend_cap"] is False
    assert payload["claims"] == {
        "diagnostic_only": True,
        "ranking_admission": False,
        "promotion_admission": False,
        "formal_e12_execution_admission": False,
        "adaptive_feedback": False,
    }
    assert payload["parallelism"]["max_parallel_per_backend"] == 1
    assert payload["parallelism"]["first_block_serial"] is True
    assert payload["limits"]["planner_count"] == 3


def test_proposed_cell_argv_is_exact_but_confers_no_dispatch_authority(
    planned: QualityCohortPlan,
) -> None:
    first = planned.cells[0]
    argv = first.proposed_argv
    assert argv[1:4] == ("-m", "crucible.workflows", "deepswe")
    assert argv[argv.index("--spec") + 1] == str(first.root)
    assert argv.count("--task-id") == 1
    assert argv[argv.index("--task-id") + 1] == first.task_id
    assert argv[argv.index("--backend-config") + 1].endswith(
        "backends_codex_astra.yaml"
    )
    assert argv[argv.index("--budget-usd") + 1] == "12"
    assert argv[argv.index("--budget-usd-per-instance") + 1] == "12"
    assert argv.count("--planner-count") == 1
    assert argv[argv.index("--planner-count") + 1] == "3"
    assert argv[argv.index("--backend-limit-mode") + 1] == "diagnostic"
    assert argv[argv.index("--max-patch-bytes") + 1] == "67108864"
    assert "--implement" in argv
    cell_payload = first.identity()
    assert cell_payload["route_authority"] == "static-planning-input-not-dispatch-route"
    assert "dispatch" not in cell_payload


def test_all_proposed_cell_argv_use_importable_project_runtime_and_parse(
    planned: QualityCohortPlan,
) -> None:
    expected_python = _ROOT / ".venv" / "bin" / "python"
    assert {cell.proposed_argv[0] for cell in planned.cells} == {
        str(expected_python)
    }

    prefix = planned.cells[0].proposed_argv[:4]
    probe = subprocess.run(
        (*prefix, "--help"),
        cwd="/",
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr

    for cell in planned.cells:
        args = deepswe._parse_args(list(cell.proposed_argv[4:]))
        assert args.spec == cell.root
        assert args.task_id == [cell.task_id]
        assert (
            args.backend_config.name
            == planned.matrix.arms[cell.arm].routing_path.name
        )
        assert args.lifecycle_out == cell.root / "lifecycle.jsonl"
        assert args.implement is True


def test_plan_digest_and_bytes_are_deterministic(planned: QualityCohortPlan) -> None:
    assert planned.to_json_bytes() == planned.to_json_bytes()
    payload = json.loads(planned.to_json_bytes())
    digest = payload.pop("plan_digest")
    assert digest.startswith("sha256:")
    assert planned.payload()["plan_digest"] == digest


def test_frozen_plan_validation_is_exact_byte_for_byte(
    planned: QualityCohortPlan, tmp_path: Path
) -> None:
    path = tmp_path / "plan.json"
    path.write_bytes(planned.to_json_bytes())
    validate_frozen_plan(path, expected=planned)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DeepSWEQualityCohortError, match="bytes differ"):
        validate_frozen_plan(path, expected=planned)


def test_output_root_cannot_modify_the_frozen_corpus(
    matrix_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)
    with pytest.raises(DeepSWEQualityCohortError, match="must remain outside"):
        build_quality_cohort_plan(
            matrix,
            dataset_path=corpus.manifest_path,
            output_root=corpus.root / "run",
        )


def test_output_root_symlink_with_nonexistent_tail_cannot_enter_corpus(
    matrix_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    alias = tmp_path / "corpus-alias"
    alias.symlink_to(corpus.root, target_is_directory=True)
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)

    with pytest.raises(DeepSWEQualityCohortError, match="outside"):
        build_quality_cohort_plan(
            matrix,
            dataset_path=corpus.manifest_path,
            output_root=alias / "not-yet-created" / "run",
        )
    assert not (corpus.root / "not-yet-created").exists()


def test_caller_cannot_mutate_a_loader_produced_matrix(
    matrix_bundle: Path, tmp_path: Path
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    changed = dataclasses.replace(matrix, source_digest="sha256:" + "0" * 64)
    with pytest.raises(DeepSWEQualityCohortError, match="differs from source bytes"):
        build_quality_cohort_plan(
            changed,
            dataset_path=tmp_path / "manifest.json",
            output_root=tmp_path / "output",
        )


@pytest.mark.parametrize("mutate_after_load", (False, True))
def test_selected_instances_leave_the_same_before_after_corpus_authority(
    matrix_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate_after_load: bool,
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus_root = tmp_path / "authority-corpus"
    manifest = corpus_root / "tasks" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("fixture\n", encoding="utf-8")
    selected = tuple(
        SimpleNamespace(
            task_id=definition.task_id,
            repo=definition.repo,
            language=definition.language,
            category=definition.category,
            base_commit_hash=f"{index:040x}",
            full_record_marker=f"full-{index}",
        )
        for index, definition in enumerate(matrix.tasks, start=1)
    )
    fillers = tuple(
        SimpleNamespace(
            task_id=f"fixture-filler-{index}",
            repo=f"fixture/filler-{index}",
            language="python",
            category="feature_request",
            base_commit_hash=f"{index + 100:040x}",
        )
        for index in range(matrix.corpus.task_count - len(selected))
    )
    expected_git = (
        matrix.corpus.git_commit,
        matrix.corpus.git_tree,
        matrix.corpus.clean_status_digest,
    )
    changed_git = (expected_git[0], "0" * 40, expected_git[2])
    measurements = iter(
        (expected_git, changed_git if mutate_after_load else expected_git)
    )
    manifest_bytes = json.dumps(
        {
            "schema_version": matrix.corpus.manifest_schema,
            "task_count": matrix.corpus.task_count,
        }
    ).encode()
    monkeypatch.setattr(convergence, "load_quality_cohort_matrix", lambda _path: matrix)
    monkeypatch.setattr(convergence, "_measure_git", lambda _root: next(measurements))
    monkeypatch.setattr(
        convergence,
        "_read_regular",
        lambda *_args, **_kwargs: manifest_bytes,
    )
    monkeypatch.setattr(
        convergence, "_digest_bytes", lambda _payload: matrix.corpus.manifest_digest
    )
    monkeypatch.setattr(
        convergence, "load_dataset", lambda _path: (*selected, *fillers)
    )
    monkeypatch.setattr(convergence, "_validate_audited_selection", lambda *_args: None)
    monkeypatch.setattr(
        convergence,
        "fingerprint_task_tests_tree",
        lambda *_args: DeepSWETestsTreeIdentity(
            sha256="sha256:" + "a" * 64,
            file_count=2,
            directory_count=1,
            total_bytes=10,
        ),
    )

    if mutate_after_load:
        with pytest.raises(
            DeepSWEQualityCohortError, match="changed during validation"
        ):
            convergence.validate_quality_corpus(matrix, manifest)
    else:
        binding = convergence.validate_quality_corpus(matrix, manifest)
        assert binding.instances == selected
        assert [item.full_record_marker for item in binding.instances] == [
            f"full-{index}" for index in range(1, 11)
        ]


def test_cli_has_only_provider_free_actions() -> None:
    with pytest.raises(SystemExit):
        cli._parse_args(
            [
                "run",
                "--dataset",
                "/dataset/tasks/manifest.json",
                "--output-root",
                "/output",
            ]
        )
    with pytest.raises(SystemExit):
        cli._parse_args(
            [
                "implement",
                "--dataset",
                "/dataset/tasks/manifest.json",
                "--output-root",
                "/output",
            ]
        )


def test_cli_dry_run_freeze_and_validate_never_dispatch(
    matrix_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)
    dataset = str(corpus.manifest_path)
    output = str(tmp_path / "proposed-output")
    common = [
        "--matrix",
        str(matrix_bundle),
        "--dataset",
        dataset,
        "--output-root",
        output,
    ]

    assert cli.main(["dry-run", *common]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["dispatch"]["enabled"] is False

    plan_path = tmp_path / "frozen" / "plan.json"
    assert cli.main(["freeze", *common, "--plan-out", str(plan_path)]) == 0
    frozen = plan_path.read_bytes()
    assert json.loads(frozen)["spend"]["authorized"] is False
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(plan_path.parent.stat().st_mode) == 0o700
    capsys.readouterr()

    assert cli.main(["validate", *common, "--plan-in", str(plan_path)]) == 0
    capsys.readouterr()
    assert cli.main(["freeze", *common, "--plan-out", str(plan_path)]) == 2
    assert plan_path.read_bytes() == frozen
    assert "refusing overwrite" in capsys.readouterr().err


def test_cli_freeze_rejects_corpus_destinations_before_any_mutation(
    matrix_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    marker = corpus.root / "tracked.txt"
    marker.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(corpus.root)], check=True)
    subprocess.run(["git", "-C", str(corpus.root), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(corpus.root),
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    before = subprocess.run(
        ["git", "-C", str(corpus.root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before == ""
    alias = tmp_path / "corpus-link"
    alias.symlink_to(corpus.root, target_is_directory=True)
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)
    common = [
        "--matrix",
        str(matrix_bundle),
        "--dataset",
        str(corpus.manifest_path),
        "--output-root",
        str(tmp_path / "proposed-output"),
    ]
    destinations = (
        corpus.root,
        corpus.root / "direct-tail" / "plan.json",
        alias / "symlink-tail" / "plan.json",
    )
    for destination in destinations:
        assert cli.main(["freeze", *common, "--plan-out", str(destination)]) == 2
        assert "must remain outside" in capsys.readouterr().err

    after = subprocess.run(
        ["git", "-C", str(corpus.root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert after == before
    assert not (corpus.root / "direct-tail").exists()
    assert not (corpus.root / "symlink-tail").exists()


def test_cli_freeze_rejects_plan_below_proposed_output_before_mkdir(
    matrix_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)
    output_root = tmp_path / "proposed-output"
    plan_out = output_root / "control" / "plan.json"

    assert (
        cli.main(
            [
                "freeze",
                "--matrix",
                str(matrix_bundle),
                "--dataset",
                str(corpus.manifest_path),
                "--output-root",
                str(output_root),
                "--plan-out",
                str(plan_out),
            ]
        )
        == 2
    )
    assert "must remain outside" in capsys.readouterr().err
    assert not output_root.exists()


def test_cli_freeze_rejects_symlink_inserted_after_path_validation(
    matrix_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix = load_quality_cohort_matrix(matrix_bundle)
    corpus = _fake_corpus(matrix, tmp_path)
    marker = corpus.root / "tracked.txt"
    marker.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(corpus.root)], check=True)
    subprocess.run(["git", "-C", str(corpus.root), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(corpus.root),
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(convergence, "validate_quality_corpus", lambda *_args: corpus)
    destination = tmp_path / "post-validation-race" / "plan.json"
    original_validate = convergence.validate_plan_output_path
    raced = False

    def validate_then_insert_symlink(
        plan: QualityCohortPlan, plan_path: Path | str
    ) -> Path:
        nonlocal raced
        resolved = original_validate(plan, plan_path)
        assert not destination.parent.exists()
        destination.parent.symlink_to(corpus.root, target_is_directory=True)
        raced = True
        return resolved

    monkeypatch.setattr(
        convergence,
        "validate_plan_output_path",
        validate_then_insert_symlink,
    )
    result = cli.main(
        [
            "freeze",
            "--matrix",
            str(matrix_bundle),
            "--dataset",
            str(corpus.manifest_path),
            "--output-root",
            str(tmp_path / "proposed-output"),
            "--plan-out",
            str(destination),
        ]
    )

    assert raced is True
    assert result == 2
    assert "cannot safely freeze" in capsys.readouterr().err
    assert destination.parent.is_symlink()
    assert not (corpus.root / "plan.json").exists()
    status = subprocess.run(
        ["git", "-C", str(corpus.root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""


def test_freeze_removes_exact_created_leaf_when_write_fails(
    planned: QualityCohortPlan,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "write-failure" / "plan.json"

    def fail_write(_descriptor: int, _payload: object) -> int:
        raise OSError("deterministic write failure")

    monkeypatch.setattr(convergence.os, "write", fail_write)
    with pytest.raises(DeepSWEQualityCohortError, match="cannot safely freeze"):
        convergence.freeze_quality_cohort_plan(planned, destination)

    assert not destination.exists()
    assert not (planned.corpus.root / "plan.json").exists()


def test_freeze_dirfd_confinement_cleans_leaf_after_parent_name_swap(
    planned: QualityCohortPlan,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "replaceable-parent" / "plan.json"
    destination.parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    original_open = os.open
    raced = False

    def swap_parent_before_leaf_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal raced
        if path == destination.name and flags & os.O_CREAT and not raced:
            destination.parent.rename(displaced_parent)
            destination.parent.symlink_to(
                planned.corpus.root,
                target_is_directory=True,
            )
            raced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(convergence.os, "open", swap_parent_before_leaf_open)
    with pytest.raises(DeepSWEQualityCohortError, match="identity changed"):
        convergence.freeze_quality_cohort_plan(planned, destination)

    assert raced is True
    assert not (planned.corpus.root / "plan.json").exists()
    assert not (displaced_parent / "plan.json").exists()


def test_exclusions_are_exact_and_selected_tasks_are_disjoint() -> None:
    assert EXCLUDED_TASK_IDS == (
        "adaptix-name-mapping-aliases",
        "ts-pattern-match-each",
    )
    assert EXCLUDED_REPOS == ("prometheus/prometheus",)
    assert not set(EXCLUDED_TASK_IDS) & set(QUALITY_PILOT_TASK_IDS)
