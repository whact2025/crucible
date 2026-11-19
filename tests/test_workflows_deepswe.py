# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows.deepswe``.

Hermetic — uses ``tmp_path`` to construct a fake deep-swe tasks
tree per test (same fixture builders as
``tests/test_deepswe_dataset.py``); validates CLI parsing, dataset
load + filter dispatch, and the validation-only modes (``--list``,
``--dry-run``). The ``--implement`` cascade path is covered separately by
container-backed smoke tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
import stat
import subprocess
from pathlib import Path

import pytest
from assay import Usage, UsagePricingEvidence, UsagePricingSegment
from assay.corpus.runs import RunHandle, RunResult
from assay.corpus.store import Corpus

from crucible.products.deepswe.secure_artifacts import RewardTestInventory
from crucible.workflows import (
    _deepswe_baseline_preflight as baseline_preflight,
    _outcome,
    deepswe as wf,
)
from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows._run import open_run_with_capture


# ---------- fixture builders (mirrors test_deepswe_dataset.py) ----------


def _prepare_parent_run(path: Path) -> Path:
    path.mkdir(parents=True)
    _outcome.begin_run(path, workflow="deepswe")
    return path


def _init_preflight_checkout(path: Path, *, label: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / ".gitignore").write_text("ignored/\nnew-ignored/\nexisting/generated/\n")
    (path / "tracked.txt").write_text(f"{label}-committed\n")
    (path / "existing").mkdir()
    (path / "existing" / "tracked.txt").write_text(f"{label}-nested\n")
    subprocess.run(
        ["git", "add", ".gitignore", "tracked.txt", "existing/tracked.txt"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=path,
        check=True,
    )
    # Inherited dirt is part of the exact preflight baseline and must survive.
    (path / "tracked.txt").write_text(f"{label}-inherited-dirty\n")
    (path / "inherited-untracked.txt").write_text(f"{label}-untracked\n")
    (path / "ignored").mkdir()
    (path / "ignored" / "state.bin").write_bytes(f"{label}-ignored".encode())


def _add_preflight_submodule(
    workspace: Path,
    *,
    origin: Path,
) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(origin),
            "existing/nested",
        ],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
            "-q",
        ],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-qm",
            "add nested dependency",
        ],
        cwd=workspace,
        check=True,
    )


def _preflight_checkout_image(path: Path) -> tuple[bytes, tuple]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    entries = []
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        metadata = item.lstat()
        if item.is_symlink():
            payload = item.readlink().as_posix()
        elif item.is_file():
            payload = item.read_bytes()
        else:
            payload = None
        entries.append(
            (
                relative,
                stat.S_IFMT(metadata.st_mode),
                metadata.st_mode & 0o7777,
                payload,
            )
        )
    return status, tuple(entries)


def _empty_candidate() -> wf.DeepSWECandidate:
    """Build a valid zero-byte candidate for deterministic workflow tests."""

    digest = hashlib.sha256(b"").hexdigest()
    return wf.DeepSWECandidate(
        patch=b"",
        status=wf.CandidateCaptureStatus.EMPTY,
        diff_ref=f"sha256:{digest}",
        sha256=digest,
        diff_bytes=0,
        files_modified=(),
    )


def _exact_empty_candidate_evidence(inst) -> dict:
    candidate = _empty_candidate()
    return {
        "candidate": candidate,
        "pristine_replay": wf.PristineReplayReport(
            status=wf.PristineReplayStatus.PASSED,
            base_commit=inst.base_commit_hash,
            candidate_sha256=candidate.sha256,
            observed_sha256=candidate.sha256,
        ),
        "verifier_patch": wf.VerifierPatchReport(
            status=wf.VerifierPatchStatus.MATCHED,
            candidate_sha256=candidate.sha256,
            observed_sha256=candidate.sha256,
            observed_bytes=candidate.diff_bytes,
        ),
    }


def _publish_architect_child(
    spec_dir: Path,
    *,
    cost_usd: float,
    raise_after_summary: bool = False,
    status: str = "complete",
    decision: str | None = None,
    tiers_executed: int | None = None,
    failure_phase: str | None = None,
    failure_kind: str | None = None,
    reason: str | None = None,
    development_budget: dict | None = None,
) -> int:
    """Publish a real typed Architect child for saga-bound unit tests."""

    corpus = Corpus(spec_dir)
    with open_run_with_capture(corpus, "architect") as handle:
        summary = {
            "status": status,
            "authoritative_spent_usd": cost_usd,
            "costs": {
                "total_usd": cost_usd,
                "rollup_incomplete": False,
            },
        }
        if decision is not None:
            summary["decision"] = decision
        if tiers_executed is not None:
            summary.update(
                {
                    "tiers_executed": tiers_executed,
                    "tier_results": [],
                }
            )
        if failure_phase is not None:
            summary["failure_phase"] = failure_phase
        if failure_kind is not None:
            summary["failure_kind"] = failure_kind
        if reason is not None:
            summary["reason"] = reason
        if development_budget is not None:
            summary["development_budget"] = development_budget
        handle.write_summary(summary)
        if raise_after_summary:
            raise RuntimeError("architect exploded after billing")
        handle.finalize(RunResult(status=status, cost_usd=cost_usd))
    return 0


def _write_task_dir(
    tasks_root: Path,
    *,
    task_id: str,
    ext_id: str = "khfixturedefault",
    repository_url: str = "https://github.com/ex/ex",
    base_commit_hash: str = "0123456789abcdef0123456789abcdef01234567",
    language: str = "python",
    category: str = "feature_request",
    instruction: str = "Default instruction body.",
) -> Path:
    task_dir = tasks_root / task_id
    task_dir.mkdir(parents=True)
    docker_image = f"public.ecr.aws/d3j8x8q7/swe-bench-202605:{ext_id}"
    (task_dir / "task.toml").write_text(f'''\
schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "datacurve/{task_id}"
description = ""
authors = []
keywords = []

[metadata]
ext_id = "{ext_id}"
task_id = "{task_id}"
display_title = "T"
display_description = "D"
original_title = "Original title"
repository_url = "{repository_url}"
base_commit_hash = "{base_commit_hash}"
category = "{category}"
language = "{language}"

[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0
env = {{}}

[verifier.environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 8192
storage_mb = 20480

[agent]
network_mode = "no-network"
timeout_sec = 5400.0

[environment]
docker_image = "{docker_image}"
build_timeout_sec = 1800.0
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
mcp_servers = []
env = {{}}

[solution]
env = {{}}
''')
    (task_dir / "instruction.md").write_text(instruction)
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    test_script = tests_dir / "test.sh"
    test_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    test_script.chmod(0o755)
    return task_dir


def _write_manifest(tasks_root: Path, entries: list[dict]) -> Path:
    manifest = {
        "schema_version": "1.0",
        "dataset": "deep-swe-1-1",
        "source_dataset": "swe-bench-ultra",
        "task_count": len(entries),
        "id_policy": {
            "format": "lowercase kebab-case, filesystem-safe, <=56 chars",
            "source": (
                "wide-research AI summary of repository, original title, "
                "and instruction excerpt"
            ),
            "original_id_location": "task.toml [metadata].ext_id",
            "new_id_location": "task.toml [metadata].task_id",
        },
        "label_generation": {
            "tool": "fixture",
            "model": "fixture",
            "run_dir": "fixture",
            "results_jsonl": "fixture",
            "prompt_config": "fixture",
        },
        "tasks": entries,
    }
    p = tasks_root / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _entry(
    task_id: str,
    *,
    ext_id: str = "khfixture",
    repo: str = "ex/ex",
    language: str = "python",
    category: str = "feature_request",
) -> dict:
    return {
        "ext_id": ext_id,
        "task_id": task_id,
        "display_title": "T",
        "display_description": "D",
        "original_title": "Original title",
        "repo": repo,
        "repository_url": f"https://github.com/{repo}",
        "language": language,
        "category": category,
    }


def _declare_test_whitelist(
    manifest: Path,
    task_id: str,
    *,
    f2p: list[str],
    p2p: list[str],
) -> None:
    """Give one fixture task the ``tests/config.json`` of a v1.1 task."""

    (manifest.parent / task_id / "tests" / "config.json").write_text(
        json.dumps(
            {
                "base_commit": "0123456789abcdef0123456789abcdef01234567",
                "f2p_node_ids": f2p,
                "grade": {"format": "ctrf", "reports": []},
                "p2p_node_ids": p2p,
            }
        ),
        encoding="utf-8",
    )


def _v11_reward(
    *,
    f2p_passed: int,
    f2p_total: int,
    p2p_passed: int,
    p2p_total: int,
) -> dict:
    """Build a reward document exactly as the dataset's ``grader.py`` does."""

    total = f2p_total + p2p_total
    return {
        "reward": int(
            f2p_total > 0
            and f2p_passed == f2p_total
            and p2p_passed == p2p_total
        ),
        "f2p_total": f2p_total,
        "f2p_passed": f2p_passed,
        "p2p_total": p2p_total,
        "p2p_passed": p2p_passed,
        "f2p": f2p_passed / f2p_total if f2p_total else 0.0,
        "p2p": p2p_passed / p2p_total if p2p_total else 1.0,
        "partial": (f2p_passed + p2p_passed) / total if total else 0.0,
    }


def _build_corpus(tmp_path: Path) -> Path:
    """Construct a small 4-task fake corpus and return the manifest
    path. Languages: python/python/go/rust; categories:
    feature_request × 3 + bugfix × 1."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    specs = [
        ("task-py-1", "kh1", "ex/py-a", "python", "feature_request"),
        ("task-py-2", "kh2", "ex/py-b", "python", "bugfix"),
        ("task-go-1", "kh3", "ex/go-a", "go", "feature_request"),
        ("task-rs-1", "kh4", "ex/rs-a", "rust", "feature_request"),
    ]
    entries = []
    for tid, eid, repo, lang, cat in specs:
        _write_task_dir(
            tasks_root,
            task_id=tid,
            ext_id=eid,
            repository_url=f"https://github.com/{repo}",
            language=lang,
            category=cat,
        )
        entries.append(_entry(tid, ext_id=eid, repo=repo, language=lang, category=cat))
    return _write_manifest(tasks_root, entries)


# ---------- argparse ----------


def test_parse_args_required(tmp_path: Path):
    """--spec and --dataset are required."""
    with pytest.raises(SystemExit):
        wf._parse_args([])


def test_parse_args_minimum(tmp_path: Path):
    ns = wf._parse_args(
        ["--spec", str(tmp_path / "s"), "--dataset", str(tmp_path / "d"), "--list"]
    )
    assert ns.list_only is True
    assert ns.dry_run is False
    assert ns.implement is False
    assert ns.task_id == []
    assert ns.limit is None
    assert ns.planner_count == 3
    # --no-cascade removed from argparse (slice 5a) — accessing the
    # attribute should fail, confirming the flag is gone.
    assert not hasattr(ns, "no_cascade")


def test_parse_args_canonicalizes_trusted_roots_once(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            "spec-root",
            "--dataset",
            "dataset/manifest.json",
            "--backend-config",
            "routing.yaml",
            "--lifecycle-out",
            "state/lifecycle.jsonl",
            "--implement", "--planner-count", "1",
        ]
    )

    assert ns.spec == (tmp_path / "spec-root").resolve()
    assert ns.dataset == (tmp_path / "dataset/manifest.json").resolve()
    assert ns.backend_config == (tmp_path / "routing.yaml").resolve()
    assert ns.lifecycle_out == (tmp_path / "state/lifecycle.jsonl").resolve()


def test_parse_args_filters_repeatable(tmp_path: Path):
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path),
            "--dataset",
            str(tmp_path / "d.json"),
            "--task-id",
            "t1",
            "--task-id",
            "t2",
            "--repo",
            "r/a",
            "--repo",
            "r/b",
            "--language",
            "python",
            "--language",
            "go",
            "--category",
            "feature_request",
            "--limit",
            "5",
            "--list",
        ]
    )
    assert ns.task_id == ["t1", "t2"]
    assert ns.repo == ["r/a", "r/b"]
    assert ns.language == ["python", "go"]
    assert ns.category == ["feature_request"]
    assert ns.limit == 5


def test_parse_args_modes_mutually_exclusive(tmp_path: Path):
    with pytest.raises(SystemExit):
        wf._parse_args(
            [
                "--spec",
                str(tmp_path),
                "--dataset",
                str(tmp_path / "d.json"),
                "--list",
                "--implement", "--planner-count", "1",
            ]
        )


def test_parse_args_cascade_knobs(tmp_path: Path):
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path),
            "--dataset",
            str(tmp_path / "d.json"),
            "--implement",
            "--budget-usd",
            "100",
            "--budget-usd-per-instance",
            "15",
            "--max-turns",
            "75",
            "--no-include-breach",
            "--skip-disk-check",
        ]
    )
    assert ns.budget_usd == 100.0
    assert ns.budget_usd_per_instance == 15.0
    assert ns.max_turns == 75
    assert ns.planner_tool_call_limit == wf.DEFAULT_PLANNER_TOOL_CALL_LIMIT
    assert ns.planner_tool_result_bytes == wf.DEFAULT_PLANNER_TOOL_RESULT_BYTES
    assert ns.no_include_breach is True
    assert ns.skip_disk_check is True
    assert ns.planner_count == 3


@pytest.mark.parametrize("budget", [None, "inf", "nan", "-1"])
def test_competition_rejects_missing_or_nonfinite_budget_before_setup(
    tmp_path: Path, monkeypatch, budget: str | None,
) -> None:
    monkeypatch.setattr(
        wf, "docker_available", lambda: pytest.fail("must reject before Docker"),
    )
    argv = ["--spec", str(tmp_path / "s"), "--dataset", "missing", "--implement"]
    if budget is not None:
        argv.extend(["--budget-usd-per-instance", budget])
    with pytest.raises(SystemExit) as exc:
        wf.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("count", ["0", "9", "1.5"])
def test_competition_rejects_invalid_planner_count(tmp_path: Path, count: str) -> None:
    with pytest.raises(SystemExit) as exc:
        wf._parse_args([
            "--spec", str(tmp_path), "--dataset", "missing", "--list",
            "--planner-count", count,
        ])
    assert exc.value.code == 2


def test_competition_shares_one_task_budget_and_source_tranche(tmp_path: Path) -> None:
    manifest = _build_corpus(tmp_path)
    args = wf._parse_args([
        "--spec", str(tmp_path / "s"), "--dataset", str(manifest),
        "--implement", "--budget-usd-per-instance", "12",
    ])
    inst = wf._load_and_filter(args)[0]
    fingerprint = wf._deepswe_checkpoint_fingerprint(
        inst, args=args, child_budget_usd=12.0,
    )
    kwargs = wf._build_run_architect_kwargs(
        args=args, inst=inst, instance_spec_dir=tmp_path / "instance",
        container_id="author", instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace", budget_usd=12.0,
        execution_fingerprint=fingerprint,
    )
    assert kwargs["planner_count"] == 3
    assert kwargs["budget_usd"] == 12.0
    assert kwargs["planner_tool_call_tranches"] == (32,)
    assert kwargs["planner_tool_call_limit"] == 32
    assert kwargs["planner_tool_result_bytes"] == 262_144
    assert kwargs["planning_policy"].planner_count == 3
    assert kwargs["planning_policy"].planning_budget_share == 0.4
    assert kwargs["planning_policy"].execution_fingerprint == fingerprint
    policy = kwargs["development_budget_policy"]
    assert policy.task_budget_usd == 12.0
    assert policy.planning_review_cap_usd == pytest.approx(4.8)
    assert policy.implementation_verifier_reserve_usd == 6.0
    assert policy.contingency_usd == pytest.approx(1.2)


# ---------- _load_and_filter ----------


def test_load_and_filter_no_filters(tmp_path: Path):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--list",
        ]
    )
    filtered = wf._load_and_filter(ns)
    assert len(filtered) == 4


def test_load_and_filter_by_language(tmp_path: Path):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--language",
            "python",
            "--list",
        ]
    )
    filtered = wf._load_and_filter(ns)
    assert len(filtered) == 2
    assert all(i.language == "python" for i in filtered)


def test_load_and_filter_combined(tmp_path: Path):
    """language=python AND category=feature_request → exactly 1."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--language",
            "python",
            "--category",
            "feature_request",
            "--list",
        ]
    )
    filtered = wf._load_and_filter(ns)
    assert len(filtered) == 1
    assert filtered[0].task_id == "task-py-1"


# ---------- --list mode ----------


def test_list_mode_prints_tab_separated(tmp_path: Path, capsys):
    manifest = _build_corpus(tmp_path)
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--list",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 4
    # Each line: task_id\trepo\tlanguage\tcategory
    fields = [line.split("\t") for line in captured]
    for row in fields:
        assert len(row) == 4
    # All four tasks present
    assert {row[0] for row in fields} == {
        "task-py-1",
        "task-py-2",
        "task-go-1",
        "task-rs-1",
    }


def test_list_mode_honors_filters(tmp_path: Path, capsys):
    manifest = _build_corpus(tmp_path)
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--language",
            "rust",
            "--list",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 1
    assert captured[0].startswith("task-rs-1\t")


# ---------- --dry-run mode ----------


def test_dry_run_prints_plan(tmp_path: Path, capsys):
    manifest = _build_corpus(tmp_path)
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "DeepSWE dry-run plan: 4 tasks" in out
    # Per-task line should carry image + base commit prefix + language.
    assert "public.ecr.aws/d3j8x8q7/swe-bench-202605:" in out
    assert "instruction=" in out
    # Verify all four tasks appear
    for tid in ("task-py-1", "task-py-2", "task-go-1", "task-rs-1"):
        assert tid in out


# ---------- error paths ----------


def test_main_missing_dataset(tmp_path: Path):
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(tmp_path / "missing.json"),
            "--list",
        ]
    )
    assert rc == 2


def test_main_empty_filter_match(tmp_path: Path):
    """Filters that match zero tasks → rc=1 with the filter values
    surfaced in the log."""
    manifest = _build_corpus(tmp_path)
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--language",
            "haskell",
            "--list",
        ]
    )
    assert rc == 1


def test_main_no_mode_set(tmp_path: Path):
    """No --list / --dry-run / --implement / --no-cascade → fail
    fast with rc=2 (operator forgot to pick a mode)."""
    manifest = _build_corpus(tmp_path)
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
        ]
    )
    assert rc == 2


def test_main_no_cascade_flag_absent(tmp_path: Path):
    """--no-cascade was removed from argparse in slice 5a (was a
    placeholder rc=2 land mine). Passing it now should fail at
    argparse with rc=2 from a "unrecognized argument" error,
    NOT a silent accept that maps to an unimplemented branch."""
    manifest = _build_corpus(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        wf.main(
            [
                "--spec",
                str(tmp_path / "s"),
                "--dataset",
                str(manifest),
                "--no-cascade",
            ]
        )
    assert excinfo.value.code == 2


def test_main_implement_no_docker(tmp_path: Path, monkeypatch):
    """--implement gates on docker_available; missing docker → rc=2."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    manifest = _build_corpus(tmp_path)
    backend_config = tmp_path / "backends.yaml"
    backend_config.write_text(
        "schema_version: crucible.backend-routing.v1\n"
        "default:\n"
        "  backend: codex\n"
        "  model: gpt-5.6-sol\n"
        "  effort: max\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wf, "docker_available", lambda: False)
    rc = wf.main(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--backend-config",
            str(backend_config),
            "--implement", "--planner-count", "1",
        ]
    )
    assert rc == 2

    from crucible.backends.config import (
        load_routing,
        routing_feature_fingerprint,
        routing_fingerprint,
        routing_source_fingerprint,
    )

    routing = load_routing(
        config_path=backend_config,
        spec_dir=tmp_path / "s",
    )
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    provenance_inputs = run_manifest["provenance"]["inputs"]
    assert provenance_inputs["backend_routing"] == routing_fingerprint(routing)
    assert provenance_inputs["backend_features"] == (
        routing_feature_fingerprint(routing)
    )
    assert provenance_inputs["backend_routing_source_sha256"] == (
        routing_source_fingerprint(routing)
    )
    assert provenance_inputs["backend_routing"] != (f"sha256:{routing.source_sha256}")


# ---------- helper: _build_run_architect_kwargs ----------


def test_build_run_architect_kwargs_plumbs_deepswe_metadata(tmp_path: Path):
    """The kwargs dict passed to run_architect carries
    deepswe_metadata in extra_parameters with container_id +
    logs_dir + task_id + ext_id + language, and roots cwd at the
    host workspace (NOT the instance spec dir)."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    insts = wf._load_and_filter(ns)
    inst = insts[0]

    host_workspace = tmp_path / "workspace"
    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="container_abc",
        instance_log_dir=tmp_path / "logs",
        host_workspace=host_workspace,
    )
    assert "extra_parameters" in kwargs
    meta = kwargs["extra_parameters"]["deepswe_metadata"]
    assert meta["container_id"] == "container_abc"
    assert meta["logs_dir"] == str(tmp_path / "logs")
    assert meta["task_id"] == inst.task_id
    assert meta["ext_id"] == inst.ext_id
    assert meta["max_patch_bytes"] == ns.max_patch_bytes
    # P0 fix: language threads through metadata so deepswe_dev_verify
    # can dispatch its per-language test command (the starter spec
    # carries no LanguagesConstraint, so _inject_languages never sets a
    # top-level ``language``).
    assert meta["language"] == inst.language
    assert "baseline_container_id" not in meta
    assert "image_digest" not in meta
    assert "base_commit" not in meta
    checkpoint_hook = kwargs["checkpoint_hook"]
    assert isinstance(checkpoint_hook, wf.DeepSWEArchitectCheckpointHook)
    assert checkpoint_hook.task_id == inst.task_id
    assert checkpoint_hook.ext_id == inst.ext_id
    assert checkpoint_hook.expected_base_commit == inst.base_commit_hash
    assert checkpoint_hook.max_patch_bytes == ns.max_patch_bytes
    # P1b fix: cwd is the host workspace (the real git checkout where
    # edits land + the diff is captured + MCP tools root), NOT the
    # Corpus spec dir. spec stays the Corpus dir.
    assert kwargs["cwd"] == str(host_workspace)
    assert kwargs["spec"] == tmp_path / "instance_spec"
    assert kwargs["backend_limit_mode"] is None
    assert kwargs["backend_limit_deadline_seconds"] is None
    # Policies should be wired through
    assert kwargs["review_policy"] is not None
    assert kwargs["breach_policy"] is not None
    # Default breach is enabled (no --no-include-breach)
    assert kwargs["breach_policy"].enabled is True
    # Feature-work plan critic threaded; replan defaults off so a
    # resumed run reuses the prior tier_dag rather than re-paying plan.
    assert kwargs["critic_rubric"] == "deepswe_plan_critic"
    # Benchmark planning is source-grounded, but Architect exposes this
    # ToolSet through its read_only capability only.
    assert kwargs["enable_source_tools"] is True
    assert kwargs["planner_tool_call_limit"] == wf.DEFAULT_PLANNER_TOOL_CALL_LIMIT
    assert kwargs["planner_count"] == 1
    assert kwargs["budget_usd"] is None
    assert kwargs["planning_policy"].planning_budget_share is None
    assert wf._deepswe_planning_envelope(
        None, planner_count=1,
    )["planning_budget_share"] is None


def test_build_run_architect_kwargs_forwards_invocation_limit_policy(
    tmp_path: Path,
) -> None:
    manifest = _build_corpus(tmp_path)
    args = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--backend-limit-mode",
            "diagnostic",
            "--backend-limit-deadline-seconds",
            "1200",
        ]
    )
    instance = wf._load_and_filter(args)[0]

    kwargs = wf._build_run_architect_kwargs(
        args=args,
        inst=instance,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="container_abc",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )

    assert kwargs["backend_limit_mode"] == "diagnostic"
    assert kwargs["backend_limit_deadline_seconds"] == 1200.0
    assert kwargs["planner_tool_result_bytes"] == wf.DEFAULT_PLANNER_TOOL_RESULT_BYTES
    assert kwargs["replan"] is False
    # In-container model: single persistent bind-mounted workspace, so
    # the cascade must NOT re-apply dependency diffs between tiers (the
    # tier-handoff double-apply bug the adaptix pilot surfaced).
    assert kwargs["persistent_workspace"] is True
    assert kwargs["fresh_lineage"] is False


def test_build_run_architect_kwargs_selects_generic_critic_at_compact_boundary(
    tmp_path: Path,
) -> None:
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]

    def build(budget: float) -> dict:
        return wf._build_run_architect_kwargs(
            args=ns,
            inst=inst,
            instance_spec_dir=tmp_path / f"instance-{budget}",
            container_id="container",
            instance_log_dir=tmp_path / "logs",
            host_workspace=tmp_path / "workspace",
            budget_usd=budget,
        )

    compact = build(12.0)
    standard = build(12.01)
    assert compact["critic_rubric"] == "plan_critic"
    assert compact["budget_usd"] == 12.0
    assert standard["critic_rubric"] == "deepswe_plan_critic"
    assert standard["budget_usd"] == 12.01
    assert wf._deepswe_planning_envelope(12.0) == {
        **wf.benchmark_planning_identity(3),
        "class": "compact_one_tier_one_slice",
        "compact_max_budget_usd": 12.0,
        "critic_rubric": "plan_critic",
        "max_total_tiers": 1,
        "max_total_slices": 1,
    }
    assert wf._deepswe_planning_envelope(12.01)["class"] == (
        "standard_feature_decomposition"
    )


def test_task_corpus_rejects_planning_envelope_class_drift(
    tmp_path: Path,
) -> None:
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(tmp_path / "instance-corpus")
    corpus.put_spec(synthesize_starter_spec(inst, task_budget_usd=12.01))

    with pytest.raises(ValueError, match="planning-envelope policy"):
        wf._prepare_task_corpus(
            corpus,
            synthesize_starter_spec(inst, task_budget_usd=12.0),
            task_id=inst.task_id,
        )


def test_build_run_architect_kwargs_threads_distinct_baseline_identity(
    tmp_path: Path,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="candidate-cid",
        baseline_container_id="baseline-cid",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )
    meta = kwargs["extra_parameters"]["deepswe_metadata"]
    assert meta["container_id"] == "candidate-cid"
    assert meta["baseline_container_id"] == "baseline-cid"
    assert meta["baseline_container_id"] != meta["container_id"]
    assert meta["image_digest"] == inst.docker_image
    assert meta["base_commit"] == inst.base_commit_hash


def test_development_preflight_parameters_match_architect_projection(
    tmp_path: Path,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    logs_dir = tmp_path / "logs"
    parameters = baseline_preflight.build_development_preflight_parameters(
        instance=inst,
        candidate_container_id="candidate-cid",
        baseline_container_id="baseline-cid",
        logs_dir=logs_dir,
        max_patch_bytes=ns.max_patch_bytes,
    )
    architect = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="candidate-cid",
        baseline_container_id="baseline-cid",
        instance_log_dir=logs_dir,
        host_workspace=tmp_path / "workspace",
    )

    assert parameters == architect["extra_parameters"]
    assert set(parameters) == {"deepswe_metadata"}
    assert set(parameters["deepswe_metadata"]) == {
        "base_commit",
        "baseline_container_id",
        "container_id",
        "ext_id",
        "image_digest",
        "language",
        "logs_dir",
        "max_patch_bytes",
        "task_id",
    }
    assert not any(
        "backend" in key or "provider" in key for key in parameters["deepswe_metadata"]
    )


def test_development_preflight_has_closed_provider_free_imports():
    source = Path(baseline_preflight.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "assay" or node.module.startswith("crucible."))
        for alias in node.names
    }

    assert imports == {
        ("assay", "Spec"),
        ("crucible.products.deepswe.dataset", "DeepSWEInstance"),
        (
            "crucible.products.deepswe.dev_verify_overrides",
            "resolve_dev_verify_command",
        ),
        ("crucible.common.benchmark_runtime", "canonical_fingerprint"),
        ("crucible.workflows", "_build_verify"),
        ("crucible.workflows", "_mutation_journal"),
        ("crucible.workflows._build_verify", "BuildVerifyOutcome"),
    }
    assert not any(
        alias.name == "assay"
        or alias.name.startswith("assay.")
        or alias.name == "crucible"
        or alias.name.startswith("crucible.")
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "BackendPool" not in source
    assert "run_architect" not in source


def _restoration_receipt_payload(tmp_path: Path) -> dict[str, object]:
    candidate = tmp_path / "candidate-receipt-root"
    baseline = tmp_path / "baseline-receipt-root"
    candidate.mkdir()
    baseline.mkdir()
    candidate_stat = candidate.stat()
    baseline_stat = baseline.stat()
    nested_device = max(candidate_stat.st_dev, baseline_stat.st_dev) + 1
    receipt = baseline_preflight.DevelopmentPreflightRestorationReceipt(
        workspaces=(
            baseline_preflight.DevelopmentPreflightWorkspaceRestoration(
                label="candidate",
                workspace=str(candidate.resolve()),
                workspace_device=candidate_stat.st_dev,
                workspace_inode=candidate_stat.st_ino,
                namespace=("nested",),
                split_namespaces=(
                    ("nested", ("keep.txt", "repo"), nested_device, 1),
                ),
                initialized_gitlinks=(
                    (
                        "nested/repo",
                        nested_device,
                        2,
                        "c" * 32,
                        "dev_verify_preflight__candidate::gitlink::nested/repo",
                    ),
                    (
                        "nested/repo/deep",
                        nested_device,
                        3,
                        "d" * 32,
                        (
                            "dev_verify_preflight__candidate::gitlink::nested/repo"
                            "::gitlink::deep"
                        ),
                    ),
                ),
                journal_transaction_id="a" * 32,
                journal_family=(
                    ("a" * 32, None),
                    ("c" * 32, "a" * 32),
                    ("d" * 32, "c" * 32),
                ),
                gate_created_paths=("gate-report", "nested/gate-created.lock"),
            ),
            baseline_preflight.DevelopmentPreflightWorkspaceRestoration(
                label="baseline",
                workspace=str(baseline.resolve()),
                workspace_device=baseline_stat.st_dev,
                workspace_inode=baseline_stat.st_ino,
                namespace=(),
                split_namespaces=(),
                initialized_gitlinks=(),
                journal_transaction_id="b" * 32,
                journal_family=(("b" * 32, None),),
                gate_created_paths=(),
            ),
        )
    )
    return receipt.to_payload()


def test_development_preflight_restoration_receipt_round_trips_without_fs(
    tmp_path: Path,
):
    payload = _restoration_receipt_payload(tmp_path)
    expected = baseline_preflight.DevelopmentPreflightRestorationReceipt.from_payload(
        payload
    )
    for raw_workspace in payload["workspaces"]:
        workspace = Path(raw_workspace["workspace"])
        workspace.rmdir()

    decoded = baseline_preflight.DevelopmentPreflightRestorationReceipt.from_payload(
        json.loads(json.dumps(payload))
    )

    assert decoded == expected
    assert decoded.to_payload() == payload
    assert decoded.fingerprint == expected.fingerprint
    assert set(decoded.to_payload()) == {"schema_version", "workspaces"}

    class _ReceiptSubclass(
        baseline_preflight.DevelopmentPreflightRestorationReceipt
    ):
        pass

    with pytest.raises(ValueError, match="subclasses"):
        _ReceiptSubclass(workspaces=decoded.workspaces)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_receipt_field",
        "unknown_workspace_field",
        "string_subclass",
        "noncanonical_workspace",
        "noncanonical_transaction",
        "bool_identity",
        "unsorted_namespace",
        "escaping_nested_path",
        "cross_duplicate_nested_path",
        "split_below_gitlink",
        "overlapping_identity",
        "malformed_split_shape",
        "incomplete_family",
        "reparented_gitlink_guard",
        "padded_missing_gitlink_guard",
        "duplicate_gitlink_guard",
        "tampered_gitlink_guard_slice",
        "disconnected_family",
        "missing_gate_created_paths",
        "gate_created_paths_not_a_list",
        "unsorted_gate_created_paths",
        "nested_gate_created_paths",
        "escaping_gate_created_path",
        "gate_created_path_below_gitlink",
        "gate_created_path_in_namespace",
        "gate_created_path_in_split_namespace",
    ],
)
def test_development_preflight_restoration_receipt_rejects_tamper(
    tmp_path: Path,
    case: str,
):
    payload = json.loads(json.dumps(_restoration_receipt_payload(tmp_path)))
    workspaces = payload["workspaces"]
    candidate = workspaces[0]
    if case == "unknown_receipt_field":
        payload["unexpected"] = None
    elif case == "unknown_workspace_field":
        candidate["unexpected"] = None
    elif case == "string_subclass":

        class _StringSubclass(str):
            pass

        candidate["label"] = _StringSubclass("candidate")
    elif case == "noncanonical_workspace":
        candidate["workspace"] += "/../candidate-receipt-root"
    elif case == "noncanonical_transaction":
        candidate["journal_transaction_id"] = "A" * 32
    elif case == "bool_identity":
        candidate["workspace_inode"] = True
    elif case == "unsorted_namespace":
        candidate["namespace"] = ["nested", "alpha"]
    elif case == "escaping_nested_path":
        candidate["split_namespaces"][0]["relative"] = "../nested"
    elif case == "cross_duplicate_nested_path":
        candidate["initialized_gitlinks"][0]["relative"] = "nested"
    elif case == "split_below_gitlink":
        candidate["split_namespaces"].append(
            {
                "relative": "nested/repo/child",
                "namespace": [],
                "device": 900_001,
                "inode": 900_001,
            }
        )
    elif case == "overlapping_identity":
        workspaces[1]["workspace_device"] = candidate["initialized_gitlinks"][0][
            "device"
        ]
        workspaces[1]["workspace_inode"] = candidate["initialized_gitlinks"][0][
            "inode"
        ]
    elif case == "malformed_split_shape":
        candidate["split_namespaces"][0] = ["nested"]
    elif case == "incomplete_family":
        candidate["journal_family"].pop()
    elif case == "reparented_gitlink_guard":
        candidate["journal_family"][2]["parent_transaction_id"] = "a" * 32
    elif case == "padded_missing_gitlink_guard":
        candidate["journal_family"].pop()
        candidate["journal_family"].append(
            {
                "transaction_id": "e" * 32,
                "parent_transaction_id": "c" * 32,
            }
        )
    elif case == "duplicate_gitlink_guard":
        candidate["initialized_gitlinks"][1]["guard_transaction_id"] = "c" * 32
    elif case == "tampered_gitlink_guard_slice":
        candidate["initialized_gitlinks"][1]["guard_slice_id"] += "-tampered"
    elif case == "disconnected_family":
        candidate["journal_family"][1]["parent_transaction_id"] = "d" * 32
    elif case == "missing_gate_created_paths":
        del candidate["gate_created_paths"]
    elif case == "gate_created_paths_not_a_list":
        candidate["gate_created_paths"] = "gate-report"
    elif case == "unsorted_gate_created_paths":
        candidate["gate_created_paths"].reverse()
    elif case == "nested_gate_created_paths":
        candidate["gate_created_paths"].insert(1, "gate-report/index.html")
    elif case == "escaping_gate_created_path":
        candidate["gate_created_paths"] = ["../gate-report"]
    elif case == "gate_created_path_below_gitlink":
        candidate["gate_created_paths"] = ["nested/repo/build"]
    elif case == "gate_created_path_in_namespace":
        # A name the checkout had before the gate ran was not created by it.
        candidate["gate_created_paths"] = ["nested"]
    elif case == "gate_created_path_in_split_namespace":
        candidate["gate_created_paths"] = ["nested/keep.txt"]
    else:  # pragma: no cover - parameter inventory is closed above
        raise AssertionError(case)

    with pytest.raises(ValueError, match="development preflight"):
        baseline_preflight.DevelopmentPreflightRestorationReceipt.from_payload(
            payload
        )


def test_architect_metadata_uses_preflight_composer_with_baseline(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    seen: dict[str, object] = {}

    def _compose(**kwargs):
        seen.update(kwargs)
        return {"deepswe_metadata": {"closed_projection": True}}

    monkeypatch.setattr(
        wf,
        "build_development_preflight_parameters",
        _compose,
    )
    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="candidate-cid",
        baseline_container_id="baseline-cid",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )

    assert seen == {
        "instance": inst,
        "candidate_container_id": "candidate-cid",
        "baseline_container_id": "baseline-cid",
        "logs_dir": tmp_path / "logs",
        "max_patch_bytes": ns.max_patch_bytes,
    }
    assert kwargs["extra_parameters"] == {
        "deepswe_metadata": {"closed_projection": True}
    }


def test_build_run_architect_kwargs_threads_dev_verify_override(
    tmp_path: Path,
    monkeypatch,
):
    """A registered per-project override lands in
    deepswe_metadata["dev_verify_command"] so deepswe_dev_verify honors
    it over the per-language default."""
    from crucible.products.deepswe import dev_verify_overrides as ov

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    monkeypatch.setattr(
        ov,
        "_TASK_OVERRIDES",
        {inst.task_id: "cd /app && nox -s test"},
    )

    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="c",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )
    meta = kwargs["extra_parameters"]["deepswe_metadata"]
    assert meta["dev_verify_command"] == "cd /app && nox -s test"


def test_build_run_architect_kwargs_omits_override_when_unregistered(
    tmp_path: Path,
    monkeypatch,
):
    """No override registered → the key is absent (the runner then falls
    back to the per-language default), not present-but-empty."""
    from crucible.products.deepswe import dev_verify_overrides as ov

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    monkeypatch.setattr(ov, "_TASK_OVERRIDES", {})
    monkeypatch.setattr(ov, "_REPO_OVERRIDES", {})

    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="c",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )
    meta = kwargs["extra_parameters"]["deepswe_metadata"]
    assert "dev_verify_command" not in meta


def test_build_run_architect_kwargs_replan_flag_forces_regen(tmp_path: Path):
    """--replan threads replan=True into run_architect so the planner
    regenerates the tier_dag (via deepswe_plan_critic) instead of
    reusing a stale plan carried on the instance Corpus."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--replan",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="c",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )
    assert kwargs["replan"] is True


def test_build_run_architect_kwargs_no_include_breach(tmp_path: Path):
    """--no-include-breach flips BreachPolicy.enabled=False while
    keeping the other policy fields intact."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--no-include-breach",
        ]
    )
    insts = wf._load_and_filter(ns)
    inst = insts[0]

    kwargs = wf._build_run_architect_kwargs(
        args=ns,
        inst=inst,
        instance_spec_dir=tmp_path / "instance_spec",
        container_id="container_abc",
        instance_log_dir=tmp_path / "logs",
        host_workspace=tmp_path / "workspace",
    )
    assert kwargs["breach_policy"].enabled is False
    # max_rounds + rubrics preserved
    from crucible.products.deepswe.spec_synthesis import DEEPSWE_BREACH_POLICY

    assert kwargs["breach_policy"].max_rounds == DEEPSWE_BREACH_POLICY.max_rounds
    assert kwargs["breach_policy"].breaker_rubric == "impl_breaker"


# ---------- _emit_summary ----------


def test_emit_summary_rolls_up_cost_and_counts(tmp_path: Path):
    """_emit_summary writes summary.json with the cost roll-up + task
    tallies. Regression guard: cost_total_usd must read the
    CostAccumulator's ``total_cost_usd`` PROPERTY (not a ``total_usd()``
    method that doesn't exist) — the wrong call shipped once and crashed
    every --implement run with AttributeError at summary time, after the
    cascade had already spent real money."""
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    cost = CostAccumulator()
    task_summaries = [
        {
            "task_id": "t1",
            "ext_id": "e1",
            "status": "resolved",
            "reward": 1.0,
            "command_returncode": 0.0,
            "verify_returncode": 0.0,
            "baseline_test_returncode": 0.0,
            "new_test_returncode": 0.0,
            "diagnostics_persisted": True,
            "verifier_stdout_truncated": False,
            "verifier_stderr_truncated": False,
        },
        {"task_id": "t2", "ext_id": "e2", "status": "cascade_error"},
        {"task_id": "t3", "ext_id": "e3", "status": "unresolved", "reward": 0.0},
    ]

    corpus = wf._open_or_create_corpus(ns.spec)
    with wf.open_run_with_capture(corpus, "deepswe") as handle:
        out = wf._emit_summary(
            handle=handle,
            args=ns,
            task_summaries=task_summaries,
            cost=cost,
            status="tasks_failed",
            halt_reason=None,
            tasks_selected=5,
            tasks_not_started=2,
        )

    payload = json.loads(out.read_text())
    assert payload["tasks_total"] == 3
    assert payload["tasks_ok"] == 1
    assert payload["tasks_failed"] == 2
    assert payload["tasks_resolved"] == 1  # only t1 graded reward==1.0
    assert payload["tasks_selected"] == 5
    assert payload["tasks_not_started"] == 2
    assert payload["tasks_unrecorded"] == 0
    assert payload["tasks_graded"] == 2
    assert payload["tasks_unresolved"] == 1
    assert payload["tasks_ungraded"] == 1
    assert payload["tasks_lease_busy"] == 0
    assert payload["tasks_by_status"] == {
        "cascade_error": 1,
        "resolved": 1,
        "unresolved": 1,
    }
    assert payload["reward_counts"] == {
        "tasks_counted": 0,
        "tasks_apply_failed": 0,
        "f2p_passed": 0,
        "f2p_total": 0,
        "p2p_passed": 0,
        "p2p_total": 0,
    }
    assert payload["cost_total_usd"] == cost.total_cost_usd
    assert payload["tasks"][0]["command_returncode"] == 0.0
    assert payload["tasks"][0]["verify_returncode"] == 0.0
    assert payload["tasks"][0]["baseline_test_returncode"] == 0.0
    assert payload["tasks"][0]["new_test_returncode"] == 0.0
    assert payload["tasks"][0]["diagnostics_persisted"] is True
    assert payload["tasks"][0]["verifier_stdout_truncated"] is False
    assert payload["tasks"][0]["verifier_stderr_truncated"] is False
    assert "costs" not in payload
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_emit_summary_persists_optional_physical_pricing_ledger(tmp_path: Path):
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    segment = UsagePricingSegment(
        input_tokens=20,
        output_tokens=3,
        cache_creation_tokens=4,
        cache_read_tokens=6,
        cost_usd=0.2,
    )
    evidence = UsagePricingEvidence(segments=(segment,), complete=True)
    cost = CostAccumulator()
    cost.record(
        role="planner",
        usage=evidence.usage,
        backend_name="claude",
        model="opus",
        pricing_evidence=evidence,
    )
    corpus = wf._open_or_create_corpus(ns.spec)
    with wf.open_run_with_capture(corpus, "deepswe") as handle:
        out = wf._emit_summary(
            handle=handle,
            args=ns,
            task_summaries=[
                {
                    "task_id": "t1",
                    "ext_id": "e1",
                    "status": "resolved",
                    "cost_incomplete": False,
                }
            ],
            cost=cost,
            status="deepswe_complete",
            halt_reason=None,
            tasks_selected=1,
            tasks_not_started=0,
        )

    payload = json.loads(out.read_text())
    assert payload["cost_total_usd"] == pytest.approx(0.2)
    assert payload["costs"]["total_cache_creation_tokens"] == 4
    assert payload["costs"]["total_cache_read_tokens"] == 6
    assert payload["costs"]["pricing_evidence"]["complete"] is True
    assert [call["role"] for call in payload["costs"]["pricing_evidence_by_call"]] == [
        "planner"
    ]


@pytest.mark.parametrize(
    ("spent_usd", "cost_incomplete", "expected_remaining"),
    [
        (2.0, True, 0.0),
        (6.0, False, 0.0),
        (2.0, False, 3.0),
    ],
)
def test_emit_summary_never_advertises_negative_or_untrusted_budget(
    tmp_path: Path,
    spent_usd: float,
    cost_incomplete: bool,
    expected_remaining: float,
):
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement",
            "--budget-usd",
            "5.0",
        ]
    )
    cost = CostAccumulator()
    cost.record(
        role="deepswe_task",
        usage=Usage(cost_usd=spent_usd),
        backend_name="stub",
    )
    task_summaries = [
        {
            "task_id": "t1",
            "ext_id": "e1",
            "status": "cascade_error",
            "cost_incomplete": cost_incomplete,
        }
    ]
    corpus = wf._open_or_create_corpus(ns.spec)

    with wf.open_run_with_capture(corpus, "deepswe") as handle:
        out = wf._emit_summary(
            handle=handle,
            args=ns,
            task_summaries=task_summaries,
            cost=cost,
            status=("cost_incomplete" if cost_incomplete else "tasks_failed"),
            halt_reason=("cost_incomplete" if cost_incomplete else None),
            tasks_selected=1,
            tasks_not_started=0,
        )

    payload = json.loads(out.read_text())
    assert payload["cost_incomplete"] is cost_incomplete
    assert payload["cost_total_usd"] == pytest.approx(spent_usd)
    assert payload["budget_remaining_usd"] == pytest.approx(expected_remaining)


def _emit_summary_payload(
    tmp_path: Path,
    task_summaries: list[dict],
    *,
    tasks_selected: int,
    tasks_not_started: int,
) -> dict:
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    corpus = wf._open_or_create_corpus(ns.spec)
    with wf.open_run_with_capture(corpus, "deepswe") as handle:
        out = wf._emit_summary(
            handle=handle,
            args=ns,
            task_summaries=task_summaries,
            cost=CostAccumulator(),
            status="tasks_failed",
            halt_reason=None,
            tasks_selected=tasks_selected,
            tasks_not_started=tasks_not_started,
        )
    return json.loads(out.read_text())


def test_emit_summary_separates_ungraded_and_not_started_from_unsolved(
    tmp_path: Path,
):
    """A task without a grade is never counted with the tasks graded zero."""

    statuses = (
        "resolved",
        "unresolved",
        "verification_failed",
        "verification_incomplete",
        "setup_failed",
        "candidate_failed",
        "task_lease_busy",
    )
    payload = _emit_summary_payload(
        tmp_path,
        [
            {"task_id": f"t{index}", "ext_id": f"e{index}", "status": status}
            for index, status in enumerate(statuses)
        ],
        # Ten were selected: seven have a summary, one ended without one, and
        # two were never handed to execution.
        tasks_selected=10,
        tasks_not_started=2,
    )

    assert payload["tasks_selected"] == 10
    assert payload["tasks_total"] == 7
    assert payload["tasks_unrecorded"] == 1
    assert payload["tasks_not_started"] == 2
    assert payload["tasks_resolved"] == 1
    assert payload["tasks_unresolved"] == 1
    assert payload["tasks_graded"] == 2
    assert payload["tasks_ungraded"] == 4
    assert payload["tasks_lease_busy"] == 1
    # setup_failed and candidate_failed carry no reuse authority. The two
    # hidden-verifier terminals are one-shot, and the lease-busy task is
    # another process's work, not a result of this run.
    assert payload["tasks_rerun_by_default"] == 2
    assert payload["tasks_by_status"] == {
        status: 1 for status in sorted(statuses)
    }
    # Existing keys keep their meaning.
    assert payload["tasks_ok"] == 1
    assert payload["tasks_failed"] == 6
    assert payload["tasks_selected"] == (
        payload["tasks_total"]
        + payload["tasks_unrecorded"]
        + payload["tasks_not_started"]
    )
    assert payload["tasks_total"] == (
        payload["tasks_graded"]
        + payload["tasks_ungraded"]
        + payload["tasks_lease_busy"]
    )
    assert payload["tasks_graded"] == (
        payload["tasks_resolved"] + payload["tasks_unresolved"]
    )
    assert payload["tasks_failed"] == (
        payload["tasks_unresolved"]
        + payload["tasks_ungraded"]
        + payload["tasks_lease_busy"]
    )


def test_emit_summary_sums_reward_counts_over_graded_tasks(tmp_path: Path):
    def _counts(f2p_passed, f2p_total, p2p_passed, p2p_total, apply_failed=False):
        return {
            "apply_failed": apply_failed,
            "f2p_passed": f2p_passed,
            "f2p_total": f2p_total,
            "p2p_passed": p2p_passed,
            "p2p_total": p2p_total,
        }

    payload = _emit_summary_payload(
        tmp_path,
        [
            {
                "task_id": "t1",
                "ext_id": "e1",
                "status": "resolved",
                "reward_counts": _counts(3, 3, 120, 120),
            },
            {
                "task_id": "t2",
                "ext_id": "e2",
                "status": "unresolved",
                "reward_counts": _counts(1, 4, 7, 9),
            },
            {
                "task_id": "t3",
                "ext_id": "e3",
                "status": "unresolved",
                "reward_counts": _counts(0, 5, 0, 11, apply_failed=True),
            },
            # Graded from a count-less v1.0 token: graded, but not counted.
            {
                "task_id": "t4",
                "ext_id": "e4",
                "status": "resolved",
                "reward_counts": None,
            },
            # Counts of a task whose verifier result is not a grade stay out.
            {
                "task_id": "t5",
                "ext_id": "e5",
                "status": "verification_failed",
                "reward_counts": _counts(2, 2, 6, 6),
            },
        ],
        tasks_selected=5,
        tasks_not_started=0,
    )

    assert payload["tasks_graded"] == 4
    assert payload["reward_counts"] == {
        "tasks_counted": 3,
        "tasks_apply_failed": 1,
        "f2p_passed": 4,
        "f2p_total": 12,
        "p2p_passed": 127,
        "p2p_total": 140,
    }
    assert payload["tasks"][0]["reward_counts"] == _counts(3, 3, 120, 120)


# ---------- isolated lifecycle: _run_task_cascade ----------


def _patch_container_lifecycle(
    monkeypatch,
    tmp_path: Path,
    calls: dict,
    *,
    candidate=None,
):
    """Stub out every docker-touching surface _run_task_cascade uses,
    recording the calls in ``calls`` so a test can assert the
    workspace/container ordering and mount contract without a live daemon."""

    def _pull(image):
        calls.setdefault("pull", []).append(image)
        return image.split(":", 1)[0] + "@sha256:" + "a" * 64

    monkeypatch.setattr(wf, "pull_image", _pull)

    def _populate(
        image,
        ws,
        *,
        base_commit=None,
        labels=None,
        register_container=None,
    ):
        calls.setdefault("populate", []).append(
            (image, str(ws), base_commit, labels, register_container)
        )
        # The real call returns the checkout's full base-commit id. Fixture
        # tasks pin a full id, so the pin is that id.
        return base_commit

    monkeypatch.setattr(wf, "populate_workspace_from_image", _populate)

    def _auth(
        inst,
        *,
        host_workspace,
        host_logs_dir,
        container_name=None,
        labels=None,
        register_container=None,
    ):
        calls.setdefault("authoring", []).append(
            {
                "name": container_name,
                "workspace": str(host_workspace),
                "logs": str(host_logs_dir),
                "labels": labels,
                "register_container": register_container,
            }
        )
        return (
            "baseline-cid"
            if labels and labels.get("io.crucible.phase") == "baseline"
            else "authoring-cid"
        )

    def _verify(
        inst,
        *,
        host_workspace,
        host_logs_dir,
        host_tests_dir,
        container_name=None,
        labels=None,
        register_container=None,
    ):
        calls.setdefault("events", []).append("start_verifier")
        calls.setdefault("verifier", []).append(
            {
                "name": container_name,
                "workspace": str(host_workspace),
                "logs": str(host_logs_dir),
                "tests_dir": str(host_tests_dir),
                "labels": labels,
                "register_container": register_container,
            }
        )
        return "verifier-cid"

    def _render(
        inst,
        *,
        host_workspace,
        container_name=None,
        labels=None,
        register_container=None,
    ):
        phase = labels.get("io.crucible.phase") if labels else None
        container_id = (
            "source-render-cid"
            if phase == "source-patch-render"
            else "proof-render-cid"
        )
        calls.setdefault("render", []).append(
            {
                "id": container_id,
                "name": container_name,
                "workspace": str(host_workspace),
                "labels": labels,
                "register_container": register_container,
            }
        )
        calls.setdefault("events", []).append(f"start_{container_id}")
        return container_id

    def _teardown(cid, image=None):
        calls.setdefault("teardown", []).append((cid, image))

    def _strict_stop(cid):
        calls.setdefault("events", []).append(f"stop_{cid}")
        calls.setdefault("strict_stop", []).append(cid)

    class _Quota:
        exceeded = False
        peak_growth_bytes = 0

        def __init__(
            self,
            paths,
            *,
            limit_bytes,
            on_exceeded,
            require_kernel,
        ):
            assert require_kernel is True
            calls.setdefault("quota", []).append((tuple(paths), limit_bytes))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(wf, "start_authoring_container", _auth)
    monkeypatch.setattr(wf, "start_patch_render_container", _render)
    monkeypatch.setattr(wf, "start_verifier_container", _verify)
    monkeypatch.setattr(wf, "stop_instance_container_strict", _strict_stop)
    monkeypatch.setattr(wf, "teardown_instance_container", _teardown)
    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _Quota)
    monkeypatch.setattr(
        wf,
        "_run_dev_verify_preflight",
        lambda **_kwargs: BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=RunnerResult(
                role="dev_verify",
                runner=("crucible.products.deepswe.runners.deepswe_dev_verify"),
                returncode=0,
                values={"returncode": 0.0},
                raw_output=b"{}",
                stderr=b"",
                duration_s=0.0,
            ),
        ),
    )
    candidate = candidate or _empty_candidate()
    monkeypatch.setattr(
        wf,
        "_establish_capture_mutation_settlement",
        lambda *args, **kwargs: {
            "schema_version": "crucible.deepswe.mutation-settlement.v1",
            "tiers_executed": 0,
            "tier_attestations": [],
            "final_attestation": {},
            "capture_attestation": {
                "workspace_patch_sha256": candidate.sha256,
                "workspace_patch_bytes": candidate.diff_bytes,
                "workspace_files_modified": list(candidate.files_modified),
            },
        },
    )
    monkeypatch.setattr(
        wf,
        "capture_model_patch_from_container",
        lambda *args, **kwargs: candidate.patch,
    )
    monkeypatch.setattr(
        wf,
        "_validate_stopped_verifier_reward",
        lambda verdict, **kwargs: verdict,
    )
    monkeypatch.setattr(
        wf,
        "prepare_candidate_replay",
        lambda _workspace, applied_candidate, **kwargs: wf.CandidateApplyReport(
            status=wf.CandidateApplyStatus.APPLIED,
            base_commit=candidate.sha256,
            candidate_sha256=applied_candidate.sha256,
        ),
    )
    monkeypatch.setattr(
        wf,
        "prove_image_replay",
        lambda *args, **kwargs: wf.PristineReplayReport(
            status=wf.PristineReplayStatus.PASSED,
            base_commit=candidate.sha256,
            candidate_sha256=candidate.sha256,
            observed_sha256=candidate.sha256,
        ),
    )
    monkeypatch.setattr(
        wf,
        "verify_hidden_model_patch",
        lambda *args, **kwargs: wf.VerifierPatchReport(
            status=wf.VerifierPatchStatus.MATCHED,
            candidate_sha256=candidate.sha256,
            observed_sha256=candidate.sha256,
            observed_bytes=candidate.diff_bytes,
        ),
    )


def _patch_partial_tier_settlement(monkeypatch) -> None:
    """Make the generic container stub represent one implemented tier."""

    establish_settlement = wf._establish_capture_mutation_settlement

    def _partial_tier_settlement(*args, **kwargs):
        settlement = establish_settlement(*args, **kwargs)
        return {
            **settlement,
            "tiers_executed": 1,
            "tier_attestations": [{}],
        }

    monkeypatch.setattr(
        wf,
        "_establish_capture_mutation_settlement",
        _partial_tier_settlement,
    )


@pytest.mark.timeout(600)
def test_dev_verify_preflight_exactly_restores_both_writable_checkouts(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
    from crucible.workflows import _build_verify

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "spec"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    spec = synthesize_starter_spec(wf._load_and_filter(ns)[0])
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _init_preflight_checkout(candidate, label="candidate")
    _init_preflight_checkout(baseline, label="baseline")
    leaf_origin = tmp_path / "leaf-origin"
    leaf_origin.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=leaf_origin, check=True)
    (leaf_origin / ".gitignore").write_text("generated/\n")
    (leaf_origin / "leaf.txt").write_text("leaf baseline\n")
    subprocess.run(
        ["git", "add", ".gitignore", "leaf.txt"],
        cwd=leaf_origin,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-qm",
            "leaf baseline",
        ],
        cwd=leaf_origin,
        check=True,
    )
    nested_origin = tmp_path / "nested-origin"
    nested_origin.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=nested_origin, check=True)
    (nested_origin / ".gitignore").write_text("generated/\n")
    (nested_origin / "nested.txt").write_text("nested baseline\n")
    subprocess.run(
        ["git", "add", ".gitignore", "nested.txt"],
        cwd=nested_origin,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-qm",
            "nested baseline",
        ],
        cwd=nested_origin,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(leaf_origin),
            "leaf",
        ],
        cwd=nested_origin,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Crucible Test",
            "-c",
            "user.email=crucible@example.invalid",
            "commit",
            "-qam",
            "add leaf dependency",
        ],
        cwd=nested_origin,
        check=True,
    )
    _add_preflight_submodule(candidate, origin=nested_origin)
    _add_preflight_submodule(baseline, origin=nested_origin)
    before = {
        workspace: _preflight_checkout_image(workspace)
        for workspace in (candidate, baseline)
    }
    original_parameters = {
        "deepswe_metadata": {
            "language": "python",
            "container_id": "candidate-container",
            "baseline_container_id": "baseline-container",
        },
    }
    seen: dict[str, object] = {}
    restoration_receipts = []

    def _mutating_verify(*_args, **kwargs):
        seen["parameters"] = kwargs["extra_parameters"]
        assert (
            kwargs["extra_parameters"]["deepswe_metadata"]["dev_verify_clean_probe"]
            is True
        )
        for workspace in (candidate, baseline):
            (workspace / "tracked.txt").write_text("preflight-overwrite\n")
            (workspace / "inherited-untracked.txt").write_text("preflight-overwrite\n")
            (workspace / "ignored" / "state.bin").write_bytes(b"overwritten")
            (workspace / "existing" / "generated").mkdir()
            (workspace / "existing" / "generated" / "cache.bin").write_bytes(b"created")
            (workspace / "new-ignored").mkdir()
            (workspace / "new-ignored" / "cache.bin").write_bytes(b"created")
            (workspace / "new-top-level.txt").write_text("created\n")
            subprocess.run(
                ["git", "add", "tracked.txt"],
                cwd=workspace,
                check=True,
            )
        return BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=None,
        )

    monkeypatch.setattr(_build_verify, "build_and_verify", _mutating_verify)

    outcome = wf._run_dev_verify_preflight(
        spec=spec,
        cwd=candidate,
        baseline_cwd=baseline,
        log_dir=tmp_path / "logs",
        journal_root=(
            tmp_path / "task-corpus" / "runs" / "attempt" / "final_gate_mutations"
        ),
        extra_parameters=original_parameters,
        restoration_callback=restoration_receipts.append,
    )

    assert outcome.classification == "success"
    assert seen["parameters"] is not original_parameters
    assert "dev_verify_clean_probe" not in original_parameters["deepswe_metadata"]
    assert len(restoration_receipts) == 1
    restoration = restoration_receipts[0]
    assert type(restoration) is (
        baseline_preflight.DevelopmentPreflightRestorationReceipt
    )
    assert [item.label for item in restoration.workspaces] == [
        "candidate",
        "baseline",
    ]
    assert restoration.fingerprint.startswith("sha256:")
    assert len(restoration.fingerprint) == 71
    # Of everything the gate run touched, only what a slice gate would
    # charge is recorded: the new untracked file and the new ignored
    # directory below an existing one.  The new ignored top-level tree, the
    # tracked file and the inherited untracked and ignored files are not.
    assert [item.gate_created_paths for item in restoration.workspaces] == [
        ("existing/generated", "new-top-level.txt"),
        ("existing/generated", "new-top-level.txt"),
    ]
    for workspace in (candidate, baseline):
        assert _preflight_checkout_image(workspace) == before[workspace]
        assert not (workspace / "new-ignored").exists()
        assert not (workspace / "new-top-level.txt").exists()

    def _nested_artifact_verify(*_args, **_kwargs):
        for workspace in (candidate, baseline):
            generated = workspace / "existing" / "nested" / "leaf" / "generated"
            generated.mkdir()
            (generated / "cache.bin").write_bytes(b"created")
        return BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=None,
        )

    monkeypatch.setattr(
        _build_verify,
        "build_and_verify",
        _nested_artifact_verify,
    )
    failed_restoration_receipts = []
    with pytest.raises(
        RuntimeError,
        match="did not exactly restore 2 writable checkout",
    ):
        wf._run_dev_verify_preflight(
            spec=spec,
            cwd=candidate,
            baseline_cwd=baseline,
            log_dir=tmp_path / "nested-logs",
            journal_root=(
                tmp_path
                / "task-corpus"
                / "runs"
                / "nested-attempt"
                / "final_gate_mutations"
            ),
            extra_parameters=original_parameters,
            restoration_callback=failed_restoration_receipts.append,
        )
    assert failed_restoration_receipts == []
    for workspace in (candidate, baseline):
        assert _preflight_checkout_image(workspace) == before[workspace]

    baseline_residue = candidate / "existing" / "nested" / "leaf" / "generated"
    baseline_residue.mkdir()
    (baseline_residue / "cache.bin").write_bytes(b"preexisting")
    monkeypatch.setattr(
        _build_verify,
        "build_and_verify",
        lambda *_args, **_kwargs: pytest.fail(
            "preflight must reject nested untracked residue before execution"
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="requires fresh initialized submodules without untracked residue",
    ):
        wf._run_dev_verify_preflight(
            spec=spec,
            cwd=candidate,
            baseline_cwd=baseline,
            log_dir=tmp_path / "residue-logs",
            journal_root=(
                tmp_path
                / "task-corpus"
                / "runs"
                / "residue-attempt"
                / "final_gate_mutations"
            ),
            extra_parameters=original_parameters,
        )
    (baseline_residue / "cache.bin").unlink()
    baseline_residue.rmdir()
    assert _preflight_checkout_image(candidate) == before[candidate]

    monkeypatch.setattr(
        _build_verify,
        "build_and_verify",
        _mutating_verify,
    )

    def _reject_restoration(receipt):
        assert receipt.fingerprint.startswith("sha256:")
        for workspace in (candidate, baseline):
            assert _preflight_checkout_image(workspace) == before[workspace]
        raise RuntimeError("injected restoration receipt sink failure")

    with pytest.raises(RuntimeError, match="receipt sink failure"):
        wf._run_dev_verify_preflight(
            spec=spec,
            cwd=candidate,
            baseline_cwd=baseline,
            log_dir=tmp_path / "callback-failure-logs",
            journal_root=(
                tmp_path
                / "task-corpus"
                / "runs"
                / "callback-failure-attempt"
                / "final_gate_mutations"
            ),
            extra_parameters=original_parameters,
            restoration_callback=_reject_restoration,
        )
    for workspace in (candidate, baseline):
        assert _preflight_checkout_image(workspace) == before[workspace]


@pytest.mark.parametrize(
    "mode",
    [
        "returned_failure",
        "exception",
        "exception_and_callback_interrupt",
        "keyboard_interrupt",
        "keyboard_interrupt_and_callback_failure",
    ],
)
def test_dev_verify_restoration_receipt_is_independent_of_build_outcome(
    tmp_path: Path,
    monkeypatch,
    mode: str,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
    from crucible.workflows import _build_verify

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        ["--spec", str(tmp_path / "spec"), "--dataset", str(manifest)]
    )
    spec = synthesize_starter_spec(wf._load_and_filter(ns)[0])
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _init_preflight_checkout(candidate, label="candidate")
    _init_preflight_checkout(baseline, label="baseline")
    before = {
        workspace: _preflight_checkout_image(workspace)
        for workspace in (candidate, baseline)
    }

    def _build(*_args, **_kwargs):
        for workspace in (candidate, baseline):
            (workspace / "tracked.txt").write_text("preflight mutation\n")
            (workspace / "gate-created.lock").write_text("locked\n")
        if mode.startswith("exception"):
            raise RuntimeError("injected build exception")
        if mode.startswith("keyboard_interrupt"):
            raise KeyboardInterrupt("injected build interrupt")
        return BuildVerifyOutcome(
            "build_failed",
            build=None,
            verify=None,
            dev_verify=None,
        )

    receipts = []

    def _observe(receipt):
        for workspace in (candidate, baseline):
            assert _preflight_checkout_image(workspace) == before[workspace]
        receipts.append(receipt)
        if mode == "keyboard_interrupt_and_callback_failure":
            raise RuntimeError("injected restoration sink failure")
        if mode == "exception_and_callback_interrupt":
            raise KeyboardInterrupt("injected restoration sink interrupt")

    monkeypatch.setattr(_build_verify, "build_and_verify", _build)
    kwargs = {
        "spec": spec,
        "cwd": candidate,
        "baseline_cwd": baseline,
        "log_dir": tmp_path / "logs",
        "journal_root": tmp_path / "journals",
        "extra_parameters": {"deepswe_metadata": {}},
        "restoration_callback": _observe,
    }
    if mode == "returned_failure":
        outcome = wf._run_dev_verify_preflight(**kwargs)
        assert outcome.classification == "build_failed"
    elif mode == "exception":
        with pytest.raises(RuntimeError, match="injected build exception"):
            wf._run_dev_verify_preflight(**kwargs)
    elif mode == "exception_and_callback_interrupt":
        with pytest.raises(
            KeyboardInterrupt,
            match="injected restoration sink interrupt",
        ) as raised:
            wf._run_dev_verify_preflight(**kwargs)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert "injected build exception" in str(raised.value.__cause__)
    else:
        with pytest.raises(KeyboardInterrupt, match="injected build interrupt") as raised:
            wf._run_dev_verify_preflight(**kwargs)
        if mode == "keyboard_interrupt_and_callback_failure":
            assert isinstance(raised.value.__cause__, RuntimeError)
            assert "restoration sink failure" in str(raised.value.__cause__)

    assert len(receipts) == 1
    assert type(receipts[0]) is (
        baseline_preflight.DevelopmentPreflightRestorationReceipt
    )
    # Whatever the gate run returned or raised, the receipt names the path
    # it created.
    assert [item.gate_created_paths for item in receipts[0].workspaces] == [
        ("gate-created.lock",),
        ("gate-created.lock",),
    ]
    for workspace in (candidate, baseline):
        assert _preflight_checkout_image(workspace) == before[workspace]


@pytest.mark.parametrize(
    "failure_boundary",
    [
        "receipt_family_decode",
        "receipt_family_decode_during_interrupt",
        "restore_keyboard_interrupt",
    ],
)
def test_dev_verify_restores_both_before_evidence_boundary_failure(
    tmp_path: Path,
    monkeypatch,
    failure_boundary: str,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
    from crucible.workflows import _build_verify, _mutation_journal

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        ["--spec", str(tmp_path / "spec"), "--dataset", str(manifest)]
    )
    spec = synthesize_starter_spec(wf._load_and_filter(ns)[0])
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _init_preflight_checkout(candidate, label="candidate")
    _init_preflight_checkout(baseline, label="baseline")
    before = {
        workspace: _preflight_checkout_image(workspace)
        for workspace in (candidate, baseline)
    }

    def _build(*_args, **_kwargs):
        for workspace in (candidate, baseline):
            (workspace / "tracked.txt").write_text("preflight mutation\n")
        if failure_boundary == "receipt_family_decode_during_interrupt":
            raise KeyboardInterrupt("injected build interrupt")
        return BuildVerifyOutcome("success", None, None, None)

    restored_labels = []
    real_restore = _mutation_journal.WorkspaceMutationJournal.restore

    def _restore(journal, *args, **kwargs):
        result = real_restore(journal, *args, **kwargs)
        restored_labels.append(journal.slice_id)
        if (
            failure_boundary == "restore_keyboard_interrupt"
            and journal.slice_id == "dev_verify_preflight__baseline"
        ):
            raise KeyboardInterrupt("injected rollback interrupt")
        return result

    real_family_decode = _mutation_journal._generation_zero_writer_epoch_family

    def _family_decode(raw_manifest):
        if (
            failure_boundary.startswith("receipt_family_decode")
            and len(restored_labels) == 2
            and raw_manifest["slice_id"] == "dev_verify_preflight__baseline"
        ):
            raise RuntimeError("injected receipt family decode failure")
        return real_family_decode(raw_manifest)

    monkeypatch.setattr(_build_verify, "build_and_verify", _build)
    monkeypatch.setattr(
        _mutation_journal.WorkspaceMutationJournal,
        "restore",
        _restore,
    )
    monkeypatch.setattr(
        _mutation_journal,
        "_generation_zero_writer_epoch_family",
        _family_decode,
    )
    receipts = []
    if failure_boundary == "receipt_family_decode":
        expected_error = pytest.raises(
            RuntimeError,
            match="injected receipt family decode failure",
        )
    elif failure_boundary == "receipt_family_decode_during_interrupt":
        expected_error = pytest.raises(
            KeyboardInterrupt,
            match="injected build interrupt",
        )
    else:
        expected_error = pytest.raises(
            KeyboardInterrupt,
            match="injected rollback interrupt",
        )
    with expected_error as raised:
        wf._run_dev_verify_preflight(
            spec=spec,
            cwd=candidate,
            baseline_cwd=baseline,
            log_dir=tmp_path / "logs",
            journal_root=tmp_path / "journals",
            extra_parameters={"deepswe_metadata": {}},
            restoration_callback=receipts.append,
        )

    assert set(restored_labels) == {
        "dev_verify_preflight__candidate",
        "dev_verify_preflight__baseline",
    }
    assert receipts == []
    if failure_boundary == "restore_keyboard_interrupt":
        assert isinstance(raised.value.__cause__, _mutation_journal.MutationJournalError)
    elif failure_boundary == "receipt_family_decode_during_interrupt":
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert "receipt family decode failure" in str(raised.value.__cause__)
    for workspace in (candidate, baseline):
        assert _preflight_checkout_image(workspace) == before[workspace]


@pytest.mark.parametrize("failed_label", ["candidate", "baseline"])
def test_dev_verify_restoration_failure_never_emits_receipt(
    tmp_path: Path,
    monkeypatch,
    failed_label: str,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
    from crucible.workflows import _build_verify, _mutation_journal

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        ["--spec", str(tmp_path / "spec"), "--dataset", str(manifest)]
    )
    spec = synthesize_starter_spec(wf._load_and_filter(ns)[0])
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _init_preflight_checkout(candidate, label="candidate")
    _init_preflight_checkout(baseline, label="baseline")
    before = {
        workspace: _preflight_checkout_image(workspace)
        for workspace in (candidate, baseline)
    }

    def _build(*_args, **_kwargs):
        for workspace in (candidate, baseline):
            (workspace / "tracked.txt").write_text("preflight mutation\n")
        return BuildVerifyOutcome("success", None, None, None)

    real_restore = _mutation_journal.WorkspaceMutationJournal.restore

    def _restore_then_report_failure(journal, *args, **kwargs):
        restored = real_restore(journal, *args, **kwargs)
        if journal.slice_id == f"dev_verify_preflight__{failed_label}":
            return False
        return restored

    monkeypatch.setattr(_build_verify, "build_and_verify", _build)
    monkeypatch.setattr(
        _mutation_journal.WorkspaceMutationJournal,
        "restore",
        _restore_then_report_failure,
    )
    receipts = []
    with pytest.raises(
        _mutation_journal.MutationJournalError,
        match="did not exactly restore 1 writable checkout",
    ):
        wf._run_dev_verify_preflight(
            spec=spec,
            cwd=candidate,
            baseline_cwd=baseline,
            log_dir=tmp_path / "logs",
            journal_root=tmp_path / "journals",
            extra_parameters={"deepswe_metadata": {}},
            restoration_callback=receipts.append,
        )

    assert receipts == []
    for workspace in (candidate, baseline):
        assert _preflight_checkout_image(workspace) == before[workspace]


def test_dev_verify_preflight_failure_stops_before_provider_invocation(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        wf,
        "_run_dev_verify_preflight",
        lambda **_kwargs: BuildVerifyOutcome(
            "dev_verify_unavailable",
            build=None,
            verify=None,
            dev_verify=RunnerResult(
                role="dev_verify",
                runner=("crucible.products.deepswe.runners.deepswe_dev_verify"),
                returncode=2,
                values={
                    "returncode": 2.0,
                    "dev_verify_dependency_unavailable": 1.0,
                    "dev_verify_infrastructure_unavailable": 1.0,
                },
                raw_output=b"{}",
                stderr=b"",
                duration_s=0.0,
            ),
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **_kwargs: pytest.fail(
            "Architect must not run after verifier preflight failure"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(
            tmp_path / "s" / "runs" / "dev-verify-preflight-failed"
        ),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "setup_failed"
    assert summary["failure_phase"] == "dev_verify_preflight"
    assert summary["failure_kind"] == "dev_verify_unavailable"
    assert summary["cost_usd"] == 0.0
    assert summary["dev_verify_preflight"]["returncode"] == 2
    assert summary["cleanup_status"] == "complete"
    assert "verifier" not in calls
    # Architect was never started; the real summary of this stop is what the
    # run-status projection files as unjudged.
    assert "architect_operation_id" not in summary
    assert summary["architect_terminal_class"] is None
    assert wf._all_task_failures_are_unjudged([summary]) is True


def test_dev_verify_preflight_exception_publishes_classifiable_failure_rc(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows._mutation_journal import MutationJournalError

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _preflight_failure(**_kwargs):
        raise MutationJournalError(
            "development-verifier preflight did not exactly restore "
            "2 writable checkout(s)"
        )

    monkeypatch.setattr(
        wf,
        "_run_dev_verify_preflight",
        _preflight_failure,
    )
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **_kwargs: pytest.fail(
            "Architect must not run after verifier preflight exception"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(
            tmp_path / "s" / "runs" / "dev-verify-preflight-exception"
        ),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "cascade_error"
    assert summary["failure_phase"] == "workflow"
    assert summary["failure_kind"] == "MutationJournalError"
    assert type(summary["rc"]) is int
    assert summary["rc"] == 3
    assert summary["cost_usd"] == 0.0
    assert summary["cleanup_status"] == "complete"
    assert "verifier" not in calls
    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["rc"] == 3
    assert wf._validate_lifecycle_record(lifecycle, line_number=1)["rc"] == 3


def test_rust_toolchain_unavailable_is_retryable_before_hidden_verifier(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _toolchain_failure(*args, **kwargs):
        raise wf.RustToolchainUnavailable(
            "Rust toolchain probe failed under runtime UID"
        )

    monkeypatch.setattr(wf, "start_authoring_container", _toolchain_failure)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: pytest.fail("Architect must not run without Rust"),
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: pytest.fail(
            "hidden verifier must not run without Rust"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "toolchain-unavailable"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "toolchain_unavailable"
    assert summary["failure_phase"] == "container_setup"
    assert summary["failure_kind"] == "toolchain_unavailable"
    assert summary["verifier_status"] == "not_run"
    assert summary["hidden_invocation_measurement_ref"] is None
    assert summary["hidden_invocation_lifecycle_ref"] is None
    assert summary["hidden_verifier_measurement_ref"] is None
    assert summary["hidden_verifier_lifecycle_ref"] is None
    assert "verifier" not in calls
    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["status_policy"] == "retryable"
    assert wf._checkpoint_is_reusable(lifecycle, args=ns) is False
    # Architect was never started: the run-status projection files this real
    # summary as unjudged.
    assert "architect_operation_id" not in summary
    assert summary["architect_terminal_class"] is None
    assert wf._all_task_failures_are_unjudged([summary]) is True


def test_verifier_toolchain_failure_never_executes_hidden_command(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    _patch_partial_tier_settlement(monkeypatch)

    def _architect(**kwargs):
        return _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            tiers_executed=1,
        )

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)

    def _verifier_toolchain_failure(*args, **kwargs):
        calls.setdefault("events", []).append("verifier_toolchain_probe")
        raise wf.RustToolchainUnavailable(
            "Rust toolchain probe failed under verifier UID"
        )

    monkeypatch.setattr(
        wf,
        "start_verifier_container",
        _verifier_toolchain_failure,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: pytest.fail(
            "hidden verifier command must not run after probe failure"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(
            tmp_path / "s" / "runs" / "verifier-toolchain-unavailable"
        ),
        cost=CostAccumulator(),
    )

    # Mount authority is durably consumed before the verifier container is
    # attempted, so the task remains one-shot even though no hidden command
    # executed. The failure kind still identifies the toolchain root cause.
    assert summary["status"] == "verification_incomplete"
    assert summary["failure_phase"] == "hidden_verifier"
    assert summary["failure_kind"] == "toolchain_unavailable"
    assert summary["verifier_status"] == "not_run"
    assert summary["hidden_invocation_measurement_ref"]
    assert summary["hidden_invocation_lifecycle_ref"]
    assert summary["hidden_verifier_measurement_ref"] is None
    assert summary["hidden_verifier_lifecycle_ref"] is None
    assert calls["events"].count("verifier_toolchain_probe") == 1
    # A one-shot oracle failure is never relabelled as unjudged work.
    assert wf._all_task_failures_are_unjudged([summary]) is False
    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["status_policy"] == "reusable"
    assert wf._checkpoint_is_reusable(lifecycle, args=ns) is True


def test_run_task_cascade_uses_fresh_hidden_grading_checkout(
    tmp_path: Path,
    monkeypatch,
):
    """Replay proves candidate bytes; hidden grading starts from fresh base."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--budget-usd-per-instance", "12",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    # The verifier reads <dataset>.parent/<task_id>/tests — the
    # fixture builder already created tasks/<id>/, add a tests dir.
    (manifest.parent / inst.task_id / "tests").mkdir(parents=True, exist_ok=True)

    candidate = wf.candidate_from_image_patch(
        b"fresh grading candidate\n",
        files_modified=("adaptix/_internal.py",),
        raw_diff_bytes=24,
        max_bytes=ns.max_patch_bytes,
    )
    calls: dict = {}
    _patch_container_lifecycle(
        monkeypatch,
        tmp_path,
        calls,
        candidate=candidate,
    )
    _patch_partial_tier_settlement(monkeypatch)

    class _SettlementClient:
        def labels_for(self, labels):
            return {
                **labels,
                "io.crucible.settlement": "true",
                "io.crucible.run-generation": "run-a",
                "io.crucible.arm-generation": "arm-a",
                "io.crucible.daemon-binding": "daemon-a",
            }

        def register_container(self, _reference):
            raise AssertionError("mocked Docker helpers do not create containers")

    settlement_client = _SettlementClient()
    monkeypatch.setattr(
        wf.DockerSettlementRegistrationClient,
        "from_environment",
        classmethod(lambda _cls, **_kwargs: settlement_client),
    )
    persist_marker = wf._persist_hidden_invocation_marker

    def _record_marker(*args, **kwargs):
        verifier_patch_path = (
            run_dir
            / "instances"
            / inst.task_id
            / "logs"
            / "hidden-verifier"
            / "artifacts"
            / "model.patch"
        )
        assert verifier_patch_path.read_bytes() == candidate.patch
        calls.setdefault("events", []).append("candidate_patch_preloaded")
        calls.setdefault("events", []).append("persist_hidden_marker")
        return persist_marker(*args, **kwargs)

    monkeypatch.setattr(wf, "_persist_hidden_invocation_marker", _record_marker)
    # Use a non-zero local cascade rc with a nonempty candidate: the hidden
    # reward remains the authoritative benchmark outcome. The authoring
    # container id must be what run_architect sees in deepswe_metadata.
    seen_cid: dict = {}

    def _fake_run_architect(**kwargs):
        assert "plan_candidate_transform" not in kwargs
        assert kwargs["planner_count"] == 3
        assert kwargs["budget_usd"] == 12.0
        assert kwargs["planner_tool_call_tranches"] == (32,)
        assert kwargs["development_budget_policy"].planning_review_cap_usd == pytest.approx(4.8)
        seen_cid["checkpoint_fingerprint"] = kwargs["planning_policy"].execution_fingerprint
        seen_cid["cid"] = kwargs["extra_parameters"]["deepswe_metadata"]["container_id"]
        seen_cid["baseline"] = kwargs["extra_parameters"]["deepswe_metadata"][
            "baseline_container_id"
        ]
        return 1

    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        _fake_run_architect,
    )
    # Dataset verify: report reward=1.
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda inst, *, container_id, instance_log_dir: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "command_returncode": 0.0,
            "verify_returncode": 0.0,
            "baseline_test_returncode": 0.0,
            "new_test_returncode": 0.0,
            "diagnostics_persisted": True,
            "verifier_stdout_bytes": 4096.0,
            "verifier_stderr_bytes": 128.0,
            "verifier_stdout_truncated": False,
            "verifier_stderr_truncated": False,
        },
    )

    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "rd")
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
    )

    # run_architect saw the AUTHORING container, never the verifier.
    assert seen_cid["cid"] == "authoring-cid"
    assert seen_cid["baseline"] == "baseline-cid"
    assert seen_cid["baseline"] != seen_cid["cid"]
    assert seen_cid["checkpoint_fingerprint"] == summary["checkpoint_fingerprint"]
    # Author/baseline, two no-tests image renderers, then one verifier.
    assert len(calls["authoring"]) == 2
    assert len(calls["render"]) == 2
    assert len(calls["verifier"]) == 1
    assert len(calls["populate"]) == 4
    assert {Path(entry[1]).name for entry in calls["populate"]} == {
        "candidate",
        "baseline",
        "verifier",
        "grading",
    }
    replay_workspace = Path(calls["render"][1]["workspace"])
    grading_workspace = Path(calls["verifier"][0]["workspace"])
    assert replay_workspace.name == "verifier"
    assert grading_workspace.name == "grading"
    assert grading_workspace != replay_workspace
    grading_population = next(
        entry for entry in calls["populate"] if Path(entry[1]).name == "grading"
    )
    assert grading_population[3]["io.crucible.phase"] == (
        "grading-workspace-population"
    )
    all_container_calls = [
        *(entry[3:] for entry in calls["populate"]),
        *(
            (entry["labels"], entry["register_container"])
            for entry in calls["authoring"]
        ),
        *((entry["labels"], entry["register_container"]) for entry in calls["render"]),
        *(
            (entry["labels"], entry["register_container"])
            for entry in calls["verifier"]
        ),
    ]
    assert len(all_container_calls) == 9
    registration_callback = all_container_calls[0][1]
    assert registration_callback is not None
    assert grading_population[4] is registration_callback
    assert all(
        labels["io.crucible.settlement"] == "true"
        and register_container is registration_callback
        for labels, register_container in all_container_calls
    )
    assert summary["baseline_container_id"] == "baseline-cid"
    assert calls["strict_stop"] == [
        "authoring-cid",
        "baseline-cid",
        "source-render-cid",
        "proof-render-cid",
        "verifier-cid",
    ]
    assert calls["events"].index("persist_hidden_marker") < calls["events"].index(
        "start_verifier"
    )
    assert calls["events"].index("candidate_patch_preloaded") < calls["events"].index(
        "persist_hidden_marker"
    )
    assert summary["verifier_patch_preload"]["status"] == "matched"
    # /logs is a dedicated mount; no candidate, baseline, or pristine
    # checkout is reachable through it.
    log_dirs = {
        Path(container["logs"])
        for container in (*calls["authoring"], *calls["verifier"])
    }
    assert len(log_dirs) == 3
    for container in (*calls["authoring"], *calls["render"], *calls["verifier"]):
        assert all(
            not Path(container["workspace"]).is_relative_to(log_dir)
            for log_dir in log_dirs
        )
    # The verifier mounts only a private snapshot, never the live dataset path.
    mounted_tests = Path(calls["verifier"][0]["tests_dir"])
    assert mounted_tests != manifest.parent / inst.task_id / "tests"
    assert mounted_tests.is_relative_to(run_dir / "instances" / inst.task_id)
    assert (
        summary["tests_snapshot"]["sha256"]
        == summary["tests_source_identity"]["sha256"]
    )
    # Dataset reward propagated into the summary.
    assert summary["status"] == "resolved"
    assert summary["rc"] == 1
    assert summary["cascade_status"] == "failed"
    assert summary["reward"] == 1.0
    assert summary["resolved"] == 1.0
    assert summary["verification_rc"] == 0
    assert summary["command_returncode"] == 0.0
    assert summary["verify_returncode"] == 0.0
    assert summary["baseline_test_returncode"] == 0.0
    assert summary["new_test_returncode"] == 0.0
    assert summary["diagnostics_persisted"] is True
    assert summary["verifier_stdout_bytes"] == 4096.0
    assert summary["verifier_stderr_bytes"] == 128.0
    assert summary["verifier_stdout_truncated"] is False
    assert summary["verifier_stderr_truncated"] is False
    assert summary["container_id"] == "authoring-cid"
    assert summary["verifier_container_id"] == "verifier-cid"
    from assay import LifecycleEvent
    from assay.corpus.store import Corpus

    instance_corpus = Corpus(
        tmp_path / "s" / "instances" / inst.task_id / "corpus",
    )
    latest = instance_corpus.latest_spec()
    assert summary["hidden_verifier_measurement_ref"] in latest.measurement_refs
    assert summary["candidate_measurement_ref"] in latest.measurement_refs
    assert summary["hidden_verifier_lifecycle_ref"] in latest.lifecycle_event_refs
    assert summary["instance_spec_version"] == latest.version
    measurement = instance_corpus.get_measurement(
        latest.measurement_refs[-1],
    )
    assert measurement.bench_id == f"deepswe/{inst.task_id}/hidden_verifier"
    assert measurement.values["reward"] == 1.0
    assert measurement.values["command_returncode"] == 0.0
    assert measurement.values["verify_returncode"] == 0.0
    assert measurement.values["baseline_test_returncode"] == 0.0
    assert measurement.values["new_test_returncode"] == 0.0
    assert measurement.values["diagnostics_persisted"] == 1.0
    assert measurement.values["verifier_stdout_bytes"] == 4096.0
    assert measurement.values["verifier_stderr_bytes"] == 128.0
    assert measurement.values["verifier_stdout_truncated"] == 0.0
    assert measurement.values["verifier_stderr_truncated"] == 0.0
    raw_hidden = json.loads(instance_corpus.get_artifact(measurement.raw_output_ref))
    assert raw_hidden["schema_version"] == 3
    assert raw_hidden["candidate"]["diff_ref"] == summary["candidate"]["diff_ref"]
    assert raw_hidden["pristine_replay"]["status"] == "passed"
    assert raw_hidden["verifier_patch"]["status"] == "matched"
    assert raw_hidden["verifier"]["command_returncode"] == 0.0
    assert raw_hidden["verifier"]["baseline_test_returncode"] == 0.0
    assert raw_hidden["verifier"]["new_test_returncode"] == 0.0
    assert raw_hidden["checkpoint"]["command_returncode"] == 0.0
    assert raw_hidden["checkpoint"]["diagnostics_persisted"] is True
    invocation = instance_corpus.get_measurement(
        summary["hidden_invocation_measurement_ref"]
    )
    raw_invocation = json.loads(instance_corpus.get_artifact(invocation.raw_output_ref))
    assert raw_invocation["tests_snapshot"] == summary["tests_snapshot"]
    assert (
        raw_invocation["checkpoint"]["tests_source_identity"]
        == summary["tests_snapshot"]
    )
    terminal = instance_corpus.get_record(
        latest.lifecycle_event_refs[-1],
        as_type=LifecycleEvent,
    )
    assert terminal.kind == "landed"
    assert terminal.affected_versions == (latest.version,)
    # Both containers torn down; image reclaimed exactly once.
    image_reclaims = [c for c in calls["teardown"] if c[1] is not None]
    assert len(image_reclaims) == 1


def test_image_rerender_mismatch_blocks_snapshot_and_hidden_mount(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    actual_prove_image_replay = wf.prove_image_replay
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(wf, "prove_image_replay", actual_prove_image_replay)
    monkeypatch.setattr(
        wf,
        "prepare_candidate_replay",
        lambda _workspace, applied_candidate, **kwargs: wf.CandidateApplyReport(
            status=wf.CandidateApplyStatus.APPLIED,
            base_commit=inst.base_commit_hash,
            candidate_sha256=applied_candidate.sha256,
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    renderings = iter((b"image-render-B", b"image-render-C"))
    monkeypatch.setattr(
        wf,
        "capture_model_patch_from_container",
        lambda *args, **kwargs: next(renderings),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "mismatch"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "pristine_replay_failed"
    assert summary["failure_kind"] == "patch_mismatch"
    assert summary["tests_snapshot"] is None
    assert "verifier" not in calls
    assert calls["strict_stop"] == [
        "authoring-cid",
        "baseline-cid",
        "source-render-cid",
        "proof-render-cid",
    ]


def test_hidden_tests_change_after_checkpoint_blocks_private_mount(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    real_snapshot = wf.snapshot_task_tests_tree

    def _mutate_then_snapshot(*args, **kwargs):
        test_script = manifest.parent / inst.task_id / "tests" / "test.sh"
        test_script.write_text(
            test_script.read_text(encoding="utf-8") + "# changed\n",
            encoding="utf-8",
        )
        return real_snapshot(*args, **kwargs)

    monkeypatch.setattr(wf, "snapshot_task_tests_tree", _mutate_then_snapshot)

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "tests-changed"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "cascade_error"
    assert "changed after checkpoint identity" in summary["error"]
    assert summary["tests_snapshot"] is None
    assert "verifier" not in calls
    assert summary["hidden_invocation_measurement_ref"] is None


def test_strict_authoring_stop_failure_blocks_capture_and_hidden_verifier(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _architect(**kwargs):
        return _publish_architect_child(Path(kwargs["spec"]), cost_usd=0.0)

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)

    def _strict_stop(container_id):
        calls.setdefault("strict_stop", []).append(container_id)
        raise wf.InstanceContainerSetupError("container remains running")

    monkeypatch.setattr(wf, "stop_instance_container_strict", _strict_stop)
    monkeypatch.setattr(
        wf,
        "capture_model_patch_from_container",
        lambda *args, **kwargs: pytest.fail("must quiesce before capture"),
    )

    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "stop-failed")
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
    )

    assert summary["status"] == "setup_failed"
    assert calls["strict_stop"] == ["authoring-cid"]
    assert "verifier" not in calls
    # Architect ran and completed: this setup stop is not "Architect never
    # started". The row stays retryable, but the run-status projection does
    # not file the task as unjudged.
    assert summary["architect_operation_id"]
    assert summary["architect_terminal_class"] == "succeeded"
    assert wf._lifecycle_record(summary)["status_policy"] == "retryable"
    assert wf._all_task_failures_are_unjudged([summary]) is False
    # The id is retained after the failed proof and retried by final cleanup.
    assert ("authoring-cid", None) in calls["teardown"]
    assert ("baseline-cid", None) in calls["teardown"]


def test_post_architect_stop_failure_is_secondary_and_retryable(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _architect(**kwargs):
        _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            status="inflation_veto",
            tiers_executed=0,
            failure_phase="architect_planning",
            failure_kind="inflation_veto",
        )
        return 2

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "stop_instance_container_strict",
        lambda _container_id: (_ for _ in ()).throw(
            wf.InstanceContainerSetupError("container remains running")
        ),
    )
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "stop-secondary"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "candidate_failed"
    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == "inflation_veto"
    assert summary["secondary_failures"] == [
        {
            "phase": "container_setup",
            "kind": "InstanceContainerSetupError",
            "error": "container remains running",
        }
    ]
    assert summary["zero_tier_settlement_proven"] is False
    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["status_policy"] == "retryable"
    assert wf._checkpoint_is_reusable(lifecycle, args=ns) is False


def test_post_architect_cleanup_failure_is_secondary_and_retryable(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _architect(**kwargs):
        _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            status="inflation_veto",
            tiers_executed=0,
            failure_phase="architect_planning",
            failure_kind="inflation_veto",
        )
        return 2

    def _teardown(container_id, image=None):
        calls.setdefault("teardown", []).append((container_id, image))
        if image is not None:
            raise RuntimeError("image cleanup failed")

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(wf, "teardown_instance_container", _teardown)
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "cleanup-secondary"),
        cost=CostAccumulator(),
    )

    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == "inflation_veto"
    assert summary["cleanup_status"] == "failed"
    assert summary["secondary_failures"] == [
        {
            "phase": "cleanup",
            "kind": "cleanup_failed",
            "error": "image: RuntimeError: image cleanup failed",
        }
    ]
    assert wf._lifecycle_record(summary)["status_policy"] == "retryable"


def test_post_architect_quota_failure_is_secondary_and_retryable(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    class _Quota:
        exceeded = False
        peak_growth_bytes = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            raise wf.StorageQuotaExceeded("quota tripped during settlement")

    def _architect(**kwargs):
        _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            status="inflation_veto",
            tiers_executed=0,
            failure_phase="architect_planning",
            failure_kind="inflation_veto",
        )
        return 2

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _Quota)
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "quota-secondary"),
        cost=CostAccumulator(),
    )

    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == "inflation_veto"
    assert summary["secondary_failures"] == [
        {
            "phase": "storage_quota",
            "kind": "StorageQuotaExceeded",
            "error": "quota tripped during settlement",
        }
    ]
    assert wf._lifecycle_record(summary)["status_policy"] == "retryable"


def test_unproven_tier_journal_settlement_blocks_full_candidate_capture(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        wf,
        "_establish_capture_mutation_settlement",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("active mutation journal remains")
        ),
    )
    monkeypatch.setattr(
        wf,
        "start_patch_render_container",
        lambda *args, **kwargs: pytest.fail(
            "full candidate capture requires positive settlement proof"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "unsettled"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "candidate_failed"
    assert summary["failure_phase"] == "mutation_journal_settlement"
    assert summary["failure_kind"] == "mutation_journal_unsettled"
    assert summary["candidate_capture_status"] == "failed"
    assert summary["candidate_measurement_ref"]
    assert "verifier" not in calls


@pytest.mark.parametrize(
    ("detail_status", "decision", "terminal_class", "status_policy"),
    [
        ("inflation_veto", "rejected", "permanent_failure", "permanent"),
        (
            "judge_protocol_failure",
            "inconclusive",
            "inconclusive",
            "retryable",
        ),
        # The planner backend was lost: no plan was ruled on, so the task
        # is neither cached as a permanent veto nor filed as judged.
        (
            "planner_infrastructure_failure",
            "inconclusive",
            "inconclusive",
            "retryable",
        ),
    ],
)
def test_zero_tier_architect_failure_skips_candidate_and_hidden_verifier(
    tmp_path: Path,
    monkeypatch,
    detail_status: str,
    decision: str,
    terminal_class: str,
    status_policy: str,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    development_metrics = {
        "time_to_first_mutation_seconds": None,
        "cost_to_first_mutation_usd": None,
        "planning_to_implementation_spend_ratio": None,
        "calls_by_role": {"architect_planner": 1},
        "validation_repair_count": 1,
        "source_calls_dispatched": 24,
        "source_calls_denied": 0,
        "implementation_reserve_remaining_usd": 3.0,
    }
    development_budget = {
        "schema_version": "crucible.development-budget.v1",
        "autonomous_development_metrics": development_metrics,
    }

    def _architect(**kwargs):
        _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            status=detail_status,
            decision=decision,
            tiers_executed=0,
            failure_phase="architect_planning",
            failure_kind=detail_status,
            reason="planning did not produce an executable DAG",
            development_budget=development_budget,
        )
        return 2

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(
            tmp_path / "s" / "runs" / f"zero-tier-{detail_status}"
        ),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "candidate_failed"
    assert summary["architect_detail_status"] == detail_status
    assert summary["architect_decision"] == decision
    assert summary["architect_terminal_class"] == terminal_class
    assert summary["architect_tiers_executed"] == 0
    assert summary["development_budget"] == development_budget
    assert summary["autonomous_development_metrics"] == development_metrics
    assert summary["zero_tier_settlement_proven"] is True
    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == detail_status
    assert summary["candidate"] is None
    assert summary["candidate_measurement_ref"] is None
    assert summary["hidden_invocation_measurement_ref"] is None
    assert summary["hidden_verifier_measurement_ref"] is None
    assert summary["verifier_status"] == "not_run"
    assert "render" not in calls
    assert "verifier" not in calls

    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["architect_detail_status"] == detail_status
    assert lifecycle["architect_decision"] == decision
    assert lifecycle["architect_terminal_class"] == terminal_class
    assert lifecycle["status_policy"] == status_policy
    assert wf._checkpoint_is_reusable(lifecycle, args=ns) is (
        status_policy == "permanent"
    )
    # The run-status projection reads the same task summary: a deterministic
    # veto is a judged failure, a lost ruling is not.
    assert wf._all_task_failures_are_unjudged([summary]) is (
        status_policy == "retryable"
    )
    assert wf._all_task_failures_are_retryable([summary]) is False


def test_zero_tier_settlement_failure_is_secondary_to_architect_failure(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _architect(**kwargs):
        _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            status="inflation_veto",
            decision="rejected",
            tiers_executed=0,
            failure_phase="architect_planning",
            failure_kind="inflation_veto",
        )
        return 2

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "_establish_capture_mutation_settlement",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("final settlement proof is invalid")
        ),
    )
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "zero-tier-unsettled"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "candidate_failed"
    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == "inflation_veto"
    assert summary["secondary_failures"] == [
        {
            "phase": "mutation_journal_settlement",
            "kind": "mutation_journal_unsettled",
            "error": (
                "mutation journal settlement is unproven: ValueError: "
                "final settlement proof is invalid"
            ),
        }
    ]
    assert summary["candidate"] is None
    assert summary["zero_tier_settlement_proven"] is False
    assert "render" not in calls
    assert "verifier" not in calls
    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["status_policy"] == "retryable"
    assert wf._checkpoint_is_reusable(lifecycle, args=ns) is False


def test_architect_settlement_requires_proof_for_every_invoked_tier(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.common.benchmark_runtime import canonical_fingerprint, file_digest

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    architect_run = tmp_path / "architect-run"
    architect_run.mkdir()
    tier_run = tmp_path / "tier-run"
    tier_run.mkdir()
    for name in ("summary.json", "result.json", "outcome.json"):
        (tier_run / name).write_text("{}\n", encoding="utf-8")

    def _proof(tier_id, operation_id, run_dir=None):
        body = {
            "schema_version": "crucible.tier-mutation-settlement.v1",
            "settled": True,
            "tier_id": tier_id,
            "operation_id": operation_id,
            "task_id": inst.task_id,
            "ext_id": inst.ext_id,
            "workspace": str(workspace.resolve()),
            "base_commit": inst.base_commit_hash,
            "max_patch_bytes": ns.max_patch_bytes,
            "workspace_patch_sha256": hashlib.sha256(b"").hexdigest(),
            "workspace_patch_bytes": 0,
            "workspace_files_modified": [],
            "spec_version": 1,
            "spec_hash": "sha256:" + "1" * 64,
            "tier_run_id": run_dir.name if run_dir is not None else None,
            "tier_run_artifacts": (
                {
                    name: file_digest(run_dir / name)
                    for name in ("summary.json", "result.json", "outcome.json")
                }
                if run_dir is not None
                else {}
            ),
            "journal_recovery": {
                "recovered": [],
                "preserved": [],
                "failures": [],
            },
            "remaining_journal_manifests": [],
            "registry_clear": True,
        }
        return {**body, "attestation_sha256": canonical_fingerprint(body)}

    tier_result = {
        "tier_id": "t0",
        "operation_id": "tier-0",
        "run_dir": str(tier_run),
        "mutation_journal_settlement": _proof("t0", "tier-0", tier_run),
    }
    payload = {
        "tiers_executed": 1,
        "tier_results": [tier_result],
        "mutation_journal_settlement": _proof("__architect_final__", "architect-final"),
    }
    zero_tier_payload = {
        "tiers_executed": 0,
        "tier_results": [],
        "mutation_journal_settlement": _proof("__architect_final__", "architect-final"),
    }
    (architect_run / "summary.json").write_text(json.dumps(zero_tier_payload))
    zero_tier = wf._require_architect_mutation_settlement(
        architect_run,
        inst=inst,
        host_workspace=workspace,
        max_patch_bytes=ns.max_patch_bytes,
    )
    assert zero_tier["tiers_executed"] == 0
    assert zero_tier["tier_attestations"] == []

    dirty_zero_tier = dict(zero_tier_payload)
    dirty_final = dict(dirty_zero_tier["mutation_journal_settlement"])
    dirty_final.pop("attestation_sha256")
    dirty_final.update(
        {
            "workspace_patch_sha256": hashlib.sha256(b"dirty").hexdigest(),
            "workspace_patch_bytes": 5,
            "workspace_files_modified": ["dirty.py"],
        }
    )
    dirty_final["attestation_sha256"] = canonical_fingerprint(dirty_final)
    dirty_zero_tier["mutation_journal_settlement"] = dirty_final
    (architect_run / "summary.json").write_text(json.dumps(dirty_zero_tier))
    with pytest.raises(ValueError, match="does not prove a clean workspace"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    bool_zero_tier = dict(zero_tier_payload, tiers_executed=False)
    (architect_run / "summary.json").write_text(json.dumps(bool_zero_tier))
    with pytest.raises(ValueError, match="inventory is incomplete"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    (architect_run / "summary.json").write_text(json.dumps(payload))

    validated = wf._require_architect_mutation_settlement(
        architect_run,
        inst=inst,
        host_workspace=workspace,
        max_patch_bytes=ns.max_patch_bytes,
    )
    assert validated["tiers_executed"] == 1

    namespaced_payload = {
        "tiers_executed": 1,
        "tier_results": [
            {
                "tier_id": "t0",
                "operation_id": "tier-0",
                "run_dir": str(tier_run),
            }
        ],
        "checkpoint_extensions": {
            wf.DEEPSWE_CHECKPOINT_EXTENSION_KEY: {
                "schema_version": wf.DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA,
                "mutation_settlement": {
                    "schema_version": wf.DEEPSWE_MUTATION_SETTLEMENT_SCHEMA,
                    "tiers_executed": 1,
                    "tier_attestations": [_proof("t0", "tier-0", tier_run)],
                    "final_attestation": _proof(
                        "__architect_final__", "architect-final"
                    ),
                },
            },
        },
    }
    (architect_run / "summary.json").write_text(json.dumps(namespaced_payload))
    namespaced = wf._require_architect_mutation_settlement(
        architect_run,
        inst=inst,
        host_workspace=workspace,
        max_patch_bytes=ns.max_patch_bytes,
    )
    assert namespaced["tiers_executed"] == 1
    assert len(namespaced["tier_attestations"]) == 1

    # Exercise the real product producer and reader together so their owner
    # key, wrapper schemas, and individual Tier-proof shape cannot drift while
    # separately constructed fixtures continue to pass.
    from crucible.products.deepswe import mutation_settlement as settlement
    from crucible.workflows.architect import ArchitectCheckpoint

    def _producer_attest(**kwargs):
        run_dir = kwargs.get("tier_run_dir")
        return _proof(
            kwargs["tier_id"],
            kwargs["operation_id"],
            Path(run_dir) if run_dir is not None else None,
        )

    monkeypatch.setattr(
        settlement,
        "attest_mutation_journal_settlement",
        _producer_attest,
    )
    producer = settlement.DeepSWEArchitectCheckpointHook(
        task_id=inst.task_id,
        ext_id=inst.ext_id,
        expected_base_commit=inst.base_commit_hash,
        max_patch_bytes=ns.max_patch_bytes,
    )
    producer(
        ArchitectCheckpoint(
            kind="tier",
            corpus=Corpus(tmp_path / "producer-corpus"),
            cwd=str(workspace),
            tier_id="t0",
            operation_id="tier-0",
            child_run_dir=str(tier_run),
            tiers_executed=1,
        )
    )
    produced_extensions = producer(
        ArchitectCheckpoint(
            kind="final",
            corpus=Corpus(tmp_path / "producer-corpus"),
            cwd=str(workspace),
            tier_id="__architect_final__",
            operation_id="architect-final",
            child_run_dir=None,
            tiers_executed=1,
        )
    )
    produced_payload = {
        "tiers_executed": 1,
        "tier_results": namespaced_payload["tier_results"],
        "checkpoint_extensions": produced_extensions,
    }
    (architect_run / "summary.json").write_text(json.dumps(produced_payload))
    produced = wf._require_architect_mutation_settlement(
        architect_run,
        inst=inst,
        host_workspace=workspace,
        max_patch_bytes=ns.max_patch_bytes,
    )
    assert produced["tier_attestations"][0]["tier_id"] == "t0"
    assert produced["final_attestation"]["tier_id"] == ("__architect_final__")

    # Namespace presence selects the new contract authoritatively. A malformed
    # or missing DeepSWE extension must not downgrade to valid legacy fields.
    mixed_payload = {
        **payload,
        "checkpoint_extensions": {"unrelated.extension": {}},
    }
    (architect_run / "summary.json").write_text(json.dumps(mixed_payload))
    with pytest.raises(ValueError, match="required DeepSWE checkpoint proof"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    # Exception envelopes and their selected partial summaries are both part
    # of the format boundary. Advertising the new namespace at either level
    # must prevent fallback to otherwise-valid legacy settlement fields.
    outer_advertised = {
        "status": "failed",
        "checkpoint_extensions": {"unrelated.extension": {}},
        "partial_summary": payload,
    }
    (architect_run / "summary.json").write_text(json.dumps(outer_advertised))
    with pytest.raises(ValueError, match="namespace is invalid"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    partial_advertised = {
        "status": "failed",
        "partial_summary": mixed_payload,
    }
    (architect_run / "summary.json").write_text(json.dumps(partial_advertised))
    with pytest.raises(ValueError, match="required DeepSWE checkpoint proof"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    malformed_count = json.loads(json.dumps(namespaced_payload))
    malformed_count["checkpoint_extensions"][wf.DEEPSWE_CHECKPOINT_EXTENSION_KEY][
        "mutation_settlement"
    ]["tiers_executed"] = True
    (architect_run / "summary.json").write_text(json.dumps(malformed_count))
    with pytest.raises(ValueError, match="settlement is incomplete"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    (architect_run / "summary.json").write_text(json.dumps(payload))

    stale_final = dict(payload["mutation_journal_settlement"])
    stale_final.pop("attestation_sha256")
    stale_final["max_patch_bytes"] = ns.max_patch_bytes - 1
    payload["mutation_journal_settlement"] = {
        **stale_final,
        "attestation_sha256": canonical_fingerprint(stale_final),
    }
    (architect_run / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity/proof"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )

    payload["mutation_journal_settlement"] = _proof(
        "__architect_final__", "architect-final"
    )

    del tier_result["mutation_journal_settlement"]
    (architect_run / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing"):
        wf._require_architect_mutation_settlement(
            architect_run,
            inst=inst,
            host_workspace=workspace,
            max_patch_bytes=ns.max_patch_bytes,
        )


def test_reward_is_revalidated_only_after_verifier_is_strictly_stopped(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    (manifest.parent / inst.task_id / "tests").mkdir(exist_ok=True)
    calls: dict = {}
    actual_reward_validation = wf._validate_stopped_verifier_reward
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        wf,
        "_validate_stopped_verifier_reward",
        actual_reward_validation,
    )

    def _strict_stop(container_id):
        calls.setdefault("strict_stop", []).append(container_id)
        if container_id == "verifier-cid":
            reward = Path(calls["verifier"][0]["logs"]) / "verifier/reward.txt"
            reward.parent.mkdir(parents=True, exist_ok=True)
            reward.write_text("1\n", encoding="utf-8")

    monkeypatch.setattr(wf, "stop_instance_container_strict", _strict_stop)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
        ),
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
        },
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "reward-order"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "resolved"
    assert summary["reward_state"] == "valid"
    assert calls["strict_stop"][-1] == "verifier-cid"
    # A tests tree without config.json declares no whitelist: the v1.0 token
    # is the grade and carries no per-test counts.
    assert summary["test_inventory"] is None
    assert summary["reward_counts"] is None
    assert summary["reward_error"] is None


def _run_cascade_with_reward_artifact(
    tmp_path: Path,
    monkeypatch,
    *,
    whitelist: tuple[list[str], list[str]] | None,
    artifact: str,
    payload: str,
):
    """Run one stubbed cascade whose verifier leaves ``payload`` at its stop."""

    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    if whitelist is not None:
        _declare_test_whitelist(
            manifest,
            inst.task_id,
            f2p=whitelist[0],
            p2p=whitelist[1],
        )
    calls: dict = {}
    actual_reward_validation = wf._validate_stopped_verifier_reward
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        wf,
        "_validate_stopped_verifier_reward",
        actual_reward_validation,
    )

    def _strict_stop(container_id):
        calls.setdefault("strict_stop", []).append(container_id)
        if container_id == "verifier-cid":
            reward = Path(calls["verifier"][0]["logs"]) / "verifier" / artifact
            reward.parent.mkdir(parents=True, exist_ok=True)
            reward.write_text(payload, encoding="utf-8")

    monkeypatch.setattr(wf, "stop_instance_container_strict", _strict_stop)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
        ),
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 0,
            "reward": None,
            "resolved": None,
            "command_returncode": 0.0,
            "verify_returncode": 0.0,
        },
    )
    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "reward-artifact"),
        cost=CostAccumulator(),
    )
    corpus = Corpus(tmp_path / "s" / "instances" / inst.task_id / "corpus")
    return summary, corpus


_FIXTURE_WHITELIST = (["f2p::a", "f2p::b"], ["p2p::a", "p2p::b", "p2p::c"])


def test_run_task_cascade_reports_v11_reward_counts(
    tmp_path: Path,
    monkeypatch,
):
    summary, corpus = _run_cascade_with_reward_artifact(
        tmp_path,
        monkeypatch,
        whitelist=_FIXTURE_WHITELIST,
        artifact="reward.json",
        payload=json.dumps(
            _v11_reward(f2p_passed=2, f2p_total=2, p2p_passed=3, p2p_total=3)
        ),
    )

    counts = {
        "apply_failed": False,
        "f2p_passed": 2,
        "f2p_total": 2,
        "p2p_passed": 3,
        "p2p_total": 3,
    }
    assert summary["status"] == "resolved"
    assert summary["reward"] == 1.0
    assert summary["reward_state"] == "valid"
    assert summary["reward_error"] is None
    assert summary["test_inventory"] == {"f2p_total": 2, "p2p_total": 3}
    assert summary["reward_counts"] == counts
    record = wf._validate_lifecycle_record(
        wf._lifecycle_record(summary),
        line_number=1,
    )
    assert record["reward_counts"] == counts
    measurement = corpus.get_measurement(
        summary["hidden_verifier_measurement_ref"]
    )
    assert measurement.values["reward"] == 1.0
    assert {name: measurement.values[name] for name in counts} == {
        "apply_failed": 0.0,
        "f2p_passed": 2.0,
        "f2p_total": 2.0,
        "p2p_passed": 3.0,
        "p2p_total": 3.0,
    }
    raw = json.loads(corpus.get_artifact(measurement.raw_output_ref))
    assert raw["schema_version"] == 3
    assert raw["verifier"]["reward_counts"] == counts
    assert raw["checkpoint"]["reward_counts"] == counts


def test_run_task_cascade_reports_unresolved_counts(
    tmp_path: Path,
    monkeypatch,
):
    summary, _corpus = _run_cascade_with_reward_artifact(
        tmp_path,
        monkeypatch,
        whitelist=_FIXTURE_WHITELIST,
        artifact="reward.json",
        payload=json.dumps(
            _v11_reward(f2p_passed=1, f2p_total=2, p2p_passed=3, p2p_total=3)
        ),
    )

    assert summary["status"] == "unresolved"
    assert summary["reward"] == 0.0
    assert summary["reward_counts"] == {
        "apply_failed": False,
        "f2p_passed": 1,
        "f2p_total": 2,
        "p2p_passed": 3,
        "p2p_total": 3,
    }


@pytest.mark.parametrize(
    ("artifact", "payload", "reason"),
    (
        # A self-consistent, all-passing grade of some other whitelist.
        (
            "reward.json",
            json.dumps(
                _v11_reward(f2p_passed=5, f2p_total=5, p2p_passed=3, p2p_total=3)
            ),
            "whitelist",
        ),
        # The v1.0 token: this task's grader writes reward.json.
        ("reward.txt", "1\n", "reward.txt cannot grade"),
    ),
    ids=("totals_of_another_whitelist", "token_for_whitelisted_task"),
)
def test_run_task_cascade_rejects_reward_not_bound_to_its_whitelist(
    tmp_path: Path,
    monkeypatch,
    artifact: str,
    payload: str,
    reason: str,
):
    summary, corpus = _run_cascade_with_reward_artifact(
        tmp_path,
        monkeypatch,
        whitelist=_FIXTURE_WHITELIST,
        artifact=artifact,
        payload=payload,
    )

    assert summary["status"] == "verification_failed"
    assert summary["verifier_status"] == "reward_mismatch"
    assert summary["failure_phase"] == "hidden_verifier"
    assert summary["failure_kind"] == "reward_mismatch"
    assert summary["reward"] is None
    assert summary["resolved"] is None
    assert summary["reward_state"] == "mismatch"
    assert summary["reward_counts"] is None
    assert reason in summary["reward_error"]
    # The one-shot verifier ran: its ungraded terminal needs explicit rerun.
    assert wf._lifecycle_record(summary)["status_policy"] == "reusable"
    measurement = corpus.get_measurement(
        summary["hidden_verifier_measurement_ref"]
    )
    assert measurement.values["reward_valid"] == 0.0
    assert "reward" not in measurement.values
    assert "f2p_passed" not in measurement.values


def test_run_task_cascade_rejects_malformed_tests_config_before_any_spend(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    (manifest.parent / inst.task_id / "tests" / "config.json").write_text(
        '{"f2p_node_ids": ["a"], "p2p_node_ids": "not a list"}',
        encoding="utf-8",
    )
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: pytest.fail(
            "a malformed tests/config.json must stop the task before Architect"
        ),
    )

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "bad-config"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "cascade_error"
    assert summary["failure_kind"] == "DeepSWEInstanceError"
    assert "tests/config.json p2p_node_ids" in summary["error"]
    assert summary["cost_usd"] == 0.0
    assert summary["hidden_invocation_measurement_ref"] is None
    # Nothing was populated or started for a task that cannot be graded.
    assert "populate" not in calls
    assert "authoring" not in calls


def test_snapshot_whitelist_failure_consumes_no_oracle_authority(
    tmp_path: Path,
    monkeypatch,
):
    """The snapshot whitelist is read before the one-shot invocation marker."""

    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    _declare_test_whitelist(
        manifest,
        inst.task_id,
        f2p=_FIXTURE_WHITELIST[0],
        p2p=_FIXTURE_WHITELIST[1],
    )
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    marker_calls = 0
    persist_marker = wf._persist_hidden_invocation_marker

    def _record_marker(*args, **kwargs):
        nonlocal marker_calls
        marker_calls += 1
        return persist_marker(*args, **kwargs)

    monkeypatch.setattr(wf, "_persist_hidden_invocation_marker", _record_marker)
    inventory_reads: list = []

    def _unreadable_snapshot_whitelist(snapshot):
        inventory_reads.append(snapshot)
        raise wf.DeepSWEInstanceError("cannot safely read tests/config.json")

    monkeypatch.setattr(
        wf,
        "load_snapshot_test_inventory",
        _unreadable_snapshot_whitelist,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: pytest.fail(
            "hidden verifier must not run without a readable whitelist"
        ),
    )

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "bad-snapshot"),
        cost=CostAccumulator(),
    )

    # The live tree parsed and the private copy exists: the failure is the
    # read of that copy, the last step before hidden authority is consumed.
    assert len(inventory_reads) == 1
    assert summary["tests_snapshot"] == inventory_reads[0].to_payload()
    assert summary["status"] == "cascade_error"
    assert summary["status"] not in wf._GRADED_TASK_STATUSES
    assert summary["failure_kind"] == "DeepSWEInstanceError"
    assert "cannot safely read tests/config.json" in summary["error"]
    assert summary["test_inventory"] is None
    assert summary["reward"] is None
    assert summary["reward_counts"] is None
    assert summary["hidden_invocation_measurement_ref"] is None
    assert summary["hidden_invocation_lifecycle_ref"] is None
    assert marker_calls == 0
    assert "verifier" not in calls
    # No one-shot authority was spent, so a plain re-run may try again.
    assert wf._lifecycle_record(summary)["status_policy"] == "retryable"


_ABBREVIATED_PIN = "68dafce"
_FULL_BASE_COMMIT = _ABBREVIATED_PIN + "0123456789abcdef" * 2 + "0"


def _run_cascade_with_abbreviated_base_commit(tmp_path: Path, monkeypatch):
    """Run one stubbed cascade for a task the dataset pins by 7 characters."""

    from crucible.common.cost import CostAccumulator

    assert len(_FULL_BASE_COMMIT) == 40
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(
        tasks_root,
        task_id="task-abbrev",
        ext_id="khabbrev",
        base_commit_hash=_ABBREVIATED_PIN,
    )
    manifest = _write_manifest(
        tasks_root,
        [_entry("task-abbrev", ext_id="khabbrev")],
    )
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    dataset_inst = wf._load_and_filter(ns)[0]
    assert dataset_inst.base_commit_hash == _ABBREVIATED_PIN
    candidate = wf.candidate_from_image_patch(
        b"abbreviated base candidate\n",
        files_modified=("module.py",),
        raw_diff_bytes=27,
        max_bytes=ns.max_patch_bytes,
    )
    calls: dict = {}
    observed: dict = {}
    _patch_container_lifecycle(
        monkeypatch,
        tmp_path,
        calls,
        candidate=candidate,
    )

    def _populate(
        image,
        ws,
        *,
        base_commit=None,
        labels=None,
        register_container=None,
    ):
        calls.setdefault("populate", []).append(
            (image, str(ws), base_commit, labels, register_container)
        )
        return _FULL_BASE_COMMIT

    monkeypatch.setattr(wf, "populate_workspace_from_image", _populate)
    preflight = wf._run_dev_verify_preflight

    def _preflight(**kwargs):
        observed["preflight_metadata"] = kwargs["extra_parameters"][
            "deepswe_metadata"
        ]["base_commit"]
        observed["preflight_spec"] = kwargs["spec"].project_context.base_commit
        return preflight(**kwargs)

    monkeypatch.setattr(wf, "_run_dev_verify_preflight", _preflight)
    settlement = wf._establish_capture_mutation_settlement

    def _settlement(*args, **kwargs):
        observed["settlement"] = kwargs["inst"].base_commit_hash
        return settlement(*args, **kwargs)

    monkeypatch.setattr(wf, "_establish_capture_mutation_settlement", _settlement)

    def _capture(_container_id, *, base_commit, max_bytes):
        observed.setdefault("renders", []).append(base_commit)
        return candidate.patch

    monkeypatch.setattr(wf, "capture_model_patch_from_container", _capture)
    prepare = wf.prepare_candidate_replay

    def _prepare(workspace, applied, *, expected_base_commit, max_bytes):
        observed["apply"] = expected_base_commit
        return prepare(
            workspace,
            applied,
            expected_base_commit=expected_base_commit,
            max_bytes=max_bytes,
        )

    monkeypatch.setattr(wf, "prepare_candidate_replay", _prepare)
    prove = wf.prove_image_replay

    def _prove(*args, expected_base_commit, max_bytes):
        observed["proof"] = expected_base_commit
        return prove(
            *args,
            expected_base_commit=expected_base_commit,
            max_bytes=max_bytes,
        )

    monkeypatch.setattr(wf, "prove_image_replay", _prove)

    def _architect(**kwargs):
        observed["checkpoint_hook"] = kwargs["checkpoint_hook"].expected_base_commit
        observed["architect_metadata"] = kwargs["extra_parameters"][
            "deepswe_metadata"
        ]["base_commit"]
        return _publish_architect_child(Path(kwargs["spec"]), cost_usd=0.0)

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )
    summary = wf._run_task_cascade(
        dataset_inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "abbreviated"),
        cost=CostAccumulator(),
    )
    corpus = Corpus(tmp_path / "s" / "instances" / dataset_inst.task_id / "corpus")
    return ns, dataset_inst, calls, observed, summary, corpus


def test_abbreviated_base_commit_is_bound_as_full_id_at_every_git_comparison(
    tmp_path: Path,
    monkeypatch,
):
    _ns, _inst, calls, observed, summary, corpus = (
        _run_cascade_with_abbreviated_base_commit(tmp_path, monkeypatch)
    )

    assert summary["status"] == "resolved"
    assert summary["base_commit"] == _FULL_BASE_COMMIT
    # Only the first population sees the dataset pin; it resolves the id that
    # the other three checkouts must then match exactly.
    assert [
        (Path(entry[1]).name, entry[2]) for entry in calls["populate"]
    ] == [
        ("candidate", _ABBREVIATED_PIN),
        ("baseline", _FULL_BASE_COMMIT),
        ("verifier", _FULL_BASE_COMMIT),
        ("grading", _FULL_BASE_COMMIT),
    ]
    assert observed == {
        "preflight_metadata": _FULL_BASE_COMMIT,
        "preflight_spec": _FULL_BASE_COMMIT,
        "checkpoint_hook": _FULL_BASE_COMMIT,
        "architect_metadata": _FULL_BASE_COMMIT,
        "settlement": _FULL_BASE_COMMIT,
        "renders": [_FULL_BASE_COMMIT, _FULL_BASE_COMMIT],
        "apply": _FULL_BASE_COMMIT,
        "proof": _FULL_BASE_COMMIT,
    }
    assert corpus.latest_spec().project_context.base_commit == _FULL_BASE_COMMIT


def test_abbreviated_base_commit_keeps_the_dataset_pin_as_checkpoint_identity(
    tmp_path: Path,
    monkeypatch,
):
    ns, dataset_inst, _calls, _observed, summary, corpus = (
        _run_cascade_with_abbreviated_base_commit(tmp_path, monkeypatch)
    )

    # Resume recomputes the fingerprint from the dataset instance, so the
    # resolved id must never enter task identity.
    assert summary["checkpoint_fingerprint"] == wf._deepswe_checkpoint_fingerprint(
        wf.dataclasses.replace(
            dataset_inst,
            docker_image=summary["image_digest"],
        ),
        args=ns,
        child_budget_usd=summary["budget_usd"],
    )
    for ref in (
        summary["candidate_measurement_ref"],
        summary["hidden_invocation_measurement_ref"],
        summary["hidden_verifier_measurement_ref"],
    ):
        assert corpus.get_measurement(ref).commit == _ABBREVIATED_PIN
    checkpoint = wf._load_hidden_oracle_checkpoint(ns.spec, dataset_inst)
    assert checkpoint["status"] == "resolved"
    assert checkpoint["checkpoint_fingerprint"] == summary["checkpoint_fingerprint"]


@pytest.mark.parametrize(
    "returned",
    (None, _ABBREVIATED_PIN, "F" * 40, "0" * 41),
    ids=("none", "abbreviated", "upper_case", "wrong_length"),
)
def test_run_task_cascade_binds_only_a_full_base_commit_id(
    tmp_path: Path,
    monkeypatch,
    returned,
):
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _populate(image, ws, **kwargs):
        calls.setdefault("populate", []).append(str(ws))
        return returned

    monkeypatch.setattr(wf, "populate_workspace_from_image", _populate)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: pytest.fail(
            "an unresolved base commit must stop the task before Architect"
        ),
    )

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "unresolved-base"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "setup_failed"
    assert summary["failure_phase"] == "container_setup"
    assert summary["failure_kind"] == "InstanceContainerSetupError"
    assert "full base commit id" in summary["error"]
    assert summary["base_commit"] is None
    assert len(calls["populate"]) == 1
    assert "authoring" not in calls
    assert "architect_operation_id" not in summary
    assert wf._all_task_failures_are_unjudged([summary]) is True
    # No starter Spec was synthesized from an id that is not the checkout's.
    assert (
        Corpus(
            tmp_path / "s" / "instances" / inst.task_id / "corpus"
        ).latest_spec()
        is None
    )


def test_run_task_cascade_grades_salvaged_candidate_when_cascade_raises(
    tmp_path: Path,
    monkeypatch,
):
    """Authoring failure is diagnostic when exact candidate bytes survive."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]

    candidate = wf.candidate_from_image_patch(
        b"nonempty candidate patch\n",
        files_modified=("adaptix/_internal.py",),
        raw_diff_bytes=25,
        max_bytes=ns.max_patch_bytes,
    )
    calls: dict = {}
    _patch_container_lifecycle(
        monkeypatch,
        tmp_path,
        calls,
        candidate=candidate,
    )
    _patch_partial_tier_settlement(monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError("cascade blew up")

    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        _boom,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 1,
            "reward": 0.0,
            "resolved": 0.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )

    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "rd")
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
    )

    assert summary["status"] == "unresolved"
    assert summary["authoring_status"] == "error"
    assert summary["failure_phase"] == "authoring"
    assert summary["failure_kind"] == "RuntimeError"
    assert summary["mutation_journal_settlement"] is not None
    assert summary["candidate_capture_status"] == "captured"
    assert len(calls["verifier"]) == 1
    # Authoring container torn down + image reclaimed once.
    image_reclaims = [c for c in calls["teardown"] if c[1] is not None]
    assert len(image_reclaims) == 1


def test_failed_authoring_empty_candidate_skips_hidden_verifier_after_tier(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    _patch_partial_tier_settlement(monkeypatch)

    def _architect(**kwargs):
        _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=0.0,
            status="partial_failed",
            tiers_executed=1,
            failure_phase="tier",
            failure_kind="dev_verify_failed",
        )
        return 1

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "prepare_candidate_replay",
        lambda *args, **kwargs: pytest.fail(
            "failed authoring with no candidate must not enter replay"
        ),
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: pytest.fail(
            "failed authoring with no candidate must not invoke hidden verifier"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "failed-empty-candidate"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "candidate_failed"
    assert summary["rc"] == 1
    assert summary["authoring_status"] == "failed"
    assert summary["architect_tiers_executed"] == 1
    assert summary["zero_tier_settlement_proven"] is False
    assert summary["failure_phase"] == "tier"
    assert summary["failure_kind"] == "dev_verify_failed"
    assert summary["candidate_capture_status"] == "empty"
    assert summary["candidate"]["diff_bytes"] == 0
    assert summary["candidate_measurement_ref"]
    assert summary["hidden_invocation_measurement_ref"] is None
    assert summary["hidden_verifier_measurement_ref"] is None
    assert summary["verifier_status"] == "not_run"
    assert "verifier" not in calls
    lifecycle = wf._lifecycle_record(summary)
    assert lifecycle["status_policy"] == "retryable"
    assert wf._checkpoint_is_reusable(lifecycle, args=ns) is False


def test_run_task_cascade_enforces_instance_storage_growth_cap(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.common.storage_quota import StorageQuotaExceeded

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _architect(**kwargs):
        child_run = Path(kwargs["spec"]) / "runs" / "quota_architect"
        child_run.mkdir(parents=True)
        (child_run / "summary.json").write_text(
            json.dumps(
                {
                    "costs": {"total_usd": 0.0, "rollup_incomplete": False},
                }
            )
        )
        return 0

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )

    quota_init: dict = {}
    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "outer")

    class _TripQuota:
        peak_growth_bytes = 12345
        exceeded = False

        def __init__(self, paths, *, limit_bytes, on_exceeded, require_kernel):
            quota_init["paths"] = tuple(Path(path) for path in paths)
            quota_init["limit_bytes"] = limit_bytes
            quota_init["require_kernel"] = require_kernel
            self.on_exceeded = on_exceeded

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.on_exceeded()
            raise StorageQuotaExceeded("test quota exceeded")

    docker_calls: list[list[str]] = []
    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _TripQuota)
    monkeypatch.setattr(
        wf.subprocess,
        "run",
        lambda command, **kwargs: docker_calls.append(command),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
    )

    # The one-shot marker was already committed before /tests was mounted, so
    # a later quota failure remains verification_incomplete, never retryable.
    assert summary["status"] == "verification_incomplete"
    assert summary["failure_phase"] == "storage_quota"
    assert summary["storage_limit_bytes"] == inst.storage_mb * 1024 * 1024
    assert summary["storage_peak_growth_bytes"] == 12345
    assert quota_init["limit_bytes"] == inst.storage_mb * 1024 * 1024
    assert quota_init["require_kernel"] is True
    assert len(quota_init["paths"]) == 3
    # The verifier is strictly stopped and inspected before final quota
    # closure, so the quota callback has no live container left to kill.
    assert docker_calls == []
    # Quota authority closes before hidden-verifier evidence is published.
    assert summary["hidden_verifier_measurement_ref"] is None
    assert summary["hidden_verifier_lifecycle_ref"] is None
    assert summary["hidden_invocation_measurement_ref"]


def test_hidden_patch_mismatch_fails_closed_but_persists_atomic_evidence(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    (manifest.parent / inst.task_id / "tests").mkdir(parents=True, exist_ok=True)
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )
    candidate = _empty_candidate()
    patch_proof_calls = 0

    def _patch_proof(*args, **kwargs):
        nonlocal patch_proof_calls
        patch_proof_calls += 1
        if patch_proof_calls == 1:
            return wf.VerifierPatchReport(
                status=wf.VerifierPatchStatus.MATCHED,
                candidate_sha256=candidate.sha256,
                observed_sha256=candidate.sha256,
                observed_bytes=candidate.diff_bytes,
            )
        return wf.VerifierPatchReport(
            status=wf.VerifierPatchStatus.MISMATCH,
            candidate_sha256=candidate.sha256,
            observed_sha256="f" * 64,
            observed_bytes=1,
            error="drift",
        )

    monkeypatch.setattr(wf, "verify_hidden_model_patch", _patch_proof)

    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "mismatch")
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
    )

    assert summary["status"] == "verification_failed"
    assert patch_proof_calls == 2
    assert len(calls["verifier"]) == 1
    assert summary["verifier_status"] == "patch_mismatch"
    assert summary["failure_phase"] == "hidden_verifier_artifact"
    corpus = Corpus(ns.spec / "instances" / inst.task_id / "corpus")
    latest = corpus.latest_spec()
    assert summary["candidate_measurement_ref"] in latest.measurement_refs
    assert summary["hidden_verifier_measurement_ref"] in latest.measurement_refs
    hidden = corpus.get_measurement(summary["hidden_verifier_measurement_ref"])
    payload = json.loads(corpus.get_artifact(hidden.raw_output_ref))
    assert payload["verifier_patch"]["status"] == "mismatch"
    assert payload["candidate"]["sha256"] == candidate.sha256
    checkpoint = wf._load_hidden_oracle_checkpoint(ns.spec, inst)
    assert checkpoint["status"] == "verification_failed"

    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *args, **kwargs: pytest.fail(
            "finalized failed oracle must not implicitly rerun"
        ),
    )

    class _NoStart:
        def write_event(self, event):
            pytest.fail(f"failed oracle should resume without start: {event}")

    from crucible.common.cost import CostAccumulator

    restored, resumed = wf._execute_task_with_checkpoint(
        inst,
        args=ns,
        run_dir=tmp_path / "s" / "runs" / "mismatch-resume",
        cost=CostAccumulator(),
        child_budget_usd=summary["budget_usd"],
        handle=_NoStart(),
        index=1,
    )
    assert resumed is True
    assert restored["status"] == "verification_failed"


def test_hidden_patch_preload_failure_consumes_no_oracle_authority(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    (manifest.parent / inst.task_id / "tests").mkdir(parents=True, exist_ok=True)
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    marker_calls = 0
    persist_marker = wf._persist_hidden_invocation_marker

    def _record_marker(*args, **kwargs):
        nonlocal marker_calls
        marker_calls += 1
        return persist_marker(*args, **kwargs)

    monkeypatch.setattr(wf, "_persist_hidden_invocation_marker", _record_marker)
    candidate = _empty_candidate()
    monkeypatch.setattr(
        wf,
        "verify_hidden_model_patch",
        lambda *args, **kwargs: wf.VerifierPatchReport(
            status=wf.VerifierPatchStatus.MISMATCH,
            candidate_sha256=candidate.sha256,
            observed_sha256="f" * 64,
            observed_bytes=1,
            error="preload drift",
        ),
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: pytest.fail(
            "hidden verifier must not run after preload proof failure"
        ),
    )

    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=_prepare_parent_run(tmp_path / "s" / "runs" / "preload-fail"),
        cost=CostAccumulator(),
    )

    assert summary["status"] == "cascade_error"
    assert summary["verifier_patch_preload"]["status"] == "mismatch"
    assert summary["hidden_invocation_measurement_ref"] is None
    assert summary["hidden_invocation_lifecycle_ref"] is None
    assert marker_calls == 0
    assert "verifier" not in calls


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (
            {
                "rc": 0,
                "reward": 1.0,
                "resolved": 1.0,
                "verify_returncode": 0.0,
                "reward_valid": True,
            },
            "resolved",
        ),
        (
            {
                "rc": 1,
                "reward": 0.0,
                "resolved": 0.0,
                "verify_returncode": 0.0,
                "reward_valid": True,
            },
            "unresolved",
        ),
        (
            {
                "rc": 1,
                "reward": 0.0,
                "resolved": 0.0,
                "verify_returncode": 137.0,
                "reward_valid": False,
            },
            "verification_failed",
        ),
        (
            {
                "rc": 1,
                "reward": 0.0,
                "resolved": 0.0,
                "command_returncode": 137.0,
                "verify_returncode": 0.0,
                "reward_valid": False,
            },
            "verification_failed",
        ),
    ],
)
def test_dataset_verdict_is_authoritative(verdict, expected):
    assert wf._dataset_verdict_status(verdict) == expected


def test_post_stop_reward_token_is_sole_kpi_authority(tmp_path: Path):
    reward = tmp_path / "verifier" / "reward.txt"
    reward.parent.mkdir()
    reward.write_text("1\n", encoding="utf-8")
    verdict = {
        "rc": 0,
        "reward": None,
        "resolved": None,
        "verify_returncode": 0.0,
        "reward_pending": True,
    }

    validated = wf._validate_stopped_verifier_reward(
        verdict,
        instance_log_dir=tmp_path,
        test_inventory=None,
    )

    assert validated["raw_reward"] == "1"
    assert validated["reward"] == 1.0
    assert validated["resolved"] == 1.0
    assert validated["reward_pending"] is False
    assert validated["reward_valid"] is True
    assert validated["reward_state"] == "valid"
    assert validated["reward_counts"] is None
    assert validated["reward_error"] is None
    assert validated["verifier_status"] == "resolved"
    assert wf._dataset_verdict_status(validated) == "resolved"


def _pending_verdict() -> dict:
    return {
        "rc": 0,
        "reward": None,
        "resolved": None,
        "verify_returncode": 0.0,
        "reward_pending": True,
    }


def test_post_stop_v11_json_reward_is_sole_kpi_authority(tmp_path: Path):
    reward = tmp_path / "verifier" / "reward.json"
    reward.parent.mkdir()
    reward.write_text(
        json.dumps(
            _v11_reward(f2p_passed=44, f2p_total=44, p2p_passed=12, p2p_total=12)
        ),
        encoding="utf-8",
    )

    validated = wf._validate_stopped_verifier_reward(
        _pending_verdict(),
        instance_log_dir=tmp_path,
        test_inventory=RewardTestInventory(f2p_total=44, p2p_total=12),
    )

    assert validated["raw_reward"] == "1"
    assert validated["reward"] == 1.0
    assert validated["resolved"] == 1.0
    assert validated["reward_pending"] is False
    assert validated["reward_valid"] is True
    assert validated["reward_state"] == "valid"
    assert validated["reward_error"] is None
    assert validated["reward_counts"] == {
        "apply_failed": False,
        "f2p_passed": 44,
        "f2p_total": 44,
        "p2p_passed": 12,
        "p2p_total": 12,
    }
    assert validated["verifier_status"] == "resolved"
    assert wf._dataset_verdict_status(validated) == "resolved"


@pytest.mark.parametrize(
    ("artifact", "document", "inventory", "reason"),
    (
        # reward 1 although a fail-to-pass test failed
        (
            "reward.json",
            {
                **_v11_reward(
                    f2p_passed=43, f2p_total=44, p2p_passed=12, p2p_total=12
                ),
                "reward": 1,
            },
            RewardTestInventory(f2p_total=44, p2p_total=12),
            "reward disagrees with its counts",
        ),
        # a self-consistent grade of some other whitelist
        (
            "reward.json",
            _v11_reward(f2p_passed=5, f2p_total=5, p2p_passed=3, p2p_total=3),
            RewardTestInventory(f2p_total=44, p2p_total=12),
            "whitelist",
        ),
        # the same, with only the pass-to-pass total of another whitelist
        (
            "reward.json",
            _v11_reward(f2p_passed=44, f2p_total=44, p2p_passed=11, p2p_total=11),
            RewardTestInventory(f2p_total=44, p2p_total=12),
            "whitelist",
        ),
        # the same, with only the fail-to-pass total of another whitelist
        (
            "reward.json",
            _v11_reward(f2p_passed=43, f2p_total=43, p2p_passed=12, p2p_total=12),
            RewardTestInventory(f2p_total=44, p2p_total=12),
            "whitelist",
        ),
        # counts although the tests snapshot declares no whitelist
        (
            "reward.json",
            _v11_reward(f2p_passed=44, f2p_total=44, p2p_passed=12, p2p_total=12),
            None,
            "cannot be bound to a tests/config.json whitelist",
        ),
        # a count-less token although the tests snapshot declares a whitelist
        (
            "reward.txt",
            None,
            RewardTestInventory(f2p_total=44, p2p_total=12),
            "reward.txt cannot grade a task whose tests declare a whitelist",
        ),
    ),
    ids=(
        "self_inconsistent",
        "totals_of_another_whitelist",
        "pass_to_pass_total_of_another_whitelist",
        "fail_to_pass_total_of_another_whitelist",
        "counts_without_whitelist",
        "token_with_whitelist",
    ),
)
def test_post_stop_reward_mismatch_is_not_a_grade(
    tmp_path: Path,
    artifact: str,
    document: dict | None,
    inventory: RewardTestInventory | None,
    reason: str,
):
    reward = tmp_path / "verifier" / artifact
    reward.parent.mkdir()
    reward.write_text(
        "1\n" if document is None else json.dumps(document),
        encoding="utf-8",
    )

    validated = wf._validate_stopped_verifier_reward(
        _pending_verdict(),
        instance_log_dir=tmp_path,
        test_inventory=inventory,
    )

    assert validated["raw_reward"] == ""
    assert validated["reward"] is None
    assert validated["resolved"] is None
    assert validated["reward_valid"] is False
    assert validated["reward_state"] == "mismatch"
    assert validated["reward_counts"] is None
    assert reason in validated["reward_error"]
    assert validated["verifier_status"] == "reward_mismatch"
    assert wf._dataset_verdict_status(validated) == "verification_failed"


@pytest.mark.parametrize(
    ("command_returncode", "verifier_status"),
    ((1.0, "command_failed"), (124.0, "timeout")),
    ids=("command_failed", "timeout"),
)
def test_post_stop_valid_reward_of_failed_verifier_is_not_a_grade(
    tmp_path: Path,
    command_returncode: float,
    verifier_status: str,
):
    """The task status decides what is a grade, not the artifact alone."""

    reward = tmp_path / "verifier" / "reward.json"
    reward.parent.mkdir()
    reward.write_text(
        json.dumps(
            _v11_reward(f2p_passed=44, f2p_total=44, p2p_passed=12, p2p_total=12)
        ),
        encoding="utf-8",
    )

    validated = wf._validate_stopped_verifier_reward(
        {**_pending_verdict(), "command_returncode": command_returncode},
        instance_log_dir=tmp_path,
        test_inventory=RewardTestInventory(f2p_total=44, p2p_total=12),
    )

    status = wf._dataset_verdict_status(validated)
    assert validated["verifier_status"] == verifier_status
    assert status == "verification_failed"
    assert status not in wf._GRADED_TASK_STATUSES
    # The result keeps what its valid artifact held; the run roll-up counts
    # the task ungraded and leaves these counts out of its sum.
    assert validated["reward_valid"] is True
    assert validated["reward"] == 1.0
    assert validated["reward_counts"] == {
        "apply_failed": False,
        "f2p_passed": 44,
        "f2p_total": 44,
        "p2p_passed": 12,
        "p2p_total": 12,
    }


def test_post_stop_reward_json_without_contract_fields_is_not_a_grade(
    tmp_path: Path,
):
    reward = tmp_path / "verifier" / "reward.json"
    reward.parent.mkdir()
    reward.write_text(
        '{"reward":1,"f2p_passed":44,"f2p_total":44}',
        encoding="utf-8",
    )

    validated = wf._validate_stopped_verifier_reward(
        _pending_verdict(),
        instance_log_dir=tmp_path,
        test_inventory=RewardTestInventory(f2p_total=44, p2p_total=12),
    )

    assert validated["reward"] is None
    assert validated["resolved"] is None
    assert validated["reward_state"] == "invalid"
    assert validated["reward_counts"] is None
    assert validated["reward_error"] == (
        "reward.json fields are not the exact v1.1 contract"
    )
    assert validated["verifier_status"] == "reward_invalid"
    assert wf._dataset_verdict_status(validated) == "verification_failed"


def test_dataset_runner_defers_all_reward_reads_until_stop(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.products.deepswe.runners import deepswe_verifier

    captured: dict = {}

    def fake_main(stdin, stdout):
        captured.update(json.loads(stdin.read()))
        json.dump(
            {
                "values": {
                    "command_returncode": 0.0,
                    "verify_returncode": 0.0,
                    "baseline_test_returncode": 0.0,
                    "new_test_returncode": 70.0,
                    "diagnostics_persisted": 1.0,
                    "verifier_stdout_bytes": 131201.0,
                    "verifier_stderr_bytes": 2048.0,
                    "verifier_stdout_truncated": 1.0,
                    "verifier_stderr_truncated": 0.0,
                    "reward_pending": 1.0,
                },
            },
            stdout,
        )
        return 0

    def forbidden_read(_logs_dir):
        raise AssertionError("pre-stop workflow must not open reward.txt")

    monkeypatch.setattr(deepswe_verifier, "main", fake_main)
    monkeypatch.setattr(wf, "read_reward_artifact", forbidden_read)

    verdict = wf._run_dataset_verify(
        object(),
        container_id="verifier-cid",
        instance_log_dir=tmp_path,
    )

    metadata = captured["parameters"]["deepswe_metadata"]
    assert metadata["defer_reward_read"] is True
    assert verdict["reward"] is None
    assert verdict["resolved"] is None
    assert verdict["reward_pending"] is True
    assert verdict["reward_valid"] is False
    assert verdict["reward_state"] == "pending_stop_validation"
    assert verdict["command_returncode"] == 0.0
    assert verdict["verify_returncode"] == 0.0
    assert verdict["baseline_test_returncode"] == 0.0
    assert verdict["new_test_returncode"] == 70.0
    assert verdict["diagnostics_persisted"] is True
    assert verdict["verifier_stdout_bytes"] == 131201.0
    assert verdict["verifier_stderr_bytes"] == 2048.0
    assert verdict["verifier_stdout_truncated"] is True
    assert verdict["verifier_stderr_truncated"] is False


def test_dataset_runner_preserves_legacy_returncode_alias(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.products.deepswe.runners import deepswe_verifier

    def legacy_main(_stdin, stdout):
        json.dump(
            {
                "values": {
                    "verify_returncode": 0.0,
                    "reward_pending": 1.0,
                },
            },
            stdout,
        )
        return 0

    monkeypatch.setattr(deepswe_verifier, "main", legacy_main)

    verdict = wf._run_dataset_verify(
        object(),
        container_id="verifier-cid",
        instance_log_dir=tmp_path,
    )

    assert verdict["verify_returncode"] == 0.0
    assert verdict["command_returncode"] == 0.0
    assert verdict["baseline_test_returncode"] is None
    assert verdict["new_test_returncode"] is None
    assert verdict["diagnostics_persisted"] is None


def test_post_stop_reward_leaf_symlink_fails_closed(tmp_path: Path):
    forged = tmp_path / "forged-reward"
    forged.write_text("1\n", encoding="utf-8")
    reward = tmp_path / "verifier" / "reward.txt"
    reward.parent.mkdir()
    reward.symlink_to(forged)

    validated = wf._validate_stopped_verifier_reward(
        {
            "rc": 0,
            "reward": None,
            "resolved": None,
            "verify_returncode": 0.0,
            "reward_pending": True,
        },
        instance_log_dir=tmp_path,
        test_inventory=None,
    )

    assert validated["raw_reward"] == ""
    assert validated["reward"] is None
    assert validated["resolved"] is None
    assert validated["reward_valid"] is False
    assert validated["reward_state"] == "unreadable"
    assert validated["reward_error"]
    assert validated["verifier_status"] == "reward_invalid"


def test_post_stop_reward_oversize_fails_closed(tmp_path: Path):
    reward = tmp_path / "verifier" / "reward.txt"
    reward.parent.mkdir()
    reward.write_bytes(b"1\nforged")

    validated = wf._validate_stopped_verifier_reward(
        {
            "rc": 0,
            "reward": None,
            "resolved": None,
            "verify_returncode": 0.0,
            "reward_pending": True,
        },
        instance_log_dir=tmp_path,
        test_inventory=None,
    )

    assert validated["raw_reward"] == ""
    assert validated["reward"] is None
    assert validated["resolved"] is None
    assert validated["reward_valid"] is False
    assert validated["reward_state"] == "unreadable"


def test_unresolved_hidden_verifier_supersedes_prior_landed_lifecycle(
    tmp_path: Path,
):
    """The benchmark oracle must be terminal even after cascade landing."""
    from assay import LifecycleEvent
    from assay.corpus.store import Corpus
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(tmp_path / "task-corpus")
    corpus.put_spec(synthesize_starter_spec(inst))
    prior_commit, _ = corpus.commit_lifecycle_event(
        lambda committed_version: LifecycleEvent(
            kind="landed",
            reason="cascade completed before hidden evaluation",
            affected_versions=(committed_version,),
            execution=wf.exec_ctx_now("test"),
        ),
    )
    prior = prior_commit.spec

    persisted = wf._persist_dataset_outcome(
        corpus,
        inst,
        {
            "rc": 1,
            "reward": 0.0,
            "resolved": 0.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
        status="unresolved",
        run_dir=tmp_path / "runs" / "attempt-1",
        checkpoint_record=wf._lifecycle_record(
            _lifecycle_summary(
                inst.task_id,
                inst.ext_id,
                status="unresolved",
            )
        ),
        **_exact_empty_candidate_evidence(inst),
    )

    latest = corpus.latest_spec()
    assert latest.version == prior.version + 1
    assert persisted["instance_spec_version"] == latest.version
    assert persisted["hidden_verifier_measurement_ref"] in latest.measurement_refs
    terminal = corpus.get_record(
        latest.lifecycle_event_refs[-1],
        as_type=LifecycleEvent,
    )
    assert terminal.kind == "abandoned"
    assert terminal.affected_versions == (latest.version,)
    assert "status=unresolved" in terminal.reason


def test_hidden_oracle_corpus_checkpoint_closes_lifecycle_crash_window(
    tmp_path: Path,
    monkeypatch,
):
    """A committed oracle is resumed when lifecycle JSONL was never written."""

    from crucible.common.cost import CostAccumulator
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(ns.spec / "instances" / inst.task_id / "corpus")
    corpus.put_spec(synthesize_starter_spec(inst))
    checkpoint = _lifecycle_summary(inst.task_id, inst.ext_id)
    immutable_image = "public.ecr.aws/example/task@sha256:" + "a" * 64
    checkpoint.update(
        {
            "repo": inst.repo,
            "language": inst.language,
            "category": inst.category,
            "image_digest": immutable_image,
        }
    )
    checkpoint_inst = wf.dataclasses.replace(inst, docker_image=immutable_image)
    checkpoint["checkpoint_fingerprint"] = wf._deepswe_checkpoint_fingerprint(
        checkpoint_inst,
        args=ns,
        child_budget_usd=None,
    )
    checkpoint_record = wf._lifecycle_record(checkpoint)
    wf._persist_dataset_outcome(
        corpus,
        inst,
        {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
        status="resolved",
        run_dir=tmp_path / "runs" / "crashed-attempt",
        checkpoint_record=checkpoint_record,
        **_exact_empty_candidate_evidence(inst),
    )
    assert not lifecycle.exists()  # the simulated process died here
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *args, **kwargs: pytest.fail("hidden verifier must not rerun"),
    )

    class _Handle:
        def write_event(self, event):
            pytest.fail(f"resumed task must not start: {event}")

    restored, resumed = wf._execute_task_with_checkpoint(
        inst,
        args=ns,
        run_dir=tmp_path / "runs" / "resume",
        cost=CostAccumulator(),
        child_budget_usd=None,
        handle=_Handle(),
        index=1,
    )
    assert resumed is True
    assert restored["status"] == "resolved"
    assert restored["resumed"] is True
    persisted = wf._load_lifecycle_jsonl(lifecycle)[inst.task_id]
    assert persisted["checkpoint_fingerprint"] == checkpoint["checkpoint_fingerprint"]


def test_newer_candidate_only_terminal_supersedes_old_hidden_oracle(
    tmp_path: Path,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(ns.spec / "instances" / inst.task_id / "corpus")
    corpus.put_spec(synthesize_starter_spec(inst))
    evidence = _exact_empty_candidate_evidence(inst)
    wf._persist_dataset_outcome(
        corpus,
        inst,
        {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
        status="resolved",
        run_dir=tmp_path / "runs" / "old-hidden",
        checkpoint_record=wf._lifecycle_record(
            _lifecycle_summary(inst.task_id, inst.ext_id, status="resolved")
        ),
        **evidence,
    )
    wf._persist_candidate_only_outcome(
        corpus,
        inst,
        evidence["candidate"],
        status="candidate_failed",
        run_dir=tmp_path / "runs" / "new-failed-rerun",
        checkpoint_record=wf._lifecycle_record(
            _lifecycle_summary(
                inst.task_id,
                inst.ext_id,
                status="candidate_failed",
            )
        ),
    )

    assert wf._load_hidden_oracle_checkpoint(ns.spec, inst) is None


def test_hidden_resume_requires_adjacent_paired_candidate_measurement(
    tmp_path: Path,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(ns.spec / "instances" / inst.task_id / "corpus")
    corpus.put_spec(synthesize_starter_spec(inst))
    evidence = _exact_empty_candidate_evidence(inst)
    persisted = wf._persist_dataset_outcome(
        corpus,
        inst,
        {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
        status="resolved",
        run_dir=tmp_path / "runs" / "hidden",
        checkpoint_record=wf._lifecycle_record(
            _lifecycle_summary(inst.task_id, inst.ext_id, status="resolved")
        ),
        **evidence,
    )
    hidden = corpus.get_measurement(persisted["hidden_verifier_measurement_ref"])
    corpus.commit(
        measurements=(
            wf.dataclasses.replace(
                hidden,
                config_hash=hidden.config_hash + ":unpaired",
            ),
        )
    )

    with pytest.raises(wf._LifecycleCheckpointError, match="not adjacent"):
        wf._load_hidden_oracle_checkpoint(ns.spec, inst)


def test_legacy_unpaired_hidden_oracle_is_non_reusable_not_fatal(
    tmp_path: Path,
):
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(ns.spec / "instances" / inst.task_id / "corpus")
    corpus.put_spec(synthesize_starter_spec(inst))
    raw_ref = corpus.put_artifact(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "deepswe_hidden_verifier",
                "task_id": inst.task_id,
                "ext_id": inst.ext_id,
                "status": "resolved",
                "checkpoint": wf._lifecycle_record(
                    _lifecycle_summary(inst.task_id, inst.ext_id, status="resolved")
                ),
            }
        ).encode(),
        kind="bench_raw",
    )
    corpus.commit(
        measurements=(
            wf.Measurement(
                bench_id=f"deepswe/{inst.task_id}/hidden_verifier",
                commit=inst.base_commit_hash,
                config_hash="legacy-unpaired",
                values={"reward": 1.0},
                raw_output_ref=raw_ref,
                execution=wf.exec_ctx_now("test"),
            ),
        )
    )

    assert wf._load_hidden_oracle_checkpoint(ns.spec, inst) is None


def test_hidden_invocation_marker_blocks_implicit_rerun_after_process_gap(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.common.cost import CostAccumulator

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    (manifest.parent / inst.task_id / "tests").mkdir(exist_ok=True)
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("process died after oracle invocation")
        ),
    )
    assigned_budget = ns.budget_usd_per_instance
    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "crashed")

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
        child_budget_usd=assigned_budget,
    )

    assert summary["status"] == "verification_incomplete"
    assert summary["hidden_invocation_measurement_ref"]
    assert summary["hidden_verifier_measurement_ref"] is None
    marker = wf._load_hidden_oracle_checkpoint(ns.spec, inst)
    assert marker["status"] == "verification_incomplete"
    assert marker["hidden_invocation_state"] == "started"

    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *args, **kwargs: pytest.fail("implicit oracle rerun forbidden"),
    )

    class _NoStart:
        def write_event(self, event):
            pytest.fail(f"blocked invocation must resume without start: {event}")

    restored, resumed = wf._execute_task_with_checkpoint(
        inst,
        args=ns,
        run_dir=tmp_path / "s" / "runs" / "resume",
        cost=CostAccumulator(),
        child_budget_usd=assigned_budget,
        handle=_NoStart(),
        index=1,
    )
    assert resumed is True
    assert restored["status"] == "verification_incomplete"

    # Explicit authority is the only way to consume another invocation.
    ns.rerun = True
    rerun_calls: list[str] = []

    def _explicit_rerun(current, **kwargs):
        rerun_calls.append(current.task_id)
        return {
            "task_id": current.task_id,
            "ext_id": current.ext_id,
            "status": "candidate_failed",
            "cost_usd": 0.0,
            "cost_incomplete": False,
            "budget_usd": assigned_budget,
            "image_digest": current.docker_image,
        }

    monkeypatch.setattr(wf, "_run_task_cascade", _explicit_rerun)

    class _Handle:
        def write_event(self, event):
            pass

    rerun_summary, rerun_resumed = wf._execute_task_with_checkpoint(
        inst,
        args=ns,
        run_dir=tmp_path / "s" / "runs" / "explicit-rerun",
        cost=CostAccumulator(),
        child_budget_usd=assigned_budget,
        handle=_Handle(),
        index=1,
    )
    assert rerun_resumed is False
    assert rerun_summary["status"] == "candidate_failed"
    assert rerun_calls == [inst.task_id]


def test_hidden_oracle_persistence_failure_has_no_visible_terminal(
    tmp_path: Path,
    monkeypatch,
):
    """Failed atomic publication leaves no resumable oracle checkpoint."""

    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    corpus = Corpus(tmp_path / "task-corpus")
    corpus.put_spec(synthesize_starter_spec(inst))
    checkpoint = wf._lifecycle_record(_lifecycle_summary(inst.task_id, inst.ext_id))
    monkeypatch.setattr(
        corpus,
        "commit",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        wf._persist_dataset_outcome(
            corpus,
            inst,
            {
                "rc": 0,
                "reward": 1.0,
                "resolved": 1.0,
                "verify_returncode": 0.0,
                "reward_valid": True,
            },
            status="resolved",
            run_dir=tmp_path / "runs" / "attempt",
            checkpoint_record=checkpoint,
            **_exact_empty_candidate_evidence(inst),
        )
    assert corpus.latest_spec().measurement_refs == ()


def test_existing_task_lineage_forces_restart_on_fresh_workspace(
    tmp_path: Path,
    monkeypatch,
):
    """A stable Corpus plus a run-local workspace must replay slices."""
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    (manifest.parent / inst.task_id / "tests").mkdir(exist_ok=True)
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)
    restart_modes: list[tuple[bool, bool]] = []

    def _architect(**kwargs):
        restart_modes.append((kwargs["restart"], kwargs["fresh_lineage"]))
        return 0

    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        _architect,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *a, **kw: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )
    from crucible.common.cost import CostAccumulator

    runs = tmp_path / "s" / "runs"
    for name in ("r1", "r2"):
        run_dir = _prepare_parent_run(runs / name)
        result = wf._run_task_cascade(
            inst,
            args=ns,
            run_dir=run_dir,
            cost=CostAccumulator(),
        )
        assert result["status"] == "resolved"
    assert restart_modes == [(False, False), (True, True)]


def test_implement_exit_fails_when_hidden_reward_is_zero(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
        ]
    )
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *a, **kw: {
            "task_id": "t",
            "ext_id": "e",
            "status": "unresolved",
            "reward": 0.0,
            "resolved": 0.0,
        },
    )
    monkeypatch.setattr(
        RunHandle,
        "elapsed_s",
        property(lambda _: 37.5),
    )
    assert wf._implement_serial(wf._load_and_filter(ns), args=ns) == 1
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert run_dir.name.startswith("20")
    assert "_deepswe_" in run_dir.name
    assert (run_dir / "summary.json").is_file()
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "tasks_failed"
    assert result["duration_s"] == 37.5
    events = (run_dir / "events.jsonl").read_text().splitlines()
    assert any(json.loads(line)["phase"] == "task_complete" for line in events)


def test_implement_preserves_retryable_architect_failure_at_run_boundary(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
        ]
    )
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *a, **kw: {
            "task_id": "t",
            "ext_id": "e",
            "status": "candidate_failed",
            "architect_terminal_class": "retryable_failure",
            "hidden_invocation_measurement_ref": None,
            "cost_incomplete": False,
            "cleanup_status": "complete",
        },
    )

    assert wf._implement_serial(wf._load_and_filter(ns), args=ns) == 1
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "retryable_tasks_failed"
    assert outcome["detail_status"] == "retryable_tasks_failed"
    assert outcome["terminal_class"] == "retryable_failure"


@pytest.mark.parametrize(
    ("drop_field", "override"),
    [
        (None, {"architect_terminal_class": "permanent_failure"}),
        (
            None,
            {"hidden_invocation_measurement_ref": "sha256:" + "a" * 64},
        ),
        ("hidden_invocation_measurement_ref", {}),
        (None, {"cost_incomplete": True}),
        (None, {"cost_incomplete": 0}),
        ("cost_incomplete", {}),
        (None, {"cleanup_status": "failed"}),
        ("cleanup_status", {}),
        (None, {"status": "unresolved"}),
    ],
)
def test_retryable_task_projection_requires_exact_preoracle_settlement(
    drop_field,
    override,
):
    summary = {
        "status": "candidate_failed",
        "architect_terminal_class": "retryable_failure",
        "hidden_invocation_measurement_ref": None,
        "cost_incomplete": False,
        "cleanup_status": "complete",
        **override,
    }
    if drop_field is not None:
        summary.pop(drop_field)

    assert wf._all_task_failures_are_retryable([summary]) is False


def test_retryable_task_projection_requires_every_failed_task_to_match():
    retryable = {
        "status": "candidate_failed",
        "architect_terminal_class": "retryable_failure",
        "hidden_invocation_measurement_ref": None,
        "cost_incomplete": False,
        "cleanup_status": "complete",
    }
    resolved = {"status": "resolved"}

    assert (
        wf._all_task_failures_are_retryable([resolved, retryable, dict(retryable)])
        is True
    )
    assert (
        wf._all_task_failures_are_retryable(
            [
                retryable,
                {**retryable, "architect_terminal_class": "permanent_failure"},
            ]
        )
        is False
    )


def _unjudged_task_summary(inst, **override) -> dict:
    """A settled pre-oracle task failure whose Architect reached no ruling."""

    return {
        "task_id": inst.task_id,
        "ext_id": inst.ext_id,
        "status": "candidate_failed",
        "architect_operation_id": "architect-test",
        "architect_detail_status": "inconclusive_tier_failure",
        "architect_terminal_class": "inconclusive",
        "rc": 1,
        "reward": None,
        "resolved": None,
        "hidden_invocation_measurement_ref": None,
        "cost_usd": 0.0,
        "cost_incomplete": False,
        "cleanup_status": "complete",
        **override,
    }


def _setup_stop_task_summary(inst, **override) -> dict:
    """A settled stop before Architect was started."""

    return {
        "task_id": inst.task_id,
        "ext_id": inst.ext_id,
        "status": "setup_failed",
        "failure_phase": "dev_verify_preflight",
        "failure_kind": "dev_verify_unavailable",
        "architect_detail_status": None,
        "architect_terminal_class": None,
        "rc": 2,
        "reward": None,
        "resolved": None,
        "hidden_invocation_measurement_ref": None,
        "cost_usd": 0.0,
        "cost_incomplete": False,
        "cleanup_status": "complete",
        **override,
    }


def _run_status_for_task_summaries(tmp_path: Path, monkeypatch, build_rows):
    """Run the serial implement loop over stubbed task results.

    ``build_rows`` receives the selected instances and returns one task
    summary per instance. Returns ``(rc, result.json, outcome.json)``.
    """

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    selected = wf._load_and_filter(ns)
    rows = build_rows(selected)
    selected = selected[: len(rows)]
    by_task = {
        inst.task_id: row for inst, row in zip(selected, rows, strict=True)
    }
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda inst, **kwargs: by_task[inst.task_id],
    )

    rc = wf._implement_serial(selected, args=ns)

    run_dir = next((tmp_path / "s" / "runs").iterdir())
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    return rc, result, outcome


@pytest.mark.parametrize(
    "build_rows",
    [
        pytest.param(
            lambda insts: [_unjudged_task_summary(insts[0])],
            id="inconclusive-architect",
        ),
        pytest.param(
            lambda insts: [_unjudged_task_summary(
                insts[0],
                architect_detail_status="cancelled",
                architect_terminal_class="cancelled",
            )],
            id="cancelled-architect",
        ),
        pytest.param(
            lambda insts: [_unjudged_task_summary(
                insts[0],
                architect_detail_status="planner_infrastructure_failure",
                rc=2,
            )],
            id="planner-backend-lost",
        ),
        pytest.param(
            lambda insts: [
                _unjudged_task_summary(
                    insts[0],
                    architect_detail_status="retryable_tier_failure",
                    architect_terminal_class="retryable_failure",
                ),
                _unjudged_task_summary(insts[1]),
            ],
            id="one-retryable-one-inconclusive",
        ),
        pytest.param(
            lambda insts: [
                _lifecycle_summary(insts[0].task_id, insts[0].ext_id),
                _unjudged_task_summary(insts[1]),
            ],
            id="one-resolved-one-inconclusive",
        ),
    ],
)
def test_implement_reports_unjudged_failures_as_inconclusive_at_run_boundary(
    tmp_path: Path,
    monkeypatch,
    build_rows,
):
    """A run whose failed tasks were never judged is not ``tasks_failed``."""

    rc, result, outcome = _run_status_for_task_summaries(
        tmp_path, monkeypatch, build_rows,
    )

    assert rc == 1
    assert result["status"] == "inconclusive_tasks_failed"
    assert outcome["detail_status"] == "inconclusive_tasks_failed"
    assert outcome["terminal_class"] == "inconclusive"


@pytest.mark.parametrize(
    "build_rows",
    [
        pytest.param(
            lambda insts: [
                _unjudged_task_summary(insts[0]),
                {
                    **_lifecycle_summary(
                        insts[1].task_id, insts[1].ext_id, status="unresolved",
                    ),
                    "hidden_invocation_measurement_ref": "sha256:" + "a" * 64,
                },
            ],
            id="one-unresolved",
        ),
        pytest.param(
            lambda insts: [
                _unjudged_task_summary(insts[0]),
                _unjudged_task_summary(
                    insts[1],
                    architect_detail_status="partial_failed",
                    architect_terminal_class="permanent_failure",
                ),
            ],
            id="one-judged-architect-failure",
        ),
        pytest.param(
            lambda insts: [
                _setup_stop_task_summary(insts[0]),
                _unjudged_task_summary(
                    insts[1],
                    architect_detail_status="inflation_veto",
                    architect_terminal_class="permanent_failure",
                ),
            ],
            id="setup-stop-and-judged-architect-failure",
        ),
    ],
)
def test_one_judged_failure_keeps_tasks_failed(
    tmp_path: Path,
    monkeypatch,
    build_rows,
):
    rc, result, outcome = _run_status_for_task_summaries(
        tmp_path, monkeypatch, build_rows,
    )

    assert rc == 1
    assert result["status"] == "tasks_failed"
    assert outcome["terminal_class"] == "permanent_failure"


def test_all_retryable_failures_keep_retryable_tasks_failed(
    tmp_path: Path,
    monkeypatch,
):
    """The retryable projection is tested first and is not widened."""

    rc, result, outcome = _run_status_for_task_summaries(
        tmp_path,
        monkeypatch,
        lambda insts: [_unjudged_task_summary(
            insts[0],
            architect_detail_status="retryable_tier_failure",
            architect_terminal_class="retryable_failure",
        )],
    )

    assert rc == 1
    assert result["status"] == "retryable_tasks_failed"
    assert outcome["terminal_class"] == "retryable_failure"


@pytest.mark.parametrize(
    ("drop_field", "override"),
    [
        # The settlement terms of the retryable projection.
        (None, {"hidden_invocation_measurement_ref": "sha256:" + "a" * 64}),
        ("hidden_invocation_measurement_ref", {}),
        (None, {"cost_incomplete": True}),
        (None, {"cost_incomplete": 0}),
        ("cost_incomplete", {}),
        (None, {"cleanup_status": "failed"}),
        ("cleanup_status", {}),
        (None, {"status": "unresolved"}),
        (None, {"status": "verification_incomplete"}),
        # A class that is a ruling, a budget stop or a contract error.
        (None, {"architect_terminal_class": "permanent_failure"}),
        (None, {"architect_terminal_class": "contract_error"}),
        (None, {"architect_terminal_class": "succeeded"}),
        (None, {"architect_terminal_class": "budget_exhausted"}),
        (None, {"architect_terminal_class": "interrupted"}),
        # A setup stop raised after Architect returned: Architect ran, and
        # its class decides.
        (None, {"status": "setup_failed", "architect_terminal_class": "succeeded"}),
        (
            None,
            {
                "status": "toolchain_unavailable",
                "architect_terminal_class": "permanent_failure",
            },
        ),
        # Without a typed class only a setup stop before Architect counts,
        # and only when Architect was never started.
        (None, {"architect_terminal_class": None}),
        ("architect_terminal_class", {}),
        (
            None,
            {"status": "cascade_error", "architect_terminal_class": None},
        ),
        (
            None,
            {"status": "storage_quota_exceeded", "architect_terminal_class": None},
        ),
        (None, {"status": "setup_failed", "architect_terminal_class": None}),
        # Architect never started, but the stop is not a setup stop.
        (
            None,
            {
                "status": "cascade_error",
                "architect_terminal_class": None,
                "architect_operation_id": None,
            },
        ),
        (
            "architect_operation_id",
            {"status": "storage_quota_exceeded", "architect_terminal_class": None},
        ),
        ("architect_operation_id", {"architect_terminal_class": None}),
    ],
)
def test_unjudged_task_projection_requires_exact_preoracle_settlement(
    drop_field,
    override,
):
    summary = {
        "status": "candidate_failed",
        "architect_operation_id": "architect-test",
        "architect_terminal_class": "inconclusive",
        "hidden_invocation_measurement_ref": None,
        "cost_incomplete": False,
        "cleanup_status": "complete",
        **override,
    }
    if drop_field is not None:
        summary.pop(drop_field)

    assert wf._all_task_failures_are_unjudged([summary]) is False


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"architect_terminal_class": "cancelled"},
        {"architect_terminal_class": "retryable_failure"},
        {"status": "cascade_error"},
        {
            "status": "setup_failed",
            "architect_terminal_class": None,
            "architect_operation_id": None,
        },
        {
            "status": "toolchain_unavailable",
            "architect_terminal_class": None,
            "architect_operation_id": None,
        },
    ],
)
def test_unjudged_task_projection_accepts_only_unjudged_failures(override):
    summary = {
        "status": "candidate_failed",
        "architect_operation_id": "architect-test",
        "architect_terminal_class": "inconclusive",
        "hidden_invocation_measurement_ref": None,
        "cost_incomplete": False,
        "cleanup_status": "complete",
        **override,
    }
    resolved = {"status": "resolved"}

    assert wf._all_task_failures_are_unjudged([summary]) is True
    assert wf._all_task_failures_are_unjudged([resolved, summary]) is True
    assert wf._all_task_failures_are_unjudged([resolved]) is False
    assert wf._all_task_failures_are_unjudged([]) is False
    assert wf._all_task_failures_are_unjudged(
        [summary, {**summary, "status": "unresolved"}]
    ) is False


@pytest.mark.parametrize(
    "override",
    [
        pytest.param(
            {"failure_kind": "dev_verify_unavailable"},
            id="preflight-gate-could-not-run",
        ),
        pytest.param(
            {"failure_kind": "dev_verify_failed", "rc": 1},
            id="preflight-baseline-red",
        ),
        pytest.param(
            {
                "failure_phase": "container_setup",
                "failure_kind": "InstanceContainerSetupError",
            },
            id="container-setup",
        ),
        pytest.param(
            {
                "status": "toolchain_unavailable",
                "failure_phase": "container_setup",
                "failure_kind": "toolchain_unavailable",
            },
            id="toolchain-unavailable",
        ),
    ],
)
def test_setup_stop_only_run_ends_inconclusive_tasks_failed(
    tmp_path: Path,
    monkeypatch,
    override,
):
    """No model was called and no work exists: nothing was judged.

    The rows of such a run re-run by default, so the run is not filed under
    the permanent ``tasks_failed``.
    """

    rc, result, outcome = _run_status_for_task_summaries(
        tmp_path,
        monkeypatch,
        lambda insts: [_setup_stop_task_summary(insts[0], **override)],
    )

    assert rc == 1
    assert result["status"] == "inconclusive_tasks_failed"
    assert outcome["detail_status"] == "inconclusive_tasks_failed"
    assert outcome["terminal_class"] == "inconclusive"


@pytest.mark.parametrize(
    "override",
    [
        pytest.param(
            {
                "status": "toolchain_unavailable",
                "failure_phase": "container_setup",
                "failure_kind": "toolchain_unavailable",
                "architect_operation_id": "architect-test",
                "architect_detail_status": "partial_failed",
                "architect_terminal_class": "permanent_failure",
            },
            id="after-a-judged-architect-failure",
        ),
        pytest.param(
            {
                "failure_phase": "container_setup",
                "failure_kind": "InstanceContainerSetupError",
                "architect_operation_id": "architect-test",
                "architect_detail_status": "complete",
                "architect_terminal_class": "succeeded",
            },
            id="after-a-completed-architect",
        ),
        pytest.param(
            {
                "failure_phase": "container_setup",
                "failure_kind": "InstanceContainerSetupError",
                "architect_operation_id": "architect-test",
            },
            id="architect-started-without-a-typed-outcome",
        ),
        pytest.param({"cost_incomplete": True}, id="cost-incomplete"),
        pytest.param({"cleanup_status": "failed"}, id="cleanup-failed"),
    ],
)
def test_setup_stop_after_architect_started_keeps_its_run_status(
    tmp_path: Path,
    monkeypatch,
    override,
):
    """The setup arm needs positive proof that Architect never started."""

    rc, result, _outcome_json = _run_status_for_task_summaries(
        tmp_path,
        monkeypatch,
        lambda insts: [_setup_stop_task_summary(insts[0], **override)],
    )

    assert rc == 1
    assert result["status"] == (
        "cost_incomplete" if override.get("cost_incomplete") else "tasks_failed"
    )


def test_implement_composes_remaining_budget_and_fails_last_task_overshoot(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement",
            "--budget-usd",
            "10",
            "--budget-usd-per-instance",
            "8",
        ]
    )
    child_caps: list[float | None] = []

    def _task(inst, *, cost, child_budget_usd, **kwargs):
        child_caps.append(child_budget_usd)
        cost.record(
            role=inst.task_id,
            usage=wf._CostShim(6.0),
            backend_name="stub",
            model="stub",
        )
        return {
            "task_id": inst.task_id,
            "ext_id": inst.ext_id,
            "status": "resolved",
            "cost_usd": 6.0,
            "cost_incomplete": False,
        }

    monkeypatch.setattr(wf, "_run_task_cascade", _task)
    assert wf._implement_serial(wf._load_and_filter(ns), args=ns) == 1

    assert child_caps == [8.0, 4.0]
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["cost_total_usd"] == pytest.approx(12.0)
    assert summary["halt_reason"] == "aggregate_budget_exceeded"
    assert result["status"] == "budget_exhausted"
    assert summary["tasks_selected"] == 4
    assert summary["tasks_total"] == 2
    assert summary["tasks_unrecorded"] == 0
    assert summary["tasks_not_started"] == 2


def test_implement_reports_tasks_not_started_after_budget_halt(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement",
            "--budget-usd",
            "10",
            "--budget-usd-per-instance",
            "10",
        ]
    )
    executed: list[str] = []

    def _task(inst, *, cost, **kwargs):
        executed.append(inst.task_id)
        cost.record(
            role=inst.task_id,
            usage=wf._CostShim(10.0),
            backend_name="stub",
            model="stub",
        )
        return {
            "task_id": inst.task_id,
            "ext_id": inst.ext_id,
            "status": "resolved",
            "cost_usd": 10.0,
            "cost_incomplete": False,
        }

    monkeypatch.setattr(wf, "_run_task_cascade", _task)
    filtered = wf._load_and_filter(ns)

    assert wf._implement_serial(filtered, args=ns) == 1

    assert executed == [filtered[0].task_id]
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["halt_reason"] == "aggregate_budget_reached"
    assert result["status"] == "budget_exhausted"
    assert summary["tasks_selected"] == 4
    assert summary["tasks_total"] == 1
    assert summary["tasks_resolved"] == 1
    assert summary["tasks_failed"] == 0
    assert summary["tasks_unrecorded"] == 0
    # The three tasks the halt never reached are not failures and not zeros.
    assert summary["tasks_not_started"] == 3
    assert summary["tasks_ungraded"] == 0


def test_implement_counts_a_task_without_summary_apart_from_not_started(
    tmp_path: Path,
    monkeypatch,
):
    """A task that ran but could not be checkpointed is not 'not started'."""

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--lifecycle-out",
            str(tmp_path / "lifecycle.jsonl"),
        ]
    )
    filtered = wf._load_and_filter(ns)
    executed: list[str] = []

    def _task(inst, **kwargs):
        executed.append(inst.task_id)
        return _lifecycle_summary(inst.task_id, inst.ext_id)

    def _append(*, out_path, summary):
        raise OSError("lifecycle volume is read-only")

    monkeypatch.setattr(wf, "_run_task_cascade", _task)
    monkeypatch.setattr(wf, "_append_lifecycle_record", _append)

    assert wf._implement_serial(filtered, args=ns) == 1

    # The cascade of the first task ran to a resolved grade.
    assert executed == [filtered[0].task_id]
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "lifecycle_output_failed"
    assert summary["halt_reason"] == "lifecycle_output_failed"
    assert summary["tasks"] == []
    assert summary["tasks_selected"] == 4
    assert summary["tasks_total"] == 0
    assert summary["tasks_unrecorded"] == 1
    assert summary["tasks_not_started"] == 3


def test_implement_setup_stop_is_counted_ungraded_and_rerun_by_default(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
        ]
    )
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda inst, **kwargs: {
            "task_id": inst.task_id,
            "ext_id": inst.ext_id,
            "status": "setup_failed",
            "failure_phase": "dev_verify_preflight",
            "failure_kind": "dev_verify_failed",
            "rc": 1,
            "reward": None,
            "resolved": None,
            "hidden_invocation_measurement_ref": None,
            "cost_usd": 0.0,
            "cost_incomplete": False,
            "cleanup_status": "complete",
        },
    )

    # Exit 1 as for any non-resolved task; there is no separate exit code for
    # a task that was never graded.
    assert wf._implement_serial(wf._load_and_filter(ns), args=ns) == 1

    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["tasks_selected"] == 1
    assert summary["tasks_total"] == 1
    assert summary["tasks_failed"] == 1
    assert summary["tasks_graded"] == 0
    assert summary["tasks_unresolved"] == 0
    assert summary["tasks_ungraded"] == 1
    assert summary["tasks_rerun_by_default"] == 1
    assert summary["tasks_not_started"] == 0
    assert summary["tasks_unrecorded"] == 0
    assert summary["tasks_by_status"] == {"setup_failed": 1}
    assert summary["tasks"][0]["reward"] is None


def _lifecycle_summary(
    task_id: str,
    ext_id: str,
    *,
    status: str = "resolved",
) -> dict:
    return {
        "task_id": task_id,
        "ext_id": ext_id,
        "repo": "ex/repo",
        "language": "python",
        "category": "feature_request",
        "status": status,
        "rc": 0,
        "reward": 1.0 if status == "resolved" else 0.0,
        "resolved": 1.0 if status == "resolved" else 0.0,
        "cleanup_status": "complete",
        "cleanup_errors": [],
        "wall_seconds": 1.0,
        "cost_usd": 0.0,
        "cost_incomplete": False,
        "checkpoint_fingerprint": "sha256:" + "1" * 64,
        "image_digest": "test/image@sha256:" + "a" * 64,
    }


def _permanent_planning_summary(task_id: str, ext_id: str) -> dict:
    """A deterministic zero-tier Architect veto with every proof complete."""

    summary = _lifecycle_summary(task_id, ext_id, status="candidate_failed")
    summary.update(
        {
            "rc": 2,
            "reward": None,
            "resolved": None,
            "authoring_status": "failed",
            "candidate_capture_status": "failed",
            "candidate": None,
            "pristine_replay": None,
            "verifier_status": "not_run",
            "verifier_patch": None,
            "tests_snapshot": None,
            "architect_detail_status": "inflation_veto",
            "architect_terminal_class": "permanent_failure",
            "architect_failure_phase": "architect_planning",
            "architect_failure_kind": "inflation_veto",
            "architect_tiers_executed": 0,
            "zero_tier_settlement_proven": True,
            "secondary_failures": [],
            "failure_phase": "architect_planning",
            "failure_kind": "inflation_veto",
        }
    )
    return summary


@pytest.mark.parametrize(
    ("summary", "policy"),
    (
        (_lifecycle_summary("task", "ext", status="resolved"), "reusable"),
        (_lifecycle_summary("task", "ext", status="unresolved"), "reusable"),
        (
            _lifecycle_summary("task", "ext", status="verification_failed"),
            "reusable",
        ),
        (
            _lifecycle_summary("task", "ext", status="verification_incomplete"),
            "reusable",
        ),
        (_lifecycle_summary("task", "ext", status="setup_failed"), "retryable"),
        (_lifecycle_summary("task", "ext", status="cascade_error"), "retryable"),
        (
            _lifecycle_summary("task", "ext", status="candidate_failed"),
            "retryable",
        ),
        (_permanent_planning_summary("task", "ext"), "permanent"),
        (
            {
                **_lifecycle_summary("task", "ext", status="resolved"),
                "diagnostic_cost_policy": "any-declared-policy",
            },
            "diagnostic_only",
        ),
        (
            {
                **_permanent_planning_summary("task", "ext"),
                "diagnostic_cost_policy": "any-declared-policy",
            },
            "diagnostic_only",
        ),
    ),
)
def test_task_status_policy_is_literal_for_each_lifecycle_class(
    summary: dict,
    policy: str,
):
    assert wf._task_status_policy(summary) == policy
    # The durable row and the run roll-up classify through the same rule.
    assert wf._lifecycle_record(summary)["status_policy"] == policy


def test_lifecycle_checkpoint_restores_physical_pricing_ledger():
    from crucible.common.cost import CostAccumulator

    summary = _lifecycle_summary("task", "ext")
    summary["cost_usd"] = 0.3
    summary["architect_costs"] = _architect_pricing_costs()

    record = wf._lifecycle_record(summary)
    assert record["architect_costs"] == summary["architect_costs"]
    validated = wf._validate_lifecycle_record(record, line_number=1)
    accumulator = CostAccumulator()
    wf._restore_checkpoint_cost(accumulator, validated, task_id="task")

    restored = accumulator.to_dict()
    assert restored["total_usd"] == pytest.approx(0.3)
    assert restored["total_cache_creation_tokens"] == 4
    assert restored["total_cache_read_tokens"] == 7
    assert [call["role"] for call in restored["pricing_evidence_by_call"]] == [
        "planner",
        "judge",
    ]
    legacy = wf._lifecycle_record(_lifecycle_summary("old", "old-ext"))
    assert "architect_costs" not in legacy


def test_checkpoint_fingerprint_binds_effective_model_pricing(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    pricing = wf.load_pricing(spec_dir=ns.spec)
    model = wf.load_routing(spec_dir=ns.spec).default.model
    original = pricing[model]
    pricing[model] = wf.dataclasses.replace(
        original,
        output_per_mtok=original.output_per_mtok + 0.01,
    )
    monkeypatch.setattr(wf, "load_pricing", lambda **kwargs: pricing)

    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_checkpoint_fingerprint_binds_exact_backend_routing_bytes(tmp_path: Path):
    manifest = _build_corpus(tmp_path)
    routing = tmp_path / "routing.yaml"
    payload = (
        "schema_version: crucible.backend-routing.v1\n"
        "default: {backend: codex, model: gpt-6-astra, effort: max}\n"
    )
    routing.write_text(payload, encoding="utf-8")
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--backend-config",
            str(routing),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]

    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    routing.write_text("# changed exact source bytes\n" + payload, encoding="utf-8")
    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_checkpoint_fingerprint_binds_all_backend_runtime_recipes(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    crucible_root = Path(wf.__file__).resolve().parents[1]
    observed: set[str] = set()
    revisions: dict[str, str] = {}

    def _digest(path):
        candidate = Path(path).resolve()
        relative = (
            candidate.relative_to(crucible_root).as_posix()
            if candidate.is_relative_to(crucible_root)
            else candidate.as_posix()
        )
        observed.add(relative)
        return {
            "name": candidate.name,
            "sha256": revisions.get(relative, "0" * 64),
            "size": 1,
        }

    monkeypatch.setattr(wf, "file_digest", _digest)
    monkeypatch.setattr(
        wf,
        "file_set_fingerprint",
        lambda *_args, **_kwargs: "sha256:test-source-set",
    )
    monkeypatch.setattr(wf, "read_assay_version", lambda: "test-assay")
    monkeypatch.setattr(
        wf,
        "image_verification_policy_identity",
        lambda: {"schema": "test-image-policy.v1"},
    )

    def _fingerprint() -> str:
        return wf._deepswe_checkpoint_fingerprint(
            inst,
            args=ns,
            child_budget_usd=3.0,
        )

    baseline = _fingerprint()
    expected_backend_modules = {
        f"backends/{name}"
        for name in (
            "__init__.py",
            "_classify.py",
            "_claude_session_transport.py",
            "_physical_telemetry.py",
            "_sdk_debug.py",
            "capabilities.py",
            "claude.py",
            "codex.py",
            "codex_official.py",
            "codex_official_worker.py",
            "config.py",
            "deepseek.py",
            "evidence.py",
            "features.py",
            "kimi.py",
            "limits.py",
            "openai_compat.py",
            "pricing.py",
            "roles.py",
            "tool_authority.py",
            "trusted_codex.py",
        )
    }
    expected_worker_recipes = {
        f"backends/{project}/{relative}"
        for project, package in (
            ("codex_worker", "crucible_codex_worker"),
            ("deepseek_worker", "crucible_deepseek_worker"),
            ("kimi_worker", "crucible_kimi_worker"),
        )
        for relative in (
            ".python-version",
            "pyproject.toml",
            "uv.lock",
            f"src/{package}/__init__.py",
            f"src/{package}/__main__.py",
        )
    }
    expected_shared_runtime = {
        "common/authority_io.py",
        "common/process_settlement.py",
        "tools/__init__.py",
        "tools/_dispatch_budget.py",
        "tools/_isolated_mutation_host.py",
        "tools/_mcp_bridge.py",
        "tools/_mcp_http_server.py",
        "workflows/_backends.py",
        "workflows/_backend_recipe.py",
        "workflows/_process_supervisor_contract.py",
        "workflows/_process_supervisor.py",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    }
    expected_planning_runtime = {
        "workflows/_benchmark_planning.py",
        "workflows/_plan_competition.py",
        "workflows/_plan_selection.py",
        "workflows/_readonly_scheduler.py",
    }
    assert (
        expected_backend_modules | expected_worker_recipes | expected_shared_runtime
        | expected_planning_runtime
    ) <= observed

    for relative in (
        "common/authority_io.py",
        "common/process_settlement.py",
        "backends/claude.py",
        "backends/codex.py",
        "backends/codex_official.py",
        "backends/deepseek.py",
        "backends/kimi.py",
        "backends/openai_compat.py",
        "backends/features.py",
        "backends/codex_worker/uv.lock",
        "backends/deepseek_worker/src/crucible_deepseek_worker/__main__.py",
        "backends/kimi_worker/pyproject.toml",
        "tools/__init__.py",
        "tools/_isolated_mutation_host.py",
        *sorted(expected_planning_runtime),
        "workflows/_backend_recipe.py",
        "workflows/_process_supervisor_contract.py",
        "workflows/_process_supervisor.py",
        ".python-version",
        "uv.lock",
    ):
        revisions[relative] = "1" * 64
        assert _fingerprint() != baseline, relative
        revisions.clear()


def test_checkpoint_fingerprint_binds_planner_and_development_budgets(
    tmp_path: Path,
):
    manifest = _build_corpus(tmp_path)
    baseline_args = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    changed_tool_args = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--planner-tool-call-limit",
            "31",
        ]
    )
    inst = wf._load_and_filter(baseline_args)[0]

    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=baseline_args,
        child_budget_usd=3.0,
    )
    changed_tool_budget = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=changed_tool_args,
        child_budget_usd=3.0,
    )
    changed_development_budget = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=baseline_args,
        child_budget_usd=2.0,
    )
    competing_args = wf._parse_args([
        "--spec", str(tmp_path / "s"), "--dataset", str(manifest),
        "--implement", "--budget-usd-per-instance", "3",
    ])
    changed_planner_count = wf._deepswe_checkpoint_fingerprint(
        inst, args=competing_args, child_budget_usd=3.0,
    )

    assert changed_tool_budget != baseline
    assert changed_development_budget != baseline
    assert changed_planner_count != baseline


def test_checkpoint_fingerprint_binds_complete_hidden_tests_tree(
    tmp_path: Path,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    hidden_file = manifest.parent / inst.task_id / "tests" / "nested" / "case.py"
    hidden_file.parent.mkdir()
    hidden_file.write_text("assert True\n", encoding="utf-8")

    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_checkpoint_fingerprint_binds_tier_and_assay_build_identity(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_tier_digest(path):
        digest = original_file_digest(path)
        if Path(path).name == "tier.py":
            return {**digest, "sha256": "f" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_tier_digest)
    workflow_changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    monkeypatch.setattr(wf, "file_digest", original_file_digest)
    monkeypatch.setattr(wf, "read_assay_version", lambda: "changed-assay")
    assay_changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert workflow_changed != baseline
    assert assay_changed != baseline


@pytest.mark.parametrize("policy_file", ["git_safety.py", "rust_toolchain.py"])
def test_checkpoint_fingerprint_binds_shared_container_runtime_policy(
    tmp_path: Path,
    monkeypatch,
    policy_file: str,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_git_policy_digest(path):
        digest = original_file_digest(path)
        if Path(path).name == policy_file:
            return {**digest, "sha256": "e" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_git_policy_digest)

    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


@pytest.mark.parametrize("authority_file", ["secure_artifacts.py", "dataset.py"])
def test_checkpoint_fingerprint_binds_reward_grade_authority(
    tmp_path: Path,
    monkeypatch,
    authority_file: str,
):
    """The reward reader and the whitelist rule decide what is a grade."""

    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest
    product_root = Path(wf.__file__).resolve().parents[1] / "products" / "deepswe"

    def _changed_authority_digest(path):
        digest = original_file_digest(path)
        if Path(path) == product_root / authority_file:
            return {**digest, "sha256": "e" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_authority_digest)

    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_checkpoint_fingerprint_binds_product_mutation_settlement_adapter(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_settlement_digest(path):
        digest = original_file_digest(path)
        if Path(path).name == "mutation_settlement.py":
            return {**digest, "sha256": "c" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_settlement_digest)
    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


@pytest.mark.parametrize(
    "protocol_file",
    ["pytest_observer.py", "observations.py"],
)
def test_checkpoint_fingerprint_binds_pytest_observation_protocol(
    tmp_path: Path,
    monkeypatch,
    protocol_file: str,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_observer_digest(path):
        digest = original_file_digest(path)
        if Path(path).as_posix().endswith(f"common/runners/{protocol_file}"):
            return {**digest, "sha256": "b" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_observer_digest)
    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


@pytest.mark.parametrize(
    "protocol_file",
    ["pytest_policy.py", "lang/python.py"],
)
def test_checkpoint_fingerprint_binds_python_dev_verify_protocol(
    tmp_path: Path,
    monkeypatch,
    protocol_file: str,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_dev_verify_digest(path):
        digest = original_file_digest(path)
        if Path(path).as_posix().endswith(f"common/runners/{protocol_file}"):
            return {**digest, "sha256": "a" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_dev_verify_digest)
    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_checkpoint_fingerprint_binds_development_verifier_preflight(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_preflight_digest(path):
        digest = original_file_digest(path)
        if Path(path).name == "_deepswe_baseline_preflight.py":
            return {**digest, "sha256": "b" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_preflight_digest)
    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_checkpoint_fingerprint_binds_planner_source_tool_recipe(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    baseline = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )
    original_file_digest = wf.file_digest

    def _changed_source_tool_digest(path):
        digest = original_file_digest(path)
        if Path(path).as_posix().endswith("tools/source/handlers.py"):
            return {**digest, "sha256": "d" * 64}
        return digest

    monkeypatch.setattr(wf, "file_digest", _changed_source_tool_digest)

    changed = wf._deepswe_checkpoint_fingerprint(
        inst,
        args=ns,
        child_budget_usd=3.0,
    )

    assert changed != baseline


def test_lifecycle_v2_is_strictly_normalized_with_unknown_new_fields():
    current = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    legacy = {
        key: value
        for key, value in current.items()
        if key
        not in {
            "assigned_budget_usd",
            "authoring_status",
            "candidate_capture_status",
            "candidate",
            "pristine_replay",
            "verifier_status",
            "verifier_patch",
            "failure_phase",
            "failure_kind",
            "architect_detail_status",
            "architect_decision",
            "architect_terminal_class",
            "architect_failure_phase",
            "architect_failure_kind",
            "architect_tiers_executed",
            "zero_tier_settlement_proven",
            "secondary_failures",
            "verification_rc",
            "verify_returncode",
            "command_returncode",
            "baseline_test_returncode",
            "new_test_returncode",
            "diagnostics_persisted",
            "verifier_stdout_bytes",
            "verifier_stderr_bytes",
            "verifier_stdout_truncated",
            "verifier_stderr_truncated",
            "reward_counts",
        }
    }
    legacy["schema_version"] = "crucible.deepswe.lifecycle.v2"

    normalized = wf._validate_lifecycle_record(legacy, line_number=1)

    assert normalized["schema_version"] == "crucible.deepswe.lifecycle.v5"
    assert normalized["assigned_budget_usd"] is None
    assert normalized["candidate"] is None
    assert normalized["verifier_status"] is None
    assert normalized["verification_rc"] is None
    assert normalized["verify_returncode"] is None
    assert normalized["command_returncode"] is None
    assert normalized["baseline_test_returncode"] is None
    assert normalized["new_test_returncode"] is None
    assert normalized["diagnostics_persisted"] is None
    assert normalized["reward_counts"] is None


def test_lifecycle_v3_is_normalized_without_inventing_architect_provenance():
    legacy = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    legacy["schema_version"] = "crucible.deepswe.lifecycle.v3"
    for field in (
        "architect_detail_status",
        "architect_decision",
        "architect_terminal_class",
        "architect_failure_phase",
        "architect_failure_kind",
        "architect_tiers_executed",
        "zero_tier_settlement_proven",
        "secondary_failures",
        "verification_rc",
        "verify_returncode",
        "command_returncode",
        "baseline_test_returncode",
        "new_test_returncode",
        "diagnostics_persisted",
        "verifier_stdout_bytes",
        "verifier_stderr_bytes",
        "verifier_stdout_truncated",
        "verifier_stderr_truncated",
        "reward_counts",
    ):
        legacy.pop(field)

    normalized = wf._validate_lifecycle_record(legacy, line_number=1)

    assert normalized["schema_version"] == "crucible.deepswe.lifecycle.v5"
    assert normalized["architect_detail_status"] is None
    assert normalized["architect_terminal_class"] is None
    assert normalized["zero_tier_settlement_proven"] is False
    assert normalized["secondary_failures"] == []
    assert normalized["status_policy"] == "reusable"
    assert normalized["verification_rc"] is None
    assert normalized["verify_returncode"] is None
    assert normalized["command_returncode"] is None
    assert normalized["baseline_test_returncode"] is None
    assert normalized["new_test_returncode"] is None
    assert normalized["diagnostics_persisted"] is None
    assert normalized["reward_counts"] is None


def test_lifecycle_v4_is_normalized_without_inventing_reward_counts():
    legacy = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    legacy["schema_version"] = "crucible.deepswe.lifecycle.v4"
    legacy.pop("reward_counts")

    normalized = wf._validate_lifecycle_record(legacy, line_number=1)

    assert normalized["schema_version"] == "crucible.deepswe.lifecycle.v5"
    assert normalized["reward_counts"] is None
    assert normalized["reward"] == 1.0
    assert normalized["status_policy"] == "reusable"


def test_lifecycle_rejects_unknown_schema_version():
    record = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    record["schema_version"] = "crucible.deepswe.lifecycle.v6"

    with pytest.raises(ValueError, match="unsupported schema"):
        wf._validate_lifecycle_record(record, line_number=1)


_LIFECYCLE_COUNTS = {
    "apply_failed": False,
    "f2p_passed": 3,
    "f2p_total": 3,
    "p2p_passed": 120,
    "p2p_total": 120,
}


def test_lifecycle_persists_reward_counts():
    summary = _lifecycle_summary("task", "ext")
    summary["reward_counts"] = dict(_LIFECYCLE_COUNTS)

    record = wf._lifecycle_record(summary)
    validated = wf._validate_lifecycle_record(record, line_number=1)

    assert validated["schema_version"] == "crucible.deepswe.lifecycle.v5"
    assert validated["reward_counts"] == _LIFECYCLE_COUNTS
    # The row survives its JSON round trip through the checkpoint file.
    assert (
        wf._validate_lifecycle_record(
            json.loads(json.dumps(validated)),
            line_number=1,
        )
        == validated
    )
    countless = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    assert countless["reward_counts"] is None
    assert (
        wf._validate_lifecycle_record(countless, line_number=1)["reward_counts"]
        is None
    )


@pytest.mark.parametrize(
    ("reward", "reward_counts"),
    (
        (1.0, {**_LIFECYCLE_COUNTS, "partial": 1.0}),
        (
            1.0,
            {
                key: value
                for key, value in _LIFECYCLE_COUNTS.items()
                if key != "apply_failed"
            },
        ),
        (1.0, {**_LIFECYCLE_COUNTS, "p2p_passed": -1}),
        (1.0, {**_LIFECYCLE_COUNTS, "f2p_total": True}),
        (1.0, {**_LIFECYCLE_COUNTS, "f2p_passed": 3.0}),
        (1.0, {**_LIFECYCLE_COUNTS, "f2p_passed": 4}),
        (1.0, {**_LIFECYCLE_COUNTS, "apply_failed": 0}),
        (1.0, [3, 3, 120, 120]),
        # counts without a reward
        (None, dict(_LIFECYCLE_COUNTS)),
        # counts that earn the other reward
        (0.0, dict(_LIFECYCLE_COUNTS)),
        (1.0, {**_LIFECYCLE_COUNTS, "f2p_passed": 2}),
        (True, dict(_LIFECYCLE_COUNTS)),
    ),
    ids=(
        "unknown_key",
        "missing_key",
        "negative",
        "boolean_count",
        "float_count",
        "passed_above_total",
        "apply_failed_not_boolean",
        "not_a_mapping",
        "no_reward",
        "reward_zero_all_passing",
        "reward_one_with_failure",
        "boolean_reward",
    ),
)
def test_lifecycle_rejects_malformed_or_ungraded_reward_counts(
    reward,
    reward_counts,
):
    record = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    record["reward"] = reward
    record["resolved"] = reward
    record["reward_counts"] = reward_counts

    with pytest.raises(ValueError, match="reward_counts"):
        wf._validate_lifecycle_record(record, line_number=1)


def test_lifecycle_persists_original_assigned_budget():
    summary = _lifecycle_summary("task", "ext")
    summary["budget_usd"] = 3.5
    record = wf._lifecycle_record(summary)
    validated = wf._validate_lifecycle_record(record, line_number=1)
    assert validated["assigned_budget_usd"] == 3.5


def test_lifecycle_persists_verifier_observability():
    summary = _lifecycle_summary("task", "ext")
    summary.update(
        {
            "verification_rc": 0,
            "verify_returncode": 0.0,
            "command_returncode": 0.0,
            "baseline_test_returncode": 0.0,
            "new_test_returncode": 70.0,
            "diagnostics_persisted": True,
            "verifier_stdout_bytes": 131201.0,
            "verifier_stderr_bytes": 2048.0,
            "verifier_stdout_truncated": True,
            "verifier_stderr_truncated": False,
        }
    )

    validated = wf._validate_lifecycle_record(
        wf._lifecycle_record(summary),
        line_number=1,
    )

    assert validated["verification_rc"] == 0
    assert validated["verify_returncode"] == 0.0
    assert validated["command_returncode"] == 0.0
    assert validated["baseline_test_returncode"] == 0.0
    assert validated["new_test_returncode"] == 70.0
    assert validated["diagnostics_persisted"] is True
    assert validated["verifier_stdout_bytes"] == 131201.0
    assert validated["verifier_stderr_bytes"] == 2048.0
    assert validated["verifier_stdout_truncated"] is True
    assert validated["verifier_stderr_truncated"] is False


def test_lifecycle_checkpoint_deduplicates_and_recovers_torn_tail(
    tmp_path: Path,
):
    out = tmp_path / "state" / "lifecycle.jsonl"
    first = _lifecycle_summary("task-1", "ext-1")

    wf._emit_lifecycle_jsonl(
        out_path=out,
        task_summaries=[first, first],
    )
    assert len(out.read_text().splitlines()) == 1

    # Simulate process death part-way through the next O_APPEND record. The
    # next locked append must discard only this unterminated tail.
    out.write_bytes(out.read_bytes() + b'{"schema_version":"crucible')
    wf._append_lifecycle_record(
        out_path=out,
        summary=_lifecycle_summary("task-2", "ext-2"),
    )

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [record["task_id"] for record in records] == ["task-1", "task-2"]
    assert all(
        record["schema_version"] == "crucible.deepswe.lifecycle.v5"
        for record in records
    )


def test_lifecycle_checkpoint_rejects_corrupt_completed_record(tmp_path: Path):
    out = tmp_path / "lifecycle.jsonl"
    out.write_bytes(b'{"not":"valid lifecycle"}\n{broken}\n')

    with pytest.raises(ValueError, match="line 1|line 2"):
        wf._load_lifecycle_jsonl(out)

    # Fail closed: a newline-terminated corrupt record is not truncated as a
    # recoverable tail and must remain available for diagnosis.
    assert out.read_bytes() == b'{"not":"valid lifecycle"}\n{broken}\n'


def test_invalid_lifecycle_terminal_records_run_duration(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    lifecycle.write_text("{broken}\n", encoding="utf-8")
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    monkeypatch.setattr(
        RunHandle,
        "elapsed_s",
        property(lambda _: 12.25),
    )

    assert wf._implement_serial(wf._load_and_filter(ns), args=ns) == 1

    run_dir = next((tmp_path / "s" / "runs").iterdir())
    result = json.loads((run_dir / "result.json").read_text())
    assert result["status"] == "lifecycle_output_invalid"
    assert result["duration_s"] == 12.25
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["tasks_selected"] == 4
    assert summary["tasks_total"] == 0
    assert summary["tasks_unrecorded"] == 0
    assert summary["tasks_not_started"] == 4


def test_implement_resumes_checkpoint_without_duplicate_task_work(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    filtered = wf._load_and_filter(ns)
    first = filtered[0]
    prior_summary = _lifecycle_summary(first.task_id, first.ext_id)
    prior_summary["checkpoint_fingerprint"] = wf._deepswe_checkpoint_fingerprint(
        wf.dataclasses.replace(
            first,
            docker_image=prior_summary["image_digest"],
        ),
        args=ns,
        child_budget_usd=None,
    )
    wf._append_lifecycle_record(
        out_path=lifecycle,
        summary=prior_summary,
    )

    executed: list[str] = []

    def _task(inst, **kwargs):
        executed.append(inst.task_id)
        return _lifecycle_summary(inst.task_id, inst.ext_id)

    monkeypatch.setattr(wf, "_run_task_cascade", _task)
    assert wf._implement_serial(filtered, args=ns) == 0
    assert executed == [inst.task_id for inst in filtered[1:]]

    records = [json.loads(line) for line in lifecycle.read_text().splitlines()]
    assert [record["task_id"] for record in records] == [
        inst.task_id for inst in filtered
    ]
    assert len({record["task_id"] for record in records}) == len(records)

    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["tasks_total"] == len(filtered)
    assert summary["tasks"][0]["resumed"] is True


def test_resume_restores_durable_cost_into_sweep_summary(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    prior = _lifecycle_summary(inst.task_id, inst.ext_id)
    prior["cost_usd"] = 2.5
    prior["checkpoint_fingerprint"] = wf._deepswe_checkpoint_fingerprint(
        wf.dataclasses.replace(inst, docker_image=prior["image_digest"]),
        args=ns,
        child_budget_usd=None,
    )
    wf._append_lifecycle_record(out_path=lifecycle, summary=prior)
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *args, **kwargs: pytest.fail("reusable task must not execute"),
    )

    assert wf._implement_serial((inst,), args=ns) == 0
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["cost_total_usd"] == pytest.approx(2.5)
    assert summary["tasks"][0]["cost_usd"] == pytest.approx(2.5)
    assert summary["tasks"][0]["resumed"] is True


def test_completed_single_track_checkpoint_cannot_be_relabelled_as_competition(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    argv = [
        "--spec", str(tmp_path / "s"), "--dataset", str(manifest),
        "--implement", "--budget-usd-per-instance", "12",
        "--lifecycle-out", str(lifecycle),
    ]
    old_args = wf._parse_args([*argv, "--planner-count", "1"])
    new_args = wf._parse_args(argv)
    inst = wf._load_and_filter(new_args)[0]
    prior = _lifecycle_summary(inst.task_id, inst.ext_id)
    prior["budget_usd"] = 12.0
    prior["checkpoint_fingerprint"] = wf._deepswe_checkpoint_fingerprint(
        wf.dataclasses.replace(inst, docker_image=prior["image_digest"]),
        args=old_args, child_budget_usd=12.0,
    )
    wf._append_lifecycle_record(out_path=lifecycle, summary=prior)
    original_record = lifecycle.read_bytes()
    monkeypatch.setattr(
        wf, "_run_task_cascade",
        lambda *args, **kwargs: pytest.fail("hidden-oracle rerun needs explicit authority"),
    )

    assert wf._implement_serial((inst,), args=new_args) == 1
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["halt_reason"] == "lifecycle_identity_conflict"
    assert lifecycle.read_bytes() == original_record


def test_retryable_checkpoint_status_runs_again_by_default(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    prior = _lifecycle_summary(
        inst.task_id,
        inst.ext_id,
        status="setup_failed",
    )
    wf._append_lifecycle_record(out_path=lifecycle, summary=prior)
    executed: list[str] = []

    def _task(task, **kwargs):
        executed.append(task.task_id)
        return _lifecycle_summary(task.task_id, task.ext_id)

    monkeypatch.setattr(wf, "_run_task_cascade", _task)
    assert wf._implement_serial((inst,), args=ns) == 0
    assert executed == [inst.task_id]
    record = wf._load_lifecycle_jsonl(lifecycle)[inst.task_id]
    assert record["status"] == "resolved"
    assert record["status_policy"] == "reusable"


def test_permanent_planning_checkpoint_is_reused_for_identical_recipe(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    prior = _lifecycle_summary(
        inst.task_id,
        inst.ext_id,
        status="candidate_failed",
    )
    prior.update(
        {
            "rc": 2,
            "reward": None,
            "resolved": None,
            "authoring_status": "failed",
            "candidate_capture_status": "failed",
            "candidate": None,
            "pristine_replay": None,
            "verifier_status": "not_run",
            "verifier_patch": None,
            "tests_snapshot": None,
            "architect_detail_status": "inflation_veto",
            "architect_decision": "rejected",
            "architect_terminal_class": "permanent_failure",
            "architect_failure_phase": "architect_planning",
            "architect_failure_kind": "inflation_veto",
            "architect_tiers_executed": 0,
            "zero_tier_settlement_proven": True,
            "secondary_failures": [],
            "failure_phase": "architect_planning",
            "failure_kind": "inflation_veto",
        }
    )
    prior["checkpoint_fingerprint"] = wf._deepswe_checkpoint_fingerprint(
        wf.dataclasses.replace(inst, docker_image=prior["image_digest"]),
        args=ns,
        child_budget_usd=None,
    )
    wf._append_lifecycle_record(out_path=lifecycle, summary=prior)
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *args, **kwargs: pytest.fail(
            "identical permanent planning terminal must not rerun"
        ),
    )

    assert wf._implement_serial((inst,), args=ns) == 1
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["tasks"][0]["resumed"] is True
    assert summary["tasks"][0]["status"] == "candidate_failed"
    assert summary["tasks"][0]["architect_detail_status"] == "inflation_veto"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architect_tiers_executed", 1),
        ("zero_tier_settlement_proven", False),
        (
            "secondary_failures",
            [{"phase": "cleanup", "kind": "cleanup_failed", "error": "x"}],
        ),
        ("cleanup_status", "failed"),
        ("candidate", {"schema": "crucible.deepswe.candidate.v1"}),
        ("verifier_status", "passed"),
    ],
)
def test_only_clean_zero_tier_terminal_can_be_permanent(field, value):
    summary = _lifecycle_summary("task", "ext", status="candidate_failed")
    summary.update(
        {
            "rc": 2,
            "reward": None,
            "resolved": None,
            "authoring_status": "failed",
            "candidate_capture_status": "failed",
            "candidate": None,
            "pristine_replay": None,
            "verifier_status": "not_run",
            "verifier_patch": None,
            "tests_snapshot": None,
            "architect_detail_status": "inflation_veto",
            "architect_terminal_class": "permanent_failure",
            "architect_failure_phase": "architect_planning",
            "architect_failure_kind": "inflation_veto",
            "architect_tiers_executed": 0,
            "zero_tier_settlement_proven": True,
            "secondary_failures": [],
            "failure_phase": "architect_planning",
            "failure_kind": "inflation_veto",
        }
    )
    summary[field] = value

    record = wf._lifecycle_record(summary)
    assert record["status_policy"] == "retryable"

    record["status_policy"] = "permanent"
    with pytest.raises(ValueError, match="unproven permanent policy"):
        wf._validate_lifecycle_record(record, line_number=1)


def test_lifecycle_rejects_boolean_architect_tier_count():
    record = wf._lifecycle_record(_lifecycle_summary("task", "ext"))
    record["architect_tiers_executed"] = False

    with pytest.raises(ValueError, match="invalid Architect tier count"):
        wf._validate_lifecycle_record(record, line_number=1)


def test_lifecycle_checkpoint_is_published_before_next_task(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "2",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    filtered = wf._load_and_filter(ns)
    calls = 0

    def _task(inst, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _lifecycle_summary(inst.task_id, inst.ext_id)

    monkeypatch.setattr(wf, "_run_task_cascade", _task)
    with pytest.raises(KeyboardInterrupt):
        wf._implement_serial(filtered, args=ns)

    records = [json.loads(line) for line in lifecycle.read_text().splitlines()]
    assert [record["task_id"] for record in records] == [filtered[0].task_id]


def test_task_execution_lease_prevents_concurrent_duplicate_work(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    lifecycle = tmp_path / "lifecycle.jsonl"
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
            "--lifecycle-out",
            str(lifecycle),
        ]
    )
    filtered = wf._load_and_filter(ns)
    calls: list[str] = []
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda inst, **kwargs: calls.append(inst.task_id),
    )

    # Model a separate live worker holding this task from before container
    # startup until its eventual checkpoint publication.
    with wf._task_execution_lease(ns.spec, filtered[0].task_id):
        assert wf._implement_serial(filtered, args=ns) == 1

    assert calls == []
    assert not lifecycle.exists()
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert summary["status"] == "task_lease_busy"
    assert summary["halt_reason"] == "task_lease_busy"
    assert summary["tasks"][0]["status"] == "task_lease_busy"
    # Another process holds the task: it is neither an ungraded result of
    # this run nor a row this run leaves behind for a re-run.
    assert summary["tasks_selected"] == 1
    assert summary["tasks_total"] == 1
    assert summary["tasks_lease_busy"] == 1
    assert summary["tasks_ungraded"] == 0
    assert summary["tasks_rerun_by_default"] == 0
    assert summary["tasks_not_started"] == 0
    assert summary["tasks_unrecorded"] == 0
    assert summary["tasks_by_status"] == {"task_lease_busy": 1}
    assert result["status"] == "task_lease_busy"
    assert any(event.get("reason") == "task_lease_busy" for event in events)


def test_architect_exception_still_rolls_up_current_attempt_cost(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    candidate = wf.candidate_from_image_patch(
        b"cost-rollup candidate\n",
        files_modified=("adaptix/_internal.py",),
        raw_diff_bytes=22,
        max_bytes=ns.max_patch_bytes,
    )
    calls: dict = {}
    _patch_container_lifecycle(
        monkeypatch,
        tmp_path,
        calls,
        candidate=candidate,
    )
    _patch_partial_tier_settlement(monkeypatch)

    def _raise_after_summary(**kwargs):
        return _publish_architect_child(
            Path(kwargs["spec"]),
            cost_usd=2.5,
            raise_after_summary=True,
        )

    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        _raise_after_summary,
    )
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 1,
            "reward": 0.0,
            "resolved": 0.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )
    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "outer")
    from crucible.common.cost import CostAccumulator

    cost = CostAccumulator()

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=cost,
        child_budget_usd=3.0,
    )

    assert summary["status"] == "unresolved"
    assert summary["authoring_status"] == "error"
    assert summary["cost_usd"] == pytest.approx(2.5)
    assert summary["cost_incomplete"] is False
    assert summary["architect_operation_id"] == wf._architect_operation_id(inst.task_id)
    assert summary["architect_run_id"]
    assert summary["architect_terminal_class"] == "permanent_failure"
    assert cost.total_cost_usd == pytest.approx(2.5)
    assert len(calls["verifier"]) == 1


def test_cleanup_failure_is_recorded_without_escaping_task(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
        ]
    )
    inst = wf._load_and_filter(ns)[0]
    calls: dict = {}
    _patch_container_lifecycle(monkeypatch, tmp_path, calls)

    def _architect(**kwargs):
        run = Path(kwargs["spec"]) / "runs" / "now_architect_attempt"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "costs": {"total_usd": 0.0, "rollup_incomplete": False},
                }
            )
        )
        return 0

    monkeypatch.setattr("crucible.workflows.architect.run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "_run_dataset_verify",
        lambda *args, **kwargs: {
            "rc": 0,
            "reward": 1.0,
            "resolved": 1.0,
            "verify_returncode": 0.0,
            "reward_valid": True,
        },
    )

    def _cleanup(container_id, image=None):
        if image is not None:
            raise RuntimeError("image cleanup failed")

    monkeypatch.setattr(wf, "teardown_instance_container", _cleanup)
    run_dir = _prepare_parent_run(tmp_path / "s" / "runs" / "outer")
    from crucible.common.cost import CostAccumulator

    summary = wf._run_task_cascade(
        inst,
        args=ns,
        run_dir=run_dir,
        cost=CostAccumulator(),
    )

    assert summary["status"] == "resolved"
    assert summary["cleanup_status"] == "failed"
    assert summary["cleanup_errors"]


def test_cleanup_failure_is_outer_failure_without_rewriting_domain_status(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    ns = wf._parse_args(
        [
            "--spec",
            str(tmp_path / "s"),
            "--dataset",
            str(manifest),
            "--implement", "--planner-count", "1",
            "--limit",
            "1",
        ]
    )
    monkeypatch.setattr(
        wf,
        "_run_task_cascade",
        lambda *args, **kwargs: {
            "task_id": "task-py-1",
            "ext_id": "e1",
            "status": "resolved",
            "reward": 1.0,
            "resolved": 1.0,
            "cost_usd": 0.0,
            "cost_incomplete": False,
            "cleanup_status": "failed",
            "cleanup_errors": ["image cleanup failed"],
        },
    )

    assert wf._implement_serial(wf._load_and_filter(ns), args=ns) == 1
    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == "cleanup_failed"
    assert summary["tasks_resolved"] == 1
    assert summary["tasks_cleanup_failed"] == 1
    assert summary["tasks"][0]["status"] == "resolved"
    assert summary["tasks"][0]["cleanup_status"] == "failed"
    assert result["status"] == "cleanup_failed"


def test_implement_empty_dataset_is_no_work_terminal(
    tmp_path: Path,
    monkeypatch,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    manifest = _write_manifest(tasks_root, [])
    spec_dir = tmp_path / "s"
    monkeypatch.setattr(
        RunHandle,
        "elapsed_s",
        property(lambda _: 4.75),
    )

    assert (
        wf.main(
            [
                "--spec",
                str(spec_dir),
                "--dataset",
                str(manifest),
                "--implement", "--planner-count", "1",
            ]
        )
        == 1
    )
    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == "no_work"
    assert summary["tasks_total"] == 0
    assert summary["tasks_selected"] == 0
    assert summary["tasks_not_started"] == 0
    assert summary["tasks_unrecorded"] == 0
    assert result["status"] == "no_work"
    assert result["duration_s"] == 4.75


def test_implement_preflight_failure_has_summary_result_and_event(
    tmp_path: Path,
    monkeypatch,
):
    manifest = _build_corpus(tmp_path)
    monkeypatch.setattr(wf, "docker_available", lambda: False)

    assert (
        wf.main(
            [
                "--spec",
                str(tmp_path / "s"),
                "--dataset",
                str(manifest),
                "--implement", "--planner-count", "1",
            ]
        )
        == 2
    )

    run_dir = next((tmp_path / "s" / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert summary["status"] == "container_runtime_required"
    assert result["status"] == "container_runtime_required"
    assert events[-1]["status"] == "container_runtime_required"
    # Four tasks were selected before Docker turned out to be unavailable.
    assert summary["tasks_selected"] == 4
    assert summary["tasks_total"] == 0
    assert summary["tasks_unrecorded"] == 0
    assert summary["tasks_not_started"] == 4


def test_implement_dataset_failure_is_a_visible_run(tmp_path: Path):
    spec_dir = tmp_path / "s"
    assert (
        wf.main(
            [
                "--spec",
                str(spec_dir),
                "--dataset",
                str(tmp_path / "missing.json"),
                "--implement", "--planner-count", "1",
            ]
        )
        == 2
    )

    run_dir = next((spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == "dataset_missing"
    assert result["status"] == "dataset_missing"
    assert summary["tasks_selected"] == 0
    assert summary["tasks_not_started"] == 0


# ---------- per-task cost roll-up (_architect_run_cost / _rollup_architect_cost) ----------


def test_architect_run_cost_uses_saga_identity_not_newest_directory(tmp_path: Path):
    """A newer unrelated/partial run cannot become this task's child."""

    parent = _prepare_parent_run(tmp_path / "parent-runs" / "deepswe-parent")
    child_corpus = tmp_path / "cross-corpus-child"
    operation_id = wf._architect_operation_id("task-1")
    with _outcome.child_operation(
        parent,
        operation_id=operation_id,
        workflow="architect",
    ):
        _publish_architect_child(child_corpus, cost_usd=8.85)

    unrelated = child_corpus / "runs" / "zzzz_architect_unrelated_partial"
    unrelated.mkdir(parents=True)
    unrelated.joinpath("summary.json").write_text(
        json.dumps(
            {
                "costs": {"total_usd": 999.0, "rollup_incomplete": False},
            }
        )
    )

    costs, outcome, run_dir = wf._architect_run_cost(parent, operation_id)

    assert costs is not None and costs["total_usd"] == 8.85
    assert outcome is not None and outcome.workflow == "architect"
    assert run_dir is not None and run_dir != unrelated
    assert run_dir.parent.parent == child_corpus


def test_architect_run_cost_none_when_absent(tmp_path: Path):
    parent = _prepare_parent_run(tmp_path / "runs" / "deepswe-parent")
    assert wf._architect_run_cost(parent, "architect-missing") == (
        None,
        None,
        None,
    )


def test_rollup_architect_cost_feeds_accumulator(tmp_path: Path):
    """_rollup_architect_cost replays per-role cost into the sweep
    CostAccumulator so total_cost_usd (and the budget halt) see it."""
    from crucible.common.cost import CostAccumulator
    from crucible.workflows import deepswe as wf

    acc = CostAccumulator()
    arch_costs = {
        "total_usd": 8.85,
        "by_role": {
            "breach_judge": {"cost_usd": 1.22, "backends": ["claude"], "models": []},
            "impl_breaker": {"cost_usd": 7.63, "backends": ["claude"], "models": []},
        },
    }
    total = wf._rollup_architect_cost(acc, arch_costs, task_id="t1")
    assert total == 8.85
    assert acc.total_cost_usd == pytest.approx(8.85)


def test_rollup_architect_cost_aggregate_fallback(tmp_path: Path):
    """With no by_role detail, fall back to a single aggregate entry."""
    from crucible.common.cost import CostAccumulator
    from crucible.workflows import deepswe as wf

    acc = CostAccumulator()
    total = wf._rollup_architect_cost(acc, {"total_usd": 3.5}, task_id="t2")
    assert total == 3.5
    assert acc.total_cost_usd == pytest.approx(3.5)


def _architect_pricing_costs() -> dict:
    first = {
        "input_tokens": 20,
        "output_tokens": 3,
        "cache_creation_tokens": 4,
        "cache_read_tokens": 6,
        "cost_usd": 0.2,
    }
    second = {
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 1,
        "cost_usd": 0.1,
    }
    calls = [
        {
            "role": "planner",
            "backend": "claude",
            "model": "opus",
            "usage": dict(first),
            "pricing_evidence": {
                "segments": [dict(first)],
                "complete": True,
            },
        },
        {
            "role": "judge",
            "backend": "codex",
            "model": "gpt",
            "usage": dict(second),
            "pricing_evidence": {
                "segments": [dict(second)],
                "complete": True,
            },
        },
    ]
    return {
        "pricing_evidence_schema": "crucible.usage-pricing-ledger.v1",
        "total_usd": 0.3,
        "total_input_tokens": 30,
        "total_output_tokens": 5,
        "total_cache_creation_tokens": 4,
        "total_cache_read_tokens": 7,
        # This role projection must not be replayed in addition to the calls.
        "by_role": {
            "planner": {
                "cost_usd": 0.2,
                "input_tokens": 20,
                "output_tokens": 3,
                "cache_creation_tokens": 4,
                "cache_read_tokens": 6,
                "calls": 1,
                "backends": ["claude"],
                "models": ["opus"],
            },
            "judge": {
                "cost_usd": 0.1,
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 1,
                "calls": 1,
                "backends": ["codex"],
                "models": ["gpt"],
            },
        },
        "by_backend": {"claude": 0.2, "codex": 0.1},
        "by_model": {"gpt": 0.1, "opus": 0.2},
        "unloaded_refs": 0,
        "accounting_debt": False,
        "accounting_debt_by_role": {},
        "rollup_incomplete": False,
        "pricing_rollup_incomplete": False,
        "pricing_evidence": {
            "segments": [dict(first), dict(second)],
            "complete": True,
        },
        "pricing_evidence_by_call": calls,
    }


def test_rollup_architect_cost_preserves_physical_pricing_call_order():
    from crucible.common.cost import CostAccumulator

    accumulator = CostAccumulator()
    architect_costs = _architect_pricing_costs()

    total = wf._rollup_architect_cost(
        accumulator,
        architect_costs,
        task_id="priced-task",
    )

    assert total == pytest.approx(0.3)
    assert accumulator.total_cost_usd == pytest.approx(0.3)
    assert accumulator.total_input_tokens == 30
    assert accumulator.total_output_tokens == 5
    assert accumulator.total_cache_creation_tokens == 4
    assert accumulator.total_cache_read_tokens == 7
    summary = accumulator.to_dict()
    assert summary["pricing_evidence"]["complete"] is True
    assert (
        summary["pricing_evidence_by_call"]
        == architect_costs["pricing_evidence_by_call"]
    )
    assert summary["by_role"]["planner"]["calls"] == 1
    assert summary["by_role"]["judge"]["calls"] == 1


def test_rollup_architect_cost_rejects_unversioned_cache_projection():
    from crucible.common.cost import CostAccumulator

    accumulator = CostAccumulator()

    with pytest.raises(ValueError, match="evidence schema is invalid"):
        wf._rollup_architect_cost(
            accumulator,
            {
                "total_usd": 0.5,
                "total_input_tokens": 15,
                "total_output_tokens": 3,
                "total_cache_creation_tokens": 2,
                "total_cache_read_tokens": 4,
                "by_role": {
                    "implementer": {
                        "cost_usd": 0.4,
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_creation_tokens": 2,
                        "cache_read_tokens": 3,
                        "backends": ["claude"],
                        "models": ["sonnet"],
                    },
                },
            },
            task_id="cache-task",
        )
    assert accumulator.total_cost_usd == 0.0
    assert accumulator.total_input_tokens == 0


def test_rollup_architect_cost_rejects_ledger_before_parent_mutation():
    from crucible.common.cost import CostAccumulator

    accumulator = CostAccumulator()
    architect_costs = _architect_pricing_costs()
    architect_costs["total_usd"] = 0.4

    with pytest.raises(ValueError, match="does not reconcile aggregate usage"):
        wf._rollup_architect_cost(
            accumulator,
            architect_costs,
            task_id="damaged-task",
        )
    assert accumulator.total_cost_usd == 0.0
    assert accumulator.total_input_tokens == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda costs: costs.pop("pricing_evidence_schema"),
            "evidence schema is invalid",
        ),
        (
            lambda costs: costs["by_role"]["planner"].__setitem__(
                "input_tokens",
                21,
            ),
            "by_role.*does not reconcile",
        ),
        (
            lambda costs: costs["by_backend"].__setitem__("claude", 0.21),
            "by_backend projection does not reconcile",
        ),
        (
            lambda costs: costs["by_model"].__setitem__("opus", 0.21),
            "by_model projection does not reconcile",
        ),
    ],
)
def test_rollup_architect_cost_rejects_version_or_projection_corruption(
    mutation,
    message,
):
    from crucible.common.cost import CostAccumulator

    accumulator = CostAccumulator()
    architect_costs = _architect_pricing_costs()
    mutation(architect_costs)

    with pytest.raises(ValueError, match=message):
        wf._rollup_architect_cost(
            accumulator,
            architect_costs,
            task_id="damaged-task",
        )
    assert accumulator.total_cost_usd == 0.0
    assert accumulator.total_input_tokens == 0


def test_rollup_architect_cost_rejects_schema_only_before_parent_mutation():
    from crucible.common.cost import CostAccumulator

    accumulator = CostAccumulator()

    with pytest.raises(ValueError, match="pricing_evidence_by_call is invalid"):
        wf._rollup_architect_cost(
            accumulator,
            {"pricing_evidence_schema": "crucible.usage-pricing-ledger.v1"},
            task_id="damaged-task",
        )
    assert accumulator.total_cost_usd == 0.0


def test_rollup_architect_cost_propagates_incomplete_pricing_state():
    from crucible.common.cost import CostAccumulator

    accumulator = CostAccumulator()
    architect_costs = _architect_pricing_costs()
    architect_costs["pricing_evidence"]["complete"] = False
    architect_costs["pricing_rollup_incomplete"] = True
    architect_costs["rollup_incomplete"] = True

    wf._rollup_architect_cost(
        accumulator,
        architect_costs,
        task_id="incomplete-task",
    )

    summary = accumulator.to_dict()
    assert summary["pricing_evidence"]["complete"] is False
    assert summary["rollup_incomplete"] is True
