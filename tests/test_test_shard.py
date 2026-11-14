# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The CI partition must cover pytest's file inventory exactly once."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import test_shard
from scripts.test_shard import SHARD_COUNT, build_shards, discover_test_files, main, verify_shard_evidence


def _write_module(root: Path, relative: str, source: str = "def test_ok():\n    pass\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_partition_matches_real_pytest_collection_and_includes_new_files(tmp_path):
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nnorecursedirs = .venv __pycache__\n", encoding="utf-8",
    )
    modules = {
        "tests/test_workflows_breach.py": 0,
        "tests/test_workflows_tier_mutation_transaction.py": 1,
        "tests/test_workflows_new_feature.py": None,
        "tests/test_workflow_new_contract.py": None,
        "tests/test_new_runtime.py": None,
        "tests/nested/new_feature_test.py": None,
    }
    for path in modules:
        _write_module(tmp_path, path)
    _write_module(tmp_path, "tests/helpers.py")
    _write_module(tmp_path, "tests/.venv/test_dependency.py")
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    collected_files = {
        line.split("::", 1)[0]
        for line in collected.stdout.splitlines()
        if ".py::" in line
    }
    assert collected_files == set(modules)
    shards = build_shards(discover_test_files(tmp_path))
    flattened = [path for shard in shards for path in shard]
    assert set(flattened) == set(modules)
    assert len(flattened) == len(set(flattened))
    assert shards[0] == ("tests/test_workflows_breach.py",)
    assert shards[1] == ("tests/test_workflows_tier_mutation_transaction.py",)
    assert len(shards) == SHARD_COUNT
    assert build_shards(reversed(flattened)) == shards


def test_listing_writes_the_complete_plan_without_starting_pytest(tmp_path, monkeypatch, capsys):
    _write_module(tmp_path, "tests/test_workflows_breach.py")
    _write_module(tmp_path, "tests/test_added_runtime.py")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected pytest dispatch"))
    assert main(["--root", str(tmp_path), "--index", "2", "--list", "--plan-out", "plan.json"]) == 0
    assert capsys.readouterr().out.strip() == "tests/test_added_runtime.py"
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["shard_index"] == 2
    assert plan["shard_count"] == SHARD_COUNT
    assert plan["shards"]["0"] == ["tests/test_workflows_breach.py"]
    assert plan["shards"]["2"] == ["tests/test_added_runtime.py"]


def test_empty_shard_never_falls_back_to_running_the_full_suite(tmp_path, monkeypatch):
    _write_module(tmp_path, "tests/test_other.py")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected pytest dispatch"))
    with pytest.raises(SystemExit) as exc:
        main(["--root", str(tmp_path), "--index", "0"])
    assert exc.value.code == 2


def test_shard_propagates_pytest_failure_for_only_the_selected_files(tmp_path, monkeypatch):
    _write_module(tmp_path, "tests/test_workflows_breach.py")
    _write_module(tmp_path, "tests/test_unrelated.py")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", run)
    assert main(["--root", str(tmp_path), "--index", "0"]) == 7
    command, options = calls[0]
    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert [arg for arg in command if arg.startswith("tests/")] == ["tests/test_workflows_breach.py"]
    assert options["cwd"] == tmp_path


def test_duplicate_or_escaping_inventory_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_shards(["tests/test_one.py", "tests/test_one.py"])
    with pytest.raises(ValueError, match="non-canonical"):
        build_shards(["tests/../outside/test_one.py"])



def test_weighted_partition_spreads_slow_files_without_omitting_new_modules(monkeypatch):
    slow = [f"tests/test_large_{index}.py" for index in range(6)]
    fresh = [f"tests/test_added_{index}.py" for index in range(60)]
    monkeypatch.setattr(test_shard, "_RUNTIME_WEIGHT_SECONDS", {path: 1200 for path in slow})
    shards = build_shards([*slow, *fresh])

    assert all(len(set(shard).intersection(slow)) == 1 for shard in shards[2:])
    assert set(path for shard in shards for path in shard) == set(slow + fresh)
    assert build_shards(reversed(slow + fresh)) == shards


def _write_shard_evidence(root: Path) -> None:
    for path in (
        "tests/test_workflows_breach.py",
        "tests/test_workflows_tier_mutation_transaction.py",
        *(f"tests/test_added_{index}.py" for index in range(6)),
    ):
        _write_module(root, path)
    shards = build_shards(discover_test_files(root))
    for index in range(SHARD_COUNT):
        (root / f".coverage.shard-{index}").write_bytes(b"coverage database")
        (root / f"test-results-{index}.xml").write_text("<testsuite/>\n", encoding="utf-8")
        plan = {
            "schema_version": "crucible.test-shards.v2",
            "shard_count": SHARD_COUNT,
            "shard_index": index,
            "shards": {str(lane): list(files) for lane, files in enumerate(shards)},
        }
        (root / f"test-shard-plan-{index}.json").write_text(json.dumps(plan), encoding="utf-8")


@pytest.mark.parametrize("name", (
    f".coverage.shard-{SHARD_COUNT - 1}",
    f"test-results-{SHARD_COUNT - 1}.xml",
    f"test-shard-plan-{SHARD_COUNT - 1}.json",
))
def test_combination_requires_every_lane_coverage_report_and_plan(tmp_path, name):
    _write_shard_evidence(tmp_path)
    verify_shard_evidence(tmp_path)
    (tmp_path / name).unlink()
    with pytest.raises(ValueError, match="missing or empty required shard artifact"):
        verify_shard_evidence(tmp_path)


def test_combination_rejects_a_plan_that_omits_a_new_test_file(tmp_path):
    _write_shard_evidence(tmp_path)
    _write_module(tmp_path, "tests/test_new_contract.py")
    with pytest.raises(ValueError, match="current complete inventory"):
        verify_shard_evidence(tmp_path)


def test_ci_matrix_and_final_gate_require_the_complete_partition():
    import yaml

    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    assert jobs["tests"]["strategy"]["matrix"]["shard"] == list(range(SHARD_COUNT))
    namespace_step = next(
        step for step in jobs["tests"]["steps"]
        if step["name"] == "Enable hosted-runner user namespaces for descriptor tests"
    )
    assert "test -f /proc/sys/kernel/apparmor_restrict_unprivileged_userns" in namespace_step["run"]
    assert 'test "$(unshare -Ur id -u)" = 0' in namespace_step["run"]
    assert jobs["coverage"]["needs"] == ["tests"]
    assert set(jobs["gate"]["needs"]) == {"quality", "tests", "coverage"}
    coverage_steps = jobs["coverage"]["steps"]
    assert any("verify_shard_evidence" in step.get("run", "") for step in coverage_steps)
    assert any("--fail-under=55" in step.get("run", "") for step in coverage_steps)
    upload = next(step for step in coverage_steps if step["name"] == "Upload combined coverage evidence")
    assert "hashFiles('crucible/coverage.xml')" in upload["if"]
    assert upload["with"]["include-hidden-files"] is True
