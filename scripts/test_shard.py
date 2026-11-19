#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Run one deterministic whole-file shard of the non-live test suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Iterable


SHARD_COUNT = 8
_DEDICATED_SHARDS = {
    "tests/test_workflows_breach.py": 0,
    "tests/test_workflows_tier_mutation_transaction.py": 1,
}


# Scheduling hints from hosted run 43345350017: completed-file JUnit times
# plus estimates apportioned from timestamped workflow progress. They affect
# placement only; every discovered file still runs. The unfinished writer tail
# has a conservative allowance until a complete duration is available.
# The hints for test_deepswe_gate_residue.py and
# test_workflows_tier_unjudged_outcomes.py, and the raised hint for
# test_workflows_tier_budget_terminal.py (18 to 34 cases), are estimates from
# a local eight-shard run, not from a hosted run.
_DEFAULT_WEIGHT_SECONDS = 30
_RUNTIME_WEIGHT_SECONDS = {
    'tests/test_deepswe_gate_residue.py': 450,
    'tests/test_mutation_journal_active_parent_handoff.py': 270,
    'tests/test_mutation_journal_authority_regressions.py': 240,
    'tests/test_mutation_journal_external_root_lock.py': 330,
    'tests/test_mutation_journal_git_admin_adversarial.py': 210,
    'tests/test_mutation_journal_restore_confinement.py': 210,
    'tests/test_mutation_journal_writer_epoch_integration.py': 120,
    'tests/test_reaper.py': 600,
    'tests/test_tier_build_progress_retry.py': 120,
    'tests/test_tier_restored_path_k3_compatibility.py': 150,
    'tests/test_uir01_final_adversarial.py': 390,
    'tests/test_workflow_diagnostic_cost.py': 120,
    'tests/test_workflows_breach.py': 5430,
    'tests/test_workflows_deepswe.py': 570,
    'tests/test_workflows_handoff.py': 780,
    'tests/test_workflows_implement_transaction.py': 480,
    'tests/test_workflows_journal_recover.py': 750,
    'tests/test_workflows_swebench.py': 480,
    'tests/test_workflows_tier_breach_policy.py': 300,
    'tests/test_workflows_tier_budget_terminal.py': 510,
    'tests/test_workflows_tier_cargo_options.py': 270,
    'tests/test_workflows_tier_diagnostic_recovery.py': 180,
    'tests/test_workflows_tier_empty_diff_retry.py': 2490,
    'tests/test_workflows_tier_final_binding.py': 1980,
    'tests/test_workflows_tier_impl_system_prompt.py': 150,
    'tests/test_workflows_tier_mutation_transaction.py': 2910,
    'tests/test_workflows_tier_restored_path_recovery.py': 540,
    'tests/test_workflows_tier_review_policy.py': 330,
    'tests/test_workflows_tier_review_retry.py': 810,
    'tests/test_workflows_tier_transport.py': 180,
    'tests/test_workflows_tier_unjudged_outcomes.py': 330,
    'tests/test_workflows_writer_scheduler.py': 1710,
}


def discover_test_files(root: Path) -> tuple[str, ...]:
    """Discover the repository's pytest file patterns without importing tests."""

    tests = root / "tests"
    if not tests.is_dir():
        raise ValueError(f"test directory does not exist: {tests}")
    return tuple(sorted(
        path.relative_to(root).as_posix()
        for path in tests.rglob("*.py")
        if path.is_file()
        and not {".venv", "__pycache__"}.intersection(path.relative_to(tests).parts)
        and (path.name.startswith("test_") or path.name.endswith("_test.py"))
    ))


def _validate_test_path(path: str) -> None:
    candidate = PurePosixPath(path)
    if (
        candidate.as_posix() != path
        or candidate.is_absolute()
        or "\\" in path
        or ".." in candidate.parts
        or not candidate.parts
        or candidate.parts[0] != "tests"
        or candidate.suffix != ".py"
        or not (
            candidate.name.startswith("test_")
            or candidate.name.endswith("_test.py")
        )
    ):
        raise ValueError(f"non-canonical test module: {path!r}")


def _weight(path: str) -> int:
    return _RUNTIME_WEIGHT_SECONDS.get(path, _DEFAULT_WEIGHT_SECONDS)


def build_shards(paths: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Exhaustively balance whole files, keeping two measured heavy files alone."""

    files = tuple(sorted(paths))
    if len(set(files)) != len(files):
        raise ValueError("test inventory contains duplicate module paths")
    shards: list[list[str]] = [[] for _ in range(SHARD_COUNT)]
    weights = [0] * SHARD_COUNT
    for path in files:
        _validate_test_path(path)
        if path in _DEDICATED_SHARDS:
            index = _DEDICATED_SHARDS[path]
            shards[index].append(path)
            weights[index] += _weight(path)
    remaining = (path for path in files if path not in _DEDICATED_SHARDS)
    for path in sorted(remaining, key=lambda path: (-_weight(path), path)):
        index = min(
            range(len(_DEDICATED_SHARDS), SHARD_COUNT),
            key=lambda index: (weights[index], len(shards[index]), index),
        )
        shards[index].append(path)
        weights[index] += _weight(path)
    return tuple(tuple(sorted(shard)) for shard in shards)


def _plan_shards(shards: tuple[tuple[str, ...], ...]) -> dict[str, list[str]]:
    return {str(index): list(files) for index, files in enumerate(shards)}


def verify_shard_evidence(root: Path) -> None:
    """Require every lane's complete artifacts and its exact current inventory."""

    expected = _plan_shards(build_shards(discover_test_files(root)))
    for index in range(SHARD_COUNT):
        for name in (
            f".coverage.shard-{index}",
            f"test-results-{index}.xml",
            f"test-shard-plan-{index}.json",
        ):
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty required shard artifact: {name}")
        plan = json.loads((root / f"test-shard-plan-{index}.json").read_text(encoding="utf-8"))
        if (
            not isinstance(plan, dict)
            or plan.get("schema_version") != "crucible.test-shards.v2"
            or type(plan.get("shard_count")) is not int
            or plan["shard_count"] != SHARD_COUNT
            or type(plan.get("shard_index")) is not int
            or plan["shard_index"] != index
            or plan.get("shards") != expected
        ):
            raise ValueError(f"shard {index} plan does not match the current complete inventory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, choices=range(SHARD_COUNT), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--list", action="store_true", help="Print selected files without running pytest.")
    parser.add_argument("--plan-out", type=Path, help="Write the complete deterministic shard inventory as JSON.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        shards = build_shards(discover_test_files(root))
    except ValueError as exc:
        parser.error(str(exc))
    selected = shards[args.index]
    if not selected:
        parser.error(f"shard {args.index} has no test modules; refusing an unscoped pytest run")
    if args.plan_out is not None:
        destination = args.plan_out if args.plan_out.is_absolute() else root / args.plan_out
        destination.write_text(json.dumps({
            "schema_version": "crucible.test-shards.v2",
            "shard_count": SHARD_COUNT,
            "shard_index": args.index,
            "shards": _plan_shards(shards),
            "estimated_seconds": {str(index): sum(map(_weight, files)) for index, files in enumerate(shards)},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.list:
        print("\n".join(selected))
        return 0
    print(f"Shard {args.index}: {len(selected)} whole test modules", flush=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-v", "--durations=25",
            "--cov=.", "--cov-report=", "--cov-fail-under=0",
            f"--junitxml=test-results-{args.index}.xml", *selected,
        ],
        cwd=root,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
