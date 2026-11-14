# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Wrap the official ``swebench`` evaluation harness as a subprocess.

The official harness (from the ``swebench`` PyPI package) is the canonical
scorer. Crucible invokes it through
``python -m crucible.products.swebench.evaluator_compat``, a version-gated
cleanup guard that otherwise preserves the upstream CLI. For SWE-bench
Verified it builds a Docker image per repo, applies each
prediction's ``model_patch`` plus the instance's ``test_patch`` to a
clean checkout at ``base_commit``, runs FAIL_TO_PASS + PASS_TO_PASS,
and emits a per-instance report classifying the prediction as
``resolved`` or not.

This module subprocess-invokes that harness and parses its output
into typed ``InstanceEvalReport`` records the workflow can fold into
Corpus Measurement records.

Requirements:

- The Crucible-pinned ``swebench`` package installed in the Python
  environment. The harness pulls in ``docker`` + ~30 other deps.
- A working Docker daemon. The harness fails fast with a clear
  error when Docker isn't reachable.
- Network for the initial image pulls / builds; subsequent runs
  reuse the cached images.

Tests in this repo mock ``_invoke_harness`` to avoid Docker; the
parser is exercised against a stub output directory shaped to the
harness's documented per-instance report contract.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from crucible.common.benchmark_runtime import run_process_tree

logger = logging.getLogger(__name__)


class SwebenchEvalError(RuntimeError):
    """Raised when the harness subprocess fails or its output can't be parsed."""


@dataclass(frozen=True)
class InstanceEvalReport:
    """One instance's evaluation outcome."""

    instance_id: str
    resolved: bool
    patch_applied: bool
    fail_to_pass_passed: int = 0
    fail_to_pass_failed: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_failed: int = 0
    raw_report: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    """Aggregate of all per-instance reports for one harness run."""

    run_id: str
    model: str
    per_instance: tuple[InstanceEvalReport, ...]
    output_dir: Path

    @property
    def total(self) -> int:
        return len(self.per_instance)

    @property
    def resolved_count(self) -> int:
        return sum(1 for r in self.per_instance if r.resolved)

    @property
    def patch_applied_count(self) -> int:
        return sum(1 for r in self.per_instance if r.patch_applied)

    @property
    def resolved_rate(self) -> float:
        return self.resolved_count / self.total if self.total else 0.0


def _harness_available() -> bool:
    """Probe whether the ``swebench`` package is importable.

    SWE-bench is a pinned Crucible base dependency.  Keeping this explicit
    probe turns an incomplete/broken installation into a focused workflow
    error before any Docker setup begins.
    """
    try:
        import swebench  # noqa: F401
    except ImportError:
        return False
    return True


def _invoke_harness(
    cmd: list[str],
    *,
    cwd: Path | None,
    timeout: int,
    log_path: Path | None,
) -> subprocess.CompletedProcess:
    """Run the harness subprocess. Hookable for tests via monkeypatch.

    Captures stdout + stderr; if ``log_path`` is set, persists the
    combined log there for postmortem. Does NOT raise on non-zero
    exit — the caller inspects ``returncode``.
    """
    logger.info("swebench eval: invoking %s", " ".join(cmd))
    result = run_process_tree(cmd, cwd=cwd, timeout=timeout)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_bytes(
            b"# stdout\n" + (result.stdout or b"")
            + b"\n# stderr\n" + (result.stderr or b"")
        )
    return result


def _read_instance_reports(
    output_dir: Path,
    instance_ids: Iterable[str],
) -> tuple[InstanceEvalReport, ...]:
    """Walk the harness's output directory and parse per-instance
    report.json files.

    The harness writes per-instance reports under a nested layout
    that varies slightly across releases; this parser searches the
    output tree for ``<instance_id>.json`` or ``<instance_id>/report.json``
    and tolerates either shape.

    Returns reports in the same order as ``instance_ids``. Missing
    reports (e.g. instances the harness skipped due to image-build
    failure) land as ``resolved=False / patch_applied=False`` with
    ``raw_report={}``.
    """
    requested = list(instance_ids)
    by_id: dict[str, dict] = {}

    candidates: list[Path] = []
    candidates.extend(output_dir.rglob("report.json"))
    candidates.extend(output_dir.rglob("*.json"))

    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Two known shapes:
        # 1) {"<instance_id>": {...}} — one or more instances per file
        # 2) {"resolved": bool, "instance_id": "...", ...} — single instance
        if not isinstance(data, dict):
            continue
        if "instance_id" in data and isinstance(data["instance_id"], str):
            by_id.setdefault(data["instance_id"], data)
            continue
        for key, payload in data.items():
            if not isinstance(payload, dict):
                continue
            if key in requested:
                by_id.setdefault(key, payload)

    out: list[InstanceEvalReport] = []
    for iid in requested:
        payload = by_id.get(iid, {})
        out.append(_parse_one_report(iid, payload))
    return tuple(out)


def _parse_one_report(instance_id: str, payload: dict) -> InstanceEvalReport:
    """Render one harness payload into a typed InstanceEvalReport.

    Tolerates the published harness shape (``tests_status`` dict
    with PASS_TO_PASS / FAIL_TO_PASS nested success/failure lists)
    plus the common shortcut where ``resolved`` is a top-level bool
    and the test counts are absent.
    """
    if not payload:
        return InstanceEvalReport(
            instance_id=instance_id,
            resolved=False,
            patch_applied=False,
        )
    resolved = bool(payload.get("resolved", False))
    patch_applied = bool(payload.get("patch_successfully_applied", False))
    tests = payload.get("tests_status") or {}
    f2p = tests.get("FAIL_TO_PASS") or {}
    p2p = tests.get("PASS_TO_PASS") or {}
    return InstanceEvalReport(
        instance_id=instance_id,
        resolved=resolved,
        patch_applied=patch_applied,
        fail_to_pass_passed=len((f2p.get("success") or [])),
        fail_to_pass_failed=len((f2p.get("failure") or [])),
        pass_to_pass_passed=len((p2p.get("success") or [])),
        pass_to_pass_failed=len((p2p.get("failure") or [])),
        raw_report=payload,
    )


def run_evaluation(
    predictions_path: Path,
    instance_ids: Iterable[str],
    *,
    run_id: str,
    model: str,
    output_dir: Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    max_workers: int = 4,
    timeout: int = 7200,
    log_path: Path | None = None,
    harness_module: str = "crucible.products.swebench.evaluator_compat",
) -> EvalReport:
    """Invoke the swebench harness against ``predictions_path`` and
    parse its per-instance reports.

    Args:
      predictions_path: JSONL produced by ``crucible swebench
        --implement`` (or any other agent in the leaderboard shape).
      instance_ids: the instance ids that appear in predictions —
        used to scope parsing and detect skipped instances.
      run_id: harness run identifier (used for output-dir naming).
      model: predictor's ``model_name_or_path`` (forwarded to the
        harness for its output-dir naming).
      output_dir: directory the harness writes per-instance reports
        into (``--output_dir`` to the harness).
      dataset_name: HuggingFace dataset id; default is the SWE-bench
        Verified canonical id.
      max_workers: harness concurrency cap (default 4 — Docker is
        the bottleneck, not Python).
      timeout: subprocess timeout in seconds (default 2h — full
        Verified runs can exceed this on a cold image cache).
      log_path: combined harness stdout/stderr persistence path.
      harness_module: override for tests / forks.

    Raises ``SwebenchEvalError`` when the pinned harness package isn't
    importable, the subprocess exits non-zero, or the output
    directory contains no parseable reports.
    """
    if not _harness_available():
        raise SwebenchEvalError(
            "Crucible's pinned swebench base dependency is not importable; "
            "reinstall/sync the Crucible environment before evaluation. "
            "The harness also needs a working Docker daemon to run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_id_list = list(instance_ids)
    cmd = [
        sys.executable, "-m", harness_module,
        "--dataset_name", dataset_name,
        "--predictions_path", str(predictions_path),
        "--run_id", run_id,
        "--max_workers", str(max_workers),
    ]
    if instance_id_list:
        # Without --instance_ids the harness evaluates every record in
        # the predictions file — meaning a workflow filter like --repo
        # django/django ends up paying Docker cost for every instance
        # in the predictions.jsonl, not just the filtered subset.
        # Forward the scope explicitly so eval cost matches the
        # filter the operator passed.
        cmd.extend(["--instance_ids", *instance_id_list])
    result = _invoke_harness(
        cmd, cwd=output_dir, timeout=timeout, log_path=log_path,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or b"")[-2048:].decode(
            "utf-8", errors="replace",
        )
        raise SwebenchEvalError(
            f"harness exited {result.returncode}: {stderr_tail}"
        )

    per_instance = _read_instance_reports(output_dir, instance_id_list)

    # Distinguish two failure modes with distinct error text so the
    # operator knows whether it's an infra problem (no JSON written
    # at all → check Docker/disk/permissions) vs. a parser problem
    # (JSON written but reports all-empty → harness skipped every
    # instance due to image-build failure, or schema drift the
    # parser doesn't recognize).
    if not per_instance:
        raise SwebenchEvalError(
            f"harness output dir {output_dir} contained zero reports — "
            f"instance_ids list was empty, or the harness did not "
            f"write any JSON. Check {log_path} for the harness's "
            f"stdout / stderr."
        )
    if all(not r.raw_report for r in per_instance):
        json_files = list(output_dir.rglob("*.json"))
        if not json_files:
            raise SwebenchEvalError(
                f"harness output dir {output_dir} has NO .json files — "
                f"the harness almost certainly failed before writing "
                f"any per-instance report (Docker unreachable / "
                f"image-build failure / disk full). Check "
                f"{log_path} for the harness's stdout / stderr."
            )
        raise SwebenchEvalError(
            f"harness output dir {output_dir} has {len(json_files)} "
            f".json file(s) but every per-instance report parsed "
            f"empty — usually means the harness aborted each "
            f"instance during image-build before running the "
            f"FAIL_TO_PASS / PASS_TO_PASS suite. Inspect the JSONs "
            f"directly or {log_path} for the harness's per-instance "
            f"failure reason."
        )
    return EvalReport(
        run_id=run_id,
        model=model,
        per_instance=per_instance,
        output_dir=output_dir,
    )


__all__ = [
    "EvalReport",
    "InstanceEvalReport",
    "SwebenchEvalError",
    "run_evaluation",
]
