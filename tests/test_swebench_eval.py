# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.swebench.eval`` — the swebench
harness subprocess wrapper + per-instance report parser.

The actual harness needs Docker; these tests stub
``_invoke_harness`` and pre-populate the output directory with
canonical-shape report.json files so the parser + aggregator can
be exercised without leaving the test process.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from crucible.products.swebench import eval as swebench_eval
from crucible.products.swebench.eval import (
    EvalReport,
    InstanceEvalReport,
    SwebenchEvalError,
    _parse_one_report,
    _read_instance_reports,
    run_evaluation,
)


# ----------------------------------------------------------------------
# Single-payload parser
# ----------------------------------------------------------------------


def test_parse_resolved_with_full_tests_status():
    payload = {
        "resolved": True,
        "patch_successfully_applied": True,
        "tests_status": {
            "FAIL_TO_PASS": {
                "success": ["test_a", "test_b"],
                "failure": [],
            },
            "PASS_TO_PASS": {
                "success": ["test_x"],
                "failure": ["test_y"],
            },
        },
    }
    r = _parse_one_report("django__django-1", payload)
    assert r.resolved is True
    assert r.patch_applied is True
    assert r.fail_to_pass_passed == 2
    assert r.fail_to_pass_failed == 0
    assert r.pass_to_pass_passed == 1
    assert r.pass_to_pass_failed == 1


def test_parse_not_resolved_patch_failed():
    payload = {
        "resolved": False,
        "patch_successfully_applied": False,
    }
    r = _parse_one_report("django__django-2", payload)
    assert r.resolved is False
    assert r.patch_applied is False
    assert r.fail_to_pass_passed == 0
    assert r.pass_to_pass_failed == 0


def test_parse_empty_payload_yields_skipped():
    """A missing per-instance report (e.g. harness skipped the instance
    due to image-build failure) lands as resolved=False / patch_applied=False
    with no tests_status counts."""
    r = _parse_one_report("django__django-3", {})
    assert r.resolved is False
    assert r.patch_applied is False
    assert r.fail_to_pass_passed == 0
    assert r.raw_report == {}


# ----------------------------------------------------------------------
# Output-directory walker
# ----------------------------------------------------------------------


def test_read_instance_reports_dict_keyed_by_instance_id(tmp_path: Path):
    """One report.json carrying a single instance_id-keyed dict — the
    most common shape the harness emits."""
    (tmp_path / "django__django-1").mkdir()
    (tmp_path / "django__django-1" / "report.json").write_text(json.dumps({
        "django__django-1": {
            "resolved": True,
            "patch_successfully_applied": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["t1"], "failure": []},
                "PASS_TO_PASS": {"success": [], "failure": []},
            },
        }
    }))
    reports = _read_instance_reports(tmp_path, ["django__django-1"])
    assert len(reports) == 1
    assert reports[0].resolved is True
    assert reports[0].fail_to_pass_passed == 1


def test_read_instance_reports_flat_self_describing(tmp_path: Path):
    """Alt shape: top-level dict with instance_id field."""
    (tmp_path / "django__django-1.json").write_text(json.dumps({
        "instance_id": "django__django-1",
        "resolved": False,
        "patch_successfully_applied": True,
    }))
    reports = _read_instance_reports(tmp_path, ["django__django-1"])
    assert reports[0].resolved is False
    assert reports[0].patch_applied is True


def test_read_instance_reports_preserves_request_order(tmp_path: Path):
    for iid in ["a-1", "b-2", "c-3"]:
        (tmp_path / f"{iid}.json").write_text(json.dumps({
            "instance_id": iid, "resolved": True,
            "patch_successfully_applied": True,
        }))
    # Request in a non-alphabetical order.
    reports = _read_instance_reports(tmp_path, ["c-3", "a-1", "b-2"])
    assert [r.instance_id for r in reports] == ["c-3", "a-1", "b-2"]


def test_read_instance_reports_missing_lands_as_skipped(tmp_path: Path):
    (tmp_path / "found.json").write_text(json.dumps({
        "instance_id": "found", "resolved": True,
        "patch_successfully_applied": True,
    }))
    reports = _read_instance_reports(tmp_path, ["found", "missing"])
    assert len(reports) == 2
    assert reports[0].instance_id == "found"
    assert reports[0].resolved is True
    assert reports[1].instance_id == "missing"
    assert reports[1].resolved is False
    assert reports[1].raw_report == {}


def test_read_instance_reports_skips_malformed_files(tmp_path: Path):
    (tmp_path / "good.json").write_text(json.dumps({
        "instance_id": "good", "resolved": True,
        "patch_successfully_applied": True,
    }))
    (tmp_path / "bad.json").write_text("not json")
    reports = _read_instance_reports(tmp_path, ["good", "bad"])
    # bad.json is silently skipped; "bad" lands as missing → skipped.
    assert reports[0].resolved is True
    assert reports[1].resolved is False


# ----------------------------------------------------------------------
# EvalReport aggregate
# ----------------------------------------------------------------------


def test_eval_report_aggregate_properties():
    per_instance = (
        InstanceEvalReport(
            instance_id="a", resolved=True, patch_applied=True,
        ),
        InstanceEvalReport(
            instance_id="b", resolved=False, patch_applied=True,
        ),
        InstanceEvalReport(
            instance_id="c", resolved=False, patch_applied=False,
        ),
    )
    report = EvalReport(
        run_id="r1", model="m", per_instance=per_instance,
        output_dir=Path("/tmp"),
    )
    assert report.total == 3
    assert report.resolved_count == 1
    assert report.patch_applied_count == 2
    assert abs(report.resolved_rate - 1/3) < 1e-6


def test_eval_report_resolved_rate_handles_empty():
    report = EvalReport(
        run_id="r", model="m", per_instance=(), output_dir=Path("/tmp"),
    )
    assert report.total == 0
    assert report.resolved_rate == 0.0


# ----------------------------------------------------------------------
# run_evaluation end-to-end (subprocess mocked)
# ----------------------------------------------------------------------


def test_run_evaluation_requires_swebench_package(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: False)
    with pytest.raises(SwebenchEvalError) as exc:
        run_evaluation(
            tmp_path / "preds.jsonl",
            instance_ids=["a"],
            run_id="r", model="m",
            output_dir=tmp_path / "out",
        )
    assert "swebench" in str(exc.value)


def test_run_evaluation_subprocess_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: True)

    def fake_invoke(cmd, *, cwd, timeout, log_path):
        return subprocess.CompletedProcess(
            cmd, returncode=1, stdout=b"", stderr=b"boom: Docker unreachable",
        )

    monkeypatch.setattr(swebench_eval, "_invoke_harness", fake_invoke)

    with pytest.raises(SwebenchEvalError) as exc:
        run_evaluation(
            tmp_path / "preds.jsonl",
            instance_ids=["a"],
            run_id="r", model="m",
            output_dir=tmp_path / "out",
        )
    assert "exited 1" in str(exc.value)
    assert "Docker unreachable" in str(exc.value)


def test_run_evaluation_no_json_files_raises_infra_error(
    tmp_path: Path, monkeypatch,
):
    """Harness exits 0 but the output dir has zero .json files —
    error message points at infrastructure (Docker / disk / perms)
    rather than parser / schema-drift."""
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: True)
    monkeypatch.setattr(
        swebench_eval, "_invoke_harness",
        lambda cmd, *, cwd, timeout, log_path: subprocess.CompletedProcess(
            cmd, returncode=0, stdout=b"", stderr=b"",
        ),
    )

    with pytest.raises(SwebenchEvalError) as exc:
        run_evaluation(
            tmp_path / "preds.jsonl",
            instance_ids=["a"],
            run_id="r", model="m",
            output_dir=tmp_path / "out",
        )
    assert "NO .json files" in str(exc.value)


def test_run_evaluation_json_present_but_empty_raises_parser_error(
    tmp_path: Path, monkeypatch,
):
    """Harness wrote JSON but every per-instance report parsed empty
    — error message points at the harness aborting during image-
    build, distinguishable from the infra-error path above."""
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: True)
    output_dir = tmp_path / "out"

    def fake_invoke(cmd, *, cwd, timeout, log_path):
        # Write a JSON that the parser will find but yields an
        # empty raw_report (e.g. harness wrote a wrapper but
        # never populated the per-instance keys).
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "wrapper.json").write_text(json.dumps({
            "unrelated_key": "unrelated_value",
        }))
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=b"", stderr=b"",
        )

    monkeypatch.setattr(swebench_eval, "_invoke_harness", fake_invoke)

    with pytest.raises(SwebenchEvalError) as exc:
        run_evaluation(
            tmp_path / "preds.jsonl",
            instance_ids=["a"],
            run_id="r", model="m",
            output_dir=output_dir,
        )
    assert ".json file(s)" in str(exc.value)
    assert "image-build" in str(exc.value)


def test_run_evaluation_forwards_instance_ids_to_harness(
    tmp_path: Path, monkeypatch,
):
    """The harness subprocess receives the filter's instance_ids via
    ``--instance_ids id1 id2 ...`` so Docker cost matches the
    filtered subset, not the full predictions file."""
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: True)

    captured_cmds: list[list[str]] = []

    def fake_invoke(cmd, *, cwd, timeout, log_path):
        captured_cmds.append(list(cmd))
        # Stub one report so the parser doesn't raise.
        cwd.mkdir(parents=True, exist_ok=True)
        (cwd / "django__django-1.json").write_text(json.dumps({
            "instance_id": "django__django-1",
            "resolved": True, "patch_successfully_applied": True,
        }))
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=b"ok", stderr=b"",
        )

    monkeypatch.setattr(swebench_eval, "_invoke_harness", fake_invoke)

    run_evaluation(
        tmp_path / "preds.jsonl",
        instance_ids=["django__django-1", "django__django-2"],
        run_id="r1", model="m",
        output_dir=tmp_path / "out",
    )
    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    assert cmd[:3] == [
        swebench_eval.sys.executable,
        "-m",
        "crucible.products.swebench.evaluator_compat",
    ]
    assert "--instance_ids" in cmd
    idx = cmd.index("--instance_ids")
    # Both instance ids appear after --instance_ids.
    assert "django__django-1" in cmd[idx:]
    assert "django__django-2" in cmd[idx:]


def test_run_evaluation_omits_instance_ids_when_empty(
    tmp_path: Path, monkeypatch,
):
    """When the caller doesn't scope (empty instance_ids), the
    harness gets no --instance_ids flag and evaluates everything in
    the predictions file (legacy behaviour preserved)."""
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: True)
    captured_cmds: list[list[str]] = []

    def fake_invoke(cmd, *, cwd, timeout, log_path):
        captured_cmds.append(list(cmd))
        cwd.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=b"", stderr=b"",
        )

    monkeypatch.setattr(swebench_eval, "_invoke_harness", fake_invoke)
    with pytest.raises(SwebenchEvalError):
        run_evaluation(
            tmp_path / "preds.jsonl",
            instance_ids=[],
            run_id="r1", model="m",
            output_dir=tmp_path / "out",
        )
    assert "--instance_ids" not in captured_cmds[0]


def test_run_evaluation_happy_path(tmp_path: Path, monkeypatch):
    """Harness exits 0 and writes report.json files; the wrapper
    parses them into a complete EvalReport."""
    monkeypatch.setattr(swebench_eval, "_harness_available", lambda: True)

    output_dir = tmp_path / "out"

    def fake_invoke(cmd, *, cwd, timeout, log_path):
        # Pre-populate the output dir with the canonical shape.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "django__django-1.json").write_text(json.dumps({
            "instance_id": "django__django-1",
            "resolved": True,
            "patch_successfully_applied": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["t1"], "failure": []},
                "PASS_TO_PASS": {"success": [], "failure": []},
            },
        }))
        (output_dir / "django__django-2.json").write_text(json.dumps({
            "instance_id": "django__django-2",
            "resolved": False,
            "patch_successfully_applied": False,
        }))
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(b"# stdout\n# stderr\n")
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout=b"ok", stderr=b"",
        )

    monkeypatch.setattr(swebench_eval, "_invoke_harness", fake_invoke)

    report = run_evaluation(
        tmp_path / "preds.jsonl",
        instance_ids=["django__django-1", "django__django-2"],
        run_id="r1",
        model="aws/anthropic/bedrock-claude-opus-4-8",
        output_dir=output_dir,
        log_path=tmp_path / "harness.log",
    )
    assert report.total == 2
    assert report.resolved_count == 1
    assert report.patch_applied_count == 1
    assert abs(report.resolved_rate - 0.5) < 1e-6
    assert (tmp_path / "harness.log").exists()
