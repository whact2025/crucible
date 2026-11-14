# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._run``: stdio tee + run-folder capture.

Coverage:
- ``capture_to_run_dir`` writes both stdout.log and stderr.log into
  the run folder, with original streams still receiving output.
- ``open_run_with_capture`` integrates with ``Corpus.open_run`` and
  yields a real ``RunHandle``.
- Each workflow ``main()`` produces ``<run_dir>/summary.json`` and
  ``<run_dir>/result.json`` (via ``handle.finalize``). No ``print()``
  reaches the real terminal — everything goes through Python logging
  or the tee.
"""

from __future__ import annotations

import io
import json
import logging
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from crucible.workflows._run import (
    FailureTerminal,
    capture_to_run_dir,
    open_run_with_capture,
    record_handled_terminal,
    write_atomic_json,
)


def test_capture_to_run_dir_writes_stdout_and_stderr(tmp_path: Path):
    run_dir = tmp_path / "run"
    # Capture also routes a copy through the original streams. Swap
    # them for in-memory buffers so the test doesn't pollute real stdio.
    orig_out = sys.stdout
    orig_err = sys.stderr
    fake_out = io.StringIO()
    fake_err = io.StringIO()
    sys.stdout = fake_out
    sys.stderr = fake_err
    try:
        with capture_to_run_dir(run_dir):
            sys.stdout.write("hello stdout\n")
            sys.stderr.write("hello stderr\n")
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err

    # The log files capture the writes.
    assert (run_dir / "stdout.log").read_text() == "hello stdout\n"
    assert (run_dir / "stderr.log").read_text() == "hello stderr\n"
    # The original streams (our in-memory buffers) ALSO received them.
    assert "hello stdout" in fake_out.getvalue()
    assert "hello stderr" in fake_err.getvalue()


def test_capture_restores_streams_on_exception(tmp_path: Path):
    run_dir = tmp_path / "run"
    orig_out = sys.stdout
    orig_err = sys.stderr
    try:
        with capture_to_run_dir(run_dir):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sys.stdout is orig_out
    assert sys.stderr is orig_err


def test_capture_routes_python_logging_to_stderr_log(tmp_path: Path):
    run_dir = tmp_path / "run"
    # logging's default StreamHandler writes to sys.stderr — the tee
    # picks it up because we swap sys.stderr.
    logger = logging.getLogger("crucible.test_capture")
    # Add a handler that writes to current sys.stderr (rebound each call).
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with capture_to_run_dir(run_dir):
            logger.info("hello from logging")
    finally:
        logger.removeHandler(handler)
    body = (run_dir / "stderr.log").read_text()
    assert "hello from logging" in body


def test_nested_capture_emits_each_log_record_once_and_restores_handlers(
    tmp_path: Path,
):
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"
    nested_token = "nested-capture-unique-token"
    parent_token = "parent-capture-unique-token"
    original_stderr = sys.stderr
    terminal = io.StringIO()
    root = logging.getLogger()
    original_handlers = tuple(root.handlers)
    original_level = root.level
    test_logger = logging.getLogger("crucible.test_capture.nested")
    logger_level = test_logger.level
    logger_propagate = test_logger.propagate
    observer_output = io.StringIO()
    observer = logging.StreamHandler(observer_output)
    observer.setLevel(logging.INFO)

    sys.stderr = terminal
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = True
    try:
        with capture_to_run_dir(parent_dir):
            with capture_to_run_dir(child_dir):
                # Add a non-Crucible handler after both live handlers. It must
                # still receive the record after the private marker is set.
                root.addHandler(observer)
                try:
                    test_logger.info(nested_token)
                finally:
                    root.removeHandler(observer)
            test_logger.info(parent_token)
    finally:
        test_logger.setLevel(logger_level)
        test_logger.propagate = logger_propagate
        if observer in root.handlers:
            root.removeHandler(observer)
        sys.stderr = original_stderr

    terminal_body = terminal.getvalue()
    parent_body = (parent_dir / "stderr.log").read_text()
    child_body = (child_dir / "stderr.log").read_text()
    assert terminal_body.count(nested_token) == 1
    assert parent_body.count(nested_token) == 1
    assert child_body.count(nested_token) == 1
    assert terminal_body.count(parent_token) == 1
    assert parent_body.count(parent_token) == 1
    assert parent_token not in child_body
    assert observer_output.getvalue().count(nested_token) == 1
    assert tuple(root.handlers) == original_handlers
    assert root.level == original_level


def test_open_run_with_capture_returns_real_handle(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")
    with open_run_with_capture(corpus, "test_workflow") as handle:
        assert handle.run_dir.is_dir()
        # Workflow name is in the folder name.
        assert "test_workflow" in handle.run_dir.name
        # Events file should be writeable.
        handle.write_event({"step": 0})
    # After exit: events.jsonl + stdout.log + stderr.log all present.
    assert (handle.run_dir / "events.jsonl").read_text().strip() == '{"step":0}'
    assert (handle.run_dir / "stdout.log").is_file()
    assert (handle.run_dir / "stderr.log").is_file()
    result = json.loads((handle.run_dir / "result.json").read_text())
    assert result["status"] == "completed"
    assert stat.S_IMODE(handle.run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((handle.run_dir / "stdout.log").stat().st_mode) == 0o600


def test_open_run_with_capture_forwards_explicit_no_target(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")

    with open_run_with_capture(
        corpus,
        "diagnostic",
        provenance_root=None,
    ) as handle:
        handle.write_summary({"status": "completed"})

    manifest = json.loads((handle.run_dir / "run_manifest.json").read_text())
    workspace = manifest["provenance"]["source"]["workspace"]
    assert workspace["binding"] == "none"
    assert workspace["identity_complete"] is False
    assert workspace["root"] is None


def test_open_run_with_capture_finalizes_exception(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")
    with pytest.raises(RuntimeError, match="boom"):
        with open_run_with_capture(corpus, "fails") as handle:
            raise RuntimeError("boom")
    result = json.loads((handle.run_dir / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["error_kind"] == "RuntimeError"
    assert result["duration_s"] > 0.0
    summary = json.loads((handle.run_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["workflow"] == "fails"
    assert summary["error_kind"] == "RuntimeError"


def test_open_run_with_capture_retains_supplied_failure_cost(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")
    known_cost = 5.251292

    with pytest.raises(RuntimeError, match="after paid child"):
        with open_run_with_capture(
            corpus,
            "paid_parent",
            failure_cost_usd=lambda: known_cost,
        ) as handle:
            known_cost += 2.726243
            raise RuntimeError("after paid child")

    summary = json.loads((handle.run_dir / "summary.json").read_text())
    result = json.loads((handle.run_dir / "result.json").read_text())
    outcome = json.loads((handle.run_dir / "outcome.json").read_text())
    assert summary["cost_usd"] == pytest.approx(7.977535)
    assert result["cost_usd"] == pytest.approx(7.977535)
    assert outcome["cost_usd"] == pytest.approx(7.977535)
    assert result["status"] == "failed"
    assert result["error_kind"] == "RuntimeError"
    assert outcome["terminal_class"] == "permanent_failure"


def test_failure_cost_error_does_not_mask_workflow_exception(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")

    def broken_cost_accounting() -> float:
        raise KeyboardInterrupt("cost ledger unavailable")

    with pytest.raises(RuntimeError, match="primary failure"):
        with open_run_with_capture(
            corpus,
            "broken_cost_parent",
            failure_cost_usd=broken_cost_accounting,
        ) as handle:
            raise RuntimeError("primary failure")

    summary = json.loads((handle.run_dir / "summary.json").read_text())
    result = json.loads((handle.run_dir / "result.json").read_text())
    assert result["error_kind"] == "RuntimeError"
    assert result["error_message"] == "primary failure"
    assert result["cost_usd"] == 0.0
    assert result["status"] == "incomplete_accounting"
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt"] is True
    assert summary["accounting_debt_by_gate"] == {
        "failure_cost_usd": ["cost_complete", "token_usage_complete"],
    }
    assert summary["cost_accounting_error"] == (
        "KeyboardInterrupt: cost ledger unavailable"
    )


@pytest.mark.parametrize("status", ("complete", "pass", "no_work"))
def test_failure_terminal_rejects_success_and_no_work(status: str):
    with pytest.raises(ValueError, match="non-success"):
        FailureTerminal(status=status)


@pytest.mark.parametrize("failed_operation", ("write_summary", "finalize_elapsed"))
def test_typed_failure_terminal_survives_one_shot_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_operation: str,
):
    from assay.corpus.runs import RunHandle

    corpus = Corpus(tmp_path / "spec")
    original = getattr(RunHandle, failed_operation)
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(f"one-shot {failed_operation} failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RunHandle, failed_operation, fail_once)

    with pytest.raises(RuntimeError, match="after unknown paid child"):
        with open_run_with_capture(
            corpus,
            "debt_parent",
            failure_cost_usd=lambda: 0.625,
            failure_terminal=lambda _error: FailureTerminal(
                status="incomplete_accounting",
                summary_fields={
                    "cost_usd_is_lower_bound": True,
                    "accounting_debt": True,
                    "accounting_debt_by_gate": {
                        "s0:breach": [
                            "cost_complete",
                            "token_usage_complete",
                        ],
                    },
                },
            ),
        ) as handle:
            raise RuntimeError("after unknown paid child")

    summary = json.loads((handle.run_dir / "summary.json").read_text())
    result = json.loads((handle.run_dir / "result.json").read_text())
    outcome = json.loads((handle.run_dir / "outcome.json").read_text())
    assert calls >= 2
    for terminal in (summary, result, outcome):
        assert terminal["cost_usd"] == pytest.approx(0.625)
    assert summary["status"] == "incomplete_accounting"
    assert summary["accounting_debt"] is True
    assert summary["cost_usd_is_lower_bound"] is True
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == "RuntimeError"
    assert result["duration_s"] > 0.0
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"


def test_failure_result_elapsed_retry_reconciles_effect_then_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assay.corpus.runs import RunHandle

    corpus = Corpus(tmp_path / "spec")
    original_finalize_elapsed = RunHandle.finalize_elapsed
    calls = 0
    observed_durations: list[float] = []

    def finalize_elapsed_then_fail_once(self, result):
        nonlocal calls
        calls += 1
        measured = original_finalize_elapsed(self, result)
        observed_durations.append(measured.duration_s)
        if calls == 1:
            raise OSError("measured result published before acknowledgement failed")
        return measured

    monkeypatch.setattr(
        RunHandle,
        "finalize_elapsed",
        finalize_elapsed_then_fail_once,
    )

    with pytest.raises(RuntimeError, match="after paid child"):
        with open_run_with_capture(
            corpus,
            "measured_retry_parent",
            failure_cost_usd=lambda: 0.625,
        ) as handle:
            raise RuntimeError("after paid child")

    result = json.loads((handle.run_dir / "result.json").read_text())
    assert calls == 2
    assert observed_durations[0] > 0.0
    assert observed_durations == [result["duration_s"], result["duration_s"]]
    assert result["status"] == "failed"
    assert result["cost_usd"] == pytest.approx(0.625)
    assert result["error_kind"] == "RuntimeError"


def test_failure_result_fallback_replays_one_measurement_after_effect_then_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from assay.corpus.runs import RunHandle

    corpus = Corpus(tmp_path / "spec")
    original_finalize = RunHandle.finalize
    elapsed_calls = 0
    finalize_calls = 0

    def fail_elapsed(_self, _result):
        nonlocal elapsed_calls
        elapsed_calls += 1
        raise OSError("measured publication unavailable")

    def finalize_then_fail_once(self, result):
        nonlocal finalize_calls
        finalize_calls += 1
        original_finalize(self, result)
        if finalize_calls == 1:
            raise OSError("result published before acknowledgement failed")

    monkeypatch.setattr(RunHandle, "finalize_elapsed", fail_elapsed)
    monkeypatch.setattr(RunHandle, "finalize", finalize_then_fail_once)

    with pytest.raises(RuntimeError, match="after paid child"):
        with open_run_with_capture(
            corpus,
            "measured_fallback_parent",
            failure_cost_usd=lambda: 0.625,
        ) as handle:
            raise RuntimeError("after paid child")

    result = json.loads((handle.run_dir / "result.json").read_text())
    outcome = json.loads((handle.run_dir / "outcome.json").read_text())
    assert elapsed_calls == 2
    assert finalize_calls == 2
    assert result["status"] == "failed"
    assert result["cost_usd"] == pytest.approx(0.625)
    assert result["duration_s"] > 0.0
    assert result["error_kind"] == "RuntimeError"
    assert result["error_message"] == "after paid child"
    assert outcome["detail_status"] == "failed"


def test_failure_terminal_callback_error_fails_closed_without_masking_body(
    tmp_path: Path,
):
    corpus = Corpus(tmp_path / "spec")

    def broken_terminal(_error: BaseException) -> FailureTerminal:
        raise KeyboardInterrupt("debt ledger unavailable")

    with pytest.raises(RuntimeError, match="primary workflow failure"):
        with open_run_with_capture(
            corpus,
            "broken_terminal_parent",
            failure_cost_usd=lambda: 1.25,
            failure_terminal=broken_terminal,
        ) as handle:
            raise RuntimeError("primary workflow failure")

    summary = json.loads((handle.run_dir / "summary.json").read_text())
    result = json.loads((handle.run_dir / "result.json").read_text())
    outcome = json.loads((handle.run_dir / "outcome.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["cost_usd"] == pytest.approx(1.25)
    assert summary["accounting_debt"] is True
    assert summary["cost_usd_is_lower_bound"] is True
    assert summary["accounting_debt_by_gate"] == {
        "failure_terminal": ["cost_complete", "token_usage_complete"],
    }
    assert summary["failure_terminal_error"] == (
        "KeyboardInterrupt: debt ledger unavailable"
    )
    assert result["status"] == "incomplete_accounting"
    assert result["error_kind"] == "RuntimeError"
    assert result["error_message"] == "primary workflow failure"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"


def test_persistent_typed_publication_failure_never_commits_generic_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from assay.corpus.runs import RunHandle

    corpus = Corpus(tmp_path / "spec")
    primary = RuntimeError("primary paid workflow failure")

    def fail_summary(*_args, **_kwargs):
        raise OSError("summary storage unavailable")

    def fail_result(*_args, **_kwargs):
        raise OSError("result storage unavailable")

    monkeypatch.setattr(RunHandle, "write_summary", fail_summary)
    monkeypatch.setattr(RunHandle, "finalize_elapsed", fail_result)
    monkeypatch.setattr(RunHandle, "finalize", fail_result)

    with pytest.raises(RuntimeError) as captured:
        with open_run_with_capture(
            corpus,
            "persistent_publication_failure",
            failure_cost_usd=lambda: 0.625,
            failure_terminal=lambda _error: FailureTerminal(
                status="incomplete_accounting",
                summary_fields={"accounting_debt": True},
            ),
        ) as handle:
            raise primary

    assert captured.value is primary
    assert not (handle.run_dir / "result.json").exists()
    assert not (handle.run_dir / "outcome.json").exists()


def test_exception_after_prefinalized_success_publishes_incomplete_outcome(
    tmp_path: Path,
):
    corpus = Corpus(tmp_path / "spec")
    primary = RuntimeError("cleanup failed after result")

    with pytest.raises(RuntimeError) as captured:
        with open_run_with_capture(corpus, "prefinalized") as handle:
            handle.write_summary({"status": "pass"})
            handle.finalize(RunResult(status="pass", cost_usd=7.25))
            raise primary

    assert captured.value is primary
    result = json.loads((handle.run_dir / "result.json").read_text())
    assert result["status"] == "pass"
    outcome = json.loads((handle.run_dir / "outcome.json").read_text())
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert outcome["cost_usd"] == pytest.approx(7.25)
    assert outcome["error_kind"] == "RuntimeError"
    assert outcome["implementation_refs"] == []


@pytest.mark.parametrize("failure_phase", ("enter", "exit"))
def test_capture_boundary_exception_gets_typed_failure_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
):
    from crucible.workflows import _run

    corpus = Corpus(tmp_path / "spec")
    boundary_error = RuntimeError(f"capture {failure_phase} failed")

    @contextmanager
    def failing_capture(_run_dir):
        if failure_phase == "enter":
            raise boundary_error
        try:
            yield
        finally:
            raise boundary_error

    monkeypatch.setattr(_run, "capture_to_run_dir", failing_capture)

    with pytest.raises(RuntimeError) as captured:
        with open_run_with_capture(corpus, "capture_boundary"):
            pass

    assert captured.value is boundary_error
    (run_dir,) = (corpus.spec_dir / "runs").iterdir()
    result = json.loads((run_dir / "result.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert result["status"] == "failed"
    assert result["error_message"] == str(boundary_error)
    assert outcome["terminal_class"] == "permanent_failure"


def test_debt_result_is_conservatively_recovered_after_outcome_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.workflows import _outcome

    corpus = Corpus(tmp_path / "spec")
    original_finalize = _outcome.finalize_run
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("outcome commit unavailable")
        return original_finalize(*args, **kwargs)

    monkeypatch.setattr(_outcome, "finalize_run", fail_once)
    with pytest.raises(RuntimeError, match="paid child failed"):
        with open_run_with_capture(
            corpus,
            "recover_debt",
            failure_cost_usd=lambda: 0.625,
            failure_terminal=lambda _error: FailureTerminal(
                status="incomplete_accounting",
                summary_fields={"accounting_debt": True},
            ),
        ) as handle:
            raise RuntimeError("paid child failed")

    assert (handle.run_dir / "result.json").is_file()
    assert not (handle.run_dir / "outcome.json").exists()
    manifest_path = handle.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["owner"] = {"pid": 999_999_999, "pid_start_token": "dead"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _outcome.reconcile_stale_runs(corpus) == (handle.run_dir.name,)
    outcome = json.loads((handle.run_dir / "outcome.json").read_text())
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["cost_usd"] == pytest.approx(0.625)
    assert outcome["error_kind"] == "process_lost_after_result"


def test_outcome_publication_baseexception_does_not_replace_body_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.workflows import _outcome

    corpus = Corpus(tmp_path / "spec")
    primary = KeyboardInterrupt("operator cancellation owns cleanup")
    secondary = MemoryError("outcome storage exhausted")
    monkeypatch.setattr(
        _outcome,
        "finalize_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(secondary),
    )

    with pytest.raises(KeyboardInterrupt) as captured:
        with open_run_with_capture(corpus, "owner_identity"):
            raise primary

    assert captured.value is primary
    assert any(
        "secondary typed-outcome publication failure" in note
        for note in getattr(primary, "__notes__", ())
    )


def test_failure_envelope_replaces_stale_checkpoint(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")

    with pytest.raises(RuntimeError, match="after checkpoint"):
        with open_run_with_capture(corpus, "checkpointed") as handle:
            handle.write_summary({"status": "pass", "progress": 2})
            raise RuntimeError("after checkpoint")

    summary = json.loads((handle.run_dir / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["partial_summary"] == {"progress": 2, "status": "pass"}


def test_record_handled_terminal_writes_complete_envelope(tmp_path: Path):
    corpus = Corpus(tmp_path / "spec")

    record_handled_terminal(
        corpus,
        "preflight",
        status="insufficient_input",
        message="need two measurements",
        error_kind="insufficient_input",
        extra={"observed": 1},
    )

    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert summary["observed"] == 1
    assert summary["status"] == result["status"] == "insufficient_input"
    assert summary["duration_s"] > 0
    assert result["duration_s"] == summary["duration_s"]
    workspace = manifest["provenance"]["source"]["workspace"]
    assert workspace["binding"] == "implicit_ambient_legacy"
    assert workspace["identity_complete"] is False


def test_write_atomic_json_replaces_privately_without_temp_residue(
    tmp_path: Path,
):
    path = tmp_path / "summary.json"
    write_atomic_json(path, {"checkpoint": 1, "path": tmp_path})
    write_atomic_json(path, {"checkpoint": 2})

    assert json.loads(path.read_text()) == {"checkpoint": 2}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".tmp_summary.json_*"))


def test_tee_stream_isatty_returns_false(tmp_path: Path):
    """Some libraries probe isatty() to decide on ANSI escapes; the
    tee should claim non-TTY so escapes don't leak into log files."""
    run_dir = tmp_path / "run"
    with capture_to_run_dir(run_dir):
        assert sys.stdout.isatty() is False
        assert sys.stderr.isatty() is False


def test_tee_stream_has_encoding(tmp_path: Path):
    """logging.StreamHandler reads stream.encoding; missing attr crashes."""
    run_dir = tmp_path / "run"
    with capture_to_run_dir(run_dir):
        assert sys.stdout.encoding  # truthy non-empty
        assert sys.stderr.encoding
