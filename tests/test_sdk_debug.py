# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.backends._sdk_debug.SdkDebugSink``.

Covers the dump-format contract and the no-op path when log_dir is
None. Exercises ClaudeBackend end-to-end with a monkeypatched
``claude_agent_sdk.query`` so the sink writes a real sdk_*.json file
to a tmp_path without invoking the real Claude CLI.

Coverage:
- Sink is a no-op when log_dir is None.
- Successful invocation writes ``sdk_0000_claude.json`` with the full
  invocation + result payload.
- Failure invocation writes the file with ``traceback`` populated.
- Counter is monotonic across calls (parallel use).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from assay import (
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    Message,
    Usage,
    UsageCostBounds,
    UsageCostBoundSegment,
    UsagePricingEvidence,
    UsagePricingSegment,
    TransportError,
)
import crucible.backends._sdk_debug as sdk_debug
from crucible.backends._sdk_debug import SdkDebugSink


# ----------------------------------------------------------------------
# Direct sink coverage
# ----------------------------------------------------------------------


def _load_result_sidecar(
    root: Path,
    payload: dict,
) -> tuple[Path, bytes, dict]:
    ref = payload["full_redacted_result_ref"]
    relative_path = payload["full_redacted_result_path"]
    assert ref.startswith("sha256:")
    assert relative_path == f"sdk_artifacts/{ref.removeprefix('sha256:')}.json"
    path = root / relative_path
    body = path.read_bytes()
    assert hashlib.sha256(body).hexdigest() == ref.removeprefix("sha256:")
    decoded = json.loads(body)
    assert body == json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return path, body, decoded


def test_sink_is_noop_when_log_dir_none():
    sink = SdkDebugSink(log_dir=None, backend_name="claude")
    assert not sink.enabled()
    inv = Invocation(system="s", user="u", tools=None,
                     max_turns=5, budget_usd=None)
    result = InvocationResult(
        final_message="ok", transcript=(), usage=Usage(),
        stop_reason="end_turn",
    )
    assert sink.dump(
        invocation=inv, result=result,
        started_iso="2026-05-18T12:00:00+00:00", duration_s=0.1,
    ) is None


def test_sink_writes_payload_on_success(tmp_path: Path):
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="claude")
    assert sink.enabled()
    inv = Invocation(
        system="be terse", user="hello", tools=None,
        max_turns=5, budget_usd=1.0,
    )
    result = InvocationResult(
        final_message="hi",
        transcript=(
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ),
        usage=Usage(input_tokens=3, output_tokens=2, cost_usd=0.01),
        stop_reason="end_turn",
    )
    path = sink.dump(
        invocation=inv, result=result,
        started_iso="2026-05-18T12:00:00+00:00", duration_s=0.5,
    )
    assert path is not None and path.is_file()
    assert path.name == "sdk_0000_claude.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "crucible.sdk-debug.v2"
    assert payload["parse_authority"] == "diagnostic_only"
    assert payload["authority"] == {
        "sdk_debug_payload": "diagnostic_only",
        "workflow_parse": "typed_workflow_records",
    }
    assert payload["backend"] == "claude"
    assert payload["seq"] == 0
    assert payload["invocation"]["system"] == "be terse"
    assert payload["invocation"]["user"] == "hello"
    assert payload["result"]["final_message"] == "hi"
    assert payload["result"]["usage"]["cost_usd"] == 0.01
    assert payload["result"]["pricing_evidence"] == {
        "segments": [],
        "complete": False,
    }
    assert payload["result"]["error"] is None
    assert payload["traceback"] is None
    assert payload["re_raised"] is False
    assert [item["json_pointer"] for item in payload["text_captures"]] == [
        "/invocation/system",
        "/invocation/user",
        "/result/final_message",
        "/result/transcript/0/content",
        "/result/transcript/1/content",
    ]
    for capture in payload["text_captures"]:
        assert capture["original_chars"] == (
            capture["retained_chars"] + capture["omitted_chars"]
        )
        assert not capture["truncated"]
        assert capture["redacted_sha256"].startswith("sha256:")
    artifact_path, _, artifact = _load_result_sidecar(tmp_path, payload)
    assert artifact["schema_version"] == (
        "crucible.sdk-debug-result-artifact.v2"
    )
    assert artifact["parse_authority"] == "diagnostic_only"
    assert artifact["authority"] == payload["authority"]
    assert artifact["result"] == payload["result"]
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(artifact_path.parent.stat().st_mode) == 0o700


def test_sink_projects_complete_pricing_evidence(tmp_path: Path):
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="codex")
    usage = Usage(
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.25,
        cache_creation_tokens=3,
        cache_read_tokens=4,
    )
    result = InvocationResult(
        final_message="ok",
        transcript=(),
        usage=usage,
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
        ),
        pricing_evidence=UsagePricingEvidence(
            segments=(UsagePricingSegment(
                input_tokens=10,
                output_tokens=2,
                cache_creation_tokens=3,
                cache_read_tokens=4,
                cost_usd=0.25,
            ),),
            complete=True,
        ),
    )

    path = sink.dump(
        invocation=Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=1,
            budget_usd=None,
        ),
        result=result,
        started_iso="2026-08-03T12:00:00+00:00",
        duration_s=0.1,
    )

    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["result"]["usage"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "cost_usd": 0.25,
        "cache_creation_tokens": 3,
        "cache_read_tokens": 4,
        "cost_bounds": None,
    }
    assert payload["result"]["pricing_evidence"] == {
        "segments": [{
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_creation_tokens": 3,
            "cache_read_tokens": 4,
            "cost_usd": 0.25,
        }],
        "complete": True,
    }


def test_pricing_projection_treats_legacy_absence_as_incomplete():
    legacy_result = object()

    assert sdk_debug._serialize_pricing_evidence(legacy_result) == {
        "segments": [],
        "complete": False,
    }


def test_sink_writes_traceback_on_failure(tmp_path: Path):
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="claude")
    inv = Invocation(system="s", user="u", tools=None,
                     max_turns=5, budget_usd=None)
    tb = "Traceback (most recent call last):\n  File ..."
    path = sink.dump(
        invocation=inv, result=None,
        started_iso="2026-05-18T12:00:00+00:00", duration_s=0.05,
        traceback_str=tb,
        re_raised=True,
    )
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["result"] is None
    assert payload["full_redacted_result_ref"] is None
    assert payload["full_redacted_result_path"] is None
    assert [item["json_pointer"] for item in payload["text_captures"]] == [
        "/invocation/system",
        "/invocation/user",
    ]
    assert not (tmp_path / "sdk_artifacts").exists()
    assert payload["traceback"] == tb
    assert payload["re_raised"] is True


def test_sink_serializes_error_details(tmp_path: Path):
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="claude")
    inv = Invocation(system="s", user="u", tools=None,
                     max_turns=5, budget_usd=None)
    err = TransportError(http_status=503, message="upstream timeout",
                         retriable=True)
    result = InvocationResult(
        final_message="", transcript=(), usage=Usage(),
        stop_reason="error", error=err,
    )
    path = sink.dump(
        invocation=inv, result=result,
        started_iso="2026-05-18T12:00:00+00:00", duration_s=0.05,
    )
    payload = json.loads(path.read_text())
    assert payload["result"]["error"] == {
        "kind": "transport",
        "message": "upstream timeout",
        "retriable": True,
        "http_status": 503,
    }


def test_sink_seq_counter_is_monotonic(tmp_path: Path):
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="claude")
    inv = Invocation(system="s", user="u", tools=None,
                     max_turns=5, budget_usd=None)
    paths = []
    for _ in range(3):
        paths.append(sink.dump(
            invocation=inv, result=None,
            started_iso="2026-05-18T12:00:00+00:00", duration_s=0.0,
        ))
    assert [p.name for p in paths] == [
        "sdk_0000_claude.json",
        "sdk_0001_claude.json",
        "sdk_0002_claude.json",
    ]


def test_concurrent_independent_sinks_never_overwrite_records(tmp_path: Path):
    """Sequence ownership is directory-scoped, not sink-instance scoped."""

    workers = 16
    barrier = threading.Barrier(workers)

    def dump(index: int) -> Path | None:
        sink = SdkDebugSink(log_dir=tmp_path, backend_name="codex")
        barrier.wait()
        return sink.dump(
            invocation=Invocation(
                system=f"worker-{index}",
                user="u",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            result=None,
            started_iso="2026-05-18T12:00:00+00:00",
            duration_s=0.0,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        paths = list(executor.map(dump, range(workers)))

    assert all(path is not None for path in paths)
    concrete = [path for path in paths if path is not None]
    assert len({path.name for path in concrete}) == workers
    assert {path.name for path in concrete} == {
        f"sdk_{index:04d}_codex.json" for index in range(workers)
    }
    assert {
        json.loads(path.read_text())["invocation"]["system"]
        for path in concrete
    } == {f"worker-{index}" for index in range(workers)}


def test_sink_truncates_long_prompts(tmp_path: Path):
    """User-prompt longer than the per-field cap is truncated with marker."""
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="claude")
    huge_user = "🙂" * 50_000
    inv = Invocation(system="s", user=huge_user, tools=None,
                     max_turns=5, budget_usd=None)
    path = sink.dump(
        invocation=inv, result=None,
        started_iso="2026-05-18T12:00:00+00:00", duration_s=0.0,
    )
    payload = json.loads(path.read_text())
    assert "truncated" in payload["invocation"]["user"]
    assert len(payload["invocation"]["user"]) < len(huge_user)
    capture = next(
        item
        for item in payload["text_captures"]
        if item["json_pointer"] == "/invocation/user"
    )
    assert capture == {
        "json_pointer": "/invocation/user",
        "truncated": True,
        "original_chars": len(huge_user),
        "retained_chars": 16_000,
        "omitted_chars": len(huge_user) - 16_000,
        "redacted_sha256": (
            "sha256:" + hashlib.sha256(huge_user.encode()).hexdigest()
        ),
    }


def test_sink_retains_complete_redacted_result_in_content_addressed_sidecar(
    tmp_path: Path,
):
    huge_final = json.dumps({"answer": "x" * 20_000})
    result = InvocationResult(
        final_message=huge_final,
        transcript=(Message(role="assistant", content="y" * 8_000),),
        usage=Usage(input_tokens=11, output_tokens=12, cost_usd=0.13),
        stop_reason="end_turn",
    )
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="codex")
    path = sink.dump(
        invocation=Invocation(
            system="system",
            user="user",
            tools=None,
            max_turns=1,
            budget_usd=None,
        ),
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=1.0,
    )

    payload = json.loads(path.read_text())
    assert "...[truncated" in payload["result"]["final_message"]
    capture = next(
        item
        for item in payload["text_captures"]
        if item["json_pointer"] == "/result/final_message"
    )
    assert capture["truncated"] is True
    assert capture["original_chars"] == len(huge_final)
    assert capture["retained_chars"] == 8_000
    assert capture["omitted_chars"] == len(huge_final) - 8_000
    assert capture["redacted_sha256"] == (
        "sha256:" + hashlib.sha256(huge_final.encode()).hexdigest()
    )
    artifact_path, _, artifact = _load_result_sidecar(tmp_path, payload)
    assert artifact["result"]["final_message"] == huge_final
    assert artifact["result"]["transcript"][0]["content"] == "y" * 8_000
    assert artifact["result"]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 12,
        "cost_usd": 0.13,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_bounds": None,
    }
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600


def test_sink_preserves_unknown_cache_read_bounds_without_exact_accounting(tmp_path: Path):
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="kimi")
    segment = UsageCostBoundSegment(
        input_tokens=10, output_tokens=2, cache_creation_tokens=3,
        cache_read_tokens=None, lower_bound_usd=0.1, upper_bound_usd=0.25,
        policy="test-diagnostic-policy",
    )
    result = InvocationResult(
        final_message="diagnostic", transcript=(), stop_reason="end_turn",
        usage=Usage(
            input_tokens=10, output_tokens=2, cache_creation_tokens=3,
            cache_read_tokens=0, cost_usd=0.1,
            cost_bounds=UsageCostBounds(segments=(segment,), coverage_complete=True),
        ),
        completeness=ExecutionCompleteness(
            cost_complete=False, token_usage_complete=False,
            terminal_complete=True, trace_complete=False,
        ),
    )
    path = sink.dump(
        invocation=Invocation(system="s", user="u", tools=None, max_turns=1, budget_usd=1),
        result=result, started_iso="2026-09-25T12:00:00+00:00", duration_s=0.1,
    )
    assert path is not None
    payload = json.loads(path.read_text())
    _, _, sidecar = _load_result_sidecar(tmp_path, payload)
    for document in (payload, sidecar):
        usage = document["result"]["usage"]
        assert usage["cost_bounds"] == {
            "segments": [{
                "input_tokens": 10, "output_tokens": 2, "cache_creation_tokens": 3,
                "cache_read_tokens": None, "lower_bound_usd": 0.1,
                "upper_bound_usd": 0.25, "policy": "test-diagnostic-policy",
            }],
            "coverage_complete": True,
        }
        assert usage["cache_read_tokens"] == 0
        assert usage["cost_usd"] == 0.1
        assert document["result"]["pricing_evidence"] == {"segments": [], "complete": False}
        assert document["parse_authority"] == "diagnostic_only"


def test_sink_redacts_credentials_and_writes_private_file(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv("EXAMPLE_API_TOKEN", "super-secret-token-value")
    monkeypatch.setenv("OVERLAP_SHORT_TOKEN", "overlapA")
    monkeypatch.setenv("OVERLAP_LONG_TOKEN", "overlapA-unique-tail")
    sink = SdkDebugSink(log_dir=tmp_path, backend_name="claude")
    inv = Invocation(
        system="Authorization: Bearer abc.def.ghi",
        user="api_key=super-secret-token-value",
        tools=None,
        max_turns=1,
        budget_usd=None,
    )
    result = InvocationResult(
        final_message=("z" * 7_992) + "super-secret-token-value",
        transcript=(Message(
            role="assistant",
            content=(
                "Authorization: Bearer abc.def.ghi; observed "
                "overlapA-unique-tail"
            ),
        ),),
        usage=Usage(),
        stop_reason="error",
        error=TransportError(
            message="worker saw super-secret-token-value",
            retriable=False,
        ),
    )
    path = sink.dump(
        invocation=inv,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.0,
    )
    body = path.read_text()
    assert "abc.def.ghi" not in body
    assert "super-secret-token-value" not in body
    assert "super-se" not in body
    assert "overlapA" not in body
    assert "unique-tail" not in body
    assert "[REDACTED]" in body
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(body)
    final_capture = next(
        item
        for item in payload["text_captures"]
        if item["json_pointer"] == "/result/final_message"
    )
    full_redacted_final = ("z" * 7_992) + "[REDACTED]"
    assert final_capture["original_chars"] == len(full_redacted_final)
    assert final_capture["redacted_sha256"] == (
        "sha256:"
        + hashlib.sha256(full_redacted_final.encode()).hexdigest()
    )
    artifact_path, artifact_body, _ = _load_result_sidecar(tmp_path, payload)
    assert b"abc.def.ghi" not in artifact_body
    assert b"super-secret-token-value" not in artifact_body
    assert b"overlapA" not in artifact_body
    assert b"unique-tail" not in artifact_body
    assert b"[REDACTED]" in artifact_body
    assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o600


def test_exact_env_redaction_precedes_heuristics_and_unions_overlaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    delimiter_secret = "cui04abc,cui04tail"
    overlap_left = "cui04-left-OVERLAP"
    overlap_right = "OVERLAP-right-cui04"
    overlap_joined = "cui04-left-OVERLAP-right-cui04"
    monkeypatch.setenv("CUI04_DELIMITER_API_KEY", delimiter_secret)
    monkeypatch.setenv("CUI04_OVERLAP_LEFT_TOKEN", overlap_left)
    monkeypatch.setenv("CUI04_OVERLAP_RIGHT_TOKEN", overlap_right)
    result = InvocationResult(
        final_message=(
            f"api_key={delimiter_secret}; observed={overlap_joined}"
        ),
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )

    path = SdkDebugSink(tmp_path, "codex").dump(
        invocation=Invocation(
            system="system",
            user=f"api_key={delimiter_secret} {overlap_joined}",
            tools=None,
            max_turns=1,
            budget_usd=None,
        ),
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )

    assert path is not None
    body = path.read_text()
    payload = json.loads(body)
    _, artifact_body, _ = _load_result_sidecar(tmp_path, payload)
    for forbidden in (
        delimiter_secret,
        "cui04abc",
        "cui04tail",
        overlap_joined,
        "cui04-left",
        "OVERLAP",
        "right-cui04",
    ):
        assert forbidden not in body
        assert forbidden.encode() not in artifact_body
    assert "api_key=[REDACTED]" in body
    assert "observed=[REDACTED]" in body


def test_lone_unicode_surrogates_are_safe_diagnostic_evidence(tmp_path: Path):
    invalid = "before\ud800after"
    result = InvocationResult(
        final_message=invalid,
        transcript=(Message(role="assistant", content="trace\udfffend"),),
        usage=Usage(),
        stop_reason="end_turn",
        error=TransportError(message="error\ud800detail", retriable=False),
    )

    path = SdkDebugSink(tmp_path, "claude").dump(
        invocation=Invocation(
            system="system\ud800prompt",
            user="user\udfffprompt",
            tools=None,
            max_turns=1,
            budget_usd=None,
        ),
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
        traceback_str="traceback\ud800detail",
    )

    assert path is not None
    body = path.read_bytes()
    body.decode("utf-8", errors="strict")
    payload = json.loads(body)
    replacement = "\N{REPLACEMENT CHARACTER}"
    assert payload["invocation"]["system"] == f"system{replacement}prompt"
    assert payload["invocation"]["user"] == f"user{replacement}prompt"
    assert payload["result"]["final_message"] == (
        f"before{replacement}after"
    )
    assert payload["result"]["transcript"][0]["content"] == (
        f"trace{replacement}end"
    )
    assert payload["result"]["error"]["message"] == (
        f"error{replacement}detail"
    )
    assert payload["traceback"] == f"traceback{replacement}detail"
    final_capture = next(
        item
        for item in payload["text_captures"]
        if item["json_pointer"] == "/result/final_message"
    )
    safe_final = f"before{replacement}after"
    assert final_capture["original_chars"] == len(safe_final)
    assert final_capture["redacted_sha256"] == (
        "sha256:" + hashlib.sha256(safe_final.encode()).hexdigest()
    )
    _, artifact_body, artifact = _load_result_sidecar(tmp_path, payload)
    artifact_body.decode("utf-8", errors="strict")
    assert artifact["result"]["final_message"] == safe_final
    # Diagnostic normalization never mutates the typed authoritative result.
    assert result.final_message == invalid


def test_non_oserror_diagnostic_failure_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    result = InvocationResult(
        final_message="authoritative result",
        transcript=(),
        usage=Usage(),
        stop_reason="end_turn",
    )

    def fail_diagnostic(_payload):
        raise ValueError("diagnostic-only failure")

    monkeypatch.setattr(sdk_debug, "_truncate_payload", fail_diagnostic)
    with caplog.at_level("WARNING"):
        path = SdkDebugSink(tmp_path, "codex").dump(
            invocation=Invocation(
                system="system",
                user="user",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            result=result,
            started_iso="2026-05-18T12:00:00+00:00",
            duration_s=0.1,
        )

    assert path is None
    assert result.final_message == "authoritative result"
    assert any("diagnostic-only failure" in record.message for record in caplog.records)


def test_diagnostic_warning_redacts_exception_and_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    detail_secret = "cui04-warning-detail-secret"
    path_secret = "cui04-warning-path-secret"
    monkeypatch.setenv("CUI04_WARNING_DETAIL_TOKEN", detail_secret)
    monkeypatch.setenv("CUI04_WARNING_PATH_TOKEN", path_secret)

    def fail_diagnostic(_payload):
        raise ValueError(f"diagnostic failure exposed {detail_secret}")

    monkeypatch.setattr(sdk_debug, "_truncate_payload", fail_diagnostic)
    log_dir = tmp_path / path_secret
    with caplog.at_level("WARNING"):
        path = SdkDebugSink(log_dir, "codex").dump(
            invocation=Invocation(
                system="system",
                user="user",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            result=None,
            started_iso="2026-05-18T12:00:00+00:00",
            duration_s=0.1,
        )

    assert path is None
    warning = "\n".join(record.message for record in caplog.records)
    assert detail_secret not in warning
    assert path_secret not in warning
    assert warning.count("[REDACTED]") >= 2


def test_concurrent_identical_results_deduplicate_sidecar(tmp_path: Path):
    workers = 16
    barrier = threading.Barrier(workers)
    result = InvocationResult(
        final_message="same result",
        transcript=(Message(role="assistant", content="same result"),),
        usage=Usage(input_tokens=1, output_tokens=2, cost_usd=0.03),
        stop_reason="end_turn",
    )

    def dump(_index: int) -> Path | None:
        sink = SdkDebugSink(log_dir=tmp_path, backend_name="kimi")
        barrier.wait()
        return sink.dump(
            invocation=Invocation(
                system="system",
                user="user",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            result=result,
            started_iso="2026-05-18T12:00:00+00:00",
            duration_s=0.1,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        paths = list(executor.map(dump, range(workers)))

    assert all(path is not None for path in paths)
    payloads = [json.loads(path.read_text()) for path in paths if path]
    assert len({item["full_redacted_result_ref"] for item in payloads}) == 1
    assert len(list((tmp_path / "sdk_artifacts").glob("*.json"))) == 1


def test_distinct_concurrent_results_publish_distinct_resolvable_sidecars(
    tmp_path: Path,
):
    workers = 8
    barrier = threading.Barrier(workers)

    def dump(index: int) -> Path | None:
        sink = SdkDebugSink(log_dir=tmp_path, backend_name="codex")
        barrier.wait()
        return sink.dump(
            invocation=Invocation(
                system="system",
                user=f"user-{index}",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            result=InvocationResult(
                final_message=f"result-{index}",
                transcript=(),
                usage=Usage(input_tokens=index, output_tokens=1),
                stop_reason="end_turn",
            ),
            started_iso="2026-05-18T12:00:00+00:00",
            duration_s=0.1,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        paths = list(executor.map(dump, range(workers)))

    payloads = [json.loads(path.read_text()) for path in paths if path]
    assert len(payloads) == workers
    assert len({item["full_redacted_result_ref"] for item in payloads}) == workers
    assert {
        _load_result_sidecar(tmp_path, item)[2]["result"]["final_message"]
        for item in payloads
    } == {f"result-{index}" for index in range(workers)}


def test_corrupt_existing_sidecar_is_preserved_and_dump_fails_safely(
    tmp_path: Path,
):
    invocation = Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=1,
        budget_usd=None,
    )
    result = InvocationResult(
        final_message="result",
        transcript=(),
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    first = SdkDebugSink(tmp_path, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )
    payload = json.loads(first.read_text())
    sidecar = tmp_path / payload["full_redacted_result_path"]
    sidecar.write_bytes(b"corrupt-existing-bytes")

    second = SdkDebugSink(tmp_path, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )

    assert second is None
    assert sidecar.read_bytes() == b"corrupt-existing-bytes"
    assert len(list(tmp_path.glob("sdk_*_claude.json"))) == 1


def _matching_sidecar_collision_material(
    tmp_path: Path,
) -> tuple[Invocation, InvocationResult, str, bytes]:
    invocation = Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=1,
        budget_usd=None,
    )
    result = InvocationResult(
        final_message="hostile collision result",
        transcript=(),
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    source_root = tmp_path / "source"
    first = SdkDebugSink(source_root, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )
    assert first is not None
    payload = json.loads(first.read_text())
    sidecar = source_root / payload["full_redacted_result_path"]
    return invocation, result, sidecar.name, sidecar.read_bytes()


def test_matching_sidecar_symlink_is_not_followed_or_chmodded(
    tmp_path: Path,
):
    invocation, result, sidecar_name, sidecar_body = (
        _matching_sidecar_collision_material(tmp_path)
    )
    attacked_root = tmp_path / "symlink-attack"
    artifact_dir = attacked_root / "sdk_artifacts"
    artifact_dir.mkdir(parents=True)
    victim = tmp_path / "matching-victim.json"
    victim.write_bytes(sidecar_body)
    victim.chmod(0o644)
    hostile_path = artifact_dir / sidecar_name
    hostile_path.symlink_to(victim)

    path = SdkDebugSink(attacked_root, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )

    assert path is None
    assert hostile_path.is_symlink()
    assert victim.read_bytes() == sidecar_body
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert list(attacked_root.glob("sdk_*_claude.json")) == []


@pytest.mark.timeout(5)
def test_matching_sidecar_fifo_fails_without_blocking(tmp_path: Path):
    invocation, result, sidecar_name, _sidecar_body = (
        _matching_sidecar_collision_material(tmp_path)
    )
    attacked_root = tmp_path / "fifo-attack"
    artifact_dir = attacked_root / "sdk_artifacts"
    artifact_dir.mkdir(parents=True)
    hostile_path = artifact_dir / sidecar_name
    os.mkfifo(hostile_path)

    started = time.monotonic()
    path = SdkDebugSink(attacked_root, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )
    elapsed = time.monotonic() - started

    assert path is None
    assert elapsed < 1.0
    assert stat.S_ISFIFO(hostile_path.lstat().st_mode)
    assert list(attacked_root.glob("sdk_*_claude.json")) == []


def test_matching_hard_link_sidecar_is_not_accepted_or_chmodded(
    tmp_path: Path,
):
    invocation, result, sidecar_name, sidecar_body = (
        _matching_sidecar_collision_material(tmp_path)
    )
    attacked_root = tmp_path / "hard-link-attack"
    artifact_dir = attacked_root / "sdk_artifacts"
    artifact_dir.mkdir(parents=True)
    victim = tmp_path / "hard-linked-victim.json"
    victim.write_bytes(sidecar_body)
    victim.chmod(0o644)
    hostile_path = artifact_dir / sidecar_name
    os.link(victim, hostile_path)
    assert victim.stat().st_nlink == 2

    path = SdkDebugSink(attacked_root, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )

    assert path is None
    assert victim.read_bytes() == sidecar_body
    assert victim.stat().st_nlink == 2
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert list(attacked_root.glob("sdk_*_claude.json")) == []


def test_artifact_directory_symlink_swap_cannot_escape_verified_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    invocation, result, sidecar_name, sidecar_body = (
        _matching_sidecar_collision_material(tmp_path)
    )
    attacked_root = tmp_path / "parent-swap-attack"
    attacked_root.mkdir()
    artifact_dir = attacked_root / "sdk_artifacts"
    displaced_dir = attacked_root / "verified-artifacts-displaced"
    external_dir = tmp_path / "external-victim-dir"
    external_dir.mkdir()
    external_victim = external_dir / sidecar_name
    external_victim.write_bytes(sidecar_body)
    external_victim.chmod(0o644)
    original_write = sdk_debug._write_private_atomic_at
    swapped = False

    def swap_parent_then_write(
        directory_fd: int,
        name: str,
        payload: bytes,
    ) -> None:
        nonlocal swapped
        if not swapped:
            artifact_dir.rename(displaced_dir)
            artifact_dir.symlink_to(external_dir, target_is_directory=True)
            swapped = True
        original_write(directory_fd, name, payload)

    monkeypatch.setattr(
        sdk_debug,
        "_write_private_atomic_at",
        swap_parent_then_write,
    )

    path = SdkDebugSink(attacked_root, "claude").dump(
        invocation=invocation,
        result=result,
        started_iso="2026-05-18T12:00:00+00:00",
        duration_s=0.1,
    )

    assert path is None
    assert swapped
    assert artifact_dir.is_symlink()
    assert external_victim.read_bytes() == sidecar_body
    assert stat.S_IMODE(external_victim.stat().st_mode) == 0o644
    assert (displaced_dir / sidecar_name).read_bytes() == sidecar_body
    assert list(attacked_root.glob("sdk_*_claude.json")) == []


def test_sink_survives_unwriteable_log_dir(tmp_path: Path, caplog):
    """A bogus log_dir doesn't break the workflow — sink logs a warning."""
    # A directory cannot be created beneath a regular file for any uid; a
    # made-up absolute root can be created by root.
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    bad = blocker / "logs"
    sink = SdkDebugSink(log_dir=bad, backend_name="claude")
    inv = Invocation(system="s", user="u", tools=None,
                     max_turns=5, budget_usd=None)
    result = InvocationResult(
        final_message="", transcript=(), usage=Usage(),
        stop_reason="end_turn",
    )
    with caplog.at_level("WARNING"):
        path = sink.dump(
            invocation=inv, result=result,
            started_iso="2026-05-18T12:00:00+00:00", duration_s=0.0,
        )
    assert path is None
    assert not bad.exists()
    # Warning was emitted; workflow proceeds.
    assert any(
        "SdkDebugSink" in rec.message for rec in caplog.records
    )


def test_mkdir_failure_warning_redacts_secret_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    secret = "cui04-mkdir-warning-secret"
    monkeypatch.setenv("CUI04_MKDIR_WARNING_TOKEN", secret)
    bad = Path("/dev/null") / secret / "logs"
    sink = SdkDebugSink(log_dir=bad, backend_name="kimi")

    with caplog.at_level("WARNING"):
        path = sink.dump(
            invocation=Invocation(
                system="system",
                user="user",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            result=None,
            started_iso="2026-05-18T12:00:00+00:00",
            duration_s=0.0,
        )

    assert path is None
    warning = "\n".join(record.message for record in caplog.records)
    assert secret not in warning
    assert "/dev/null/[REDACTED]/logs" in warning


# ----------------------------------------------------------------------
# Redaction of the lines of one bounded text
# ----------------------------------------------------------------------

_LINE_WITHHELD = "[REDACTED: line withheld]"


def _made_up_value(tag: str, length: int = 20) -> str:
    """Return a value made at run time, so that no file holds it."""
    digest = hashlib.sha256(f"redaction test value {tag}".encode())
    return digest.hexdigest()[:length]


def _readable(value: str, text: str) -> bool:
    """Whether any six characters in a row of ``value`` stand in ``text``."""
    return any(
        value[start : start + 6] in text for start in range(len(value) - 5)
    )


def _as_the_sdk_frames(text: str) -> list[str]:
    """Split ``text`` as the reviewed SDK frames a CLI's stderr."""
    return [line.rstrip() for line in text.split("\n") if line.rstrip()]


def test_redact_sdk_text_knows_environment_values_of_eight_characters_up(
    monkeypatch: pytest.MonkeyPatch,
):
    """The exact values are those of variables with a sensitive name."""
    long_enough = _made_up_value("eight characters", 8)
    too_short = _made_up_value("seven characters", 7)
    other_name = _made_up_value("another name", 20)
    monkeypatch.setenv("CRUCIBLE_TEST_LINES_API_KEY", long_enough)
    monkeypatch.setenv("CRUCIBLE_TEST_LINES_SHORT_TOKEN", too_short)
    monkeypatch.setenv("CRUCIBLE_TEST_LINES_SETTING", other_name)

    redacted = sdk_debug.redact_sdk_text(
        f"saw {long_enough} and {too_short} and {other_name}"
    )

    assert redacted == f"saw [REDACTED] and {too_short} and {other_name}"


def test_redact_sdk_lines_withholds_the_lines_its_passes_disagree_on():
    """A line is returned only where every pass holds that same line.

    Each line is redacted on its own, each pair of neighbouring lines as
    one text, and all lines joined as one text.
    """
    value = _made_up_value("lines the passes disagree on")

    # By position: the joined text keeps its line breaks, and the value
    # line differs from what its own redaction returns.
    assert sdk_debug.redact_sdk_lines(["token:", f"  {value}", "  done"]) == [
        "token:",
        _LINE_WITHHELD,
        "  done",
    ]
    # By the pairs: in the joined text the outer name takes the inner name
    # as its value, so only the pair of the last two lines covers the value.
    assert sdk_debug.redact_sdk_lines(
        ["auth:", "  token:", f"    {value}"]
    ) == ["auth:", _LINE_WITHHELD, _LINE_WITHHELD]
    # By alignment: the redaction of the joined text removed a line break,
    # and the lines around the two withheld ones stay in their places.
    assert sdk_debug.redact_sdk_lines(
        ["one", "two", "Authorization: Bearer", f" {value}", "five"]
    ) == ["one", "two", _LINE_WITHHELD, _LINE_WITHHELD, "five"]
    # A joined pass over the lines as redacted on their own would print
    # the token here: its first line no longer holds the scheme word.
    assert sdk_debug.redact_sdk_lines(
        ["Authorization: Bearer", f" {value}"]
    ) == [_LINE_WITHHELD, _LINE_WITHHELD]
    assert sdk_debug.redact_sdk_lines([]) == []


def test_redact_sdk_lines_withholds_a_framed_line_of_an_environment_value(
    monkeypatch: pytest.MonkeyPatch,
):
    """A line of a sensitive value is withheld wherever it stands.

    Output that was split at line breaks and stripped does not hold a value
    with a line break as the parent holds it. A value without a line break
    and without white space at its ends is found as it stands, and is
    redacted in its place. A line of a value is looked for from 8
    characters on, counted after stripping: a shorter one withholds
    nothing.
    """
    first = _made_up_value("framed value, first line")
    second = _made_up_value("framed value, second line")
    single = _made_up_value("value on one line")
    monkeypatch.setenv("CRUCIBLE_TEST_LINES_TWO_LINE_TOKEN", f"{first}\n{second}")
    monkeypatch.setenv("CRUCIBLE_TEST_LINES_ONE_LINE_TOKEN", single)

    # Also when the other line of the value is not among the lines.
    assert sdk_debug.redact_sdk_lines([f"got {second} here", "next"]) == [
        _LINE_WITHHELD,
        "next",
    ]
    assert sdk_debug.redact_sdk_lines(
        [f"value is {first}", f"{second} end"]
    ) == [_LINE_WITHHELD, _LINE_WITHHELD]
    assert sdk_debug.redact_sdk_lines([f"value is {single}", "next"]) == [
        "value is [REDACTED]",
        "next",
    ]

    # The shortest line that is looked for has exactly 8 characters. The
    # other line of this value has 8 as the parent holds it and 7 after
    # stripping: the line that holds it is returned as redacted on its own.
    eight = _made_up_value("framed value, a line of eight characters", 8)
    seven = _made_up_value("framed value, a line of seven characters", 7)
    monkeypatch.setenv(
        "CRUCIBLE_TEST_LINES_SHORT_LINES_TOKEN",
        f"{eight}\n {seven}",
    )
    assert sdk_debug.redact_sdk_lines([f"got {eight} here", "next"]) == [
        _LINE_WITHHELD,
        "next",
    ]
    assert sdk_debug.redact_sdk_lines([f"request {seven} done", "next"]) == [
        f"request {seven} done",
        "next",
    ]


def test_redact_sdk_lines_two_checks_also_withhold_plain_lines(
    monkeypatch: pytest.MonkeyPatch,
):
    """What the pair check and the framed lines cost, as known behavior.

    A plain line after a name whose own line follows another name is
    withheld. So is every line that holds the stripped form of a one-line
    value held with white space at an end, secret or not.
    """
    edged = _made_up_value("value held with a blank at its end")
    monkeypatch.setenv("CRUCIBLE_TEST_LINES_EDGED_TOKEN", f"{edged} ")

    assert sdk_debug.redact_sdk_lines(
        ["auth:", "  token:", "plain line here"]
    ) == ["auth:", _LINE_WITHHELD, _LINE_WITHHELD]
    assert sdk_debug.redact_sdk_lines([f"rejected {edged} (401)"]) == [
        _LINE_WITHHELD
    ]


def _texts_of_lines_with_their_values() -> list[tuple[list[str], list[str]]]:
    """Texts of one to several lines, each with the made-up values it holds."""
    value = _made_up_value("texts of lines")
    first = _made_up_value("texts of lines, first half")
    second = _made_up_value("texts of lines, second half")
    texts: list[tuple[list[str], list[str]]] = [
        (["upstream said token:", f"  {value}"], [value]),
        (["Authorization: Bearer", f" {value}"], [value]),
        ([f'password = "{first}', f'{second}"'], [first, second]),
        ([f"upstream said token: {value}"], [value]),
        ([f"Authorization: Bearer {value}"], [value]),
        (["auth:", f"  token: {value}"], [value]),
        (["secret:", f"  password: {value}", "  user: bob"], [value]),
        (["expected scheme Bearer", f"token: {value}"], [value]),
        (['password = "abc', "more text", f'password = "{value}"'], [value]),
        (["auth:", "  token:", f"    {value}"], [value]),
        (["retry with secret:", "No conversation found", "exit"], []),
        (["a: b", "c = d", "[e]"], []),
        (["only line"], []),
    ]
    # A line break at every position of four one-line shapes.
    for shape, values in (
        (f"upstream said token: {value} ok", [value]),
        (f"api_key={value};next", [value]),
        (f"Authorization: Bearer {value} ok", [value]),
        (f'password = "{first} {second}" ok', [first, second]),
    ):
        for position in range(1, len(shape)):
            lines = _as_the_sdk_frames(
                shape[:position] + "\n" + shape[position:]
            )
            texts.append((lines, values))
    return texts


def test_redact_sdk_lines_returns_each_line_alone_or_the_marker():
    """No line is returned in another form than its own redaction.

    So the function shows nothing that redacting each line on its own
    hides. It shows nothing that redacting the joined lines hides either,
    because a line is returned only where the joined result holds it.
    """
    texts = _texts_of_lines_with_their_values()
    assert len(texts) > 100

    for lines, values in texts:
        alone = [sdk_debug.redact_sdk_text(line) for line in lines]
        together = sdk_debug.redact_sdk_text("\n".join(lines))

        returned = sdk_debug.redact_sdk_lines(lines)

        assert len(returned) == len(lines)
        for item, line_alone in zip(returned, alone, strict=True):
            assert item in (line_alone, _LINE_WITHHELD)
        for value in values:
            if _readable(value, "\n".join(returned)):
                assert _readable(value, "\n".join(alone))
                assert _readable(value, together)

    # Where every pass agrees, the lines are returned as redacted alone.
    for plain in (["a: b", "c = d", "[e]"], ["only line"]):
        assert sdk_debug.redact_sdk_lines(plain) == plain


@pytest.mark.parametrize(
    "case",
    [
        "name_then_value",
        "name_separator_value_on_three_lines",
        "assignment_of_bearer_then_token",
        "bearer_alone_then_token",
        "name_and_bearer_then_token",
        "name_then_bearer_then_token",
        "other_name_then_bearer_then_token",
        "double_quoted_value_over_two_line_breaks",
        "single_quoted_value_over_a_line_break",
        "value_then_name_then_second_value",
        "value_and_bearer_then_token",
        "environment_value_of_three_lines",
        "bearer_then_bearer_then_token",
        "quoted_value_over_a_line_break_after_the_pairing_text",
    ],
)
def test_redact_sdk_lines_hides_a_value_whose_shape_spans_lines(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    """No part of a value is readable in what is returned.

    In the last two cases the joined text does not redact the value. Only
    the pair of the last two lines does, and its redaction is a single
    line: a pair whose redaction lost its line break withholds both of its
    lines.
    """
    value = _made_up_value(case)
    first = _made_up_value(case + " first part")
    second = _made_up_value(case + " second part")
    third = _made_up_value(case + " third part")
    hidden = [value]
    if case == "name_then_value":
        lines = ["upstream said token:", f"  {value}"]
    elif case == "name_separator_value_on_three_lines":
        lines = ["api_key", "=", value]
    elif case == "assignment_of_bearer_then_token":
        lines = ["authorization=Bearer", value]
    elif case == "bearer_alone_then_token":
        lines = ["Bearer", value]
    elif case == "name_and_bearer_then_token":
        lines = ["auth: Bearer", value]
    elif case == "name_then_bearer_then_token":
        lines = ["Authorization:", "Bearer", value]
    elif case == "other_name_then_bearer_then_token":
        lines = ["password:", "Bearer", value]
    elif case == "double_quoted_value_over_two_line_breaks":
        lines = [f'password = "{first}', second, f'{third}"']
        hidden = [first, second, third]
    elif case == "single_quoted_value_over_a_line_break":
        lines = [f"password = '{first}", f"{second}'"]
        hidden = [first, second]
    elif case == "value_then_name_then_second_value":
        lines = [f"token: {first}", "password:", second]
        hidden = [first, second]
    elif case == "value_and_bearer_then_token":
        lines = [f"token={first} Bearer", second]
        hidden = [first, second]
    elif case == "environment_value_of_three_lines":
        monkeypatch.setenv(
            "CRUCIBLE_TEST_LINES_THREE_LINE_TOKEN",
            f"{first}\n{second}\n{third}",
        )
        lines = [f"value is {first}", second, f"{third} end"]
        hidden = [first, second, third]
    elif case == "bearer_then_bearer_then_token":
        # In the joined text the first scheme word takes the second as its
        # token, and the token line stays a line of its own.
        lines = ["Bearer", "Bearer", value]
    else:
        assert case == "quoted_value_over_a_line_break_after_the_pairing_text"
        # In the joined text the first quote is closed by the second, and
        # the last line stays a line of its own.
        lines = [
            'password = "abc',
            "more text",
            f'password = "{first}',
            f'{second}"',
        ]
        hidden = [first, second]

    returned = sdk_debug.redact_sdk_lines(lines)

    assert len(returned) == len(lines)
    for part in hidden:
        assert not _readable(part, "\n".join(returned))


_LINES_FUNCTION = "redact_sdk_lines"
# Every place of the package where the name stands: the function, the
# adapter's import of it, and the one call, with the entries of the stderr
# window of one attempt.
_USES_OF_THE_LINES_FUNCTION = [
    ("backends/_sdk_debug.py", "definition", "redact_sdk_lines"),
    (
        "backends/claude.py",
        "call",
        "redact_sdk_lines(cli_stderr_window.entries)",
    ),
    ("backends/claude.py", "import", "redact_sdk_lines"),
]


def _uses_of_the_lines_function(root: Path) -> list[tuple[str, str, str]]:
    """Every use of the name ``redact_sdk_lines`` in the Python under ``root``.

    Read is every ``*.py`` file outside the directories named ``tests`` or
    ``__pycache__`` and those whose name begins with a dot. A use is a
    definition by the name, an import of it (as written, so with the name
    it is bound to), a call of it (as written, so with its arguments), and
    every other place where it stands as a name, as an attribute or as a
    whole string: the function handed over as a value, bound to another
    name, or looked up by its name. Comments and longer strings are not
    uses, and a name put together at run time is not seen.
    """
    uses: list[tuple[str, str, str]] = []
    for current, directories, files in os.walk(root):
        directories[:] = [
            name
            for name in directories
            if not name.startswith(".") and name not in {"__pycache__", "tests"}
        ]
        for file_name in files:
            if not file_name.endswith(".py"):
                continue
            path = Path(current) / file_name
            source = path.read_bytes()
            if _LINES_FUNCTION.encode() not in source:
                continue
            where = path.relative_to(root).as_posix()
            tree = ast.parse(source)
            calls = {
                id(node.func): node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
            }
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    if node.name == _LINES_FUNCTION:
                        uses.append((where, "definition", node.name))
                elif isinstance(node, ast.alias):
                    imported = node.name.rpartition(".")[2]
                    if _LINES_FUNCTION in (imported, node.asname):
                        uses.append((where, "import", ast.unparse(node)))
                elif (
                    (isinstance(node, ast.Name) and node.id == _LINES_FUNCTION)
                    or (
                        isinstance(node, ast.Attribute)
                        and node.attr == _LINES_FUNCTION
                    )
                    or (
                        isinstance(node, ast.Constant)
                        and node.value == _LINES_FUNCTION
                    )
                ):
                    call = calls.get(id(node))
                    if call is None:
                        uses.append((where, "value", ast.unparse(node)))
                    else:
                        uses.append((where, "call", ast.unparse(call)))
    return sorted(uses)


def test_redact_sdk_lines_has_one_caller_which_hands_it_a_bounded_text():
    """``redact_sdk_lines`` has no bound of its own, and one caller.

    It reads every line four times, so its caller has to bound the text.
    The one caller is the Claude adapter's handler of a failed attempt,
    which hands it the entries of that attempt's stderr window and nothing
    else. A second caller, another argument, the function under another
    name or handed over as a value needs a bound of its own and fails
    here, in whichever module of the package it stands.
    """
    import crucible

    root = Path(crucible.__file__).resolve().parent

    assert _uses_of_the_lines_function(root) == _USES_OF_THE_LINES_FUNCTION


_UNREAD_USE = (
    "from crucible.backends._sdk_debug import redact_sdk_lines\n"
    "\n"
    "redact_sdk_lines([])\n"
)
# One more use of the function in a made-up package tree: the file it is
# appended to, its text, and what it adds to the uses found.
_FURTHER_USES_OF_THE_LINES_FUNCTION = {
    "a_call_in_rules": (
        "rules/late.py",
        "from ..backends import _sdk_debug\n"
        "\n"
        "\n"
        "def publish(lines):\n"
        "    return _sdk_debug.redact_sdk_lines(lines)\n",
        [("rules/late.py", "call", "_sdk_debug.redact_sdk_lines(lines)")],
    ),
    "a_call_in_scripts": (
        "scripts/report.py",
        "from crucible.backends._sdk_debug import redact_sdk_lines\n"
        "\n"
        "print(redact_sdk_lines(['a']))\n",
        [
            ("scripts/report.py", "call", "redact_sdk_lines(['a'])"),
            ("scripts/report.py", "import", "redact_sdk_lines"),
        ],
    ),
    "a_call_in_a_module_at_the_root": (
        "orchestration.py",
        "from .backends._sdk_debug import redact_sdk_lines\n"
        "\n"
        "LINES = redact_sdk_lines(())\n",
        [
            ("orchestration.py", "call", "redact_sdk_lines(())"),
            ("orchestration.py", "import", "redact_sdk_lines"),
        ],
    ),
    "a_second_call_in_the_adapter": (
        "backends/claude.py",
        "\n"
        "\n"
        "def publish_again(lines):\n"
        "    return redact_sdk_lines(lines)\n",
        [("backends/claude.py", "call", "redact_sdk_lines(lines)")],
    ),
    "a_call_under_another_name": (
        "backends/other.py",
        "from ._sdk_debug import redact_sdk_lines as lines_of\n"
        "\n"
        "\n"
        "def publish(lines):\n"
        "    return lines_of(lines)\n",
        [("backends/other.py", "import", "redact_sdk_lines as lines_of")],
    ),
    "the_function_handed_over_as_a_value": (
        "backends/claude.py",
        "\n"
        "\n"
        "def publish_each(texts):\n"
        "    return list(map(redact_sdk_lines, texts))\n",
        [("backends/claude.py", "value", "redact_sdk_lines")],
    ),
    "the_function_looked_up_by_its_name": (
        "workflows/report.py",
        "from ..backends import _sdk_debug\n"
        "\n"
        "publish = getattr(_sdk_debug, 'redact_sdk_lines')\n",
        [("workflows/report.py", "value", "'redact_sdk_lines'")],
    ),
}


def _package_with_the_one_caller(root: Path) -> None:
    """Write a made-up package tree that uses the function as the real one.

    Beside the three uses it holds the name in a comment and in a longer
    string, and uses of it where nothing is read: under ``tests`` at two
    depths and in a virtual environment.
    """
    files = {
        "backends/_sdk_debug.py": (
            "def redact_sdk_lines(lines):\n"
            '    """Stand in for ``redact_sdk_lines``."""\n'
            "    return list(lines)\n"
        ),
        "backends/claude.py": (
            "from ._sdk_debug import redact_sdk_lines\n"
            "\n"
            "\n"
            "def publish(cli_stderr_window):\n"
            "    # The kept entries go through ``redact_sdk_lines``.\n"
            "    return redact_sdk_lines(cli_stderr_window.entries)\n"
        ),
        "tests/test_lines.py": _UNREAD_USE,
        "products/demo/tests/test_demo.py": _UNREAD_USE,
        ".venv/lib/site.py": _UNREAD_USE,
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())


@pytest.mark.parametrize("case", sorted(_FURTHER_USES_OF_THE_LINES_FUNCTION))
def test_a_further_use_of_redact_sdk_lines_is_found_wherever_it_stands(
    tmp_path: Path,
    case: str,
):
    """What the test of the one caller reads, shown on a made-up package.

    The tree holds the three uses of the real package. One more use, in a
    module of any directory or at the root, and in any spelling, is found
    with its place and its text.
    """
    relative, text, further = _FURTHER_USES_OF_THE_LINES_FUNCTION[case]
    _package_with_the_one_caller(tmp_path)
    assert _uses_of_the_lines_function(tmp_path) == _USES_OF_THE_LINES_FUNCTION

    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(text.encode())

    assert _uses_of_the_lines_function(tmp_path) == sorted(
        [*_USES_OF_THE_LINES_FUNCTION, *further]
    )


# ----------------------------------------------------------------------
# Integration: ClaudeBackend uses the sink end-to-end
# ----------------------------------------------------------------------


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch):
    """Replace claude_agent_sdk.query with a generator we control."""

    @dataclass
    class _TextBlock:
        text: str

    @dataclass
    class _AssistantMessage:
        content: list
        model: str = "claude-opus-4-8"
        usage: dict | None = None

    @dataclass
    class _ResultMessage:
        usage: dict
        total_cost_usd: float
        stop_reason: str
        is_error: bool = False
        num_turns: int = 1

    @dataclass
    class _SystemMessage:
        subtype: str
        data: dict
        session_id: str | None = None

    def make_query(responses):
        async def query(*, prompt, options):
            for r in responses:
                yield r
        return query

    class _Options:
        def __init__(self, **kw):
            pass

    # Inject a fake claude_agent_sdk module hierarchy.
    import types as _types
    pkg = _types.ModuleType("claude_agent_sdk")
    types_mod = _types.ModuleType("claude_agent_sdk.types")
    types_mod.AssistantMessage = _AssistantMessage
    types_mod.ResultMessage = _ResultMessage
    types_mod.SystemMessage = _SystemMessage
    types_mod.TextBlock = _TextBlock
    pkg.ClaudeAgentOptions = _Options
    pkg.types = types_mod
    pkg.query = make_query  # will be replaced per-test

    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk", pkg)
    monkeypatch.setitem(
        __import__("sys").modules, "claude_agent_sdk.types", types_mod,
    )
    return pkg, _AssistantMessage, _ResultMessage, _TextBlock


def test_claude_backend_writes_sdk_debug_on_success(
    tmp_path: Path, fake_sdk
):
    pkg, AssistantMessage, ResultMessage, TextBlock = fake_sdk
    pkg.query = (lambda *, prompt, options: _ag(
        [AssistantMessage(content=[TextBlock(text="hi there")]),
         ResultMessage(usage={"input_tokens": 4, "output_tokens": 2},
                       total_cost_usd=0.02, stop_reason="end_turn")]
    ))

    from crucible.backends import ClaudeBackend
    backend = ClaudeBackend(
        model="aws/anthropic/bedrock-claude-opus-4-8",
        log_dir=tmp_path,
    )
    result = backend.invoke(Invocation(
        system="be terse", user="hello",
        tools=None, max_turns=5, budget_usd=None,
    ))
    assert result.final_message == "hi there"
    files = sorted(tmp_path.glob("sdk_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["backend"] == "claude"
    assert payload["result"]["final_message"] == "hi there"
    assert payload["result"]["error"] is None


async def _ag(items):
    for x in items:
        yield x
