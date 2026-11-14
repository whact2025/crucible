# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
import signal
import time
from pathlib import Path

import pytest

from crucible.backends import codex_official_worker as transport


def _terminal_payload():
    return {
        "protocol": "crucible.codex-worker.v1",
        "type": "terminal",
        "ok": True,
        "sdk_version": "0.144.4",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "status": "completed",
        "error": None,
        "final_response": "OK",
        "usage": {
            "input_tokens": 3,
            "cached_input_tokens": 0,
            "output_tokens": 1,
            "reasoning_output_tokens": 0,
            "total_tokens": 4,
        },
        "tool_items": [],
    }


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "worker"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_worker_environment_is_closed_and_requires_private_home(
    monkeypatch, tmp_path
):
    home = tmp_path / "codex-home"
    home.mkdir()
    environment = transport.worker_environment(
        codex_home=str(home),
        inherited={
            "HOME": "/operator",
            "LANG": "C.UTF-8",
            "OPENAI_API_KEY": "secret",
            "ANTHROPIC_API_KEY": "secret",
            "HTTP_PROXY": "http://proxy",
        },
    )

    assert environment == {
        "HOME": "/operator",
        "LANG": "C.UTF-8",
        "CODEX_HOME": str(home),
        "PYTHONNOUSERSITE": "1",
    }
    with pytest.raises(
        transport.CodexOfficialSDKQualificationError,
        match="existing absolute",
    ):
        transport.worker_environment(
            codex_home=str(tmp_path / "missing")
        )

    worker = tmp_path / "worker-python"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o700)
    invoked = None

    def capture_probe(argv, **_kwargs):
        nonlocal invoked
        invoked = tuple(argv)

        class Completed:
            returncode = 0
            stdout = json.dumps(
                {
                    "protocol": "crucible.codex-worker.v1",
                    "type": "probe",
                    "ok": True,
                    "sdk_version": "0.144.4",
                    "cli_version": "0.144.4",
                }
            ).encode("utf-8")

        return Completed()

    monkeypatch.setattr(transport.subprocess, "run", capture_probe)
    transport.qualify_worker_runtime.cache_clear()
    try:
        transport.qualify_worker_runtime(str(worker))
    finally:
        transport.qualify_worker_runtime.cache_clear()
    assert invoked == (
        str(worker),
        "-B",
        "-I",
        "-m",
        "crucible_codex_worker",
        "--probe",
    )


def test_success_response_rejects_unsettled_tool_evidence():
    payload = _terminal_payload()
    payload["tool_items"] = [
        {
            "call_id": "call-1",
            "server": "crucible",
            "tool": "read_source",
            "status": "inProgress",
            "has_result": False,
            "has_error": False,
        }
    ]
    with pytest.raises(
        transport.CodexOfficialSDKQualificationError,
        match="unsettled",
    ):
        transport._validate_success_response(payload)


def test_transport_accepts_one_bounded_terminal_json(
    monkeypatch, tmp_path
):
    payload = json.dumps(
        _terminal_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )
    event = json.dumps(
        {
            "protocol": "crucible.codex-worker.v1",
            "type": "thread_started",
            "thread_id": "thread-1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    worker = _script(
        tmp_path,
        "read ignored\n"
        f"printf '%s\\n' '{event}'\n"
        f"printf '%s\\n' '{payload}'\n",
    )
    monkeypatch.setattr(
        transport,
        "qualify_worker_runtime",
        lambda _path: None,
    )
    invoked = None
    original_create_subprocess_exec = asyncio.create_subprocess_exec

    async def capture_worker_argv(*argv, **kwargs):
        nonlocal invoked
        invoked = tuple(argv)
        return await original_create_subprocess_exec(*argv, **kwargs)

    monkeypatch.setattr(
        transport.asyncio,
        "create_subprocess_exec",
        capture_worker_argv,
    )

    thread_ids = []
    result = asyncio.run(
        transport.run_official_worker(
            {"request": "value"},
            worker_python=worker,
            env={},
            timeout_seconds=2.0,
            thread_id_holder=thread_ids,
        )
    )
    assert result["turn_id"] == "turn-1"
    assert result["usage"]["total_tokens"] == 4
    assert thread_ids == ["thread-1"]
    assert invoked == (
        str(worker),
        "-B",
        "-I",
        "-m",
        "crucible_codex_worker",
    )


def test_transport_failure_preserves_pre_turn_thread_for_retry(
    monkeypatch, tmp_path
):
    event = json.dumps(
        {
            "protocol": "crucible.codex-worker.v1",
            "type": "thread_started",
            "thread_id": "thread-retry",
        },
        separators=(",", ":"),
    )
    terminal = json.dumps(
        {
            "protocol": "crucible.codex-worker.v1",
            "type": "terminal",
            "ok": False,
            "error_type": "BrokenPipeError",
            "error": "redacted",
        },
        separators=(",", ":"),
    )
    worker = _script(
        tmp_path,
        "read ignored\n"
        f"printf '%s\\n' '{event}'\n"
        f"printf '%s\\n' '{terminal}'\n",
    )
    monkeypatch.setattr(
        transport,
        "qualify_worker_runtime",
        lambda _path: None,
    )
    thread_ids = []

    with pytest.raises(
        transport.CodexOfficialWorkerInvocationError,
        match="BrokenPipeError",
    ):
        asyncio.run(
            transport.run_official_worker(
                {"request": "value"},
                worker_python=worker,
                env={},
                timeout_seconds=2.0,
                thread_id_holder=thread_ids,
            )
        )
    assert thread_ids == ["thread-retry"]


def test_timeout_kills_worker_process_group_and_prevents_late_child_action(
    monkeypatch, tmp_path
):
    marker = tmp_path / "late"
    worker = _script(
        tmp_path,
        "(trap '' TERM; "
        f"/bin/sleep 0.4; /usr/bin/touch '{marker}'; /bin/sleep 20) &\n"
        "/bin/sleep 20\n",
    )
    monkeypatch.setattr(
        transport,
        "qualify_worker_runtime",
        lambda _path: None,
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            transport.run_official_worker(
                {"request": "value"},
                worker_python=worker,
                env={},
                timeout_seconds=0.05,
            )
        )
    time.sleep(0.6)
    assert not marker.exists()


def test_normal_leader_exit_kills_term_resistant_descendant_before_return(
    monkeypatch, tmp_path
):
    marker = tmp_path / "late-after-leader-exit"
    payload = json.dumps(
        _terminal_payload(),
        sort_keys=True,
        separators=(",", ":"),
    )
    event = json.dumps(
        {
            "protocol": "crucible.codex-worker.v1",
            "type": "thread_started",
            "thread_id": "thread-1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    worker = _script(
        tmp_path,
        "read ignored\n"
        "(trap '' TERM; "
        f"/bin/sleep 0.4; /usr/bin/touch '{marker}'; /bin/sleep 20) "
        ">/dev/null 2>&1 &\n"
        f"printf '%s\\n' '{event}'\n"
        f"printf '%s\\n' '{payload}'\n",
    )
    monkeypatch.setattr(
        transport,
        "qualify_worker_runtime",
        lambda _path: None,
    )

    result = asyncio.run(
        transport.run_official_worker(
            {"request": "value"},
            worker_python=worker,
            env={},
            timeout_seconds=2.0,
        )
    )

    assert result["turn_id"] == "turn-1"
    time.sleep(0.6)
    assert not marker.exists()


def test_unproven_process_group_settlement_is_typed(monkeypatch):
    class NeverSettledProcess:
        pid = 987_654_321
        returncode = None

        async def wait(self):
            return 0

    signals = []

    def fake_killpg(_process_group_id, sig):
        signals.append(sig)

    monkeypatch.setattr(transport.os, "killpg", fake_killpg)
    monkeypatch.setattr(transport, "_STOP_GRACE_SECONDS", 0.001)

    with pytest.raises(
        transport.CodexOfficialSDKQualificationError,
        match="lifecycle did not settle",
    ):
        asyncio.run(transport._stop_process_group(NeverSettledProcess()))

    assert signal.SIGTERM in signals
    assert signal.SIGKILL in signals


def test_failed_settlement_does_not_report_process_stopped(
    monkeypatch, tmp_path
):
    worker = _script(tmp_path, "read ignored\n")
    monkeypatch.setattr(
        transport,
        "qualify_worker_runtime",
        lambda _path: None,
    )

    async def fail_settlement(_process):
        raise transport.CodexOfficialSDKQualificationError(
            "settlement unproven"
        )

    monkeypatch.setattr(transport, "_stop_process_group", fail_settlement)
    started = []
    stopped = []

    with pytest.raises(
        transport.CodexOfficialSDKQualificationError,
        match="settlement unproven",
    ):
        asyncio.run(
            transport.run_official_worker(
                {"request": "value"},
                worker_python=worker,
                env={},
                timeout_seconds=2.0,
                on_process_start=started.append,
                on_process_stop=stopped.append,
            )
        )

    assert len(started) == 1
    assert stopped == []
