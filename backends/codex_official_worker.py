# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded parent-side transport for the isolated official Codex SDK worker."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

_PROTOCOL = "crucible.codex-worker.v1"
_SDK_VERSION = "0.144.4"
_WORKER_PROJECT = Path(__file__).with_name("codex_worker")
_DEFAULT_WORKER_PYTHON = _WORKER_PROJECT / ".venv" / "bin" / "python"
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_STOP_GRACE_SECONDS = 2.0
_STOP_POLL_SECONDS = 0.01
_PROBE_TIMEOUT_SECONDS = 15.0


class CodexOfficialSDKUnavailable(RuntimeError):
    """The independently locked official SDK worker is not installed."""


class CodexOfficialSDKQualificationError(RuntimeError):
    """The installed worker or its wire result differs from the reviewed ABI."""


class CodexOfficialWorkerInvocationError(RuntimeError):
    """The qualified worker reported a provider/transport invocation error."""

    def __init__(self, error_type: str) -> None:
        self.error_type = error_type
        super().__init__(
            f"{error_type}: official Codex worker invocation failed"
        )


def resolve_worker_python(explicit: Path | str | None = None) -> Path:
    """Resolve one absolute worker interpreter without a PATH lookup."""

    configured = explicit
    if configured is None:
        configured = os.environ.get("CRUCIBLE_CODEX_WORKER_PYTHON")
    candidate = (
        Path(configured).expanduser()
        if configured
        else _DEFAULT_WORKER_PYTHON
    )
    if not candidate.is_absolute():
        raise CodexOfficialSDKQualificationError(
            "official Codex worker Python must be an absolute path"
        )
    return Path(os.path.abspath(candidate))


def worker_environment(
    *,
    codex_home: str,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a minimal SDK/CLI environment rooted at a private Codex home."""

    if not isinstance(codex_home, str) or not codex_home:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker requires an explicit CODEX_HOME"
        )
    home = Path(codex_home)
    if not home.is_absolute() or not home.is_dir():
        raise CodexOfficialSDKQualificationError(
            "official Codex worker CODEX_HOME must be an existing "
            "absolute directory"
        )
    source = os.environ if inherited is None else inherited
    allowed = (
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
    )
    result = {name: source[name] for name in allowed if name in source}
    result.update(
        {
            "CODEX_HOME": str(home),
            "PYTHONNOUSERSITE": "1",
        }
    )
    return result


@functools.lru_cache(maxsize=8)
def qualify_worker_runtime(worker_python_text: str) -> None:
    """Fail closed unless the worker reports the exact SDK and CLI tuple."""

    worker_python = Path(worker_python_text)
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise CodexOfficialSDKUnavailable(
            "official Codex worker runtime is not installed; run "
            f"`uv sync --project {_WORKER_PROJECT}` or set "
            "CRUCIBLE_CODEX_WORKER_PYTHON to its absolute interpreter path"
        )
    try:
        completed = subprocess.run(
            [
                str(worker_python),
                "-B",
                "-I",
                "-m",
                "crucible_codex_worker",
                "--probe",
            ],
            check=False,
            capture_output=True,
            env={"PYTHONNOUSERSITE": "1"},
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodexOfficialSDKUnavailable(
            "failed to probe the isolated official Codex worker "
            f"({type(exc).__name__})"
        ) from exc
    if len(completed.stdout) > _MAX_STDERR_BYTES:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker probe exceeded its output limit"
        )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker probe did not return JSON identity"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(payload, dict)
        or payload.get("protocol") != _PROTOCOL
        or payload.get("type") != "probe"
        or payload.get("ok") is not True
        or payload.get("sdk_version") != _SDK_VERSION
        or payload.get("cli_version") != _SDK_VERSION
    ):
        raise CodexOfficialSDKQualificationError(
            "official Codex worker does not match the qualified "
            f"openai-codex=={_SDK_VERSION} / "
            f"openai-codex-cli-bin=={_SDK_VERSION} tuple"
        )


async def _read_bounded(
    stream: asyncio.StreamReader,
    *,
    limit: int,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(min(64 * 1024, limit + 1 - size))
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise CodexOfficialSDKQualificationError(
                f"official Codex worker {label} exceeded its byte limit"
            )
        chunks.append(chunk)


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a worker process group still has any members."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise CodexOfficialSDKQualificationError(
            "could not inspect the official Codex worker process group "
            f"({type(exc).__name__})"
        ) from exc
    return True


def _signal_process_group(process_group_id: int, sig: signal.Signals) -> bool:
    """Signal a process group, returning false only when it is already gone."""

    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise CodexOfficialSDKQualificationError(
            "could not signal the official Codex worker process group "
            f"with {sig.name} ({type(exc).__name__})"
        ) from exc
    return True


async def _wait_for_process_group_settlement(
    process: asyncio.subprocess.Process,
    *,
    process_group_id: int,
    timeout: float,
    return_on_leader_reap: bool = False,
) -> tuple[bool, bool]:
    """Wait boundedly for both leader reap and process-group disappearance."""

    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        leader_reaped = process.returncode is not None
        if leader_reaped:
            await process.wait()
        group_gone = not _process_group_exists(process_group_id)
        if leader_reaped and (group_gone or return_on_leader_reap):
            return leader_reaped, group_gone
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return leader_reaped, group_gone
        await asyncio.sleep(min(_STOP_POLL_SECONDS, remaining))


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop and prove settlement of one isolated worker process group."""

    process_group_id = process.pid
    if _process_group_exists(process_group_id):
        _signal_process_group(process_group_id, signal.SIGTERM)
    elif process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()

    leader_reaped, group_gone = await _wait_for_process_group_settlement(
        process,
        process_group_id=process_group_id,
        timeout=_STOP_GRACE_SECONDS,
        # Once the leader exits there is no worker left to coordinate its
        # descendants, so residual members are killed without another grace
        # period in which they could perform late actions.
        return_on_leader_reap=True,
    )
    if leader_reaped and group_gone:
        return

    if not group_gone and _process_group_exists(process_group_id):
        _signal_process_group(process_group_id, signal.SIGKILL)
    elif not leader_reaped and process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    leader_reaped, group_gone = await _wait_for_process_group_settlement(
        process,
        process_group_id=process_group_id,
        timeout=_STOP_GRACE_SECONDS,
    )
    if not leader_reaped or not group_gone:
        unsettled = []
        if not leader_reaped:
            unsettled.append("leader was not reaped")
        if not group_gone:
            unsettled.append("process group still exists")
        raise CodexOfficialSDKQualificationError(
            "official Codex worker lifecycle did not settle after SIGKILL: "
            + "; ".join(unsettled)
        )


def _validate_success_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CodexOfficialSDKQualificationError(
            "official Codex worker result must be an object"
        )
    if payload.get("protocol") != _PROTOCOL:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker protocol mismatch"
        )
    if payload.get("ok") is not True:
        error_type = payload.get("error_type")
        if not isinstance(error_type, str) or not error_type:
            error_type = "WorkerError"
        raise CodexOfficialWorkerInvocationError(error_type)
    expected = {
        "protocol",
        "type",
        "ok",
        "sdk_version",
        "thread_id",
        "turn_id",
        "status",
        "error",
        "final_response",
        "usage",
        "tool_items",
    }
    if set(payload) != expected:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker result fields changed"
        )
    if payload["type"] != "terminal":
        raise CodexOfficialSDKQualificationError(
            "official Codex worker result is not terminal"
        )
    if payload["sdk_version"] != _SDK_VERSION:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker SDK version drift"
        )
    for name in ("thread_id", "turn_id", "status", "final_response"):
        if not isinstance(payload[name], str):
            raise CodexOfficialSDKQualificationError(
                f"official Codex worker returned invalid {name}"
            )
    if payload["status"] not in {"completed", "interrupted", "failed"}:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker returned a non-terminal or unknown status"
        )
    if payload["error"] is not None and not isinstance(
        payload["error"], dict
    ):
        raise CodexOfficialSDKQualificationError(
            "official Codex worker returned invalid error"
        )
    if payload["error"] is not None:
        if set(payload["error"]) != {
            "message",
            "has_additional_details",
            "has_codex_error_info",
        }:
            raise CodexOfficialSDKQualificationError(
                "official Codex worker error fields changed"
            )
        if (
            not isinstance(payload["error"]["message"], str)
            or not payload["error"]["message"]
            or type(payload["error"]["has_additional_details"]) is not bool
            or type(payload["error"]["has_codex_error_info"]) is not bool
        ):
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned malformed error evidence"
            )
    if payload["status"] == "completed" and payload["error"] is not None:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker completed with contradictory error evidence"
        )
    if payload["status"] == "failed" and payload["error"] is None:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker failed without error evidence"
        )
    usage = payload["usage"]
    if usage is not None:
        expected_usage = {
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        }
        if (
            not isinstance(usage, dict)
            or set(usage) != expected_usage
            or any(type(value) is not int or value < 0 for value in usage.values())
            or usage["cached_input_tokens"] > usage["input_tokens"]
            or usage["reasoning_output_tokens"] > usage["output_tokens"]
            or usage["total_tokens"]
            != usage["input_tokens"] + usage["output_tokens"]
        ):
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned invalid usage"
            )
    tool_items = payload["tool_items"]
    if not isinstance(tool_items, list):
        raise CodexOfficialSDKQualificationError(
            "official Codex worker returned invalid tool evidence"
        )
    expected_tool = {
        "call_id",
        "server",
        "tool",
        "status",
        "has_result",
        "has_error",
    }
    seen: set[str] = set()
    for item in tool_items:
        if not isinstance(item, dict) or set(item) != expected_tool:
            raise CodexOfficialSDKQualificationError(
                "official Codex worker tool evidence fields changed"
            )
        if any(
            not isinstance(item[name], str) or not item[name]
            for name in ("call_id", "server", "tool")
        ):
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned incomplete tool identity"
            )
        if item["server"] != "crucible":
            raise CodexOfficialSDKQualificationError(
                "official Codex worker observed an undeclared MCP server"
            )
        if item["call_id"] in seen:
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned a duplicate tool call id"
            )
        seen.add(item["call_id"])
        if item["status"] not in {"completed", "failed"}:
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned an unsettled or unknown "
                "tool status"
            )
        if type(item["has_result"]) is not bool or type(
            item["has_error"]
        ) is not bool:
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned invalid tool settlement"
            )
        if not (item["has_result"] or item["has_error"]):
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned contradictory tool settlement"
            )
    return payload


def _decode_worker_stream(
    stdout: bytes,
    *,
    thread_id_holder: list[str] | None,
) -> Any:
    lines = stdout.splitlines()
    if not lines or len(lines) > 2:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker returned an invalid event count"
        )
    decoded: list[Any] = []
    for line in lines:
        try:
            decoded.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodexOfficialSDKQualificationError(
                "official Codex worker did not return valid JSON events"
            ) from exc
    terminal = decoded[-1]
    if len(decoded) == 2:
        event = decoded[0]
        if (
            not isinstance(event, dict)
            or set(event) != {"protocol", "type", "thread_id"}
            or event.get("protocol") != _PROTOCOL
            or event.get("type") != "thread_started"
            or not isinstance(event.get("thread_id"), str)
            or not event["thread_id"]
        ):
            raise CodexOfficialSDKQualificationError(
                "official Codex worker returned an invalid thread event"
            )
        if thread_id_holder is not None:
            thread_id_holder.append(event["thread_id"])
        if (
            isinstance(terminal, dict)
            and terminal.get("ok") is True
            and terminal.get("thread_id") != event["thread_id"]
        ):
            raise CodexOfficialSDKQualificationError(
                "official Codex worker thread identity changed within a turn"
            )
    elif isinstance(terminal, dict) and terminal.get("ok") is True:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker omitted its pre-turn thread identity"
        )
    return terminal


async def run_official_worker(
    request: Mapping[str, Any],
    *,
    worker_python: Path,
    env: Mapping[str, str],
    timeout_seconds: float | None,
    thread_id_holder: list[str] | None = None,
    on_process_start: Callable[[int], None] | None = None,
    on_process_stop: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run one worker process group and return a qualified terminal payload."""

    qualify_worker_runtime(str(worker_python))
    encoded = json.dumps(
        dict(request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker request exceeds its byte limit"
        )
    try:
        process = await asyncio.create_subprocess_exec(
            str(worker_python),
            "-B",
            "-I",
            "-m",
            "crucible_codex_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(env),
            start_new_session=True,
        )
    except OSError as exc:
        raise CodexOfficialSDKUnavailable(
            "failed to start the isolated official Codex worker "
            f"({type(exc).__name__})"
        ) from exc
    if on_process_start is not None:
        try:
            on_process_start(process.pid)
        except BaseException:
            await _stop_process_group(process)
            raise
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(
        _read_bounded(
            process.stdout,
            limit=_MAX_RESPONSE_BYTES,
            label="stdout",
        )
    )
    stderr_task = asyncio.create_task(
        _read_bounded(
            process.stderr,
            limit=_MAX_STDERR_BYTES,
            label="stderr",
        )
    )
    try:
        process.stdin.write(encoded)
        await process.stdin.drain()
        process.stdin.close()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.wait_closed()
        completed = asyncio.gather(
            process.wait(),
            stdout_task,
            stderr_task,
        )
        if timeout_seconds is None:
            returncode, stdout, stderr = await completed
        else:
            returncode, stdout, stderr = await asyncio.wait_for(
                completed,
                timeout=timeout_seconds,
            )
    except BaseException:
        for task in (stdout_task, stderr_task):
            task.cancel()
        await asyncio.gather(
            stdout_task,
            stderr_task,
            return_exceptions=True,
        )
        raise
    finally:
        try:
            await _stop_process_group(process)
        except BaseException:
            for task in (stdout_task, stderr_task):
                task.cancel()
            await asyncio.gather(
                stdout_task,
                stderr_task,
                return_exceptions=True,
            )
            raise
        else:
            if on_process_stop is not None:
                on_process_stop(process.pid)
    decoded = _decode_worker_stream(
        stdout,
        thread_id_holder=thread_id_holder,
    )
    if returncode != 0:
        raise CodexOfficialSDKQualificationError(
            "official Codex worker exited without a valid terminal result "
            f"(exit code {returncode}; private stderr {len(stderr)} bytes)"
        )
    return _validate_success_response(decoded)


__all__ = [
    "CodexOfficialSDKQualificationError",
    "CodexOfficialSDKUnavailable",
    "CodexOfficialWorkerInvocationError",
    "qualify_worker_runtime",
    "resolve_worker_python",
    "run_official_worker",
    "worker_environment",
]
