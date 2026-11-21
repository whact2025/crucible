# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE verifier runner.

Wraps the upstream ``tests/test.sh`` shell that ships baked into
each per-task image. Run via ``docker exec <container_id>`` against
a live per-task container; reads one versioned reward artifact from a
host-bind-mounted logs dir (v1.0 ``reward.txt`` or v1.1 ``reward.json``) to
extract the binary reward.

Subprocess contract (per ``assay.bench``):

- Reads BenchConfig JSON on stdin.
- Emits JSON KPI payload on stdout.
- In direct/read-reward mode, exit 0 means the verifier command completed and
  wrote ``reward=1``. Reward 0 or invalid/missing reward evidence exits
  non-zero. This mode holds no tests-tree authority: a v1.1 ``reward.json`` is
  checked against the dataset contract and its own counts, not against the
  task's whitelist sizes.
- In workflow ``defer_reward_read`` mode, the runner cannot classify reward
  before strict container stop. Its process exit is therefore only the hidden
  command return code; reward resolution is published later by the workflow's
  post-stop artifact validation. In particular, command exit 0 does *not* mean
  the task resolved.

Recognized ``parameters``:

- ``role``: only ``"verify"`` is accepted by this runner.
  ``"build"`` is intentionally routed to the per-language runner in
  ``products/deepswe/spec_synthesis.py::_runners_for_instance``
  (DeepSWE images are dep-baked; the per-language build runner
  handles whatever build sanity-check the language needs).
- ``deepswe_metadata``: dict with::
    {
      "container_id": "<live docker container id>",
      "logs_dir": "<host path bind-mounted at /logs in container>",
    }
  Both are populated by the DeepSWE workflow via
  ``extra_parameters`` plumbing through tier → architect →
  run_architect. The workflow-only ``defer_reward_read: true`` flag prevents
  any pre-stop reward read; the workflow then performs the authoritative
  descriptor-safe read after strictly stopping the container.

KPI shape emitted (``reward`` and ``resolved`` are present only when a valid
reward artifact was read)::

    {
      "values": {
        "reward": 1.0 | 0.0,
        "resolved": 1.0 | 0.0,
        "command_returncode": <int>,
        "verify_returncode": <int>,  # compatibility alias
        "baseline_test_returncode": <int>,  # when unambiguous
        "new_test_returncode": <int>        # when unambiguous
      },
      "wall_seconds": <float>
    }

The command's stdout and stderr are streamed through bounded head/tail
captures. Private, host-only diagnostic artifacts retain complete output up to
the bound and explicit truncation metadata above it. Exact baseline/new-test
exit marker lines are extracted while streaming, so they remain available even
when the surrounding output is too large to retain. Console logging remains a
short tail only.

Verifier failure modes captured in the KPI:

A missing or rejected reward artifact is never reported as a score of zero:

- ``docker exec`` returned non-zero and left no valid reward: no reward is
  emitted, ``verify_returncode=<rc>``; exit with that code.
- reward artifact missing: no reward is emitted; exit non-zero. (The verifier
  exited before Step 5 — likely a setup-time error inside the shell.)
- reward evidence is ambiguous, malformed, unsafe, non-regular, oversized, or
  contradicts its own counts: same as missing, without logging
  attacker-controlled contents.

The per-task image's ``tests/test.sh`` handles its own workspace staging;
Crucible publishes the captured candidate at ``/logs/artifacts/model.patch``
for v1.1 but does not pre-commit the task workspace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import selectors
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.products.deepswe.git_safety import (
    ContainerGitEnvironmentError,
    hermetic_container_git_exec_prefix,
    hermetic_container_git_shell_function,
    inspect_docker_container_environment,
)
from crucible.products.deepswe.secure_artifacts import read_reward_artifact

logger = logging.getLogger(__name__)


# Per-call subprocess timeout for the ``docker exec`` invocation.
# Pier's per-task agent_timeout is 90 min; the verifier inside that
# is bounded by the task.toml verifier_timeout (typically 30 min).
# Default here is 60 min — adequate for any single task's verifier
# pass with headroom for slow Python startup + test runs.
_VERIFY_TIMEOUT_SEC = 3600

# Retain the first and last 64 KiB of each command stream. For output at or
# below 128 KiB this is the complete stream; larger output receives a visible
# omission marker and immutable size/hash metadata. The diagnostic directory
# is a sibling of the bind-mounted /logs directory, so verifier code cannot
# rewrite the host's evidence files.
_STREAM_EDGE_BYTES = 64 * 1024
_CONSOLE_TAIL_CHARS = 2_000
_MARKER_SCAN_BYTES = 256
_DIAGNOSTIC_SCHEMA = "crucible.deepswe.verifier-command-evidence.v1"

_BASELINE_MARKER = b"[verifier] Baseline exit code: "
_NEW_TEST_MARKER = b"[verifier] New tests exit code: "
_COMBINED_RC_MARKER = b"[verifier] base pytest rc="
_COMBINED_RC_LINE = re.compile(
    rb"\[verifier\] base pytest rc=([0-9]{1,3}); new pytest rc=([0-9]{1,3})"
    rb"(?: \(nonzero on failing tests is normal; graded from XML\))?"
)


class _BoundedStreamCapture:
    """Incrementally retain bounded stream evidence and exact exit markers."""

    def __init__(self) -> None:
        self._head = bytearray()
        self._tail = bytearray()
        self._sha256 = hashlib.sha256()
        self._total_bytes = 0
        self._marker_scan_tail = b""
        self._last_marker_end = {
            "baseline_test_returncode": -1,
            "new_test_returncode": -1,
        }
        self._marker_counts = {
            "baseline_test_returncode": 0,
            "new_test_returncode": 0,
        }
        self._marker_values: dict[str, int | None] = {
            "baseline_test_returncode": None,
            "new_test_returncode": None,
        }

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        prior_total = self._total_bytes
        self._total_bytes += len(chunk)
        self._sha256.update(chunk)

        head_remaining = max(0, _STREAM_EDGE_BYTES - len(self._head))
        if head_remaining:
            self._head.extend(chunk[:head_remaining])
        remainder = chunk[head_remaining:]
        if remainder:
            self._tail.extend(remainder)
            if len(self._tail) > _STREAM_EDGE_BYTES:
                del self._tail[:-_STREAM_EDGE_BYTES]

        scan = self._marker_scan_tail + chunk
        scan_offset = prior_total - len(self._marker_scan_tail)
        self._scan_markers(scan, scan_offset)
        self._marker_scan_tail = scan[-_MARKER_SCAN_BYTES:]

    def _scan_markers(self, data: bytes, absolute_offset: int) -> None:
        for marker, field in (
            (_BASELINE_MARKER, "baseline_test_returncode"),
            (_NEW_TEST_MARKER, "new_test_returncode"),
        ):
            search_from = 0
            while True:
                marker_at = data.find(marker, search_from)
                if marker_at < 0:
                    break
                value_at = marker_at + len(marker)
                newline_at = data.find(b"\n", value_at, value_at + 6)
                if newline_at < 0:
                    search_from = marker_at + 1
                    continue
                raw_value = data[value_at:newline_at].removesuffix(b"\r")
                absolute_end = absolute_offset + newline_at + 1
                if marker_at > 0 and data[marker_at - 1 : marker_at] != b"\n":
                    search_from = marker_at + 1
                    continue
                if absolute_end <= self._last_marker_end[field]:
                    search_from = marker_at + 1
                    continue
                if raw_value.isdigit() and len(raw_value) <= 3:
                    value = int(raw_value)
                    if 0 <= value <= 255:
                        self._record_marker(field, value, absolute_end)
                search_from = marker_at + 1

        search_from = 0
        while True:
            marker_at = data.find(_COMBINED_RC_MARKER, search_from)
            if marker_at < 0:
                break
            newline_at = data.find(
                b"\n",
                marker_at,
                marker_at + _MARKER_SCAN_BYTES,
            )
            if newline_at < 0:
                search_from = marker_at + 1
                continue
            if marker_at > 0 and data[marker_at - 1 : marker_at] != b"\n":
                search_from = marker_at + 1
                continue
            line = data[marker_at:newline_at].removesuffix(b"\r")
            match = _COMBINED_RC_LINE.fullmatch(line)
            absolute_end = absolute_offset + newline_at + 1
            if match is not None:
                values = (int(match.group(1)), int(match.group(2)))
                if all(0 <= value <= 255 for value in values):
                    for field, value in zip(
                        ("baseline_test_returncode", "new_test_returncode"),
                        values,
                        strict=True,
                    ):
                        if absolute_end > self._last_marker_end[field]:
                            self._record_marker(field, value, absolute_end)
            search_from = marker_at + 1

    def _record_marker(self, field: str, value: int, absolute_end: int) -> None:
        """Retain one unambiguous, complete verifier-owned exit marker."""

        self._last_marker_end[field] = absolute_end
        self._marker_counts[field] += 1
        if self._marker_counts[field] == 1:
            self._marker_values[field] = value
        else:
            # Candidate output can spoof the human-readable marker. Multiple
            # exact lines are ambiguous and never become structured evidence.
            self._marker_values[field] = None

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def truncated(self) -> bool:
        return self._total_bytes > 2 * _STREAM_EDGE_BYTES

    @property
    def sha256(self) -> str:
        return self._sha256.hexdigest()

    @property
    def baseline_test_returncode(self) -> int | None:
        return self._marker_values["baseline_test_returncode"]

    @property
    def new_test_returncode(self) -> int | None:
        return self._marker_values["new_test_returncode"]

    def evidence_bytes(self) -> bytes:
        if not self.truncated:
            return bytes(self._head + self._tail)
        omitted = self._total_bytes - len(self._head) - len(self._tail)
        marker = (
            "\n[crucible: verifier output truncated; "
            f"omitted {omitted} bytes; sha256={self.sha256}]\n"
        ).encode("ascii")
        return bytes(self._head) + marker + bytes(self._tail)

    def text(self) -> str:
        return self.evidence_bytes().decode("utf-8", errors="replace")

    def console_tail(self) -> str:
        return bytes(self._tail or self._head).decode("utf-8", errors="replace")[
            -_CONSOLE_TAIL_CHARS:
        ]

    def metadata(self, *, filename: str) -> dict[str, Any]:
        return {
            "filename": filename,
            "total_bytes": self.total_bytes,
            "retained_bytes": len(self.evidence_bytes()),
            "truncated": self.truncated,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _DockerExecResult:
    returncode: int
    stdout: _BoundedStreamCapture
    stderr: _BoundedStreamCapture


def _capture_text(value: str) -> _BoundedStreamCapture:
    """Build bounded evidence for local errors and focused tests."""

    capture = _BoundedStreamCapture()
    capture.feed(value.encode("utf-8", errors="replace"))
    return capture


def _diagnostics_dir(logs_dir: Path) -> Path:
    return logs_dir.parent / f"{logs_dir.name}-command-evidence"


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        info = path.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise OSError("verifier diagnostic directory is unsafe") from None


def _write_private_atomic(path: Path, data: bytes) -> None:
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary)


def _persist_command_evidence(
    logs_dir: Path,
    result: _DockerExecResult,
) -> bool:
    """Persist bounded, private diagnostics outside the verifier bind mount."""

    directory = _diagnostics_dir(logs_dir)
    stdout_name = "stdout.log"
    stderr_name = "stderr.log"
    metadata_name = "result.json"
    try:
        _ensure_private_directory(directory)
        _write_private_atomic(
            directory / stdout_name,
            result.stdout.evidence_bytes(),
        )
        _write_private_atomic(
            directory / stderr_name,
            result.stderr.evidence_bytes(),
        )
        metadata = {
            "schema": _DIAGNOSTIC_SCHEMA,
            "command_returncode": result.returncode,
            "baseline_test_returncode": result.stdout.baseline_test_returncode,
            "new_test_returncode": result.stdout.new_test_returncode,
            "stdout": result.stdout.metadata(filename=stdout_name),
            "stderr": result.stderr.metadata(filename=stderr_name),
        }
        encoded = (
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        _write_private_atomic(directory / metadata_name, encoded)
    except OSError as exc:
        logger.warning(
            "deepswe_verifier: bounded command evidence could not be persisted: %s",
            exc,
        )
        return False
    logger.info(
        "deepswe_verifier: bounded command evidence persisted under %s",
        directory,
    )
    return True


def _run_docker_exec(
    container_id: str,
    cmd: list[str],
    *,
    timeout: int = _VERIFY_TIMEOUT_SEC,
) -> _DockerExecResult:
    """Run docker exec with bounded streaming stdout/stderr evidence.

    Timeouts surface as command return code 124, matching the shell timeout
    convention. Output is never accumulated without a fixed memory bound.
    """
    try:
        inherited_environment = inspect_docker_container_environment(container_id)
        exec_prefix = hermetic_container_git_exec_prefix(inherited_environment)
    except ContainerGitEnvironmentError as exc:
        return _DockerExecResult(
            125,
            _capture_text(""),
            _capture_text(str(exc)),
        )
    full = [
        "docker",
        "exec",
        container_id,
        *exec_prefix,
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        f"{timeout}s",
        *cmd,
    ]
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout_capture = _BoundedStreamCapture()
    stderr_capture = _BoundedStreamCapture()
    deadline = time.monotonic() + timeout + 15
    try:
        process = subprocess.Popen(
            full,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, stdout_capture)
        selector.register(process.stderr, selectors.EVENT_READ, stderr_capture)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(full, timeout + 15)
            events = selector.select(timeout=min(remaining, 1.0))
            if not events:
                continue
            for key, _mask in events:
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                capture = key.data
                assert isinstance(capture, _BoundedStreamCapture)
                capture.feed(chunk)
        remaining = max(0.0, deadline - time.monotonic())
        returncode = process.wait(timeout=remaining)
        return _DockerExecResult(
            returncode,
            stdout_capture,
            stderr_capture,
        )
    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
            process.wait()
        stderr_capture.feed(f"verify timed out after {timeout}s".encode())
        return _DockerExecResult(124, stdout_capture, stderr_capture)
    except FileNotFoundError as exc:
        # docker binary missing
        return _DockerExecResult(
            127,
            stdout_capture,
            _capture_text(f"docker not on PATH: {exc}"),
        )
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


def _emit_kpi(
    stdout: Any,
    *,
    reward: float | None,
    command_returncode: int,
    wall_seconds: float,
    reward_pending: bool = False,
    command_evidence: _DockerExecResult | None = None,
    diagnostics_persisted: bool = False,
) -> None:
    """Write the KPI JSON payload to ``stdout``."""
    values: dict[str, float] = {
        # Keep verify_returncode as a compatibility alias. It has always been
        # the docker/test.sh command status, not the binary reward verdict.
        "command_returncode": float(command_returncode),
        "verify_returncode": float(command_returncode),
        "reward_pending": float(reward_pending),
        "diagnostics_persisted": float(diagnostics_persisted),
    }
    if command_evidence is not None:
        values.update(
            {
                "verifier_stdout_bytes": float(command_evidence.stdout.total_bytes),
                "verifier_stderr_bytes": float(command_evidence.stderr.total_bytes),
                "verifier_stdout_truncated": float(command_evidence.stdout.truncated),
                "verifier_stderr_truncated": float(command_evidence.stderr.truncated),
            }
        )
        baseline_rc = command_evidence.stdout.baseline_test_returncode
        new_test_rc = command_evidence.stdout.new_test_returncode
        if baseline_rc is not None:
            values["baseline_test_returncode"] = float(baseline_rc)
        if new_test_rc is not None:
            values["new_test_returncode"] = float(new_test_rc)
    if reward is not None:
        values.update(
            {
                "reward": reward,
                "resolved": reward,
            }
        )
    payload = {
        "values": values,
        "wall_seconds": wall_seconds,
    }
    json.dump(payload, stdout)
    stdout.write("\n")


def _run_verify(
    parameters: dict[str, Any],
    stdout: Any,
) -> int:
    """Run the verifier shell inside the per-task container and emit
    a KPI payload.

    Direct mode returns 0 only for command success plus reward 1. Deferred
    workflow mode returns the command status because reward authority is not
    consumed until strict post-stop validation.
    """
    metadata = parameters.get("deepswe_metadata") or {}
    if not isinstance(metadata, dict):
        logger.error(
            "deepswe_verifier: ``deepswe_metadata`` parameter must be a dict, got %s",
            type(metadata).__name__,
        )
        return 2

    container_id = metadata.get("container_id")
    logs_dir_str = metadata.get("logs_dir")
    if not container_id or not logs_dir_str:
        logger.error(
            "deepswe_verifier: ``deepswe_metadata`` missing required "
            "keys ``container_id`` / ``logs_dir`` (got: %s)",
            sorted(metadata.keys()),
        )
        return 2

    logs_dir = Path(logs_dir_str)
    defer_reward_read = metadata.get("defer_reward_read", False)
    if not isinstance(defer_reward_read, bool):
        logger.error(
            "deepswe_verifier: defer_reward_read must be a boolean, got %s",
            type(defer_reward_read).__name__,
        )
        return 2
    start = time.monotonic()
    harness_script = f"{hermetic_container_git_shell_function()}\nsource /tests/test.sh"
    command = _run_docker_exec(
        str(container_id),
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            harness_script,
            "deepswe-hidden-verifier",
        ],
    )
    wall = time.monotonic() - start
    diagnostics_persisted = _persist_command_evidence(logs_dir, command)
    stdout_tail = command.stdout.console_tail()
    stderr_tail = command.stderr.console_tail()
    if stdout_tail:
        logger.info("deepswe_verifier: test.sh stdout (tail):\n%s", stdout_tail)
    if stderr_tail:
        logger.info("deepswe_verifier: test.sh stderr (tail):\n%s", stderr_tail)

    if defer_reward_read:
        _emit_kpi(
            stdout,
            reward=None,
            command_returncode=command.returncode,
            wall_seconds=wall,
            reward_pending=True,
            command_evidence=command,
            diagnostics_persisted=diagnostics_persisted,
        )
        return command.returncode

    # Direct mode holds no tests-tree authority, so a reward.json is checked
    # against its own counts only, not against the task's whitelist sizes.
    reward_read = read_reward_artifact(logs_dir, inventory=None)
    reward = reward_read.reward
    if reward_read.state != "valid" or reward is None:
        # No valid artifact, no score: a missing or rejected reward is never
        # published as reward 0. The non-zero exit still fails the gate.
        if reward_read.state == "missing":
            logger.error(
                "deepswe_verifier: reward artifact missing under %s "
                "(verifier exited rc=%d before publishing it — see stderr above)",
                logs_dir / "verifier",
                command.returncode,
            )
        else:
            logger.error(
                "deepswe_verifier: reward artifact rejected "
                "(state=%s, reason=%s); "
                "no reward is reported",
                reward_read.state,
                reward_read.error or "unspecified",
            )
        _emit_kpi(
            stdout,
            reward=None,
            command_returncode=command.returncode,
            wall_seconds=wall,
            command_evidence=command,
            diagnostics_persisted=diagnostics_persisted,
        )
        return 1 if command.returncode == 0 else command.returncode

    _emit_kpi(
        stdout,
        reward=reward,
        command_returncode=command.returncode,
        wall_seconds=wall,
        command_evidence=command,
        diagnostics_persisted=diagnostics_persisted,
    )
    # Exit 0 iff verifier said resolved AND docker exec itself
    # succeeded; otherwise non-zero so build_verify classification
    # can flip success → verify_failed.
    return 0 if (reward == 1.0 and command.returncode == 0) else 1


def main(stdin: object = None, stdout: object = None) -> int:
    """Entry point. Reads BenchConfig JSON on stdin, dispatches
    on ``parameters.role``, writes KPI JSON on stdout."""
    stdin_obj = stdin if stdin is not None else sys.stdin
    stdout_obj = stdout if stdout is not None else sys.stdout

    try:
        cfg = json.loads(stdin_obj.read())
    except (
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
    ) as e:
        logger.error("deepswe_verifier: BenchConfig stdin parse failed: %s", e)
        return 2

    parameters = (cfg.get("parameters") or {}) if isinstance(cfg, dict) else {}
    role = parameters.get("role", "verify")

    if role != "verify":
        logger.error(
            "deepswe_verifier: role=%r not supported. Build-role calls "
            "should route to crucible.common.runners.lang.<lang> per "
            "products/deepswe/spec_synthesis.py — this is a routing "
            "bug, not a runner extension point.",
            role,
        )
        return 2

    real_stdout = sys.stdout
    try:
        sys.stdout = stdout_obj  # type: ignore[assignment]
        return _run_verify(parameters, stdout_obj)
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
