# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.deepswe.runners.deepswe_verifier``.

Hermetic — uses ``monkeypatch`` against ``subprocess.Popen`` to fake
docker exec, and ``tmp_path`` to fake the host-bind-mounted
``/logs`` dir where the verifier shell would write reward.txt.
No real docker, no real image, no real test.sh execution.

Coverage:
- happy path (reward=1)
- unresolved path (reward=0)
- docker exec failure (returncode != 0)
- reward.txt missing (verifier exited before Step 5)
- reward.txt contents not parseable
- role refusal (build / unknown)
- malformed BenchConfig stdin
- missing deepswe_metadata keys
- timeout surfaces as returncode 124
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from crucible.products.deepswe.runners import deepswe_verifier


@pytest.fixture(autouse=True)
def _hermetic_image_environment(monkeypatch):
    monkeypatch.setattr(
        deepswe_verifier,
        "inspect_docker_container_environment",
        lambda _container_id: (
            "PATH=/app/bin:/tmp/crucible-rust-immutable/cargo-bin:"
            "/root/.cargo/bin:/usr/bin:/bin",
            "CARGO_HOME=/tmp/crucible-cargo-home",
            "RUSTUP_HOME=/tmp/crucible-rust-immutable/rustup",
            "CARGO_TARGET_DIR=/app/target",
            "CUDA_HOME=/opt/cuda",
            "HOME=/attacker/home",
            "GIT_CONFIG_COUNT=1",
            "GIT_CONFIG_KEY_0=alias.apply",
            "GIT_CONFIG_VALUE_0=!touch /logs/image-config-owned",
            "BASH_ENV=/app/evil-bashrc",
            "ENV=/app/evil-shrc",
            "BASH_FUNC_git%%=() { touch /logs/function-owned; }",
        ),
    )


# ---------- fixture helpers ----------


def _write_reward(logs_dir: Path, contents: str) -> None:
    """Write the verifier's reward.txt into the host-bind-mounted
    logs dir (same path the verifier shell writes to inside the
    container)."""
    (logs_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (logs_dir / "verifier" / "reward.txt").write_text(contents)


def _write_json_reward(logs_dir: Path, payload: str) -> None:
    """Write the DeepSWE v1.1 reward.json artifact."""

    (logs_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (logs_dir / "verifier" / "reward.json").write_text(payload)


def _make_stdin(parameters: dict) -> io.StringIO:
    """Build the BenchConfig stdin shape the runner reads."""
    return io.StringIO(json.dumps({"parameters": parameters}))


def _fake_docker_exec(
    rc: int = 0,
    stdout: str = "",
    stderr: str = "",
    *,
    timeout_on_wait: bool = False,
):
    """Return a Popen-shaped factory backed by small, closed pipes."""
    calls: list[list[str]] = []

    class _FakeProcess:
        def __init__(self, args, **_kwargs):
            calls.append(list(args))
            self.args = args
            self.returncode = None
            self._configured_returncode = rc
            self._timeout_on_wait = timeout_on_wait
            self._killed = False
            stdout_read, stdout_write = os.pipe()
            stderr_read, stderr_write = os.pipe()
            stdout_bytes = stdout.encode("utf-8")
            stderr_bytes = stderr.encode("utf-8")
            assert len(stdout_bytes) < 32 * 1024
            assert len(stderr_bytes) < 32 * 1024
            os.write(stdout_write, stdout_bytes)
            os.write(stderr_write, stderr_bytes)
            os.close(stdout_write)
            os.close(stderr_write)
            self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
            self.stderr = os.fdopen(stderr_read, "rb", buffering=0)

        def wait(self, timeout=None):
            if self._timeout_on_wait and not self._killed:
                raise subprocess.TimeoutExpired(self.args, timeout)
            if self.returncode is None:
                self.returncode = -9 if self._killed else self._configured_returncode
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            self._killed = True
            self.returncode = -9

    def _fake(args, **kwargs):
        return _FakeProcess(args, **kwargs)

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


def _run_with_logs(logs_dir: Path, monkeypatch) -> tuple[int, dict]:
    monkeypatch.setattr(subprocess, "Popen", _fake_docker_exec(rc=0))
    stdout = io.StringIO()
    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(logs_dir),
                },
            }
        ),
        stdout=stdout,
    )
    return rc, json.loads(stdout.getvalue())


def _captured_command(
    *,
    rc: int = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    chunk_bytes: int = 997,
):
    stdout_capture = deepswe_verifier._BoundedStreamCapture()
    stderr_capture = deepswe_verifier._BoundedStreamCapture()
    for offset in range(0, len(stdout), chunk_bytes):
        stdout_capture.feed(stdout[offset : offset + chunk_bytes])
    for offset in range(0, len(stderr), chunk_bytes):
        stderr_capture.feed(stderr[offset : offset + chunk_bytes])
    return deepswe_verifier._DockerExecResult(
        rc,
        stdout_capture,
        stderr_capture,
    )


# ---------- happy path ----------


def test_verify_happy_path(tmp_path: Path, monkeypatch):
    """reward.txt = '1', docker exec rc=0 → emit reward=1 KPI,
    return 0."""
    _write_reward(tmp_path, "1")
    command_stdout = (
        "[verifier] Baseline exit code: 0\n"
        "[verifier] New tests exit code: 0\n"
        "[verifier] reward=1\n"
    )
    command_stderr = "diagnostic warning\n"
    fake = _fake_docker_exec(
        rc=0,
        stdout=command_stdout,
        stderr=command_stderr,
    )
    monkeypatch.setattr(subprocess, "Popen", fake)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 0

    payload = json.loads(stdout.getvalue())
    assert payload["values"]["reward"] == 1.0
    assert payload["values"]["resolved"] == 1.0
    assert payload["values"]["command_returncode"] == 0.0
    assert payload["values"]["verify_returncode"] == 0.0
    assert payload["values"]["baseline_test_returncode"] == 0.0
    assert payload["values"]["new_test_returncode"] == 0.0
    assert payload["values"]["diagnostics_persisted"] == 1.0
    assert payload["wall_seconds"] >= 0.0

    diagnostics = deepswe_verifier._diagnostics_dir(tmp_path)
    assert (diagnostics / "stdout.log").read_text() == command_stdout
    assert (diagnostics / "stderr.log").read_text() == command_stderr
    evidence = json.loads((diagnostics / "result.json").read_text())
    assert evidence["schema"].endswith("verifier-command-evidence.v1")
    assert evidence["command_returncode"] == 0
    assert evidence["baseline_test_returncode"] == 0
    assert evidence["new_test_returncode"] == 0
    assert evidence["stdout"]["truncated"] is False
    assert evidence["stderr"]["truncated"] is False
    assert diagnostics.stat().st_mode & 0o077 == 0
    assert (diagnostics / "stdout.log").stat().st_mode & 0o077 == 0

    # The complete hidden shell, including timeout, receives only the shared
    # hermetic environment; image Git aliases and exported functions vanish.
    assert len(fake.calls) == 1
    command = fake.calls[0]
    assert command[:6] == [
        "docker",
        "exec",
        "abc123",
        "/usr/bin/env",
        "-i",
        "--",
    ]
    assert "CUDA_HOME=/opt/cuda" in command
    assert "HOME=/tmp/home" in command
    assert (
        "PATH=/tmp/crucible-rust-immutable/cargo-bin:/root/.cargo/bin:/usr/bin:/bin"
    ) in command
    assert "CARGO_HOME=/tmp/crucible-cargo-home" in command
    assert "RUSTUP_HOME=/tmp/crucible-rust-immutable/rustup" in command
    assert "CARGO_TARGET_DIR=/app/target" in command
    assert not any(argument.startswith("GIT_") for argument in command[6:])
    assert not any("evil" in arg or "function-owned" in arg for arg in command)
    assert not any("image-config-owned" in arg for arg in command)
    timeout_index = command.index("/usr/bin/timeout")
    assert command[timeout_index : timeout_index + 5] == [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=5s",
        "3600s",
        "/bin/bash",
    ]
    assert command[-6:-2] == [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
    ]
    harness_script = command[-2]
    assert "git() {" in harness_script
    assert "/usr/bin/git" in harness_script
    assert "--git-dir=/app/.git" in harness_script
    assert "--work-tree=/app" in harness_script
    assert "GIT_CONFIG_NOSYSTEM=1" in harness_script
    assert "source /tests/test.sh" in harness_script
    assert "export -f" not in harness_script
    assert command[-1] == "deepswe-hidden-verifier"


def test_v11_combined_exit_marker_is_structured_across_chunks():
    command = _captured_command(
        stdout=(b"prefix\n[verifier] base pytest rc=0; new pytest rc=70\ntrailer\n"),
        chunk_bytes=7,
    )

    assert command.stdout.baseline_test_returncode == 0
    assert command.stdout.new_test_returncode == 70


def test_v11_combined_exit_marker_accepts_canonical_explanatory_suffix():
    command = _captured_command(
        stdout=(
            b"[verifier] base pytest rc=1; new pytest rc=0 "
            b"(nonzero on failing tests is normal; graded from XML)\n"
        ),
        chunk_bytes=5,
    )

    assert command.stdout.baseline_test_returncode == 1
    assert command.stdout.new_test_returncode == 0


def test_duplicate_legacy_and_v11_exit_markers_are_ambiguous():
    command = _captured_command(
        stdout=(
            b"[verifier] Baseline exit code: 0\n"
            b"[verifier] New tests exit code: 0\n"
            b"[verifier] base pytest rc=0; new pytest rc=0\n"
        ),
        chunk_bytes=11,
    )

    assert command.stdout.baseline_test_returncode is None
    assert command.stdout.new_test_returncode is None


def test_verify_unresolved_reward_zero(tmp_path: Path, monkeypatch):
    """reward.txt = '0', docker exec rc=0 → emit reward=0 KPI,
    return 1 (build_verify needs non-zero to flip success →
    verify_failed)."""
    _write_reward(tmp_path, "0")
    fake = _fake_docker_exec(rc=0)
    monkeypatch.setattr(subprocess, "Popen", fake)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 1
    payload = json.loads(stdout.getvalue())
    assert payload["values"]["reward"] == 0.0
    assert payload["values"]["command_returncode"] == 0.0
    assert payload["values"]["verify_returncode"] == 0.0


@pytest.mark.parametrize(
    ("reward", "expected_rc"),
    ((1, 0), (0, 1)),
)
def test_verify_v11_json_reward(
    tmp_path: Path,
    monkeypatch,
    reward: int,
    expected_rc: int,
):
    _write_json_reward(
        tmp_path,
        json.dumps(
            {
                "reward": reward,
                "f2p_passed": 3,
                "f2p_total": 3,
                "p2p_passed": 120,
                "p2p_total": 120,
            }
        ),
    )

    rc, payload = _run_with_logs(tmp_path, monkeypatch)

    assert rc == expected_rc
    assert payload["values"]["reward"] == float(reward)
    assert payload["values"]["resolved"] == float(reward)


def test_verify_v11_marker_propagates_to_kpi_and_diagnostics(
    tmp_path: Path,
    monkeypatch,
):
    _write_json_reward(tmp_path, '{"reward":1}')
    command_stdout = (
        "[verifier] base pytest rc=0; new pytest rc=70 "
        "(nonzero on failing tests is normal; graded from XML)\n"
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _fake_docker_exec(rc=0, stdout=command_stdout),
    )
    result_stdout = io.StringIO()

    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(tmp_path),
                },
            }
        ),
        stdout=result_stdout,
    )

    assert rc == 0
    values = json.loads(result_stdout.getvalue())["values"]
    assert values["baseline_test_returncode"] == 0.0
    assert values["new_test_returncode"] == 70.0
    evidence = json.loads(
        (deepswe_verifier._diagnostics_dir(tmp_path) / "result.json").read_text()
    )
    assert evidence["baseline_test_returncode"] == 0
    assert evidence["new_test_returncode"] == 70


@pytest.mark.parametrize(
    "payload",
    (
        "[]",
        "{}",
        '{"reward":"1"}',
        '{"reward":true}',
        '{"reward":1.0}',
        '{"reward":2}',
        '{"reward":1,"reward":0}',
        '{"reward":NaN}',
        '{"reward":1,"partial":1e999}',
    ),
)
def test_verify_v11_json_reward_rejects_noncanonical_authority(
    tmp_path: Path,
    monkeypatch,
    payload: str,
):
    _write_json_reward(tmp_path, payload)

    rc, result = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert result["values"]["reward"] == 0.0


def test_verify_v11_json_reward_rejects_excessive_nesting(
    tmp_path: Path,
    monkeypatch,
):
    _write_json_reward(tmp_path, "[" * 1_200 + "0" + "]" * 1_200)

    rc, result = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert result["values"]["reward"] == 0.0


@pytest.mark.parametrize("failure", ("symlink", "oversize", "invalid_utf8"))
def test_verify_v11_json_reward_rejects_unsafe_or_unreadable_leaf(
    tmp_path: Path,
    monkeypatch,
    failure: str,
):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    reward = verifier / "reward.json"
    if failure == "symlink":
        forged = tmp_path / "forged-reward.json"
        forged.write_text('{"reward":1}', encoding="utf-8")
        reward.symlink_to(forged)
    elif failure == "oversize":
        reward.write_bytes(b"x" * (16 * 1024 + 1))
    else:
        reward.write_bytes(b'{"reward":1,"detail":"\xff"}')

    rc, result = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert result["values"]["reward"] == 0.0


def test_verify_rejects_ambiguous_text_and_json_rewards(
    tmp_path: Path,
    monkeypatch,
):
    _write_reward(tmp_path, "1\n")
    _write_json_reward(tmp_path, '{"reward":1}')

    rc, payload = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert payload["values"]["reward"] == 0.0


# ---------- failure modes ----------


def test_verify_docker_exec_failed(tmp_path: Path, monkeypatch):
    """docker exec returned non-zero (e.g., container died,
    bash error). Emit reward=0, surface the docker rc, return
    non-zero."""
    # NOTE: deliberately do NOT write reward.txt — docker exec
    # failure typically means the verifier shell never reached
    # Step 5.
    fake = _fake_docker_exec(rc=137, stderr="OOMKilled")
    monkeypatch.setattr(subprocess, "Popen", fake)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 137
    payload = json.loads(stdout.getvalue())
    assert payload["values"]["reward"] == 0.0
    assert payload["values"]["verify_returncode"] == 137.0


def test_verify_reward_txt_missing(tmp_path: Path, monkeypatch):
    """docker exec rc=0 but reward.txt absent → verifier exited
    before Step 5 (early-error in test.sh). Emit reward=0, return 1."""
    fake = _fake_docker_exec(rc=0)
    monkeypatch.setattr(subprocess, "Popen", fake)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 1
    payload = json.loads(stdout.getvalue())
    assert payload["values"]["reward"] == 0.0


def test_verify_reward_txt_unparseable(tmp_path: Path, monkeypatch):
    """reward.txt contains something other than '0' or '1' (e.g.
    truncated, race, corrupt). Emit reward=0, return 1."""
    _write_reward(tmp_path, "garbage")
    fake = _fake_docker_exec(rc=0)
    monkeypatch.setattr(subprocess, "Popen", fake)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 1
    payload = json.loads(stdout.getvalue())
    assert payload["values"]["reward"] == 0.0


def test_verify_reward_txt_whitespace(tmp_path: Path, monkeypatch):
    """The upstream echo-written single trailing newline is accepted."""
    _write_reward(tmp_path, "1\n")
    fake = _fake_docker_exec(rc=0)
    monkeypatch.setattr(subprocess, "Popen", fake)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 0


def test_verify_reward_leaf_symlink_is_never_followed(tmp_path: Path, monkeypatch):
    target = tmp_path / "forged-reward"
    target.write_text("1\n", encoding="utf-8")
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "reward.txt").symlink_to(target)

    rc, payload = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert payload["values"]["reward"] == 0.0
    reward = deepswe_verifier.read_reward_artifact(tmp_path)
    assert reward.reward is None
    assert reward.token == ""


def test_verify_reward_parent_symlink_is_never_followed(tmp_path: Path, monkeypatch):
    forged = tmp_path / "forged-verifier"
    forged.mkdir()
    (forged / "reward.txt").write_text("1\n", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "verifier").symlink_to(forged, target_is_directory=True)

    rc, payload = _run_with_logs(logs, monkeypatch)

    assert rc == 1
    assert payload["values"]["reward"] == 0.0


def test_verify_reward_nonregular_leaf_is_rejected(tmp_path: Path, monkeypatch):
    reward = tmp_path / "verifier" / "reward.txt"
    reward.mkdir(parents=True)

    rc, payload = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert payload["values"]["reward"] == 0.0


def test_verify_reward_oversize_is_bounded_and_not_logged(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    secret = "1\nATTACKER_CONTROLLED_SECRET"
    _write_reward(tmp_path, secret)

    rc, payload = _run_with_logs(tmp_path, monkeypatch)

    assert rc == 1
    assert payload["values"]["reward"] == 0.0
    assert secret not in caplog.text
    assert "ATTACKER_CONTROLLED_SECRET" not in caplog.text


def test_workflow_deferred_mode_never_reads_reward_before_stop(
    tmp_path: Path,
    monkeypatch,
):
    _write_reward(tmp_path, "1\n")
    monkeypatch.setattr(subprocess, "Popen", _fake_docker_exec(rc=0))

    def forbidden_read(_logs_dir):
        raise AssertionError("workflow reward must not be read before strict stop")

    monkeypatch.setattr(deepswe_verifier, "read_reward_artifact", forbidden_read)
    stdout = io.StringIO()
    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(tmp_path),
                    "defer_reward_read": True,
                },
            }
        ),
        stdout=stdout,
    )

    assert rc == 0
    values = json.loads(stdout.getvalue())["values"]
    assert values["command_returncode"] == 0.0
    assert values["verify_returncode"] == 0.0
    assert values["reward_pending"] == 1.0
    assert "reward" not in values
    assert "resolved" not in values


def test_bounded_evidence_keeps_edges_and_middle_exit_markers(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    edge = deepswe_verifier._STREAM_EDGE_BYTES
    omitted_secret = b"DO_NOT_PERSIST_MIDDLE_SECRET"
    command_stdout = (
        b"first hidden failure\n"
        + b"A" * (edge + 10_000)
        + b"\n[verifier] Baseline exit code: 0\n"
        + omitted_secret
        + b"B" * (edge + 10_000)
        + b"\n[verifier] New tests exit code: 70\n"
        + b"last hidden failure\n"
    )
    command_stderr = b"stderr-head\n" + b"C" * (3 * edge) + b"\nstderr-tail\n"
    command = _captured_command(
        stdout=command_stdout,
        stderr=command_stderr,
    )
    monkeypatch.setattr(
        deepswe_verifier,
        "_run_docker_exec",
        lambda *_args, **_kwargs: command,
    )

    result_stdout = io.StringIO()
    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(tmp_path),
                    "defer_reward_read": True,
                },
            }
        ),
        stdout=result_stdout,
    )

    assert rc == 0
    values = json.loads(result_stdout.getvalue())["values"]
    assert values["command_returncode"] == 0.0
    assert values["verify_returncode"] == 0.0
    assert values["baseline_test_returncode"] == 0.0
    assert values["new_test_returncode"] == 70.0
    assert values["verifier_stdout_bytes"] == float(len(command_stdout))
    assert values["verifier_stderr_bytes"] == float(len(command_stderr))
    assert values["verifier_stdout_truncated"] == 1.0
    assert values["verifier_stderr_truncated"] == 1.0
    assert values["diagnostics_persisted"] == 1.0

    diagnostics = deepswe_verifier._diagnostics_dir(tmp_path)
    persisted_stdout = (diagnostics / "stdout.log").read_bytes()
    persisted_stderr = (diagnostics / "stderr.log").read_bytes()
    max_evidence = 2 * edge + 256
    assert len(persisted_stdout) <= max_evidence
    assert len(persisted_stderr) <= max_evidence
    assert persisted_stdout.startswith(b"first hidden failure\n")
    assert persisted_stdout.endswith(b"last hidden failure\n")
    assert persisted_stderr.startswith(b"stderr-head\n")
    assert persisted_stderr.endswith(b"\nstderr-tail\n")
    assert omitted_secret not in persisted_stdout
    assert omitted_secret.decode() not in caplog.text

    evidence = json.loads((diagnostics / "result.json").read_text())
    assert evidence["baseline_test_returncode"] == 0
    assert evidence["new_test_returncode"] == 70
    assert evidence["stdout"]["total_bytes"] == len(command_stdout)
    assert evidence["stdout"]["truncated"] is True
    assert evidence["stderr"]["total_bytes"] == len(command_stderr)
    assert evidence["stderr"]["truncated"] is True


def test_duplicate_exit_markers_are_not_promoted(
    tmp_path: Path,
    monkeypatch,
):
    command = _captured_command(
        stdout=(
            b"[verifier] Baseline exit code: 0\n"
            b"[verifier] Baseline exit code: 1\n"
            b"[verifier] New tests exit code: 4\n"
        ),
    )
    monkeypatch.setattr(
        deepswe_verifier,
        "_run_docker_exec",
        lambda *_args, **_kwargs: command,
    )
    result_stdout = io.StringIO()

    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(tmp_path),
                    "defer_reward_read": True,
                },
            }
        ),
        stdout=result_stdout,
    )

    assert rc == 0
    values = json.loads(result_stdout.getvalue())["values"]
    assert "baseline_test_returncode" not in values
    assert values["new_test_returncode"] == 4.0
    evidence = json.loads(
        (deepswe_verifier._diagnostics_dir(tmp_path) / "result.json").read_text()
    )
    assert evidence["baseline_test_returncode"] is None
    assert evidence["new_test_returncode"] == 4


def test_unsafe_diagnostic_directory_fails_closed_without_changing_reward(
    tmp_path: Path,
    monkeypatch,
):
    _write_reward(tmp_path, "1\n")
    forged = tmp_path.parent / "forged-diagnostics"
    forged.mkdir()
    deepswe_verifier._diagnostics_dir(tmp_path).symlink_to(
        forged,
        target_is_directory=True,
    )
    monkeypatch.setattr(subprocess, "Popen", _fake_docker_exec(rc=0))
    result_stdout = io.StringIO()

    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(tmp_path),
                },
            }
        ),
        stdout=result_stdout,
    )

    assert rc == 0
    values = json.loads(result_stdout.getvalue())["values"]
    assert values["reward"] == 1.0
    assert values["diagnostics_persisted"] == 0.0
    assert list(forged.iterdir()) == []


def test_secure_reader_rejects_leaf_replacement_during_read(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.products.deepswe import secure_artifacts

    reward = tmp_path / "verifier" / "reward.txt"
    reward.parent.mkdir()
    reward.write_bytes(b"1\n")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"0\n")
    original_read = secure_artifacts.os.read
    replaced = False

    def racing_read(descriptor, size):
        nonlocal replaced
        data = original_read(descriptor, size)
        if data and not replaced:
            replaced = True
            replacement.replace(reward)
        return data

    monkeypatch.setattr(secure_artifacts.os, "read", racing_read)

    result = secure_artifacts.read_bounded_regular_artifact(
        tmp_path,
        "verifier/reward.txt",
        max_bytes=2,
    )

    assert result.status == secure_artifacts.ArtifactReadStatus.CHANGED
    assert result.data == b""


def test_verify_docker_timeout(tmp_path: Path, monkeypatch):
    """docker exec raises TimeoutExpired → runner surfaces as
    rc=124, emits reward=0, returns non-zero."""
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _fake_docker_exec(rc=0, timeout_on_wait=True),
    )

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 124
    payload = json.loads(stdout.getvalue())
    assert payload["values"]["verify_returncode"] == 124.0


def test_verify_docker_not_on_path(tmp_path: Path, monkeypatch):
    """docker binary missing → returncode 127, no crash."""

    def _enoent(args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'docker'")

    monkeypatch.setattr(subprocess, "Popen", _enoent)

    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 127


def test_verify_refuses_uninspectable_container_environment(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        deepswe_verifier,
        "inspect_docker_container_environment",
        lambda _container_id: (_ for _ in ()).throw(
            deepswe_verifier.ContainerGitEnvironmentError("invalid image env")
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(
            "hidden command must not run without environment authority"
        ),
    )

    stdout = io.StringIO()
    rc = deepswe_verifier.main(
        stdin=_make_stdin(
            {
                "role": "verify",
                "deepswe_metadata": {
                    "container_id": "abc123",
                    "logs_dir": str(tmp_path),
                },
            }
        ),
        stdout=stdout,
    )

    assert rc == 125
    assert json.loads(stdout.getvalue())["values"]["verify_returncode"] == 125.0


# ---------- role + parameter validation ----------


def test_main_rejects_build_role(tmp_path: Path):
    """``role="build"`` is a routing bug — the verifier refuses it
    explicitly (build goes to crucible.common.runners.lang.<lang>
    per spec_synthesis._runners_for_instance)."""
    stdin = _make_stdin(
        {
            "role": "build",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 2


def test_main_rejects_unknown_role(tmp_path: Path):
    """Unknown role values are refused — no silent fallback."""
    stdin = _make_stdin(
        {
            "role": "something-else",
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 2


def test_main_missing_metadata():
    """No ``deepswe_metadata`` parameter → fail fast with rc=2."""
    stdin = _make_stdin({"role": "verify"})
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 2


def test_main_metadata_missing_container_id(tmp_path: Path):
    """``deepswe_metadata`` present but no container_id → rc=2."""
    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {"logs_dir": str(tmp_path)},
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 2


def test_main_metadata_missing_logs_dir():
    """``deepswe_metadata`` present but no logs_dir → rc=2."""
    stdin = _make_stdin(
        {
            "role": "verify",
            "deepswe_metadata": {"container_id": "abc123"},
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 2


def test_main_malformed_stdin():
    """Stdin not parseable as JSON → fail fast with rc=2."""
    stdin = io.StringIO("not json{")
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 2


@pytest.mark.parametrize(
    "stdin",
    [
        pytest.param(
            lambda: io.StringIO("[" * 50_000 + "0" + "]" * 50_000),
            id="recursive-json",
        ),
        pytest.param(lambda: io.BytesIO(b"\xff"), id="invalid-utf8"),
    ],
)
def test_main_decode_failures_exit_2(stdin):
    rc = deepswe_verifier.main(stdin=stdin(), stdout=io.StringIO())

    assert rc == 2


def test_main_default_role_is_verify(tmp_path: Path, monkeypatch):
    """If ``role`` is omitted from the parameters, default to
    ``"verify"`` (same convention as swebench_pytest)."""
    _write_reward(tmp_path, "1")
    monkeypatch.setattr(subprocess, "Popen", _fake_docker_exec(rc=0))

    stdin = _make_stdin(
        {
            # No "role" key.
            "deepswe_metadata": {
                "container_id": "abc123",
                "logs_dir": str(tmp_path),
            },
        }
    )
    stdout = io.StringIO()
    rc = deepswe_verifier.main(stdin=stdin, stdout=stdout)
    assert rc == 0
