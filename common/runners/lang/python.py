# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Python language runner — ``uv run pytest`` over a project.

Subprocess contract (per ``assay.bench``):

- Reads a JSON BenchConfig from stdin.
- Runs the configured pytest command in ``working_dir`` (or process cwd).
- Emits a JSON KPI payload on stdout.
- Exits 0 on success (pytest returncode 0), non-zero on failure.

Recognized ``parameters``:

- ``build_command``: ``list[str]`` — defaults to
  ``["pip", "install", "-e", "."]``. Projects whose committed lockfile
  depends on sibling/path sources should select their repository-native
  locked command explicitly (for example ``["uv", "sync", "--locked"]``).
- ``build_args``: ``list[str]`` — appended to ``build_command``.
- ``test_command``: ``list[str]`` — defaults to
  ``["uv", "run", "pytest", "-q", "--maxfail=16"]``.  The shared
  development-verification cap reports a repairable failure batch without
  allowing a pathological suite to produce unbounded failure output.
- ``test_args``: ``list[str]`` — appended to ``test_command``.
- ``per_language.python.breach_test_command``: static ``list[str]`` selecting
  the Python/pytest environment used only for exact breach observations. It
  never replaces or alters the project ``test_command``.
- ``working_dir``: str — overrides process cwd.
- ``project_subdir``: str — confined relative project directory beneath
  ``working_dir``; absolute paths, traversal, and symlink escapes are rejected.

Emitted ``values``:

- ``python_returncode``  — pytest exit code (0 on success).
- ``python_tests_passed`` — parsed from the pytest summary line.
- ``python_tests_failed`` — parsed from the pytest summary line.
- ``python_tests_skipped``— parsed from the pytest summary line.
- ``python_duration_s``  — wall time of the pytest subprocess.
- ``tests_passed`` / ``tests_failed`` / ``returncode`` — language-
  agnostic aliases. ``_build_verify``'s classification logic reads
  ``tests_failed`` directly so the polyglot aggregator can preserve
  that key across languages without name-mangling.

The runner does NOT classify "soft fail vs hard fail" — that
belongs to the workflow layer. Non-zero pytest exit fails the
subprocess; the polyglot runner aggregates.

This adapter does not implement benchmark execution. A workflow-owned
``runner_mode == "bench"`` request exits 2 before invoking pytest; it never
silently substitutes tests for a benchmark gate.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from crucible.common.runners.breach_observation import (
    BREACH_OBSERVATION_FAILURE_KEY,
    BreachObservationChannelError,
    BreachObservationConfigurationError,
    BreachObservationInfrastructureError,
    PythonObservationBinding,
    assert_python_observation_binding,
    breach_observation_failure,
    preflight_python_observation_channel,
    pytest_python_prefix,
)
from crucible.common.runners.parameters import (
    for_language,
    resolve_project_working_dir,
)
from crucible.common.runners.pytest_policy import PYTEST_DEV_VERIFY_MAXFAIL_ARG


_DEFAULT_TEST_COMMAND = [
    "uv", "run", "pytest", "-q", PYTEST_DEV_VERIFY_MAXFAIL_ARG,
]
_DEFAULT_BUILD_COMMAND = "pip install -e ."

_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_PYTEST_SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)
# ---------------------------------------------------------------------------
# Per-language conventions consumed by SWE-bench synthesis and breach.
# These helpers are language-uniform contracts (every lang/<X>.py module
# exposes the same three functions) so workflows that need
# "what's the test runner for language X?" get one place to ask. Used by
# products/swebench/spec_synthesis.py for synthesized runners and by breach's
# Defender for per-language attack-test file paths.
# ---------------------------------------------------------------------------


def breach_test_path(repo_root: Path) -> Path:
    """Path the breach Defender writes attack tests to for a Python repo.

    The path is under ``tests/`` because every Python project SWE-bench
    Verified covers uses ``pytest`` test discovery against ``tests/``
    (or a top-level test directory pytest auto-discovers). The
    ``swebench_breach`` prefix avoids colliding with the project's own
    test files and survives the SWE-bench eval harness's test_patch
    layering (which targets the project's existing test files, not
    new ones the agent added).
    """
    return repo_root / "tests" / "test_swebench_breach.py"


def build_command(repo_root: Path) -> str:
    """Build command for a Python project (editable install of the local
    source so tests can resolve imports).

    SWE-bench Verified repos universally have a ``setup.py`` or
    ``pyproject.toml`` at the repo root and expect to be pip-installed
    in editable mode before tests run. ``-e .`` is the lowest-common-
    denominator. Projects with a more complex install path (django's
    nested module structure, sphinx's optional deps) are still
    importable from this baseline because the eval harness's Docker
    image pre-installs the dev deps.
    """
    return "pip install -e ."


def verify_command(repo_root: Path) -> str:
    """Verify (test) command for a Python project.

    Plain ``pytest`` against the repo root. The agent's tier-cascade
    The workflow build/verify pass uses this to run the project's
    *visible* tests as a regression check between breach rounds.
    The hidden ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` tests are the
    leaderboard ground truth, scored separately by the SWE-bench
    eval harness — outside the cascade.
    """
    return f"pytest {PYTEST_DEV_VERIFY_MAXFAIL_ARG}"


def _parse_pytest_summary(text: str) -> dict[str, float]:
    """Extract ``N passed / M failed / K skipped`` counts from pytest output."""
    out: dict[str, float] = {}
    last_summary = ""
    for line in text.splitlines():
        if "passed" in line or "failed" in line or "skipped" in line:
            last_summary = line
    if not last_summary:
        return out
    if m := _PYTEST_PASSED_RE.search(last_summary):
        out["tests_passed"] = float(m.group(1))
    if m := _PYTEST_FAILED_RE.search(last_summary):
        out["tests_failed"] = float(m.group(1))
    if m := _PYTEST_SKIPPED_RE.search(last_summary):
        out["tests_skipped"] = float(m.group(1))
    return out


def _pytest_python_prefix(
    command_prefix: list[str],
    *,
    cwd: str | Path | None = None,
) -> list[str]:
    """Resolve the Python environment that owns a configured pytest command."""

    return pytest_python_prefix(
        command_prefix,
        cwd=cwd or os.getcwd(),
        explicit=False,
    )


def _run_exact_pytest_observations(
    command_prefix: list[str] | None = None,
    *,
    python_prefix: list[str] | None = None,
    observation_binding: PythonObservationBinding | None = None,
    working_dir: str,
    env: dict[str, str],
    test_file: object,
    attack_ids: object,
) -> list[dict[str, Any]]:
    from crucible.common.runners.observations import (
        canonical_attack_test_ids,
        exact_pytest_command,
        pytest_observer_failure,
        pytest_observer_status,
    )

    if not isinstance(test_file, str) or not test_file:
        raise BreachObservationChannelError(
            "breach_observations requires breach_test_file"
        )
    try:
        identities = canonical_attack_test_ids(attack_ids, "python")
    except ValueError as exc:
        raise BreachObservationChannelError(str(exc)) from exc
    if python_prefix is None:
        if command_prefix is None:
            raise BreachObservationChannelError(
                "exact pytest observation requires a Python binding"
            )
        python_prefix = _pytest_python_prefix(
            command_prefix,
            cwd=working_dir,
        )
    observations: list[dict[str, Any]] = []
    for _attack_id, test_id in identities:
        if observation_binding is not None:
            assert_python_observation_binding(observation_binding)
        node_id = f"{test_file}::{test_id}"
        completed = subprocess.run(
            exact_pytest_command(node_id, python_prefix=python_prefix),
            cwd=str(Path(working_dir)),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        status = pytest_observer_status(completed.returncode)
        if status is None:
            raise BreachObservationChannelError(
                "exact pytest observer did not emit a reserved outcome: "
                f"node={node_id!r}, returncode={completed.returncode}, "
                f"stderr={completed.stderr[-1000:]!r}"
            )
        failure = pytest_observer_failure(
            completed.stdout or "",
            expected_node_id=node_id,
        )
        if failure is not None and status not in {"failed", "runner_error"}:
            raise BreachObservationChannelError(
                "exact pytest observer emitted failure metadata for "
                f"incompatible status: node={node_id!r}, status={status!r}"
            )
        observation: dict[str, Any] = {
            "language": "python",
            "id": node_id,
            "status": status,
            "line": node_id,
        }
        if failure is not None:
            observation["failure"] = failure
        observations.append(observation)
    return observations


def main(stdin: object = None, stdout: object = None) -> int:
    stdin_obj = stdin if stdin is not None else sys.stdin
    stdout_obj = stdout if stdout is not None else sys.stdout

    cfg = json.loads(stdin_obj.read())
    raw_parameters = (
        (cfg.get("parameters") or {}) if isinstance(cfg, dict) else {}
    )
    parameters = for_language(raw_parameters, "python")

    # ``role`` is injected by ``_build_verify._run_runner`` so a single
    # runner module can serve both the "build" and "verify" phases for
    # a Python project — the SWE-bench shape, where every Verified
    # instance needs the same ``pip install -e .`` build then a
    # bounded multi-failure pytest regression check. Role-unaware callers (the
    # pre-SWE-bench bench fan-out path) still hit verify by default.
    role = parameters.get("role", "verify")
    requested_mode = parameters.get("runner_mode")
    expected_mode = "build" if role == "build" else "tests"
    if role not in {"build", "verify", "dev_verify"} or requested_mode not in {
        None,
        expected_mode,
    }:
        stdout_obj.write(json.dumps({
            "values": {"returncode": 2.0},
            "raw_output": (
                "Python runner does not support "
                f"role={role!r}, runner_mode={requested_mode!r}; "
                "benchmark verification requires a benchmark-capable adapter"
            ),
        }))
        stdout_obj.flush()
        return 2
    if role == "build":
        cmd_base = list(
            parameters.get("build_command")
            or shlex.split(_DEFAULT_BUILD_COMMAND)
        )
        cmd_args = list(parameters.get("build_args") or [])
    else:
        cmd_base = list(parameters.get("test_command") or _DEFAULT_TEST_COMMAND)
        cmd_args = list(parameters.get("test_args") or [])
    cmd = cmd_base + cmd_args

    try:
        working_dir = str(resolve_project_working_dir(
            parameters,
            process_cwd=os.getcwd(),
        ))
    except ValueError as exc:
        stdout_obj.write(json.dumps({
            "values": {"returncode": 2.0},
            "raw_output": str(exc),
            "test_observations": [],
        }))
        stdout_obj.flush()
        return 2
    env = {**os.environ, **(cfg.get("env") or {})}

    observation_binding: PythonObservationBinding | None = None
    if role != "build" and parameters.get("breach_observations"):
        try:
            observation_binding = preflight_python_observation_channel(
                parameters,
                default_test_command=_DEFAULT_TEST_COMMAND,
                cwd=working_dir,
                candidate_root=os.getcwd(),
                env=env,
            )
        except BreachObservationConfigurationError as exc:
            failure = breach_observation_failure("configuration", exc)
            stdout_obj.write(json.dumps({
                "values": {
                    "returncode": 2.0,
                    "python_returncode": 2.0,
                },
                "raw_output": failure["detail"],
                "runner_mode": expected_mode,
                "test_observations": [],
                BREACH_OBSERVATION_FAILURE_KEY: failure,
            }))
            stdout_obj.flush()
            return 2
        except BreachObservationInfrastructureError as exc:
            failure = breach_observation_failure("infrastructure", exc)
            stdout_obj.write(json.dumps({
                "values": {
                    "returncode": 2.0,
                    "python_returncode": 2.0,
                },
                "raw_output": failure["detail"],
                "runner_mode": expected_mode,
                "test_observations": [],
                BREACH_OBSERVATION_FAILURE_KEY: failure,
            }))
            stdout_obj.flush()
            return 2

    start = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=str(Path(working_dir)),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.monotonic() - start
    reporter_observations: list[dict[str, Any]] = []
    reporter_failure: dict[str, str] | None = None
    if role != "build" and parameters.get("breach_observations"):
        assert observation_binding is not None
        try:
            reporter_observations = _run_exact_pytest_observations(
                python_prefix=list(observation_binding.python_prefix),
                observation_binding=observation_binding,
                working_dir=str(working_dir),
                env=env,
                test_file=parameters.get("breach_test_file"),
                attack_ids=parameters.get("breach_attack_ids"),
            )
        except BreachObservationChannelError as exc:
            reporter_failure = breach_observation_failure("channel", exc)
        except ValueError as exc:
            reporter_failure = breach_observation_failure("channel", exc)
        except BreachObservationInfrastructureError as exc:
            reporter_failure = breach_observation_failure(
                "infrastructure",
                exc,
            )
        except OSError as exc:
            reporter_failure = breach_observation_failure(
                "infrastructure",
                f"{type(exc).__name__}: {exc}",
            )

    combined = completed.stdout + completed.stderr
    summary = _parse_pytest_summary(combined)
    values: dict[str, float] = {
        "python_returncode": float(completed.returncode),
        "python_duration_s": float(duration),
        "returncode": float(completed.returncode),
        "duration_s": float(duration),
    }
    for key, val in summary.items():
        values[f"python_{key}"] = val
        values[key] = val

    payload = {
        "values": values,
        "raw_output": combined[-4_000:],
        "runner_mode": expected_mode,
    }
    result_returncode = completed.returncode
    if role != "build" and parameters.get("breach_observations"):
        from crucible.common.runners.observations import filter_test_observations

        try:
            payload["test_observations"] = filter_test_observations(
                reporter_observations or [],
                parameters.get("breach_attack_ids"),
                language="python",
            )
            if reporter_failure is not None:
                payload[BREACH_OBSERVATION_FAILURE_KEY] = reporter_failure
                raise ValueError(reporter_failure["detail"])
        except ValueError as exc:
            if reporter_failure is None:
                reporter_failure = breach_observation_failure("channel", exc)
                payload[BREACH_OBSERVATION_FAILURE_KEY] = reporter_failure
            result_returncode = 2
            values["returncode"] = 2.0
            values["python_returncode"] = 2.0
            payload["raw_output"] = str(exc)
            payload["test_observations"] = []
    stdout_obj.write(json.dumps(payload))
    stdout_obj.flush()
    return result_returncode


if __name__ == "__main__":
    sys.exit(main())
