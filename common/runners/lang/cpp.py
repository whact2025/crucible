# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""C++ language runner — ``cmake --build`` + ``ctest`` over a build dir.

Subprocess contract (per ``assay.bench``):

- Reads a JSON BenchConfig from stdin.
- The build role runs only ``cmake --build <build_dir>``.
- Verification roles build, then run ``ctest --test-dir <build_dir>``.
- Emits a JSON KPI payload on stdout.
- Exits 0 on success, non-zero on build OR test failure (the runner
  short-circuits on build failure and does not attempt ctest).

Recognized ``parameters``:

- ``build_dir``: str — defaults to ``"build"`` (relative to
  ``working_dir`` or process cwd).
- ``cmake_target``: str — when present, ``cmake --build <build_dir>
  --target <target>``.
- ``ctest_args``: ``list[str]`` — extra args to ctest
  (e.g., ``["-R", "MyTest"]``, ``["-j", "8"]``).
- ``working_dir``: str — overrides process cwd.
- ``project_subdir``: str — confined relative CMake project directory beneath
  ``working_dir``; absolute paths, traversal, and symlink escapes are rejected.
- ``skip_build``: bool — when true, verification skips the cmake step (use
  when another role already built). A build role cannot skip its build.

This adapter does not define a benchmark command. A workflow-owned
``runner_mode == "bench"`` request exits 2 before invoking CMake or CTest;
ordinary tests are never substituted for a requested benchmark.

Emitted ``values``:

- ``cpp_build_returncode``   — cmake exit code.
- ``cpp_ctest_returncode``   — ctest exit code (or -1 when skipped
  due to build failure); absent for a successful build-only invocation.
- ``cpp_tests_passed``       — parsed from ctest summary.
- ``cpp_tests_failed``       — parsed.
- ``cpp_build_duration_s``   — cmake wall time.
- ``cpp_ctest_duration_s``   — ctest wall time.
- ``tests_passed`` / ``tests_failed`` / ``returncode`` — language-
  agnostic aliases for the polyglot aggregator.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from crucible.common.runners.parameters import (
    for_language,
    resolve_project_working_dir,
)


_CTEST_SUMMARY_RE = re.compile(
    r"(\d+)%\s+tests passed,\s+(\d+)\s+tests failed out of\s+(\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Per-language conventions — see python.py for the contract.
# These defaults follow common CMake/CTest layouts. Projects with different
# conventions should provide explicit commands and breach-test paths.
# ---------------------------------------------------------------------------


def breach_test_path(repo_root: Path) -> Path:
    """Path the breach Defender writes attack tests to for a C++ repo.

    Default landing: ``tests/swebench_breach.cpp``. Real C++ projects
    have wildly varying test conventions (gtest vs ctest vs custom);
    operators must override via ``--breach-tests`` when this default
    doesn't fit.
    """
    return repo_root / "tests" / "swebench_breach.cpp"


def build_command(repo_root: Path) -> str:
    """Build command for a C++ project (CMake convention)."""
    return "cmake --build build"


def verify_command(repo_root: Path) -> str:
    """Verify (test) command for a C++ project."""
    return "ctest --test-dir build --output-on-failure"


def _parse_ctest_summary(text: str) -> dict[str, float]:
    """Pull ``X% tests passed, Y tests failed out of Z`` from ctest output."""
    m = _CTEST_SUMMARY_RE.search(text)
    if not m:
        return {}
    failed = int(m.group(2))
    total = int(m.group(3))
    passed = total - failed
    return {
        "tests_passed": float(passed),
        "tests_failed": float(failed),
        "tests_total": float(total),
    }


def _run(cmd: list[str], cwd: str, env: dict[str, str]) -> tuple[int, str, float]:
    start = time.monotonic()
    completed = subprocess.run(
        cmd, cwd=cwd, env=env,
        capture_output=True, text=True, check=False,
    )
    return (
        completed.returncode,
        completed.stdout + completed.stderr,
        time.monotonic() - start,
    )


def main(stdin: object = None, stdout: object = None) -> int:
    stdin_obj = stdin if stdin is not None else sys.stdin
    stdout_obj = stdout if stdout is not None else sys.stdout

    cfg = json.loads(stdin_obj.read())
    raw_parameters = (
        (cfg.get("parameters") or {}) if isinstance(cfg, dict) else {}
    )
    parameters = for_language(raw_parameters, "cpp")

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
                "C++ runner does not support "
                f"role={role!r}, runner_mode={requested_mode!r}; "
                "benchmark verification requires a benchmark-capable adapter"
            ),
        }))
        stdout_obj.flush()
        return 2

    skip_build = bool(parameters.get("skip_build"))
    if role == "build" and skip_build:
        stdout_obj.write(json.dumps({
            "values": {"returncode": 2.0},
            "raw_output": "C++ build role cannot set skip_build",
        }))
        stdout_obj.flush()
        return 2

    try:
        working_dir = str(resolve_project_working_dir(
            parameters,
            process_cwd=os.getcwd(),
        ))
    except ValueError as exc:
        stdout_obj.write(json.dumps({
            "values": {"returncode": 2.0},
            "raw_output": str(exc),
        }))
        stdout_obj.flush()
        return 2
    build_dir = parameters.get("build_dir") or "build"
    env = {**os.environ, **(cfg.get("env") or {})}

    values: dict[str, float] = {}
    combined_output: list[str] = []

    # ---- Build ----
    if not skip_build:
        build_cmd: list[str] = ["cmake", "--build", build_dir]
        if target := parameters.get("cmake_target"):
            build_cmd.extend(["--target", str(target)])
        rc, output, dur = _run(build_cmd, working_dir, env)
        combined_output.append(output)
        values["cpp_build_returncode"] = float(rc)
        values["cpp_build_duration_s"] = float(dur)
        if rc != 0 or role == "build":
            if rc != 0:
                values["cpp_ctest_returncode"] = -1.0
            values["returncode"] = float(rc)
            values["duration_s"] = float(dur)
            stdout_obj.write(json.dumps({
                "values": values,
                "raw_output": "\n".join(combined_output)[-4_000:],
                "runner_mode": expected_mode,
            }))
            stdout_obj.flush()
            return rc

    # ---- ctest ----
    ctest_cmd: list[str] = ["ctest", "--test-dir", build_dir, "--output-on-failure"]
    extra = parameters.get("ctest_args") or []
    if isinstance(extra, list):
        ctest_cmd.extend(str(a) for a in extra)
    rc, output, dur = _run(ctest_cmd, working_dir, env)
    combined_output.append(output)
    values["cpp_ctest_returncode"] = float(rc)
    values["cpp_ctest_duration_s"] = float(dur)
    summary = _parse_ctest_summary(output)
    for key, val in summary.items():
        values[f"cpp_{key}"] = val
        values[key] = val
    values["returncode"] = float(rc)
    values["duration_s"] = float(
        values.get("cpp_build_duration_s", 0.0) + dur
    )

    payload = {
        "values": values,
        "raw_output": "\n".join(combined_output)[-4_000:],
        "runner_mode": expected_mode,
    }
    stdout_obj.write(json.dumps(payload))
    stdout_obj.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
