# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Rust language runner — Cargo build, test, or benchmark gate.

Subprocess contract (per ``assay.bench``):

- Reads a JSON BenchConfig from stdin.
- Runs the workflow-selected Cargo subcommand in ``working_dir`` (or process
  cwd): ``build`` for the build role, ``test`` for ordinary verification, and
  ``bench`` for ``runner_mode == "bench"``.
- Emits a JSON KPI payload on stdout.
- Exits 0 on success, non-zero on test or compile failure.

Recognized ``parameters``:

- ``crate``: str — when present, runs ``cargo test -p <crate>``
  instead of ``cargo test --workspace``.
- ``cargo_features``: ``list[str]`` — feature selection shared by the
  build and test roles, emitted as ``--features <comma-separated-list>``.
- ``cargo_test_args``: ``list[str]`` — arguments appended only to
  ``cargo test``. This supports target selection and harness arguments such
  as ``["--test", "protocol", "--", "--test-threads=1"]``. Breach mode
  independently rejects selectors, filters, and uncaptured-output controls so
  generated adversarial tests cannot be omitted or forge observations.
- ``cargo_bench_args``: ``list[str]`` — arguments appended only to
  ``cargo bench``. These are canonical Spec parameters, never workflow CLI
  state. ``cargo_test_args`` and legacy shared ``test_args`` are deliberately
  not applied to a benchmark invocation.
- ``test_args``: ``list[str]`` — extra args appended to Cargo for the legacy
  build and test modes. This field is retained for callers that need raw
  shared arguments; new callers should use the mode-specific fields above.
- ``working_dir``: str — overrides process cwd.
- ``project_subdir``: str — confined relative Cargo project directory beneath
  ``working_dir``; absolute paths, traversal, and symlink escapes are rejected.

Workflow-owned ``role`` / ``runner_mode`` are not authorable Spec settings.
The runner echoes the executed mode in its result so ``_build_verify`` can
fail closed when an adapter silently ignores a requested benchmark.

Emitted ``values``:

- ``rust_returncode``     — cargo exit code (0 on success).
- ``rust_tests_passed``   — parsed from cargo's ``test result:`` line.
- ``rust_tests_failed``   — parsed.
- ``rust_tests_ignored``  — parsed.
- ``rust_duration_s``     — wall time of the cargo subprocess.
- ``tests_passed`` / ``tests_failed`` / ``returncode`` — language-
  agnostic aliases for the polyglot aggregator.

Note: cargo test prints multiple ``test result:`` lines (one per
crate). The runner sums across them to get workspace totals.
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


_CARGO_RESULT_RE = re.compile(
    r"test result:\s+(?:ok|FAILED)\.\s+"
    r"(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored",
    re.IGNORECASE,
)
_LIBTEST_OBSERVATION_RE = re.compile(
    r"^test (?P<id>(?:[A-Za-z0-9_]+::)*[A-Za-z0-9_]+) "
    r"\.\.\. (?:(?P<status>ok|FAILED)|ignored"
    r"(?:, (?P<ignore_reason>[^\r\n]+))?)$"
)


# ---------------------------------------------------------------------------
# Per-language conventions — see python.py for the contract.
# ---------------------------------------------------------------------------


def breach_test_path(repo_root: Path) -> Path:
    """Path the breach Defender writes attack tests to for a Rust repo.

    Default landing: ``tests/swebench_breach.rs`` at the workspace
    root (cargo integration tests). For workspace repos where the
    affected crate is not the workspace root, callers may need to
    override via ``--breach-tests`` to point inside the specific
    crate (``crates/<name>/tests/swebench_breach.rs``).
    """
    return repo_root / "tests" / "swebench_breach.rs"


def build_command(repo_root: Path) -> str:
    """Canonical default; typed selection comes from Spec parameters."""
    return "cargo build --workspace"


def verify_command(repo_root: Path) -> str:
    """Canonical default; typed selection comes from Spec parameters."""
    return "cargo test --workspace"


def _parse_cargo_summary(text: str) -> dict[str, float]:
    """Sum ``test result: ok. X passed; Y failed; Z ignored`` across lines.

    Cargo emits one summary per crate in a workspace; we aggregate to
    workspace totals so the polyglot aggregator sees one number per
    language.
    """
    passed = failed = ignored = 0
    matched = False
    for m in _CARGO_RESULT_RE.finditer(text):
        matched = True
        passed += int(m.group(1))
        failed += int(m.group(2))
        ignored += int(m.group(3))
    if not matched:
        return {}
    return {
        "tests_passed": float(passed),
        "tests_failed": float(failed),
        "tests_ignored": float(ignored),
    }


def _parse_libtest_observations(text: str) -> list[dict[str, str]]:
    status_map = {
        "ok": "passed",
        "FAILED": "failed",
        "ignored": "not_observed",
    }
    observations: list[dict[str, str]] = []
    for line in text.splitlines():
        match = _LIBTEST_OBSERVATION_RE.fullmatch(line.rstrip())
        if match is None:
            continue
        raw_status = match.group("status") or "ignored"
        observations.append({
            "language": "rust",
            "id": match.group("id"),
            "status": status_map[raw_status],
            "line": line.rstrip(),
        })
    return observations


def _run_exact_libtest_observations(
    parameters: dict,
    *,
    working_dir: str,
    env: dict[str, str],
) -> list[dict[str, str]]:
    from crucible.common.runners.observations import canonical_attack_test_ids

    attack_ids = parameters.get("breach_attack_ids")
    identities = canonical_attack_test_ids(attack_ids, "rust")
    full_command = _build_cmd(parameters)
    delimiter = full_command.index("--") if "--" in full_command else len(full_command)
    cargo_prefix = full_command[:delimiter]
    listed = subprocess.run(
        [*cargo_prefix, "--", "--list"],
        cwd=str(Path(working_dir)),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        raise ValueError(
            "cargo test discovery failed: "
            + (listed.stderr or listed.stdout)[-1000:]
        )
    discovered = {
        line[:-6]
        for line in listed.stdout.splitlines()
        if line.endswith(": test")
    }
    observations: list[dict[str, str]] = []
    for attack_id, canonical_id in identities:
        matches = sorted(
            test_id
            for test_id in discovered
            if test_id.rsplit("::", 1)[-1] == canonical_id
        )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous libtest attack identity {attack_id!r}: {matches!r}"
            )
        if not matches:
            observations.append({
                "language": "rust",
                "id": canonical_id,
                "status": "runner_error",
                "line": (
                    f"{canonical_id} (canonical libtest node absent from "
                    "discovery)"
                ),
            })
            continue
        test_id = matches[0]
        completed = subprocess.run(
            [
                *cargo_prefix,
                test_id,
                "--",
                "--exact",
                "--test-threads=1",
            ],
            cwd=str(Path(working_dir)),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        exact_records = [
            record for record in _parse_libtest_observations(combined)
            if record["id"] == test_id
        ]
        if len(exact_records) != 1:
            status = "runner_error"
            line = (
                f"{test_id} (expected one exact libtest result, "
                f"observed={len(exact_records)}, rc={completed.returncode})"
            )
        else:
            record = exact_records[0]
            observed_status = record["status"]
            consistent = (
                (observed_status == "passed" and completed.returncode == 0)
                or (
                    observed_status == "failed"
                    and completed.returncode == 101
                )
                or (
                    observed_status == "not_observed"
                    and completed.returncode == 0
                )
            )
            status = observed_status if consistent else "runner_error"
            line = record["line"]
        observations.append({
            "language": "rust",
            "id": test_id,
            "status": status,
            "line": line,
        })
    return observations


def _command_mode(parameters: dict) -> str:
    """Map workflow-owned role/mode controls to one Cargo subcommand."""

    role = parameters.get("role") or "verify"
    requested_mode = parameters.get("runner_mode")
    if role == "build":
        if requested_mode not in (None, "build"):
            raise ValueError(
                "Rust runner build role requires runner_mode='build'"
            )
        return "build"
    if role not in {"verify", "dev_verify"}:
        raise ValueError(f"Rust runner role {role!r} is not supported")
    mode = requested_mode or "tests"
    if mode not in {"tests", "bench"}:
        raise ValueError(f"Rust runner mode {mode!r} is not supported")
    return mode


def _build_cmd(parameters: dict) -> list[str]:
    parameters = for_language(parameters, "rust")
    # ``role`` is injected by ``crucible.workflows._build_verify._run_runner``
    # so language runners can pick the right cargo subcommand. Without
    # this, both build and verify roles fall through to ``cargo test``,
    # which makes ``verify_mode: build_only`` slices spuriously gate
    # on the full workspace test suite (compile + run every test in
    # every crate) instead of just compile-checking.
    mode = _command_mode(parameters)
    crate = parameters.get("crate")
    if crate is not None and (not isinstance(crate, str) or not crate):
        raise ValueError("Rust runner parameter 'crate' must be a non-empty string")
    subcommand = {
        "build": "build",
        "tests": "test",
        "bench": "bench",
    }[mode]
    if crate:
        cmd: list[str] = ["cargo", subcommand, "-p", crate]
    else:
        cmd = ["cargo", subcommand, "--workspace"]

    def _string_list(key: str) -> list[str]:
        raw = parameters.get(key)
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)) or not all(
            isinstance(value, str) and value for value in raw
        ):
            raise ValueError(
                f"Rust runner parameter {key!r} must be a string list"
            )
        return list(raw)

    features = _string_list("cargo_features")
    if features:
        cmd.extend(["--features", ",".join(str(feature) for feature in features)])

    # Backward-compatible raw arguments remain shared by the pre-existing
    # build/test roles. They do not leak into the newly added benchmark mode;
    # benchmark selection has its own typed, Spec-bound field.
    extra = _string_list("test_args")
    if mode in {"build", "tests"}:
        cmd.extend(extra)

    # Test target selectors and harness arguments are invalid for
    # ``cargo build``. Apply them only to the verify role so one tier-level
    # selection can safely drive both build and test.
    cargo_test_args = _string_list("cargo_test_args")
    if subcommand == "test":
        rendered_test_args = list(cargo_test_args)
        if parameters.get("breach_observations"):
            forbidden = {"--nocapture", "--show-output"}.intersection(
                rendered_test_args
            )
            if forbidden:
                raise ValueError(
                    "breach observations require captured libtest output; "
                    f"forbidden arguments: {sorted(forbidden)!r}"
                )
            if "--" not in rendered_test_args:
                rendered_test_args.extend(("--", "--test-threads=1"))
            elif not any(
                argument == "--test-threads=1"
                or argument.startswith("--test-threads=")
                for argument in rendered_test_args
            ):
                rendered_test_args.append("--test-threads=1")
        cmd.extend(rendered_test_args)
    cargo_bench_args = _string_list("cargo_bench_args")
    if subcommand == "bench":
        if parameters.get("breach_observations"):
            raise ValueError(
                "breach observations require runner_mode='tests'; "
                "benchmark harness output is not libtest evidence"
            )
        cmd.extend(cargo_bench_args)
    return cmd


def main(stdin: object = None, stdout: object = None) -> int:
    stdin_obj = stdin if stdin is not None else sys.stdin
    stdout_obj = stdout if stdout is not None else sys.stdout

    cfg = json.loads(stdin_obj.read())
    raw_parameters = (
        (cfg.get("parameters") or {}) if isinstance(cfg, dict) else {}
    )
    try:
        parameters = for_language(raw_parameters, "rust")
        cmd = _build_cmd(parameters)
        executed_mode = _command_mode(parameters)
    except ValueError as exc:
        stdout_obj.write(json.dumps({
            "values": {"returncode": 2.0},
            "raw_output": str(exc),
            "test_observations": [],
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
            "test_observations": [],
        }))
        stdout_obj.flush()
        return 2
    env = {**os.environ, **(cfg.get("env") or {})}

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

    combined = completed.stdout + completed.stderr
    summary = _parse_cargo_summary(combined)
    values: dict[str, float] = {
        "rust_returncode": float(completed.returncode),
        "rust_duration_s": float(duration),
        "returncode": float(completed.returncode),
        "duration_s": float(duration),
    }
    for key, val in summary.items():
        values[f"rust_{key}"] = val
        values[key] = val

    payload = {
        "values": values,
        "raw_output": combined[-4_000:],
        "runner_mode": executed_mode,
    }
    result_returncode = completed.returncode
    # Build and verify receive the same breach envelope. Exact libtest
    # discovery is meaningful only for the test-mode verify command; appending
    # `-- --list` to the preceding `cargo build` command is invalid Cargo CLI.
    if parameters.get("breach_observations") and executed_mode == "tests":
        from crucible.common.runners.observations import filter_test_observations

        try:
            payload["test_observations"] = filter_test_observations(
                _run_exact_libtest_observations(
                    parameters,
                    working_dir=str(working_dir),
                    env=env,
                ),
                parameters.get("breach_attack_ids"),
                language="rust",
            )
        except ValueError as exc:
            result_returncode = 2
            values["returncode"] = 2.0
            values["rust_returncode"] = 2.0
            payload["raw_output"] = str(exc)
            payload["test_observations"] = []
    stdout_obj.write(json.dumps(payload))
    stdout_obj.flush()
    return result_returncode


if __name__ == "__main__":
    sys.exit(main())
