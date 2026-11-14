# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""SubprocessTestRunner and CargoTestRunner — default test runners."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from crucible.common.runner_protocol import RunnerCommandContext, RunnerContext, TestResult
from crucible.common.runners.base import BaseRunner

logger = logging.getLogger(__name__)


class SubprocessTestRunner(BaseRunner):
    """Run a test command and return a structured ``TestResult``."""

    def __init__(
        self,
        context: RunnerContext,
        cmd_ctx: RunnerCommandContext | None = None,
    ) -> None:
        super().__init__(context, cmd_ctx=cmd_ctx)

    def test(self) -> TestResult:
        result = self._run_command()
        combined = result.stdout + result.stderr
        if result.misconfigured:
            return TestResult(
                status="misconfigured", exit_code=-1, output=combined,
            )
        if result.timed_out:
            return TestResult(
                status="timed_out", exit_code=-1, output=combined,
            )
        status = "passed" if result.exit_code == 0 else "failed"
        return TestResult(
            status=status, exit_code=result.exit_code, output=combined,
        )


class CargoTestRunner(BaseRunner):
    """Rust-aware test runner that scopes ``cargo test`` to modified crates.

    Reads ``self.context.files_to_modify`` to determine which crates
    were touched, then runs ``cargo test -p <crate>`` for each.  Falls
    back to ``cargo test --workspace`` when no crate can be inferred.
    """

    def __init__(
        self,
        context: RunnerContext,
        cmd_ctx: RunnerCommandContext | None = None,
    ) -> None:
        super().__init__(context, cmd_ctx=cmd_ctx)

    def _infer_crates(self) -> list[str]:
        """Extract crate names from ``files_to_modify`` paths."""
        crates: list[str] = []
        for f in self.context.files_to_modify:
            parts = Path(f).parts
            for i, part in enumerate(parts):
                if part == "crates" and i + 1 < len(parts):
                    crate = parts[i + 1]
                    if crate not in crates:
                        crates.append(crate)
                    break
        return crates

    def _find_cargo(self) -> str:
        cargo_home = Path.home() / ".cargo" / "bin" / "cargo"
        if cargo_home.is_file():
            return str(cargo_home)
        return "cargo"

    def test(self) -> TestResult:
        cargo = self._find_cargo()
        crates = self._infer_crates()

        if crates:
            cmd = [cargo, "test", "--release"]
            for crate in crates:
                cmd.extend(["-p", crate])
            logger.info("CargoTestRunner: testing crates %s", crates)
        else:
            cmd = [cargo, "test", "--workspace"]
            logger.info("CargoTestRunner: testing full workspace")

        result = self._run_command(cmd)
        combined = result.stdout + result.stderr

        if result.misconfigured:
            return TestResult(
                status="misconfigured", exit_code=-1,
                output=combined or result.stderr,
            )
        if result.command_not_found:
            return TestResult(
                status="failed", exit_code=-1,
                output="cargo not found on PATH",
            )
        if result.timed_out:
            return TestResult(
                status="timed_out", exit_code=-1, output=combined,
            )

        tests_run = 0
        tests_failed = 0
        for m in re.finditer(r"(\d+) passed", combined):
            tests_run += int(m.group(1))
        for m in re.finditer(r"(\d+) failed", combined):
            tests_failed += int(m.group(1))
            tests_run += int(m.group(1))

        status = "passed" if result.exit_code == 0 else "failed"
        return TestResult(
            status=status,
            exit_code=result.exit_code,
            output=combined,
            tests_run=tests_run,
            tests_failed=tests_failed,
        )
