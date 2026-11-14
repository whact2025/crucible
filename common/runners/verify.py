# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""SubprocessVerifyRunner — default verify runner backed by a command list."""

from __future__ import annotations

from crucible.common.runner_protocol import RunnerCommandContext, RunnerContext, VerifyResult
from crucible.common.runners.base import BaseRunner


class SubprocessVerifyRunner(BaseRunner):
    """Run a verification command and return a structured ``VerifyResult``."""

    DEFAULT_TIMEOUT: float = 600.0

    def __init__(
        self,
        context: RunnerContext,
        cmd_ctx: RunnerCommandContext | None = None,
    ) -> None:
        super().__init__(context, cmd_ctx=cmd_ctx)

    def verify(self) -> VerifyResult:
        result = self._run_command()
        combined = result.stdout + result.stderr
        if result.misconfigured:
            return VerifyResult(
                status="misconfigured", exit_code=-1, output=combined,
            )
        if result.timed_out:
            return VerifyResult(
                status="timed_out", exit_code=-1, output=combined,
            )
        status = "passed" if result.exit_code == 0 else "failed"
        return VerifyResult(
            status=status, exit_code=result.exit_code, output=combined,
        )
