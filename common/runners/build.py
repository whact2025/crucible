# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""SubprocessBuildRunner — default build runner backed by a command list."""

from __future__ import annotations

from crucible.common.runner_protocol import BuildResult, RunnerCommandContext, RunnerContext
from crucible.common.runners.base import BaseRunner


class SubprocessBuildRunner(BaseRunner):
    """Run a build command and return a structured ``BuildResult``."""

    def __init__(
        self,
        context: RunnerContext,
        cmd_ctx: RunnerCommandContext | None = None,
    ) -> None:
        super().__init__(context, cmd_ctx=cmd_ctx)

    def build(self) -> BuildResult:
        result = self._run_command()
        combined = result.stdout + result.stderr
        if result.misconfigured:
            return BuildResult(
                status="misconfigured", exit_code=-1, output=combined,
            )
        if result.command_not_found:
            return BuildResult(
                status="command_not_found", exit_code=-1, output=combined,
            )
        if result.timed_out:
            return BuildResult(
                status="timed_out", exit_code=-1, output=combined,
            )
        status = "ok" if result.exit_code == 0 else "failed"
        return BuildResult(
            status=status,
            exit_code=result.exit_code,
            output=combined,
            artifact_path=self.context.build_artifact_path,
        )
