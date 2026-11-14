# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""BaseRunner — shared subprocess execution for all runner types.

Centralises timeout handling, output capture, environment setup, and
logging so that concrete runners only implement their action method.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from crucible.common.runner_protocol import RunnerCommandContext, RunnerContext

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 200_000

# The crucible interpreter's bin/ — ``sys.prefix / "bin"`` is the venv's
# bin where ``pytest``, ``ruff``, etc. live.  Spec ``test_command``
# entries like ``[pytest]`` would otherwise fail with "command not found"
# in subprocess shells (no PATH inheritance from venv activation).
# Using ``sys.prefix`` rather than ``Path(sys.executable).resolve()``
# because uv-managed venvs symlink ``.venv/bin/python`` → the
# uv-managed interpreter dir, which does NOT contain pytest; the venv
# bin/ does.
_CRUCIBLE_VENV_BIN = str(Path(sys.prefix) / "bin")


@dataclass
class CommandResult:
    """Internal result from a single subprocess invocation."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    command_not_found: bool = False
    misconfigured: bool = False
    duration_s: float = 0.0


class BaseRunner:
    """Shared subprocess machinery for build/test/verify runners.

    Subclasses set ``self.command`` and ``self.timeout`` at construction,
    then call ``_run_command()`` from their action method.

    Passing cmd_ctx with timeout=None is a misuse signal and will cause
    _run_command to refuse to spawn; pass cmd_ctx=None to inherit the
    subclass default.
    """

    DEFAULT_TIMEOUT: float = 1800.0

    def __init__(
        self,
        context: RunnerContext,
        cmd_ctx: RunnerCommandContext | None = None,
    ) -> None:
        self.context = context
        if cmd_ctx is None:
            self.command: list[str] = []
            self.timeout = self.DEFAULT_TIMEOUT
            self._timeout_misconfigured = False
        elif cmd_ctx.timeout is None:
            self.command = list(cmd_ctx.command)
            self.timeout = self.DEFAULT_TIMEOUT
            self._timeout_misconfigured = True
        else:
            self.command = list(cmd_ctx.command)
            self.timeout = cmd_ctx.timeout
            self._timeout_misconfigured = False

    def _build_env(self) -> dict[str, str]:
        """Merge ``os.environ`` with Crucible workspace variables."""
        env = os.environ.copy()
        env["CRUCIBLE_WORKSPACE"] = str(self.context.workspace_root)
        env["CRUCIBLE_REPO_ROOT"] = str(self.context.repo_root)
        # Repository-controlled hooks are executable code.  Deterministic
        # workflow/build subprocesses must never run them on the host.
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "core.hooksPath"
        env["GIT_CONFIG_VALUE_0"] = "/dev/null"
        if self.context.worktree:
            env["CRUCIBLE_WORKTREE"] = str(self.context.worktree)
        else:
            env["CRUCIBLE_WORKTREE"] = str(self.context.cwd)
        existing_path = env.get("PATH", "")
        if _CRUCIBLE_VENV_BIN and _CRUCIBLE_VENV_BIN not in existing_path.split(os.pathsep):
            env["PATH"] = (
                _CRUCIBLE_VENV_BIN + os.pathsep + existing_path
                if existing_path else _CRUCIBLE_VENV_BIN
            )
        return env

    def _run_command(
        self,
        cmd: list[str] | None = None,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Spawn a subprocess and return a structured result.

        Never raises — returns ``CommandResult`` with status flags on
        timeout or missing executable.
        """
        effective_cmd = cmd or self.command
        if not effective_cmd:
            return CommandResult(
                exit_code=-1, stdout="", stderr="no command specified",
                command_not_found=True,
            )

        # Refuse to spawn a subprocess if the runner was configured with
        # ``timeout=None`` and the caller did not supply an explicit override.
        # Allowing this would silently disable the kill switch for the run.
        if timeout is None and getattr(self, "_timeout_misconfigured", False):
            return CommandResult(
                exit_code=-1, stdout="",
                stderr=(
                    "refusing to run subprocess: RunnerCommandContext.timeout "
                    "was None (no kill window configured)"
                ),
                misconfigured=True,
            )

        effective_cwd = cwd or self.context.cwd
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_env = env or self._build_env()

        logger.info(
            "Running: %s (cwd=%s, timeout=%.0fs)",
            " ".join(effective_cmd), effective_cwd, effective_timeout,
        )
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                effective_cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=str(effective_cwd),
                env=effective_env,
            )
            duration = time.monotonic() - t0
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if len(stdout) > _MAX_OUTPUT_CHARS:
                stdout = stdout[-_MAX_OUTPUT_CHARS:]
            if len(stderr) > _MAX_OUTPUT_CHARS:
                stderr = stderr[-_MAX_OUTPUT_CHARS:]
            return CommandResult(
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - t0
            partial_out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            partial_err = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandResult(
                exit_code=-1,
                stdout=partial_out[-_MAX_OUTPUT_CHARS:],
                stderr=partial_err[-_MAX_OUTPUT_CHARS:],
                timed_out=True,
                duration_s=duration,
            )
        except FileNotFoundError:
            duration = time.monotonic() - t0
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"command not found: {effective_cmd[0]!r}",
                command_not_found=True,
                duration_s=duration,
            )
