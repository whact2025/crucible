# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Runner protocols and structured result types.

Defines the core abstractions for pluggable build, test, verify, and
benchmark runners.  All runners receive a frozen ``RunnerContext`` with
workspace paths and return structured result dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class RunnerContext:
    """Immutable context injected into every runner.

    ``workspace_root`` is the top-level workspace and ``repo_root`` is the
    repository being operated on. ``project_root`` identifies the selected
    project within that repository; ``worktree`` is populated when execution
    is isolated in a Git worktree.
    """

    workspace_root: Path
    repo_root: Path
    project: str
    project_root: Path
    worktree: Path | None
    cwd: Path
    build_artifact_path: str = ""
    files_to_modify: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunnerCommandContext:
    """Command and timeout selected for one runner subprocess."""

    command: tuple[str, ...]
    timeout: float | None


@dataclass
class BuildResult:
    """Normalized outcome from a build runner."""

    status: Literal["ok", "failed", "timed_out", "skipped", "command_not_found", "misconfigured"]
    exit_code: int
    output: str
    artifact_path: str = ""


@dataclass
class TestResult:
    """Normalized outcome from a test runner."""

    status: Literal["passed", "failed", "timed_out", "skipped", "misconfigured"]
    exit_code: int
    output: str
    tests_run: int = 0
    tests_failed: int = 0


@dataclass
class VerifyResult:
    """Normalized outcome from a verification runner."""

    status: Literal["passed", "failed", "timed_out", "skipped", "misconfigured"]
    exit_code: int
    output: str


@runtime_checkable
class BuildRunnerProtocol(Protocol):
    """Structural interface for build runners."""

    context: RunnerContext

    def build(self) -> BuildResult: ...


@runtime_checkable
class TestRunnerProtocol(Protocol):
    """Structural interface for test runners."""

    context: RunnerContext

    def test(self) -> TestResult: ...


@runtime_checkable
class VerifyRunnerProtocol(Protocol):
    """Structural interface for verification runners."""

    context: RunnerContext

    def verify(self) -> VerifyResult: ...


@runtime_checkable
class BenchmarkRunnerProtocol(Protocol):
    """Structural interface for benchmark runners.

    Concrete products define the object returned by :meth:`run`; callers that
    need a specific result type must narrow it at the product boundary.
    """

    context: RunnerContext

    def run(self) -> object: ...

    def cleanup(self) -> None: ...
