# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Container-only visible-test runner for SWE-bench authoring.

The official evaluator owns all hidden test patches and expected-test IDs.
This runner accepts only public repository identity plus a live authoring
container id. Any hidden evaluation field is rejected before a subprocess
can run, and there is no host or virtualenv fallback.

Implicit direct-pytest and Django-native verification is conservatively
narrowed to visible, tracked tests matching every changed Python source.
Ambiguous, incomplete, or unsafe selection falls back to the configured full
suite and a clean-base comparison.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, TextIO

from crucible.common.benchmark_runtime import (
    canonical_fingerprint,
    label_args,
    managed_container_labels,
    managed_container_name,
    reconcile_orphaned_containers,
)
from crucible.common.runners.breach_observation import (
    BREACH_OBSERVATION_FAILURE_KEY,
    BreachObservationChannelError,
    breach_observation_failure,
)
from crucible.common.runners.parameters import for_language
from crucible.common.runners.container_exec import (
    ContainerExecProfile,
    run_contained_docker_exec,
)
from crucible.products.swebench.instance_container import CONDA_ACTIVATE_PREFIX
from crucible.products.swebench.paths import is_standard_test_path


_HIDDEN_FIELDS = frozenset({
    "test_patch", "fail_to_pass", "pass_to_pass",
    "FAIL_TO_PASS", "PASS_TO_PASS", "patch",
})
SWEBENCH_EXEC_PROFILE = ContainerExecProfile(
    name="swebench",
    workspace_destination="/testbed",
    git_destination="/testbed/.git",
    logs_destination=None,
    allowed_phases=frozenset({"authoring", "visible-verifier"}),
    helper_cache_env="CRUCIBLE_SWEBENCH_HELPER_DIR",
)
_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_PYTEST_ERROR_RE = re.compile(r"(\d+)\s+error", re.IGNORECASE)
_PYTEST_SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)
_PYTEST_TERMINAL_RE = re.compile(
    r"^(?P<counts>.+?)\s+in\s+\d+(?:\.\d+)?(?:s|\s+seconds?)"
    r"(?:\s+\([^\n]*\))?$"
)
_PYTEST_COUNT_RE = re.compile(
    r"(?P<count>\d{1,9})\s+"
    r"(?P<kind>passed|failed|errors?|skipped|xfailed|xpassed|deselected|"
    r"warnings?|reruns?)"
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PYTEST_EXECUTABLES = frozenset({"pytest", "py.test"})
_MAX_CHANGED_PATHS = 16
_MAX_VISIBLE_PATHS = 100_000
_MAX_SELECTION_PATH_CHARS = 240
_MAX_NEARBY_TEST_DISTANCE = 6
_MAX_FAILURE_RECORDS = 2_048
_MAX_FAILURE_NODE_CHARS = 2_000
_MAX_FAILURE_REASON_CHARS = 2_000
_RAW_OUTPUT_LIMIT = 8_000
_BASELINE_TIMEOUT_S = 1_800
_BASELINE_CLEANUP_RESERVE_S = 15.0
_DOCKER_CONTROL_TIMEOUT_S = 30.0
_BASELINE_TESTBED_TMPFS_MB = 4_096
_BASELINE_TMP_TMPFS_MB = 4_096
_BASELINE_CONTAINER_SCOPE = "crucible.swebench.visible-baseline.v1"
_BASELINE_TESTBED = "/testbed"
_BASELINE_SEED_TARGET = "/.crucible-baseline-seed"
_BASELINE_SOURCE_WORKTREE = "/.crucible-baseline-source/worktree"
_BASELINE_SOURCE_GIT_DIR = "/.crucible-baseline-source/git"
_BASELINE_CACHE_SCHEMA = "crucible.swebench.visible-baseline.v7"
_OUTCOME_PARSER_SCHEMA = "crucible.swebench.visible-outcome.v7"
_BASELINE_CACHE_ENV = "CRUCIBLE_SWEBENCH_BASELINE_CACHE"
_MAX_BIND_SOURCE_CHARS = 4_096
_IMMUTABLE_IMAGE_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")
_CANDIDATE_PATCH_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTAINER_REFERENCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
_SYMPY_NATIVE_COMMAND_PREFIX = (
    "PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning",
    "bin/test",
    "-C",
    "--verbose",
)
_SYMPY_NATIVE_HASH_SEED = "PYTHONHASHSEED=1"
_SYMPY_NATIVE_TEST_SEED = "--seed=1"
_SYMPY_NATIVE_TRAILER_RE = re.compile(
    r"^=*[ \t]*tests finished: (?P<passed>\d{1,9}) passed,[ \t\r\n]*"
    r"(?:(?P<failed_1>\d{1,9}) failed,[ \t\r\n]*)?"
    r"(?:(?P<failed_2>\d{1,9}) failed,[ \t\r\n]*)?"
    r"(?:(?P<skipped>\d{1,9}) skipped,[ \t\r\n]*)?"
    r"(?:(?P<expected_failed>\d{1,9}) expected to fail,[ \t\r\n]*)?"
    r"(?:(?P<unexpected_successes>\d{1,9}) expected to fail but passed,"
    r"[ \t\r\n]*)?"
    r"(?:(?P<exceptions>\d{1,9}) exceptions,[ \t\r\n]*)?"
    r"in \d+(?:\.\d+)? seconds[ \t]*(?:=+)?\r?\n"
    r"(?P<do_not_commit>DO \*NOT\* COMMIT!\r?\n)?\Z",
    re.MULTILINE,
)
_SYMPY_NATIVE_FAILURE_HEADER_RE = re.compile(
    r"^_{20,}\r?\n_+[ \t]+"
    r"(?P<node>sympy/[A-Za-z0-9_.\-/]+\.py"
    r"(?::[A-Za-z_][A-Za-z0-9_]*)?)"
    r"[ \t]+_+\r?\n",
    re.MULTILINE,
)
_PYTEST_FAILURE_RE = re.compile(r"^(FAILED|ERROR)\s+(.+)$")
_PYTEST_SHORT_SUMMARY_HEADER_RE = re.compile(
    r"^=+\s+short test summary info\s+=+$"
)
_PYTEST_SHORT_SUMMARY_RECORD_RE = re.compile(
    r"^(?:PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS|RERUN)\b"
)
_PYTEST_SHORT_SUMMARY_POSTAMBLE_RE = re.compile(
    r"^!+\s+Interrupted:\s+\d{1,9}\s+errors?\s+during\s+collection\s+!+$"
)
_DJANGO_TRAILER_PREFIX = (
    r"^-{20,}\nRan\s+(?P<ran>\d{1,9})\s+tests?\s+in\s+[^\n]+\n\n"
    r"(?P<status>OK|FAILED)(?:\s*\((?P<counts>[^\n]*)\))?"
)
_DJANGO_DATABASE_POSTAMBLE = (
    r"(?:\n(?:Destroying|Preserving) test database for alias "
    r"'[^'\n]{1,128}' \([^\n]{1,1000}\)\.\.\.)*"
)
_DJANGO_TRAILER_RE = re.compile(
    _DJANGO_TRAILER_PREFIX + _DJANGO_DATABASE_POSTAMBLE + r"\s*\Z",
    re.MULTILINE,
)
_DJANGO_PRIOR_TRAILER_RE = re.compile(
    _DJANGO_TRAILER_PREFIX + r"[^\S\n]*(?:\n|\Z)",
    re.MULTILINE,
)
_DJANGO_FAILURE_HEADER_RE = re.compile(
    r"^={20,}\n(ERROR|FAIL):\s+([^\n]+)\n"
    r"(?:[^\n]*\n){0,16}?^-{20,}\n",
    re.MULTILINE,
)
_DJANGO_RESULT_COUNT_RE = re.compile(
    r"(failures|errors|skipped|expected failures|"
    r"unexpected successes)=(\d{1,9})"
)
_VOLATILE_FAILURE_TEXT_RE = re.compile(
    r"(?:0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?s\b)"
)
_SELECTION_PROVENANCE = (
    "container git diff --name-only HEAD -- (NUL-delimited); "
    "candidate tests from container git ls-files"
)
_DJANGO_VISIBLE_TEST_APP_MAP = {
    "django/core/validators.py": "validators",
    "django/db/models/sql/compiler.py": "queries",
    "django/db/models/sql/query.py": "queries",
}
_ASTROPY_VISIBLE_TEST_PATH_MAP = {
    "astropy/timeseries/core.py": (
        "astropy/timeseries/tests/test_common.py",
        "astropy/timeseries/tests/test_sampled.py",
        "astropy/timeseries/tests/test_binned.py",
    ),
}
_PYTEST_VISIBLE_TEST_PATH_MAP = {
    "src/_pytest/logging.py": (
        "testing/logging/test_fixture.py",
        "testing/logging/test_formatter.py",
        "testing/logging/test_reporting.py",
    ),
    "src/_pytest/unittest.py": ("testing/test_unittest.py",),
}
_SYMPY_VISIBLE_TEST_PATH_MAP = {
    "sympy/geometry/point.py": ("sympy/geometry/tests/test_point.py",),
    "sympy/core/function.py": (
        "sympy/core/tests/test_function.py",
        "sympy/utilities/tests/test_lambdify.py",
    ),
}
_TRUSTED_RUNNER_CONTROL_PATHS = frozenset({".crucible_build.lock"})


@dataclasses.dataclass(frozen=True)
class _VisibleTestSelection:
    """Public-repository-only provenance for one visible-test command."""

    mode: str
    reason: str
    changed_source_count: int = 0
    targets: tuple[str, ...] = ()
    git_diff_attempted: bool = False


@dataclasses.dataclass(frozen=True)
class _PytestOutcome:
    """Cache-safe test outcome derived only from visible runner output."""

    returncode: int
    # (node id, FAILED|ERROR, normalized short-summary reason)
    failures: tuple[tuple[str, str, str], ...]
    summary: tuple[tuple[str, float], ...]
    output_sha256: str
    parser_kind: str = "pytest"
    tests_run: int | None = None


@dataclasses.dataclass(frozen=True)
class _DjangoResult:
    tests_run: int
    status: str
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    expected_failures: int = 0
    unexpected_successes: int = 0
    suite_start: int = 0
    trailer_start: int = 0


@dataclasses.dataclass(frozen=True)
class _SympyResult:
    passed: int
    failed: int
    skipped: int
    expected_failed: int
    unexpected_successes: int
    exceptions: int
    trailer_start: int


@dataclasses.dataclass(frozen=True)
class _BaselineComparison:
    """Effect of a clean-base run on one candidate verification result."""

    mode: str
    effective_returncode: int
    cache_hit: bool = False
    baseline_returncode: int | None = None
    baseline_summary: tuple[tuple[str, float], ...] = ()
    new_failures: int = 0
    worsened_failures: int = 0
    preexisting_failures: int = 0
    cache_key: str = ""
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class _VisibleTestRun:
    completed: subprocess.CompletedProcess[str]
    selection: _VisibleTestSelection
    comparison: _BaselineComparison
    command: tuple[str, ...]
    candidate: _PytestOutcome


@dataclasses.dataclass(frozen=True)
class _AuthoringBindSources:
    """Validated host sources for a standalone candidate checkout."""

    worktree: str
    git_dir: str


def _is_safe_repo_path(raw_path: str) -> bool:
    """Accept bounded, printable, project-relative POSIX paths only."""

    if not raw_path or len(raw_path) > _MAX_SELECTION_PATH_CHARS:
        return False
    if "\\" in raw_path or any(not char.isprintable() for char in raw_path):
        return False
    path = PurePosixPath(raw_path)
    return (
        not path.is_absolute()
        and raw_path == path.as_posix()
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not path.parts[0].startswith("-")
    )


def _parse_git_paths(
    output: bytes | str, *, max_paths: int
) -> tuple[str, ...] | None:
    """Parse bounded ``git -z`` output without path quoting ambiguity."""

    raw = output.encode("utf-8") if isinstance(output, str) else output
    if raw and not raw.endswith(b"\0"):
        return None
    chunks = raw.split(b"\0")[:-1] if raw else []
    if len(chunks) > max_paths:
        return None
    paths: list[str] = []
    for chunk in chunks:
        try:
            path = chunk.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
        if not _is_safe_repo_path(path):
            return None
        paths.append(path)
    return tuple(paths)


def _docker_git_paths(
    metadata: dict[str, Any],
    *git_arguments: str,
    max_paths: int,
    deadline: float | None = None,
) -> tuple[str, ...] | None:
    """Run one non-shell Git path query inside the authoring container."""

    try:
        completed = subprocess.run(
            [
                "docker",
                "exec",
                "--workdir",
                "/testbed",
                str(metadata["container_id"]),
                "git",
                *git_arguments,
            ],
            capture_output=True,
            check=False,
            timeout=_deadline_timeout(
                deadline, cap=_DOCKER_CONTROL_TIMEOUT_S
            ),
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if completed.returncode != 0:
        return None
    return _parse_git_paths(completed.stdout, max_paths=max_paths)


def _is_pytest_command(command: list[str]) -> bool:
    """Recognize direct pytest and ``python -m pytest`` commands."""

    if not command:
        return False
    executable = PurePosixPath(str(command[0])).name
    if executable in _PYTEST_EXECUTABLES:
        return True
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
        return False
    return any(
        command[index] == "-m"
        and PurePosixPath(str(command[index + 1])).name in _PYTEST_EXECUTABLES
        for index in range(1, len(command) - 1)
    )


def _django_runtests_index(
    command: list[str], metadata: dict[str, Any]
) -> int | None:
    """Recognize Django's repository-native test script, not lookalikes."""

    if str(metadata.get("repo") or "") != "django/django":
        return None
    for index, raw_token in enumerate(command[:3]):
        token = PurePosixPath(str(raw_token))
        normalized = token.as_posix().removeprefix("./")
        if normalized == "tests/runtests.py":
            if index == 0:
                return index
            executable = PurePosixPath(str(command[0])).name
            if index == 1 and re.fullmatch(
                r"python(?:\d+(?:\.\d+)*)?", executable
            ):
                return index
            return None
    return None


def _sympy_runtests_index(
    command: list[str], metadata: dict[str, Any]
) -> int | None:
    """Recognize only the reviewed SymPy native command and test paths."""

    if str(metadata.get("repo") or "") != "sympy/sympy":
        return None
    if tuple(str(token) for token in command[:4]) != _SYMPY_NATIVE_COMMAND_PREFIX:
        return None
    for raw_target in command[4:]:
        target = str(raw_target)
        path = PurePosixPath(target)
        if (
            not _is_safe_repo_path(target)
            or len(path.parts) < 4
            or path.parts[0] != "sympy"
            or "tests" not in path.parts[1:-1]
            or not path.name.startswith("test_")
            or path.suffix != ".py"
        ):
            return None
    return 1


def _is_supported_test_command(
    command: list[str], metadata: dict[str, Any]
) -> bool:
    return _is_pytest_command(command) or _django_runtests_index(
        command, metadata
    ) is not None or _sympy_runtests_index(command, metadata) is not None


def _test_runner_kind(
    command: list[str], metadata: dict[str, Any]
) -> str | None:
    if _django_runtests_index(command, metadata) is not None:
        return "django-unittest"
    if _sympy_runtests_index(command, metadata) is not None:
        return "sympy-native"
    if _is_pytest_command(command):
        return "pytest"
    return None


def _is_python_source_path(path: str) -> bool:
    """Exclude tests and non-Python files from source-to-test mapping."""

    candidate = PurePosixPath(path)
    return (
        candidate.suffix.lower() == ".py"
        and not is_standard_test_path(path)
    )


def _is_visible_test_path(path: str, expected_name: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        candidate.name == expected_name
        and "tests" in {part.lower() for part in candidate.parts[:-1]}
    )


def _directory_distance(source: PurePosixPath, test: PurePosixPath) -> int:
    source_parts = source.parent.parts
    test_parts = test.parent.parts
    common = _common_directory_prefix(source, test)
    return len(source_parts) + len(test_parts) - 2 * common


def _common_directory_prefix(source: PurePosixPath, test: PurePosixPath) -> int:
    source_parts = source.parent.parts
    test_parts = test.parent.parts
    common = 0
    for source_part, test_part in zip(
        source_parts, test_parts, strict=False,
    ):
        if source_part != test_part:
            break
        common += 1
    return common


def _is_nearby_test(source: PurePosixPath, test: PurePosixPath) -> bool:
    """Reject same-stem tests in unrelated non-root package trees."""

    is_root_test_tree = bool(test.parts) and test.parts[0].lower() == "tests"
    return is_root_test_tree or _common_directory_prefix(source, test) > 0


def _map_source_to_visible_test(
    source_path: str,
    visible_paths: frozenset[str],
) -> str | None:
    """Return one unambiguous nearby ``test_<stem>.py`` target."""

    source = PurePosixPath(source_path)
    expected_name = f"test_{source.stem}.py"
    exact = (source.parent / "tests" / expected_name).as_posix()
    if exact in visible_paths:
        return exact

    candidates = [
        PurePosixPath(path)
        for path in visible_paths
        if _is_visible_test_path(path, expected_name)
        and _is_nearby_test(source, PurePosixPath(path))
    ]
    if not candidates:
        return None
    by_distance: dict[int, list[PurePosixPath]] = {}
    for candidate in candidates:
        by_distance.setdefault(
            _directory_distance(source, candidate), []
        ).append(candidate)
    nearest_distance = min(by_distance)
    nearest = by_distance[nearest_distance]
    if nearest_distance > _MAX_NEARBY_TEST_DISTANCE or len(nearest) != 1:
        return None
    return nearest[0].as_posix()


def _map_changed_sources_to_visible_tests(
    changed_paths: tuple[str, ...],
    visible_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Map every changed source or decline targeted verification entirely."""

    source_paths = tuple(
        path for path in changed_paths if _is_python_source_path(path)
    )
    if not source_paths:
        return None
    safe_visible_paths = frozenset(
        path
        for path in visible_paths
        if _is_safe_repo_path(path) and PurePosixPath(path).suffix == ".py"
    )
    targets: list[str] = []
    for source_path in source_paths:
        target = _map_source_to_visible_test(source_path, safe_visible_paths)
        if target is None:
            return None
        if target not in targets:
            targets.append(target)
    return source_paths, tuple(targets)


def _map_changed_pytest_project_sources(
    changed_paths: tuple[str, ...],
    visible_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Apply the reviewed pytest-project source-to-public-suite map."""

    source_paths = tuple(
        path for path in changed_paths if _is_python_source_path(path)
    )
    if not source_paths:
        return None
    safe_visible_paths = frozenset(
        path for path in visible_paths if _is_safe_repo_path(path)
    )
    targets: list[str] = []
    for source_path in source_paths:
        mapped_targets = _PYTEST_VISIBLE_TEST_PATH_MAP.get(source_path)
        if (
            mapped_targets is None
            or any(target not in safe_visible_paths for target in mapped_targets)
        ):
            return None
        for target in mapped_targets:
            if target not in targets:
                targets.append(target)
    return source_paths, tuple(targets)


def _map_changed_astropy_sources(
    changed_paths: tuple[str, ...],
    visible_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Apply reviewed Astropy source-to-public-suite mappings."""

    source_paths = tuple(
        path for path in changed_paths if _is_python_source_path(path)
    )
    if not source_paths:
        return None
    safe_visible_paths = frozenset(
        path for path in visible_paths if _is_safe_repo_path(path)
    )
    targets: list[str] = []
    for source_path in source_paths:
        mapped_targets = _ASTROPY_VISIBLE_TEST_PATH_MAP.get(source_path)
        if (
            mapped_targets is None
            or any(target not in safe_visible_paths for target in mapped_targets)
        ):
            return None
        for target in mapped_targets:
            if target not in targets:
                targets.append(target)
    return source_paths, tuple(targets)


def _map_changed_sympy_sources(
    changed_paths: tuple[str, ...],
    visible_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Apply reviewed SymPy source-to-native-suite mappings."""

    source_paths = tuple(
        path for path in changed_paths if _is_python_source_path(path)
    )
    if not source_paths:
        return None
    safe_visible_paths = frozenset(
        path for path in visible_paths if _is_safe_repo_path(path)
    )
    targets: list[str] = []
    for source_path in source_paths:
        mapped_targets = _SYMPY_VISIBLE_TEST_PATH_MAP.get(source_path)
        if (
            mapped_targets is None
            or any(target not in safe_visible_paths for target in mapped_targets)
        ):
            return None
        for target in mapped_targets:
            if target not in targets:
                targets.append(target)
    return source_paths, tuple(targets)


def _django_test_apps(visible_paths: tuple[str, ...]) -> frozenset[str]:
    """Derive native test labels solely from tracked test packages."""

    apps: set[str] = set()
    for raw_path in visible_paths:
        if not _is_safe_repo_path(raw_path):
            continue
        path = PurePosixPath(raw_path)
        if (
            len(path.parts) == 3
            and path.parts[0] == "tests"
            and path.parts[2] == "__init__.py"
        ):
            apps.add(path.parts[1])
    return frozenset(apps)


def _map_django_source_to_test_app(
    source_path: str,
    test_apps: frozenset[str],
) -> str | None:
    """Apply the narrow reviewed Django source-to-public-suite map."""

    source = PurePosixPath(source_path)
    if (
        len(source.parts) < 2
        or source.parts[0] != "django"
        or source.suffix.lower() != ".py"
    ):
        return None
    target = _DJANGO_VISIBLE_TEST_APP_MAP.get(source.as_posix())
    return target if target in test_apps else None


def _map_changed_django_sources_to_visible_tests(
    changed_paths: tuple[str, ...],
    visible_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Map every changed Django source to a bounded native test-app label."""

    source_paths = tuple(
        path for path in changed_paths if _is_python_source_path(path)
    )
    if (
        not source_paths
        or any(
            PurePosixPath(path).parts[:1] != ("django",)
            for path in source_paths
        )
    ):
        return None
    test_apps = _django_test_apps(visible_paths)
    targets: list[str] = []
    for source_path in source_paths:
        target = _map_django_source_to_test_app(
            source_path, test_apps
        )
        if target is None:
            return None
        if target not in targets:
            targets.append(target)
    return source_paths, tuple(targets)


def _select_visible_pytest_targets(
    parameters: dict[str, Any],
    metadata: dict[str, Any],
    command: list[str],
    *,
    deadline: float | None = None,
) -> _VisibleTestSelection:
    """Select public relevant tests, retaining full-suite fallback on doubt."""

    is_pytest = _is_pytest_command(command)
    is_django = _django_runtests_index(command, metadata) is not None
    is_sympy = _sympy_runtests_index(command, metadata) is not None
    if not is_pytest and not is_django and not is_sympy:
        return _VisibleTestSelection(
            mode="preserved_non_pytest",
            reason="configured command is not a supported visible-test runner",
        )
    if "test_args" in parameters:
        return _VisibleTestSelection(
            mode="preserved_explicit_test_args",
            reason="explicit test_args supplied by the Spec",
        )
    changed_paths = _docker_git_paths(
        metadata,
        "diff",
        "--no-ext-diff",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        "-z",
        "HEAD",
        "--",
        max_paths=_MAX_CHANGED_PATHS,
        deadline=deadline,
    )
    if changed_paths is None:
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason="changed-source Git query failed or returned unsafe paths",
            git_diff_attempted=True,
        )
    source_count = sum(_is_python_source_path(path) for path in changed_paths)
    if source_count == 0:
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason="no changed non-test Python source files",
            git_diff_attempted=True,
        )
    if source_count != len(changed_paths):
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason=(
                "at least one changed tracked path is not a mappable "
                "non-test Python source"
            ),
            changed_source_count=source_count,
            git_diff_attempted=True,
        )
    untracked_paths = _docker_git_paths(
        metadata,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        max_paths=_MAX_CHANGED_PATHS,
        deadline=deadline,
    )
    if untracked_paths is None:
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason="untracked-path Git query failed or returned unsafe paths",
            changed_source_count=source_count,
            git_diff_attempted=True,
        )
    uncovered_untracked_paths = tuple(
        path
        for path in untracked_paths
        if path not in _TRUSTED_RUNNER_CONTROL_PATHS
    )
    if uncovered_untracked_paths:
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason="at least one untracked path is outside reviewed test coverage",
            changed_source_count=source_count,
            git_diff_attempted=True,
        )
    visible_paths = _docker_git_paths(
        metadata,
        "ls-files",
        "--cached",
        "-z",
        "--",
        "*.py",
        max_paths=_MAX_VISIBLE_PATHS,
        deadline=deadline,
    )
    if visible_paths is None:
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason="visible tracked-test Git query failed or returned unsafe paths",
            changed_source_count=source_count,
            git_diff_attempted=True,
        )
    if is_django:
        mapped = _map_changed_django_sources_to_visible_tests(
            changed_paths, visible_paths
        )
    elif is_sympy:
        mapped = _map_changed_sympy_sources(changed_paths, visible_paths)
    elif (
        str(metadata.get("repo") or "") == "astropy/astropy"
        and any(path in _ASTROPY_VISIBLE_TEST_PATH_MAP for path in changed_paths)
    ):
        mapped = _map_changed_astropy_sources(changed_paths, visible_paths)
    elif (
        str(metadata.get("repo") or "") == "pytest-dev/pytest"
        and any(path in _PYTEST_VISIBLE_TEST_PATH_MAP for path in changed_paths)
    ):
        mapped = _map_changed_pytest_project_sources(
            changed_paths, visible_paths
        )
    else:
        mapped = _map_changed_sources_to_visible_tests(
            changed_paths, visible_paths
        )
    if mapped is None:
        return _VisibleTestSelection(
            mode="full_suite_fallback",
            reason="at least one changed source has no safe unambiguous test target",
            changed_source_count=source_count,
            git_diff_attempted=True,
        )
    source_paths, targets = mapped
    return _VisibleTestSelection(
        mode="targeted",
        reason="all changed sources mapped to visible tracked tests",
        changed_source_count=len(source_paths),
        targets=targets,
        git_diff_attempted=True,
    )


def _selection_values(selection: _VisibleTestSelection) -> dict[str, float]:
    return {
        "visible_test_selection_applied": float(selection.mode == "targeted"),
        "visible_test_selection_target_count": float(len(selection.targets)),
        "visible_test_selection_changed_source_count": float(
            selection.changed_source_count
        ),
        "visible_test_selection_git_diff": float(selection.git_diff_attempted),
        "visible_test_selection_full_suite_fallback": float(
            selection.mode == "full_suite_fallback"
        ),
        "visible_test_selection_explicit_args": float(
            selection.mode == "preserved_explicit_test_args"
        ),
        "visible_test_selection_non_pytest": float(
            selection.mode == "preserved_non_pytest"
        ),
    }


def _comparison_values(comparison: _BaselineComparison) -> dict[str, float]:
    values = {
        "visible_test_baseline_attempted": float(
            bool(comparison.cache_key)
            or comparison.mode
            in {"incomparable", "regressed", "preexisting_only"}
        ),
        "visible_test_baseline_cache_hit": float(comparison.cache_hit),
        "visible_test_baseline_unavailable": float(
            comparison.mode == "unavailable"
        ),
        "visible_test_baseline_preexisting_only": float(
            comparison.mode == "preexisting_only"
        ),
        "visible_test_new_failures": float(comparison.new_failures),
        "visible_test_worsened_failures": float(comparison.worsened_failures),
        "visible_test_preexisting_failures": float(
            comparison.preexisting_failures
        ),
    }
    if comparison.baseline_returncode is not None:
        values["visible_test_baseline_returncode"] = float(
            comparison.baseline_returncode
        )
    for key, value in comparison.baseline_summary:
        values[f"visible_test_baseline_{key}"] = value
    return values


def _selection_raw_output(
    combined: str,
    selection: _VisibleTestSelection,
    comparison: _BaselineComparison | None = None,
) -> str:
    provenance = (
        _SELECTION_PROVENANCE
        if selection.git_diff_attempted
        else "configured command preserved; no repository selection query"
    )
    note_lines = [
        "[crucible visible-test selection]",
        f"mode={selection.mode}",
        f"reason={selection.reason}",
        f"provenance={provenance}",
        "selected_targets=" + json.dumps(list(selection.targets)),
    ]
    if comparison is not None:
        note_lines.extend((
            "[crucible clean-base comparison]",
            f"mode={comparison.mode}",
            f"reason={comparison.reason}",
            f"cache_hit={str(comparison.cache_hit).lower()}",
            f"cache_key={comparison.cache_key[:16]}",
            f"baseline_returncode={comparison.baseline_returncode}",
            f"new_failures={comparison.new_failures}",
            f"worsened_failures={comparison.worsened_failures}",
            f"preexisting_failures={comparison.preexisting_failures}",
        ))
    note = "\n".join(note_lines)
    available = max(0, _RAW_OUTPUT_LIMIT - len(note) - 1)
    test_output = combined[-available:] if available else ""
    return test_output + "\n" + note


def _resolve_test_command(repo: str, version: str) -> list[str] | None:
    """Resolve the public project test command from SWE-bench metadata."""
    if not repo or not version:
        return None
    try:
        from swebench.harness.constants import (  # noqa: PLC0415
            MAP_REPO_VERSION_TO_SPECS,
        )
    except ImportError:
        return None
    spec = (MAP_REPO_VERSION_TO_SPECS.get(repo) or {}).get(version) or {}
    command = spec.get("test_cmd")
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, (list, tuple)):
        return [str(token) for token in command]
    return None


def _parse_pytest_summary(text: str) -> dict[str, float]:
    """Parse pytest's true terminal summary into bounded exact counts."""

    lines = _ANSI_ESCAPE_RE.sub("", text).splitlines()
    if not lines:
        return {}
    terminal = lines[-1].strip().strip("=").strip()
    match = _PYTEST_TERMINAL_RE.fullmatch(terminal)
    if match is None:
        return {}
    fields = match.group("counts").split(", ")
    parsed: dict[str, int] = {}
    key_by_kind = {
        "passed": "tests_passed",
        "failed": "tests_failed",
        "error": "tests_errored",
        "errors": "tests_errored",
        "skipped": "tests_skipped",
        "xfailed": "tests_expected_failed",
        "xpassed": "tests_unexpected_successes",
        "deselected": "tests_deselected",
        "warning": "tests_warnings",
        "warnings": "tests_warnings",
        "rerun": "tests_rerun",
        "reruns": "tests_rerun",
    }
    for field in fields:
        count_match = _PYTEST_COUNT_RE.fullmatch(field)
        if count_match is None:
            return {}
        key = key_by_kind[count_match.group("kind")]
        if key in parsed:
            return {}
        parsed[key] = int(count_match.group("count"))
    executed = sum(
        parsed.get(key, 0)
        for key in (
            "tests_passed",
            "tests_failed",
            "tests_errored",
            "tests_skipped",
            "tests_expected_failed",
            "tests_unexpected_successes",
        )
    )
    if executed <= 0:
        return {}
    return {key: float(value) for key, value in parsed.items()}


def _django_result_counts(raw_counts: str) -> dict[str, int] | None:
    if not raw_counts:
        return {}
    fields = raw_counts.split(", ")
    parsed: dict[str, int] = {}
    for field in fields:
        match = _DJANGO_RESULT_COUNT_RE.fullmatch(field)
        if match is None or match.group(1) in parsed:
            return None
        parsed[match.group(1)] = int(match.group(2))
    return parsed


def _parse_django_result(text: str) -> _DjangoResult | None:
    """Pair the final ``Ran`` line with its adjacent unittest status."""

    matches = list(_DJANGO_TRAILER_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    counts = _django_result_counts(match.group("counts") or "")
    if counts is None:
        return None
    status = match.group("status")
    failures = counts.get("failures", 0)
    errors = counts.get("errors", 0)
    skipped = counts.get("skipped", 0)
    expected_failures = counts.get("expected failures", 0)
    unexpected_successes = counts.get("unexpected successes", 0)
    try:
        tests_run = int(match.group("ran"))
    except (TypeError, ValueError):
        return None
    if (
        status == "OK" and (failures or errors or unexpected_successes)
        or status == "FAILED"
        and not (failures or errors or unexpected_successes)
    ):
        return None
    prior_matches = list(
        _DJANGO_PRIOR_TRAILER_RE.finditer(text, 0, match.start())
    )
    return _DjangoResult(
        tests_run=tests_run,
        status=status,
        failures=failures,
        errors=errors,
        skipped=skipped,
        expected_failures=expected_failures,
        unexpected_successes=unexpected_successes,
        suite_start=prior_matches[-1].end() if prior_matches else 0,
        trailer_start=match.start(),
    )


def _parse_django_summary(text: str) -> dict[str, float]:
    """Extract bounded unittest counts from Django's native runner output."""

    result = _parse_django_result(text)
    if result is None:
        return {}
    passed = max(
        0,
        result.tests_run
        - result.failures
        - result.errors
        - result.skipped
        - result.expected_failures
        - result.unexpected_successes,
    )
    return {
        "tests_passed": float(passed),
        "tests_failed": float(
            result.failures + result.unexpected_successes
        ),
        "tests_errored": float(result.errors),
        "tests_skipped": float(result.skipped),
        "tests_expected_failed": float(result.expected_failures),
        "tests_unexpected_successes": float(result.unexpected_successes),
    }


def _normalize_failure_reason(reason: str) -> str:
    """Remove common nondeterministic fragments from a short failure reason."""

    normalized = _VOLATILE_FAILURE_TEXT_RE.sub("<volatile>", reason)
    return " ".join(normalized.split())


def _outer_pytest_short_summary(text: str) -> str | None:
    """Return the final outer pytest short-summary body, if authoritative.

    Repository tests often invoke pytest recursively and include an inner
    pytest failure report in captured output. Only the final short-summary
    block immediately paired with the stream's terminal result belongs to the
    authoring run itself. Reject intervening or unrecognized text so a nested
    report cannot be mistaken for that outer result.
    """

    normalized = _ANSI_ESCAPE_RE.sub("", text)
    lines = normalized.splitlines()
    if len(lines) < 2 or not _parse_pytest_summary(normalized):
        return None
    terminal_index = len(lines) - 1
    header_indexes = [
        index
        for index, line in enumerate(lines[:terminal_index])
        if _PYTEST_SHORT_SUMMARY_HEADER_RE.fullmatch(line.strip())
    ]
    if not header_indexes:
        return None
    body_lines = lines[header_indexes[-1] + 1 : terminal_index]
    records: list[str] = []
    saw_postamble = False
    for raw_line in body_lines:
        line = raw_line.strip()
        if not line:
            continue
        if _PYTEST_SHORT_SUMMARY_RECORD_RE.match(line) is not None:
            if saw_postamble:
                return None
            records.append(line)
            continue
        if (
            records
            and not saw_postamble
            and _PYTEST_SHORT_SUMMARY_POSTAMBLE_RE.fullmatch(line) is not None
        ):
            saw_postamble = True
            continue
        return None
    if not records:
        return None
    return "\n".join(records)


def _parse_pytest_failures(
    text: str,
) -> tuple[tuple[str, str, str], ...] | None:
    """Extract failures only from pytest's authoritative outer summary."""

    short_summary = _outer_pytest_short_summary(text)
    if short_summary is None:
        return None

    failures: dict[str, tuple[str, str]] = {}
    for raw_line in short_summary.splitlines():
        match = _PYTEST_FAILURE_RE.match(raw_line.strip())
        if match is None:
            continue
        status, remainder = match.groups()
        node_id, separator, reason = remainder.partition(" - ")
        node_id = node_id.strip()
        normalized_reason = _normalize_failure_reason(reason) if separator else ""
        if (
            not node_id
            or "::" not in node_id and not node_id.endswith(".py")
            or len(node_id) > _MAX_FAILURE_NODE_CHARS
            or len(normalized_reason) > _MAX_FAILURE_REASON_CHARS
            or any(not character.isprintable() for character in node_id)
        ):
            continue
        if node_id not in failures and len(failures) >= _MAX_FAILURE_RECORDS:
            return None
        failures[node_id] = (status, normalized_reason)
    return tuple(
        (node_id, status, reason)
        for node_id, (status, reason) in sorted(failures.items())
    )


def _django_failure_fingerprint(body: str) -> str:
    """Bind forgiveness to the exact stable body of a unittest failure."""

    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_django_failures(
    text: str,
    *,
    suite_start: int = 0,
    trailer_start: int | None = None,
) -> tuple[tuple[str, str, str], ...] | None:
    """Extract exact identity and body fingerprints from unittest sections."""

    upper_bound = len(text) if trailer_start is None else trailer_start
    matches = [
        match
        for match in _DJANGO_FAILURE_HEADER_RE.finditer(text)
        if suite_start <= match.start() < upper_bound
    ]
    if len(matches) > _MAX_FAILURE_RECORDS:
        return None
    failures: dict[str, tuple[str, str]] = {}
    for index, match in enumerate(matches):
        raw_status, raw_node_id = match.groups()
        node_id = raw_node_id.strip()
        if (
            not node_id
            or len(node_id) > _MAX_FAILURE_NODE_CHARS
            or any(not character.isprintable() for character in node_id)
            or node_id in failures
            or match.end() > upper_bound
        ):
            return None
        body_end = min(
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text),
            trailer_start
            if trailer_start is not None and trailer_start > match.end()
            else len(text),
        )
        body = text[match.end() : body_end]
        if not body.strip():
            return None
        reason = _django_failure_fingerprint(body)
        failures[node_id] = (
            "ERROR" if raw_status == "ERROR" else "FAILED",
            reason,
        )
    return tuple(
        (node_id, status, reason)
        for node_id, (status, reason) in sorted(failures.items())
    )


def _parse_sympy_result(text: str) -> _SympyResult | None:
    """Parse SymPy's final native-runner trailer, anchored to stdout EOF."""

    normalized = _ANSI_ESCAPE_RE.sub("", text)
    matches = list(_SYMPY_NATIVE_TRAILER_RE.finditer(normalized))
    if len(matches) != 1:
        return None
    match = matches[0]

    def count(name: str) -> int:
        raw = match.group(name)
        return int(raw) if raw is not None else 0

    result = _SympyResult(
        passed=count("passed"),
        failed=count("failed_1") + count("failed_2"),
        skipped=count("skipped"),
        expected_failed=count("expected_failed"),
        unexpected_successes=count("unexpected_successes"),
        exceptions=count("exceptions"),
        trailer_start=match.start(),
    )
    has_failures = result.failed + result.exceptions > 0
    if bool(match.group("do_not_commit")) != has_failures:
        return None
    return result


def _sympy_failure_fingerprint(body: str) -> str:
    """Bind forgiveness to one exact deterministic SymPy failure body."""

    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_sympy_failures(
    text: str,
    *,
    result: _SympyResult,
) -> tuple[tuple[str, str, str], ...] | None:
    """Extract exact native failure identities and body fingerprints."""

    normalized = _ANSI_ESCAPE_RE.sub("", text)
    matches = [
        match
        for match in _SYMPY_NATIVE_FAILURE_HEADER_RE.finditer(normalized)
        if match.start() < result.trailer_start
    ]
    failure_total = result.failed + result.exceptions
    if len(matches) != failure_total or len(matches) > _MAX_FAILURE_RECORDS:
        return None
    failures: dict[str, tuple[str, str]] = {}
    for index, match in enumerate(matches):
        node_id = match.group("node")
        if (
            not _is_safe_repo_path(node_id.partition(":")[0])
            or len(node_id) > _MAX_FAILURE_NODE_CHARS
            or node_id in failures
        ):
            return None
        body_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else result.trailer_start
        )
        body = normalized[match.end() : body_end]
        if not body.strip():
            return None
        status = "ERROR" if index < result.exceptions else "FAILED"
        failures[node_id] = (status, _sympy_failure_fingerprint(body))
    return tuple(
        (node_id, status, reason)
        for node_id, (status, reason) in sorted(failures.items())
    )


def _pytest_outcome(
    completed: subprocess.CompletedProcess[str],
) -> _PytestOutcome:
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    combined = stdout_text + stderr_text
    parsed_streams = [
        (stream, summary)
        for stream in (stdout_text, stderr_text)
        if (summary := _parse_pytest_summary(stream))
    ]
    authoritative_stream, summary = (
        parsed_streams[0] if len(parsed_streams) == 1 else ("", {})
    )
    tests_run = int(sum(
        summary.get(key, 0.0)
        for key in (
            "tests_passed",
            "tests_failed",
            "tests_errored",
            "tests_skipped",
            "tests_expected_failed",
            "tests_unexpected_successes",
        )
    )) if summary else None
    failed = int(
        summary.get("tests_failed", 0.0)
        + summary.get("tests_errored", 0.0)
    )
    parsed_failures = (
        _parse_pytest_failures(authoritative_stream) if failed else ()
    )
    structurally_valid = bool(
        summary
        and tests_run is not None
        and tests_run > 0
        and parsed_failures is not None
        and (
            completed.returncode == 0 and failed == 0
            or completed.returncode == 1 and failed > 0
        )
    )
    return _PytestOutcome(
        returncode=int(completed.returncode) if structurally_valid else 2,
        failures=parsed_failures if structurally_valid else (),
        summary=tuple(sorted(summary.items())) if structurally_valid else (),
        output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        tests_run=tests_run if structurally_valid else None,
    )


def _django_outcome(
    completed: subprocess.CompletedProcess[str],
) -> _PytestOutcome:
    combined = (completed.stdout or "") + (completed.stderr or "")
    result = _parse_django_result(combined)
    failures = _parse_django_failures(
        combined,
        suite_start=result.suite_start if result is not None else 0,
        trailer_start=result.trailer_start if result is not None else None,
    )
    parsed_failures = failures if failures is not None else ()
    status_counts = {
        "FAILED": sum(status == "FAILED" for _, status, _ in parsed_failures),
        "ERROR": sum(status == "ERROR" for _, status, _ in parsed_failures),
    }
    structurally_valid = bool(
        result is not None
        and failures is not None
        and result.tests_run > 0
        and status_counts["FAILED"] == result.failures
        and status_counts["ERROR"] == result.errors
        and (
            completed.returncode == 0 and result.status == "OK"
            or completed.returncode == 1 and result.status == "FAILED"
        )
    )
    effective_returncode = int(completed.returncode) if structurally_valid else 2
    return _PytestOutcome(
        returncode=effective_returncode,
        failures=parsed_failures if structurally_valid else (),
        summary=(
            tuple(sorted(_parse_django_summary(combined).items()))
            if structurally_valid
            else ()
        ),
        output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        parser_kind="django-unittest",
        tests_run=(
            result.tests_run if structurally_valid and result is not None else None
        ),
    )


def _sympy_outcome(
    completed: subprocess.CompletedProcess[str],
) -> _PytestOutcome:
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    combined = stdout_text + stderr_text
    result = _parse_sympy_result(stdout_text)
    failure_total = (
        result.failed + result.exceptions if result is not None else 0
    )
    failures = (
        _parse_sympy_failures(stdout_text, result=result)
        if result is not None and failure_total
        else ()
    )
    tests_run = (
        result.passed
        + result.failed
        + result.skipped
        + result.expected_failed
        + result.unexpected_successes
        + result.exceptions
        if result is not None
        else None
    )
    structurally_valid = bool(
        result is not None
        and tests_run is not None
        and tests_run > 0
        and failures is not None
        and len(failures) == failure_total
        and (
            completed.returncode == 0 and failure_total == 0
            or completed.returncode == 1 and failure_total > 0
        )
    )
    summary = {
        "tests_passed": float(result.passed),
        "tests_failed": float(result.failed),
        "tests_errored": float(result.exceptions),
        "tests_skipped": float(result.skipped),
        "tests_expected_failed": float(result.expected_failed),
        "tests_unexpected_successes": float(result.unexpected_successes),
    } if result is not None else {}
    return _PytestOutcome(
        returncode=int(completed.returncode) if structurally_valid else 2,
        failures=failures if structurally_valid else (),
        summary=tuple(sorted(summary.items())) if structurally_valid else (),
        output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        parser_kind="sympy-native",
        tests_run=tests_run if structurally_valid else None,
    )


def _visible_test_outcome(
    completed: subprocess.CompletedProcess[str],
    command: list[str],
    metadata: dict[str, Any],
) -> _PytestOutcome:
    if _django_runtests_index(command, metadata) is not None:
        return _django_outcome(completed)
    if _sympy_runtests_index(command, metadata) is not None:
        return _sympy_outcome(completed)
    if _is_pytest_command(command):
        return _pytest_outcome(completed)
    combined = (completed.stdout or "") + (completed.stderr or "")
    return _PytestOutcome(
        returncode=int(completed.returncode),
        failures=(),
        summary=(),
        output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        parser_kind="opaque",
    )


def _parse_visible_test_summary(
    text: str, command: list[str], metadata: dict[str, Any]
) -> dict[str, float]:
    if _django_runtests_index(command, metadata) is not None:
        return _parse_django_summary(text)
    if _sympy_runtests_index(command, metadata) is not None:
        result = _parse_sympy_result(text)
        if result is None:
            return {}
        return {
            "tests_passed": float(result.passed),
            "tests_failed": float(result.failed),
            "tests_errored": float(result.exceptions),
            "tests_skipped": float(result.skipped),
            "tests_expected_failed": float(result.expected_failed),
            "tests_unexpected_successes": float(result.unexpected_successes),
        }
    return _parse_pytest_summary(text)


def _outcome_failure_total(outcome: _PytestOutcome) -> int | None:
    summary = dict(outcome.summary)
    if not summary:
        return None
    return int(summary.get("tests_failed", 0.0) + summary.get("tests_errored", 0.0))


def _compare_pytest_outcomes(
    candidate: _PytestOutcome,
    baseline: _PytestOutcome,
    *,
    cache_hit: bool,
    cache_key: str,
) -> _BaselineComparison:
    """Forgive only demonstrably unchanged or improved base failures."""

    common = {
        "cache_hit": cache_hit,
        "baseline_returncode": baseline.returncode,
        "baseline_summary": baseline.summary,
        "cache_key": cache_key,
    }
    candidate_total = _outcome_failure_total(candidate)
    baseline_total = _outcome_failure_total(baseline)
    blocking_returncode = max(1, candidate.returncode)
    # Exit code 1 is the ordinary test-failure result for both supported
    # runners. Any usage, interruption, or internal error remains blocking.
    if candidate.returncode not in {0, 1} or baseline.returncode not in {0, 1}:
        return _BaselineComparison(
            mode="incomparable",
            effective_returncode=blocking_returncode,
            reason="candidate result is not comparable with a valid clean base",
            **common,
        )
    if (
        candidate.parser_kind != baseline.parser_kind
        or candidate.tests_run != baseline.tests_run
    ):
        return _BaselineComparison(
            mode="incomparable",
            effective_returncode=blocking_returncode,
            reason="test parser kind or executed-test count changed",
            **common,
        )
    candidate_summary = dict(candidate.summary)
    baseline_summary = dict(baseline.summary)
    regressed_nonpasses = sum(
        max(
            0,
            int(candidate_summary.get(key, 0.0))
            - int(baseline_summary.get(key, 0.0)),
        )
        for key in (
            "tests_skipped",
            "tests_expected_failed",
            "tests_unexpected_successes",
            "tests_deselected",
            "tests_rerun",
        )
    )
    if regressed_nonpasses:
        return _BaselineComparison(
            mode="regressed",
            effective_returncode=blocking_returncode,
            baseline_returncode=baseline.returncode,
            baseline_summary=baseline.summary,
            cache_key=cache_key,
            cache_hit=cache_hit,
            worsened_failures=regressed_nonpasses,
            reason="candidate increased skipped or non-passing test outcomes",
        )
    if candidate.returncode == 0:
        if candidate_total != 0 or candidate.failures:
            return _BaselineComparison(
                mode="incomparable",
                effective_returncode=blocking_returncode,
                reason="passing candidate outcome was structurally incomplete",
                **common,
            )
        if baseline.returncode == 0:
            if baseline_total != 0 or baseline.failures:
                return _BaselineComparison(
                    mode="incomparable",
                    effective_returncode=blocking_returncode,
                    reason="passing clean-base outcome was structurally incomplete",
                    **common,
                )
        elif (
            baseline_total is None
            or baseline_total != len(baseline.failures)
            or baseline_total == 0
        ):
            return _BaselineComparison(
                mode="incomparable",
                effective_returncode=blocking_returncode,
                reason="clean-base failure identities were incomplete",
                **common,
            )
        return _BaselineComparison(
            mode="baseline_match",
            effective_returncode=0,
            preexisting_failures=int(baseline_total or 0),
            reason="candidate passed the same number of tests as the clean base",
            **common,
        )
    if (
        candidate_total is None
        or candidate_total != len(candidate.failures)
        or candidate_total == 0
    ):
        return _BaselineComparison(
            mode="incomparable",
            effective_returncode=blocking_returncode,
            reason="visible-test failure identities were incomplete",
            **common,
        )
    if baseline.returncode == 0:
        return _BaselineComparison(
            mode="regressed",
            effective_returncode=blocking_returncode,
            new_failures=candidate_total,
            reason="candidate failed tests that pass on the clean base",
            **common,
        )
    if (
        baseline_total is None
        or baseline_total != len(baseline.failures)
        or baseline_total == 0
    ):
        return _BaselineComparison(
            mode="incomparable",
            effective_returncode=blocking_returncode,
            reason="clean-base failure identities were incomplete",
            **common,
        )

    baseline_failures = {
        node_id: (status, reason)
        for node_id, status, reason in baseline.failures
    }
    new_failures = 0
    worsened_failures = 0
    preexisting_failures = 0
    for node_id, status, reason in candidate.failures:
        prior = baseline_failures.get(node_id)
        if prior is None:
            new_failures += 1
            continue
        prior_status, prior_reason = prior
        if status != prior_status or reason != prior_reason:
            # Only the exact same status and body is a known base failure.
            worsened_failures += 1
        else:
            preexisting_failures += 1

    if (
        new_failures
        or worsened_failures
        or candidate_total > baseline_total
    ):
        return _BaselineComparison(
            mode="regressed",
            effective_returncode=blocking_returncode,
            new_failures=new_failures,
            worsened_failures=worsened_failures,
            preexisting_failures=preexisting_failures,
            reason="candidate introduced or worsened visible test failures",
            **common,
        )
    return _BaselineComparison(
        mode="preexisting_only",
        effective_returncode=0,
        preexisting_failures=preexisting_failures,
        reason="all candidate failures are unchanged or improved from clean base",
        **common,
    )


def _serialize_outcome(outcome: _PytestOutcome) -> dict[str, Any]:
    return {
        "returncode": outcome.returncode,
        "failures": [list(failure) for failure in outcome.failures],
        "summary": dict(outcome.summary),
        "output_sha256": outcome.output_sha256,
        "parser_kind": outcome.parser_kind,
        "tests_run": outcome.tests_run,
    }


def _deserialize_outcome(raw: object) -> _PytestOutcome | None:
    if not isinstance(raw, dict):
        return None
    try:
        returncode = int(raw["returncode"])
        failures = tuple(
            (str(item[0]), str(item[1]), str(item[2]))
            for item in raw["failures"]
        )
        summary = tuple(
            sorted((str(key), float(value)) for key, value in raw["summary"].items())
        )
        output_sha256 = str(raw["output_sha256"])
        parser_kind = str(raw["parser_kind"])
        raw_tests_run = raw["tests_run"]
        tests_run = None if raw_tests_run is None else int(raw_tests_run)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None
    if (
        returncode not in {0, 1}
        or parser_kind not in {"pytest", "django-unittest", "sympy-native"}
        or tests_run is not None and tests_run < 0
        or tests_run is None
        or tests_run <= 0
        or len(failures) > _MAX_FAILURE_RECORDS
        or len({node_id for node_id, _, _ in failures}) != len(failures)
        or any(status not in {"FAILED", "ERROR"} for _, status, _ in failures)
        or any(
            not node_id
            or len(node_id) > _MAX_FAILURE_NODE_CHARS
            or len(reason) > _MAX_FAILURE_REASON_CHARS
            or any(not character.isprintable() for character in node_id)
            for node_id, _, reason in failures
        )
        or any(
            key not in {
                "tests_passed",
                "tests_failed",
                "tests_errored",
                "tests_skipped",
                "tests_expected_failed",
                "tests_unexpected_successes",
                "tests_deselected",
                "tests_warnings",
                "tests_rerun",
            }
            or value < 0
            or not value.is_integer()
            for key, value in summary
        )
        or not re.fullmatch(r"[0-9a-f]{64}", output_sha256)
    ):
        return None
    return _PytestOutcome(
        returncode,
        failures,
        summary,
        output_sha256,
        parser_kind,
        tests_run,
    )


def _baseline_cache_root() -> Path:
    configured = os.environ.get(_BASELINE_CACHE_ENV)
    if configured:
        return Path(configured)
    # Keep baseline evidence on the mounted data disk without assuming the
    # shared ``/data/crucible-swebench`` deployment directory is writable by
    # the workflow UID.  A direct UID-scoped child of /data also avoids an
    # attacker-controlled intermediate path; ``_secure_cache_root`` verifies
    # that the final directory is private, non-symlink, and owned by this UID.
    return Path(
        f"/data/.crucible-swebench-visible-baselines-{os.geteuid()}"
    )


def _secure_cache_root(root: Path) -> bool:
    """Create a private cache root, rejecting unsafe pre-existing roots."""

    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = root.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.geteuid()
        and info.st_mode & 0o077 == 0
    )


def _runner_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _baseline_identity(
    metadata: dict[str, Any],
    base_commit: str,
    command: list[str],
    selection: _VisibleTestSelection,
) -> dict[str, Any] | None:
    image_digest = str(metadata.get("image_digest") or "")
    instance_id = str(metadata.get("instance_id") or "")
    parser_kind = _test_runner_kind(command, metadata)
    if (
        _IMMUTABLE_IMAGE_RE.fullmatch(image_digest) is None
        or _GIT_COMMIT_RE.fullmatch(base_commit) is None
        or not instance_id
        or len(instance_id) > _MAX_FAILURE_NODE_CHARS
        or any(not character.isprintable() for character in instance_id)
        or parser_kind is None
    ):
        return None
    return {
        "schema": _BASELINE_CACHE_SCHEMA,
        "outcome_parser": {
            "schema": _OUTCOME_PARSER_SCHEMA,
            "kind": parser_kind,
        },
        "instance_id": instance_id,
        "image_digest": image_digest,
        "base_commit": base_commit,
        "repo": str(metadata.get("repo") or ""),
        "version": str(metadata.get("version") or ""),
        "command": command,
        "environment": {
            "conda_activate_prefix": CONDA_ACTIVATE_PREFIX,
            "workdir": _BASELINE_TESTBED,
            "pythonpath": [
                _BASELINE_TESTBED,
                _BASELINE_TESTBED + "/src",
            ],
            "uid": os.getuid(),
            "gid": os.getgid(),
            "network": "none",
            "read_only_root": True,
            "source_recipe": (
                "git-object-base-only"
                if parser_kind == "django-unittest"
                else "immutable-image-provisioned-plus-git-object-base"
            ),
            "sympy_determinism": (
                {
                    "pythonhashseed": _SYMPY_NATIVE_HASH_SEED,
                    "test_seed": _SYMPY_NATIVE_TEST_SEED,
                }
                if parser_kind == "sympy-native"
                else None
            ),
        },
        "selected_targets": list(selection.targets),
        "runner_sha256": _runner_digest(),
    }


def _identity_key(identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cached_outcome(
    path: Path, identity: dict[str, Any]
) -> _PytestOutcome | None:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("identity") != identity:
        return None
    return _deserialize_outcome(raw.get("outcome"))


def _write_cached_outcome(
    root: Path,
    path: Path,
    identity: dict[str, Any],
    outcome: _PytestOutcome,
) -> None:
    payload = json.dumps(
        {"identity": identity, "outcome": _serialize_outcome(outcome)},
        sort_keys=True,
        separators=(",", ":"),
    )
    temporary: str | None = None
    try:
        file_descriptor, temporary = tempfile.mkstemp(
            dir=root, prefix=f".{path.name}.", suffix=".tmp"
        )
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as cache_file:
            cache_file.write(payload)
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _cached_baseline_outcome(
    identity: dict[str, Any],
    run_baseline: Callable[[], _PytestOutcome | None],
    *,
    deadline: float | None = None,
) -> tuple[_PytestOutcome | None, bool, str, str]:
    """Load or atomically populate a baseline under an interprocess lock."""

    key = _identity_key(identity)
    root = _baseline_cache_root()
    if not _secure_cache_root(root):
        return None, False, key, "baseline cache root is unavailable or unsafe"
    cache_path = root / f"{key}.json"
    lock_path = root / f"{key}.lock"
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError:
        return None, False, key, "baseline cache lock could not be opened"
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if deadline is not None and deadline <= time.monotonic():
                os.close(lock_fd)
                return (
                    None,
                    False,
                    key,
                    "baseline cache lock deadline expired",
                )
            time.sleep(
                0.05
                if deadline is None
                else min(0.05, max(0.0, deadline - time.monotonic()))
            )
        except OSError:
            os.close(lock_fd)
            return None, False, key, "baseline cache lock could not be acquired"
    try:
        cached = _read_cached_outcome(cache_path, identity)
        if cached is not None:
            return cached, True, key, ""
        if deadline is not None and deadline <= time.monotonic():
            return None, False, key, "baseline runner deadline expired"
        outcome = run_baseline()
        if outcome is None:
            return None, False, key, "clean-base test run could not be reproduced"
        _write_cached_outcome(root, cache_path, identity, outcome)
        return outcome, False, key, ""
    except OSError:
        return None, False, key, "baseline cache operation failed"
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)


def _validated_metadata(
    parameters: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    metadata = parameters.get("swebench_metadata")
    if not isinstance(metadata, dict):
        return None, "SWE-bench authoring metadata is required"
    leaked = sorted(_HIDDEN_FIELDS.intersection(metadata))
    if leaked:
        return None, (
            "authoring runner rejected hidden evaluation fields: "
            + ", ".join(leaked)
        )
    if not metadata.get("container_id"):
        return None, (
            "SWE-bench authoring requires an isolated container; "
            "host execution is forbidden"
        )
    return metadata, None


def _emit_error(stdout: TextIO, message: str, *, hidden: bool = False) -> int:
    stdout.write(json.dumps({
        "values": {
            "returncode": 2.0,
            "python_returncode": 2.0,
            (
                "hidden_evaluation_data_rejected"
                if hidden else "container_required"
            ): 1.0,
        },
        "raw_output": message,
    }))
    stdout.flush()
    return 2


def _shell_test_command(
    command: list[str],
    *,
    testbed: str = "/testbed",
    metadata: dict[str, Any] | None = None,
) -> str:
    activate_prefix = CONDA_ACTIVATE_PREFIX
    if testbed != "/testbed":
        activate_prefix = activate_prefix.replace(
            "cd /testbed ", f"cd {shlex.quote(testbed)} ", 1
        )
    rendered_command = [str(token) for token in command]
    if (
        metadata is not None
        and _sympy_runtests_index(rendered_command, metadata) is not None
    ):
        rendered_command = [
            rendered_command[0],
            _SYMPY_NATIVE_HASH_SEED,
            *rendered_command[1:4],
            _SYMPY_NATIVE_TEST_SEED,
            *rendered_command[4:],
        ]
    if rendered_command and _ENV_ASSIGNMENT_RE.fullmatch(rendered_command[0]):
        rendered_command.insert(0, "env")
    return activate_prefix + "&& " + " ".join(
        shlex.quote(token) for token in rendered_command
    )


def _run_managed_baseline_docker(
    arguments: list[str],
    *,
    step_name: str,
    timeout: float,
    deadline: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one Docker lifecycle command for managed baseline cleanup."""

    try:
        completed = subprocess.run(
            ["docker", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=_deadline_timeout(deadline, cap=timeout),
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        raise OSError(f"baseline Docker {step_name} failed") from exc
    if completed.returncode != 0:
        raise OSError(
            f"baseline Docker {step_name} returned {completed.returncode}: "
            f"{(completed.stderr or '')[-500:]}"
        )
    return completed


def _process_start_token(pid: int) -> str | None:
    """Return Linux's PID-reuse-resistant process start token."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        command_end = raw.rfind(")")
        if command_end < 0:
            return None
        fields_after_command = raw[command_end + 1 :].split()
        token = fields_after_command[19]
        int(token)
        return token
    except (OSError, IndexError, ValueError):
        return None


def _validated_bind_source(raw_source: object) -> str | None:
    """Return one owned real directory safe for Docker ``--mount``."""

    if (
        not isinstance(raw_source, str)
        or not raw_source
        or len(raw_source) > _MAX_BIND_SOURCE_CHARS
        or "," in raw_source
        or any(not character.isprintable() for character in raw_source)
    ):
        return None
    source = PurePosixPath(raw_source)
    if (
        not source.is_absolute()
        or source == PurePosixPath("/")
        or source.as_posix() != raw_source
    ):
        return None
    host_source = Path(raw_source)
    try:
        source_info = host_source.lstat()
        resolved = host_source.resolve(strict=True)
    except OSError:
        return None
    if (
        not stat.S_ISDIR(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or source_info.st_uid != os.geteuid()
        or resolved.as_posix() != raw_source
    ):
        return None
    return raw_source


def _validated_authoring_bind_sources(
    metadata: dict[str, Any],
    *,
    deadline: float | None = None,
) -> _AuthoringBindSources | None:
    """Validate exact candidate-container bind sources for clean-base replay.

    The immutable image supplies dependencies, but its baked ``/testbed`` may
    be at a different revision than the dataset instance. The active authoring
    or patch-only visible-verifier container has the exact standalone
    checkout/object store. Accept its host sources only when it is
    Crucible-managed, running, image/task-bound, and exposes exactly the
    expected workspace plus nested read-only Git bind.

    ``authoring`` retains its original compatibility contract. A
    ``visible-verifier`` container is accepted only when its managed label is
    bound to the exact lowercase SHA-256 supplied in
    ``metadata.candidate_patch_sha256``. No other phase is a source of trusted
    clean-base material.
    """

    container_id = str(metadata.get("container_id") or "")
    if _CONTAINER_REFERENCE_RE.fullmatch(container_id) is None:
        return None
    try:
        inspected = _run_managed_baseline_docker(
            ["inspect", "--format", "{{json .}}", container_id],
            step_name="inspect candidate container",
            timeout=_DOCKER_CONTROL_TIMEOUT_S,
            deadline=deadline,
        )
        payload = json.loads((inspected.stdout or "{}").strip())
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("State")
    config = payload.get("Config")
    mounts = payload.get("Mounts")
    if (
        not isinstance(state, dict)
        or state.get("Running") is not True
        or not isinstance(config, dict)
        or not isinstance(mounts, list)
    ):
        return None
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        return None
    expected_image = str(metadata.get("image_digest") or "")
    instance_id = str(metadata.get("instance_id") or "")
    if not instance_id or _IMMUTABLE_IMAGE_RE.fullmatch(expected_image) is None:
        return None
    expected_task = canonical_fingerprint(instance_id).split(":", 1)[1]
    try:
        owner_pid = int(labels.get("io.crucible.owner-pid", "0"))
    except (TypeError, ValueError):
        return None
    owner_start = labels.get("io.crucible.owner-start")
    phase = labels.get("io.crucible.phase")
    if phase == "authoring":
        phase_bound = True
    elif phase == "visible-verifier":
        candidate_patch_sha256 = metadata.get("candidate_patch_sha256")
        phase_bound = bool(
            isinstance(candidate_patch_sha256, str)
            and _CANDIDATE_PATCH_SHA256_RE.fullmatch(candidate_patch_sha256)
            and labels.get("io.crucible.candidate-patch-sha256")
            == candidate_patch_sha256
        )
    else:
        phase_bound = False
    if (
        labels.get("io.crucible.managed") != "true"
        or not phase_bound
        or labels.get("io.crucible.image-digest") != expected_image
        or labels.get("io.crucible.task") != expected_task
        or labels.get("io.crucible.owner-host") != socket.gethostname()
        or owner_pid <= 0
        or not isinstance(owner_start, str)
        or _process_start_token(owner_pid) != owner_start
        or config.get("Image") != expected_image
    ):
        return None

    by_destination: dict[str, dict[str, Any]] = {}
    for raw_mount in mounts:
        if not isinstance(raw_mount, dict):
            return None
        destination = raw_mount.get("Destination")
        if not isinstance(destination, str) or destination in by_destination:
            return None
        by_destination[destination] = raw_mount
    if set(by_destination) != {"/testbed", "/testbed/.git"}:
        return None
    workspace_mount = by_destination["/testbed"]
    git_mount = by_destination["/testbed/.git"]
    if (
        workspace_mount.get("Type") != "bind"
        or workspace_mount.get("RW") is not True
        or workspace_mount.get("Propagation") != "rprivate"
        or git_mount.get("Type") != "bind"
        or git_mount.get("RW") is not False
        or git_mount.get("Propagation") != "rprivate"
    ):
        return None
    worktree_source = _validated_bind_source(workspace_mount.get("Source"))
    git_source = _validated_bind_source(git_mount.get("Source"))
    if (
        worktree_source is None
        or git_source is None
        or PurePosixPath(git_source)
        != PurePosixPath(worktree_source) / ".git"
    ):
        return None
    return _AuthoringBindSources(worktree_source, git_source)


def _runner_deadline(parameters: dict[str, Any]) -> float:
    raw_timeout = parameters.get("runner_timeout_s", _BASELINE_TIMEOUT_S)
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not math.isfinite(float(raw_timeout))
        or float(raw_timeout) <= 0.0
    ):
        raise ValueError("SWE-bench runner timeout must be a positive number")
    return time.monotonic() + float(raw_timeout)


def _deadline_timeout(
    deadline: float | None,
    *,
    cap: float,
) -> float:
    """Return one subprocess timeout bounded by an absolute runner deadline."""

    if deadline is None:
        return cap
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise ValueError("SWE-bench runner deadline expired")
    return min(cap, remaining)


def _best_effort_docker_cleanup(
    arguments: list[str],
    *,
    deadline: float,
) -> None:
    """Spend only the remaining shared cleanup window on one Docker action."""

    try:
        timeout = _deadline_timeout(
            deadline,
            cap=_DOCKER_CONTROL_TIMEOUT_S,
        )
        subprocess.run(
            ["docker", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass


def _contained_candidate_command(
    metadata: dict[str, Any],
    shell_command: str,
    *,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    """Run candidate code below the attested container subreaper."""

    remaining = deadline - time.monotonic()
    cleanup_timeout = min(5.0, max(0.5, remaining * 0.10))
    execution_timeout = remaining - cleanup_timeout
    if execution_timeout <= 0.0:
        raise ValueError("SWE-bench runner deadline expired before execution")
    container_id = str(metadata["container_id"])
    returncode, stdout, stderr = run_contained_docker_exec(
        container_id,
        shell_command,
        timeout=execution_timeout,
        cleanup_timeout=cleanup_timeout,
        profile=SWEBENCH_EXEC_PROFILE,
    )
    return subprocess.CompletedProcess(
        ["contained-docker-exec", container_id],
        returncode,
        stdout,
        stderr,
    )


def _run_candidate_tests(
    metadata: dict[str, Any],
    command: list[str],
    *,
    deadline: float,
) -> subprocess.CompletedProcess[str]:
    shell_command = _shell_test_command(command, metadata=metadata)
    return _contained_candidate_command(
        metadata,
        shell_command,
        deadline=deadline,
    )


def _container_base_commit(
    metadata: dict[str, Any],
    *,
    deadline: float | None = None,
) -> str | None:
    try:
        completed = subprocess.run(
            [
                "docker", "exec", "--workdir", "/testbed",
                str(metadata["container_id"]),
                "git", "rev-parse", "--verify", "HEAD^{commit}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_deadline_timeout(
                deadline, cap=_DOCKER_CONTROL_TIMEOUT_S
            ),
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if completed.returncode != 0:
        return None
    commit = str(completed.stdout or "").strip().lower()
    return commit if _GIT_COMMIT_RE.fullmatch(commit) is not None else None


def _prepare_immutable_pytest_seed(
    *,
    image_digest: str,
    task_id: str,
    cache_key: str,
    deadline: float | None = None,
    cleanup_deadline: float | None = None,
) -> tuple[str, str] | None:
    """Keep one bounded immutable-image seed volume mounted for a baseline."""

    now = time.monotonic()
    if cleanup_deadline is None:
        cleanup_deadline = now + _BASELINE_TIMEOUT_S
    if deadline is None:
        deadline = cleanup_deadline - min(
            _BASELINE_CLEANUP_RESERVE_S,
            max(0.5, (cleanup_deadline - now) * 0.10),
        )

    seed_name = managed_container_name(
        prefix="crucible-visible-seed",
        scope=_BASELINE_CONTAINER_SCOPE,
        task_id=task_id,
        phase="seed",
    )
    volume_name = (
        "crucible-visible-seed-"
        + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]
    )
    seed_labels = managed_container_labels(
        scope=_BASELINE_CONTAINER_SCOPE,
        task_id=task_id,
        phase="visible-baseline-seed",
    )
    seed_labels["io.crucible.visible-baseline-key"] = cache_key
    try:
        created = _run_managed_baseline_docker(
            [
                "volume", "create",
                "--driver", "local",
                "--opt", "type=tmpfs",
                "--opt", "device=tmpfs",
                "--opt", (
                    f"o=size={_BASELINE_TESTBED_TMPFS_MB}m,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
                *label_args(seed_labels),
                volume_name,
            ],
            step_name="create bounded seed volume",
            timeout=_DOCKER_CONTROL_TIMEOUT_S,
            deadline=deadline,
        )
        if (created.stdout or "").strip() != volume_name:
            raise OSError("baseline volume create returned an unexpected name")
        seed_script = (
            "cd /testbed"
            " && tar --exclude=./.git -cf - . | "
            "tar --no-same-owner -xf - -C "
            + shlex.quote(_BASELINE_SEED_TARGET)
            + " && : > "
            + shlex.quote(_BASELINE_SEED_TARGET + "/.crucible-seed-ready")
            + " && exec tail -f /dev/null"
        )
        _run_managed_baseline_docker(
            [
                "run", "-d", "--rm",
                "--name", seed_name,
                *label_args(seed_labels),
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--pids-limit", "64",
                "--cpus", "1.0",
                "--memory", "1024m",
                "--memory-swap", "1024m",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", (
                    "type=volume,source="
                    + volume_name
                    + ",target="
                    + _BASELINE_SEED_TARGET
                    + ",volume-nocopy"
                ),
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
                "--entrypoint", "bash",
                image_digest,
                "-eu", "-o", "pipefail", "-c", seed_script,
            ],
            step_name="start immutable seed holder",
            timeout=60,
            deadline=deadline,
        )
        ready_path = _BASELINE_SEED_TARGET + "/.crucible-seed-ready"
        for _ in range(300):
            ready = subprocess.run(
                ["docker", "exec", seed_name, "test", "-e", ready_path],
                capture_output=True,
                text=True,
                check=False,
                timeout=_deadline_timeout(deadline, cap=5.0),
            )
            if ready.returncode == 0:
                return volume_name, seed_name
            container_state = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", seed_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=_deadline_timeout(deadline, cap=5.0),
            )
            if (
                container_state.returncode != 0
                or (container_state.stdout or "").strip() != "true"
            ):
                break
            time.sleep(min(0.1, _deadline_timeout(deadline, cap=0.1)))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    _best_effort_docker_cleanup(
        ["rm", "-f", seed_name],
        deadline=cleanup_deadline,
    )
    _best_effort_docker_cleanup(
        ["volume", "rm", "-f", volume_name],
        deadline=cleanup_deadline,
    )
    return None


def _run_clean_baseline(
    metadata: dict[str, Any],
    base_commit: str,
    command: list[str],
    cache_key: str,
    *,
    deadline: float | None = None,
) -> _PytestOutcome | None:
    """Run the same public tests in the pristine immutable benchmark image."""

    now = time.monotonic()
    if deadline is None:
        deadline = now + _BASELINE_TIMEOUT_S
    cleanup_reserve = min(
        _BASELINE_CLEANUP_RESERVE_S,
        max(0.5, (deadline - now) * 0.10),
    )
    execution_deadline = deadline - cleanup_reserve
    if execution_deadline <= now:
        return None
    image_digest = str(metadata["image_digest"])
    parser_kind = _test_runner_kind(command, metadata)
    if parser_kind is None:
        return None
    authoring_sources = _validated_authoring_bind_sources(
        metadata,
        deadline=execution_deadline,
    )
    if authoring_sources is None:
        return None
    # Every invocation gets a Docker-global unique name.  The cache lock
    # serializes ordinary same-identity callers, but unique names also protect
    # live baselines owned by another UID/cache root from cross-run removal.
    task_id = f"{cache_key}:{secrets.token_hex(16)}"
    container_name = managed_container_name(
        prefix="crucible-visible-baseline",
        scope=_BASELINE_CONTAINER_SCOPE,
        task_id=task_id,
        phase="test",
    )
    labels = managed_container_labels(
        scope=_BASELINE_CONTAINER_SCOPE,
        task_id=task_id,
        phase="visible-baseline",
    )
    labels["io.crucible.visible-baseline-key"] = cache_key
    try:
        reconcile_orphaned_containers(
            scope=_BASELINE_CONTAINER_SCOPE,
            run_docker=lambda arguments, *, step_name, timeout: (
                _run_managed_baseline_docker(
                    arguments,
                    step_name=step_name,
                    timeout=timeout,
                    deadline=execution_deadline,
                )
            ),
        )
    except OSError:
        return None
    seed_volume: str | None = None
    seed_container: str | None = None
    if parser_kind != "django-unittest":
        seed = _prepare_immutable_pytest_seed(
            image_digest=image_digest,
            task_id=task_id,
            cache_key=cache_key,
            deadline=execution_deadline,
            cleanup_deadline=deadline,
        )
        if seed is None:
            return None
        seed_volume, seed_container = seed
    commit_guard = (
        'test "$(git rev-parse --verify HEAD^{commit})" = '
        + shlex.quote(base_commit)
    )
    clean_mode = "-ffdx" if parser_kind == "django-unittest" else "-ffd"
    clean_base = (
        commit_guard
        + " && git reset --hard --quiet "
        + shlex.quote(base_commit)
        + " && git clean "
        + clean_mode
        + " --quiet"
        + " && "
        + commit_guard
        + ' && test -z "$(git status --porcelain=v1 --untracked-files=all)"'
    )
    # Never seed a clean base from the live post-model worktree. Django's
    # pure-Python suite starts from an empty bounded tmpfs plus the exact Git
    # object store. Other projects may depend on ignored build artifacts
    # provisioned in the benchmark image. A bounded named tmpfs remains mounted
    # by an immutable-image seed holder while this container mounts it at the
    # exact /testbed path; only the instance object store is overlaid and reset.
    prepare_prefix = "mkdir -p /tmp/home"
    prepare_prefix += " && mkdir -p " + shlex.quote(
        _BASELINE_TESTBED + "/.git"
    )
    source_mounts: list[str] = []
    prepare_testbed = (
        prepare_prefix
        + " && cd "
        + shlex.quote(_BASELINE_SOURCE_GIT_DIR)
        + " && tar -cf - . | "
        + "tar --no-same-owner -xf - -C "
        + shlex.quote(_BASELINE_TESTBED + "/.git")
    )
    source_mounts.extend((
        "--mount",
        "type=bind,source="
        + authoring_sources.git_dir
        + ",target="
        + _BASELINE_SOURCE_GIT_DIR
        + ",readonly,bind-propagation=rprivate",
    ))
    if parser_kind == "django-unittest":
        testbed_storage = (
            "--tmpfs",
            _BASELINE_TESTBED + ":rw,exec,nosuid,nodev,"
            f"size={_BASELINE_TESTBED_TMPFS_MB}m,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}",
        )
    else:
        assert seed_volume is not None
        testbed_storage = (
            "--mount",
            "type=volume,source="
            + seed_volume
            + ",target="
            + _BASELINE_TESTBED
            + ",volume-nocopy",
        )
    test_script = (
        prepare_testbed
        + " && cd "
        + shlex.quote(_BASELINE_TESTBED)
        + " && "
        + clean_base
        + " && export PYTHONPATH="
        + shlex.quote(f"{_BASELINE_TESTBED}:{_BASELINE_TESTBED}/src")
        + '${PYTHONPATH:+:"$PYTHONPATH"}'
        + " && "
        + _shell_test_command(
            command,
            testbed=_BASELINE_TESTBED,
            metadata=metadata,
        )
    )
    try:
        completed = subprocess.run(
            [
                "docker", "run", "--rm",
                "--name", container_name,
                *label_args(labels),
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--pids-limit", "512",
                "--cpus", "2.0",
                "--memory", "8192m",
                "--memory-swap", "8192m",
                "--ulimit", "nofile=4096:4096",
                "--user", f"{os.getuid()}:{os.getgid()}",
                *source_mounts,
                *testbed_storage,
                "--env", "HOME=/tmp/home",
                "--env", "PYTHONDONTWRITEBYTECODE=1",
                "--tmpfs", (
                    "/tmp:rw,exec,nosuid,nodev,"
                    f"size={_BASELINE_TMP_TMPFS_MB}m,mode=1777"
                ),
                "--entrypoint", "bash",
                "--workdir", "/",
                image_digest,
                "-eu", "-o", "pipefail", "-c",
                test_script,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_deadline_timeout(
                execution_deadline,
                cap=_BASELINE_TIMEOUT_S,
            ),
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    finally:
        _best_effort_docker_cleanup(
            ["rm", "-f", container_name],
            deadline=deadline,
        )
        if seed_container is not None:
            _best_effort_docker_cleanup(
                ["rm", "-f", seed_container],
                deadline=deadline,
            )
        if seed_volume is not None:
            _best_effort_docker_cleanup(
                ["volume", "rm", "-f", seed_volume],
                deadline=deadline,
            )
    outcome = _visible_test_outcome(completed, command, metadata)
    if outcome.returncode == 0:
        return outcome
    failure_total = _outcome_failure_total(outcome)
    if (
        outcome.returncode == 1
        and failure_total is not None
        and failure_total > 0
        and failure_total == len(outcome.failures)
    ):
        return outcome
    return None


def _baseline_comparison(
    metadata: dict[str, Any],
    base_commit: str | None,
    command: list[str],
    selection: _VisibleTestSelection,
    candidate: _PytestOutcome,
    *,
    deadline: float | None = None,
) -> _BaselineComparison:
    parser_kind = _test_runner_kind(command, metadata)
    image_digest = str(metadata.get("image_digest") or "")
    mandatory_clean_base = (
        parser_kind in {"django-unittest", "sympy-native"}
        or parser_kind == "pytest"
        and _IMMUTABLE_IMAGE_RE.fullmatch(image_digest) is not None
    )
    if candidate.returncode not in {0, 1}:
        return _BaselineComparison(
            mode="incomparable",
            effective_returncode=max(1, candidate.returncode),
            reason="candidate result is not structurally comparable",
        )
    if candidate.returncode == 0 and not mandatory_clean_base:
        return _BaselineComparison(
            mode="not_needed",
            effective_returncode=0,
            reason="candidate visible tests passed",
        )
    if not _is_supported_test_command(command, metadata):
        return _BaselineComparison(
            mode="not_applicable",
            effective_returncode=candidate.returncode,
            reason="clean-base comparison requires a supported test command",
        )
    if base_commit is None:
        return _BaselineComparison(
            mode="unavailable",
            effective_returncode=max(1, candidate.returncode),
            reason="immutable clean-base identity could not be established",
        )
    identity = _baseline_identity(metadata, base_commit, command, selection)
    if identity is None:
        return _BaselineComparison(
            mode="unavailable",
            effective_returncode=max(1, candidate.returncode),
            reason="immutable image digest or base commit is invalid",
        )
    identity_key = _identity_key(identity)
    baseline, cache_hit, key, error = _cached_baseline_outcome(
        identity,
        lambda: _run_clean_baseline(
            metadata,
            base_commit,
            command,
            identity_key,
            deadline=deadline,
        ),
        deadline=deadline,
    )
    if baseline is None:
        return _BaselineComparison(
            mode="unavailable",
            effective_returncode=max(1, candidate.returncode),
            cache_hit=cache_hit,
            cache_key=key,
            reason=error,
        )
    return _compare_pytest_outcomes(
        candidate, baseline, cache_hit=cache_hit, cache_key=key
    )


def _run_visible_tests(
    parameters: dict[str, Any],
    metadata: dict[str, Any],
    *,
    deadline: float,
) -> _VisibleTestRun:
    command = _visible_test_command(parameters, metadata)
    selection = _select_visible_pytest_targets(
        parameters,
        metadata,
        command,
        deadline=deadline,
    )
    command.extend(selection.targets)
    base_commit = None
    image_digest = str(metadata.get("image_digest") or "")
    if (
        _is_supported_test_command(command, metadata)
        and _IMMUTABLE_IMAGE_RE.fullmatch(image_digest) is not None
    ):
        base_commit = _container_base_commit(metadata, deadline=deadline)
    completed = _run_candidate_tests(
        metadata,
        command,
        deadline=deadline,
    )
    candidate = _visible_test_outcome(completed, command, metadata)
    comparison = _baseline_comparison(
        metadata,
        base_commit,
        command,
        selection,
        candidate,
        deadline=deadline,
    )
    return _VisibleTestRun(
        completed,
        selection,
        comparison,
        tuple(command),
        candidate,
    )


def _visible_test_command(
    parameters: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    command = list(
        parameters.get("test_command")
        or _resolve_test_command(
            str(metadata.get("repo") or ""),
            str(metadata.get("version") or ""),
        )
        or ["pytest", "-q", "--tb=short"]
    )
    command.extend(str(arg) for arg in (parameters.get("test_args") or ()))
    return command


def _run_exact_visible_observations(
    parameters: dict[str, Any],
    metadata: dict[str, Any],
    *,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    from crucible.common.runners.observations import (
        canonical_attack_test_ids,
        exact_pytest_command,
        pytest_observer_failure,
        pytest_observer_status,
    )

    if deadline is None:
        deadline = time.monotonic() + _BASELINE_TIMEOUT_S
    attack_ids = parameters.get("breach_attack_ids")
    test_file = parameters.get("breach_test_file")
    if not isinstance(test_file, str) or not test_file:
        raise ValueError("breach_observations requires breach_test_file")
    identities = canonical_attack_test_ids(attack_ids, "python")
    observations: list[dict[str, Any]] = []
    for _attack_id, test_id in identities:
        node_id = f"{test_file}::{test_id}"
        command = exact_pytest_command(node_id, python_prefix=("python",))
        shell_command = (
            CONDA_ACTIVATE_PREFIX
            + "&& "
            + " ".join(shlex.quote(token) for token in command)
        )
        completed = _contained_candidate_command(
            metadata,
            shell_command,
            deadline=deadline,
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
    stdout_obj: TextIO = stdout if stdout is not None else sys.stdout
    config = json.loads(stdin_obj.read())
    raw_parameters = (
        config.get("parameters") or {} if isinstance(config, dict) else {}
    )
    parameters = for_language(raw_parameters, "python")
    metadata, error = _validated_metadata(parameters)
    if error is not None or metadata is None:
        return _emit_error(
            stdout_obj,
            error or "invalid authoring metadata",
            hidden="hidden evaluation fields" in (error or ""),
        )

    role = parameters.get("role", "dev_verify")
    if role == "build":
        stdout_obj.write(json.dumps({
            "values": {
                "returncode": 0.0,
                "python_returncode": 0.0,
                "duration_s": 0.0,
                "swebench_build_skipped": 1.0,
            },
            "raw_output": "build already provisioned in authoring image",
        }))
        stdout_obj.flush()
        return 0
    if role not in {"verify", "dev_verify"}:
        return _emit_error(stdout_obj, f"unsupported runner role: {role!r}")

    started = time.monotonic()
    try:
        deadline = _runner_deadline(parameters)
        run = _run_visible_tests(
            parameters,
            metadata,
            deadline=deadline,
        )
    except ValueError as exc:
        return _emit_error(stdout_obj, str(exc))
    completed = run.completed
    selection = run.selection
    comparison = run.comparison
    result_returncode = comparison.effective_returncode
    duration = time.monotonic() - started
    combined = (completed.stdout or "") + (completed.stderr or "")
    candidate_summary = dict(run.candidate.summary)
    effective_summary = dict(candidate_summary)
    if comparison.mode == "preexisting_only":
        # The workflow gate consumes generic ``tests_failed`` independently of
        # the subprocess return code.  Publish the clean-base-adjusted counts
        # there, while retaining the raw candidate counts in an explicit
        # namespace for diagnostics and repair prompts.
        effective_summary["tests_failed"] = 0.0
        effective_summary["tests_errored"] = 0.0
    values: dict[str, float] = {
        "returncode": float(result_returncode),
        "python_returncode": float(result_returncode),
        "visible_test_candidate_returncode": float(completed.returncode),
        "duration_s": float(duration),
        "python_duration_s": float(duration),
        "visible_tests_only": 1.0,
    }
    values.update(_selection_values(selection))
    values.update(_comparison_values(comparison))
    for key in sorted({*candidate_summary, "tests_failed", "tests_errored"}):
        value = candidate_summary.get(key, 0.0)
        values[f"visible_test_candidate_{key}"] = value
    for key, value in effective_summary.items():
        values[key] = value
        values[f"python_{key}"] = value
    payload = {
        "values": values,
        "raw_output": _selection_raw_output(combined, selection, comparison),
    }
    if parameters.get("breach_observations"):
        from crucible.common.runners.observations import filter_test_observations

        try:
            payload["test_observations"] = filter_test_observations(
                _run_exact_visible_observations(
                    parameters,
                    metadata,
                    deadline=deadline,
                ),
                parameters.get("breach_attack_ids"),
                language="python",
            )
        except BreachObservationChannelError as exc:
            failure = breach_observation_failure("channel", exc)
            values["returncode"] = 2.0
            values["python_returncode"] = 2.0
            payload["raw_output"] = _selection_raw_output(
                failure["detail"], selection, comparison
            )
            payload["test_observations"] = []
            payload[BREACH_OBSERVATION_FAILURE_KEY] = failure
            result_returncode = 2
        except OSError as exc:
            failure = breach_observation_failure(
                "infrastructure",
                f"{type(exc).__name__}: {exc}",
            )
            values["returncode"] = 2.0
            values["python_returncode"] = 2.0
            payload["raw_output"] = _selection_raw_output(
                failure["detail"], selection, comparison
            )
            payload["test_observations"] = []
            payload[BREACH_OBSERVATION_FAILURE_KEY] = failure
            result_returncode = 2
        except ValueError as exc:
            failure = breach_observation_failure("channel", exc)
            values["returncode"] = 2.0
            values["python_returncode"] = 2.0
            payload["raw_output"] = _selection_raw_output(
                failure["detail"], selection, comparison
            )
            payload["test_observations"] = []
            payload[BREACH_OBSERVATION_FAILURE_KEY] = failure
            result_returncode = 2
    stdout_obj.write(json.dumps(payload))
    stdout_obj.flush()
    return result_returncode


if __name__ == "__main__":
    raise SystemExit(main())
