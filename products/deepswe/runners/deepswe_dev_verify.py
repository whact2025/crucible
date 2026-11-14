# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE per-round development verifier — project-native COMPILE + tests.

The ``dev_verify`` role's runner (distinct from ``deepswe_verifier``,
which runs the dataset's hidden test gate). This runner COMPILES then runs
the PROJECT'S OWN tests inside the per-task AUTHORING container via
``docker exec``, so the Implementer and Defender get a real,
project-native pass/fail signal at author time — and every breach
round gates on the Defender's ``test_<attack_id>`` files, which the
project's test discovery picks up.  Every command runs below a trusted
subreaper which fails the gate and terminates the complete descendant tree if
a test leaves background work behind.

Build gate: deepswe skips the host build phase entirely (``_build_verify``
sets ``b = None`` for the dev_verify gate), so the per-language command here
is compile-then-test (see ``_LANG_TO_COMMAND``) — making a build a part of
EVERY language flow, so a signature/symbol/wiring deviation fails as a build
error rather than slipping through to a runtime test error.

DeepSWE images are dependency-baked while the host workspace is not. The
in-container runner therefore provides authoring-time feedback in the same
environment used by the task.

It is deliberately NOT the dataset gate. ``/tests/`` is unmounted in
the authoring container, so this runner can only ever run the
project's own tests + the Defender's tests — leakage of the hidden
suite is structurally impossible here.

Subprocess contract (per ``assay.bench``):

- Reads BenchConfig JSON on stdin.
- Emits a JSON KPI payload on stdout.
- Exit 0 iff the project's tests passed (``tests_failed == 0`` and the
  in-container command returned 0).
- Exit non-zero on any failure (tests failed, docker exec failure,
  unknown language, missing metadata).

Recognized ``parameters``:

- ``language``: dispatch key (``python`` / ``rust`` / ``go`` /
  ``typescript`` / ``javascript``). Case-insensitive. Also drives
  output-count parsing even when a ``dev_verify_command`` override is in
  effect.
- ``deepswe_metadata``: dict with ``{"container_id": "<id>"}`` — the
  AUTHORING container's id, populated by the deepswe worker. May also
  carry ``dev_verify_command``: a crucible-owned per-project test-command
  override (see ``crucible.products.deepswe.dev_verify_overrides``) that
  is honored OVER the per-language default in ``_LANG_TO_COMMAND``. Strict
  clean-base comparison additionally consumes ``baseline_container_id``
  (a separate pristine authoring container), immutable ``image_digest``, and
  ``base_commit``. The optional ``dev_verify_parser`` selects one of
  ``pytest``, ``rust-libtest``, ``go-json``, ``jest``, ``vitest``, ``mocha``,
  ``node-tap``, or ``ava-tap`` when the exact command/output cannot be
  detected safely.
- ``role``: injected by ``_build_verify``; only ``"dev_verify"`` (or
  the unset default) is honored. Other roles are a routing bug.
- ``declared_verification_surface``: workflow-owned, normalized slice paths.
  Strong native test-path matches are executed candidate-only, one file at a
  time, and each file must produce a strict non-vacuous result. The primary
  suite and its clean-base cache identity remain unchanged.
- ``crate`` / ``cargo_features`` / ``cargo_test_args``: sealed Rust breach
  selection. The bundled Rust command applies these to the in-container Cargo
  invocation so a generated integration test cannot be omitted by workspace
  defaults. Project command overrides are rejected by breach preflight.

KPI shape emitted::

    {
      "values": {
        "tests_passed": N, "tests_failed": M, "tests_skipped": K,
        "returncode": <int>,
      },
      "wall_seconds": <float>,
    }

Per-language commands are defaults. Some projects (adaptix,
mashumaro) carry custom ``nox`` / ``tox`` orchestrators that a plain
``pytest`` invocation may bypass — those get a per-project
``dev_verify_command`` override via
``crucible.products.deepswe.dev_verify_overrides``, honored here over the
per-language default.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import PurePosixPath
from typing import Any

from crucible.common.runners.breach_observation import (
    BREACH_OBSERVATION_FAILURE_KEY,
    breach_observation_failure,
)
from crucible.common.runners.declared_verification import (
    ACCEPTED_VERIFICATION_REFERENCE_KEY,
    DECLARED_VERIFICATION_REFERENCE_OUTPUT_KEY,
    DECLARED_VERIFICATION_SURFACE_KEY,
    accepted_verification_reference_payload,
    parse_accepted_verification_reference,
    parse_declared_verification_surface,
)
from crucible.common.runners.parameters import for_language
from crucible.common.runners.pytest_policy import PYTEST_DEV_VERIFY_MAXFAIL_ARG
from crucible.common.runners.repair_progress import (
    REPAIR_FAILURE_SET_OUTPUT_KEY,
    RepairFailure,
    build_repair_failure_set,
)
from crucible.common.runners.container_exec import (
    CONTAINMENT_UNUSABLE_RC,
    run_contained_docker_exec,
)
from crucible.products.deepswe.runners.dev_verify_outcome import (
    BaselineComparison,
    TestOutcome,
    baseline_identity,
    cached_baseline_outcome,
    compare_outcomes,
    parse_test_outcome,
    strict_tap_test_records,
)

logger = logging.getLogger(__name__)


# Per-call timeout for the in-container test run. Author-time test
# runs are bounded; a single focused suite clears this comfortably. A
# runaway suite that exceeds it surfaces as returncode 124.
_DEV_VERIFY_TIMEOUT_SEC = 1800
# Tier repair prompts retain at most 8,000 characters per build/verify record.
# Match that outer bound so a bounded multi-failure pytest run can carry all
# of its short tracebacks and terminal failure identities into one repair
# turn.  The subprocess timeout and pytest max-failure policy bound the work
# and memory before this final diagnostic cap is applied.
_RAW_OUTPUT_LIMIT = 8_000
_IMMUTABLE_IMAGE_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_BASE_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")
_CONTAINER_REFERENCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_DECLARED_JS_TS_TEST_RE = re.compile(
    r"(?:\.test|\.spec|_test|_tests)\.(?:tsx?|jsx?|mjs|cjs)$",
    re.IGNORECASE,
)
_DECLARED_TEST_DIRS = frozenset({"test", "tests", "testing"})
_DECLARED_TEST_HELPER_DIRS = frozenset({
    "data",
    "fixture",
    "fixtures",
    "helper",
    "helpers",
    "resource",
    "resources",
    "sample",
    "samples",
    "support",
    "test_data",
    "testdata",
})
_MAX_DECLARED_TEST_FILES = 64
_RUNNER_CLEANUP_RESERVE_MAX_SEC = 5.0
_RUNNER_CLEANUP_RESERVE_FRACTION = 0.10
_SEALED_NETWORK_FAILURE_RE = re.compile(
    r"(?:network is unreachable|temporary failure in name resolution|"
    r"could not resolve host|no such host|dial (?:tcp|udp): lookup)",
    re.IGNORECASE,
)
_GO_REMOTE_DEPENDENCY_RE = re.compile(
    r"(?:proxy\.golang\.org|sum\.golang\.org|"
    r"github\.com/[^\s]+/@v/|go:\s+downloading)",
    re.IGNORECASE,
)
_TS_CLEAN_SUPERVISOR_FAILURE_RE = re.compile(
    r"(?:Error:\s+write\s+EPIPE|code:\s*['\"]EPIPE['\"])",
    re.IGNORECASE,
)
_CONTAINMENT_EXECUTABLE_FAILURE_RE = re.compile(
    r"(?:docker not on PATH|cc is required to build the static exec "
    r"supervisor|cannot stage static exec supervisor)",
    re.IGNORECASE,
)
CLEAN_INFRASTRUCTURE_PROVENANCE_KEY = (
    "dev_verify_clean_infrastructure_provenance"
)
CLEAN_INFRASTRUCTURE_PROVENANCE_SCHEMA = (
    "crucible.deepswe-clean-infrastructure.v1"
)
PROVENANCE_INFRASTRUCTURE_FLAGS = (
    "dev_verify_supervisor_unavailable",
    "dev_verify_dependency_unavailable",
    "dev_verify_parser_unavailable",
)


class _RunnerDeadlineExceeded(RuntimeError):
    """The workflow-owned runner execution window has closed."""


class _RunnerDeadline:
    """One absolute execution deadline shared by every nested command."""

    def __init__(self, *, expires_at: float, cleanup_reserve: float) -> None:
        self.expires_at = expires_at
        self.cleanup_reserve = cleanup_reserve

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> _RunnerDeadline:
        raw_timeout = parameters.get("runner_timeout_s")
        if raw_timeout is None:
            timeout = float(_DEV_VERIFY_TIMEOUT_SEC)
            cleanup_reserve = _RUNNER_CLEANUP_RESERVE_MAX_SEC
            execution_seconds = timeout
        else:
            if (
                isinstance(raw_timeout, bool)
                or not isinstance(raw_timeout, (int, float))
                or not math.isfinite(float(raw_timeout))
                or float(raw_timeout) <= 0.0
            ):
                raise ValueError("runner_timeout_s must be a positive number")
            timeout = float(raw_timeout)
            cleanup_reserve = min(
                _RUNNER_CLEANUP_RESERVE_MAX_SEC,
                max(0.001, timeout * _RUNNER_CLEANUP_RESERVE_FRACTION),
            )
            execution_seconds = max(0.0, timeout - cleanup_reserve)
        return cls(
            expires_at=time.monotonic() + execution_seconds,
            cleanup_reserve=cleanup_reserve,
        )

    def remaining(self, *, cap: float | None = None) -> float:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0.0:
            raise _RunnerDeadlineExceeded(
                "workflow-owned development-verifier deadline expired"
            )
        return min(remaining, cap) if cap is not None else remaining

# Per-language project-native test command, run as ``bash -c`` inside
# the authoring container at /app. Strawman per the goal-doc; calibrate
# against the pilot.
#
# Deliberately the FULL project suite, NOT a slice-scoped subset. The
# per-round gate must catch a slice that *regresses* the baseline, and
# scoping to the slice's own files would hide exactly that: e.g. adding
# a required field to a shared schema without seeding its default
# breaks every construction site project-wide, which a scoped run would
# miss. A correct incomplete increment (optional field + inline default)
# does not red the suite, so the full suite is already the right
# discriminator. The plan-critic's required-field-atomicity rule
# (deepswe_plan_critic rule 7) is the upstream lever that keeps each
# slice non-regressing — not a looser gate here.
# Each command is COMPILE-then-TEST in a single in-container run (``&&``),
# so a signature / symbol / wiring deviation fails as a BUILD error before
# the test phase — instead of slipping through to a murky runtime failure.
# This closes the gap the smoke RCA found: deepswe SKIPS the host build phase
# entirely (``_build_verify`` sets ``b = None`` for the dev_verify gate), so
# this in-container gate command is the ONLY place a build can run. Previously
# only rust compiled (as a side-effect of ``cargo test``) while go / ts / js
# had no compile gate at all — a wrong public signature on those languages
# surfaced (if at all) only as a runtime test error.
#
# Per language:
#   - python:  interpreted — pytest's collection import IS the compile gate
#              (a syntax/import error fails collection), so no separate step.
#   - rust:    ``cargo build`` is explicit; ``cargo test`` also compiles.
#   - go:      ``go build ./...`` compiles ALL packages (``go test`` alone
#              only compiles the packages it tests — the real gap).
#   - ts / js: ``npm run build --if-present`` runs the project's build script
#              (e.g. ``tsc``) when defined, and is a no-op (rc 0) when absent.
# A per-project ``dev_verify_command`` override REPLACES the whole command, so
# an override that needs a compile gate must include its own build prefix.
_LANG_TO_COMMAND: dict[str, str] = {
    "python": (
        f"cd /app && pytest {PYTEST_DEV_VERIFY_MAXFAIL_ARG} --tb=short -q"
    ),
    "rust": "cd /app && cargo build --quiet && cargo test --quiet",
    "go": "cd /app && go build ./... && go test -json ./...",
    "typescript": "cd /app && npm run build --if-present && npm test --silent",
    "javascript": "cd /app && npm run build --if-present && npm test --silent",
}


def _is_declared_native_test_path(path: str, language: str) -> bool:
    """Recognize only strong, language-native test path conventions.

    DeepSWE explicitly permits production files whose names contain ``test``
    (for example ``browser_test_runner.js`` or ``base_test.py``).  Avoid the
    broad SWE-bench source-stripping predicate here: a declared path becomes a
    blocking exact-test obligation only when its language convention is
    sufficiently unambiguous.
    """

    parsed = PurePosixPath(path)
    filename = parsed.name.casefold()
    directories = tuple(part.casefold() for part in parsed.parts[:-1])
    in_test_tree = any(
        part in _DECLARED_TEST_DIRS or part.endswith("_tests")
        for part in directories
    )
    if language == "python":
        return (
            in_test_tree
            and filename.endswith(".py")
            and (
                filename.startswith(("test_", "test-", "test."))
                or filename.endswith("_test.py")
            )
        )
    if language in {"javascript", "typescript"}:
        strong_convention = _DECLARED_JS_TS_TEST_RE.search(filename) is not None
        javascript_suffix = parsed.suffix.casefold() in {
            ".js", ".jsx", ".mjs", ".cjs",
        }
        typescript_suffix = (
            language == "typescript"
            and parsed.suffix.casefold() in {".ts", ".tsx"}
            and not filename.endswith(".d.ts")
        )
        helper_subtree = any(
            part in _DECLARED_TEST_HELPER_DIRS for part in directories
        )
        return strong_convention or (
            in_test_tree
            and not helper_subtree
            and (javascript_suffix or typescript_suffix)
        )
    if language == "rust":
        return (
            len(parsed.parts) >= 2
            and parsed.parts[-2] == "tests"
            and parsed.suffix == ".rs"
        )
    # Go's package runner cannot prove that one file, rather than a sibling,
    # contributed a non-vacuous test without a separate AST authority.  The
    # unchanged full-suite command remains authoritative until that protocol
    # exists.
    return False


def _rust_breach_command(parameters: dict[str, Any]) -> str:
    """Render the Cargo-aware default command from a sealed payload."""
    crate = parameters.get("crate")
    if crate is not None and (not isinstance(crate, str) or not crate):
        raise ValueError("Rust runner parameter 'crate' must be a non-empty string")
    scope = f" -p {shlex.quote(crate)}" if crate else ""

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
    feature_args = (
        " --features " + shlex.quote(",".join(features))
        if features else ""
    )
    raw_test_args = _string_list("cargo_test_args")
    if parameters.get("breach_observations"):
        forbidden = {"--nocapture", "--show-output"}.intersection(
            str(argument) for argument in raw_test_args
        )
        if forbidden:
            raise ValueError(
                "breach observations require captured libtest output; "
                f"forbidden arguments: {sorted(forbidden)!r}"
            )
        raw_test_args = list(raw_test_args)
        if "--" not in raw_test_args:
            raw_test_args.extend(("--", "--test-threads=1"))
        elif not any(
            argument.startswith("--test-threads=")
            for argument in raw_test_args
        ):
            raw_test_args.append("--test-threads=1")
    test_args = (
        " " + " ".join(shlex.quote(str(arg)) for arg in raw_test_args)
        if raw_test_args else ""
    )
    return (
        f"cd /app && cargo build --quiet{scope}{feature_args} && "
        f"cargo test{scope}{feature_args}{test_args}"
    )

# pytest summary parsing — shared shape with
# crucible.common.runners.lang.python._parse_pytest_summary.
_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_PYTEST_SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)

# cargo test summary: "test result: ok. 12 passed; 0 failed; 1 ignored"
_CARGO_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_CARGO_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_CARGO_IGNORED_RE = re.compile(r"(\d+)\s+ignored", re.IGNORECASE)

# go test: "ok  pkg  0.1s" / "--- FAIL" / "FAIL pkg"; counts are not
# summarized uniformly, so we fall back to returncode-only semantics
# for go/js/ts and only parse counts where the framework summary is
# stable (pytest, cargo).


def _run_docker_exec(
    container_id: str,
    command: str,
    *,
    timeout: float = _DEV_VERIFY_TIMEOUT_SEC,
    deadline: _RunnerDeadline | None = None,
) -> tuple[int, str, str]:
    """Run one contained command and return ``(rc, stdout, stderr)``.

    The containment helper preserves the source container's exact workspace,
    environment, and staged toolchain while proving that no command descendant
    survives this call.  Timeout surfaces as returncode 124; missing Docker or
    the static-helper compiler as 127.
    """
    bounded_timeout = (
        deadline.remaining(cap=float(timeout))
        if deadline is not None else float(timeout)
    )
    cleanup_grace = (
        deadline.cleanup_reserve
        if deadline is not None else 5.0
    )
    return run_contained_docker_exec(
        container_id,
        command,
        timeout=bounded_timeout,
        cleanup_timeout=cleanup_grace,
    )


def _declared_container_path_state(
    container_id: str,
    test_file: str,
    *,
    deadline: _RunnerDeadline | None = None,
) -> str:
    """Return ``safe``/``absent`` or reject an unsafe ``/app`` path."""

    absolute = (PurePosixPath("/app") / PurePosixPath(test_file)).as_posix()
    quoted = shlex.quote(absolute)
    command = (
        f"candidate={quoted}; "
        'if [ ! -e "$candidate" ] && [ ! -L "$candidate" ]; then '
        'printf "absent\\n"; exit 0; fi; '
        '[ -f "$candidate" ] && [ ! -L "$candidate" ] || exit 41; '
        'resolved=$(readlink -f "$candidate") || exit 42; '
        'case "$resolved" in /app/*) ;; *) exit 43 ;; esac; '
        '[ "$resolved" = "$candidate" ] || exit 44; '
        'printf "safe\\n"'
    )
    rc, stdout, stderr = _run_docker_exec(
        container_id,
        command,
        timeout=30,
        deadline=deadline,
    )
    observed = [line.strip() for line in stdout.splitlines() if line.strip()]
    if rc == 0 and observed in (["safe"], ["absent"]):
        return observed[0]
    raise ValueError(
        "declared verification path is missing, aliased, or outside /app: "
        f"path={test_file!r}, rc={rc}, stderr={stderr[-500:]!r}"
    )


def _validate_declared_container_path(
    container_id: str,
    test_file: str,
    *,
    deadline: _RunnerDeadline | None = None,
) -> None:
    """Prove one candidate file is an ordinary, unaliased ``/app`` file."""

    state = _declared_container_path_state(
        container_id,
        test_file,
        deadline=deadline,
    )
    if state != "safe":
        raise ValueError(
            f"declared candidate test file is absent: path={test_file!r}"
        )


def _baseline_metadata(
    metadata: dict[str, Any],
    *,
    candidate_container_id: str,
) -> tuple[dict[str, str] | None, str]:
    """Validate the workflow-owned clean-base container identity.

    Baseline comparison is opt-in until the DeepSWE workflow supplies all
    identity fields.  A partially supplied envelope fails closed instead of
    silently degrading to candidate-only verification.
    """

    keys = ("baseline_container_id", "image_digest", "base_commit")
    present = [
        key
        for key in keys
        if metadata.get(key) is not None and metadata.get(key) != ""
    ]
    if not present:
        return None, ""
    if len(present) != len(keys):
        return None, (
            "clean-base verification metadata is incomplete; required="
            + ",".join(keys)
        )
    baseline_container_id = metadata["baseline_container_id"]
    image_digest = metadata["image_digest"]
    base_commit = metadata["base_commit"]
    if (
        not isinstance(baseline_container_id, str)
        or _CONTAINER_REFERENCE_RE.fullmatch(baseline_container_id) is None
        or baseline_container_id == candidate_container_id
    ):
        return None, "baseline_container_id is invalid or aliases the candidate"
    if (
        not isinstance(image_digest, str)
        or _IMMUTABLE_IMAGE_RE.fullmatch(image_digest) is None
    ):
        return None, "image_digest must be an immutable sha256 reference"
    if (
        not isinstance(base_commit, str)
        or _BASE_COMMIT_RE.fullmatch(base_commit.lower()) is None
    ):
        return None, "base_commit must be an immutable Git object id"
    return {
        "baseline_container_id": baseline_container_id,
        "image_digest": image_digest,
        "base_commit": base_commit.lower(),
    }, ""


def _probe_clean_baseline_container(
    baseline: dict[str, str],
    *,
    deadline: _RunnerDeadline | None = None,
) -> str:
    """Return an error unless the baseline sandbox is the claimed clean base."""

    container_id = baseline["baseline_container_id"]
    try:
        probe_timeout = (
            deadline.remaining(cap=30.0)
            if deadline is not None else 30.0
        )
        inspected = subprocess.run(
            [
                "docker", "inspect", "--format",
                '{{index .Config.Labels "io.crucible.image-digest"}}',
                container_id,
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=probe_timeout,
            check=False,
        )
    except _RunnerDeadlineExceeded:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"baseline image digest probe failed: {type(exc).__name__}"
    if (
        inspected.returncode != 0
        or (inspected.stdout or "").strip().lower()
        != baseline["image_digest"].lower()
    ):
        return (
            "baseline image digest does not match clean-base authority: "
            f"returncode={inspected.returncode}"
        )

    # Every command inside either authoring container uses the trusted
    # subreaper.  Killing only a timed-out Docker CLI can otherwise leave its
    # in-container process (including a Git fsmonitor hook) running against the
    # workspace after this clean-base probe has returned.
    contained_probes = (
        (
            "cd /app && git rev-parse --verify 'HEAD^{commit}'",
            baseline["base_commit"],
            "baseline Git HEAD",
        ),
        (
            "cd /app && git status --porcelain=v1 -z --untracked-files=all",
            "",
            "baseline worktree cleanliness",
        ),
    )
    for command, expected, label in contained_probes:
        try:
            returncode, output, _stderr = _run_docker_exec(
                container_id,
                command,
                timeout=30.0,
                deadline=deadline,
            )
        except _RunnerDeadlineExceeded:
            raise
        except OSError as exc:
            return f"{label} probe failed: {type(exc).__name__}"
        observed = output.strip().lower()
        if returncode != 0 or observed != expected.lower():
            return (
                f"{label} does not match clean-base authority: "
                f"returncode={returncode}"
            )
    return ""


def _expected_breach_test_delta(parameters: dict[str, Any]) -> int:
    if not parameters.get("breach_observations"):
        return 0
    raw_ids = parameters.get("breach_attack_ids")
    if not isinstance(raw_ids, (list, tuple)):
        return 0
    return len({attack_id for attack_id in raw_ids if isinstance(attack_id, str)})


def _outcome_values(outcome: TestOutcome) -> dict[str, float]:
    values = {
        key: float(value)
        for key, value in outcome.summary
    }
    values["dev_verify_structurally_valid"] = float(
        outcome.structurally_valid
    )
    values["dev_verify_failure_set_complete"] = float(
        outcome.failure_set_complete
    )
    if outcome.tests_run is not None:
        values["dev_verify_tests_run"] = float(outcome.tests_run)
    return values


def _repair_failure_set_payload(
    *,
    effective_rc: int,
    values: dict[str, float],
    candidate: TestOutcome,
    command: str,
    declared_reference: dict[str, object] | None,
) -> dict[str, object] | None:
    """Emit progress evidence only for one complete declared-file failure set."""

    if (
        effective_rc != 1
        or declared_reference is None
        or not candidate.structurally_valid
        or candidate.returncode != 0
        or candidate.tests_run is None
        or candidate.tests_run <= 0
        or candidate.failures
        or not candidate.failure_set_complete
    ):
        return None

    def counter(name: str, *, default: int | None = None) -> int | None:
        value = values.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or not float(value).is_integer()
        ):
            return None
        return int(value)

    broad_failed = counter("tests_failed", default=0)
    broad_errored = counter("tests_errored", default=0)
    broad_passed = counter("tests_passed")
    broad_tests_run = counter("dev_verify_tests_run")
    broad_structurally_valid = counter("dev_verify_structurally_valid")
    broad_neutral = {
        name: counter(name, default=0)
        for name in (
            "tests_skipped",
            "tests_expected_failed",
            "tests_unexpected_successes",
            "tests_deselected",
            "tests_rerun",
        )
    }
    declared_required = counter("dev_verify_declared_test_files_required")
    declared_executed = counter("dev_verify_declared_test_files_executed")
    declared_passed = counter("dev_verify_declared_test_files_passed")
    declared_failed = counter("dev_verify_declared_test_files_failed")
    declared_structural = counter(
        "dev_verify_declared_test_files_structural_failures",
    )
    if (
        None in {
            broad_failed,
            broad_errored,
            broad_passed,
            broad_tests_run,
            broad_structurally_valid,
            declared_required,
            declared_executed,
            declared_passed,
            declared_failed,
            declared_structural,
            *broad_neutral.values(),
        }
        or broad_failed != 0
        or broad_errored != 0
        or broad_passed <= 0
        or broad_tests_run != candidate.tests_run
        or broad_structurally_valid != 1
        or broad_tests_run != (
            broad_passed
            + broad_neutral["tests_skipped"]
            + broad_neutral["tests_expected_failed"]
            + broad_neutral["tests_unexpected_successes"]
        )
        or declared_structural != 0
        or declared_failed == 0
        or declared_required != declared_executed
        or declared_required != declared_passed + declared_failed
    ):
        return None
    try:
        reference = parse_accepted_verification_reference(declared_reference)
    except (TypeError, ValueError):
        return None
    if len(reference.outcomes) != declared_required:
        return None

    neutral_keys = {
        "tests_passed", "tests_failed", "tests_errored",
    }
    roster: list[dict[str, object]] = []
    failures: list[RepairFailure] = []
    observed_passed = 0
    observed_failed = 0
    for entry in reference.outcomes:
        try:
            outcome = TestOutcome.from_payload(entry.outcome)
        except ValueError:
            return None
        summary = dict(outcome.summary)
        failed_count = (
            summary.get("tests_failed", 0)
            + summary.get("tests_errored", 0)
        )
        counted_tests = sum(
            summary.get(key, 0)
            for key in (
                "tests_passed",
                "tests_failed",
                "tests_errored",
                "tests_skipped",
                "tests_expected_failed",
                "tests_unexpected_successes",
            )
        )
        if (
            not outcome.structurally_valid
            or not outcome.failure_set_complete
            or outcome.tests_run is None
            or outcome.tests_run <= 0
            or outcome.tests_run != counted_tests
            or failed_count != len(outcome.failures)
            or (outcome.returncode == 0 and outcome.failures)
            or (outcome.returncode == 1 and not outcome.failures)
        ):
            return None
        observed_passed += int(outcome.returncode == 0)
        observed_failed += int(outcome.returncode == 1)
        roster.append({
            "command_sha256": entry.command_sha256,
            "neutral_summary": [
                [key, value]
                for key, value in outcome.summary
                if key not in neutral_keys
            ],
            "parser_kind": outcome.parser_kind,
            "path": entry.path,
            "tests_run": outcome.tests_run,
        })
        failures.extend(
            RepairFailure(
                failure_id=json.dumps(
                    [entry.path, node_id],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                status=status,
                fingerprint=fingerprint,
            )
            for node_id, status, fingerprint in outcome.failures
        )
    if (
        not failures
        or observed_passed != declared_passed
        or observed_failed != declared_failed
    ):
        return None
    try:
        return build_repair_failure_set(
            surface={
                "schema_version": "crucible.deepswe-repair-surface.v1",
                "broad_command_sha256": hashlib.sha256(
                    command.encode("utf-8")
                ).hexdigest(),
                "broad_neutral_counters": broad_neutral,
                "broad_parser_kind": candidate.parser_kind,
                "broad_tests_run": candidate.tests_run,
                "declared_language": reference.language,
                "declared_parser_kind": reference.parser_kind,
                "declared_roster": sorted(
                    roster,
                    key=lambda item: str(item["path"]),
                ),
                "declared_surface_sha256": (
                    reference.predecessor_surface_sha256
                ),
            },
            failures=tuple(failures),
        )
    except ValueError:
        # Progress evidence is optional. A bounded but unusually large
        # product result must preserve the real verifier failure instead of
        # crashing merely because it cannot earn an extra repair attempt.
        return None


def _comparison_values(
    comparison: BaselineComparison,
    *,
    baseline_attempted: bool,
) -> dict[str, float]:
    return {
        "dev_verify_baseline_attempted": float(baseline_attempted),
        "dev_verify_baseline_cache_hit": float(comparison.cache_hit),
        "dev_verify_baseline_passed": float(
            comparison.mode in {"baseline_match", "preexisting_only"}
        ),
        "dev_verify_baseline_unavailable": float(
            comparison.mode == "unavailable"
        ),
        "dev_verify_candidate_incomparable": float(
            comparison.mode == "incomparable"
        ),
        "dev_verify_preexisting_failures": float(
            comparison.preexisting_failures
        ),
        "dev_verify_new_failures": float(comparison.new_failures),
        "dev_verify_worsened_failures": float(
            comparison.worsened_failures
        ),
    }


def _candidate_infrastructure_values(
    *,
    language: str,
    process_returncode: int,
    output: str,
    parser_kind: str,
    clean_probe: bool = False,
    proven_clean_flags: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Identify failures proven runner-owned outside the candidate checkout."""

    supervisor_signature = bool(
        language in {"javascript", "typescript"}
        and process_returncode == 125
        and "crucible supervisor:" in output
        and _TS_CLEAN_SUPERVISOR_FAILURE_RE.search(output)
    )
    dependency_signature = bool(
        language == "go"
        and _SEALED_NETWORK_FAILURE_RE.search(output)
        and _GO_REMOTE_DEPENDENCY_RE.search(output)
    )
    supervisor_unavailable = supervisor_signature and (
        clean_probe
        or "dev_verify_supervisor_unavailable" in proven_clean_flags
    )
    dependency_unavailable = dependency_signature and (
        clean_probe
        or "dev_verify_dependency_unavailable" in proven_clean_flags
    )
    executable_unavailable = bool(
        process_returncode == 127
        and _CONTAINMENT_EXECUTABLE_FAILURE_RE.search(output)
    )
    parser_signature = parser_kind in {"unsupported", "ava-unsupported"}
    parser_unavailable = parser_signature and (
        clean_probe
        or "dev_verify_parser_unavailable" in proven_clean_flags
    )
    return {
        "dev_verify_supervisor_signature": float(supervisor_signature),
        "dev_verify_dependency_signature": float(dependency_signature),
        "dev_verify_parser_signature": float(parser_signature),
        "dev_verify_supervisor_unavailable": float(supervisor_unavailable),
        "dev_verify_dependency_unavailable": float(dependency_unavailable),
        "dev_verify_executable_unavailable": float(executable_unavailable),
        "dev_verify_parser_unavailable": float(parser_unavailable),
        "dev_verify_infrastructure_unavailable": float(
            supervisor_unavailable
            or dependency_unavailable
            or executable_unavailable
            or parser_unavailable
        ),
    }


def _validated_clean_infrastructure_flags(
    metadata: dict[str, Any],
    *,
    language: str,
    command: str,
) -> frozenset[str]:
    """Return only exact workflow-bound clean-probe infrastructure evidence."""

    raw = metadata.get(CLEAN_INFRASTRUCTURE_PROVENANCE_KEY)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "language",
        "command_sha256",
        "flags",
    }:
        return frozenset()
    if (
        raw.get("schema_version") != CLEAN_INFRASTRUCTURE_PROVENANCE_SCHEMA
        or raw.get("language") != language
        or raw.get("command_sha256")
        != hashlib.sha256(command.encode("utf-8")).hexdigest()
    ):
        return frozenset()
    flags = raw.get("flags")
    if (
        not isinstance(flags, dict)
        or set(flags) != set(PROVENANCE_INFRASTRUCTURE_FLAGS)
        or any(type(value) is not bool for value in flags.values())
    ):
        return frozenset()
    return frozenset(name for name, value in flags.items() if value)


def _clean_infrastructure_provenance(
    *,
    language: str,
    command: str,
    values: dict[str, float],
) -> dict[str, object]:
    """Build bounded evidence from one workflow-owned pristine probe."""

    return {
        "schema_version": CLEAN_INFRASTRUCTURE_PROVENANCE_SCHEMA,
        "language": language,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "flags": {
            name: values.get(name) == 1.0
            for name in PROVENANCE_INFRASTRUCTURE_FLAGS
        },
    }


def _diagnostic_output(
    combined: str,
    candidate: TestOutcome,
    comparison: BaselineComparison | None,
) -> str:
    lines = [
        "[crucible dev-verify outcome]",
        f"parser={candidate.parser_kind}",
        f"structurally_valid={str(candidate.structurally_valid).lower()}",
        f"structural_error={candidate.structural_error}",
    ]
    if comparison is not None:
        lines.extend((
            "[crucible clean-base comparison]",
            f"mode={comparison.mode}",
            f"reason={comparison.reason}",
            f"cache_hit={str(comparison.cache_hit).lower()}",
            f"cache_key={comparison.cache_key[:16]}",
            f"preexisting_failures={comparison.preexisting_failures}",
            f"new_failures={comparison.new_failures}",
            f"worsened_failures={comparison.worsened_failures}",
        ))
    note = "\n".join(lines)
    available = max(0, _RAW_OUTPUT_LIMIT - len(note) - 1)
    return combined[-available:] + "\n" + note


def _parse_counts(language: str, text: str) -> dict[str, float]:
    """Parse passed/failed/skipped counts from test output where the
    framework summary is stable (python, rust). Other languages return
    an empty dict — classification falls back to returncode."""
    out: dict[str, float] = {}
    if language == "python":
        last = ""
        for line in text.splitlines():
            if "passed" in line or "failed" in line or "skipped" in line:
                last = line
        if last:
            if m := _PYTEST_PASSED_RE.search(last):
                out["tests_passed"] = float(m.group(1))
            if m := _PYTEST_FAILED_RE.search(last):
                out["tests_failed"] = float(m.group(1))
            if m := _PYTEST_SKIPPED_RE.search(last):
                out["tests_skipped"] = float(m.group(1))
    elif language == "rust":
        # Sum across multiple "test result:" lines (cargo emits one per
        # binary/crate).
        passed = failed = skipped = 0
        seen = False
        for line in text.splitlines():
            if "test result:" not in line:
                continue
            seen = True
            if m := _CARGO_PASSED_RE.search(line):
                passed += int(m.group(1))
            if m := _CARGO_FAILED_RE.search(line):
                failed += int(m.group(1))
            if m := _CARGO_IGNORED_RE.search(line):
                skipped += int(m.group(1))
        if seen:
            out["tests_passed"] = float(passed)
            out["tests_failed"] = float(failed)
            out["tests_skipped"] = float(skipped)
    return out


def _python_exact_observations(
    container_id: str,
    parameters: dict[str, Any],
    *,
    deadline: _RunnerDeadline | None = None,
) -> list[dict[str, Any]]:
    from crucible.common.runners.observations import (
        canonical_attack_test_ids,
        exact_pytest_command,
        pytest_observer_failure,
        pytest_observer_status,
    )

    attack_ids = parameters.get("breach_attack_ids")
    test_file = parameters.get("breach_test_file")
    if not isinstance(test_file, str) or not test_file:
        raise ValueError("breach_observations requires breach_test_file")
    identities = canonical_attack_test_ids(attack_ids, "python")
    observations: list[dict[str, Any]] = []
    for _attack_id, test_id in identities:
        node_id = f"{test_file}::{test_id}"
        argv = exact_pytest_command(node_id, python_prefix=("python",))
        command = "cd /app && " + " ".join(shlex.quote(arg) for arg in argv)
        rc, probe_stdout, stderr = _run_docker_exec(
            container_id,
            command,
            deadline=deadline,
        )
        status = pytest_observer_status(rc)
        if status is None:
            raise ValueError(
                "exact pytest observer did not emit a reserved outcome: "
                f"node={node_id!r}, returncode={rc}, stderr={stderr[-1000:]!r}"
            )
        failure = pytest_observer_failure(
            probe_stdout,
            expected_node_id=node_id,
        )
        if failure is not None and status not in {"failed", "runner_error"}:
            raise ValueError(
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


def _rust_exact_observations(
    container_id: str,
    parameters: dict[str, Any],
    *,
    deadline: _RunnerDeadline | None = None,
) -> list[dict[str, str]]:
    from crucible.common.runners.lang.rust import _parse_libtest_observations
    from crucible.common.runners.observations import canonical_attack_test_ids

    attack_ids = parameters.get("breach_attack_ids")
    identities = canonical_attack_test_ids(attack_ids, "rust")
    crate = parameters.get("crate")
    scope = f" -p {shlex.quote(crate)}" if isinstance(crate, str) and crate else ""
    raw_features = parameters.get("cargo_features") or []
    features = (
        [str(feature) for feature in raw_features]
        if isinstance(raw_features, list) else []
    )
    feature_args = (
        " --features " + shlex.quote(",".join(features)) if features else ""
    )
    prefix = f"cargo test{scope}{feature_args}"
    rc, listed_stdout, listed_stderr = _run_docker_exec(
        container_id,
        f"cd /app && {prefix} -- --list",
        deadline=deadline,
    )
    if rc != 0:
        raise ValueError(
            "cargo test discovery failed: "
            + (listed_stderr or listed_stdout)[-1000:]
        )
    discovered = {
        line[:-6]
        for line in listed_stdout.splitlines()
        if line.endswith(": test")
    }
    observations: list[dict[str, str]] = []
    for attack_id, canonical_id in identities:
        matches = sorted(
            test_id for test_id in discovered
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
        test_rc, test_stdout, test_stderr = _run_docker_exec(
            container_id,
            f"cd /app && {prefix} {shlex.quote(test_id)} -- "
            "--exact --test-threads=1",
            deadline=deadline,
        )
        exact_records = [
            record
            for record in _parse_libtest_observations(test_stdout + test_stderr)
            if record["id"] == test_id
        ]
        if len(exact_records) != 1:
            status = "runner_error"
            line = (
                f"{test_id} (expected one exact libtest result, "
                f"observed={len(exact_records)}, rc={test_rc})"
            )
        else:
            record = exact_records[0]
            observed_status = record["status"]
            consistent = (
                (observed_status == "passed" and test_rc == 0)
                or (observed_status == "failed" and test_rc == 101)
                or (observed_status == "not_observed" and test_rc == 0)
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


def _go_exact_observations(
    container_id: str,
    parameters: dict[str, Any],
    *,
    deadline: _RunnerDeadline | None = None,
) -> list[dict[str, str]]:
    """Run each canonical Go attack in its one containing package."""

    from crucible.common.runners.observations import canonical_attack_test_ids

    test_file = parameters.get("breach_test_file")
    if not isinstance(test_file, str) or not test_file:
        raise ValueError("breach_observations requires breach_test_file")
    path = PurePosixPath(test_file)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("breach_test_file must be relative to /app")
    parent = path.parent.as_posix()
    package = "." if parent in {"", "."} else f"./{parent}"
    list_command = (
        "cd /app && go list -f '{{.ImportPath}}' " + shlex.quote(package)
    )
    list_rc, list_stdout, list_stderr = _run_docker_exec(
        container_id,
        list_command,
        deadline=deadline,
    )
    import_paths = [line.strip() for line in list_stdout.splitlines() if line.strip()]
    if list_rc != 0 or len(import_paths) != 1:
        raise ValueError(
            "cannot resolve exact Go breach package: "
            f"package={package!r}, rc={list_rc}, "
            f"stdout={list_stdout[-500:]!r}, stderr={list_stderr[-500:]!r}"
        )
    expected_package = import_paths[0]
    identities = canonical_attack_test_ids(
        parameters.get("breach_attack_ids"),
        "go",
    )

    observations: list[dict[str, str]] = []
    for _attack_id, test_id in identities:
        command = (
            "cd /app && go test -json "
            f"{shlex.quote(package)} -run {shlex.quote('^' + test_id + '$')} "
            "-count=1"
        )
        rc, stdout, stderr = _run_docker_exec(
            container_id,
            command,
            deadline=deadline,
        )
        terminal: list[tuple[str, str, str]] = []
        for line in stdout.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            action = record.get("Action")
            observed_id = record.get("Test")
            observed_package = record.get("Package")
            if (
                action in {"pass", "fail", "skip"}
                and observed_id == test_id
                and observed_package == expected_package
            ):
                terminal.append((action, observed_package, line))
        if len(terminal) != 1:
            status = "runner_error"
            line = (
                f"{test_id} (expected one exact go test terminal event, "
                f"observed={len(terminal)}, rc={rc}, stderr={stderr[-500:]!r})"
            )
        else:
            action, _package, line = terminal[0]
            observed_status = {
                "pass": "passed",
                "fail": "failed",
                "skip": "not_observed",
            }[action]
            consistent = (
                (observed_status == "passed" and rc == 0)
                or (observed_status == "failed" and rc == 1)
                or (observed_status == "not_observed" and rc == 0)
            )
            status = observed_status if consistent else "runner_error"
        observations.append({
            "language": "go",
            "id": test_id,
            "status": status,
            "line": line,
        })
    return observations


def _last_json_object(text: str) -> dict[str, Any] | None:
    """Return the last complete JSON object embedded in runner output."""

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            trailing = text[index + end :].strip()
            if not trailing:
                return value
            candidates.append((end, value))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _nearest_javascript_package_dir(
    container_id: str,
    test_file: str,
    *,
    deadline: _RunnerDeadline | None = None,
) -> str:
    """Resolve the nearest package.json without trusting project scripts."""

    absolute = (PurePosixPath("/app") / PurePosixPath(test_file)).as_posix()
    command = (
        f"candidate={shlex.quote(absolute)}; "
        'directory=${candidate%/*}; '
        'while [ "$directory" != "/" ]; do '
        'if [ -f "$directory/package.json" ]; then '
        'printf "%s\\n" "$directory"; exit 0; fi; '
        '[ "$directory" = "/app" ] && break; '
        'directory=${directory%/*}; done; '
        'printf "/app\\n"'
    )
    rc, stdout, stderr = _run_docker_exec(
        container_id,
        command,
        timeout=30,
        deadline=deadline,
    )
    directories = [line.strip() for line in stdout.splitlines() if line.strip()]
    if (
        rc != 0
        or len(directories) != 1
        or not (
            directories[0] == "/app"
            or directories[0].startswith("/app/")
        )
    ):
        raise ValueError(
            "cannot resolve nearest JavaScript package for exact probe: "
            f"file={test_file!r}, rc={rc}, stderr={stderr[-500:]!r}"
        )
    return directories[0]


def _relative_javascript_test_file(test_file: str, workdir: str) -> str:
    absolute = PurePosixPath("/app") / PurePosixPath(test_file)
    try:
        relative = absolute.relative_to(PurePosixPath(workdir))
    except ValueError as exc:
        raise ValueError(
            "exact-probe profile does not own the generated test file: "
            f"workdir={workdir!r}, file={test_file!r}"
        ) from exc
    if not relative.parts:
        raise ValueError("exact-probe test path cannot equal its package directory")
    return relative.as_posix()


def _javascript_package_prefix(configured_command: str) -> tuple[str, ...]:
    """Select a local-only package binary launcher from the sealed suite."""

    if re.search(r"\bpnpm\b", configured_command):
        return ("pnpm", "exec")
    if re.search(r"\byarn\b", configured_command):
        return ("yarn", "exec")
    if re.search(r"\bbun\b", configured_command):
        return ("bunx",)
    return ("npx", "--no-install")


def _javascript_probe_context(
    container_id: str,
    test_file: str,
    *,
    language: str,
    parser_kind: str,
    metadata: dict[str, Any],
    deadline: _RunnerDeadline | None = None,
) -> tuple[str, tuple[str, ...], str, str]:
    """Resolve one runner-owned package context for an exact JS/TS file."""

    if parser_kind not in {"jest", "vitest", "mocha", "node-tap", "ava-tap"}:
        raise ValueError(
            "JavaScript/TypeScript exact verification requires a strict "
            "Jest, Vitest, Mocha, node:test, or AVA TAP adapter; "
            f"parser={parser_kind!r}"
        )
    configured_command = str(metadata.get("dev_verify_command") or "")
    task_id = str(metadata.get("task_id") or "")
    from crucible.products.deepswe.dev_verify_overrides import (
        resolve_javascript_probe_profile,
    )

    profile = resolve_javascript_probe_profile(task_id=task_id)
    if profile is not None and profile.parser_kind != parser_kind:
        raise ValueError(
            "sealed JavaScript exact-probe profile disagrees with suite parser: "
            f"task={task_id!r}, profile={profile.parser_kind!r}, "
            f"parser={parser_kind!r}"
        )
    if profile is not None:
        workdir = profile.workdir
        executable = profile.executable
        setup_command = profile.setup_command
    else:
        workdir = _nearest_javascript_package_dir(
            container_id,
            test_file,
            deadline=deadline,
        )
        setup_command = ""
        if parser_kind == "node-tap":
            if language == "typescript":
                raise ValueError(
                    "TypeScript node:test exact verification requires a "
                    f"sealed task probe profile; task={task_id!r}"
                )
            executable = ("node",)
        else:
            runner_name = {
                "jest": "jest",
                "vitest": "vitest",
                "mocha": "mocha",
                "ava-tap": "ava",
            }[parser_kind]
            executable = (
                *_javascript_package_prefix(configured_command),
                runner_name,
            )
    relative_test_file = _relative_javascript_test_file(test_file, workdir)
    return workdir, executable, setup_command, relative_test_file


def _javascript_exact_observations(
    container_id: str,
    parameters: dict[str, Any],
    *,
    parser_kind: str,
    deadline: _RunnerDeadline | None = None,
) -> list[dict[str, str]]:
    """Run one package-aware exact probe per canonical JS/TS title."""

    from crucible.common.runners.observations import canonical_attack_test_ids

    language = str(
        (parameters.get("deepswe_metadata") or {}).get("language") or ""
    ).strip().lower()
    test_file = parameters.get("breach_test_file")
    if not isinstance(test_file, str) or not test_file:
        raise ValueError("breach_observations requires breach_test_file")
    path = PurePosixPath(test_file)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("breach_test_file must be relative to /app")
    identities = canonical_attack_test_ids(
        parameters.get("breach_attack_ids"), language
    )
    metadata = parameters.get("deepswe_metadata") or {}
    workdir, executable, setup_command, relative_test_file = (
        _javascript_probe_context(
            container_id,
            test_file,
            language=language,
            parser_kind=parser_kind,
            metadata=metadata,
            deadline=deadline,
        )
    )

    observations: list[dict[str, str]] = []
    for _attack_id, test_id in identities:
        if parser_kind == "jest":
            argv = (
                *executable, "--runInBand", "--json",
                "--runTestsByPath", relative_test_file, "--testNamePattern",
                re.escape(test_id),
            )
        elif parser_kind == "vitest":
            argv = (
                *executable, "run", relative_test_file, "--reporter=json",
                "--testNamePattern", re.escape(test_id),
            )
        elif parser_kind == "mocha":
            argv = (
                *executable, relative_test_file,
                "--grep", re.escape(test_id), "--reporter", "json",
            )
        elif parser_kind == "node-tap":
            argv = (
                *executable,
                "--test",
                "--test-reporter=tap",
                "--test-name-pattern",
                re.escape(test_id),
                relative_test_file,
            )
        else:
            argv = (
                *executable,
                "--tap",
                "--match",
                test_id,
                relative_test_file,
            )
        invocation = " ".join(shlex.quote(arg) for arg in argv)
        command = f"cd {shlex.quote(workdir)} && "
        if setup_command:
            command += setup_command + " && "
        command += invocation
        rc, stdout, stderr = _run_docker_exec(
            container_id,
            command,
            deadline=deadline,
        )
        payload = _last_json_object(stdout)
        matches: list[tuple[str, str]] = []
        if parser_kind in {"jest", "vitest"} and isinstance(payload, dict):
            raw_results = payload.get("testResults")
            if isinstance(raw_results, list):
                for suite in raw_results:
                    assertions = (
                        suite.get("assertionResults")
                        if isinstance(suite, dict) else None
                    )
                    if not isinstance(assertions, list):
                        continue
                    for assertion in assertions:
                        if (
                            isinstance(assertion, dict)
                            and assertion.get("title") == test_id
                            and isinstance(assertion.get("status"), str)
                        ):
                            matches.append((assertion["status"], json.dumps(
                                assertion, sort_keys=True, ensure_ascii=True
                            )))
        elif parser_kind == "mocha" and isinstance(payload, dict):
            for key, status in (
                ("passes", "passed"),
                ("failures", "failed"),
                ("pending", "pending"),
            ):
                records = payload.get(key)
                if not isinstance(records, list):
                    continue
                for record in records:
                    if (
                        isinstance(record, dict)
                        and record.get("title") == test_id
                    ):
                        matches.append((status, json.dumps(
                            record, sort_keys=True, ensure_ascii=True
                        )))
        elif parser_kind in {"node-tap", "ava-tap"}:
            records = strict_tap_test_records(
                stdout,
                stderr,
                node_style=parser_kind == "node-tap",
            )
            matches.extend(
                (record.status, record.line)
                for record in records
                if record.title == test_id
            )
        if len(matches) != 1:
            status = "runner_error"
            line = (
                f"{test_id} (expected one exact {parser_kind} result, "
                f"observed={len(matches)}, rc={rc}, stderr={stderr[-500:]!r})"
            )
        else:
            raw_status, line = matches[0]
            observed_status = {
                "passed": "passed",
                "failed": "failed",
                "pending": "not_observed",
                "skipped": "not_observed",
                "todo": "not_observed",
                "not_observed": "not_observed",
            }.get(raw_status, "runner_error")
            consistent = (
                observed_status == "passed" and rc == 0
                or observed_status == "failed" and rc != 0
                or observed_status == "not_observed" and rc == 0
            )
            status = observed_status if consistent else "runner_error"
        observations.append({
            "language": language,
            "id": f"{test_file}::{test_id}",
            "status": status,
            "line": line,
        })
    return observations


def _declared_test_files_from_surface(
    raw_surface: object,
    language: str,
) -> tuple[str, ...]:
    slices = parse_declared_verification_surface(raw_surface)
    paths = (
        path
        for item in slices
        for path in (*item.files_in_scope, *item.creates_paths)
    )
    selected = tuple(dict.fromkeys(
        path for path in paths
        if _is_declared_native_test_path(path, language)
    ))
    if len(selected) > _MAX_DECLARED_TEST_FILES:
        raise ValueError("declared verification selected too many test files")
    return selected


def _declared_test_files(
    parameters: dict[str, Any],
    language: str,
) -> tuple[str, ...]:
    raw_surface = parameters.get(DECLARED_VERIFICATION_SURFACE_KEY)
    if raw_surface is None:
        return ()
    return _declared_test_files_from_surface(raw_surface, language)


def _accepted_predecessor_outcomes(
    parameters: dict[str, Any],
    *,
    language: str,
    parser_kind: str,
    cumulative_test_files: tuple[str, ...],
) -> dict[str, tuple[str, TestOutcome]]:
    raw_reference = parameters.get(ACCEPTED_VERIFICATION_REFERENCE_KEY)
    if raw_reference is None:
        return {}
    reference = parse_accepted_verification_reference(raw_reference)
    if reference.language != language:
        raise ValueError(
            "accepted verification language does not match the active runner"
        )
    if reference.parser_kind != parser_kind:
        raise ValueError(
            "accepted verification parser does not match the active runner"
        )
    predecessor_test_files = _declared_test_files_from_surface(
        reference.predecessor_surface,
        language,
    )
    by_path: dict[str, tuple[str, TestOutcome]] = {}
    for item in reference.outcomes:
        outcome = TestOutcome.from_payload(item.outcome)
        if not outcome.structurally_valid:
            raise ValueError(
                "accepted verification outcome must be structurally valid"
            )
        if outcome.parser_kind != parser_kind:
            raise ValueError(
                "accepted verification outcome parser does not match its "
                "reference"
            )
        by_path[item.path] = (item.command_sha256, outcome)
    if set(by_path) != set(predecessor_test_files):
        raise ValueError(
            "accepted verification reference does not bind every predecessor "
            "declared test file"
        )
    if not set(predecessor_test_files).issubset(cumulative_test_files):
        raise ValueError(
            "accepted verification predecessor surface is not contained in "
            "the cumulative declared surface"
        )
    return by_path


def _nearest_rust_package_dir(
    container_id: str,
    test_file: str,
    *,
    deadline: _RunnerDeadline | None = None,
) -> str:
    absolute = (PurePosixPath("/app") / PurePosixPath(test_file)).as_posix()
    command = (
        f"candidate={shlex.quote(absolute)}; "
        'directory=${candidate%/*}; '
        'while [ "$directory" != "/" ]; do '
        'if [ -f "$directory/Cargo.toml" ]; then '
        'printf "%s\\n" "$directory"; exit 0; fi; '
        '[ "$directory" = "/app" ] && break; '
        'directory=${directory%/*}; done; exit 1'
    )
    rc, stdout, stderr = _run_docker_exec(
        container_id,
        command,
        timeout=30,
        deadline=deadline,
    )
    directories = [line.strip() for line in stdout.splitlines() if line.strip()]
    if (
        rc != 0
        or len(directories) != 1
        or not (
            directories[0] == "/app"
            or directories[0].startswith("/app/")
        )
    ):
        raise ValueError(
            "cannot resolve Cargo package for declared integration test: "
            f"file={test_file!r}, rc={rc}, stderr={stderr[-500:]!r}"
        )
    return directories[0]


def _declared_test_probe_command(
    container_id: str,
    test_file: str,
    *,
    language: str,
    parser_kind: str,
    metadata: dict[str, Any],
    parameters: dict[str, Any],
    deadline: _RunnerDeadline | None = None,
) -> str:
    if language == "python":
        argv = (
            "pytest",
            test_file,
            PYTEST_DEV_VERIFY_MAXFAIL_ARG,
            "--tb=short",
            "-q",
        )
        return "cd /app && " + " ".join(shlex.quote(arg) for arg in argv)
    if language in {"javascript", "typescript"}:
        workdir, executable, setup_command, relative = (
            _javascript_probe_context(
                container_id,
                test_file,
                language=language,
                parser_kind=parser_kind,
                metadata=metadata,
                deadline=deadline,
            )
        )
        if parser_kind == "jest":
            argv = (*executable, "--runInBand", "--runTestsByPath", relative)
        elif parser_kind == "vitest":
            argv = (*executable, "run", relative)
        elif parser_kind == "mocha":
            argv = (*executable, relative, "--reporter", "spec")
        elif parser_kind == "node-tap":
            argv = (*executable, "--test", "--test-reporter=tap", relative)
        else:
            argv = (*executable, "--tap", relative)
        command = f"cd {shlex.quote(workdir)} && "
        if setup_command:
            command += setup_command + " && "
        return command + " ".join(shlex.quote(arg) for arg in argv)
    if language == "rust":
        workdir = _nearest_rust_package_dir(
            container_id,
            test_file,
            deadline=deadline,
        )
        relative = _relative_javascript_test_file(test_file, workdir)
        relative_path = PurePosixPath(relative)
        if (
            len(relative_path.parts) != 2
            or relative_path.parts[0] != "tests"
            or relative_path.suffix != ".rs"
        ):
            raise ValueError(
                "declared Rust test must be a direct Cargo integration target: "
                f"file={test_file!r}"
            )
        raw_features = parameters.get("cargo_features") or []
        if not isinstance(raw_features, (list, tuple)) or not all(
            isinstance(feature, str) and feature for feature in raw_features
        ):
            raise ValueError("cargo_features must be a string list")
        argv: tuple[str, ...] = (
            "cargo", "test", "--test", relative_path.stem,
        )
        if raw_features:
            argv += ("--features", ",".join(raw_features))
        argv += ("--", "--test-threads=1")
        return (
            f"cd {shlex.quote(workdir)} && "
            + " ".join(shlex.quote(arg) for arg in argv)
        )
    raise ValueError(
        f"declared exact-test verification is unsupported for {language!r}"
    )


def _run_declared_test_probes(
    container_id: str,
    parameters: dict[str, Any],
    *,
    language: str,
    parser_kind: str,
    baseline: dict[str, str] | None = None,
    deadline: _RunnerDeadline | None = None,
) -> tuple[int, dict[str, float], str, float, dict[str, object]]:
    """Run each declared native test file independently and non-vacuously."""

    test_files = _declared_test_files(parameters, language)
    accepted_outcomes = _accepted_predecessor_outcomes(
        parameters,
        language=language,
        parser_kind=parser_kind,
        cumulative_test_files=test_files,
    )
    reference_entries: list[tuple[str, str, dict[str, object]]] = []
    if not test_files:
        reference = accepted_verification_reference_payload(
            predecessor_surface=parameters[DECLARED_VERIFICATION_SURFACE_KEY],
            language=language,
            parser_kind=parser_kind,
            outcomes=(),
        )
        return 0, {
            "dev_verify_declared_test_files_required": 0.0,
            "dev_verify_declared_test_files_executed": 0.0,
            "dev_verify_declared_test_files_passed": 0.0,
            "dev_verify_declared_test_files_failed": 0.0,
            "dev_verify_declared_test_files_structural_failures": 0.0,
            "dev_verify_declared_test_baseline_files_compared": 0.0,
            "dev_verify_declared_test_baseline_files_absent": 0.0,
            "dev_verify_declared_test_baseline_cache_hits": 0.0,
            "dev_verify_declared_test_preexisting_failures": 0.0,
            "dev_verify_declared_test_accepted_files_compared": 0.0,
        }, "", 0.0, reference

    metadata = parameters.get("deepswe_metadata") or {}
    required = len(test_files)
    executed = passed = failed = structural = 0
    baseline_compared = baseline_absent = baseline_cache_hits = 0
    preexisting_failures = 0
    accepted_compared = 0
    effective_rc = 0
    wall_seconds = 0.0
    diagnostics = ["[crucible declared-test verification]"]
    if baseline is not None:
        probe_error = _probe_clean_baseline_container(
            baseline,
            deadline=deadline,
        )
        if probe_error:
            raise ValueError(probe_error)
    for test_file in test_files:
        try:
            _validate_declared_container_path(
                container_id,
                test_file,
                deadline=deadline,
            )
            command = _declared_test_probe_command(
                container_id,
                test_file,
                language=language,
                parser_kind=parser_kind,
                metadata=metadata,
                parameters=parameters,
                deadline=deadline,
            )
            start = time.monotonic()
            rc, probe_stdout, probe_stderr = _run_docker_exec(
                container_id,
                command,
                deadline=deadline,
            )
            wall_seconds += time.monotonic() - start
            executed += 1
            candidate_outcome = parse_test_outcome(
                language=language,
                command=command,
                actual_returncode=rc,
                stdout=probe_stdout,
                stderr=probe_stderr,
                parser_hint=parser_kind,
            )
            command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
            if candidate_outcome.structurally_valid:
                reference_entries.append((
                    test_file,
                    command_sha256,
                    candidate_outcome.to_payload(),
                ))
            comparison: BaselineComparison | None = None
            baseline_state = "unavailable"
            accepted = accepted_outcomes.get(test_file)
            if accepted is not None:
                accepted_command_sha256, accepted_outcome = accepted
                if accepted_command_sha256 != command_sha256:
                    raise ValueError(
                        "accepted verification focused command changed for "
                        f"path={test_file!r}"
                    )
                comparison = compare_outcomes(
                    candidate_outcome,
                    accepted_outcome,
                    expected_extra_tests=0,
                )
                baseline_state = "accepted_predecessor"
                accepted_compared += 1
                preexisting_failures += comparison.preexisting_failures
            elif baseline is not None and candidate_outcome.structurally_valid:
                baseline_state = _declared_container_path_state(
                    baseline["baseline_container_id"],
                    test_file,
                    deadline=deadline,
                )
                if baseline_state == "safe":
                    identity = baseline_identity(
                        image_digest=baseline["image_digest"],
                        base_commit=baseline["base_commit"],
                        command=command,
                        language=language,
                        parser_hint=parser_kind,
                    )

                    def _produce_focused_baseline(
                        command: str = command,
                    ) -> TestOutcome | None:
                        baseline_rc, baseline_stdout, baseline_stderr = (
                            _run_docker_exec(
                                baseline["baseline_container_id"],
                                command,
                                deadline=deadline,
                            )
                        )
                        return parse_test_outcome(
                            language=language,
                            command=command,
                            actual_returncode=baseline_rc,
                            stdout=baseline_stdout,
                            stderr=baseline_stderr,
                            parser_hint=parser_kind,
                        )

                    (
                        baseline_outcome,
                        cache_hit,
                        cache_key,
                        cache_error,
                    ) = cached_baseline_outcome(
                        identity,
                        _produce_focused_baseline,
                    )
                    baseline_compared += 1
                    baseline_cache_hits += int(cache_hit)
                    if (
                        baseline_outcome is None
                        or not baseline_outcome.structurally_valid
                    ):
                        raise ValueError(
                            cache_error
                            or "focused clean-base outcome is invalid"
                        )
                    comparison = compare_outcomes(
                        candidate_outcome,
                        baseline_outcome,
                        expected_extra_tests=0,
                        cache_hit=cache_hit,
                        cache_key=cache_key,
                    )
                    preexisting_failures += comparison.preexisting_failures
                else:
                    baseline_absent += 1

            probe_rc = (
                comparison.effective_returncode
                if comparison is not None else candidate_outcome.returncode
            )
            if probe_rc == 0 and candidate_outcome.structurally_valid:
                passed += 1
                status = (
                    comparison.mode
                    if comparison is not None else "passed"
                )
            else:
                failed += 1
                if (
                    not candidate_outcome.structurally_valid
                    or comparison is not None
                    and comparison.mode in {"unavailable", "incomparable"}
                ):
                    structural += 1
                    effective_rc = max(effective_rc, 2)
                    status = "structurally_invalid"
                else:
                    effective_rc = max(effective_rc, probe_rc, 1)
                    status = "failed"
            reason = (
                comparison.reason
                if comparison is not None else candidate_outcome.structural_error
            )
            diagnostics.append(
                f"path={test_file} status={status} "
                f"parser={candidate_outcome.parser_kind} "
                f"tests_run={candidate_outcome.tests_run} "
                f"baseline={baseline_state} "
                f"comparison={comparison.mode if comparison else 'none'} "
                f"reason={reason[:500]}"
            )
            if probe_rc != 0:
                output_tail = (probe_stdout + probe_stderr)[-1_500:]
                if output_tail:
                    diagnostics.append(
                        f"path={test_file} output_tail:\n{output_tail}"
                    )
        except _RunnerDeadlineExceeded:
            raise
        except (OSError, TypeError, ValueError) as exc:
            failed += 1
            structural += 1
            effective_rc = max(effective_rc, 2)
            diagnostics.append(
                f"path={test_file} status=runner_error "
                f"reason={str(exc)[:500]}"
            )
    values = {
        "dev_verify_declared_test_files_required": float(required),
        "dev_verify_declared_test_files_executed": float(executed),
        "dev_verify_declared_test_files_passed": float(passed),
        "dev_verify_declared_test_files_failed": float(failed),
        "dev_verify_declared_test_files_structural_failures": float(structural),
        "dev_verify_declared_test_baseline_files_compared": float(
            baseline_compared
        ),
        "dev_verify_declared_test_baseline_files_absent": float(
            baseline_absent
        ),
        "dev_verify_declared_test_baseline_cache_hits": float(
            baseline_cache_hits
        ),
        "dev_verify_declared_test_preexisting_failures": float(
            preexisting_failures
        ),
        "dev_verify_declared_test_accepted_files_compared": float(
            accepted_compared
        ),
    }
    reference = accepted_verification_reference_payload(
        predecessor_surface=parameters[DECLARED_VERIFICATION_SURFACE_KEY],
        language=language,
        parser_kind=parser_kind,
        outcomes=reference_entries,
    )
    return (
        effective_rc,
        values,
        "\n".join(diagnostics),
        wall_seconds,
        reference,
    )


def _emit_kpi(
    stdout: Any,
    *,
    values: dict[str, float],
    wall_seconds: float,
    raw_output: str = "",
    test_observations: list[dict[str, Any]] | None = None,
    declared_verification_reference: dict[str, object] | None = None,
    repair_failure_set: dict[str, object] | None = None,
    clean_infrastructure_provenance: dict[str, object] | None = None,
    breach_observation_failure_payload: dict[str, str] | None = None,
) -> None:
    payload = {
        "values": values,
        "wall_seconds": wall_seconds,
        "raw_output": raw_output,
    }
    if test_observations is not None:
        payload["test_observations"] = test_observations
    if declared_verification_reference is not None:
        payload[DECLARED_VERIFICATION_REFERENCE_OUTPUT_KEY] = (
            declared_verification_reference
        )
    if repair_failure_set is not None:
        payload[REPAIR_FAILURE_SET_OUTPUT_KEY] = repair_failure_set
    if clean_infrastructure_provenance is not None:
        payload[CLEAN_INFRASTRUCTURE_PROVENANCE_KEY] = (
            clean_infrastructure_provenance
        )
    if breach_observation_failure_payload is not None:
        payload[BREACH_OBSERVATION_FAILURE_KEY] = (
            breach_observation_failure_payload
        )
    json.dump(payload, stdout)
    stdout.write("\n")


def _emit_runner_unavailable(
    stdout: Any,
    *,
    diagnostic: str,
    specific_flag: str,
    returncode: int = 2,
    runner_started: float | None = None,
) -> None:
    """Emit machine-readable setup unavailability before returning non-zero."""

    _emit_kpi(
        stdout,
        values={
            "returncode": float(returncode),
            "dev_verify_infrastructure_unavailable": 1.0,
            specific_flag: 1.0,
        },
        wall_seconds=(
            0.0
            if runner_started is None
            else time.monotonic() - runner_started
        ),
        raw_output=diagnostic,
        test_observations=[],
    )


def _run_dev_verify(parameters: dict[str, Any], stdout: Any) -> int:
    runner_started = time.monotonic()
    raw_metadata = parameters.get("deepswe_metadata")
    metadata = {} if raw_metadata is None else raw_metadata
    if not isinstance(metadata, dict):
        diagnostic = (
            "deepswe_dev_verify: ``deepswe_metadata`` must be a dict, "
            f"got {type(metadata).__name__}"
        )
        logger.error("%s", diagnostic)
        _emit_runner_unavailable(
            stdout,
            diagnostic=diagnostic,
            specific_flag="dev_verify_metadata_unavailable",
            runner_started=runner_started,
        )
        return 2

    # The dispatch language is threaded through deepswe_metadata by the
    # worker (``_build_run_architect_kwargs``) because the DeepSWE
    # starter spec carries no LanguagesConstraint, so _build_verify's
    # languages-injection path never sets a top-level ``language``. The
    # top-level read is a fallback for direct/unit invocations.
    raw_language = metadata.get("language") or parameters.get("language") or ""
    if not isinstance(raw_language, str):
        diagnostic = (
            "deepswe_dev_verify: language must be a string, "
            f"got {type(raw_language).__name__}"
        )
        logger.error("%s", diagnostic)
        _emit_runner_unavailable(
            stdout,
            diagnostic=diagnostic,
            specific_flag="dev_verify_language_unavailable",
            runner_started=runner_started,
        )
        return 2
    language = raw_language.strip().lower()
    try:
        parameters = for_language(parameters, language)
    except ValueError as exc:
        _emit_runner_unavailable(
            stdout,
            diagnostic=str(exc),
            specific_flag="dev_verify_language_unavailable",
            runner_started=runner_started,
        )
        return 2
    raw_metadata = parameters.get("deepswe_metadata")
    metadata = {} if raw_metadata is None else raw_metadata
    if not isinstance(metadata, dict):
        diagnostic = (
            "deepswe_dev_verify: language-specific ``deepswe_metadata`` "
            f"must be a dict, got {type(metadata).__name__}"
        )
        _emit_runner_unavailable(
            stdout,
            diagnostic=diagnostic,
            specific_flag="dev_verify_metadata_unavailable",
            runner_started=runner_started,
        )
        return 2
    try:
        deadline = _RunnerDeadline.from_parameters(parameters)
    except ValueError as exc:
        _emit_runner_unavailable(
            stdout,
            diagnostic=str(exc),
            specific_flag="dev_verify_runner_configuration_unavailable",
            runner_started=runner_started,
        )
        return 2
    # A crucible-owned per-project override (threaded by the worker from
    # dev_verify_overrides) wins over the per-language default — it's the
    # escape hatch for projects whose native orchestrator the default
    # bypasses, and it also rescues languages absent from _LANG_TO_COMMAND.
    # ``language`` is still read above for output-count parsing.
    raw_override = metadata.get("dev_verify_command")
    if raw_override is not None and not isinstance(raw_override, str):
        diagnostic = (
            "deepswe_dev_verify: dev_verify_command must be a string, "
            f"got {type(raw_override).__name__}"
        )
        _emit_runner_unavailable(
            stdout,
            diagnostic=diagnostic,
            specific_flag="dev_verify_runner_configuration_unavailable",
            runner_started=runner_started,
        )
        return 2
    override = (raw_override or "").strip()
    breach_observations = bool(parameters.get("breach_observations"))
    if language == "rust" and not override:
        try:
            command = _rust_breach_command(parameters)
        except ValueError as exc:
            _emit_runner_unavailable(
                stdout,
                diagnostic=str(exc),
                specific_flag="dev_verify_runner_configuration_unavailable",
                runner_started=runner_started,
            )
            return 2
    else:
        command = override or _LANG_TO_COMMAND.get(language)
    if command is None:
        diagnostic = (
            "deepswe_dev_verify: no project-native test command for "
            f"language {language!r} and no ``dev_verify_command`` override in "
            "deepswe_metadata. Supported languages: "
            f"{sorted(_LANG_TO_COMMAND)}. Register a "
            "per-project override in "
            "crucible.products.deepswe.dev_verify_overrides, or add an "
            "entry to _LANG_TO_COMMAND (calibrate against the pilot)."
        )
        logger.error("%s", diagnostic)
        _emit_runner_unavailable(
            stdout,
            diagnostic=diagnostic,
            specific_flag="dev_verify_language_unavailable",
            runner_started=runner_started,
        )
        return 2

    container_id = metadata.get("container_id")
    if (
        not isinstance(container_id, str)
        or _CONTAINER_REFERENCE_RE.fullmatch(container_id) is None
    ):
        diagnostic = (
            "deepswe_dev_verify: ``deepswe_metadata`` missing required "
            "valid ``container_id`` (got keys: "
            f"{sorted(str(key) for key in metadata)})"
        )
        logger.error("%s", diagnostic)
        _emit_runner_unavailable(
            stdout,
            diagnostic=diagnostic,
            specific_flag="dev_verify_container_unavailable",
            runner_started=runner_started,
        )
        return 2

    try:
        rc, exec_stdout, exec_stderr = _run_docker_exec(
            str(container_id),
            command,
            deadline=deadline,
        )
    except _RunnerDeadlineExceeded as exc:
        _emit_kpi(
            stdout,
            values={"returncode": 124.0},
            wall_seconds=time.monotonic() - runner_started,
            raw_output=str(exc),
            test_observations=[],
        )
        return 124
    if exec_stdout:
        logger.info("deepswe_dev_verify: stdout (tail):\n%s", exec_stdout[-2000:])
    if exec_stderr:
        logger.info("deepswe_dev_verify: stderr (tail):\n%s", exec_stderr[-2000:])

    if rc == CONTAINMENT_UNUSABLE_RC:
        diagnostic = (exec_stdout + exec_stderr)[-_RAW_OUTPUT_LIMIT:]
        _emit_kpi(
            stdout,
            values={
                "returncode": float(CONTAINMENT_UNUSABLE_RC),
                "dev_verify_candidate_process_returncode": float(rc),
                "dev_verify_source_unusable": 1.0,
                "dev_verify_infrastructure_unavailable": 1.0,
            },
            wall_seconds=time.monotonic() - runner_started,
            raw_output=diagnostic,
            test_observations=[],
        )
        return CONTAINMENT_UNUSABLE_RC

    combined = exec_stdout + exec_stderr
    raw_parser_hint = metadata.get("dev_verify_parser")
    parser_hint = (
        raw_parser_hint.strip()
        if isinstance(raw_parser_hint, str) and raw_parser_hint.strip()
        else None
    )
    candidate = parse_test_outcome(
        language=language,
        command=command,
        actual_returncode=rc,
        stdout=exec_stdout,
        stderr=exec_stderr,
        parser_hint=parser_hint,
    )
    clean_probe = metadata.get("dev_verify_clean_probe") is True
    proven_clean_flags = _validated_clean_infrastructure_flags(
        metadata,
        language=language,
        command=command,
    )
    values: dict[str, float] = {
        "dev_verify_candidate_process_returncode": float(rc),
        "dev_verify_candidate_returncode": float(candidate.returncode),
        **_outcome_values(candidate),
        **_candidate_infrastructure_values(
            language=language,
            process_returncode=rc,
            output=combined,
            parser_kind=candidate.parser_kind,
            clean_probe=clean_probe,
            proven_clean_flags=proven_clean_flags,
        ),
    }
    effective_rc = (
        rc if rc in {124, 127} else candidate.returncode
    )
    comparison: BaselineComparison | None = None
    baseline_attempted = False
    baseline, baseline_error = _baseline_metadata(
        metadata,
        candidate_container_id=str(container_id),
    )
    if baseline_error:
        effective_rc = 2
        comparison = BaselineComparison(
            mode="unavailable",
            effective_returncode=2,
            reason=baseline_error,
        )
    elif baseline is None:
        values["dev_verify_baseline_attempted"] = 0.0
    elif candidate.structurally_valid:
        baseline_attempted = True
        identity = baseline_identity(
            image_digest=baseline["image_digest"],
            base_commit=baseline["base_commit"],
            command=command,
            language=language,
            parser_hint=parser_hint,
        )

        def _produce_baseline() -> TestOutcome | None:
            probe_error = _probe_clean_baseline_container(
                baseline,
                deadline=deadline,
            )
            if probe_error:
                return TestOutcome(
                    parser_kind=parser_hint or language,
                    returncode=2,
                    tests_run=None,
                    summary=(),
                    failures=(),
                    failure_set_complete=False,
                    output_sha256="0" * 64,
                    structural_error=probe_error,
                )
            baseline_rc, baseline_stdout, baseline_stderr = _run_docker_exec(
                baseline["baseline_container_id"],
                command,
                deadline=deadline,
            )
            return parse_test_outcome(
                language=language,
                command=command,
                actual_returncode=baseline_rc,
                stdout=baseline_stdout,
                stderr=baseline_stderr,
                parser_hint=parser_hint,
            )

        try:
            baseline_outcome, cache_hit, cache_key, cache_error = (
                cached_baseline_outcome(identity, _produce_baseline)
            )
        except _RunnerDeadlineExceeded as exc:
            comparison = BaselineComparison(
                mode="unavailable",
                effective_returncode=124,
                reason=str(exc),
            )
            effective_rc = 124
            values.update(_comparison_values(
                comparison,
                baseline_attempted=baseline_attempted,
            ))
        else:
            if (
                baseline_outcome is None
                or not baseline_outcome.structurally_valid
            ):
                comparison = BaselineComparison(
                    mode="unavailable",
                    effective_returncode=2,
                    reason=cache_error or "clean-base outcome is invalid",
                    cache_hit=cache_hit,
                    cache_key=cache_key,
                    baseline_returncode=(
                        baseline_outcome.returncode
                        if baseline_outcome is not None else None
                    ),
                )
            else:
                comparison = compare_outcomes(
                    candidate,
                    baseline_outcome,
                    expected_extra_tests=(
                        _expected_breach_test_delta(parameters)
                    ),
                    cache_hit=cache_hit,
                    cache_key=cache_key,
                )
            effective_rc = comparison.effective_returncode
            values.update(_comparison_values(
                comparison,
                baseline_attempted=baseline_attempted,
            ))
            if comparison.mode == "preexisting_only":
                # The generic build/verify layer treats tests_failed > 0 as a
                # partial failure independently of the process return code.
                values["dev_verify_candidate_tests_failed"] = values.get(
                    "tests_failed", 0.0
                )
                values["dev_verify_candidate_tests_errored"] = values.get(
                    "tests_errored", 0.0
                )
                values["tests_failed"] = 0.0
                values["tests_errored"] = 0.0
    elif baseline is not None:
        comparison = BaselineComparison(
            mode="incomparable",
            effective_returncode=effective_rc,
            reason="candidate outcome is structurally invalid",
        )
        values.update(_comparison_values(
            comparison,
            baseline_attempted=baseline_attempted,
        ))
    if comparison is not None and "dev_verify_baseline_attempted" not in values:
        values.update(_comparison_values(
            comparison,
            baseline_attempted=baseline_attempted,
        ))
    if comparison is not None and comparison.mode == "unavailable":
        values["dev_verify_infrastructure_unavailable"] = 1.0
    declared_diagnostic = ""
    declared_reference: dict[str, object] | None = None
    if DECLARED_VERIFICATION_SURFACE_KEY in parameters:
        try:
            (
                declared_rc,
                declared_values,
                declared_diagnostic,
                _declared_wall,
                declared_reference,
            ) = _run_declared_test_probes(
                str(container_id),
                parameters,
                language=language,
                parser_kind=candidate.parser_kind,
                baseline=baseline,
                deadline=deadline,
            )
            effective_rc = max(effective_rc, declared_rc)
            values.update(declared_values)
        except _RunnerDeadlineExceeded as exc:
            effective_rc = max(effective_rc, 124)
            values.update({
                "dev_verify_infrastructure_unavailable": 1.0,
                "dev_verify_declared_probe_unavailable": 1.0,
            })
            declared_diagnostic = (
                "[crucible declared-test verification]\n"
                f"status=timeout reason={str(exc)[:500]}"
            )
        except (OSError, TypeError, ValueError) as exc:
            effective_rc = max(effective_rc, 2)
            values.update({
                "dev_verify_infrastructure_unavailable": 1.0,
                "dev_verify_declared_probe_unavailable": 1.0,
                "dev_verify_declared_test_files_required": 0.0,
                "dev_verify_declared_test_files_executed": 0.0,
                "dev_verify_declared_test_files_passed": 0.0,
                "dev_verify_declared_test_files_failed": 1.0,
                "dev_verify_declared_test_files_structural_failures": 1.0,
            })
            declared_diagnostic = (
                "[crucible declared-test verification]\n"
                f"path=<surface> status=runner_error reason={str(exc)[:500]}"
            )
    if (
        clean_probe
        and values.get("dev_verify_infrastructure_unavailable") == 1.0
    ):
        # Clean-checkout infrastructure evidence is a setup failure even when
        # differential comparison would otherwise forgive identical candidate
        # and baseline outcomes as preexisting. Never admit paid model
        # transport behind such a preflight.
        effective_rc = 2
        values["dev_verify_clean_probe_unavailable"] = 1.0
    values["returncode"] = float(effective_rc)
    # raw_output mirrors the language-runner contract (python.py:176,
    # rust.py:155): combined stdout+stderr tail, capped to the tier repair
    # diagnostic budget.
    # breach.py::_extract_pytest_output reads this top-level key to give
    # the Defender round-over-round test feedback.
    observations = None
    observation_error: str | None = None
    observation_failure_payload: dict[str, str] | None = None
    if breach_observations:
        from crucible.common.runners.observations import filter_test_observations

        try:
            if language == "python":
                raw_observations = _python_exact_observations(
                    str(container_id), parameters, deadline=deadline,
                )
            elif language == "rust":
                raw_observations = _rust_exact_observations(
                    str(container_id), parameters, deadline=deadline,
                )
            elif language == "go":
                raw_observations = _go_exact_observations(
                    str(container_id), parameters, deadline=deadline,
                )
            elif language in {"javascript", "typescript"}:
                raw_observations = _javascript_exact_observations(
                    str(container_id),
                    parameters,
                    parser_kind=candidate.parser_kind,
                    deadline=deadline,
                )
            else:
                raise ValueError(
                    "breach observations require an exact runner adapter; "
                    f"language={language!r} is unsupported"
                )
            observations = filter_test_observations(
                raw_observations,
                parameters.get("breach_attack_ids"),
                language=language,
            )
        except _RunnerDeadlineExceeded as exc:
            observations = []
            observation_error = str(exc)
            observation_failure_payload = breach_observation_failure(
                "infrastructure",
                exc,
            )
            effective_rc = max(effective_rc, 124)
            values["returncode"] = float(effective_rc)
        except OSError as exc:
            observations = []
            observation_error = str(exc)
            observation_failure_payload = breach_observation_failure(
                "infrastructure",
                f"{type(exc).__name__}: {exc}",
            )
            effective_rc = max(effective_rc, 2)
            values["returncode"] = float(effective_rc)
        except ValueError as exc:
            observations = []
            observation_error = str(exc)
            observation_failure_payload = breach_observation_failure(
                "channel",
                exc,
            )
            effective_rc = max(effective_rc, 2)
            values["returncode"] = float(effective_rc)
    raw_diagnostic = (
        observation_error
        or _diagnostic_output(combined, candidate, comparison)
    )
    if declared_diagnostic:
        raw_diagnostic = (
            raw_diagnostic + "\n" + declared_diagnostic
        )[-_RAW_OUTPUT_LIMIT:]
    try:
        deadline.remaining()
    except _RunnerDeadlineExceeded as exc:
        effective_rc = max(effective_rc, 124)
        values["returncode"] = float(effective_rc)
        raw_diagnostic = (
            raw_diagnostic + "\n[crucible runner deadline]\n" + str(exc)
        )[-_RAW_OUTPUT_LIMIT:]
    repair_failure_set = _repair_failure_set_payload(
        effective_rc=effective_rc,
        values=values,
        candidate=candidate,
        command=command,
        declared_reference=declared_reference,
    )
    _emit_kpi(
        stdout,
        values=values,
        wall_seconds=time.monotonic() - runner_started,
        raw_output=raw_diagnostic,
        test_observations=observations,
        declared_verification_reference=declared_reference,
        repair_failure_set=repair_failure_set,
        clean_infrastructure_provenance=(
            _clean_infrastructure_provenance(
                language=language,
                command=command,
                values=values,
            )
            if clean_probe
            else None
        ),
        breach_observation_failure_payload=observation_failure_payload,
    )

    return effective_rc


def main(stdin: object = None, stdout: object = None) -> int:
    """Entry point. Reads BenchConfig JSON on stdin, runs the project's
    own tests inside the authoring container, writes KPI JSON on stdout."""
    stdin_obj = stdin if stdin is not None else sys.stdin
    stdout_obj = stdout if stdout is not None else sys.stdout

    try:
        cfg = json.loads(stdin_obj.read())
    except (
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
    ) as e:
        diagnostic = (
            "deepswe_dev_verify: BenchConfig stdin parse failed: "
            f"{type(e).__name__}: {e}"
        )
        logger.error("%s", diagnostic)
        _emit_runner_unavailable(
            stdout_obj,
            diagnostic=diagnostic,
            specific_flag="dev_verify_runner_protocol_unavailable",
        )
        return 2

    if not isinstance(cfg, dict):
        _emit_runner_unavailable(
            stdout_obj,
            diagnostic="deepswe_dev_verify: BenchConfig must be an object",
            specific_flag="dev_verify_runner_protocol_unavailable",
        )
        return 2
    raw_parameters = cfg.get("parameters")
    parameters = {} if raw_parameters is None else raw_parameters
    if not isinstance(parameters, dict):
        _emit_runner_unavailable(
            stdout_obj,
            diagnostic=(
                "deepswe_dev_verify: BenchConfig parameters must be an "
                f"object, got {type(parameters).__name__}"
            ),
            specific_flag="dev_verify_runner_protocol_unavailable",
        )
        return 2
    role = parameters.get("role", "dev_verify")
    if role not in ("dev_verify", "verify"):
        # ``verify`` tolerated only because _build_verify injects the
        # role name it dispatched under; the dev_verify role maps here.
        logger.error(
            "deepswe_dev_verify: role=%r not supported by this runner. "
            "The dataset gate is crucible.products.deepswe.runners."
            "deepswe_verifier; this runner serves the dev_verify role.",
            role,
        )
        _emit_runner_unavailable(
            stdout_obj,
            diagnostic=(
                "deepswe_dev_verify: unsupported runner role "
                f"{role!r}"
            ),
            specific_flag="dev_verify_runner_protocol_unavailable",
        )
        return 2

    return _run_dev_verify(parameters, stdout_obj)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
