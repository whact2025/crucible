# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Progress-sensitive extension of Tier build-feedback retries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    InvocationResult,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

from crucible.common.policies import BreachPolicy
from crucible.common.runners.repair_progress import (
    REPAIR_FAILURE_SET_OUTPUT_KEY,
    RepairFailure,
    build_repair_failure_set,
)
from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows.tier import (
    _build_failure_signature,
    _build_failure_snapshot as _raw_build_failure_snapshot,
    _build_retry_test_surface_sha256,
    _should_extend_build_retries,
)


_TEST_SURFACE_SHA256 = "sha256:" + "0" * 64
_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub-progress-retry",
    requested_model="stub-model",
    events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
)


def _build_failure_snapshot(
    outcome,
    *,
    test_surface_sha256: str = _TEST_SURFACE_SHA256,
):
    return _raw_build_failure_snapshot(
        outcome,
        test_surface_sha256=test_surface_sha256,
    )


def _runner(
    output: str,
    *,
    returncode: int = 1,
    duration_s: float = 1.0,
) -> RunnerResult:
    payload = {
        "raw_output": output,
        "test_observations": [{
            "test_id": "tests/test_decode.py::test_invalid_packet",
            "status": "failed",
        }],
    }
    return RunnerResult(
        role="verify",
        runner="tests.progress_runner",
        returncode=returncode,
        values={"tests_failed": float(bool(returncode)), "duration_s": duration_s},
        raw_output=json.dumps(payload).encode("utf-8"),
        stderr=b"",
        duration_s=duration_s,
    )


def _failure(output: str, *, duration_s: float = 1.0) -> BuildVerifyOutcome:
    return BuildVerifyOutcome(
        "verify_failed",
        build=None,
        verify=_runner(output, duration_s=duration_s),
    )


def _success() -> BuildVerifyOutcome:
    return BuildVerifyOutcome(
        "success",
        build=None,
        verify=_runner("1 passed in 0.01s", returncode=0),
    )


def _dev_verify_unavailable() -> BuildVerifyOutcome:
    return BuildVerifyOutcome(
        "dev_verify_unavailable",
        build=None,
        verify=None,
        dev_verify=RunnerResult(
            role="dev_verify",
            runner="crucible.products.deepswe.runners.deepswe_dev_verify",
            returncode=2,
            values={
                "returncode": 2.0,
                "dev_verify_baseline_unavailable": 1.0,
                "dev_verify_infrastructure_unavailable": 1.0,
            },
            raw_output=b'{"raw_output":"sealed dependency unavailable"}',
            stderr=b"",
            duration_s=1.0,
        ),
    )


def _declared_dev_verify_failure(
    *,
    raw_output: str,
    failures: tuple[tuple[str, str, str], ...],
    output_sha256: str,
) -> BuildVerifyOutcome:
    surface = {
        "schema_version": "crucible.declared-verification-surface.v1",
        "slices": [{
            "slice_id": "s0",
            "files_in_scope": [
                "tests/runners/process_test_runner_tests.js",
            ],
            "creates_paths": [],
        }],
    }
    surface_digest = hashlib.sha256(json.dumps(
        surface,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    payload = {
        "raw_output": raw_output,
        "declared_verification_reference": {
            "schema_version": (
                "crucible.accepted-declared-verification-reference.v1"
            ),
            "language": "javascript",
            "parser_kind": "mocha",
            "predecessor_surface": surface,
            "predecessor_surface_sha256": surface_digest,
            "outcomes": [{
                "path": "tests/runners/process_test_runner_tests.js",
                "command_sha256": "c" * 64,
                "outcome": {
                    "schema": "crucible.deepswe.dev-verify-outcome.v3",
                    "failures": [list(item) for item in failures],
                    "output_sha256": output_sha256,
                    "parser_kind": "mocha",
                    "returncode": 1,
                    "structural_error": "",
                    "summary": [
                        ["tests_failed", len(failures)],
                        ["tests_passed", 7 - len(failures)],
                        ["tests_skipped", 0],
                    ],
                    "tests_run": 7,
                },
            }],
        },
    }
    payload[REPAIR_FAILURE_SET_OUTPUT_KEY] = build_repair_failure_set(
        surface={
            "schema_version": "crucible.test-repair-surface.v1",
            "broad_tests_run": 115,
            "declared_roster": [{
                "command_sha256": "c" * 64,
                "neutral_summary": [["tests_skipped", 0]],
                "path": "tests/runners/process_test_runner_tests.js",
                "tests_run": 7,
            }],
            "declared_surface_sha256": surface_digest,
        },
        failures=tuple(
            RepairFailure(
                failure_id=json.dumps(
                    [
                        "tests/runners/process_test_runner_tests.js",
                        node_id,
                    ],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                status=status,
                fingerprint=fingerprint,
            )
            for node_id, status, fingerprint in failures
        ),
    )
    runner = RunnerResult(
        role="dev_verify",
        runner="crucible.products.deepswe.runners.deepswe_dev_verify",
        returncode=1,
        values={
            "tests_failed": 0.0,
            "tests_passed": 115.0,
            "dev_verify_structurally_valid": 1.0,
            "dev_verify_tests_run": 115.0,
            "dev_verify_declared_test_files_required": 1.0,
            "dev_verify_declared_test_files_executed": 1.0,
            "dev_verify_declared_test_files_failed": 1.0,
            "dev_verify_declared_test_files_passed": 0.0,
            "dev_verify_declared_test_files_structural_failures": 0.0,
            "dev_verify_declared_test_preexisting_failures": 0.0,
        },
        raw_output=json.dumps(payload).encode("utf-8"),
        stderr=b"",
        duration_s=1.0,
    )
    return BuildVerifyOutcome(
        "dev_verify_failed",
        build=None,
        verify=None,
        dev_verify=runner,
    )


def _amend_declared_failure(
    outcome: BuildVerifyOutcome,
    *,
    mutate_payload=None,
    values: dict[str, float] | None = None,
    returncode: int | None = None,
) -> BuildVerifyOutcome:
    runner = outcome.dev_verify
    assert runner is not None
    payload = json.loads(runner.raw_output)
    if mutate_payload is not None:
        mutate_payload(payload)
    amended_runner = replace(
        runner,
        raw_output=json.dumps(payload).encode("utf-8"),
        values=(values if values is not None else dict(runner.values)),
        returncode=(runner.returncode if returncode is None else returncode),
    )
    return replace(outcome, dev_verify=amended_runner)


def _rewrite_repair_failure_set(
    payload: dict[str, object],
    *,
    mutate_surface=None,
    failures: tuple[RepairFailure, ...] | None = None,
) -> None:
    envelope = payload[REPAIR_FAILURE_SET_OUTPUT_KEY]
    assert isinstance(envelope, dict)
    surface = json.loads(json.dumps(envelope["surface"]))
    if mutate_surface is not None:
        mutate_surface(surface)
    if failures is None:
        failures = tuple(
            RepairFailure(
                failure_id=item["id"],
                status=item["status"],
                fingerprint=item["fingerprint"],
            )
            for item in envelope["failures"]
        )
    payload[REPAIR_FAILURE_SET_OUTPUT_KEY] = build_repair_failure_set(
        surface=surface,
        failures=failures,
    )


def _progress_extension_allowed(
    previous: BuildVerifyOutcome,
    current: BuildVerifyOutcome,
) -> bool:
    return _should_extend_build_retries(
        build_attempt=1,
        retry_limit=1,
        previous_signature=_build_failure_signature(previous),
        current_signature=_build_failure_signature(current),
        previous_failures=_build_failure_snapshot(previous),
        current_failures=_build_failure_snapshot(current),
        progress_extensions_used=0,
    )


FAILURE_A = """\
tests/test_decode.py::test_invalid_packet FAILED
.F..                                                                     [ 40%]
tmp/pytest-of-alpha/pytest-2/test_invalid_packet0/input.json
E   ValueError: malformed packet header
=========================== 1 failed in 1.23s ===========================
"""

FAILURE_A_NOISY = """\
tests/test_decode.py::test_invalid_packet FAILED
....F                                                                    [ 97%]
tmp/pytest-of-beta/pytest-98/test_invalid_packet0/input.json
\x1b[31mE   ValueError: malformed packet header\x1b[0m
=========================== 1 failed in 19.87s ===========================
"""

FAILURE_B = """\
tests/test_decode.py::test_invalid_packet FAILED
E   TypeError: decoder requires a bytes-like packet
=========================== 1 failed in 2.00s ===========================
"""

FAILURE_C = """\
tests/test_decode.py::test_invalid_packet FAILED
E   RuntimeError: checksum validation failed
=========================== 1 failed in 3.00s ===========================
"""

DECLARED_FAILURE_A = (
    "ProcessTestRunner > abort > returns a Promise",
    "FAILED",
    "sha256:" + "a" * 64,
)
DECLARED_FAILURE_B = (
    "ProcessTestRunner > suppression > preserves diagnostics",
    "FAILED",
    "sha256:" + "b" * 64,
)
DECLARED_FAILURE_C = (
    "ProcessTestRunner > exit > preserves status",
    "FAILED",
    "sha256:" + "c" * 64,
)


def _two_path_declared_failure() -> BuildVerifyOutcome:
    base = _declared_dev_verify_failure(
        raw_output="two failing paths",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="e" * 64,
    )

    def add_second_path(payload):
        reference = payload["declared_verification_reference"]
        surface = reference["predecessor_surface"]
        second_path = "tests/runners/secondary_runner_tests.js"
        surface["slices"][0]["files_in_scope"].append(second_path)
        reference["predecessor_surface_sha256"] = hashlib.sha256(json.dumps(
            surface,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        second = json.loads(json.dumps(reference["outcomes"][0]))
        second["path"] = second_path
        second["command_sha256"] = "d" * 64
        second["outcome"]["failures"] = [list(DECLARED_FAILURE_C)]
        second["outcome"]["output_sha256"] = "f" * 64
        reference["outcomes"].append(second)

        def extend_roster(repair_surface):
            repair_surface["declared_surface_sha256"] = reference[
                "predecessor_surface_sha256"
            ]
            repair_surface["declared_roster"].append({
                "command_sha256": "d" * 64,
                "neutral_summary": [["tests_skipped", 0]],
                "path": second_path,
                "tests_run": 7,
            })

        _rewrite_repair_failure_set(
            payload,
            mutate_surface=extend_roster,
            failures=(
                RepairFailure(
                    json.dumps([
                        "tests/runners/process_test_runner_tests.js",
                        DECLARED_FAILURE_A[0],
                    ], separators=(",", ":")),
                    DECLARED_FAILURE_A[1],
                    DECLARED_FAILURE_A[2],
                ),
                RepairFailure(
                    json.dumps(
                        [second_path, DECLARED_FAILURE_C[0]],
                        separators=(",", ":"),
                    ),
                    DECLARED_FAILURE_C[1],
                    DECLARED_FAILURE_C[2],
                ),
            ),
        )

    values = dict(base.dev_verify.values)
    values.update({
        "dev_verify_declared_test_files_required": 2.0,
        "dev_verify_declared_test_files_executed": 2.0,
        "dev_verify_declared_test_files_failed": 2.0,
    })
    return _amend_declared_failure(
        base,
        mutate_payload=add_second_path,
        values=values,
    )


def test_failure_signature_ignores_pytest_duration_progress_and_run_noise():
    assert _build_failure_signature(
        _failure(FAILURE_A, duration_s=1.23)
    ) == _build_failure_signature(
        _failure(FAILURE_A_NOISY, duration_s=19.87)
    )


def test_failure_signature_distinguishes_actual_exception_content():
    assert _build_failure_signature(_failure(FAILURE_A)) != (
        _build_failure_signature(_failure(FAILURE_B))
    )


def test_declared_failure_signature_ignores_runner_noise_and_output_digest():
    first = _declared_dev_verify_failure(
        raw_output=(
            "(node:1172) warning\n"
            "http://localhost:7357/-1\n"
            "ProcessTestRunner > abort > returns a Promise"
        ),
        failures=(DECLARED_FAILURE_A,),
        output_sha256="1" * 64,
    )
    second = _declared_dev_verify_failure(
        raw_output=(
            "(node:9821) warning\n"
            "http://localhost:8123/-1\n"
            "ProcessTestRunner > abort > returns a Promise"
        ),
        failures=(DECLARED_FAILURE_A,),
        output_sha256="2" * 64,
    )

    assert _build_failure_signature(first) == _build_failure_signature(second)


def test_declared_failure_signature_tracks_failure_identity_change():
    first = _declared_dev_verify_failure(
        raw_output="volatile output A",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="3" * 64,
    )
    second = _declared_dev_verify_failure(
        raw_output="volatile output B",
        failures=(DECLARED_FAILURE_B,),
        output_sha256="4" * 64,
    )

    assert _build_failure_signature(first) != _build_failure_signature(second)


def test_progress_extension_requires_strict_unchanged_failure_subset():
    previous = _declared_dev_verify_failure(
        raw_output="two failures",
        failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
        output_sha256="5" * 64,
    )
    current = _declared_dev_verify_failure(
        raw_output="one failure",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="6" * 64,
    )
    signature_a = _build_failure_signature(previous)
    signature_b = _build_failure_signature(current)
    snapshot_a = _build_failure_snapshot(previous)
    snapshot_b = _build_failure_snapshot(current)
    assert signature_a is not None and signature_b is not None
    assert snapshot_a is not None and snapshot_b is not None

    assert _should_extend_build_retries(
        build_attempt=1,
        retry_limit=1,
        previous_signature=signature_a,
        current_signature=signature_b,
        previous_failures=snapshot_a,
        current_failures=snapshot_b,
        progress_extensions_used=0,
    )
    assert not _should_extend_build_retries(
        build_attempt=1,
        retry_limit=1,
        previous_signature=signature_a,
        current_signature=signature_a,
        previous_failures=snapshot_a,
        current_failures=snapshot_a,
        progress_extensions_used=0,
    )
    assert not _should_extend_build_retries(
        build_attempt=1,
        retry_limit=1,
        previous_signature=None,
        current_signature=signature_b,
        previous_failures=None,
        current_failures=snapshot_b,
        progress_extensions_used=0,
    )
    assert not _should_extend_build_retries(
        build_attempt=0,
        retry_limit=1,
        previous_signature=signature_a,
        current_signature=signature_b,
        previous_failures=snapshot_a,
        current_failures=snapshot_b,
        progress_extensions_used=0,
    )
    assert not _should_extend_build_retries(
        build_attempt=2,
        retry_limit=2,
        previous_signature=signature_a,
        current_signature=signature_b,
        previous_failures=snapshot_a,
        current_failures=snapshot_b,
        progress_extensions_used=1,
    )


@pytest.mark.parametrize(
    "current_failures",
    [
        pytest.param(
            (DECLARED_FAILURE_A, DECLARED_FAILURE_B),
            id="equal",
        ),
        pytest.param((DECLARED_FAILURE_C,), id="disjoint"),
        pytest.param(
            (DECLARED_FAILURE_A, DECLARED_FAILURE_B, DECLARED_FAILURE_C),
            id="superset",
        ),
        pytest.param(
            (
                (
                    DECLARED_FAILURE_A[0],
                    DECLARED_FAILURE_A[1],
                    "sha256:" + "d" * 64,
                ),
            ),
            id="changed-survivor-fingerprint",
        ),
    ],
)
def test_progress_extension_denies_nonmonotonic_failure_sets(current_failures):
    previous = _declared_dev_verify_failure(
        raw_output="previous",
        failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
        output_sha256="7" * 64,
    )
    current = _declared_dev_verify_failure(
        raw_output="current",
        failures=current_failures,
        output_sha256="8" * 64,
    )
    assert not _should_extend_build_retries(
        build_attempt=1,
        retry_limit=1,
        previous_signature=_build_failure_signature(previous),
        current_signature=_build_failure_signature(current),
        previous_failures=_build_failure_snapshot(previous),
        current_failures=_build_failure_snapshot(current),
        progress_extensions_used=0,
    )


def test_progress_extension_denies_partial_replacement():
    previous = _declared_dev_verify_failure(
        raw_output="A and B",
        failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
        output_sha256="1" * 64,
    )
    current = _declared_dev_verify_failure(
        raw_output="B and C",
        failures=(DECLARED_FAILURE_B, DECLARED_FAILURE_C),
        output_sha256="2" * 64,
    )

    assert not _progress_extension_allowed(previous, current)


def test_progress_extension_denies_omitted_outcome_roster():
    previous = _two_path_declared_failure()
    previous_runner = previous.dev_verify
    assert previous_runner is not None

    def omit_second_path(payload):
        reference = payload["declared_verification_reference"]
        reference["outcomes"] = reference["outcomes"][:1]
        envelope = payload[REPAIR_FAILURE_SET_OUTPUT_KEY]
        first = envelope["failures"][0]

        def shrink_roster(surface):
            surface["declared_roster"] = surface["declared_roster"][:1]

        _rewrite_repair_failure_set(
            payload,
            mutate_surface=shrink_roster,
            failures=(RepairFailure(
                first["id"], first["status"], first["fingerprint"],
            ),),
        )

    values = dict(previous_runner.values)
    values.update({
        "dev_verify_declared_test_files_required": 1.0,
        "dev_verify_declared_test_files_executed": 1.0,
        "dev_verify_declared_test_files_failed": 1.0,
    })
    current = _amend_declared_failure(
        previous,
        mutate_payload=omit_second_path,
        values=values,
    )

    assert _build_failure_snapshot(current) is not None
    assert not _progress_extension_allowed(previous, current)


def test_progress_extension_denies_changed_command_roster():
    previous = _declared_dev_verify_failure(
        raw_output="A and B",
        failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
        output_sha256="3" * 64,
    )
    current = _declared_dev_verify_failure(
        raw_output="A",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="4" * 64,
    )

    def change_command(payload):
        payload["declared_verification_reference"]["outcomes"][0][
            "command_sha256"
        ] = "e" * 64

        def change_repair_command(surface):
            surface["declared_roster"][0]["command_sha256"] = "e" * 64

        _rewrite_repair_failure_set(
            payload,
            mutate_surface=change_repair_command,
        )

    current = _amend_declared_failure(
        current,
        mutate_payload=change_command,
    )

    assert not _progress_extension_allowed(previous, current)


@pytest.mark.parametrize(
    "amendment",
    [
        pytest.param("broad_failure", id="broad-suite-failure"),
        pytest.param("broad_coverage", id="inconsistent-broad-suite-coverage"),
        pytest.param("broad_structure", id="invalid-broad-suite-structure"),
        pytest.param("outer_rc", id="outer-infrastructure-return-code"),
        pytest.param("vacuous_pass", id="vacuous-inner-pass"),
        pytest.param("structural", id="structural-inner-outcome"),
    ],
)
def test_incomplete_failure_snapshot_cannot_authorize_progress(amendment):
    current = _declared_dev_verify_failure(
        raw_output="A",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="5" * 64,
    )
    runner = current.dev_verify
    assert runner is not None
    values = dict(runner.values)
    returncode = None

    def mutate_payload(payload):
        if amendment == "broad_failure":
            payload.pop(REPAIR_FAILURE_SET_OUTPUT_KEY)
        elif amendment == "vacuous_pass":
            payload[REPAIR_FAILURE_SET_OUTPUT_KEY]["complete"] = False
        elif amendment == "structural":
            payload[REPAIR_FAILURE_SET_OUTPUT_KEY]["failures"] = []
        elif amendment == "broad_coverage":
            payload[REPAIR_FAILURE_SET_OUTPUT_KEY]["surface"][
                "broad_tests_run"
            ] = 1
        elif amendment == "broad_structure":
            payload[REPAIR_FAILURE_SET_OUTPUT_KEY]["failures"][0][
                "fingerprint"
            ] = "not-a-digest"

    if amendment == "outer_rc":
        returncode = 2
    current = _amend_declared_failure(
        current,
        mutate_payload=mutate_payload,
        values=values,
        returncode=returncode,
    )

    assert _build_failure_snapshot(current) is None


@pytest.mark.parametrize(
    "coverage_change",
    [
        pytest.param("reduced_tests", id="reduced-test-count"),
        pytest.param("increased_skips", id="increased-skips"),
        pytest.param("changed_surface", id="changed-declared-surface"),
    ],
)
def test_progress_extension_requires_stable_suite_coverage(coverage_change):
    previous = _declared_dev_verify_failure(
        raw_output="A and B",
        failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
        output_sha256="6" * 64,
    )
    current = _declared_dev_verify_failure(
        raw_output="A",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="7" * 64,
    )

    def change_coverage(payload):
        reference = payload["declared_verification_reference"]
        outcome = reference["outcomes"][0]["outcome"]
        if coverage_change == "reduced_tests":
            outcome["tests_run"] = 6
            outcome["summary"] = [
                ["tests_failed", 1],
                ["tests_passed", 5],
                ["tests_skipped", 0],
            ]

            def reduce_repair_tests(surface):
                surface["declared_roster"][0]["tests_run"] = 6

            _rewrite_repair_failure_set(
                payload,
                mutate_surface=reduce_repair_tests,
            )
        elif coverage_change == "increased_skips":
            outcome["summary"] = [
                ["tests_failed", 1],
                ["tests_passed", 5],
                ["tests_skipped", 1],
            ]

            def increase_repair_skips(surface):
                surface["declared_roster"][0]["neutral_summary"] = [
                    ["tests_skipped", 1],
                ]

            _rewrite_repair_failure_set(
                payload,
                mutate_surface=increase_repair_skips,
            )
        else:
            surface = reference["predecessor_surface"]
            surface["slices"][0]["creates_paths"].append(
                "tests/runners/new_test.js"
            )
            reference["predecessor_surface_sha256"] = hashlib.sha256(
                json.dumps(
                    surface,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()

            def change_repair_surface(surface):
                surface["declared_surface_sha256"] = reference[
                    "predecessor_surface_sha256"
                ]

            _rewrite_repair_failure_set(
                payload,
                mutate_surface=change_repair_surface,
            )

    current = _amend_declared_failure(
        current,
        mutate_payload=change_coverage,
    )

    assert _build_failure_snapshot(current) is not None
    assert not _progress_extension_allowed(previous, current)


def test_progress_extension_requires_unchanged_declared_test_bytes():
    previous = _declared_dev_verify_failure(
        raw_output="A and B",
        failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
        output_sha256="8" * 64,
    )
    current = _declared_dev_verify_failure(
        raw_output="A",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="9" * 64,
    )

    assert not _should_extend_build_retries(
        build_attempt=1,
        retry_limit=1,
        previous_signature=_build_failure_signature(previous),
        current_signature=_build_failure_signature(current),
        previous_failures=_build_failure_snapshot(
            previous,
            test_surface_sha256="sha256:" + "a" * 64,
        ),
        current_failures=_build_failure_snapshot(
            current,
            test_surface_sha256="sha256:" + "b" * 64,
        ),
        progress_extensions_used=0,
    )


@pytest.mark.parametrize(
    "test_surface_sha256",
    [None, "", "sha256:not-a-digest", "sha256:" + "A" * 64],
)
def test_failure_snapshot_requires_valid_test_surface_digest(
    test_surface_sha256,
):
    outcome = _declared_dev_verify_failure(
        raw_output="A",
        failures=(DECLARED_FAILURE_A,),
        output_sha256="a" * 64,
    )

    assert _raw_build_failure_snapshot(
        outcome,
        test_surface_sha256=test_surface_sha256,
    ) is None


def test_retry_surface_binds_structural_and_root_level_test_siblings(tmp_path):
    nested = tmp_path / "tests" / "nested.spec.js"
    nested.parent.mkdir()
    nested.write_text("nested before\n", encoding="utf-8")
    root_test = tmp_path / "root.test.js"
    root_test.write_text("root before\n", encoding="utf-8")
    directive = ImplementationDirective(
        slice_id="s0",
        intent="Update both declared test files.",
        files_in_scope=("tests/nested.spec.js", "root.test.js"),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 2},
    )

    before = _build_retry_test_surface_sha256(
        cwd=str(tmp_path),
        directive=directive,
    )
    root_test.write_text("root after\n", encoding="utf-8")
    after = _build_retry_test_surface_sha256(
        cwd=str(tmp_path),
        directive=directive,
    )

    assert before != after


class _EditingBackend:
    name = "stub-progress-retry"
    model = "stub-model"

    def __init__(self, repository: Path) -> None:
        self.repository = repository
        self.implement_calls = 0

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation) -> InvocationResult:
        self.implement_calls += 1
        (self.repository / "file_0.py").write_text(
            f"# implementation attempt {self.implement_calls}\n",
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="done",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


def _init_repo(repository: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tier@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tier Retry Test"],
        cwd=repository,
        check=True,
    )
    (repository / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=repository,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _spec(base_commit: str) -> Spec:
    slice_ = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent="Implement the bounded decoder change.",
            files_in_scope=("file_0.py",),
            creates_paths=("file_0.py",),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_progress_retry_test",
            repo_url="https://example.invalid/tier-progress.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Exercise progress-sensitive build retry bounds.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),),
        ),
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _run_tier_with_gate_sequence(
    tmp_path: Path,
    monkeypatch,
    outcomes: tuple[BuildVerifyOutcome, ...],
) -> tuple[int, _EditingBackend, int]:
    repository = tmp_path / "repo"
    repository.mkdir()
    base_commit = _init_repo(repository)
    spec_dir = tmp_path / "corpus" / "spec1"
    spec_dir.mkdir(parents=True)
    Corpus(spec_dir).put_spec(_spec(base_commit))

    backend = _EditingBackend(repository)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )
    gate_calls = 0

    def build_and_verify(*args, **kwargs):
        nonlocal gate_calls
        if gate_calls >= len(outcomes):
            raise AssertionError("unexpected build/verify retry")
        outcome = outcomes[gate_calls]
        gate_calls += 1
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_progress_retry_gate_stub",
            succeeded=outcome.classification.startswith("success"),
        )
        return outcome

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        build_and_verify,
    )
    from crucible.workflows.tier import run_tier

    rc = run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repository),
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        breach_policy=BreachPolicy(enabled=False),
        build_retries=1,
    )
    return rc, backend, gate_calls


@pytest.mark.parametrize(
    ("outcomes", "expected_rc", "expected_calls"),
    [
        pytest.param(
            (
                _declared_dev_verify_failure(
                    raw_output="two failures",
                    failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
                    output_sha256="9" * 64,
                ),
                _declared_dev_verify_failure(
                    raw_output="one surviving failure",
                    failures=(DECLARED_FAILURE_A,),
                    output_sha256="a" * 64,
                ),
                _success(),
            ),
            0,
            3,
            id="strict-failure-subset-grants-one-progress-retry",
        ),
        pytest.param(
            (_failure(FAILURE_A), _failure(FAILURE_A_NOISY), _success()),
            1,
            2,
            id="normalized-identical-failure-stops-at-base-limit",
        ),
        pytest.param(
            (_dev_verify_unavailable(),),
            1,
            1,
            id="infrastructure-unavailable-never-spends-model-retry",
        ),
        pytest.param(
            (
                _declared_dev_verify_failure(
                    raw_output="three failures",
                    failures=(
                        DECLARED_FAILURE_A,
                        DECLARED_FAILURE_B,
                        DECLARED_FAILURE_C,
                    ),
                    output_sha256="b" * 64,
                ),
                _declared_dev_verify_failure(
                    raw_output="two surviving failures",
                    failures=(DECLARED_FAILURE_A, DECLARED_FAILURE_B),
                    output_sha256="c" * 64,
                ),
                _declared_dev_verify_failure(
                    raw_output="one surviving failure",
                    failures=(DECLARED_FAILURE_A,),
                    output_sha256="d" * 64,
                ),
                _success(),
            ),
            1,
            3,
            id="only-one-progress-extension-is-available",
        ),
    ],
)
def test_tier_build_retry_loop_extends_only_for_bounded_progress(
    tmp_path: Path,
    monkeypatch,
    outcomes: tuple[BuildVerifyOutcome, ...],
    expected_rc: int,
    expected_calls: int,
):
    rc, backend, gate_calls = _run_tier_with_gate_sequence(
        tmp_path,
        monkeypatch,
        outcomes,
    )

    assert rc == expected_rc
    assert gate_calls == expected_calls
    assert backend.implement_calls == expected_calls
