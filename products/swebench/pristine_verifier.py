# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Patch-only pristine visible verification for SWE-bench candidates.

The authoring checkout is intentionally absent from this module's API.  A
candidate is transferred as immutable patch bytes into a second standalone
checkout, applied only after an independently managed container is ready, and
then exercised through the public SWE-bench visible-test runner.  Hidden
dataset fields are never read or serialized.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from crucible.common.benchmark_runtime import (
    managed_container_labels,
    managed_container_name,
    verify_immutable_image,
)
from crucible.common.storage_quota import DirectoryGrowthQuota
from crucible.products.swebench.dataset import SwebenchInstance
from crucible.products.swebench.instance_container import (
    container_immutable_image,
    reconcile_orphans,
    resolve_container_spec,
    setup_instance_container,
    teardown_instance_container,
)
from crucible.products.swebench.paths import is_standard_test_path
from crucible.products.swebench.repo_workspace import repo_workspace


PRISTINE_VERIFICATION_SCHEMA = "crucible.swebench.pristine-verification.v1"
_REPORT_SCHEMA = PRISTINE_VERIFICATION_SCHEMA
_RUNNER_MODULE = "crucible.products.swebench.runners.swebench_pytest"
_VERIFIER_PHASE = "visible-verifier"
_PATCH_LABEL = "io.crucible.candidate-patch-sha256"
_RAW_OUTPUT_LIMIT = 8_000
_REASON_LIMIT = 2_000
_VERIFIER_STORAGE_GROWTH_MB = 8_192
_GIT_TIMEOUT_S = 120
_CONTAINER_PROBE_TIMEOUT_S = 30
_MAX_CHANGED_PATHS = 10_000
_MAX_CHANGED_PATH_BYTES = 4_096
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BASE_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")
_IMAGE_DIGEST_RE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_COMPARISON_MODE_RE = re.compile(
    r"(?m)^\[crucible clean-base comparison\]\r?\n"
    r"mode=(?P<mode>[a-z_]+)$"
)
_COMPARISON_MODES = frozenset(
    {
        "",
        "baseline_match",
        "incomparable",
        "not_applicable",
        "not_needed",
        "preexisting_only",
        "regressed",
        "unavailable",
    }
)
_CLEANUP_STATUSES = frozenset({"complete", "failed", "not_started"})
_INVALID_BASE_COMMIT = "0" * 40
_INVALID_IMAGE_DIGEST = "invalid.local/crucible@sha256:" + "0" * 64
_TRUSTED_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_TRUSTED_RUNNER_BOOTSTRAP = (
    "import runpy,sys;"
    "source_root=sys.argv.pop(1);module=sys.argv.pop(1);"
    "sys.path.insert(0,source_root);"
    "runpy.run_module(module,run_name='__main__',alter_sys=True)"
)


class PristineVerificationStatus(StrEnum):
    """Terminal classification for one patch-only visible verification."""

    PASSED = "passed"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    TEST_FAILED = "test_failed"
    INFRA_FAILED = "infra_failed"


@dataclass(frozen=True)
class PristineVerificationReport:
    """Serializable evidence from one pristine candidate verification."""

    schema: str
    status: PristineVerificationStatus
    reason: str
    patch_sha256: str
    patch_bytes: int
    base_commit: str
    image_digest: str
    apply_check_rc: int | None
    apply_rc: int | None
    runner_rc: int | None
    candidate_rc: int | None
    comparison_mode: str
    preexisting_failures: int
    new_failures: int
    worsened_failures: int
    duration_s: float
    cleanup_status: str
    raw_output_tail: str

    def to_payload(self) -> dict[str, object]:
        """Return the exact v1 persistence payload."""

        return {
            "schema": self.schema,
            "status": self.status.value,
            "reason": self.reason,
            "patch_sha256": self.patch_sha256,
            "patch_bytes": self.patch_bytes,
            "base_commit": self.base_commit,
            "image_digest": self.image_digest,
            "apply_check_rc": self.apply_check_rc,
            "apply_rc": self.apply_rc,
            "runner_rc": self.runner_rc,
            "candidate_rc": self.candidate_rc,
            "comparison_mode": self.comparison_mode,
            "preexisting_failures": self.preexisting_failures,
            "new_failures": self.new_failures,
            "worsened_failures": self.worsened_failures,
            "duration_s": self.duration_s,
            "cleanup_status": self.cleanup_status,
            "raw_output_tail": self.raw_output_tail,
        }

    @classmethod
    def from_payload(cls, raw: object) -> "PristineVerificationReport":
        """Strictly deserialize one v1 report for checkpoint resume.

        Unknown fields are rejected so a future schema cannot be interpreted
        accidentally with weaker v1 semantics.
        """

        expected_fields = {field.name for field in dataclasses.fields(cls)}
        if not isinstance(raw, dict):
            raise ValueError("pristine verification payload must be an object")
        if set(raw) != expected_fields:
            missing = sorted(expected_fields - set(raw))
            unknown = sorted(set(raw) - expected_fields)
            raise ValueError(
                "pristine verification payload fields do not match v1: "
                f"missing={missing!r}, unknown={unknown!r}"
            )
        if raw["schema"] != _REPORT_SCHEMA:
            raise ValueError("unsupported pristine verification schema")
        try:
            status = PristineVerificationStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid pristine verification status") from exc

        def _required_string(name: str, *, allow_empty: bool = False) -> str:
            value = raw[name]
            if not isinstance(value, str) or (not allow_empty and not value):
                raise ValueError(f"invalid pristine verification {name}")
            return value

        def _count(name: str, *, positive: bool = False) -> int:
            value = raw[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if positive else 0)
            ):
                raise ValueError(f"invalid pristine verification {name}")
            return value

        def _returncode(name: str) -> int | None:
            value = raw[name]
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"invalid pristine verification {name}")
            return value

        reason = _required_string("reason")
        patch_sha256 = _required_string("patch_sha256")
        base_commit = _required_string("base_commit")
        image_digest = _required_string("image_digest")
        comparison_mode = _required_string("comparison_mode", allow_empty=True)
        cleanup_status = _required_string("cleanup_status")
        raw_output_tail = _required_string("raw_output_tail", allow_empty=True)
        if _SHA256_RE.fullmatch(patch_sha256) is None:
            raise ValueError("invalid pristine verification patch_sha256")
        if _BASE_COMMIT_RE.fullmatch(base_commit) is None:
            raise ValueError("invalid pristine verification base_commit")
        if _IMAGE_DIGEST_RE.fullmatch(image_digest) is None:
            raise ValueError("invalid pristine verification image_digest")
        if comparison_mode not in _COMPARISON_MODES:
            raise ValueError("invalid pristine verification comparison_mode")
        if cleanup_status not in _CLEANUP_STATUSES:
            raise ValueError("invalid pristine verification cleanup_status")
        if len(reason) > _REASON_LIMIT or len(raw_output_tail) > _RAW_OUTPUT_LIMIT:
            raise ValueError("pristine verification text field exceeds v1 bound")
        duration = raw["duration_s"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            raise ValueError("invalid pristine verification duration_s")

        report = cls(
            schema=_REPORT_SCHEMA,
            status=status,
            reason=reason,
            patch_sha256=patch_sha256,
            patch_bytes=_count("patch_bytes"),
            base_commit=base_commit,
            image_digest=image_digest,
            apply_check_rc=_returncode("apply_check_rc"),
            apply_rc=_returncode("apply_rc"),
            runner_rc=_returncode("runner_rc"),
            candidate_rc=_returncode("candidate_rc"),
            comparison_mode=comparison_mode,
            preexisting_failures=_count("preexisting_failures"),
            new_failures=_count("new_failures"),
            worsened_failures=_count("worsened_failures"),
            duration_s=float(duration),
            cleanup_status=cleanup_status,
            raw_output_tail=raw_output_tail,
        )
        report._validate_cross_fields()
        return report

    def _validate_cross_fields(self) -> None:
        if self.status == PristineVerificationStatus.PASSED:
            if (
                self.apply_check_rc != 0
                or self.apply_rc != 0
                or self.runner_rc != 0
                or self.candidate_rc not in {0, 1}
                or self.comparison_mode
                not in {
                    "baseline_match",
                    "preexisting_only",
                }
                or self.new_failures
                or self.worsened_failures
                or self.cleanup_status != "complete"
            ):
                raise ValueError("inconsistent passed pristine verification")
        elif self.status == PristineVerificationStatus.PATCH_APPLY_FAILED:
            if (
                self.apply_check_rc in {None, 0} and self.apply_rc in {None, 0}
            ) or self.cleanup_status != "complete":
                raise ValueError("patch_apply_failed lacks a failing apply result")
        elif self.status == PristineVerificationStatus.TEST_FAILED:
            if (
                self.apply_check_rc != 0
                or self.apply_rc != 0
                or self.runner_rc not in {1, 2}
                or self.candidate_rc not in {0, 1, 2}
                or self.comparison_mode != "regressed"
                or self.cleanup_status != "complete"
            ):
                raise ValueError("inconsistent test_failed pristine verification")


@dataclass(frozen=True)
class _RunnerEvidence:
    status: PristineVerificationStatus
    reason: str
    runner_rc: int
    candidate_rc: int | None = None
    comparison_mode: str = ""
    preexisting_failures: int = 0
    new_failures: int = 0
    worsened_failures: int = 0
    raw_output: str = ""


def _text_tail(value: object, limit: int = _RAW_OUTPUT_LIMIT) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    elif value is None:
        text = ""
    else:
        text = str(value)
    return text[-limit:]


def _bounded_reason(reason: object) -> str:
    rendered = _text_tail(reason, _REASON_LIMIT).strip()
    return rendered or "pristine verification failed without diagnostic detail"


def _bounded_output(parts: list[str]) -> str:
    return "\n".join(part for part in parts if part)[-_RAW_OUTPUT_LIMIT:]


def _run_git(
    workspace: Path,
    arguments: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = _GIT_TIMEOUT_S,
) -> subprocess.CompletedProcess[bytes]:
    """Run trusted host Git without a shell or repository hooks."""

    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *arguments],
        cwd=str(workspace),
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _validated_source_paths(payload: bytes) -> tuple[str, ...] | None:
    """Decode a bounded NUL-delimited Git path list under source-only policy."""

    if not payload or not payload.endswith(b"\0"):
        return None
    raw_paths = payload[:-1].split(b"\0")
    if not raw_paths or len(raw_paths) > _MAX_CHANGED_PATHS:
        return None
    paths: list[str] = []
    for raw_path in raw_paths:
        if not raw_path or len(raw_path) > _MAX_CHANGED_PATH_BYTES:
            return None
        try:
            path_text = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
        path = PurePosixPath(path_text)
        if (
            "\\" in path_text
            or any(not character.isprintable() for character in path_text)
            or path.is_absolute()
            or path.as_posix() != path_text
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.parts[0].startswith("-")
            or is_standard_test_path(path_text)
        ):
            return None
        paths.append(path_text)
    if len(set(paths)) != len(paths):
        return None
    return tuple(paths)


def _run_container_git(
    container_id: str,
    arguments: list[str],
    *,
    timeout: int = _CONTAINER_PROBE_TIMEOUT_S,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "docker",
            "exec",
            "--workdir",
            "/testbed",
            container_id,
            "git",
            *arguments,
        ],
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _invoke_runner(
    workspace: Path,
    payload: dict[str, object],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    """Invoke the trusted runner without importing from the candidate tree."""

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": "/dev/null",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _TRUSTED_RUNNER_BOOTSTRAP,
            str(_TRUSTED_SOURCE_ROOT),
            _RUNNER_MODULE,
        ],
        cwd=str(workspace),
        input=json.dumps(payload, sort_keys=True).encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=timeout,
        env=environment,
    )


def _container_is_absent(
    container_id: str,
    *,
    timeout_s: float = 5.0,
) -> bool:
    """Wait briefly for Docker's asynchronous ``--rm`` removal to finish."""

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            completed = subprocess.run(
                ["docker", "inspect", container_id],
                capture_output=True,
                check=False,
                timeout=_CONTAINER_PROBE_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if completed.returncode != 0:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.1, remaining))


def _kill_container(container_id: str | None) -> None:
    if not container_id:
        return
    try:
        subprocess.run(
            ["docker", "kill", container_id],
            capture_output=True,
            check=False,
            timeout=_CONTAINER_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _metric_int(
    values: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> int | None:
    if name not in values:
        if required:
            raise ValueError(f"runner omitted required metric {name}")
        return None
    value = values[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not float(value).is_integer()
        or float(value) < 0.0
    ):
        raise ValueError(f"runner emitted invalid metric {name}")
    return int(value)


def _parse_runner_evidence(
    completed: subprocess.CompletedProcess[bytes],
) -> _RunnerEvidence:
    stdout_text = _text_tail(completed.stdout, 128_000)
    stderr_text = _text_tail(completed.stderr, 32_000)
    protocol_output = _bounded_output([stdout_text, stderr_text])
    try:
        payload = json.loads(stdout_text)
    except (json.JSONDecodeError, TypeError) as exc:
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            f"visible runner emitted malformed JSON: {exc}",
            int(completed.returncode),
            raw_output=protocol_output,
        )
    if not isinstance(payload, dict):
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            "visible runner payload was not an object",
            int(completed.returncode),
            raw_output=protocol_output,
        )
    values = payload.get("values")
    raw_output = payload.get("raw_output")
    if not isinstance(values, dict) or not isinstance(raw_output, str):
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            "visible runner payload omitted values or raw_output",
            int(completed.returncode),
            raw_output=protocol_output,
        )
    combined_output = _bounded_output([raw_output, stderr_text])
    try:
        effective_rc = _metric_int(values, "returncode")
        python_rc = _metric_int(values, "python_returncode")
        candidate_rc = _metric_int(values, "visible_test_candidate_returncode")
        preexisting = _metric_int(values, "visible_test_preexisting_failures")
        new = _metric_int(values, "visible_test_new_failures")
        worsened = _metric_int(values, "visible_test_worsened_failures")
        candidate_failed = _metric_int(values, "visible_test_candidate_tests_failed")
        candidate_errored = _metric_int(values, "visible_test_candidate_tests_errored")
        baseline_unavailable = _metric_int(values, "visible_test_baseline_unavailable")
    except ValueError as exc:
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            str(exc),
            int(completed.returncode),
            raw_output=combined_output,
        )
    assert effective_rc is not None
    assert python_rc is not None
    assert candidate_rc is not None
    assert preexisting is not None
    assert new is not None
    assert worsened is not None
    assert candidate_failed is not None
    assert candidate_errored is not None
    assert baseline_unavailable is not None
    matches = list(_COMPARISON_MODE_RE.finditer(raw_output))
    comparison_mode = matches[-1].group("mode") if matches else ""
    common = {
        "runner_rc": int(completed.returncode),
        "candidate_rc": candidate_rc,
        "comparison_mode": comparison_mode,
        "preexisting_failures": preexisting,
        "new_failures": new,
        "worsened_failures": worsened,
        "raw_output": combined_output,
    }
    if effective_rc != completed.returncode or python_rc != completed.returncode:
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            "visible runner process and reported return codes disagree",
            **common,
        )
    if comparison_mode not in _COMPARISON_MODES or not comparison_mode:
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            "visible runner omitted a recognized clean-base comparison mode",
            **common,
        )
    if completed.returncode == 0:
        if (
            candidate_rc in {0, 1}
            and comparison_mode in {"baseline_match", "preexisting_only"}
            and new == 0
            and worsened == 0
            and baseline_unavailable == 0
        ):
            return _RunnerEvidence(
                PristineVerificationStatus.PASSED,
                "candidate visible tests matched or improved the pristine base",
                **common,
            )
        return _RunnerEvidence(
            PristineVerificationStatus.INFRA_FAILED,
            "zero-return runner evidence was structurally inconsistent",
            **common,
        )
    if (
        completed.returncode in {1, 2}
        and candidate_rc in {0, 1, 2}
        and comparison_mode == "regressed"
        and baseline_unavailable == 0
    ):
        return _RunnerEvidence(
            PristineVerificationStatus.TEST_FAILED,
            "candidate introduced or worsened visible test outcomes",
            **common,
        )
    return _RunnerEvidence(
        PristineVerificationStatus.INFRA_FAILED,
        "visible runner failure lacked complete regression evidence",
        **common,
    )


def verify_source_patch(
    instance: SwebenchInstance,
    *,
    source_patch: bytes,
    expected_sha256: str,
    expected_image_digest: str,
    repo_cache_dir: Path | None,
    scope: str,
    timeout: int = 1_800,
) -> PristineVerificationReport:
    """Apply and verify one source patch in an independent clean environment.

    Expected data/setup/subprocess/cleanup failures are returned as typed
    reports.  Only an incorrect Python API type raises as a programmer error.
    """

    if not isinstance(source_patch, bytes):
        raise TypeError("source_patch must be bytes")
    started = time.monotonic()
    actual_sha256 = hashlib.sha256(source_patch).hexdigest()
    raw_parts: list[str] = []
    base_commit_raw = getattr(instance, "base_commit", "")
    base_commit = (
        str(base_commit_raw).lower()
        if isinstance(base_commit_raw, str)
        and _BASE_COMMIT_RE.fullmatch(base_commit_raw.lower()) is not None
        else _INVALID_BASE_COMMIT
    )
    image_digest = (
        expected_image_digest
        if isinstance(expected_image_digest, str)
        and _IMAGE_DIGEST_RE.fullmatch(expected_image_digest) is not None
        else _INVALID_IMAGE_DIGEST
    )
    status = PristineVerificationStatus.INFRA_FAILED
    reason = "pristine verification did not start"
    apply_check_rc: int | None = None
    apply_rc: int | None = None
    runner_rc: int | None = None
    candidate_rc: int | None = None
    comparison_mode = ""
    preexisting_failures = 0
    new_failures = 0
    worsened_failures = 0
    cleanup_status = "not_started"
    cleanup_issues: list[str] = []
    workspace_path: Path | None = None
    quota: DirectoryGrowthQuota | None = None
    container_id: str | None = None

    def _make_report() -> PristineVerificationReport:
        return PristineVerificationReport(
            schema=_REPORT_SCHEMA,
            status=status,
            reason=_bounded_reason(reason),
            patch_sha256=actual_sha256,
            patch_bytes=len(source_patch),
            base_commit=base_commit,
            image_digest=image_digest,
            apply_check_rc=apply_check_rc,
            apply_rc=apply_rc,
            runner_rc=runner_rc,
            candidate_rc=candidate_rc,
            comparison_mode=comparison_mode,
            preexisting_failures=preexisting_failures,
            new_failures=new_failures,
            worsened_failures=worsened_failures,
            duration_s=max(0.0, time.monotonic() - started),
            cleanup_status=cleanup_status,
            raw_output_tail=_bounded_output(raw_parts),
        )

    if not source_patch:
        reason = "candidate source patch is empty"
        return _make_report()
    if (
        not isinstance(expected_sha256, str)
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or expected_sha256 != actual_sha256
    ):
        reason = "candidate patch hash does not match expected_sha256"
        return _make_report()
    if base_commit == _INVALID_BASE_COMMIT:
        reason = "instance base_commit is not a valid immutable Git id"
        return _make_report()
    if image_digest == _INVALID_IMAGE_DIGEST:
        reason = "expected_image_digest is not an immutable sha256 reference"
        return _make_report()
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        reason = "verifier timeout must be a positive integer"
        return _make_report()
    try:
        verify_immutable_image(expected_image_digest)
        reconcile_orphans(scope)
        with repo_workspace(instance, cache_dir=repo_cache_dir) as workspace:
            workspace_path = Path(workspace).resolve()

            def _kill_active() -> None:
                _kill_container(container_id)

            quota = DirectoryGrowthQuota(
                (workspace_path,),
                limit_bytes=_VERIFIER_STORAGE_GROWTH_MB * 1024 * 1024,
                on_exceeded=_kill_active,
                require_kernel=True,
            )
            with quota:
                try:
                    container_spec = resolve_container_spec(instance)
                    if container_spec is None:
                        raise RuntimeError(
                            "no supported SWE-bench verifier container spec"
                        )
                    pinned_spec = dataclasses.replace(
                        container_spec,
                        image=expected_image_digest,
                    )
                    instance_id = str(getattr(instance, "instance_id", ""))
                    repo = str(getattr(instance, "repo", ""))
                    version = str(getattr(instance, "version", ""))
                    if not instance_id or not repo:
                        raise RuntimeError("instance public identity is incomplete")
                    labels = managed_container_labels(
                        scope=scope,
                        task_id=instance_id,
                        phase=_VERIFIER_PHASE,
                    )
                    labels[_PATCH_LABEL] = actual_sha256
                    container_name = managed_container_name(
                        prefix="crucible-swebench",
                        scope=scope,
                        task_id=f"{instance_id}:{actual_sha256}",
                        phase=_VERIFIER_PHASE,
                    )
                    container_id = setup_instance_container(
                        workspace_path,
                        container_name,
                        pinned_spec,
                        labels=labels,
                    )
                    actual_image = container_immutable_image(container_id)
                    if actual_image != expected_image_digest:
                        raise RuntimeError("verifier container image digest mismatch")
                    host_head = _run_git(
                        workspace_path,
                        ["rev-parse", "--verify", "HEAD^{commit}"],
                        timeout=_CONTAINER_PROBE_TIMEOUT_S,
                    )
                    container_head = _run_container_git(
                        container_id,
                        ["rev-parse", "--verify", "HEAD^{commit}"],
                    )
                    raw_parts.extend(
                        (
                            _text_tail(host_head.stderr),
                            _text_tail(container_head.stderr),
                        )
                    )
                    if (
                        host_head.returncode != 0
                        or container_head.returncode != 0
                        or _text_tail(host_head.stdout).strip().lower() != base_commit
                        or _text_tail(container_head.stdout).strip().lower()
                        != base_commit
                    ):
                        raise RuntimeError(
                            "fresh verifier checkout base commit mismatch"
                        )

                    apply_check = _run_git(
                        workspace_path,
                        [
                            "apply",
                            "--check",
                            "--index",
                            "--binary",
                            "--whitespace=nowarn",
                            "-",
                        ],
                        input_bytes=source_patch,
                    )
                    apply_check_rc = int(apply_check.returncode)
                    raw_parts.extend(
                        (
                            _text_tail(apply_check.stdout),
                            _text_tail(apply_check.stderr),
                        )
                    )
                    if apply_check_rc != 0:
                        status = PristineVerificationStatus.PATCH_APPLY_FAILED
                        reason = "candidate patch failed pristine git apply --check"
                    else:
                        applied = _run_git(
                            workspace_path,
                            [
                                "apply",
                                "--index",
                                "--binary",
                                "--whitespace=nowarn",
                                "-",
                            ],
                            input_bytes=source_patch,
                        )
                        apply_rc = int(applied.returncode)
                        raw_parts.extend(
                            (
                                _text_tail(applied.stdout),
                                _text_tail(applied.stderr),
                            )
                        )
                        if apply_rc != 0:
                            status = PristineVerificationStatus.PATCH_APPLY_FAILED
                            reason = "candidate patch failed pristine git apply"
                        else:
                            changed = _run_git(
                                workspace_path,
                                ["diff", "--quiet", "HEAD", "--"],
                                timeout=_CONTAINER_PROBE_TIMEOUT_S,
                            )
                            if changed.returncode == 0:
                                apply_rc = 1
                                status = PristineVerificationStatus.PATCH_APPLY_FAILED
                                reason = (
                                    "candidate patch applied without changing "
                                    "the pristine checkout"
                                )
                            elif changed.returncode != 1:
                                raise RuntimeError(
                                    "could not validate the applied candidate diff"
                                )
                            else:
                                staged_paths = _run_git(
                                    workspace_path,
                                    [
                                        "diff",
                                        "--cached",
                                        "--name-only",
                                        "-z",
                                        "HEAD",
                                        "--",
                                    ],
                                    timeout=_CONTAINER_PROBE_TIMEOUT_S,
                                )
                                raw_parts.append(_text_tail(staged_paths.stderr))
                                if (
                                    staged_paths.returncode != 0
                                    or _validated_source_paths(
                                        staged_paths.stdout or b""
                                    )
                                    is None
                                ):
                                    raise RuntimeError(
                                        "applied candidate violates the "
                                        "source-only changed-path policy"
                                    )
                                metadata = {
                                    "instance_id": instance_id,
                                    "repo": repo,
                                    "version": version,
                                    "base_commit": base_commit,
                                    "container_id": container_id,
                                    "image_digest": expected_image_digest,
                                    "candidate_patch_sha256": actual_sha256,
                                }
                                runner_payload: dict[str, object] = {
                                    "bench_id": (f"pristine_verify_{instance_id}"),
                                    "parameters": {
                                        "role": "dev_verify",
                                        "runner_timeout_s": timeout,
                                        "swebench_metadata": metadata,
                                    },
                                    "env": {},
                                }
                                runner_completed = _invoke_runner(
                                    workspace_path,
                                    runner_payload,
                                    timeout=timeout,
                                )
                                evidence = _parse_runner_evidence(runner_completed)
                                status = evidence.status
                                reason = evidence.reason
                                runner_rc = evidence.runner_rc
                                candidate_rc = evidence.candidate_rc
                                comparison_mode = evidence.comparison_mode
                                preexisting_failures = evidence.preexisting_failures
                                new_failures = evidence.new_failures
                                worsened_failures = evidence.worsened_failures
                                raw_parts.append(evidence.raw_output)
                finally:
                    if container_id is not None:
                        teardown_instance_container(container_id, image=None)
                        if not _container_is_absent(container_id):
                            cleanup_issues.append("verifier container leaked")
        cleanup_status = "complete"
    except subprocess.TimeoutExpired as exc:
        status = PristineVerificationStatus.INFRA_FAILED
        reason = f"pristine verifier subprocess timed out after {timeout}s"
        raw_parts.extend(
            (
                _text_tail(exc.stdout),
                _text_tail(exc.stderr),
            )
        )
    except Exception as exc:  # noqa: BLE001 - typed fail-closed boundary
        status = PristineVerificationStatus.INFRA_FAILED
        reason = f"pristine verifier infrastructure failure: {type(exc).__name__}"
        raw_parts.append(_text_tail(str(exc)))
    finally:
        if quota is not None:
            try:
                quota.retire_kernel_lease()
            except Exception as exc:  # noqa: BLE001 - cleanup evidence
                cleanup_issues.append("quota retirement failed: " + type(exc).__name__)
        if workspace_path is not None and workspace_path.exists():
            cleanup_issues.append("verifier workspace leaked")
        if workspace_path is not None and cleanup_status == "not_started":
            cleanup_status = "complete"
        if cleanup_issues:
            cleanup_status = "failed"
            status = PristineVerificationStatus.INFRA_FAILED
            reason = "pristine verifier cleanup failed: " + "; ".join(cleanup_issues)
            raw_parts.extend(cleanup_issues)
    return _make_report()


__all__ = [
    "PRISTINE_VERIFICATION_SCHEMA",
    "PristineVerificationReport",
    "PristineVerificationStatus",
    "verify_source_patch",
]
