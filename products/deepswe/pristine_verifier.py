# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""DeepSWE candidate capture and pristine replay.

Tier restores pipeline-owned breach artifacts from its provenance-bound
mutation journal before this module runs.  Every repository change that
remains after that restoration is part of the model's canonical DeepSWE patch,
including changes under test-looking paths.  This module content-addresses
that full patch, applies its exact bytes to a second checkout populated from
the immutable task image, and proves that the hidden verifier graded those
same bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any

from assay.primitives.implement import GitCaptureError, capture_workspace_delta

from crucible.products.deepswe.git_safety import git_argv, sanitized_git_env
from crucible.products.deepswe.patch_limits import (
    DEFAULT_MAX_PATCH_BYTES,
    validate_patch_byte_limit,
)
from crucible.products.deepswe.secure_artifacts import (
    ArtifactReadStatus,
    read_bounded_regular_artifact,
)


class CandidateCaptureStatus(StrEnum):
    """Outcome of normalizing an authoring checkout."""

    CAPTURED = "captured"
    EMPTY = "empty"
    FAILED = "failed"


class PristineReplayStatus(StrEnum):
    """Outcome of applying a candidate to an immutable-base checkout."""

    PASSED = "passed"
    BASE_INVALID = "base_invalid"
    PATCH_APPLY_FAILED = "patch_apply_failed"
    PATCH_MISMATCH = "patch_mismatch"


class CandidateApplyStatus(StrEnum):
    """Non-authoritative state of applying bytes to a clean checkout."""

    APPLIED = "applied"
    BASE_INVALID = "base_invalid"
    PATCH_APPLY_FAILED = "patch_apply_failed"


class VerifierPatchStatus(StrEnum):
    """Binding between the hidden verifier's artifact and the candidate."""

    MATCHED = "matched"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNREADABLE = "unreadable"


@dataclasses.dataclass(frozen=True)
class DeepSWECandidate:
    """One exact canonical repository patch retained independently of grading."""

    patch: bytes = dataclasses.field(repr=False)
    status: CandidateCaptureStatus
    diff_ref: str
    sha256: str
    diff_bytes: int
    files_modified: tuple[str, ...]
    # Backward-compatible payload field. DeepSWE no longer filters paths by
    # name, so new candidates always leave this empty. Pipeline-owned breach
    # mutations are removed earlier by Tier's provenance-bound journal.
    stripped_protected_paths: tuple[str, ...] = ()
    raw_diff_bytes: int = 0
    error: str | None = None

    @classmethod
    def failed(cls, error: str) -> "DeepSWECandidate":
        return cls(
            patch=b"",
            status=CandidateCaptureStatus.FAILED,
            diff_ref=_content_ref(b""),
            sha256=_sha256(b""),
            diff_bytes=0,
            files_modified=(),
            error=error[:2000],
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "crucible.deepswe.candidate.v1",
            "status": self.status.value,
            "diff_ref": self.diff_ref,
            "sha256": self.sha256,
            "diff_bytes": self.diff_bytes,
            "files_modified": list(self.files_modified),
            "stripped_protected_paths": list(self.stripped_protected_paths),
            "raw_diff_bytes": self.raw_diff_bytes,
            "error": self.error,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        patch: bytes,
    ) -> "DeepSWECandidate":
        if not isinstance(payload, dict) or payload.get("schema") != (
            "crucible.deepswe.candidate.v1"
        ):
            raise ValueError("invalid DeepSWE candidate payload")
        files = payload.get("files_modified")
        stripped = payload.get("stripped_protected_paths")
        if (
            not isinstance(files, list)
            or not all(isinstance(item, str) for item in files)
            or not isinstance(stripped, list)
            or not all(isinstance(item, str) for item in stripped)
        ):
            raise ValueError("invalid DeepSWE candidate path metadata")
        try:
            status = CandidateCaptureStatus(payload["status"])
            diff_ref = str(payload["diff_ref"])
            sha256 = str(payload["sha256"])
            diff_bytes = int(payload["diff_bytes"])
            raw_diff_bytes = int(payload.get("raw_diff_bytes", diff_bytes))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid DeepSWE candidate fields") from exc
        candidate = cls(
            patch=patch,
            status=status,
            diff_ref=diff_ref,
            sha256=sha256,
            diff_bytes=diff_bytes,
            files_modified=tuple(files),
            stripped_protected_paths=tuple(stripped),
            raw_diff_bytes=raw_diff_bytes,
            error=(str(payload["error"]) if payload.get("error") is not None else None),
        )
        validate_candidate(candidate)
        return candidate


@dataclasses.dataclass(frozen=True)
class PristineReplayReport:
    status: PristineReplayStatus
    base_commit: str
    candidate_sha256: str
    observed_sha256: str | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "crucible.deepswe.pristine-replay.v1",
            "status": self.status.value,
            "base_commit": self.base_commit,
            "candidate_sha256": self.candidate_sha256,
            "observed_sha256": self.observed_sha256,
            "error": self.error,
        }


@dataclasses.dataclass(frozen=True)
class CandidateApplyReport:
    status: CandidateApplyStatus
    base_commit: str
    candidate_sha256: str
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "crucible.deepswe.candidate-apply.v1",
            "status": self.status.value,
            "base_commit": self.base_commit,
            "candidate_sha256": self.candidate_sha256,
            "error": self.error,
        }


@dataclasses.dataclass(frozen=True)
class VerifierPatchReport:
    status: VerifierPatchStatus
    candidate_sha256: str
    observed_sha256: str | None = None
    observed_bytes: int | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "crucible.deepswe.verifier-patch.v1",
            "status": self.status.value,
            "candidate_sha256": self.candidate_sha256,
            "observed_sha256": self.observed_sha256,
            "observed_bytes": self.observed_bytes,
            "error": self.error,
        }


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _content_ref(blob: bytes) -> str:
    return f"sha256:{_sha256(blob)}"


def validate_candidate(candidate: DeepSWECandidate) -> None:
    """Reject candidate metadata that is not bound to its exact bytes."""

    if candidate.status == CandidateCaptureStatus.FAILED:
        if candidate.patch or candidate.diff_bytes:
            raise ValueError("failed candidate cannot carry patch bytes")
        return
    if (
        candidate.sha256 != _sha256(candidate.patch)
        or candidate.diff_ref != _content_ref(candidate.patch)
        or candidate.diff_bytes != len(candidate.patch)
        or candidate.raw_diff_bytes < candidate.diff_bytes
    ):
        raise ValueError("DeepSWE candidate metadata does not match patch")
    expected_status = (
        CandidateCaptureStatus.CAPTURED
        if candidate.patch
        else CandidateCaptureStatus.EMPTY
    )
    if candidate.status != expected_status:
        raise ValueError("DeepSWE candidate status does not match patch")


def candidate_from_image_patch(
    patch: bytes,
    *,
    files_modified: tuple[str, ...],
    raw_diff_bytes: int,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
) -> DeepSWECandidate:
    """Content-address the exact bytes rendered by the task image.

    This creates candidate evidence only. It deliberately cannot issue replay
    authority: callers must apply these bytes to a distinct clean checkout and
    prove that the task image renders the applied workspace byte-for-byte
    identically before exposing hidden tests.
    """

    limit = validate_patch_byte_limit(max_bytes)
    rendered = bytes(patch)
    if len(rendered) > limit:
        raise ValueError(
            f"rendered candidate exceeds patch byte limit: {len(rendered)} > {limit}"
        )
    if (
        isinstance(raw_diff_bytes, bool)
        or not isinstance(raw_diff_bytes, int)
        or raw_diff_bytes < 0
    ):
        raise ValueError("raw_diff_bytes must be a non-negative integer")
    if not all(isinstance(path, str) for path in files_modified):
        raise ValueError("files_modified must contain only strings")
    candidate = DeepSWECandidate(
        patch=rendered,
        status=(
            CandidateCaptureStatus.CAPTURED
            if rendered
            else CandidateCaptureStatus.EMPTY
        ),
        diff_ref=_content_ref(rendered),
        sha256=_sha256(rendered),
        diff_bytes=len(rendered),
        files_modified=tuple(files_modified),
        raw_diff_bytes=max(raw_diff_bytes, len(rendered)),
    )
    validate_candidate(candidate)
    return candidate


def capture_source_candidate(
    workspace: Path,
    *,
    expected_base_commit: str,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
) -> DeepSWECandidate:
    """Capture the full canonical repository patch left by authoring.

    The underlying Assay capture uses an isolated index/object store and
    includes tracked and untracked files, binary patches, mode changes,
    deletions, and renames. No filename or directory-name classifier is used:
    DeepSWE repositories contain production files such as ``base_test.py`` and
    ``test_set.py``, and upstream grades the complete model patch. Tier must
    restore its own breach-test mutations before this security transition.
    """

    try:
        limit = validate_patch_byte_limit(max_bytes)
        workspace = workspace.resolve()
        root = subprocess.run(
            git_argv("rev-parse", "--show-toplevel"),
            cwd=str(workspace),
            env=sanitized_git_env(),
            capture_output=True,
            check=False,
            timeout=30,
        )
        if (
            root.returncode != 0
            or Path(os.fsdecode(root.stdout).strip()).resolve() != workspace
        ):
            raise GitCaptureError("DeepSWE workspace is not its Git root")
        patch, modified, base_commit = capture_workspace_delta(
            cwd=str(workspace),
            path_scope=None,
            max_diff_bytes=limit,
        )
        if len(patch) > limit:
            raise GitCaptureError(
                "DeepSWE candidate exceeds patch byte limit: "
                f"{len(patch)} > {limit}"
            )
        if base_commit != expected_base_commit:
            raise GitCaptureError(
                "DeepSWE workspace HEAD does not match immutable base: "
                f"{base_commit!r} != {expected_base_commit!r}"
            )

        status = (
            CandidateCaptureStatus.CAPTURED
            if patch
            else CandidateCaptureStatus.EMPTY
        )
        candidate = DeepSWECandidate(
            patch=patch,
            status=status,
            diff_ref=_content_ref(patch),
            sha256=_sha256(patch),
            diff_bytes=len(patch),
            files_modified=tuple(modified),
            stripped_protected_paths=(),
            raw_diff_bytes=len(patch),
        )
        validate_candidate(candidate)
        return candidate
    except (GitCaptureError, OSError, subprocess.SubprocessError, ValueError) as exc:
        return DeepSWECandidate.failed(f"{type(exc).__name__}: {exc}")


def prepare_candidate_replay(
    workspace: Path,
    candidate: DeepSWECandidate,
    *,
    expected_base_commit: str,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
) -> CandidateApplyReport:
    """Apply candidate bytes to a clean checkout without issuing proof.

    Host Git is used only to apply the patch. Because host and task-image Git
    can serialize an equivalent worktree differently, this function never
    compares a host-rendered diff and never returns ``PristineReplayStatus``.
    Authority is issued only by :func:`prove_image_replay` after a fresh
    no-tests task-image container renders the exact resulting workspace.
    """

    validate_candidate(candidate)
    limit = validate_patch_byte_limit(max_bytes)
    if candidate.diff_bytes > limit:
        return CandidateApplyReport(
            status=CandidateApplyStatus.PATCH_APPLY_FAILED,
            base_commit=expected_base_commit,
            candidate_sha256=candidate.sha256,
            error=(
                "candidate exceeds patch byte limit: "
                f"{candidate.diff_bytes} > {limit}"
            ),
        )
    workspace = workspace.resolve()
    try:
        observed_before, files_before, base_commit = capture_workspace_delta(
            cwd=str(workspace),
            path_scope=None,
            max_diff_bytes=limit,
        )
    except (GitCaptureError, OSError, subprocess.SubprocessError) as exc:
        return CandidateApplyReport(
            status=CandidateApplyStatus.BASE_INVALID,
            base_commit=expected_base_commit,
            candidate_sha256=candidate.sha256,
            error=f"{type(exc).__name__}: {exc}"[:2000],
        )
    if base_commit != expected_base_commit or observed_before or files_before:
        return CandidateApplyReport(
            status=CandidateApplyStatus.BASE_INVALID,
            base_commit=base_commit,
            candidate_sha256=candidate.sha256,
            error="pristine DeepSWE checkout is not clean at the expected base",
        )

    if candidate.patch:
        applied = subprocess.run(
            git_argv(
                "apply",
                "--whitespace=nowarn",
                "--binary",
                "-",
            ),
            cwd=str(workspace),
            env=sanitized_git_env(),
            input=candidate.patch,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if applied.returncode != 0:
            return CandidateApplyReport(
                status=CandidateApplyStatus.PATCH_APPLY_FAILED,
                base_commit=base_commit,
                candidate_sha256=candidate.sha256,
                error=applied.stderr.decode("utf-8", errors="replace")[-2000:],
            )

    return CandidateApplyReport(
        status=CandidateApplyStatus.APPLIED,
        base_commit=base_commit,
        candidate_sha256=candidate.sha256,
    )


def prove_image_replay(
    candidate: DeepSWECandidate,
    apply_report: CandidateApplyReport,
    rendered_patch: bytes,
    *,
    expected_base_commit: str,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
) -> PristineReplayReport:
    """Bind fresh task-image rendering to the exact candidate bytes.

    The caller must render ``rendered_patch`` from the same checkout left by
    :func:`prepare_candidate_replay`, in a fresh task-image container without
    hidden mounts, and strictly stop that container before using the report.
    """

    validate_candidate(candidate)
    limit = validate_patch_byte_limit(max_bytes)
    observed = bytes(rendered_patch)
    observed_sha256 = _sha256(observed)
    if (
        apply_report.status != CandidateApplyStatus.APPLIED
        or apply_report.base_commit != expected_base_commit
        or apply_report.candidate_sha256 != candidate.sha256
    ):
        status = (
            PristineReplayStatus.BASE_INVALID
            if apply_report.status == CandidateApplyStatus.BASE_INVALID
            else PristineReplayStatus.PATCH_APPLY_FAILED
        )
        return PristineReplayReport(
            status=status,
            base_commit=apply_report.base_commit,
            candidate_sha256=candidate.sha256,
            observed_sha256=observed_sha256,
            error=apply_report.error or "candidate apply report is not authoritative",
        )
    if len(observed) > limit:
        return PristineReplayReport(
            status=PristineReplayStatus.PATCH_MISMATCH,
            base_commit=apply_report.base_commit,
            candidate_sha256=candidate.sha256,
            observed_sha256=observed_sha256,
            error=(
                "image-rendered replay exceeds patch byte limit: "
                f"{len(observed)} > {limit}"
            ),
        )
    if observed != candidate.patch:
        return PristineReplayReport(
            status=PristineReplayStatus.PATCH_MISMATCH,
            base_commit=apply_report.base_commit,
            candidate_sha256=candidate.sha256,
            observed_sha256=observed_sha256,
            error="task-image replay rendering is not byte-identical to candidate",
        )
    return PristineReplayReport(
        status=PristineReplayStatus.PASSED,
        base_commit=apply_report.base_commit,
        candidate_sha256=candidate.sha256,
        observed_sha256=observed_sha256,
    )


def verify_hidden_model_patch(
    path: Path,
    candidate: DeepSWECandidate,
) -> VerifierPatchReport:
    """Prove that the verifier graded the exact replayed candidate bytes."""

    validate_candidate(candidate)
    artifact = read_bounded_regular_artifact(
        path.parent,
        path.name,
        max_bytes=candidate.diff_bytes,
    )
    if artifact.status == ArtifactReadStatus.MISSING:
        return VerifierPatchReport(
            status=VerifierPatchStatus.MISSING,
            candidate_sha256=candidate.sha256,
            error=f"verifier model.patch is missing: {path}",
        )
    if artifact.status != ArtifactReadStatus.OK:
        return VerifierPatchReport(
            status=VerifierPatchStatus.UNREADABLE,
            candidate_sha256=candidate.sha256,
            error=(
                "verifier model.patch was rejected: "
                f"state={artifact.status.value}, reason={artifact.error or 'unspecified'}"
            )[:2000],
        )
    observed = artifact.data
    digest = _sha256(observed)
    status = (
        VerifierPatchStatus.MATCHED
        if observed == candidate.patch
        else VerifierPatchStatus.MISMATCH
    )
    return VerifierPatchReport(
        status=status,
        candidate_sha256=candidate.sha256,
        observed_sha256=digest,
        observed_bytes=len(observed),
        error=(
            None
            if status == VerifierPatchStatus.MATCHED
            else "verifier model.patch differs from replayed candidate"
        ),
    )


def encode_candidate_envelope(
    candidate: DeepSWECandidate,
    *,
    task_id: str,
    ext_id: str,
    checkpoint: dict[str, Any],
    pristine_replay: PristineReplayReport | None,
    verifier_patch: VerifierPatchReport | None,
) -> bytes:
    """Canonical durable evidence envelope used by Corpus measurements."""

    validate_candidate(candidate)
    return json.dumps(
        {
            "schema_version": 1,
            "kind": "deepswe_candidate",
            "task_id": task_id,
            "ext_id": ext_id,
            "candidate": candidate.to_payload(),
            "checkpoint": checkpoint,
            "pristine_replay": (
                pristine_replay.to_payload() if pristine_replay is not None else None
            ),
            "verifier_patch": (
                verifier_patch.to_payload() if verifier_patch is not None else None
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


__all__ = [
    "CandidateApplyReport",
    "CandidateApplyStatus",
    "CandidateCaptureStatus",
    "DeepSWECandidate",
    "PristineReplayReport",
    "PristineReplayStatus",
    "VerifierPatchReport",
    "VerifierPatchStatus",
    "candidate_from_image_patch",
    "capture_source_candidate",
    "encode_candidate_envelope",
    "prepare_candidate_replay",
    "prove_image_replay",
    "validate_candidate",
    "verify_hidden_model_patch",
]
