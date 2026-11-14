# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durable, candidate-bound evidence for deterministic and child gates.

The workflow result directories are useful operational records, but they are
not part of a terminal :class:`assay.types.artifacts.Implementation`'s
content-addressed authority.  This module projects the small portion needed by
resume and composition into strict, canonical artifacts:

* ``VerificationEvidence`` binds build/verify results to the exact candidate
  diff, executable Spec authority, slice, and runner output artifacts.
* ``HandoffAuthoringFinalizationEvidence`` binds an output-producing Handoff
  gate and its exact materialized digests to that candidate and verification.
* ``HandoffGateLedger`` retains each requested Breach/Review outcome as the
  existing typed ``WorkflowOutcome`` instead of reducing it to an exit code.

These codecs reject unknown fields and non-canonical JSON. A caller therefore
cannot silently extend either authority surface without a deliberate schema
revision and parser change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from assay.canonical import hash_record, parse, serialize
from assay.types.identity import ContentHash

from ._build_verify import BuildVerifyOutcome, RunnerResult
from ._outcome import TerminalClass, WorkflowOutcome

if TYPE_CHECKING:
    from assay.corpus.store import Corpus
    from assay.types.artifacts import Implementation


HANDOFF_VERIFICATION_ARTIFACT = "handoff_verification_evidence"
HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT = (
    "handoff_authoring_finalization_evidence"
)
CONDITIONAL_ABSENCE_ARTIFACT = "conditional_absence_evidence"
DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT = "declared_test_evidence_pending"
HANDOFF_GATE_LEDGER_ARTIFACT = "handoff_gate_ledger"
TIER_VERIFICATION_ARTIFACT = "tier_verification_evidence"
TIER_VERIFICATION_MANIFEST_ARTIFACT = "tier_verification_manifest"

_VERIFICATION_SCHEMA = "crucible.digest-bound-verification-evidence.v4"
_CONDITIONAL_ABSENCE_SCHEMA = (
    "crucible.digest-bound-conditional-absence-evidence.v2"
)
_HANDOFF_GATE_LEDGER_SCHEMA = "crucible.handoff-gate-ledger.v1"
_HANDOFF_AUTHORIZING_FINALIZATION_SCHEMA = (
    "crucible.handoff-authoring-finalization-evidence.v1"
)
_TIER_VERIFICATION_MANIFEST_SCHEMA = (
    "crucible.tier-verification-manifest.v1"
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VERIFICATION_DISPOSITIONS = frozenset({
    "passed",
    "failed",
    "unavailable",
    "waived",
})
_GATE_TERMINALS = frozenset({
    *(item.value for item in TerminalClass),
    "skipped",
})


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be 'sha256:' plus 64 lowercase hexadecimal digits"
        )
    return value


def _plain_string(value: object, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        suffix = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{label} must be a {suffix}")
    return value


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _plain_string(value, label=label, allow_empty=True)


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _canonical_payload(blob: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = parse(blob)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if serialize(payload) != blob:
        raise ValueError(f"{label} must use canonical JSON encoding")
    return payload


def _repo_path(value: object, *, label: str) -> str:
    path = _plain_string(value, label=label)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or "\\" in path or ".." in parsed.parts:
        raise ValueError(f"{label} must be a normalized repository-relative path")
    if str(parsed) != path or path == ".":
        raise ValueError(f"{label} must be a normalized repository-relative path")
    return path


def bound_workspace_path_absent(
    root_descriptor: int,
    relative_path: str,
) -> bool:
    """Prove one declared path absent beneath an exact workspace root.

    Existing ancestors must be real directories reached without following a
    symlink. A missing ancestor or leaf is rechecked through a fresh walk from
    the held root descriptor so a detached directory cannot masquerade as the
    live workspace namespace.
    """

    normalized = _repo_path(
        relative_path,
        label="conditional output path",
    )
    components = PurePosixPath(normalized).parts
    descriptor = os.dup(root_descriptor)
    ancestor_identities: list[tuple[str, tuple[int, int, int, int]]] = []
    missing_component: str | None = None
    try:
        for index, component in enumerate(components):
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                missing_component = component
                break
            if index == len(components) - 1:
                return False
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "conditional output ancestor is not a real directory"
                )
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(child)
            identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
            )
            if identity != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
            ):
                os.close(child)
                raise ValueError(
                    "conditional output ancestor changed while opening"
                )
            ancestor_identities.append((component, identity))
            os.close(descriptor)
            descriptor = child

        assert missing_component is not None
        validation_descriptor = os.dup(root_descriptor)
        try:
            for component, expected_identity in ancestor_identities:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=validation_descriptor,
                )
                opened = os.fstat(child)
                identity = (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_nlink,
                )
                if identity != expected_identity:
                    os.close(child)
                    raise ValueError(
                        "conditional output ancestor changed during proof"
                    )
                os.close(validation_descriptor)
                validation_descriptor = child
            try:
                os.stat(
                    missing_component,
                    dir_fd=validation_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return True
            return False
        finally:
            os.close(validation_descriptor)
    finally:
        os.close(descriptor)


def _json_scalar(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{label} must be a finite JSON scalar")


def _resolve_json_pointer(document: object, pointer: str) -> object:
    """Resolve one strict RFC 6901 pointer, rejecting malformed/ambiguous paths."""

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = document
    for raw_token in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw_token):
            raise ValueError("JSON pointer contains an invalid escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValueError("JSON pointer does not resolve")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ValueError("JSON pointer list token must be a canonical index")
            index = int(token)
            if index >= len(current):
                raise ValueError("JSON pointer does not resolve")
            current = current[index]
        else:
            raise ValueError("JSON pointer traverses a scalar")
    return current


def _strict_json_document(blob: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"conditional selector has duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"conditional selector contains non-finite value {value}")

    try:
        return json.loads(
            blob,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("conditional selector must be valid UTF-8 JSON") from exc


def validate_conditional_selector(
    selector_bytes: bytes,
    *,
    json_pointer: str,
    expected_value: object,
) -> None:
    """Require one unambiguous scalar selection from exact JSON bytes."""

    expected = _json_scalar(expected_value, label="conditional expected value")
    selected = _resolve_json_pointer(
        _strict_json_document(selector_bytes), json_pointer
    )
    if not isinstance(selected, (str, bool, int, float)) and selected is not None:
        raise ValueError("conditional selector must resolve to a JSON scalar")
    if selected != expected or type(selected) is not type(expected):
        raise ValueError("conditional selector value does not equal expected value")


@dataclass(frozen=True, slots=True)
class RunnerEvidence:
    """Content-addressed projection of one build/verify subprocess result."""

    role: str
    runner: str
    returncode: int
    duration_s: float
    values: dict[str, float]
    raw_output_ref: str
    stderr_ref: str

    def __post_init__(self) -> None:
        _plain_string(self.role, label="RunnerEvidence.role")
        _plain_string(self.runner, label="RunnerEvidence.runner")
        _exact_int(self.returncode, label="RunnerEvidence.returncode")
        _finite_nonnegative(self.duration_s, label="RunnerEvidence.duration_s")
        if not isinstance(self.values, dict):
            raise TypeError("RunnerEvidence.values must be a dict")
        normalized: dict[str, float] = {}
        for key, value in self.values.items():
            _plain_string(key, label="RunnerEvidence.values key")
            normalized[key] = _finite_number(
                value,
                label=f"RunnerEvidence.values[{key!r}]",
            )
        object.__setattr__(self, "values", normalized)
        _content_hash(self.raw_output_ref, label="RunnerEvidence.raw_output_ref")
        _content_hash(self.stderr_ref, label="RunnerEvidence.stderr_ref")

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "runner": self.runner,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "values": dict(self.values),
            "raw_output_ref": self.raw_output_ref,
            "stderr_ref": self.stderr_ref,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RunnerEvidence":
        fields = {
            "role",
            "runner",
            "returncode",
            "duration_s",
            "values",
            "raw_output_ref",
            "stderr_ref",
        }
        _exact_fields(value, frozenset(fields), label="RunnerEvidence")
        raw_values = value["values"]
        if not isinstance(raw_values, dict):
            raise ValueError("RunnerEvidence.values must be an object")
        return cls(
            role=_plain_string(value["role"], label="RunnerEvidence.role"),
            runner=_plain_string(value["runner"], label="RunnerEvidence.runner"),
            returncode=_exact_int(
                value["returncode"], label="RunnerEvidence.returncode"
            ),
            duration_s=_finite_nonnegative(
                value["duration_s"], label="RunnerEvidence.duration_s"
            ),
            values=dict(raw_values),
            raw_output_ref=_content_hash(
                value["raw_output_ref"], label="RunnerEvidence.raw_output_ref"
            ),
            stderr_ref=_content_hash(
                value["stderr_ref"], label="RunnerEvidence.stderr_ref"
            ),
        )


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Exact deterministic gate evidence for one candidate postimage."""

    schema_version: str
    tier_id: str
    slice_id: str
    spec_execution_authority_hash: str
    candidate_diff_sha256: str
    base_commit: str
    files_modified: tuple[str, ...]
    gate: str
    classification: str
    disposition: str
    build_timeout_s: float | None
    verify_timeout_s: float | None
    build: RunnerEvidence | None
    verify: RunnerEvidence | None
    dev_verify: RunnerEvidence | None

    def __post_init__(self) -> None:
        if self.schema_version != _VERIFICATION_SCHEMA:
            raise ValueError("unsupported VerificationEvidence schema")
        _plain_string(self.tier_id, label="VerificationEvidence.tier_id")
        _plain_string(self.slice_id, label="VerificationEvidence.slice_id")
        _content_hash(
            self.spec_execution_authority_hash,
            label="VerificationEvidence.spec_execution_authority_hash",
        )
        _content_hash(
            self.candidate_diff_sha256,
            label="VerificationEvidence.candidate_diff_sha256",
        )
        _plain_string(self.base_commit, label="VerificationEvidence.base_commit")
        if not isinstance(self.files_modified, tuple) or any(
            not isinstance(item, str) or not item for item in self.files_modified
        ):
            raise ValueError(
                "VerificationEvidence.files_modified must be a non-empty-string tuple"
            )
        if len(set(self.files_modified)) != len(self.files_modified):
            raise ValueError("VerificationEvidence.files_modified has duplicates")
        if self.gate not in {"verify", "dev_verify"}:
            raise ValueError("VerificationEvidence.gate is unsupported")
        _plain_string(
            self.classification,
            label="VerificationEvidence.classification",
        )
        if self.disposition not in _VERIFICATION_DISPOSITIONS:
            raise ValueError("VerificationEvidence.disposition is unsupported")
        for label, timeout in (
            ("build_timeout_s", self.build_timeout_s),
            ("verify_timeout_s", self.verify_timeout_s),
        ):
            if timeout is not None:
                _finite_nonnegative(timeout, label=f"VerificationEvidence.{label}")
        if self.disposition == "passed" and not self.classification.startswith(
            "success"
        ):
            raise ValueError("passed verification requires a success classification")
        if self.disposition == "passed":
            if self.gate == "verify":
                if (
                    self.build is None
                    or self.build.role != "build"
                    or self.build.returncode != 0
                    or self.dev_verify is not None
                    or (
                        self.verify is not None
                        and (
                            self.verify.role != "verify"
                            or self.verify.returncode != 0
                            or self.verify.values.get("tests_failed", 0.0) > 0
                        )
                    )
                ):
                    raise ValueError(
                        "passed verify evidence requires a successful build "
                        "and any configured verify runner"
                    )
            elif (
                self.build is not None
                or self.verify is not None
                or self.dev_verify is None
                or self.dev_verify.role != "dev_verify"
                or self.dev_verify.returncode != 0
                or self.dev_verify.values.get("tests_failed", 0.0) > 0
            ):
                raise ValueError(
                    "passed dev_verify evidence requires its successful runner"
                )
        if self.disposition == "waived" and self.classification != (
            "skipped_by_policy"
        ):
            raise ValueError("waived verification requires skipped_by_policy")
        if self.disposition == "unavailable" and self.classification != (
            "runner_missing"
        ):
            raise ValueError("unavailable verification requires runner_missing")

    @property
    def satisfies_required_gate(self) -> bool:
        """Whether this exact Spec admitted the candidate without a waiver."""

        return self.disposition == "passed"

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "tier_id": self.tier_id,
            "slice_id": self.slice_id,
            "spec_execution_authority_hash": (
                self.spec_execution_authority_hash
            ),
            "candidate_diff_sha256": self.candidate_diff_sha256,
            "base_commit": self.base_commit,
            "files_modified": list(self.files_modified),
            "gate": self.gate,
            "classification": self.classification,
            "disposition": self.disposition,
            "build_timeout_s": self.build_timeout_s,
            "verify_timeout_s": self.verify_timeout_s,
            "build": (
                self.build.to_payload()
                if self.build is not None
                else None
            ),
            "verify": (
                self.verify.to_payload()
                if self.verify is not None
                else None
            ),
            "dev_verify": (
                self.dev_verify.to_payload()
                if self.dev_verify is not None
                else None
            ),
        }
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VerificationEvidence":
        schema_version = _plain_string(
            value.get("schema_version"),
            label="VerificationEvidence.schema_version",
        )
        if schema_version != _VERIFICATION_SCHEMA:
            raise ValueError("unsupported VerificationEvidence schema")
        fields = {
            "schema_version",
            "tier_id",
            "slice_id",
            "spec_execution_authority_hash",
            "candidate_diff_sha256",
            "base_commit",
            "files_modified",
            "gate",
            "classification",
            "disposition",
            "build_timeout_s",
            "verify_timeout_s",
            "build",
            "verify",
            "dev_verify",
        }
        _exact_fields(value, frozenset(fields), label="VerificationEvidence")
        raw_files = value["files_modified"]
        if not isinstance(raw_files, list) or not all(
            isinstance(item, str) for item in raw_files
        ):
            raise ValueError("VerificationEvidence.files_modified must be a list")
        def runner(field_name: str) -> RunnerEvidence | None:
            raw = value[field_name]
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise ValueError(
                    f"VerificationEvidence.{field_name} must be an object or null"
                )
            return RunnerEvidence.from_mapping(raw)

        def timeout(field_name: str) -> float | None:
            raw = value[field_name]
            if raw is None:
                return None
            return _finite_nonnegative(
                raw,
                label=f"VerificationEvidence.{field_name}",
            )

        return cls(
            schema_version=schema_version,
            tier_id=_plain_string(
                value["tier_id"], label="VerificationEvidence.tier_id"
            ),
            slice_id=_plain_string(
                value["slice_id"], label="VerificationEvidence.slice_id"
            ),
            spec_execution_authority_hash=_content_hash(
                value["spec_execution_authority_hash"],
                label="VerificationEvidence.spec_execution_authority_hash",
            ),
            candidate_diff_sha256=_content_hash(
                value["candidate_diff_sha256"],
                label="VerificationEvidence.candidate_diff_sha256",
            ),
            base_commit=_plain_string(
                value["base_commit"], label="VerificationEvidence.base_commit"
            ),
            files_modified=tuple(raw_files),
            gate=_plain_string(value["gate"], label="VerificationEvidence.gate"),
            classification=_plain_string(
                value["classification"],
                label="VerificationEvidence.classification",
            ),
            disposition=_plain_string(
                value["disposition"],
                label="VerificationEvidence.disposition",
            ),
            build_timeout_s=timeout("build_timeout_s"),
            verify_timeout_s=timeout("verify_timeout_s"),
            build=runner("build"),
            verify=runner("verify"),
            dev_verify=runner("dev_verify"),
        )


@dataclass(frozen=True, slots=True)
class HandoffAuthoringFinalizationEvidence:
    """Durable proof that Handoff ran one exact output-producing gate."""

    schema_version: str
    tier_id: str
    slice_id: str
    spec_execution_authority_hash: str
    candidate_diff_sha256: str
    base_commit: str
    files_modified: tuple[str, ...]
    gate: str
    output_sha256: tuple[tuple[str, str], ...]
    transcript_ref: str
    verification_evidence_ref: str

    def __post_init__(self) -> None:
        if self.schema_version != _HANDOFF_AUTHORIZING_FINALIZATION_SCHEMA:
            raise ValueError(
                "unsupported Handoff authoring-finalization evidence schema"
            )
        for field_name in ("tier_id", "slice_id", "base_commit"):
            _plain_string(
                getattr(self, field_name),
                label=f"HandoffAuthoringFinalizationEvidence.{field_name}",
            )
        for field_name in (
            "spec_execution_authority_hash",
            "candidate_diff_sha256",
            "transcript_ref",
            "verification_evidence_ref",
        ):
            _content_hash(
                getattr(self, field_name),
                label=f"HandoffAuthoringFinalizationEvidence.{field_name}",
            )
        if self.gate not in {"verify", "dev_verify"}:
            raise ValueError(
                "Handoff authoring-finalization evidence gate is unsupported"
            )
        if not isinstance(self.files_modified, tuple) or any(
            not isinstance(path, str) or not path
            for path in self.files_modified
        ):
            raise ValueError(
                "Handoff authoring-finalization files_modified must be a "
                "non-empty-string tuple"
            )
        if len(set(self.files_modified)) != len(self.files_modified):
            raise ValueError(
                "Handoff authoring-finalization files_modified has duplicates"
            )
        if not isinstance(self.output_sha256, tuple) or not self.output_sha256:
            raise ValueError(
                "Handoff authoring-finalization output_sha256 must be non-empty"
            )
        paths: list[str] = []
        for index, item in enumerate(self.output_sha256):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "Handoff authoring-finalization output_sha256 entries "
                    "must be two-string tuples"
                )
            path, digest = item
            paths.append(_repo_path(
                path,
                label=(
                    "HandoffAuthoringFinalizationEvidence.output_sha256"
                    f"[{index}].path"
                ),
            ))
            _content_hash(
                digest,
                label=(
                    "HandoffAuthoringFinalizationEvidence.output_sha256"
                    f"[{index}].sha256"
                ),
            )
        if len(set(paths)) != len(paths):
            raise ValueError(
                "Handoff authoring-finalization output_sha256 has duplicates"
            )
        if any(path not in self.files_modified for path in paths):
            raise ValueError(
                "Handoff authoring-finalization outputs must be candidate files"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tier_id": self.tier_id,
            "slice_id": self.slice_id,
            "spec_execution_authority_hash": (
                self.spec_execution_authority_hash
            ),
            "candidate_diff_sha256": self.candidate_diff_sha256,
            "base_commit": self.base_commit,
            "files_modified": list(self.files_modified),
            "gate": self.gate,
            "output_sha256": [
                {"path": path, "sha256": digest}
                for path, digest in self.output_sha256
            ],
            "transcript_ref": self.transcript_ref,
            "verification_evidence_ref": self.verification_evidence_ref,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "HandoffAuthoringFinalizationEvidence":
        fields = frozenset({
            "schema_version",
            "tier_id",
            "slice_id",
            "spec_execution_authority_hash",
            "candidate_diff_sha256",
            "base_commit",
            "files_modified",
            "gate",
            "output_sha256",
            "transcript_ref",
            "verification_evidence_ref",
        })
        _exact_fields(
            value,
            fields,
            label="HandoffAuthoringFinalizationEvidence",
        )
        raw_files = value["files_modified"]
        if not isinstance(raw_files, list) or not all(
            isinstance(path, str) for path in raw_files
        ):
            raise ValueError(
                "Handoff authoring-finalization files_modified must be a list"
            )
        raw_outputs = value["output_sha256"]
        if not isinstance(raw_outputs, list):
            raise ValueError(
                "Handoff authoring-finalization output_sha256 must be a list"
            )
        outputs: list[tuple[str, str]] = []
        for index, item in enumerate(raw_outputs):
            if not isinstance(item, Mapping):
                raise ValueError(
                    "Handoff authoring-finalization output entry must be an "
                    "object"
                )
            _exact_fields(
                item,
                frozenset({"path", "sha256"}),
                label=f"output_sha256[{index}]",
            )
            outputs.append((
                _repo_path(item["path"], label=f"output_sha256[{index}].path"),
                _content_hash(
                    item["sha256"],
                    label=f"output_sha256[{index}].sha256",
                ),
            ))
        return cls(
            schema_version=_plain_string(
                value["schema_version"], label="schema_version"
            ),
            tier_id=_plain_string(value["tier_id"], label="tier_id"),
            slice_id=_plain_string(value["slice_id"], label="slice_id"),
            spec_execution_authority_hash=_content_hash(
                value["spec_execution_authority_hash"],
                label="spec_execution_authority_hash",
            ),
            candidate_diff_sha256=_content_hash(
                value["candidate_diff_sha256"],
                label="candidate_diff_sha256",
            ),
            base_commit=_plain_string(value["base_commit"], label="base_commit"),
            files_modified=tuple(raw_files),
            gate=_plain_string(value["gate"], label="gate"),
            output_sha256=tuple(outputs),
            transcript_ref=_content_hash(
                value["transcript_ref"], label="transcript_ref"
            ),
            verification_evidence_ref=_content_hash(
                value["verification_evidence_ref"],
                label="verification_evidence_ref",
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionalAbsenceEvidence:
    """Required-pass proof that a plan-selected sibling excludes this slice."""

    schema_version: str
    tier_id: str
    slice_id: str
    spec_execution_authority_hash: str
    candidate_diff_sha256: str
    base_commit: str
    files_modified: tuple[str, ...]
    creates_paths: tuple[str, ...]
    selector_path: str
    selector_sha256: str
    json_pointer: str
    expected_value: object
    decision_slice_id: str
    decision_implementation_sha256: str
    selected_slice_id: str
    selected_implementation_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _CONDITIONAL_ABSENCE_SCHEMA:
            raise ValueError("unsupported ConditionalAbsenceEvidence schema")
        for field_name in ("tier_id", "slice_id", "decision_slice_id", "selected_slice_id"):
            _plain_string(
                getattr(self, field_name),
                label=f"ConditionalAbsenceEvidence.{field_name}",
            )
        for field_name in (
            "spec_execution_authority_hash",
            "candidate_diff_sha256",
            "selector_sha256",
            "decision_implementation_sha256",
            "selected_implementation_sha256",
        ):
            _content_hash(
                getattr(self, field_name),
                label=f"ConditionalAbsenceEvidence.{field_name}",
            )
        _plain_string(self.base_commit, label="ConditionalAbsenceEvidence.base_commit")
        if self.files_modified != ():
            raise ValueError("conditional absence requires an empty files_modified tuple")
        if not isinstance(self.creates_paths, tuple) or not self.creates_paths:
            raise ValueError("conditional absence requires declared creates_paths")
        if len(set(self.creates_paths)) != len(self.creates_paths):
            raise ValueError("ConditionalAbsenceEvidence.creates_paths has duplicates")
        for path in self.creates_paths:
            _repo_path(path, label="ConditionalAbsenceEvidence.creates_paths item")
        _repo_path(self.selector_path, label="ConditionalAbsenceEvidence.selector_path")
        _plain_string(
            self.json_pointer,
            label="ConditionalAbsenceEvidence.json_pointer",
            allow_empty=True,
        )
        _json_scalar(
            self.expected_value,
            label="ConditionalAbsenceEvidence.expected_value",
        )
        if self.slice_id in {self.decision_slice_id, self.selected_slice_id}:
            raise ValueError("conditional and authority slices must be distinct")
        if self.decision_slice_id == self.selected_slice_id:
            raise ValueError("decision and selected slices must be distinct")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tier_id": self.tier_id,
            "slice_id": self.slice_id,
            "spec_execution_authority_hash": self.spec_execution_authority_hash,
            "candidate_diff_sha256": self.candidate_diff_sha256,
            "base_commit": self.base_commit,
            "files_modified": list(self.files_modified),
            "creates_paths": list(self.creates_paths),
            "selector_path": self.selector_path,
            "selector_sha256": self.selector_sha256,
            "json_pointer": self.json_pointer,
            "expected_value": self.expected_value,
            "decision_slice_id": self.decision_slice_id,
            "decision_implementation_sha256": self.decision_implementation_sha256,
            "selected_slice_id": self.selected_slice_id,
            "selected_implementation_sha256": self.selected_implementation_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConditionalAbsenceEvidence":
        fields = frozenset({
            "schema_version", "tier_id", "slice_id",
            "spec_execution_authority_hash", "candidate_diff_sha256",
            "base_commit", "files_modified", "creates_paths", "selector_path",
            "selector_sha256", "json_pointer", "expected_value",
            "decision_slice_id", "decision_implementation_sha256",
            "selected_slice_id", "selected_implementation_sha256",
        })
        _exact_fields(value, fields, label="ConditionalAbsenceEvidence")
        raw_files = value["files_modified"]
        raw_creates = value["creates_paths"]
        if not isinstance(raw_files, list) or not all(isinstance(x, str) for x in raw_files):
            raise ValueError("ConditionalAbsenceEvidence.files_modified must be a list")
        if not isinstance(raw_creates, list) or not all(isinstance(x, str) for x in raw_creates):
            raise ValueError("ConditionalAbsenceEvidence.creates_paths must be a list")
        return cls(
            schema_version=_plain_string(value["schema_version"], label="schema_version"),
            tier_id=_plain_string(value["tier_id"], label="tier_id"),
            slice_id=_plain_string(value["slice_id"], label="slice_id"),
            spec_execution_authority_hash=_content_hash(value["spec_execution_authority_hash"], label="spec_execution_authority_hash"),
            candidate_diff_sha256=_content_hash(value["candidate_diff_sha256"], label="candidate_diff_sha256"),
            base_commit=_plain_string(value["base_commit"], label="base_commit"),
            files_modified=tuple(raw_files),
            creates_paths=tuple(raw_creates),
            selector_path=_repo_path(value["selector_path"], label="selector_path"),
            selector_sha256=_content_hash(value["selector_sha256"], label="selector_sha256"),
            json_pointer=_plain_string(value["json_pointer"], label="json_pointer", allow_empty=True),
            expected_value=_json_scalar(value["expected_value"], label="expected_value"),
            decision_slice_id=_plain_string(value["decision_slice_id"], label="decision_slice_id"),
            decision_implementation_sha256=_content_hash(value["decision_implementation_sha256"], label="decision_implementation_sha256"),
            selected_slice_id=_plain_string(value["selected_slice_id"], label="selected_slice_id"),
            selected_implementation_sha256=_content_hash(value["selected_implementation_sha256"], label="selected_implementation_sha256"),
        )


def _persist_runner_evidence(
    corpus: "Corpus",
    result: RunnerResult | None,
    *,
    name_hint: str,
) -> RunnerEvidence | None:
    if result is None:
        return None
    raw_ref = corpus.put_artifact(
        result.raw_output,
        kind="verification_logs",
        name_hint=f"{name_hint}_{result.role}.stdout",
    )
    stderr_ref = corpus.put_artifact(
        result.stderr,
        kind="verification_logs",
        name_hint=f"{name_hint}_{result.role}.stderr",
    )
    return RunnerEvidence(
        role=result.role,
        runner=result.runner,
        returncode=result.returncode,
        duration_s=result.duration_s,
        values=dict(result.values),
        raw_output_ref=str(raw_ref),
        stderr_ref=str(stderr_ref),
    )


def persist_verification_evidence(
    corpus: "Corpus",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    candidate_diff_ref: ContentHash,
    base_commit: str,
    files_modified: tuple[str, ...],
    outcome: BuildVerifyOutcome | None,
    skipped_by_policy: bool,
    build_timeout_s: float | None,
    verify_timeout_s: float | None,
    gate: str = "verify",
    name_hint: str | None = None,
) -> ContentHash:
    """Persist and return one strict verification-evidence artifact."""

    if skipped_by_policy:
        if outcome is not None:
            raise ValueError("skipped verification cannot carry a runner outcome")
        classification = "skipped_by_policy"
        disposition = "waived"
    else:
        if outcome is None:
            raise ValueError("required verification must carry a runner outcome")
        classification = outcome.classification
        disposition = (
            "passed"
            if classification.startswith("success")
            else "unavailable"
            if classification == "runner_missing"
            else "failed"
        )
    artifact_hint = name_hint or f"{tier_id}_{slice_id}"
    evidence = VerificationEvidence(
        schema_version=_VERIFICATION_SCHEMA,
        tier_id=tier_id,
        slice_id=slice_id,
        spec_execution_authority_hash=spec_execution_authority_hash,
        candidate_diff_sha256=str(candidate_diff_ref),
        base_commit=base_commit,
        files_modified=files_modified,
        gate=gate,
        classification=classification,
        disposition=disposition,
        build_timeout_s=build_timeout_s,
        verify_timeout_s=verify_timeout_s,
        build=(
            _persist_runner_evidence(
                corpus, outcome.build, name_hint=artifact_hint
            )
            if outcome is not None
            else None
        ),
        verify=(
            _persist_runner_evidence(
                corpus, outcome.verify, name_hint=artifact_hint
            )
            if outcome is not None
            else None
        ),
        dev_verify=(
            _persist_runner_evidence(
                corpus, outcome.dev_verify, name_hint=artifact_hint
            )
            if outcome is not None
            else None
        ),
    )
    encoded = serialize(evidence.to_payload())
    # Exercise the strict decoder before publishing a resumability authority.
    VerificationEvidence.from_mapping(_canonical_payload(
        encoded,
        label="generated verification evidence",
    ))
    return corpus.put_artifact(
        encoded,
        kind="verification_evidence",
        name_hint=f"{artifact_hint}.json",
    )


def load_verification_evidence(
    corpus: "Corpus",
    ref: str,
) -> VerificationEvidence:
    _content_hash(ref, label="verification evidence ref")
    blob = corpus.get_artifact(ContentHash(ref))
    return VerificationEvidence.from_mapping(
        _canonical_payload(blob, label="verification evidence")
    )


def persist_handoff_authoring_finalization_evidence(
    corpus: "Corpus",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    candidate_diff_ref: ContentHash,
    base_commit: str,
    files_modified: tuple[str, ...],
    gate: str,
    output_sha256: tuple[tuple[str, str], ...],
    transcript_ref: ContentHash,
    verification_evidence_ref: ContentHash,
) -> ContentHash:
    """Persist one strict Handoff output-producing-gate proof."""

    evidence = HandoffAuthoringFinalizationEvidence(
        schema_version=_HANDOFF_AUTHORIZING_FINALIZATION_SCHEMA,
        tier_id=tier_id,
        slice_id=slice_id,
        spec_execution_authority_hash=spec_execution_authority_hash,
        candidate_diff_sha256=str(candidate_diff_ref),
        base_commit=base_commit,
        files_modified=files_modified,
        gate=gate,
        output_sha256=output_sha256,
        transcript_ref=str(transcript_ref),
        verification_evidence_ref=str(verification_evidence_ref),
    )
    encoded = serialize(evidence.to_payload())
    HandoffAuthoringFinalizationEvidence.from_mapping(_canonical_payload(
        encoded,
        label="generated Handoff authoring-finalization evidence",
    ))
    return corpus.put_artifact(
        encoded,
        kind="verification_evidence",
        name_hint=f"handoff_{tier_id}_{slice_id}_finalization.json",
    )


def load_handoff_authoring_finalization_evidence(
    corpus: "Corpus",
    ref: str,
) -> HandoffAuthoringFinalizationEvidence:
    _content_hash(ref, label="Handoff authoring-finalization evidence ref")
    blob = corpus.get_artifact(ContentHash(ref))
    return HandoffAuthoringFinalizationEvidence.from_mapping(
        _canonical_payload(
            blob,
            label="Handoff authoring-finalization evidence",
        )
    )


def verification_evidence_content_available(
    corpus: "Corpus",
    evidence: VerificationEvidence,
) -> bool:
    """Resolve every content-addressed byte claimed by gate evidence."""

    try:
        corpus.get_artifact(ContentHash(evidence.candidate_diff_sha256))
        for runner in (evidence.build, evidence.verify, evidence.dev_verify):
            if runner is None:
                continue
            corpus.get_artifact(ContentHash(runner.raw_output_ref))
            corpus.get_artifact(ContentHash(runner.stderr_ref))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def verification_matches_implementation(
    corpus: "Corpus",
    implementation: "Implementation",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    require_gate: bool,
    expected_gate: str = "verify",
) -> bool:
    """Validate a Handoff record's exact verification/resume authority."""

    if expected_gate not in {"verify", "dev_verify"}:
        return False

    ref = (implementation.artifacts or {}).get(HANDOFF_VERIFICATION_ARTIFACT)
    if ref is None:
        return not require_gate
    try:
        evidence = load_verification_evidence(corpus, str(ref))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    if not verification_evidence_content_available(corpus, evidence):
        return False
    if (
        evidence.tier_id != tier_id
        or evidence.slice_id != slice_id
        or evidence.spec_execution_authority_hash
        != spec_execution_authority_hash
        or evidence.candidate_diff_sha256 != str(implementation.diff_ref)
        or evidence.base_commit != implementation.base_commit
        or evidence.files_modified != implementation.files_modified
        or evidence.gate != expected_gate
    ):
        return False
    return not require_gate or evidence.satisfies_required_gate


def handoff_authoring_finalization_matches(
    corpus: "Corpus",
    implementation: "Implementation",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    expected_gate: str,
    expected_output_paths: tuple[str, ...],
) -> bool:
    """Match the durable exact-output proof to a Handoff candidate."""

    artifacts = implementation.artifacts or {}
    ref = artifacts.get(HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT)
    verification_ref = artifacts.get(HANDOFF_VERIFICATION_ARTIFACT)
    if ref is None or verification_ref is None:
        return False
    try:
        evidence = load_handoff_authoring_finalization_evidence(corpus, str(ref))
        corpus.get_artifact(ContentHash(evidence.candidate_diff_sha256))
        corpus.get_artifact(ContentHash(evidence.transcript_ref))
        verification = load_verification_evidence(
            corpus,
            evidence.verification_evidence_ref,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    if (
        evidence.tier_id != tier_id
        or evidence.slice_id != slice_id
        or evidence.spec_execution_authority_hash
        != spec_execution_authority_hash
        or evidence.candidate_diff_sha256 != str(implementation.diff_ref)
        or evidence.base_commit != implementation.base_commit
        or evidence.files_modified != implementation.files_modified
        or evidence.gate != expected_gate
        or tuple(path for path, _digest in evidence.output_sha256)
        != expected_output_paths
        or evidence.transcript_ref != str(implementation.transcript_ref)
        or evidence.verification_evidence_ref != str(verification_ref)
        or verification.tier_id != evidence.tier_id
        or verification.slice_id != evidence.slice_id
        or verification.spec_execution_authority_hash
        != evidence.spec_execution_authority_hash
        or verification.candidate_diff_sha256
        != evidence.candidate_diff_sha256
        or verification.base_commit != evidence.base_commit
        or verification.files_modified != evidence.files_modified
        or verification.gate != evidence.gate
        or not verification_evidence_content_available(corpus, verification)
    ):
        return False
    return True


def implementation_projection_hash(implementation: "Implementation") -> str:
    """Hash one terminal Implementation without its cyclic Tier manifest ref."""

    artifacts = dict(implementation.artifacts or {})
    artifacts.pop(TIER_VERIFICATION_MANIFEST_ARTIFACT, None)
    return str(hash_record(replace(implementation, artifacts=artifacts)))


def persist_conditional_absence_evidence(
    corpus: "Corpus",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    candidate_diff_ref: ContentHash,
    base_commit: str,
    files_modified: tuple[str, ...],
    creates_paths: tuple[str, ...],
    selector_path: str,
    selector_bytes: bytes,
    json_pointer: str,
    expected_value: object,
    decision_slice_id: str,
    decision_implementation: "Implementation",
    selected_slice_id: str,
    selected_implementation: "Implementation",
) -> ContentHash:
    """Persist strict evidence for an explicitly unselected empty slice."""

    validate_conditional_selector(
        selector_bytes,
        json_pointer=json_pointer,
        expected_value=expected_value,
    )
    selector_ref = corpus.put_artifact(
        selector_bytes,
        kind="conditional_selectors",
        name_hint=PurePosixPath(selector_path).name,
    )
    evidence = ConditionalAbsenceEvidence(
        schema_version=_CONDITIONAL_ABSENCE_SCHEMA,
        tier_id=tier_id,
        slice_id=slice_id,
        spec_execution_authority_hash=spec_execution_authority_hash,
        candidate_diff_sha256=str(candidate_diff_ref),
        base_commit=base_commit,
        files_modified=files_modified,
        creates_paths=creates_paths,
        selector_path=selector_path,
        selector_sha256=str(selector_ref),
        json_pointer=json_pointer,
        expected_value=expected_value,
        decision_slice_id=decision_slice_id,
        decision_implementation_sha256=implementation_projection_hash(
            decision_implementation
        ),
        selected_slice_id=selected_slice_id,
        selected_implementation_sha256=implementation_projection_hash(
            selected_implementation
        ),
    )
    encoded = serialize(evidence.to_payload())
    ConditionalAbsenceEvidence.from_mapping(
        _canonical_payload(encoded, label="generated conditional absence evidence")
    )
    return corpus.put_artifact(
        encoded,
        kind="conditional_absence_evidence",
        name_hint=f"{tier_id}_{slice_id}.json",
    )


def load_conditional_absence_evidence(
    corpus: "Corpus", ref: str
) -> ConditionalAbsenceEvidence:
    _content_hash(ref, label="conditional absence evidence ref")
    blob = corpus.get_artifact(ContentHash(ref))
    return ConditionalAbsenceEvidence.from_mapping(
        _canonical_payload(blob, label="conditional absence evidence")
    )


def conditional_absence_matches_implementation(
    corpus: "Corpus",
    implementation: "Implementation",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    creates_paths: tuple[str, ...],
    decision_slice_id: str,
    decision_implementation: "Implementation" | None,
    selected_slice_id: str,
    selected_implementation: "Implementation" | None,
) -> bool:
    """Validate exact selector, output, plan, and implementation bindings."""

    ref = (implementation.artifacts or {}).get(CONDITIONAL_ABSENCE_ARTIFACT)
    if ref is None or decision_implementation is None or selected_implementation is None:
        return False
    try:
        evidence = load_conditional_absence_evidence(corpus, str(ref))
        selector_bytes = corpus.get_artifact(ContentHash(evidence.selector_sha256))
        document = _strict_json_document(selector_bytes)
        selected_value = _resolve_json_pointer(document, evidence.json_pointer)
    except (
        KeyError, OSError, TypeError, ValueError, UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return False
    empty_digest = "sha256:" + hashlib.sha256(b"").hexdigest()
    return bool(
        evidence.tier_id == tier_id
        and evidence.slice_id == slice_id
        and evidence.spec_execution_authority_hash == spec_execution_authority_hash
        and evidence.candidate_diff_sha256 == str(implementation.diff_ref) == empty_digest
        and evidence.base_commit == implementation.base_commit
        and evidence.files_modified == implementation.files_modified == ()
        and evidence.creates_paths == creates_paths
        and selected_value == evidence.expected_value
        and type(selected_value) is type(evidence.expected_value)
        and evidence.decision_slice_id == decision_slice_id
        and evidence.decision_implementation_sha256
        == implementation_projection_hash(decision_implementation)
        and evidence.selected_slice_id == selected_slice_id
        and evidence.selected_implementation_sha256
        == implementation_projection_hash(selected_implementation)
    )


@dataclass(frozen=True, slots=True)
class TierImplementationBinding:
    """One declaration-ordered terminal Implementation projection."""

    tier_id: str
    slice_id: str
    implementation_projection_sha256: str

    def __post_init__(self) -> None:
        _plain_string(self.tier_id, label="TierImplementationBinding.tier_id")
        _plain_string(self.slice_id, label="TierImplementationBinding.slice_id")
        _content_hash(
            self.implementation_projection_sha256,
            label=(
                "TierImplementationBinding.implementation_projection_sha256"
            ),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "tier_id": self.tier_id,
            "slice_id": self.slice_id,
            "implementation_projection_sha256": (
                self.implementation_projection_sha256
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "TierImplementationBinding":
        fields = frozenset({
            "tier_id",
            "slice_id",
            "implementation_projection_sha256",
        })
        _exact_fields(value, fields, label="TierImplementationBinding")
        return cls(
            tier_id=_plain_string(
                value["tier_id"], label="TierImplementationBinding.tier_id"
            ),
            slice_id=_plain_string(
                value["slice_id"], label="TierImplementationBinding.slice_id"
            ),
            implementation_projection_sha256=_content_hash(
                value["implementation_projection_sha256"],
                label=(
                    "TierImplementationBinding."
                    "implementation_projection_sha256"
                ),
            ),
        )


def tier_implementation_bindings(
    values: tuple[tuple[str, "Implementation"], ...],
) -> tuple[TierImplementationBinding, ...]:
    """Project an ordered ``(tier_id, Implementation)`` sequence."""

    return tuple(
        TierImplementationBinding(
            tier_id=tier_id,
            slice_id=implementation.slice_id,
            implementation_projection_sha256=(
                implementation_projection_hash(implementation)
            ),
        )
        for tier_id, implementation in values
    )


@dataclass(frozen=True, slots=True)
class TierVerificationManifest:
    """Atomic final-candidate gate authority for a complete Tier closure."""

    schema_version: str
    tier_id: str
    spec_execution_authority_hash: str
    project_base_commit: str
    candidate_diff_sha256: str
    base_commit: str
    files_modified: tuple[str, ...]
    gate: str
    disposition: str
    obligation_slice_ids: tuple[str, ...]
    current_bindings: tuple[TierImplementationBinding, ...]
    dependency_bindings: tuple[TierImplementationBinding, ...]
    verification_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _TIER_VERIFICATION_MANIFEST_SCHEMA:
            raise ValueError("unsupported TierVerificationManifest schema")
        _plain_string(self.tier_id, label="TierVerificationManifest.tier_id")
        _content_hash(
            self.spec_execution_authority_hash,
            label="TierVerificationManifest.spec_execution_authority_hash",
        )
        _plain_string(
            self.project_base_commit,
            label="TierVerificationManifest.project_base_commit",
        )
        _content_hash(
            self.candidate_diff_sha256,
            label="TierVerificationManifest.candidate_diff_sha256",
        )
        _plain_string(
            self.base_commit,
            label="TierVerificationManifest.base_commit",
        )
        if not isinstance(self.files_modified, tuple) or any(
            not isinstance(item, str) or not item
            for item in self.files_modified
        ):
            raise ValueError(
                "TierVerificationManifest.files_modified must be a string tuple"
            )
        if len(set(self.files_modified)) != len(self.files_modified):
            raise ValueError(
                "TierVerificationManifest.files_modified has duplicates"
            )
        if self.gate not in {"verify", "dev_verify"}:
            raise ValueError("TierVerificationManifest.gate is unsupported")
        if self.disposition not in _VERIFICATION_DISPOSITIONS:
            raise ValueError(
                "TierVerificationManifest.disposition is unsupported"
            )
        if (
            not self.obligation_slice_ids
            or any(
                not isinstance(item, str) or not item
                for item in self.obligation_slice_ids
            )
            or len(set(self.obligation_slice_ids))
            != len(self.obligation_slice_ids)
        ):
            raise ValueError(
                "TierVerificationManifest obligation slices must be unique"
            )
        if not self.current_bindings:
            raise ValueError(
                "TierVerificationManifest.current_bindings must be non-empty"
            )
        identities = [
            (item.tier_id, item.slice_id)
            for item in (*self.dependency_bindings, *self.current_bindings)
        ]
        if len(set(identities)) != len(identities):
            raise ValueError(
                "TierVerificationManifest implementation bindings overlap"
            )
        if any(item.tier_id != self.tier_id for item in self.current_bindings):
            raise ValueError(
                "TierVerificationManifest current binding has wrong tier"
            )
        current_slice_ids = {item.slice_id for item in self.current_bindings}
        if not set(self.obligation_slice_ids).issubset(current_slice_ids):
            raise ValueError(
                "TierVerificationManifest obligation is not a current slice"
            )
        if not self.verification_evidence_refs:
            raise ValueError(
                "TierVerificationManifest requires verification evidence"
            )
        for ref in self.verification_evidence_refs:
            _content_hash(
                ref,
                label="TierVerificationManifest.verification_evidence_ref",
            )
        if len(set(self.verification_evidence_refs)) != len(
            self.verification_evidence_refs
        ):
            raise ValueError(
                "TierVerificationManifest verification refs have duplicates"
            )

    @property
    def satisfies_required_gate(self) -> bool:
        return self.disposition == "passed"

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tier_id": self.tier_id,
            "spec_execution_authority_hash": (
                self.spec_execution_authority_hash
            ),
            "project_base_commit": self.project_base_commit,
            "candidate_diff_sha256": self.candidate_diff_sha256,
            "base_commit": self.base_commit,
            "files_modified": list(self.files_modified),
            "gate": self.gate,
            "disposition": self.disposition,
            "obligation_slice_ids": list(self.obligation_slice_ids),
            "current_bindings": [
                item.to_payload() for item in self.current_bindings
            ],
            "dependency_bindings": [
                item.to_payload() for item in self.dependency_bindings
            ],
            "verification_evidence_refs": list(
                self.verification_evidence_refs
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TierVerificationManifest":
        fields = frozenset({
            "schema_version",
            "tier_id",
            "spec_execution_authority_hash",
            "project_base_commit",
            "candidate_diff_sha256",
            "base_commit",
            "files_modified",
            "gate",
            "disposition",
            "obligation_slice_ids",
            "current_bindings",
            "dependency_bindings",
            "verification_evidence_refs",
        })
        _exact_fields(value, fields, label="TierVerificationManifest")

        def bindings(field_name: str) -> tuple[TierImplementationBinding, ...]:
            raw = value[field_name]
            if not isinstance(raw, list) or not all(
                isinstance(item, dict) for item in raw
            ):
                raise ValueError(
                    f"TierVerificationManifest.{field_name} must be an object list"
                )
            return tuple(
                TierImplementationBinding.from_mapping(item) for item in raw
            )

        raw_files = value["files_modified"]
        if not isinstance(raw_files, list) or not all(
            isinstance(item, str) for item in raw_files
        ):
            raise ValueError(
                "TierVerificationManifest.files_modified must be a string list"
            )
        raw_refs = value["verification_evidence_refs"]
        if not isinstance(raw_refs, list) or not all(
            isinstance(item, str) for item in raw_refs
        ):
            raise ValueError(
                "TierVerificationManifest.verification_evidence_refs must be "
                "a string list"
            )
        raw_obligations = value["obligation_slice_ids"]
        if not isinstance(raw_obligations, list) or not all(
            isinstance(item, str) for item in raw_obligations
        ):
            raise ValueError(
                "TierVerificationManifest.obligation_slice_ids must be a "
                "string list"
            )
        return cls(
            schema_version=_plain_string(
                value["schema_version"],
                label="TierVerificationManifest.schema_version",
            ),
            tier_id=_plain_string(
                value["tier_id"], label="TierVerificationManifest.tier_id"
            ),
            spec_execution_authority_hash=_content_hash(
                value["spec_execution_authority_hash"],
                label=(
                    "TierVerificationManifest.spec_execution_authority_hash"
                ),
            ),
            project_base_commit=_plain_string(
                value["project_base_commit"],
                label="TierVerificationManifest.project_base_commit",
            ),
            candidate_diff_sha256=_content_hash(
                value["candidate_diff_sha256"],
                label="TierVerificationManifest.candidate_diff_sha256",
            ),
            base_commit=_plain_string(
                value["base_commit"],
                label="TierVerificationManifest.base_commit",
            ),
            files_modified=tuple(raw_files),
            gate=_plain_string(
                value["gate"], label="TierVerificationManifest.gate"
            ),
            disposition=_plain_string(
                value["disposition"],
                label="TierVerificationManifest.disposition",
            ),
            obligation_slice_ids=tuple(raw_obligations),
            current_bindings=bindings("current_bindings"),
            dependency_bindings=bindings("dependency_bindings"),
            verification_evidence_refs=tuple(raw_refs),
        )


def _manifest_disposition(
    evidence: tuple[VerificationEvidence, ...],
) -> str:
    dispositions = {item.disposition for item in evidence}
    if dispositions == {"passed"}:
        return "passed"
    if "failed" in dispositions:
        return "failed"
    if "unavailable" in dispositions:
        return "unavailable"
    return "waived"


def persist_tier_verification_manifest(
    corpus: "Corpus",
    *,
    tier_id: str,
    spec_execution_authority_hash: str,
    project_base_commit: str,
    candidate_diff_ref: ContentHash,
    base_commit: str,
    files_modified: tuple[str, ...],
    gate: str,
    obligation_slice_ids: tuple[str, ...],
    current_implementations: tuple["Implementation", ...],
    dependency_implementations: tuple[tuple[str, "Implementation"], ...],
    verification_evidence_refs: tuple[ContentHash, ...],
) -> ContentHash:
    """Persist one cycle-free, closure-bound Tier verification manifest."""

    corpus.get_artifact(candidate_diff_ref)
    loaded = tuple(
        load_verification_evidence(corpus, str(ref))
        for ref in verification_evidence_refs
    )
    if any(
        not verification_evidence_content_available(corpus, item)
        for item in loaded
    ):
        raise ValueError("Tier verification evidence content is unavailable")
    if any(
        implementation.base_commit != base_commit
        for implementation in current_implementations
    ):
        raise ValueError(
            "Tier manifest candidate base differs from a current Implementation"
        )
    current_slice_ids = {item.slice_id for item in current_implementations}
    if tuple(item.slice_id for item in loaded) != obligation_slice_ids:
        raise ValueError(
            "Tier verification evidence does not cover exact obligations"
        )
    if any(
        item.tier_id != tier_id
        or item.slice_id not in current_slice_ids
        or item.spec_execution_authority_hash
        != spec_execution_authority_hash
        or item.candidate_diff_sha256 != str(candidate_diff_ref)
        or item.base_commit != base_commit
        or item.files_modified != files_modified
        or item.gate != gate
        for item in loaded
    ):
        raise ValueError(
            "Tier verification evidence does not bind the manifest candidate"
        )
    manifest = TierVerificationManifest(
        schema_version=_TIER_VERIFICATION_MANIFEST_SCHEMA,
        tier_id=tier_id,
        spec_execution_authority_hash=spec_execution_authority_hash,
        project_base_commit=project_base_commit,
        candidate_diff_sha256=str(candidate_diff_ref),
        base_commit=base_commit,
        files_modified=files_modified,
        gate=gate,
        disposition=_manifest_disposition(loaded),
        obligation_slice_ids=obligation_slice_ids,
        current_bindings=tier_implementation_bindings(tuple(
            (tier_id, implementation)
            for implementation in current_implementations
        )),
        dependency_bindings=tier_implementation_bindings(
            dependency_implementations
        ),
        verification_evidence_refs=tuple(
            str(ref) for ref in verification_evidence_refs
        ),
    )
    encoded = serialize(manifest.to_payload())
    TierVerificationManifest.from_mapping(_canonical_payload(
        encoded,
        label="generated Tier verification manifest",
    ))
    return corpus.put_artifact(
        encoded,
        kind="tier_verification_manifests",
        name_hint=f"{tier_id}.json",
    )


def load_tier_verification_manifest(
    corpus: "Corpus",
    ref: str,
) -> TierVerificationManifest:
    _content_hash(ref, label="Tier verification manifest ref")
    blob = corpus.get_artifact(ContentHash(ref))
    return TierVerificationManifest.from_mapping(
        _canonical_payload(blob, label="Tier verification manifest")
    )


def tier_verification_manifest_matches(
    corpus: "Corpus",
    *,
    manifest_ref: str,
    tier_id: str,
    spec_execution_authority_hash: str,
    expected_project_base_commit: str,
    current_implementations: tuple["Implementation", ...],
    dependency_implementations: tuple[tuple[str, "Implementation"], ...],
    expected_obligation_slice_ids: tuple[str, ...],
    require_gate: bool,
) -> bool:
    """Validate one atomic Tier manifest against the exact retained closure."""

    try:
        manifest = load_tier_verification_manifest(corpus, manifest_ref)
        corpus.get_artifact(ContentHash(manifest.candidate_diff_sha256))
        loaded = tuple(
            load_verification_evidence(corpus, ref)
            for ref in manifest.verification_evidence_refs
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    if any(
        not verification_evidence_content_available(corpus, item)
        for item in loaded
    ):
        return False
    if (
        manifest.tier_id != tier_id
        or manifest.spec_execution_authority_hash
        != spec_execution_authority_hash
        or manifest.project_base_commit != expected_project_base_commit
        or any(
            implementation.base_commit != manifest.base_commit
            for implementation in current_implementations
        )
        or any(
            str((implementation.artifacts or {}).get(
                TIER_VERIFICATION_MANIFEST_ARTIFACT
            ))
            != manifest_ref
            for implementation in current_implementations
        )
        or manifest.current_bindings
        != tier_implementation_bindings(tuple(
            (tier_id, implementation)
            for implementation in current_implementations
        ))
        or manifest.dependency_bindings
        != tier_implementation_bindings(dependency_implementations)
        or manifest.obligation_slice_ids != expected_obligation_slice_ids
        or tuple(item.slice_id for item in loaded)
        != manifest.obligation_slice_ids
        or manifest.disposition != _manifest_disposition(loaded)
        or any(
            item.tier_id != tier_id
            or item.slice_id not in {
                implementation.slice_id
                for implementation in current_implementations
            }
            or item.spec_execution_authority_hash
            != spec_execution_authority_hash
            or item.candidate_diff_sha256
            != manifest.candidate_diff_sha256
            or item.base_commit != manifest.base_commit
            or item.files_modified != manifest.files_modified
            or item.gate != manifest.gate
            for item in loaded
        )
    ):
        return False
    return not require_gate or manifest.satisfies_required_gate


@dataclass(frozen=True, slots=True)
class GateLedgerEntry:
    """One requested child or deterministic follow-on gate outcome."""

    gate: str
    candidate_diff_sha256: str
    terminal_class: str
    decision: str
    exit_code: int | None
    workflow_outcome: WorkflowOutcome | None
    verification_evidence_ref: str | None
    error_kind: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        _plain_string(self.gate, label="GateLedgerEntry.gate")
        _content_hash(
            self.candidate_diff_sha256,
            label="GateLedgerEntry.candidate_diff_sha256",
        )
        if self.terminal_class not in _GATE_TERMINALS:
            raise ValueError("GateLedgerEntry.terminal_class is unsupported")
        _plain_string(self.decision, label="GateLedgerEntry.decision")
        if self.exit_code is not None:
            _exact_int(self.exit_code, label="GateLedgerEntry.exit_code")
        if self.verification_evidence_ref is not None:
            _content_hash(
                self.verification_evidence_ref,
                label="GateLedgerEntry.verification_evidence_ref",
            )
        _optional_string(self.error_kind, label="GateLedgerEntry.error_kind")
        _optional_string(self.error_message, label="GateLedgerEntry.error_message")
        if self.workflow_outcome is not None:
            if self.terminal_class != self.workflow_outcome.terminal_class.value:
                raise ValueError("gate terminal_class disagrees with WorkflowOutcome")
            if self.decision != self.workflow_outcome.detail_status:
                raise ValueError("gate decision disagrees with WorkflowOutcome")
        if self.terminal_class == "skipped" and (
            self.workflow_outcome is not None
            or self.verification_evidence_ref is not None
        ):
            raise ValueError("skipped gate cannot claim executed evidence")

    @property
    def succeeded(self) -> bool:
        if self.error_kind is not None or self.exit_code != 0:
            return False
        if self.gate == "breach":
            outcome = self.workflow_outcome
            return bool(
                outcome is not None
                and outcome.workflow == "breach"
                and outcome.terminal_class is TerminalClass.SUCCEEDED
                and outcome.detail_status in {"no_attacks", "proceed"}
                and outcome.error_kind is None
                and outcome.error_message is None
            )
        if self.gate.startswith("review:"):
            outcome = self.workflow_outcome
            return bool(
                outcome is not None
                and outcome.workflow == "review"
                and outcome.terminal_class is TerminalClass.SUCCEEDED
                and outcome.detail_status == "pass"
                and outcome.error_kind is None
                and outcome.error_message is None
            )
        return self.terminal_class == TerminalClass.SUCCEEDED.value

    def to_payload(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "candidate_diff_sha256": self.candidate_diff_sha256,
            "terminal_class": self.terminal_class,
            "decision": self.decision,
            "exit_code": self.exit_code,
            "workflow_outcome": (
                self.workflow_outcome.to_dict()
                if self.workflow_outcome is not None
                else None
            ),
            "verification_evidence_ref": self.verification_evidence_ref,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GateLedgerEntry":
        fields = frozenset({
            "gate",
            "candidate_diff_sha256",
            "terminal_class",
            "decision",
            "exit_code",
            "workflow_outcome",
            "verification_evidence_ref",
            "error_kind",
            "error_message",
        })
        _exact_fields(value, fields, label="GateLedgerEntry")
        raw_outcome = value["workflow_outcome"]
        if raw_outcome is not None and not isinstance(raw_outcome, dict):
            raise ValueError("GateLedgerEntry.workflow_outcome must be object or null")
        raw_rc = value["exit_code"]
        raw_verification_ref = value["verification_evidence_ref"]
        return cls(
            gate=_plain_string(value["gate"], label="GateLedgerEntry.gate"),
            candidate_diff_sha256=_content_hash(
                value["candidate_diff_sha256"],
                label="GateLedgerEntry.candidate_diff_sha256",
            ),
            terminal_class=_plain_string(
                value["terminal_class"],
                label="GateLedgerEntry.terminal_class",
            ),
            decision=_plain_string(
                value["decision"], label="GateLedgerEntry.decision"
            ),
            exit_code=(
                None
                if raw_rc is None
                else _exact_int(raw_rc, label="GateLedgerEntry.exit_code")
            ),
            workflow_outcome=(
                WorkflowOutcome.from_mapping(raw_outcome)
                if raw_outcome is not None
                else None
            ),
            verification_evidence_ref=(
                None
                if raw_verification_ref is None
                else _content_hash(
                    raw_verification_ref,
                    label="GateLedgerEntry.verification_evidence_ref",
                )
            ),
            error_kind=_optional_string(
                value["error_kind"], label="GateLedgerEntry.error_kind"
            ),
            error_message=_optional_string(
                value["error_message"], label="GateLedgerEntry.error_message"
            ),
        )


@dataclass(frozen=True, slots=True)
class HandoffGateLedger:
    """Ordered typed gate evidence attached to a terminal Handoff record."""

    schema_version: str
    tier_id: str
    slice_id: str
    spec_execution_authority_hash: str
    terminal_candidate_diff_sha256: str
    entries: tuple[GateLedgerEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _HANDOFF_GATE_LEDGER_SCHEMA:
            raise ValueError("unsupported HandoffGateLedger schema")
        _plain_string(self.tier_id, label="HandoffGateLedger.tier_id")
        _plain_string(self.slice_id, label="HandoffGateLedger.slice_id")
        _content_hash(
            self.spec_execution_authority_hash,
            label="HandoffGateLedger.spec_execution_authority_hash",
        )
        _content_hash(
            self.terminal_candidate_diff_sha256,
            label="HandoffGateLedger.terminal_candidate_diff_sha256",
        )
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("HandoffGateLedger.entries must be non-empty")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tier_id": self.tier_id,
            "slice_id": self.slice_id,
            "spec_execution_authority_hash": (
                self.spec_execution_authority_hash
            ),
            "terminal_candidate_diff_sha256": (
                self.terminal_candidate_diff_sha256
            ),
            "entries": [entry.to_payload() for entry in self.entries],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HandoffGateLedger":
        fields = frozenset({
            "schema_version",
            "tier_id",
            "slice_id",
            "spec_execution_authority_hash",
            "terminal_candidate_diff_sha256",
            "entries",
        })
        _exact_fields(value, fields, label="HandoffGateLedger")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list) or not all(
            isinstance(item, dict) for item in raw_entries
        ):
            raise ValueError("HandoffGateLedger.entries must be an object list")
        return cls(
            schema_version=_plain_string(
                value["schema_version"],
                label="HandoffGateLedger.schema_version",
            ),
            tier_id=_plain_string(
                value["tier_id"], label="HandoffGateLedger.tier_id"
            ),
            slice_id=_plain_string(
                value["slice_id"], label="HandoffGateLedger.slice_id"
            ),
            spec_execution_authority_hash=_content_hash(
                value["spec_execution_authority_hash"],
                label="HandoffGateLedger.spec_execution_authority_hash",
            ),
            terminal_candidate_diff_sha256=_content_hash(
                value["terminal_candidate_diff_sha256"],
                label="HandoffGateLedger.terminal_candidate_diff_sha256",
            ),
            entries=tuple(
                GateLedgerEntry.from_mapping(item) for item in raw_entries
            ),
        )


def persist_handoff_gate_ledger(
    corpus: "Corpus",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    terminal_candidate_diff_ref: ContentHash,
    entries: tuple[GateLedgerEntry, ...],
) -> ContentHash:
    ledger = HandoffGateLedger(
        schema_version=_HANDOFF_GATE_LEDGER_SCHEMA,
        tier_id=tier_id,
        slice_id=slice_id,
        spec_execution_authority_hash=spec_execution_authority_hash,
        terminal_candidate_diff_sha256=str(terminal_candidate_diff_ref),
        entries=entries,
    )
    encoded = serialize(ledger.to_payload())
    HandoffGateLedger.from_mapping(_canonical_payload(
        encoded,
        label="generated Handoff gate ledger",
    ))
    return corpus.put_artifact(
        encoded,
        kind="handoff_gate_ledgers",
        name_hint=f"{tier_id}_{slice_id}.json",
    )


def load_handoff_gate_ledger(
    corpus: "Corpus",
    ref: str,
) -> HandoffGateLedger:
    _content_hash(ref, label="Handoff gate ledger ref")
    blob = corpus.get_artifact(ContentHash(ref))
    return HandoffGateLedger.from_mapping(
        _canonical_payload(blob, label="Handoff gate ledger")
    )


def _typed_child_gate_passes(
    entry: GateLedgerEntry,
    *,
    workflow: str,
    decisions: frozenset[str],
) -> bool:
    outcome = entry.workflow_outcome
    return bool(
        entry.error_kind is None
        and entry.exit_code == 0
        and outcome is not None
        and outcome.workflow == workflow
        and outcome.terminal_class is TerminalClass.SUCCEEDED
        and outcome.detail_status in decisions
        and outcome.error_kind is None
        and outcome.error_message is None
    )


def handoff_gate_ledger_matches(
    corpus: "Corpus",
    implementation: "Implementation",
    *,
    tier_id: str,
    slice_id: str,
    spec_execution_authority_hash: str,
    require_breach: bool,
    review_rubric: str | None,
) -> bool:
    """Prove that a terminal Handoff ran every currently requested child gate.

    Breach may accept a Defender mutation, so its successful child outcome can
    bind the pre-Defender candidate only when a later post-Breach verification
    entry closes the chain to the terminal candidate.  Review always consumes
    the final candidate and therefore must bind it directly.  An empty patch is
    the sole deterministic Breach skip admitted by this contract.
    """

    if not require_breach and review_rubric is None:
        return True
    ref = (implementation.artifacts or {}).get(HANDOFF_GATE_LEDGER_ARTIFACT)
    if ref is None:
        return False
    try:
        ledger = load_handoff_gate_ledger(corpus, str(ref))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    terminal_digest = str(implementation.diff_ref)
    if (
        ledger.tier_id != tier_id
        or ledger.slice_id != slice_id
        or ledger.spec_execution_authority_hash
        != spec_execution_authority_hash
        or ledger.terminal_candidate_diff_sha256 != terminal_digest
    ):
        return False

    entries = ledger.entries

    def closes_terminal_candidate(entry: GateLedgerEntry) -> bool:
        expected_ref = (implementation.artifacts or {}).get(
            HANDOFF_VERIFICATION_ARTIFACT
        )
        if (
            expected_ref is None
            or entry.gate != "post_breach_verify"
            or entry.candidate_diff_sha256 != terminal_digest
            or entry.verification_evidence_ref != str(expected_ref)
            or entry.error_kind is not None
            or entry.exit_code != 0
        ):
            return False
        try:
            evidence = load_verification_evidence(
                corpus,
                entry.verification_evidence_ref,
            )
        except (KeyError, OSError, TypeError, ValueError):
            return False
        if not verification_evidence_content_available(corpus, evidence):
            return False
        if (
            evidence.tier_id != tier_id
            or evidence.slice_id != slice_id
            or evidence.spec_execution_authority_hash
            != spec_execution_authority_hash
            or evidence.candidate_diff_sha256 != terminal_digest
            or evidence.base_commit != implementation.base_commit
            or evidence.files_modified != implementation.files_modified
            or evidence.gate != "verify"
            or entry.decision != evidence.classification
        ):
            return False
        expected_terminal = (
            TerminalClass.SUCCEEDED.value
            if evidence.disposition == "passed"
            else TerminalClass.NO_WORK.value
            if evidence.disposition in {"waived", "unavailable"}
            else None
        )
        return entry.terminal_class == expected_terminal

    if require_breach:
        empty_digest = "sha256:" + hashlib.sha256(b"").hexdigest()
        breach_passed = False
        for index, entry in enumerate(entries):
            if entry.gate != "breach":
                continue
            if (
                terminal_digest == empty_digest
                and entry.candidate_diff_sha256 == terminal_digest
                and entry.terminal_class == "skipped"
                and entry.decision == "skipped_empty_diff"
                and entry.error_kind is None
                and entry.exit_code == 0
            ):
                breach_passed = True
                break
            if not _typed_child_gate_passes(
                entry,
                workflow="breach",
                decisions=frozenset({"no_attacks", "proceed"}),
            ):
                continue
            if entry.candidate_diff_sha256 == terminal_digest:
                breach_passed = True
                break
            if any(
                closes_terminal_candidate(later)
                for later in entries[index + 1 :]
            ):
                breach_passed = True
                break
        if not breach_passed:
            return False

    if review_rubric is not None:
        review_gate = f"review:{review_rubric}"
        if not any(
            entry.gate == review_gate
            and entry.candidate_diff_sha256 == terminal_digest
            and _typed_child_gate_passes(
                entry,
                workflow="review",
                decisions=frozenset({"pass"}),
            )
            for entry in entries
        ):
            return False
    return True


__all__ = [
    "CONDITIONAL_ABSENCE_ARTIFACT",
    "DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT",
    "ConditionalAbsenceEvidence",
    "GateLedgerEntry",
    "HANDOFF_AUTHORIZING_FINALIZATION_ARTIFACT",
    "HANDOFF_GATE_LEDGER_ARTIFACT",
    "HANDOFF_VERIFICATION_ARTIFACT",
    "HandoffAuthoringFinalizationEvidence",
    "TIER_VERIFICATION_ARTIFACT",
    "TIER_VERIFICATION_MANIFEST_ARTIFACT",
    "HandoffGateLedger",
    "RunnerEvidence",
    "TierImplementationBinding",
    "TierVerificationManifest",
    "VerificationEvidence",
    "bound_workspace_path_absent",
    "conditional_absence_matches_implementation",
    "handoff_gate_ledger_matches",
    "handoff_authoring_finalization_matches",
    "implementation_projection_hash",
    "load_handoff_gate_ledger",
    "load_handoff_authoring_finalization_evidence",
    "load_conditional_absence_evidence",
    "load_tier_verification_manifest",
    "load_verification_evidence",
    "persist_handoff_gate_ledger",
    "persist_handoff_authoring_finalization_evidence",
    "persist_conditional_absence_evidence",
    "persist_tier_verification_manifest",
    "persist_verification_evidence",
    "tier_implementation_bindings",
    "tier_verification_manifest_matches",
    "validate_conditional_selector",
    "verification_matches_implementation",
    "verification_evidence_content_available",
]
