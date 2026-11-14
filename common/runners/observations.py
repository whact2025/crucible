# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Canonical identities and bounded runner-owned attack observations."""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Required, Sequence, TypedDict

AttackObservationStatus = Literal[
    "passed", "failed", "not_observed", "runner_error"
]

_MAX_OBSERVATION_RECORDS = 256
_MAX_OBSERVATION_BYTES = 64 * 1024
_MAX_ATTACK_ID_BYTES = 256
_MAX_PYTEST_FAILURE_BYTES = 4096
_MAX_PYTEST_PATH_BYTES = 512
_MAX_PYTEST_SYMBOL_BYTES = 256
_MAX_PYTEST_LINE = 10_000_000

PYTEST_FAILURE_SCHEMA = "crucible.pytest.failure.v1"

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTTED_IDENTIFIER_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z"
)
_CODE_NAME_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*|<[A-Za-z_][A-Za-z0-9_ ]*>)\Z"
)


class PytestFailureTest(TypedDict):
    """Exact test identity supplied to the bounded observer."""

    path: str
    function: str


class PytestFailureFrame(TypedDict):
    """Terminal traceback frame without source or local values."""

    path: str
    function: str
    line: int


class PytestFailureException(TypedDict, total=False):
    """Bounded exception identity and optional direct class-attribute proof."""

    type: Required[str]
    subject_kind: Literal["type"]
    subject_module: str
    subject_name: str
    attribute: str
    direct_source_reference: Literal[True]


class PytestFailureMetadata(TypedDict):
    """Bounded, message-free hint for one exact pytest failure.

    The pytest hook shares an interpreter with generated test code, so schema
    validation does not make this provenance authoritative. Consumers may use
    it only for fail-closed diagnostics or bounded repair scheduling.
    """

    schema: Literal["crucible.pytest.failure.v1"]
    phase: Literal["call", "setup", "teardown"]
    test: PytestFailureTest
    frame: PytestFailureFrame
    exception: PytestFailureException

# Reserved process exits emitted by the Crucible-owned exact pytest observer.
# They intentionally do not overlap pytest's documented 0-5 exit range.
PYTEST_OBSERVER_PASSED = 70
PYTEST_OBSERVER_FAILED = 71
PYTEST_OBSERVER_NOT_OBSERVED = 72
PYTEST_OBSERVER_RUNNER_ERROR = 73

_PYTEST_OBSERVER_STATUSES: dict[int, AttackObservationStatus] = {
    PYTEST_OBSERVER_PASSED: "passed",
    PYTEST_OBSERVER_FAILED: "failed",
    PYTEST_OBSERVER_NOT_OBSERVED: "not_observed",
    PYTEST_OBSERVER_RUNNER_ERROR: "runner_error",
}


def canonical_attack_test_id(attack_id: str, language: str) -> str:
    """Return the sole executable test identity for one Breaker finding.

    Breaker ids are evidence identities, not necessarily language identifiers.
    Canonicalizing once prevents authoring, execution, ledger, and stripping
    from each accepting a different fuzzy spelling.
    """

    if not isinstance(attack_id, str) or not attack_id:
        raise ValueError("attack ids must be non-empty strings")
    if len(attack_id.encode("utf-8")) > _MAX_ATTACK_ID_BYTES:
        raise ValueError(
            f"attack id exceeds {_MAX_ATTACK_ID_BYTES} UTF-8 bytes: {attack_id!r}"
        )
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in attack_id):
        raise ValueError(f"attack id contains control characters: {attack_id!r}")

    normalized_language = (language or "").strip().lower()
    if normalized_language in {"javascript", "typescript"}:
        # These frameworks identify tests by a string literal, not a source
        # identifier. Retain the exact finding id so stripping is lossless.
        return attack_id

    words = re.findall(r"[A-Za-z0-9]+", attack_id)
    if not words:
        raise ValueError(
            f"attack id has no ASCII identifier content: {attack_id!r}"
        )
    if normalized_language in {"python", "rust"}:
        return "test_" + "_".join(word.lower() for word in words)
    if normalized_language == "go":
        return "TestBreach" + "".join(
            word[:1].upper() + word[1:] for word in words
        )
    raise ValueError(
        f"breach has no canonical test identity for language={language!r}"
    )


def canonical_attack_test_ids(
    attack_ids: object,
    language: str,
) -> tuple[tuple[str, str], ...]:
    """Validate a batch and return ``(finding_id, test_id)`` pairs.

    The mapping must be bijective. For example ``race-a`` and ``race_a`` both
    map to ``test_race_a`` in Python and are rejected before authoring or test
    execution rather than allowing one test to satisfy two findings.
    """

    if not isinstance(attack_ids, (list, tuple)) or not attack_ids:
        raise ValueError(
            "breach_observations requires a non-empty breach_attack_ids list"
        )
    if not all(isinstance(attack_id, str) and attack_id for attack_id in attack_ids):
        raise ValueError(
            "breach_observations requires a non-empty breach_attack_ids list"
        )
    if len(set(attack_ids)) != len(attack_ids):
        raise ValueError("breach attack ids must be unique")

    pairs = tuple(
        (attack_id, canonical_attack_test_id(attack_id, language))
        for attack_id in attack_ids
    )
    owners: dict[str, str] = {}
    for attack_id, test_id in pairs:
        prior = owners.setdefault(test_id, attack_id)
        if prior != attack_id:
            raise ValueError(
                "breach attack ids collide after canonicalization for "
                f"language={language!r}: {prior!r} and {attack_id!r} -> "
                f"{test_id!r}"
            )
    return pairs


def pytest_observer_status(returncode: int) -> AttackObservationStatus | None:
    """Map only the reserved observer exits; ordinary pytest exits are untrusted."""

    return _PYTEST_OBSERVER_STATUSES.get(returncode)


def _bounded_pytest_text(value: object, *, limit: int) -> str:
    if type(value) is not str or not value:
        raise ValueError("pytest failure metadata requires non-empty text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("pytest failure metadata contains invalid text") from exc
    if len(encoded) > limit or any(
        ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    ):
        raise ValueError("pytest failure metadata text is unsafe or unbounded")
    return value


def _pytest_test_identity(expected_node_id: str) -> tuple[str, str]:
    if type(expected_node_id) is not str:
        raise ValueError("pytest failure metadata requires an exact node id")
    parts = expected_node_id.split("::")
    if len(parts) != 2:
        raise ValueError("pytest failure metadata requires file::test identity")
    path = _bounded_pytest_text(parts[0], limit=_MAX_PYTEST_PATH_BYTES)
    function = _bounded_pytest_text(parts[1], limit=_MAX_PYTEST_SYMBOL_BYTES)
    if (
        _IDENTIFIER_RE.fullmatch(function) is None
        or not function.startswith("test_")
    ):
        raise ValueError("pytest failure metadata requires a canonical test function")
    return path, function


def validate_pytest_failure(
    value: object,
    *,
    expected_node_id: str,
) -> PytestFailureMetadata:
    """Validate and copy one message-free exact-pytest failure record.

    The generic form exposes only exception type and terminal traceback
    provenance.  A larger form is accepted solely for a class-owned
    ``AttributeError`` whose exact test instruction was independently proven
    to be a direct ``Type.attribute`` access by the embedded observer.
    """

    expected_path, expected_function = _pytest_test_identity(expected_node_id)
    if type(value) is not dict or set(value) != {
        "schema", "phase", "test", "frame", "exception",
    }:
        raise ValueError("pytest failure metadata has an invalid envelope")
    if value.get("schema") != PYTEST_FAILURE_SCHEMA:
        raise ValueError("pytest failure metadata has an unsupported schema")
    phase = value.get("phase")
    if phase not in {"call", "setup", "teardown"}:
        raise ValueError("pytest failure metadata has an invalid phase")

    test = value.get("test")
    if type(test) is not dict or set(test) != {"path", "function"}:
        raise ValueError("pytest failure metadata has an invalid test identity")
    test_path = _bounded_pytest_text(
        test.get("path"), limit=_MAX_PYTEST_PATH_BYTES
    )
    test_function = _bounded_pytest_text(
        test.get("function"), limit=_MAX_PYTEST_SYMBOL_BYTES
    )
    if test_path != expected_path or test_function != expected_function:
        raise ValueError("pytest failure metadata does not bind the expected test")

    frame = value.get("frame")
    if type(frame) is not dict or set(frame) != {"path", "function", "line"}:
        raise ValueError("pytest failure metadata has an invalid terminal frame")
    frame_path = _bounded_pytest_text(
        frame.get("path"), limit=_MAX_PYTEST_PATH_BYTES
    )
    frame_function = _bounded_pytest_text(
        frame.get("function"), limit=_MAX_PYTEST_SYMBOL_BYTES
    )
    frame_line = frame.get("line")
    if (
        _CODE_NAME_RE.fullmatch(frame_function) is None
        or type(frame_line) is not int
        or not 0 < frame_line <= _MAX_PYTEST_LINE
    ):
        raise ValueError("pytest failure metadata has invalid frame provenance")

    exception = value.get("exception")
    generic_keys = {"type"}
    attribute_keys = {
        "type",
        "subject_kind",
        "subject_module",
        "subject_name",
        "attribute",
        "direct_source_reference",
    }
    if type(exception) is not dict:
        raise ValueError("pytest failure metadata has invalid exception fields")
    exception_keys = frozenset(exception)
    if exception_keys not in {
        frozenset(generic_keys), frozenset(attribute_keys),
    }:
        raise ValueError("pytest failure metadata has invalid exception fields")
    exception_type = _bounded_pytest_text(
        exception.get("type"), limit=_MAX_PYTEST_SYMBOL_BYTES
    )
    if _IDENTIFIER_RE.fullmatch(exception_type) is None:
        raise ValueError("pytest failure metadata has an invalid exception type")
    exception_record: PytestFailureException = {"type": exception_type}
    if exception_keys == frozenset(attribute_keys):
        subject_module = _bounded_pytest_text(
            exception.get("subject_module"), limit=_MAX_PYTEST_SYMBOL_BYTES
        )
        subject_name = _bounded_pytest_text(
            exception.get("subject_name"), limit=_MAX_PYTEST_SYMBOL_BYTES
        )
        attribute = _bounded_pytest_text(
            exception.get("attribute"), limit=_MAX_PYTEST_SYMBOL_BYTES
        )
        if (
            exception_type != "AttributeError"
            or exception.get("subject_kind") != "type"
            or _DOTTED_IDENTIFIER_RE.fullmatch(subject_module) is None
            or _DOTTED_IDENTIFIER_RE.fullmatch(subject_name) is None
            or _IDENTIFIER_RE.fullmatch(attribute) is None
            or exception.get("direct_source_reference") is not True
        ):
            raise ValueError(
                "pytest failure metadata has invalid class-attribute proof"
            )
        exception_record.update({
            "subject_kind": "type",
            "subject_module": subject_module,
            "subject_name": subject_name,
            "attribute": attribute,
            "direct_source_reference": True,
        })

    normalized: PytestFailureMetadata = {
        "schema": PYTEST_FAILURE_SCHEMA,
        "phase": phase,
        "test": {"path": test_path, "function": test_function},
        "frame": {
            "path": frame_path,
            "function": frame_function,
            "line": frame_line,
        },
        "exception": exception_record,
    }
    encoded_size = len(
        json.dumps(normalized, ensure_ascii=True).encode("utf-8")
    )
    if encoded_size > _MAX_PYTEST_FAILURE_BYTES:
        raise ValueError("pytest failure metadata exceeded the trusted cap")
    return normalized


def pytest_observer_failure(
    output: str,
    *,
    expected_node_id: str,
) -> PytestFailureMetadata | None:
    """Parse the supervisor-relayed optional canonical stdout envelope.

    Empty output preserves the legacy status-only protocol.  Any non-empty,
    malformed, duplicated-key, non-canonical, or unbounded output is an error
    so arbitrary diagnostic text cannot enter the bounded hint shape. This
    validates structure and identity binding, not same-interpreter provenance.
    """

    if type(output) is not str:
        raise ValueError("pytest observer output must be text")
    if output == "":
        return None
    try:
        encoded = output.encode("ascii")
    except UnicodeError as exc:
        raise ValueError("pytest observer output must be canonical ASCII JSON") from exc
    if len(encoded) > _MAX_PYTEST_FAILURE_BYTES:
        raise ValueError("pytest observer output exceeded the trusted cap")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("pytest observer output contains duplicate keys")
            result[key] = item
        return result

    try:
        decoded = json.loads(output, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("pytest observer output is not valid JSON") from exc
    normalized = validate_pytest_failure(
        decoded, expected_node_id=expected_node_id
    )
    canonical = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if output != canonical:
        raise ValueError("pytest observer output is not canonical JSON")
    return normalized


@lru_cache(maxsize=1)
def _pytest_observer_inline_script() -> str:
    """Embed the dependency-free observer for host or container execution."""

    source = Path(__file__).with_name("pytest_observer.py").read_bytes()
    encoded = base64.b64encode(source).decode("ascii")
    return (
        "import base64,sys,types;"
        "m=types.ModuleType('_crucible_exact_pytest_observer');"
        "m.__file__='<crucible_exact_pytest_observer>';"
        "sys.modules[m.__name__]=m;"
        f"exec(compile(base64.b64decode('{encoded}'),m.__file__,'exec'),m.__dict__);"
        f"m.supervise_exact('{encoded}',sys.argv[1],sys.argv[1:])"
    )


def exact_pytest_command(
    node_id: str,
    *,
    python_prefix: Sequence[str],
) -> list[str]:
    """Build an exact pytest probe whose outcome is a reserved exit code."""

    if not node_id or not python_prefix:
        raise ValueError("exact pytest observation requires a node and Python")
    return [
        *python_prefix,
        "-B",
        "-I",
        "-c",
        _pytest_observer_inline_script(),
        node_id,
        "-q",
        "--tb=short",
        "--capture=fd",
    ]


def filter_test_observations(
    observations: list[dict[str, object]],
    attack_ids: object,
    *,
    language: str,
) -> list[dict[str, object]]:
    """Keep a bounded, one-to-one packet for the canonical generated tests."""

    pairs = canonical_attack_test_ids(attack_ids, language)
    allowed = {test_id for _attack_id, test_id in pairs}
    filtered: list[dict[str, object]] = []
    seen: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        observed_language = observation.get("language")
        observed_id = observation.get("id")
        status = observation.get("status")
        line = observation.get("line")
        if (
            observed_language != language
            or not isinstance(observed_id, str)
            or observed_id.rsplit("::", 1)[-1] not in allowed
            or status not in {
                "passed", "failed", "not_observed", "runner_error",
            }
            or not isinstance(line, str)
        ):
            continue
        terminal = observed_id.rsplit("::", 1)[-1]
        if terminal in seen:
            raise ValueError(
                f"duplicate exact observation for canonical test {terminal!r}"
            )
        seen.add(terminal)
        if "failure" in observation:
            failure = observation["failure"]
            if language != "python" or status not in {"failed", "runner_error"}:
                raise ValueError(
                    "pytest failure metadata requires a failed Python observation"
                )
            normalized_failure = validate_pytest_failure(
                failure,
                expected_node_id=observed_id,
            )
            if (
                (status == "failed" and normalized_failure["phase"] != "call")
                or (
                    status == "runner_error"
                    and normalized_failure["phase"] == "call"
                )
            ):
                raise ValueError(
                    "pytest failure metadata phase contradicts observation status"
                )
            filtered.append({**observation, "failure": normalized_failure})
        else:
            filtered.append(observation)

    encoded_size = len(json.dumps(filtered, ensure_ascii=True).encode("utf-8"))
    if (
        len(filtered) > min(_MAX_OBSERVATION_RECORDS, len(allowed))
        or encoded_size > _MAX_OBSERVATION_BYTES
    ):
        raise ValueError(
            "breach per-test observation envelope exceeded the trusted cap: "
            f"records={len(filtered)}, bytes={encoded_size}"
        )
    return filtered


__all__ = [
    "AttackObservationStatus",
    "PYTEST_FAILURE_SCHEMA",
    "PYTEST_OBSERVER_FAILED",
    "PYTEST_OBSERVER_NOT_OBSERVED",
    "PYTEST_OBSERVER_PASSED",
    "PYTEST_OBSERVER_RUNNER_ERROR",
    "PytestFailureException",
    "PytestFailureFrame",
    "PytestFailureMetadata",
    "PytestFailureTest",
    "canonical_attack_test_id",
    "canonical_attack_test_ids",
    "exact_pytest_command",
    "filter_test_observations",
    "pytest_observer_failure",
    "pytest_observer_status",
    "validate_pytest_failure",
]
