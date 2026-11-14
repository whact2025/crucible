# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Workflow-owned transport for plan-declared verification surfaces.

The generic build/verify layer knows which repository paths a slice may
modify, but it must not guess which of those paths are tests.  It therefore
transports a small, canonical surface to the selected runner.  Product-owned
runners may conservatively interpret that surface using their native test
conventions.

The surface is derived from the accepted ``ImplementationDirective``.  It is
never valid as canonical project runner configuration or caller-supplied
runtime metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import PurePosixPath


DECLARED_VERIFICATION_SURFACE_KEY = "declared_verification_surface"
DECLARED_VERIFICATION_SURFACE_SCHEMA = (
    "crucible.declared-verification-surface.v1"
)
ACCEPTED_VERIFICATION_REFERENCE_KEY = (
    "accepted_declared_verification_reference"
)
DECLARED_VERIFICATION_REFERENCE_OUTPUT_KEY = (
    "declared_verification_reference"
)
ACCEPTED_VERIFICATION_REFERENCE_SCHEMA = (
    "crucible.accepted-declared-verification-reference.v1"
)

_MAX_SLICES = 128
_MAX_PATHS_PER_FIELD = 512
_MAX_PATH_CHARS = 4_096
_MAX_SLICE_ID_CHARS = 256
_MAX_REFERENCE_OUTCOMES = 64
_MAX_REFERENCE_BYTES = 1_048_576
_MAX_REFERENCE_NAME_CHARS = 128
_MAX_JSON_DEPTH = 32
_MAX_REFERENCE_JSON_NODES = 65_536

# A surface is embedded verbatim in an accepted reference.  Give it only half
# of the aggregate reference budget so every max-valid surface can still be
# wrapped with the schema/name/digest envelope, while the remaining half is
# available for the bounded outcome roster and product evidence.  The durable
# workflow checkpoint permits 1,100,000 bytes, leaving a further 51,424 bytes
# around a max-valid reference for its tier and authority envelope.
_MAX_SURFACE_BYTES = _MAX_REFERENCE_BYTES // 2
_MAX_SURFACE_JSON_NODES = _MAX_REFERENCE_JSON_NODES // 2

_JSON_COMPLEXITY_ERROR = (
    "declared verification JSON is too deeply nested or has too many nodes"
)


@dataclass(frozen=True, slots=True)
class DeclaredVerificationSlice:
    """One normalized, planner-authorized slice surface."""

    slice_id: str
    files_in_scope: tuple[str, ...]
    creates_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AcceptedVerificationOutcome:
    """One accepted exact-path result bound to its focused command."""

    path: str
    command_sha256: str
    outcome: dict[str, object]


@dataclass(frozen=True, slots=True)
class AcceptedVerificationReference:
    """A bounded accepted checkpoint for predecessor focused tests."""

    language: str
    parser_kind: str
    predecessor_surface: dict[str, object]
    predecessor_surface_sha256: str
    outcomes: tuple[AcceptedVerificationOutcome, ...]


def normalize_declared_path(value: object) -> str:
    """Return one bounded repository-relative POSIX path.

    Backslashes are normalized for cross-platform plan producers before the
    absolute/traversal checks.  Resolution is deliberately lexical: the
    product runner must separately prove that an on-disk path remains inside
    its sandbox and is not a symlink.
    """

    if not isinstance(value, str):
        raise ValueError("declared verification path must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("declared verification path contains invalid Unicode") from exc
    if (
        not value
        or len(value) > _MAX_PATH_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("declared verification path is empty or too large")
    rendered = value.replace("\\", "/")
    path = PurePosixPath(rendered)
    windows_absolute = (
        len(rendered) >= 2
        and rendered[0].isalpha()
        and rendered[1] == ":"
    )
    if path.is_absolute() or windows_absolute or ".." in path.parts:
        raise ValueError(
            "declared verification path must stay repository-relative"
        )
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise ValueError("declared verification path cannot name the root")
    return PurePosixPath(*parts).as_posix()


def _normalize_paths(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"declared verification {field} must be a list")
    if len(values) > _MAX_PATHS_PER_FIELD:
        raise ValueError(f"declared verification {field} is too large")
    return tuple(dict.fromkeys(normalize_declared_path(value) for value in values))


def _slice_from_directive(directive: object) -> DeclaredVerificationSlice:
    slice_id = getattr(directive, "slice_id", None)
    if (
        not isinstance(slice_id, str)
        or not slice_id
        or len(slice_id) > _MAX_SLICE_ID_CHARS
    ):
        raise ValueError("declared verification slice_id is invalid")
    return DeclaredVerificationSlice(
        slice_id=slice_id,
        files_in_scope=_normalize_paths(
            getattr(directive, "files_in_scope", None),
            field="files_in_scope",
        ),
        creates_paths=_normalize_paths(
            getattr(directive, "creates_paths", None),
            field="creates_paths",
        ),
    )


def surface_from_directives(directives: Iterable[object]) -> dict[str, object]:
    """Build the canonical JSON surface from trusted directive objects."""

    bounded = tuple(islice(directives, _MAX_SLICES + 1))
    if not bounded or len(bounded) > _MAX_SLICES:
        raise ValueError(
            "declared verification requires a bounded non-empty slice list"
        )
    normalized = tuple(_slice_from_directive(item) for item in bounded)
    if len({item.slice_id for item in normalized}) != len(normalized):
        raise ValueError("declared verification slice_ids must be unique")
    payload = _surface_payload(normalized)
    _canonical_json_bytes(
        payload,
        max_bytes=_MAX_SURFACE_BYTES,
        max_nodes=_MAX_SURFACE_JSON_NODES,
    )
    return payload


def _surface_payload(
    slices: Iterable[DeclaredVerificationSlice],
) -> dict[str, object]:
    return {
        "schema_version": DECLARED_VERIFICATION_SURFACE_SCHEMA,
        "slices": [
            {
                "slice_id": item.slice_id,
                "files_in_scope": list(item.files_in_scope),
                "creates_paths": list(item.creates_paths),
            }
            for item in slices
        ],
    }


def _validate_json_value(value: object, *, max_nodes: int) -> None:
    """Reject non-portable or recursively unbounded JSON data."""

    # Iterator frames make traversal memory proportional to nesting depth,
    # rather than to the width of the largest list or object.  Checking a
    # container's direct width before opening its iterator also rejects a
    # pre-existing over-wide value without allocating a second wide stack.
    stack: list[tuple[Iterator[object], int]] = [(iter((value,)), 0)]
    nodes = 0
    while stack:
        iterator, depth = stack[-1]
        try:
            candidate = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if nodes > max_nodes or depth > _MAX_JSON_DEPTH:
            raise ValueError(_JSON_COMPLEXITY_ERROR)
        if candidate is None or type(candidate) in {bool, int, str}:
            if type(candidate) is int and not -(2**63) <= candidate < 2**63:
                raise ValueError("declared verification integer is out of range")
            if type(candidate) is str:
                try:
                    candidate.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "declared verification JSON contains invalid Unicode"
                    ) from exc
            continue
        if type(candidate) is list:
            if len(candidate) > max_nodes - nodes:
                raise ValueError(_JSON_COMPLEXITY_ERROR)
            stack.append((iter(candidate), depth + 1))
            continue
        if type(candidate) is dict:
            if len(candidate) > max_nodes - nodes:
                raise ValueError(_JSON_COMPLEXITY_ERROR)
            for key in candidate:
                if type(key) is not str:
                    raise ValueError(
                        "declared verification object keys must be strings"
                    )
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "declared verification JSON contains invalid Unicode"
                    ) from exc
            stack.append((iter(candidate.values()), depth + 1))
            continue
        raise ValueError("declared verification requires strict JSON data")


def _canonical_json_bytes(
    value: object,
    *,
    max_bytes: int,
    max_nodes: int,
) -> bytes:
    _validate_json_value(value, max_nodes=max_nodes)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(
            "accepted verification reference must be bounded JSON data"
        ) from exc
    if len(encoded) > max_bytes:
        raise ValueError("declared verification JSON exceeds the byte limit")
    return encoded


def _reference_name(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_REFERENCE_NAME_CHARS
        or any(not character.isprintable() for character in value)
    ):
        raise ValueError(f"accepted verification {field} is invalid")
    return value


def parse_accepted_verification_reference(
    value: object,
) -> AcceptedVerificationReference:
    """Strictly parse one workflow-owned accepted outcome checkpoint.

    The product runner separately decodes each product-specific outcome.  This
    common boundary owns transport shape, surface/path identity, command
    identity, and global memory/disk bounds.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "language",
        "parser_kind",
        "predecessor_surface",
        "predecessor_surface_sha256",
        "outcomes",
    }:
        raise ValueError("accepted verification reference has invalid fields")
    payload = dict(value)
    _canonical_json_bytes(
        payload,
        max_bytes=_MAX_REFERENCE_BYTES,
        max_nodes=_MAX_REFERENCE_JSON_NODES,
    )
    if payload.get("schema_version") != ACCEPTED_VERIFICATION_REFERENCE_SCHEMA:
        raise ValueError("accepted verification reference schema is unsupported")
    language = _reference_name(payload.get("language"), field="language")
    parser_kind = _reference_name(
        payload.get("parser_kind"), field="parser_kind"
    )
    surface_slices = parse_declared_verification_surface(
        payload.get("predecessor_surface")
    )
    predecessor_surface = _surface_payload(surface_slices)
    surface_digest = hashlib.sha256(
        _canonical_json_bytes(
            predecessor_surface,
            max_bytes=_MAX_SURFACE_BYTES,
            max_nodes=_MAX_SURFACE_JSON_NODES,
        )
    ).hexdigest()
    if payload.get("predecessor_surface_sha256") != surface_digest:
        raise ValueError("accepted verification predecessor surface mismatch")
    raw_outcomes = payload.get("outcomes")
    if (
        not isinstance(raw_outcomes, list)
        or len(raw_outcomes) > _MAX_REFERENCE_OUTCOMES
    ):
        raise ValueError("accepted verification outcomes must be bounded")
    surface_paths = {
        path
        for item in surface_slices
        for path in (*item.files_in_scope, *item.creates_paths)
    }
    outcomes: list[AcceptedVerificationOutcome] = []
    seen_paths: set[str] = set()
    for raw in raw_outcomes:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "command_sha256",
            "outcome",
        }:
            raise ValueError(
                "accepted verification outcome entry has invalid fields"
            )
        path = normalize_declared_path(raw.get("path"))
        command_sha256 = raw.get("command_sha256")
        outcome = raw.get("outcome")
        if path not in surface_paths or path in seen_paths:
            raise ValueError(
                "accepted verification outcome path is duplicated or outside "
                "its predecessor surface"
            )
        if (
            not isinstance(command_sha256, str)
            or len(command_sha256) != 64
            or any(character not in "0123456789abcdef" for character in command_sha256)
        ):
            raise ValueError(
                "accepted verification focused command digest is invalid"
            )
        if not isinstance(outcome, Mapping):
            raise ValueError("accepted verification outcome payload is invalid")
        # Round-trip to an ordinary dict so callers cannot retain a custom
        # Mapping implementation past this trust boundary.
        try:
            normalized_outcome = json.loads(
                _canonical_json_bytes(
                    dict(outcome),
                    max_bytes=_MAX_REFERENCE_BYTES,
                    max_nodes=_MAX_REFERENCE_JSON_NODES,
                ).decode("utf-8")
            )
        except (RecursionError, json.JSONDecodeError) as exc:
            raise ValueError(
                "accepted verification outcome could not be normalized"
            ) from exc
        seen_paths.add(path)
        outcomes.append(AcceptedVerificationOutcome(
            path=path,
            command_sha256=command_sha256,
            outcome=normalized_outcome,
        ))
    return AcceptedVerificationReference(
        language=language,
        parser_kind=parser_kind,
        predecessor_surface=predecessor_surface,
        predecessor_surface_sha256=surface_digest,
        outcomes=tuple(outcomes),
    )


def accepted_verification_reference_payload(
    *,
    predecessor_surface: object,
    language: str,
    parser_kind: str,
    outcomes: Iterable[tuple[str, str, Mapping[str, object]]],
) -> dict[str, object]:
    """Build and self-validate a canonical accepted reference payload."""

    bounded_outcomes = tuple(islice(outcomes, _MAX_REFERENCE_OUTCOMES + 1))
    if len(bounded_outcomes) > _MAX_REFERENCE_OUTCOMES:
        raise ValueError("accepted verification outcomes must be bounded")
    surface = _surface_payload(
        parse_declared_verification_surface(predecessor_surface)
    )
    payload = {
        "schema_version": ACCEPTED_VERIFICATION_REFERENCE_SCHEMA,
        "language": language,
        "parser_kind": parser_kind,
        "predecessor_surface": surface,
        "predecessor_surface_sha256": hashlib.sha256(
            _canonical_json_bytes(
                surface,
                max_bytes=_MAX_SURFACE_BYTES,
                max_nodes=_MAX_SURFACE_JSON_NODES,
            )
        ).hexdigest(),
        "outcomes": [
            {
                "path": path,
                "command_sha256": command_sha256,
                "outcome": dict(outcome),
            }
            for path, command_sha256, outcome in bounded_outcomes
        ],
    }
    parsed = parse_accepted_verification_reference(payload)
    return {
        "schema_version": ACCEPTED_VERIFICATION_REFERENCE_SCHEMA,
        "language": parsed.language,
        "parser_kind": parsed.parser_kind,
        "predecessor_surface": parsed.predecessor_surface,
        "predecessor_surface_sha256": parsed.predecessor_surface_sha256,
        "outcomes": [
            {
                "path": item.path,
                "command_sha256": item.command_sha256,
                "outcome": item.outcome,
            }
            for item in parsed.outcomes
        ],
    }


def encode_accepted_verification_reference(value: object) -> bytes:
    """Return canonical bounded bytes after strict reference validation."""

    parsed = parse_accepted_verification_reference(value)
    return _canonical_json_bytes(
        {
            "schema_version": ACCEPTED_VERIFICATION_REFERENCE_SCHEMA,
            "language": parsed.language,
            "parser_kind": parsed.parser_kind,
            "predecessor_surface": parsed.predecessor_surface,
            "predecessor_surface_sha256": parsed.predecessor_surface_sha256,
            "outcomes": [
                {
                    "path": item.path,
                    "command_sha256": item.command_sha256,
                    "outcome": item.outcome,
                }
                for item in parsed.outcomes
            ],
        },
        max_bytes=_MAX_REFERENCE_BYTES,
        max_nodes=_MAX_REFERENCE_JSON_NODES,
    )


def parse_declared_verification_surface(
    value: object,
) -> tuple[DeclaredVerificationSlice, ...]:
    """Strictly parse a runner-received surface for defense in depth."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "slices",
    }:
        raise ValueError("declared verification surface has invalid fields")
    payload = dict(value)
    _canonical_json_bytes(
        payload,
        max_bytes=_MAX_SURFACE_BYTES,
        max_nodes=_MAX_SURFACE_JSON_NODES,
    )
    if payload.get("schema_version") != DECLARED_VERIFICATION_SURFACE_SCHEMA:
        raise ValueError("declared verification surface schema is unsupported")
    raw_slices = payload.get("slices")
    if (
        not isinstance(raw_slices, list)
        or not raw_slices
        or len(raw_slices) > _MAX_SLICES
    ):
        raise ValueError("declared verification slices must be bounded")
    parsed: list[DeclaredVerificationSlice] = []
    for raw in raw_slices:
        if not isinstance(raw, Mapping) or set(raw) != {
            "slice_id",
            "files_in_scope",
            "creates_paths",
        }:
            raise ValueError("declared verification slice has invalid fields")
        slice_id = raw.get("slice_id")
        if (
            not isinstance(slice_id, str)
            or not slice_id
            or len(slice_id) > _MAX_SLICE_ID_CHARS
        ):
            raise ValueError("declared verification slice_id is invalid")
        parsed.append(DeclaredVerificationSlice(
            slice_id=slice_id,
            files_in_scope=_normalize_paths(
                raw.get("files_in_scope"), field="files_in_scope"
            ),
            creates_paths=_normalize_paths(
                raw.get("creates_paths"), field="creates_paths"
            ),
        ))
    if len({item.slice_id for item in parsed}) != len(parsed):
        raise ValueError("declared verification slice_ids must be unique")
    return tuple(parsed)


__all__ = [
    "ACCEPTED_VERIFICATION_REFERENCE_KEY",
    "ACCEPTED_VERIFICATION_REFERENCE_SCHEMA",
    "AcceptedVerificationOutcome",
    "AcceptedVerificationReference",
    "DECLARED_VERIFICATION_SURFACE_KEY",
    "DECLARED_VERIFICATION_SURFACE_SCHEMA",
    "DECLARED_VERIFICATION_REFERENCE_OUTPUT_KEY",
    "DeclaredVerificationSlice",
    "accepted_verification_reference_payload",
    "encode_accepted_verification_reference",
    "normalize_declared_path",
    "parse_accepted_verification_reference",
    "parse_declared_verification_surface",
    "surface_from_directives",
]
