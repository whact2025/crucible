# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Product-neutral, complete failure sets for bounded repair progress."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping


REPAIR_FAILURE_SET_OUTPUT_KEY = "repair_failure_set"
REPAIR_FAILURE_SET_SCHEMA = "crucible.runner-repair-failure-set.v1"

_MAX_SURFACE_BYTES = 256 * 1024
_MAX_ENVELOPE_BYTES = 1024 * 1024
_MAX_FAILURES = 1_024
_MAX_FAILURE_ID_BYTES = 8_192
_MAX_STATUS_BYTES = 128
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 65_536
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class RepairFailure:
    """One stable failure identity and its unchanged-survivor fingerprint."""

    failure_id: str
    status: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RepairFailureSet:
    """A complete failure set bound to one opaque executable suite surface."""

    surface_sha256: str
    failures: tuple[RepairFailure, ...]


def _validate_json_value(value: object) -> None:
    """Reject non-portable JSON values before recursive library encoding."""

    stack = [(value, 0)]
    nodes = 0
    while stack:
        candidate, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("repair failure-set JSON is too deeply nested")
        if candidate is None or type(candidate) in {bool, int, str}:
            if type(candidate) is int and not -(2**63) <= candidate < 2**63:
                raise ValueError("repair failure-set integer is out of range")
            if type(candidate) is str:
                try:
                    candidate.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "repair failure-set JSON contains invalid Unicode"
                    ) from exc
            continue
        if type(candidate) is list:
            stack.extend((item, depth + 1) for item in candidate)
            continue
        if type(candidate) is dict:
            for key, item in candidate.items():
                if type(key) is not str:
                    raise ValueError(
                        "repair failure-set object keys must be strings"
                    )
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "repair failure-set JSON contains invalid Unicode"
                    ) from exc
                stack.append((item, depth + 1))
            continue
        raise ValueError("repair failure-set surface must use strict JSON types")


def _canonical_json(
    value: object,
    *,
    max_bytes: int = _MAX_SURFACE_BYTES,
) -> bytes:
    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("repair failure-set surface must be JSON data") from exc
    if len(encoded) > max_bytes:
        raise ValueError("repair failure-set JSON exceeds the byte limit")
    return encoded


def _bounded_text(value: object, *, field: str, limit: int) -> str:
    try:
        encoded = value.encode("utf-8") if isinstance(value, str) else b""
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"repair failure-set {field} contains invalid Unicode"
        ) from exc
    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
        or len(encoded) > limit
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError(f"repair failure-set {field} is invalid")
    return value


def build_repair_failure_set(
    *,
    surface: Mapping[str, object],
    failures: tuple[RepairFailure, ...],
) -> dict[str, object]:
    """Build and self-validate a canonical complete failure-set envelope."""

    surface_payload = json.loads(_canonical_json(dict(surface)))
    payload = {
        "schema_version": REPAIR_FAILURE_SET_SCHEMA,
        "complete": True,
        "surface": surface_payload,
        "surface_sha256": (
            "sha256:" + hashlib.sha256(_canonical_json(surface_payload)).hexdigest()
        ),
        "failures": [
            {
                "id": failure.failure_id,
                "status": failure.status,
                "fingerprint": failure.fingerprint,
            }
            for failure in failures
        ],
    }
    parsed = parse_repair_failure_set(payload)
    return {
        **payload,
        "failures": [
            {
                "id": failure.failure_id,
                "status": failure.status,
                "fingerprint": failure.fingerprint,
            }
            for failure in parsed.failures
        ],
    }


def parse_repair_failure_set(value: object) -> RepairFailureSet:
    """Strictly parse one runner-owned complete failure-set envelope."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "complete",
        "surface",
        "surface_sha256",
        "failures",
    }:
        raise ValueError("repair failure-set envelope has invalid fields")
    _canonical_json(dict(value), max_bytes=_MAX_ENVELOPE_BYTES)
    if (
        value.get("schema_version") != REPAIR_FAILURE_SET_SCHEMA
        or value.get("complete") is not True
    ):
        raise ValueError("repair failure-set envelope is incomplete or unsupported")
    surface = value.get("surface")
    if not isinstance(surface, Mapping):
        raise ValueError("repair failure-set surface must be an object")
    surface_payload = dict(surface)
    if not isinstance(surface_payload.get("schema_version"), str) or not (
        surface_payload["schema_version"].strip()
    ):
        raise ValueError("repair failure-set surface schema_version is required")
    surface_bytes = _canonical_json(surface_payload)
    expected_surface_digest = (
        "sha256:" + hashlib.sha256(surface_bytes).hexdigest()
    )
    if value.get("surface_sha256") != expected_surface_digest:
        raise ValueError("repair failure-set surface digest mismatch")
    raw_failures = value.get("failures")
    if (
        not isinstance(raw_failures, list)
        or not raw_failures
        or len(raw_failures) > _MAX_FAILURES
    ):
        raise ValueError("repair failure-set failures must be bounded and nonempty")
    failures: list[RepairFailure] = []
    seen_ids: set[str] = set()
    for raw in raw_failures:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id", "status", "fingerprint",
        }:
            raise ValueError("repair failure-set entry has invalid fields")
        failure_id = _bounded_text(
            raw.get("id"), field="id", limit=_MAX_FAILURE_ID_BYTES,
        )
        status = _bounded_text(
            raw.get("status"), field="status", limit=_MAX_STATUS_BYTES,
        )
        fingerprint = raw.get("fingerprint")
        if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
            raise ValueError("repair failure-set fingerprint is invalid")
        if failure_id in seen_ids:
            raise ValueError("repair failure-set ids must be unique")
        seen_ids.add(failure_id)
        failures.append(RepairFailure(failure_id, status, fingerprint))
    return RepairFailureSet(
        surface_sha256=expected_surface_digest,
        failures=tuple(sorted(failures, key=lambda item: item.failure_id)),
    )
