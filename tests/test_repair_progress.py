# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Strict product-neutral repair-progress envelope contracts."""

from __future__ import annotations

import copy

import pytest

from crucible.common.runners.repair_progress import (
    RepairFailure,
    build_repair_failure_set,
    parse_repair_failure_set,
)


_FINGERPRINT = "sha256:" + "a" * 64


def _envelope() -> dict[str, object]:
    return build_repair_failure_set(
        surface={
            "schema_version": "crucible.test-repair-surface.v1",
            "nested": {"b": [2, 3], "a": 1},
        },
        failures=(
            RepairFailure("failure-b", "FAILED", "sha256:" + "b" * 64),
            RepairFailure("failure-a", "ERROR", _FINGERPRINT),
        ),
    )


def test_repair_failure_set_is_canonical_and_sorts_failures():
    first = _envelope()
    second = build_repair_failure_set(
        surface={
            "nested": {"a": 1, "b": [2, 3]},
            "schema_version": "crucible.test-repair-surface.v1",
        },
        failures=tuple(reversed((
            RepairFailure("failure-a", "ERROR", _FINGERPRINT),
            RepairFailure("failure-b", "FAILED", "sha256:" + "b" * 64),
        ))),
    )

    parsed = parse_repair_failure_set(first)
    assert first["surface_sha256"] == second["surface_sha256"]
    assert [failure.failure_id for failure in parsed.failures] == [
        "failure-a", "failure-b",
    ]


@pytest.mark.parametrize(
    "surface",
    [
        pytest.param({}, id="missing-schema"),
        pytest.param(
            {"schema_version": "x", "value": float("nan")},
            id="nan",
        ),
        pytest.param(
            {"schema_version": "x", "value": float("inf")},
            id="infinity",
        ),
        pytest.param(
            {"schema_version": "x", 1: "numeric-key"},
            id="numeric-key",
        ),
        pytest.param(
            {"schema_version": "x", "value": 1.5},
            id="float",
        ),
    ],
)
def test_repair_failure_set_rejects_nonportable_surfaces(surface):
    with pytest.raises(ValueError):
        build_repair_failure_set(
            surface=surface,
            failures=(RepairFailure("failure", "FAILED", _FINGERPRINT),),
        )


def test_repair_failure_set_rejects_excessive_depth_as_value_error():
    nested: object = "leaf"
    for _ in range(40):
        nested = [nested]

    with pytest.raises(ValueError, match="deeply nested"):
        build_repair_failure_set(
            surface={"schema_version": "x", "nested": nested},
            failures=(RepairFailure("failure", "FAILED", _FINGERPRINT),),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("complete", False, id="incomplete"),
        pytest.param("failures", [], id="empty-failures"),
        pytest.param("surface_sha256", "sha256:" + "0" * 64, id="bad-surface"),
    ],
)
def test_repair_failure_set_rejects_malformed_envelope_fields(field, value):
    payload = copy.deepcopy(_envelope())
    payload[field] = value

    with pytest.raises(ValueError):
        parse_repair_failure_set(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("id", "   ", id="whitespace-id"),
        pytest.param("status", "\ud800", id="surrogate-status"),
        pytest.param("fingerprint", "sha256:" + "A" * 64, id="bad-digest"),
    ],
)
def test_repair_failure_set_rejects_invalid_failure_text(field, value):
    payload = copy.deepcopy(_envelope())
    payload["failures"][0][field] = value

    with pytest.raises(ValueError):
        parse_repair_failure_set(payload)


def test_repair_failure_set_rejects_duplicate_ids_and_extra_fields():
    duplicate = copy.deepcopy(_envelope())
    duplicate["failures"][1]["id"] = duplicate["failures"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        parse_repair_failure_set(duplicate)

    extra = copy.deepcopy(_envelope())
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="invalid fields"):
        parse_repair_failure_set(extra)


def test_repair_failure_set_enforces_aggregate_byte_limit():
    failures = tuple(
        RepairFailure(
            f"failure-{index:04d}-" + "x" * 8_000,
            "FAILED",
            _FINGERPRINT,
        )
        for index in range(140)
    )

    with pytest.raises(ValueError, match="byte limit"):
        build_repair_failure_set(
            surface={"schema_version": "x"},
            failures=failures,
        )
