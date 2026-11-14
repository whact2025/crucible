# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adversarial contracts for declared-verification JSON transport."""

from __future__ import annotations

import copy
import json
import tracemalloc
from types import SimpleNamespace

import pytest

from crucible.common.runners import declared_verification as transport
from crucible.workflows import _build_verify as build_verify


_COMMAND_SHA256 = "a" * 64
_TEST_PATH = "tests/test_contract.py"


def _surface() -> dict[str, object]:
    return {
        "schema_version": transport.DECLARED_VERIFICATION_SURFACE_SCHEMA,
        "slices": [
            {
                "slice_id": "contract",
                "files_in_scope": [_TEST_PATH],
                "creates_paths": [],
            }
        ],
    }


def _reference(outcome: dict[object, object]) -> dict[str, object]:
    return transport.accepted_verification_reference_payload(
        predecessor_surface=_surface(),
        language="python",
        parser_kind="pytest",
        outcomes=((_TEST_PATH, _COMMAND_SHA256, outcome),),
    )


class _CountingIterable:
    def __init__(self, count, factory) -> None:
        self.count = count
        self.factory = factory
        self.consumed = 0

    def __iter__(self):
        for index in range(self.count):
            self.consumed += 1
            yield self.factory(index)


def test_declared_verification_canonicalizes_nested_outcome_order() -> None:
    first = _reference(
        {
            "z": [3, 2, 1],
            "a": {"right": True, "left": None},
        }
    )
    second = _reference(
        {
            "a": {"left": None, "right": True},
            "z": [3, 2, 1],
        }
    )

    assert transport.encode_accepted_verification_reference(first) == (
        transport.encode_accepted_verification_reference(second)
    )
    assert transport.parse_accepted_verification_reference(first) == (
        transport.parse_accepted_verification_reference(second)
    )


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param({"value": float("nan")}, id="nan"),
        pytest.param({"value": float("inf")}, id="infinity"),
        pytest.param({"value": 1.5}, id="float"),
        pytest.param({"nested": {1: "numeric-key"}}, id="numeric-key"),
        pytest.param({"value": "\ud800"}, id="invalid-unicode"),
        pytest.param({"value": ("tuple",)}, id="non-json-container"),
    ],
)
def test_declared_verification_builder_rejects_nonportable_json(
    outcome,
) -> None:
    with pytest.raises(ValueError):
        _reference(outcome)


@pytest.mark.parametrize(
    "invalid_slice",
    [
        pytest.param(
            {
                "slice_id": "contract",
                "files_in_scope": [float("nan")],
                "creates_paths": [],
            },
            id="nan",
        ),
        pytest.param(
            {
                "slice_id": "contract",
                "files_in_scope": [_TEST_PATH],
                "creates_paths": [],
                1: "numeric-key",
            },
            id="numeric-key",
        ),
        pytest.param(
            {
                "slice_id": "\ud800",
                "files_in_scope": [_TEST_PATH],
                "creates_paths": [],
            },
            id="invalid-unicode",
        ),
    ],
)
def test_declared_verification_surface_parser_rejects_nonportable_json(
    invalid_slice,
) -> None:
    surface = _surface()
    surface["slices"] = [invalid_slice]

    with pytest.raises(ValueError):
        transport.parse_declared_verification_surface(surface)


def test_declared_verification_parser_rejects_excess_depth_and_nodes() -> None:
    deeply_nested: object = "leaf"
    for _ in range(40):
        deeply_nested = [deeply_nested]
    with pytest.raises(ValueError, match="deeply nested"):
        _reference({"nested": deeply_nested})

    too_many_nodes = [None] * 65_536
    with pytest.raises(ValueError, match="deeply nested"):
        _reference({"nodes": too_many_nodes})


def test_declared_verification_bounds_directive_consumption_at_max_plus_one() -> None:
    directives = _CountingIterable(
        1_000,
        lambda index: SimpleNamespace(
            slice_id=f"slice-{index}",
            files_in_scope=[],
            creates_paths=[],
        ),
    )

    with pytest.raises(ValueError, match="bounded non-empty slice list"):
        transport.surface_from_directives(directives)

    assert directives.consumed == transport._MAX_SLICES + 1


def test_declared_verification_bounds_outcome_consumption_at_max_plus_one() -> None:
    outcomes = _CountingIterable(
        1_000,
        lambda _index: (_TEST_PATH, _COMMAND_SHA256, {}),
    )

    with pytest.raises(ValueError, match="outcomes must be bounded"):
        transport.accepted_verification_reference_payload(
            predecessor_surface=_surface(),
            language="python",
            parser_kind="pytest",
            outcomes=outcomes,
        )

    assert outcomes.consumed == transport._MAX_REFERENCE_OUTCOMES + 1


def test_declared_verification_wide_list_does_not_allocate_a_wide_stack() -> None:
    wide = [None] * (transport._MAX_REFERENCE_JSON_NODES + 1)
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="too many nodes"):
            transport._validate_json_value(
                wide,
                max_nodes=transport._MAX_REFERENCE_JSON_NODES,
            )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 1_000_000


def test_max_byte_surface_wraps_in_reference_and_checkpoint() -> None:
    paths = []
    for index in range(128):
        prefix = f"tests/{index:03d}/"
        paths.append(prefix + "x" * (4_092 - len(prefix)))
    directive = SimpleNamespace(
        slice_id="s",
        files_in_scope=paths,
        creates_paths=[],
    )

    surface = transport.surface_from_directives((directive,))
    encoded_surface = json.dumps(
        surface,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(encoded_surface) == transport._MAX_SURFACE_BYTES

    max_name = "\N{GRINNING FACE}" * transport._MAX_REFERENCE_NAME_CHARS
    reference = transport.accepted_verification_reference_payload(
        predecessor_surface=surface,
        language=max_name,
        parser_kind=max_name,
        outcomes=(),
    )
    assert len(transport.encode_accepted_verification_reference(reference)) < (
        transport._MAX_REFERENCE_BYTES
    )

    checkpoint = build_verify.encode_accepted_verification_checkpoint(
        reference,
        tier_id="\N{GRINNING FACE}" * 256,
        spec_execution_authority_hash="sha256:" + "a" * 64,
    )
    assert len(checkpoint) < (build_verify._ACCEPTED_VERIFICATION_CHECKPOINT_MAX_BYTES)


def test_max_node_surface_wraps_in_reference() -> None:
    paths = [f"p/{index:03x}" for index in range(512)]
    directives = [
        SimpleNamespace(
            slice_id=f"slice-{index}",
            files_in_scope=paths,
            creates_paths=paths,
        )
        for index in range(31)
    ]
    directives.append(
        SimpleNamespace(
            slice_id="slice-31",
            files_in_scope=paths,
            creates_paths=paths[:381],
        )
    )

    surface = transport.surface_from_directives(directives)
    reference = transport.accepted_verification_reference_payload(
        predecessor_surface=surface,
        language="python",
        parser_kind="pytest",
        outcomes=(),
    )
    transport.encode_accepted_verification_reference(reference)

    directives[-1].creates_paths.append("one-node-too-many")
    with pytest.raises(ValueError, match="too many nodes"):
        transport.surface_from_directives(directives)


def test_declared_verification_enforces_aggregate_byte_limit() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        _reference({"blob": "x" * 1_048_576})


def test_declared_verification_parser_rejects_mutated_nonportable_outcome() -> None:
    payload = _reference({"result": "passed"})
    mutated = copy.deepcopy(payload)
    mutated["outcomes"][0]["outcome"] = {"value": float("nan")}

    with pytest.raises(ValueError):
        transport.parse_accepted_verification_reference(mutated)


def test_declared_verification_translates_encoder_recursion_error(
    monkeypatch,
) -> None:
    def fail_encoding(*_args, **_kwargs):
        raise RecursionError("synthetic encoder recursion")

    monkeypatch.setattr(transport.json, "dumps", fail_encoding)

    with pytest.raises(ValueError, match="bounded JSON data"):
        transport.parse_declared_verification_surface(_surface())


def test_declared_verification_translates_decoder_recursion_error(
    monkeypatch,
) -> None:
    payload = _reference({"result": "passed"})

    def fail_decoding(*_args, **_kwargs):
        raise RecursionError("synthetic decoder recursion")

    monkeypatch.setattr(transport.json, "loads", fail_decoding)

    with pytest.raises(ValueError, match="could not be normalized"):
        transport.parse_accepted_verification_reference(payload)


def test_declared_path_rejects_invalid_unicode() -> None:
    with pytest.raises(ValueError, match="invalid Unicode"):
        transport.normalize_declared_path("tests/\ud800.py")
