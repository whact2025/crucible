# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Historical wire compatibility and read-only reference authority boundaries."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from assay.types.dag import (
    ImplementationDirective,
    ReadOnlySourceReference,
    Slice,
    Tier,
    TierDAG,
)
from crucible.workflows import breach


# Frozen using pre-reference Crucible 8a1aab81 and Assay 654894737.
# V1 was independently parsed by that reader. V2/V3 are its exact encoder
# output, including empty requirement_refs and decomposed (not NFC) Unicode.
_PACKETS = {
    "v1": (
        b'{"issue_text":"Keep cafe\\u0301 exact.",'
        b'"schema_version":"crucible.breach-authority.v1","slices":[{"directive":'
        b'{"blast_radius":{"max_files":2},"creates_paths":["tests/test_app.py"],'
        b'"files_in_scope":["src/app.py"],"intent":"Preserve cafe\\u0301 behavior.",'
        b'"mutation_regions":[],"slice_id":"fix","verify_mode":"tests"},'
        b'"slice_id":"fix"}],"tier_id":"t0"}',
        "sha256:80af1230b5643ad5c8c551d348e119d0bcdcddf582be693b04e8ea6b7cf531c2",
    ),
    "v2": (
        b'{"issue_text":"Keep cafe\\u0301 exact.",'
        b'"schema_version":"crucible.breach-authority.v2","slices":[{"directive":'
        b'{"blast_radius":{"max_files":2},"creates_paths":["tests/test_app.py"],'
        b'"files_in_scope":["src/app.py"],"intent":"Preserve cafe\\u0301 behavior.",'
        b'"mutation_regions":[],"requirement_refs":[],"slice_id":"fix",'
        b'"verify_mode":"tests"},"slice_id":"fix"}],"tier_id":"t0"}',
        "sha256:06418fac212ec5c8716e437a97ef9cc6adec3273eaea7b3b17ca6ed8aee124d6",
    ),
    "v3": (
        b'{"authority_kind":"selected_slices","issue_text":"Keep cafe\\u0301 exact.",'
        b'"schema_version":"crucible.breach-authority.v3","slices":[{"directive":'
        b'{"blast_radius":{"max_files":2},"creates_paths":["tests/test_app.py"],'
        b'"files_in_scope":["src/app.py"],"intent":"Preserve cafe\\u0301 behavior.",'
        b'"mutation_regions":[],"requirement_refs":[],"slice_id":"fix",'
        b'"verify_mode":"tests"},"slice_id":"fix"}],"tier_id":"t0"}',
        "sha256:ece4cb7e9f1f13466648b9eca676f9be2b113af1038273fefe4fcbdd4097cc7b",
    ),
}
_REFERENCE = {
    "reference_id": "context",
    "path": "docs/context.md",
    "sha256": "sha256:" + "a" * 64,
    "start_line": 2,
    "end_line": 4,
}


def _directive(*references):
    return ImplementationDirective(
        slice_id="fix",
        intent="Preserve cafe\u0301 behavior.",
        files_in_scope=("src/app.py",),
        creates_paths=("tests/test_app.py",),
        verify_mode="tests",
        blast_radius={"max_files": 2},
        read_only_references=tuple(references),
    )


def _spec(directive, *, selected=False):
    slices = (Slice(id=directive.slice_id, directive=directive),)
    if selected:
        sibling = dataclasses.replace(directive, slice_id="sibling")
        slices += (Slice(id=sibling.slice_id, directive=sibling),)
    return SimpleNamespace(tier_dag=TierDAG(tiers=(Tier(
        tier_id="t0", slices=slices, depends_on=(),
    ),)))


def _encode(directive, version):
    return breach.encode_breach_authority(
        tier_id="t0",
        issue_text="Keep cafe\u0301 exact.",
        slice_directives=(directive,),
        authority_kind="selected_slices" if version == "v3" else "canonical_tier",
    )


def _parse(encoded):
    return breach._parse_breach_authority_payload(
        json.loads(encoded),
        encoded=encoded,
        digest=breach.breach_authority_digest(encoded),
    )


def _payload_with_references(version, references):
    payload = json.loads(_PACKETS[version][0])
    payload["slices"][0]["directive"]["read_only_references"] = references
    return payload


@pytest.mark.parametrize("version", _PACKETS)
def test_historical_packet_bytes_digest_and_payload_remain_exact(version, tmp_path):
    encoded, digest = _PACKETS[version]
    assert "sha256:" + hashlib.sha256(encoded).hexdigest() == digest
    payload = json.loads(encoded)
    before = copy.deepcopy(payload)
    authority = breach._parse_breach_authority_payload(
        payload, encoded=encoded, digest=digest,
    )
    assert payload == before
    assert authority.payload() == before
    assert authority.encoded == encoded
    assert authority.digest == digest
    assert authority.slices[0].directive["read_only_references"] == []
    breach._validate_authority_canonical_tier(authority, _spec(_directive()))

    path = tmp_path / "authority.json"
    path.write_bytes(encoded)
    loaded = breach._load_breach_authority(
        authority_file=path, authority_digest_value=digest,
        issue_text="Keep cafe\u0301 exact.", diff_files=("src/app.py",),
    )
    assert loaded == authority
    assert path.read_bytes() == encoded


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_empty_references_preserve_pre_reference_encoder_bytes(version):
    assert _encode(_directive(), version) == _PACKETS[version]


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_explicit_empty_wire_references_match_absent_canonical_field(version):
    payload = _payload_with_references(version, [])
    encoded = json.dumps(payload).encode()
    authority = _parse(encoded)
    breach._validate_authority_canonical_tier(authority, _spec(_directive()))
    assert authority.encoded == encoded
    assert authority.payload() == payload


@pytest.mark.parametrize("references", [[], [_REFERENCE]])
def test_v1_does_not_acquire_a_new_wire_field(references):
    with pytest.raises(ValueError, match="invalid fields"):
        _parse(json.dumps(_payload_with_references("v1", references)).encode())


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_nonempty_references_roundtrip_and_stay_outside_write_authority(version, tmp_path):
    reference = ReadOnlySourceReference(**_REFERENCE)
    directive = _directive(reference)
    encoded, digest = _encode(directive, version)
    authority = _parse(encoded)
    assert authority.slices[0].directive["read_only_references"] == [_REFERENCE]
    breach._validate_authority_canonical_tier(
        authority, _spec(directive, selected=version == "v3"),
    )
    assert authority.slices[0].scope == ("src/app.py", "tests/test_app.py")
    files, creates, radius, regions = breach._defender_authority_constraints(
        authority, ("src/app.py",), language="python",
    )
    assert files == ("src/app.py",)
    assert creates == ("tests/test_app.py",)
    assert radius == {"max_files": 2}
    assert regions == ()

    path = tmp_path / "authority.json"
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="does not cover diff paths"):
        breach._load_breach_authority(
            authority_file=path, authority_digest_value=digest,
            issue_text="Keep cafe\u0301 exact.", diff_files=("docs/context.md",),
        )
    changed = json.loads(encoded)
    changed["slices"][0]["directive"]["read_only_references"][0]["path"] = "other.md"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest mismatch"):
        breach._load_breach_authority(
            authority_file=path, authority_digest_value=digest,
            issue_text="Keep cafe\u0301 exact.", diff_files=("src/app.py",),
        )


@pytest.mark.parametrize("version", _PACKETS)
def test_historical_empty_authority_cannot_bind_new_canonical_references(version):
    authority = _parse(_PACKETS[version][0])
    with pytest.raises(ValueError, match="canonical active tier"):
        breach._validate_authority_canonical_tier(
            authority, _spec(_directive(ReadOnlySourceReference(**_REFERENCE))),
        )


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("field,value", [
    ("reference_id", "renamed"),
    ("path", "docs/other.md"),
    ("sha256", "sha256:" + "b" * 64),
    ("start_line", 3),
    ("end_line", 5),
])
def test_every_reference_field_is_bound_to_the_canonical_plan(version, field, value):
    reference = ReadOnlySourceReference(**_REFERENCE)
    authority = _parse(_encode(_directive(reference), version)[0])
    changed = dataclasses.replace(reference, **{field: value})
    with pytest.raises(ValueError, match="canonical active tier"):
        breach._validate_authority_canonical_tier(
            authority, _spec(_directive(changed), selected=version == "v3"),
        )


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("field,value", [
    ("extra", "not a supported reference field"),
    ("reference_id", ""),
    ("reference_id", "a" * 65),
    ("reference_id", True),
    ("path", "/absolute"),
    ("path", "../parent"),
    ("path", "docs//file"),
    ("path", "docs/./file"),
    ("path", "docs\\file"),
    ("path", "docs/*.md"),
    ("path", "docs/\x00file"),
    ("path", "docs/\ud800file"),
    ("path", "x" * 4097),
    ("sha256", "sha256:" + "A" * 64),
    ("sha256", "a" * 64),
    ("start_line", True),
    ("start_line", 1.5),
    ("start_line", "2"),
    ("start_line", 0),
    ("end_line", False),
    ("end_line", 1),
    ("end_line", "4"),
])
def test_reference_records_are_closed_and_typed(version, field, value):
    malformed = {**_REFERENCE, field: value}
    encoded = json.dumps(_payload_with_references(version, [malformed])).encode()
    with pytest.raises(ValueError, match="read_only_references"):
        _parse(encoded)


@pytest.mark.parametrize("references", [None, {}, "reference", [None], [[]], [{}]])
def test_reference_container_and_missing_fields_fail_closed(references):
    with pytest.raises(ValueError, match="read_only_references"):
        _parse(json.dumps(_payload_with_references("v2", references)).encode())


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_duplicate_reference_ids_and_unknown_directive_fields_are_rejected(version):
    refs = [_REFERENCE, {**_REFERENCE, "path": "other.md"}]
    with pytest.raises(ValueError, match="duplicate read-only reference IDs"):
        _parse(json.dumps(_payload_with_references(version, refs)).encode())
    payload = _payload_with_references(version, [_REFERENCE])
    payload["slices"][0]["directive"]["unreviewed_field"] = []
    with pytest.raises(ValueError, match="invalid fields"):
        _parse(json.dumps(payload).encode())


@pytest.mark.parametrize("count", [64, 65])
def test_reference_count_is_bounded_across_slices(count):
    payload = _payload_with_references("v2", [
        {**_REFERENCE, "reference_id": f"reference-{i}"} for i in range(32)
    ])
    second = copy.deepcopy(payload["slices"][0])
    second["slice_id"] = second["directive"]["slice_id"] = "second"
    second["directive"]["read_only_references"] = [
        {**_REFERENCE, "reference_id": f"reference-{i}"} for i in range(32, count)
    ]
    payload["slices"].append(second)
    encoded = json.dumps(payload).encode()
    if count == 65:
        with pytest.raises(ValueError, match="reference count limit"):
            _parse(encoded)
    else:
        assert len(_parse(encoded).slices) == 2
