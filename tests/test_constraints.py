# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Current constraint authority survives the retired product removal."""

from __future__ import annotations

import json

import pytest
from assay import Spec
from assay.canonical import parse, serialize
from assay.corpus.codec import RecordDecodeError, decode_dataclass
from assay.types.spec import UnknownConstraintKind

from crucible.constraints import (
    AuthoringFinalizationConstraint,
    AuthoringInstructionConstraint,
    LanguagesConstraint,
)


def _spec_payload(constraints: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_context": {
            "project": "native-brain",
            "repo_url": "https://example.invalid/brain.git",
            "base_commit": "a" * 40,
            "runners": {},
        },
        "intent": "Build the native brain with ordinary language runners.",
        "constraints": constraints,
        "done_criteria": [],
    }


def test_current_constraint_kinds_round_trip_through_spec_authority() -> None:
    source = _spec_payload([
        {"kind": "languages", "languages": ["rust", "cpp"]},
        {
            "kind": "authoring_instruction",
            "instruction": "Retain the supported Python brain.",
            "policy_owner": "native-brain",
            "policy_rule_id": "runtime-boundary",
            "policy_revision": 1,
        },
        {
            "kind": "authoring_finalization",
            "slice_id": "native-build",
            "materializes_paths": ["reports/native-build.json"],
            "gate": "dev_verify",
        },
    ])
    authored = Spec.from_yaml(json.dumps(source))
    canonical = serialize(authored)
    decoded = decode_dataclass(parse(canonical), Spec)

    assert tuple(type(item) for item in decoded.constraints) == (
        LanguagesConstraint,
        AuthoringInstructionConstraint,
        AuthoringFinalizationConstraint,
    )
    assert decoded.constraints[0].languages == ("rust", "cpp")
    assert decoded.constraints[1].policy_rule_id == "runtime-boundary"
    assert decoded.constraints[2].materializes_paths == ("reports/native-build.json",)
    assert decoded.constraints[2].gate == "dev_verify"
    assert serialize(decoded) == canonical


@pytest.mark.parametrize("kind", ("sil_ov_runner_authority", "sil_ov_runner_authority_v2"))
def test_removed_constraint_kinds_cannot_become_executable_spec_authority(kind: str) -> None:
    with pytest.raises(UnknownConstraintKind, match="unknown constraint kind"):
        Spec.from_yaml(json.dumps(_spec_payload([{"kind": kind}])))

    current = Spec.from_yaml(json.dumps(_spec_payload([])))
    archived_payload = parse(serialize(current))
    archived_payload["constraints"] = [{"kind": kind}]
    with pytest.raises(RecordDecodeError, match="kind"):
        decode_dataclass(archived_payload, Spec)
