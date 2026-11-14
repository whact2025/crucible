# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from assay.corpus.store import Corpus

from crucible.products.deepswe import mutation_settlement as settlement
from crucible.workflows.architect import ArchitectCheckpoint


def _hook() -> settlement.DeepSWEArchitectCheckpointHook:
    return settlement.DeepSWEArchitectCheckpointHook(
        task_id="task-1",
        ext_id="ext-1",
        expected_base_commit="a" * 40,
        max_patch_bytes=4096,
    )


def _attestation(
    *,
    tier_id: str,
    operation_id: str,
    clean: bool = True,
) -> dict[str, object]:
    payload = b"" if clean else b"diff"
    return {
        "schema_version": settlement.TIER_MUTATION_SETTLEMENT_SCHEMA,
        "settled": True,
        "tier_id": tier_id,
        "operation_id": operation_id,
        "task_id": "task-1",
        "ext_id": "ext-1",
        "workspace_patch_sha256": hashlib.sha256(payload).hexdigest(),
        "workspace_patch_bytes": len(payload),
        "workspace_files_modified": [] if clean else ["src/changed.py"],
    }


@pytest.mark.parametrize("value", [0, -1, True, None, "4096"])
def test_checkpoint_hook_requires_positive_patch_limit(value):
    with pytest.raises(ValueError, match="max_patch_bytes"):
        settlement.DeepSWEArchitectCheckpointHook(
            task_id="task-1",
            ext_id="ext-1",
            expected_base_commit="a" * 40,
            max_patch_bytes=value,
        )


def test_checkpoint_hook_collects_tiers_and_namespaces_final_proof(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[dict[str, object]] = []

    def attest(**kwargs):
        calls.append(kwargs)
        return _attestation(
            tier_id=kwargs["tier_id"],
            operation_id=kwargs["operation_id"],
        )

    monkeypatch.setattr(settlement, "attest_mutation_journal_settlement", attest)
    corpus = Corpus(tmp_path / "corpus")
    hook = _hook()
    assert hook(ArchitectCheckpoint(
        kind="tier",
        corpus=corpus,
        cwd=str(tmp_path),
        tier_id="t0",
        operation_id="tier-0",
        child_run_dir=str(tmp_path / "tier-run"),
        tiers_executed=1,
    )) is None

    result = hook(ArchitectCheckpoint(
        kind="final",
        corpus=corpus,
        cwd=str(tmp_path),
        tier_id="__architect_final__",
        operation_id="architect-final",
        child_run_dir=None,
        tiers_executed=1,
    ))

    extension = result[settlement.DEEPSWE_CHECKPOINT_EXTENSION_KEY]
    aggregate = extension["mutation_settlement"]
    assert extension["schema_version"] == (
        settlement.DEEPSWE_CHECKPOINT_EXTENSION_SCHEMA
    )
    assert aggregate["tiers_executed"] == 1
    assert [item["tier_id"] for item in aggregate["tier_attestations"]] == [
        "t0"
    ]
    assert aggregate["final_attestation"]["tier_id"] == "__architect_final__"
    assert calls[0]["tier_run_dir"] == str(tmp_path / "tier-run")
    assert calls[1]["tier_run_dir"] is None
    assert all(call["expected_base_commit"] == "a" * 40 for call in calls)


def test_checkpoint_hook_turns_dirty_zero_tier_terminal_into_nonproof(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        settlement,
        "attest_mutation_journal_settlement",
        lambda **kwargs: _attestation(
            tier_id=kwargs["tier_id"],
            operation_id=kwargs["operation_id"],
            clean=False,
        ),
    )
    result = _hook()(ArchitectCheckpoint(
        kind="final",
        corpus=Corpus(tmp_path / "corpus"),
        cwd=str(tmp_path),
        tier_id="__architect_final__",
        operation_id="architect-final",
        child_run_dir=None,
        tiers_executed=0,
    ))

    final = result[settlement.DEEPSWE_CHECKPOINT_EXTENSION_KEY][
        "mutation_settlement"
    ]["final_attestation"]
    assert final["settled"] is False
    assert final["schema_version"] == (
        settlement.TIER_MUTATION_SETTLEMENT_SCHEMA
    )
    assert final["error_kind"] == "ValueError"
    assert "unchanged workspace" in final["error_message"]


def test_checkpoint_hook_preserves_attestation_exception_as_nonproof(
    monkeypatch,
    tmp_path: Path,
):
    def fail(**_kwargs):
        raise RuntimeError("attester unavailable")

    monkeypatch.setattr(settlement, "attest_mutation_journal_settlement", fail)
    result = _hook()(ArchitectCheckpoint(
        kind="final",
        corpus=Corpus(tmp_path / "corpus"),
        cwd=str(tmp_path),
        tier_id="__architect_final__",
        operation_id="architect-final",
        child_run_dir=None,
        tiers_executed=0,
    ))

    final = result[settlement.DEEPSWE_CHECKPOINT_EXTENSION_KEY][
        "mutation_settlement"
    ]["final_attestation"]
    assert final["settled"] is False
    assert final["schema_version"] == (
        settlement.TIER_MUTATION_SETTLEMENT_SCHEMA
    )
    assert final["error_kind"] == "RuntimeError"
    assert final["error_message"] == "attester unavailable"
