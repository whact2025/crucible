# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Strict writer-generation rollover and retired-member proof tests."""

from __future__ import annotations

import copy

import pytest

from .test_writer_epoch import (
    _Storage,
    _arm,
    _owner_boundary,
    _scheduler_receipt,
)
from .test_mutation_journal_writer_epoch_integration import (
    _begin_family,
    _unregister,
)


class _ProofStorage(_Storage):
    def __init__(self, transaction_id: str, parent_transaction_id: str | None):
        super().__init__(transaction_id, parent_transaction_id)
        self.proofs: tuple[object, ...] = ()

    def read_writer_epoch_terminal_proofs(self) -> tuple[object, ...]:
        return copy.deepcopy(self.proofs)


def _quiesce(lease):
    from crucible.workflows._writer_epoch import quiesce_writer_epoch_family

    quiesce_writer_epoch_family(lease, _scheduler_receipt(lease))


def _terminal_proof(root, retired, lease):
    from crucible.workflows._writer_epoch import (
        build_writer_epoch_terminal_proof,
    )

    proof = build_writer_epoch_terminal_proof(
        transaction_id=retired.transaction_id,
        parent_transaction_id=retired.parent_transaction_id,
        terminal_state="committed",
        epoch_record=retired.record,
        family=tuple(
            (member.transaction_id, member.parent_transaction_id)
            for member in lease.members
        ),
        terminal_manifest={
            "transaction_id": retired.transaction_id,
            "state": "committed",
            "writer_epoch": retired.record,
        },
    )
    root.proofs = (*root.proofs, proof.to_record())
    return proof


def _prior_generation():
    root = _ProofStorage("root", None)
    retired = _ProofStorage("retired", "root")
    lease = _arm((root, retired))
    _quiesce(lease)
    _terminal_proof(root, retired, lease)
    retired.active = False
    return root, retired, lease


def _rollover(root, *members):
    from crucible.workflows._writer_epoch import arm_writer_epoch_family

    owner, boundary = _owner_boundary()
    return arm_writer_epoch_family(
        (root, *members),
        owner=owner,
        boundary=boundary,
    )


def test_strict_rollover_accepts_exact_retirement_and_pristine_child() -> None:
    root, _retired, prior = _prior_generation()
    added = _ProofStorage("added", "root")

    lease = _rollover(root, added)

    assert prior.generation == 1
    assert lease.generation == 2
    assert lease.member_ids == ("root", "added")
    assert root.record == added.record == lease.armed_record()


def test_rollover_added_member_must_be_pristine_generation_zero() -> None:
    from crucible.workflows._writer_epoch import WriterEpochBlockedError

    root, _retired, _prior = _prior_generation()
    foreign = _ProofStorage("foreign", None)
    foreign_lease = _arm((foreign,))
    _quiesce(foreign_lease)
    added = _ProofStorage("added", "root")
    added.record = copy.deepcopy(foreign.record)

    with pytest.raises(WriterEpochBlockedError, match="pristine generation zero"):
        _rollover(root, added)


@pytest.mark.parametrize("field", ("parent_transaction_id", "family_digest"))
def test_rollover_rejects_tampered_removed_member_proof(field: str) -> None:
    from crucible.workflows._writer_epoch import WriterEpochBlockedError

    root, _retired, _prior = _prior_generation()
    proof = copy.deepcopy(root.proofs[0])
    assert isinstance(proof, dict)
    proof[field] = "wrong-parent" if field == "parent_transaction_id" else (
        "sha256:" + ("f" * 64)
    )
    root.proofs = (proof,)
    added = _ProofStorage("added", "root")

    with pytest.raises(WriterEpochBlockedError, match="proof"):
        _rollover(root, added)


def test_rollover_rejects_reparented_retained_member() -> None:
    from crucible.workflows._writer_epoch import WriterEpochBlockedError

    root = _ProofStorage("root", None)
    retained = _ProofStorage("retained", "root")
    retired = _ProofStorage("retired", "root")
    prior = _arm((root, retained, retired))
    _quiesce(prior)
    _terminal_proof(root, retired, prior)
    retired.active = False
    bridge = _ProofStorage("bridge", "root")
    retained.parent_transaction_id = "bridge"

    with pytest.raises(WriterEpochBlockedError, match="retained.*differs"):
        _rollover(root, bridge, retained)


def test_proof_for_retained_member_does_not_authorize_removed_member() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        build_writer_epoch_terminal_proof,
    )

    root = _ProofStorage("root", None)
    retained = _ProofStorage("retained", "root")
    retired = _ProofStorage("retired", "root")
    prior = _arm((root, retained, retired))
    _quiesce(prior)
    wrong = build_writer_epoch_terminal_proof(
        transaction_id=retained.transaction_id,
        parent_transaction_id=retained.parent_transaction_id,
        terminal_state="committed",
        epoch_record=retained.record,
        family=tuple(
            (member.transaction_id, member.parent_transaction_id)
            for member in prior.members
        ),
        terminal_manifest={"state": "committed", "member": "retained"},
    )
    root.proofs = (wrong.to_record(),)
    retired.active = False

    with pytest.raises(WriterEpochBlockedError, match="lacks.*proof"):
        _rollover(root, retained)


def test_stale_prior_generation_proof_cannot_authorize_next_retirement() -> None:
    from crucible.workflows._writer_epoch import WriterEpochBlockedError

    root, _retired, _prior = _prior_generation()
    generation_two_child = _ProofStorage("generation-two-child", "root")
    generation_two = _rollover(root, generation_two_child)
    _quiesce(generation_two)
    generation_two_child.active = False
    generation_three_child = _ProofStorage("generation-three-child", "root")

    with pytest.raises(WriterEpochBlockedError, match="lacks.*proof"):
        _rollover(root, generation_three_child)


def test_root_first_rollover_interruption_remains_blocking_and_retryable() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochArmError,
        WriterEpochBlockedError,
        require_writer_epoch_quiescence,
    )

    root, _retired, _prior = _prior_generation()
    added = _ProofStorage("added", "root")
    root.fail_after_state = "armed"

    with pytest.raises(WriterEpochArmError) as raised:
        _rollover(root, added)
    assert root.record["state"] == "armed"
    assert added.record["state"] == "never_armed"
    with pytest.raises(WriterEpochBlockedError):
        require_writer_epoch_quiescence((root, added))

    lease = raised.value.retry_arm()
    assert root.record == added.record == lease.armed_record()


def test_journal_cleanup_enforces_children_first_root_last(tmp_path) -> None:
    _repo, _corpus, root, child = _begin_family(
        tmp_path,
        run_id="root-last-cleanup",
    )
    root_dir = root.journal_dir
    child_dir = child.journal_dir
    try:
        for journal in (child, root):
            journal._manifest["state"] = "rolled_back"
            journal._write()

        assert root.retry_cleanup() is False
        assert root_dir.is_dir()
        assert child_dir.is_dir()

        assert child.retry_cleanup() is True
        assert not child_dir.exists()
        assert root_dir.is_dir()
        assert root._manifest["writer_epoch_terminal_proofs"].get(
            child.transaction_id
        )

        assert root.retry_cleanup() is True
        assert not root_dir.exists()
    finally:
        _unregister(root, child)


def test_journal_reuses_root_for_new_child_after_proved_retirement(
    tmp_path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, corpus, root, retired = _begin_family(
        tmp_path,
        run_id="journal-rollover",
    )
    replacement = None
    try:
        first = _arm((root, retired))
        _quiesce(first)
        retired._manifest["state"] = "rolled_back"
        retired._write()
        assert retired.retry_cleanup() is True

        replacement = journals.WorkspaceMutationJournal.begin(
            cwd=root._manifest["cwd"],
            journal_root=root.journal_dir / "nested_mutations",
            slice_id="journal-rollover-replacement",
            scope=("a.py",),
            sequence=3,
            spec=corpus.latest_spec(),
            parent_transaction_id=root.transaction_id,
        )
        family = journals.live_writer_epoch_family(root)
        assert tuple(member.transaction_id for member in family) == (
            root.transaction_id,
            replacement.transaction_id,
        )

        second = _arm(family)
        assert second.generation == first.generation + 1
        assert root.read_writer_epoch_record() == (
            replacement.read_writer_epoch_record()
        )
        _quiesce(second)
    finally:
        _unregister(root, retired, *(() if replacement is None else (replacement,)))


def test_generation_zero_root_rejects_unproved_family_shrink(tmp_path) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import WriterEpochBlockedError

    _repo, _corpus, root, child = _begin_family(
        tmp_path,
        run_id="generation-zero-unproved-shrink",
    )
    try:
        journals._unregister_live(child.transaction_id)

        with pytest.raises(
            WriterEpochBlockedError,
            match="generation-zero.*exact terminal proof",
        ):
            _arm((root,))
    finally:
        _unregister(root, child)


def test_generation_zero_root_accepts_exact_proved_family_shrink(
    tmp_path,
) -> None:
    _repo, _corpus, root, child = _begin_family(
        tmp_path,
        run_id="generation-zero-proved-shrink",
    )
    try:
        child._manifest["state"] = "rolled_back"
        child._write()
        assert child.retry_cleanup() is True

        lease = _arm((root,))

        assert lease.generation == 1
        assert lease.member_ids == (root.transaction_id,)
    finally:
        _unregister(root, child)


def test_generation_zero_projection_race_blocks_inside_root_cas(
    tmp_path,
    monkeypatch,
) -> None:
    from crucible.workflows._writer_epoch import WriterEpochArmError

    _repo, _corpus, root, _child = _begin_family(
        tmp_path,
        run_id="generation-zero-projection-race",
    )
    try:
        original = root.validate_writer_epoch_prearm_family

        def race_projection(expected_epoch, replacement_family):
            original(expected_epoch, replacement_family)
            root._manifest["writer_epoch_family"].append("racing-child")
            root._manifest["writer_epoch_family_parents"][
                "racing-child"
            ] = root.transaction_id
            root._write()

        monkeypatch.setattr(
            root,
            "validate_writer_epoch_prearm_family",
            race_projection,
        )

        with pytest.raises(
            WriterEpochArmError,
            match="generation-zero.*exact terminal proof",
        ):
            _arm((root, _child))
        assert root.read_writer_epoch_record()["state"] == "never_armed"
    finally:
        _unregister(root, _child)
