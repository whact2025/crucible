# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hostile integration tests for mutation-journal writer epochs.

These tests exercise the persistence adapter and the source/Git frontiers,
not merely the in-memory epoch state machine.  In particular, no PID/death
heuristic, mutable registry projection, or partially removed journal family
may turn an armed or incomplete epoch into settlement authority.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import shutil
import stat
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from .test_workflows_tier_mutation_transaction import (
    _setup,
    _simulate_process_death,
)


_BOOT_ID = "00000000-0000-0000-0000-000000000001"
_EVIDENCE_DIGEST = "sha256:" + ("a" * 64)


class _MemoryEpochStorage:
    """Small public-protocol storage used to mint valid competing records."""

    def __init__(self, transaction_id: str) -> None:
        from crucible.workflows._writer_epoch import initial_writer_epoch_record

        self.transaction_id = transaction_id
        self.parent_transaction_id = None
        self.active = True
        self.record: object = initial_writer_epoch_record()

    def read_writer_epoch_record(self) -> object:
        return json.loads(json.dumps(self.record))

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        if self.record != expected:
            raise RuntimeError("stale memory CAS")
        self.record = json.loads(json.dumps(replacement))
        return self.read_writer_epoch_record()


def _owner(identifier: str):
    from crucible.workflows._writer_epoch import WriterEpochOwner

    return WriterEpochOwner(
        kind="test",
        identifier=identifier,
        process_id=max(1, os.getpid()),
        process_start_time_ticks=1,
        boot_id=_BOOT_ID,
    )


def _boundary(identifier: str):
    from crucible.workflows._writer_epoch import WriterEpochBoundary

    return WriterEpochBoundary(
        kind="test",
        identifier=identifier,
        settlement_kind="scheduler_join",
        expected_owner_ids=(f"{identifier}:owner",),
        expected_worker_ids=(f"{identifier}:worker",),
    )


def _arm_family(*journals: Any, identifier: str = "hostile"):
    from crucible.workflows._writer_epoch import arm_writer_epoch_family

    return arm_writer_epoch_family(
        tuple(journals),
        owner=_owner(identifier),
        boundary=_boundary(identifier),
    )


def _quiesce_family(lease: Any) -> None:
    from crucible.workflows._writer_epoch import (
        quiesce_writer_epoch_family,
        scheduler_settlement_receipt,
    )

    identifier = lease.boundary.identifier
    receipt = scheduler_settlement_receipt(
        lease,
        settled_owner_ids=(f"{identifier}:owner",),
        joined_worker_ids=(f"{identifier}:worker",),
        evidence_digest=_EVIDENCE_DIGEST,
        outcome="succeeded",
    )
    quiesce_writer_epoch_family(lease, receipt)


def _valid_armed_record(
    identifier: str,
    *,
    transaction_id: str | None = None,
) -> dict[str, object]:
    storage = _MemoryEpochStorage(transaction_id or f"memory-{identifier}")
    _arm_family(storage, identifier=identifier)
    assert isinstance(storage.record, dict)
    return json.loads(json.dumps(storage.record))


def _cas_process(
    label: str,
    journal: Any,
    replacement: dict[str, object],
    peer_at_replace: Any,
    first_reread: Any,
    results: Any,
) -> None:
    """Force both contenders past the non-atomic compare before publication."""

    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    real_replace = journals.os.replace
    real_load = journals._load_mutation_journal_manifest_at

    def ordered_replace(source, destination, *args, **kwargs):
        if str(destination) == "manifest.json" and str(source).startswith(
            ".tmp_manifest.json_"
        ):
            if label == "second":
                peer_at_replace.set()
                # With a correct interprocess CAS lock, this process may own
                # the lock first. The timeout avoids deadlocking that valid
                # serialization while still exposing the unlocked race.
                first_reread.wait(2.0)
            else:
                peer_at_replace.wait(2.0)
        return real_replace(source, destination, *args, **kwargs)

    def load_then_release(*args, **kwargs):
        observed = real_load(*args, **kwargs)
        if label == "first":
            first_reread.set()
        return observed

    journals.os.replace = ordered_replace  # type: ignore[assignment]
    journals._load_mutation_journal_manifest_at = load_then_release
    try:
        journal.compare_and_write_writer_epoch_record(
            initial_writer_epoch_record(),
            replacement,
        )
    except BaseException as exc:  # child reports the exact CAS disposition
        results.put((label, "rejected", type(exc).__name__, str(exc)))
    else:
        results.put((label, "ok", "", ""))


def _begin_single(tmp_path: Path, *, run_id: str = "writer-epoch"):
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / run_id / "mutations",
        slice_id=run_id,
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    return repo, corpus, journal


def _begin_family(tmp_path: Path, *, run_id: str = "writer-family"):
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / run_id / "mutations",
        slice_id=f"{run_id}-parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=parent.journal_dir / "nested_mutations",
        slice_id=f"{run_id}-child",
        scope=("a.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    return repo, corpus, parent, child


def _unregister(*journals_to_drop: Any) -> None:
    from crucible.workflows import _mutation_journal as journals

    for journal in reversed(journals_to_drop):
        journals._unregister_live(journal.transaction_id)


def _publish_quiesced_terminal_receipt(
    tmp_path: Path,
    *,
    run_id: str,
):
    """Publish a terminal receipt for one settled generation without deleting it."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus, journal = _begin_single(tmp_path, run_id=run_id)
    lease = _arm_family(journal, identifier=run_id)
    _quiesce_family(lease)
    journal._manifest["state"] = "rolled_back"
    journal._write()
    journal._persist_writer_epoch_terminal_proof()
    parent_fd = os.open(
        journal.journal_dir.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    journal_fd = os.open(
        journal.journal_dir,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        receipt, _preexisting = journals._ensure_terminal_cleanup_receipt(
            journal,
            parent_fd,
            journal.journal_dir.parent,
            journal.journal_dir.name,
            journal_fd,
            os.fstat(journal_fd),
        )
    finally:
        os.close(journal_fd)
        os.close(parent_fd)
    return repo, corpus, journal, lease, receipt


def _publish_quiesced_external_tombstone(
    tmp_path: Path,
    *,
    run_id: str,
) -> tuple[Any, Any, Any, Path, Path, Path, dict[str, object]]:
    """Publish one exact v2 audit/tombstone pair for hostile reaper tests."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._journal_recovery import ExternalRecoveryRequest

    source, corpus, journal = _begin_single(tmp_path, run_id=run_id)
    lease = _arm_family(journal, identifier=run_id)
    _quiesce_family(lease)
    journal._manifest["state"] = "external_recovery_verified"
    journal._write()
    proof = journals._writer_epoch_manifest_proof(
        journal._manifest,
        terminal_state="rolled_back",
    )
    run_dir = journal.journal_dir.parent.parent
    tombstone_root = run_dir / "journal_recovery_tombstones"
    audit_root = run_dir / "journal_recovery"
    tombstone_root.mkdir()
    audit_root.mkdir()
    original = journal.journal_dir
    tombstone = tombstone_root / journal.transaction_id
    metadata = original.lstat()
    request = ExternalRecoveryRequest(
        run_id=run_dir.name,
        transaction_id=journal.transaction_id,
        expected_diff_ref="sha256:" + ("a" * 64),
        removed_linked_worktrees=(str(tmp_path / "removed-peer"),),
        appended_stash_tip="b" * 40,
        reason=f"exact quiesced writer epoch tombstone {run_id}",
    ).normalized()
    event: dict[str, object] = {
        "schema": external_recovery._ACCEPTANCE_SCHEMA,
        "stage": "tombstone_prepared",
        "request": request.to_json(),
        "source_root": str(source.resolve()),
        "parent_slice_id": journal.slice_id,
        "tombstone": str(tombstone),
        "journal_dir": str(original),
        "writer_epoch_manifest": copy.deepcopy(journal._manifest),
        "writer_epoch_terminal_proof": proof.to_record(),
        "tombstone_identity": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
        },
    }
    journal_fd = os.open(
        journal.journal_dir,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        event["tombstone_inventory"] = [
            list(record)
            for record in external_recovery._external_deletion_inventory(
                journal_fd
            )
        ]
        external_recovery._publish_prepared_control_at(journal_fd, event)
        assert external_recovery._external_deletion_inventory(
            journal_fd,
            event=event,
        ) == external_recovery._external_tombstone_inventory(event)
    finally:
        os.close(journal_fd)
    original.rename(tombstone)
    event["stage"] = "complete"
    audit = audit_root / f"{journal.transaction_id}.json"
    audit.write_bytes(external_recovery._encode_recovery_event(event))
    journals._unregister_live(journal.transaction_id)
    return source, corpus, journal, original, tombstone, audit, event


def _mutate_recovery_audit(audit: Path, *, kind: str) -> tuple[int, int]:
    """Replace audit authority by name or mutate the held inode in place."""

    before = audit.stat().st_ino
    payload = json.dumps(
        {"attacker": kind},
        sort_keys=True,
        separators=(",", ":"),
    )
    if kind == "replacement":
        replacement = audit.with_name(f".{audit.name}.attacker")
        replacement.write_text(payload, encoding="utf-8")
        os.replace(replacement, audit)
    elif kind == "same_inode":
        audit.write_text(payload, encoding="utf-8")
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported audit mutation: {kind}")
    return before, audit.stat().st_ino


def _clone_external_tombstone_authority(
    *,
    corpus: Any,
    journal: Any,
    first_tombstone: Path,
    first_event: dict[str, object],
    run_id: str,
) -> Path:
    """Clone one exact certificate under a distinct run-level authority."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows._journal_recovery import ExternalRecoveryRequest

    second_run = Path(corpus.spec_dir) / "runs" / run_id
    second_durable_root = second_run / "mutations"
    second_tombstone_root = second_run / "journal_recovery_tombstones"
    second_audit_root = second_run / "journal_recovery"
    second_durable_root.mkdir(parents=True)
    second_tombstone_root.mkdir()
    second_audit_root.mkdir()
    second_tombstone = second_tombstone_root / journal.transaction_id
    shutil.copytree(first_tombstone, second_tombstone, symlinks=True)
    second_metadata = second_tombstone.lstat()
    first_request = ExternalRecoveryRequest.from_json(
        cast(dict[str, object], first_event["request"])
    )
    second_request = ExternalRecoveryRequest(
        run_id=second_run.name,
        transaction_id=first_request.transaction_id,
        expected_diff_ref=first_request.expected_diff_ref,
        removed_linked_worktrees=first_request.removed_linked_worktrees,
        appended_stash_tip=first_request.appended_stash_tip,
        reason=first_request.reason,
        ancestor_submodule=first_request.ancestor_submodule,
    ).normalized()
    second_event = copy.deepcopy(first_event)
    second_event.update({
        "stage": "tombstone_prepared",
        "request": second_request.to_json(),
        "tombstone": str(second_tombstone),
        "journal_dir": str(second_durable_root / journal.transaction_id),
        "tombstone_identity": {
            "device": second_metadata.st_dev,
            "inode": second_metadata.st_ino,
            "mode": stat.S_IMODE(second_metadata.st_mode),
        },
    })
    second_event.pop("tombstone_control_identity", None)
    second_fd = os.open(
        second_tombstone,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        second_event["tombstone_inventory"] = [
            list(record)
            for record in external_recovery._external_deletion_inventory(
                second_fd
            )
        ]
        external_recovery._publish_prepared_control_at(
            second_fd,
            second_event,
        )
        assert external_recovery._external_deletion_inventory(
            second_fd,
            event=second_event,
        ) == external_recovery._external_tombstone_inventory(second_event)
    finally:
        os.close(second_fd)
    second_event["stage"] = "complete"
    (
        second_audit_root / f"{journal.transaction_id}.json"
    ).write_bytes(external_recovery._encode_recovery_event(second_event))
    return second_tombstone


def test_manifest_epoch_cas_is_atomic_across_processes(tmp_path: Path) -> None:
    """Two CAS operations from one expected value cannot both succeed."""

    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    _repo, _corpus, journal = _begin_single(tmp_path, run_id="cas-race")
    ctx = multiprocessing.get_context("fork")
    peer_at_replace = ctx.Event()
    first_reread = ctx.Event()
    results = ctx.Queue()
    contenders = (
        ctx.Process(
            target=_cas_process,
            args=(
                "first",
                journal,
                _valid_armed_record(
                    "cas-first",
                    transaction_id=journal.transaction_id,
                ),
                peer_at_replace,
                first_reread,
                results,
            ),
        ),
        ctx.Process(
            target=_cas_process,
            args=(
                "second",
                journal,
                _valid_armed_record(
                    "cas-second",
                    transaction_id=journal.transaction_id,
                ),
                peer_at_replace,
                first_reread,
                results,
            ),
        ),
    )
    try:
        for contender in contenders:
            contender.start()
        for contender in contenders:
            contender.join(timeout=15)
            assert not contender.is_alive(), "CAS contender deadlocked"
            assert contender.exitcode == 0
        observed = []
        for _index in range(2):
            try:
                observed.append(results.get(timeout=2))
            except Empty:
                pytest.fail("CAS contender did not report its disposition")
        assert sorted(item[1] for item in observed) == ["ok", "rejected"]
        assert journal.read_writer_epoch_record() != initial_writer_epoch_record()
    finally:
        for contender in contenders:
            if contender.is_alive():
                contender.kill()
                contender.join()
        _unregister(journal)


@pytest.mark.parametrize("foreign_field", ("state", "transaction_id"))
def test_manifest_epoch_cas_rejects_terminal_or_wrong_transaction(
    tmp_path: Path,
    foreign_field: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    _repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id=f"cas-{foreign_field}",
    )
    original_id = journal.transaction_id
    manifest_path = journal.journal_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[foreign_field] = (
        "committed" if foreign_field == "state" else "foreign-transaction"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="terminal|transaction|active|identity",
        ):
            journal.compare_and_write_writer_epoch_record(
                initial_writer_epoch_record(),
                initial_writer_epoch_record(),
            )
    finally:
        journals._unregister_live(original_id)


def test_never_armed_registry_projection_cannot_shrink_live_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, parent, child = _begin_family(
        tmp_path,
        run_id="registry-shrink",
    )
    parent._set_registry_transaction_family(
        (parent.transaction_id,),
        parents={parent.transaction_id: None},
    )
    touched: list[str] = []

    def forbidden_source_read():
        touched.append("source")
        raise AssertionError("source opened before exact epoch-family proof")

    monkeypatch.setattr(parent, "_bound_workspace_cwd", forbidden_source_read)
    try:
        with pytest.raises(journals.MutationJournalError, match="writer epoch"):
            parent.prepare_git_capture()
        assert touched == []
    finally:
        _unregister(parent, child)


def test_crash_unregister_preserves_family_without_terminal_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-owner loss cannot manufacture a terminal family shrink."""

    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, parent, child = _begin_family(
        tmp_path,
        run_id="crash-unregister-family",
    )
    expected_family = sorted((parent.transaction_id, child.transaction_id))
    expected_parents = {
        parent.transaction_id: None,
        child.transaction_id: parent.transaction_id,
    }
    assert parent._manifest["registry_transaction_family"] == expected_family
    assert parent._manifest["registry_family_parents"] == expected_parents
    assert parent._manifest["writer_epoch_terminal_proofs"] == {}
    source_touches: list[str] = []

    def forbidden_source_read():
        source_touches.append("source")
        raise AssertionError("source opened after unauthenticated family shrink")

    monkeypatch.setattr(parent, "_bound_workspace_cwd", forbidden_source_read)
    try:
        journals._unregister_live(child.transaction_id)
        durable_parent = json.loads(
            (parent.journal_dir / "manifest.json").read_text(encoding="utf-8")
        )
        assert durable_parent["registry_transaction_family"] == expected_family
        assert durable_parent["registry_family_parents"] == expected_parents
        assert durable_parent["writer_epoch_family"] == expected_family
        assert durable_parent["writer_epoch_terminal_proofs"] == {}
        with pytest.raises(journals.MutationJournalError, match="writer epoch"):
            parent.prepare_git_capture()
        assert source_touches == []
    finally:
        journals._unregister_live(parent.transaction_id)


def test_terminal_receipt_binds_typed_writer_epoch_proof(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _writer_epoch as epochs

    _repo, _corpus, journal, lease, receipt = _publish_quiesced_terminal_receipt(
        tmp_path,
        run_id="receipt-epoch-binding",
    )
    try:
        record = json.loads(receipt.receipt_payload)
        proof_record = record["writer_epoch_terminal_proof"]
        proof = epochs.parse_writer_epoch_terminal_proof(proof_record)
        expected = epochs.build_writer_epoch_terminal_proof(
            transaction_id=journal.transaction_id,
            parent_transaction_id=None,
            terminal_state="rolled_back",
            epoch_record=journal.read_writer_epoch_record(),
            family=((journal.transaction_id, None),),
            terminal_manifest=receipt.manifest,
        )
        parsed = epochs._parse_epoch_record(proof.epoch_record)

        assert proof == expected
        assert proof.transaction_id == receipt.transaction_id
        assert proof.terminal_manifest_digest == expected.terminal_manifest_digest
        assert parsed.state == "quiesced"
        assert parsed.generation == lease.generation
        assert parsed.epoch_id == lease.epoch_id
        assert parsed.family_digest == lease.family_digest
        assert parsed.settlement is not None
        assert parsed.settlement.receipt_digest
    finally:
        _unregister(journal)


@pytest.mark.parametrize(
    "tamper",
    ("armed", "family", "settlement", "terminal_manifest"),
)
def test_terminal_receipt_rejects_semantically_rebound_epoch_proof(
    tmp_path: Path,
    tamper: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows import _writer_epoch as epochs

    _repo, _corpus, journal, _lease, receipt = _publish_quiesced_terminal_receipt(
        tmp_path,
        run_id=f"receipt-rebind-{tamper}",
    )
    record = json.loads(receipt.receipt_payload)
    proof = cast(
        dict[str, Any],
        copy.deepcopy(record["writer_epoch_terminal_proof"]),
    )
    epoch_record = cast(dict[str, Any], proof["epoch_record"])

    def refresh_settlement_digest() -> None:
        settlement = cast(dict[str, Any], epoch_record["settlement"])
        settlement["receipt_digest"] = epochs._digest(
            {key: value for key, value in settlement.items() if key != "receipt_digest"}
        )

    if tamper == "armed":
        epoch_record["state"] = "armed"
        epoch_record["settlement"] = None
    elif tamper == "family":
        forged_family = [
            {
                "transaction_id": journal.transaction_id,
                "parent_transaction_id": None,
            },
            {
                "transaction_id": "forged-receipt-child",
                "parent_transaction_id": journal.transaction_id,
            },
        ]
        forged_family_digest = epochs._digest({"family": forged_family})
        proof["family"] = forged_family
        proof["family_digest"] = forged_family_digest
        epoch_record["family"] = forged_family
        epoch_record["family_digest"] = forged_family_digest
        cast(dict[str, Any], epoch_record["settlement"])["family_digest"] = (
            forged_family_digest
        )
        refresh_settlement_digest()
    elif tamper == "settlement":
        cast(dict[str, Any], epoch_record["settlement"])["evidence_digest"] = (
            "sha256:" + ("c" * 64)
        )
        refresh_settlement_digest()
    else:
        proof["terminal_manifest_digest"] = "sha256:" + ("d" * 64)

    proof["proof_digest"] = epochs._digest(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    )
    record["writer_epoch_terminal_proof"] = proof
    tampered_payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    parent_fd = os.open(
        journal.journal_dir.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="writer epoch|terminal proof|receipt",
        ):
            journals._parse_terminal_cleanup_receipt(
                parent_fd,
                journal.journal_dir.parent,
                receipt.receipt_name,
                tampered_payload,
            )
    finally:
        os.close(parent_fd)
        _unregister(journal)


@pytest.mark.parametrize("target_state", ("tombstone", "absent"))
def test_receipt_epoch_authority_survives_journal_target_removal(
    tmp_path: Path,
    target_state: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus, journal, _lease, receipt = _publish_quiesced_terminal_receipt(
        tmp_path,
        run_id=f"receipt-target-{target_state}",
    )
    receipt_path = journal.journal_dir.parent / receipt.receipt_name
    tombstone = journal.journal_dir.parent / receipt.tombstone_name
    journal.journal_dir.rename(tombstone)
    if target_state == "absent":
        shutil.rmtree(tombstone)
    journals._unregister_live(journal.transaction_id)

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.failures == ()
    assert journal.slice_id in report.recovered
    assert not receipt_path.exists()
    assert not tombstone.exists()
    assert not journals._registry_marker_path(
        repo,
        journal.transaction_id,
    ).exists()


def test_recovery_retains_exact_epoch_family_map_through_receipt_and_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    repo, corpus, journal, _lease, _receipt = _publish_quiesced_terminal_receipt(
        tmp_path,
        run_id="receipt-tls-lifetime",
    )
    journals._unregister_live(journal.transaction_id)
    sentinel = {"outer-sentinel": object()}
    missing = object()
    prior = getattr(
        journals._LOCK_STATE,
        "writer_epoch_recovery_members",
        missing,
    )
    journals._LOCK_STATE.writer_epoch_recovery_members = sentinel
    observed: list[tuple[str, frozenset[str]]] = []

    def observe_receipt(_corpus, _receipt, *, cleanup=True):
        members = getattr(
            journals._LOCK_STATE,
            "writer_epoch_recovery_members",
            {},
        )
        observed.append((f"receipt:{cleanup}", frozenset(members)))
        return journal.slice_id

    def observe_tombstone(_corpus, _root):
        members = getattr(
            journals._LOCK_STATE,
            "writer_epoch_recovery_members",
            {},
        )
        observed.append(("tombstone", frozenset(members)))
        return ()

    monkeypatch.setattr(
        journals,
        "_resume_terminal_cleanup_receipt",
        observe_receipt,
    )
    monkeypatch.setattr(
        external_recovery,
        "reap_external_recovery_tombstones",
        observe_tombstone,
    )
    try:
        journals.recover_mutation_journals(corpus, repo)
        assert observed == [
            ("receipt:False", frozenset({journal.transaction_id})),
            ("receipt:True", frozenset({journal.transaction_id})),
            ("tombstone", frozenset({journal.transaction_id})),
        ]
        assert journals._LOCK_STATE.writer_epoch_recovery_members is sentinel
    finally:
        if prior is missing:
            del journals._LOCK_STATE.writer_epoch_recovery_members
        else:
            journals._LOCK_STATE.writer_epoch_recovery_members = prior


def test_recovery_epoch_certificate_rejects_duplicate_live_owner(
    tmp_path: Path,
) -> None:
    """One transaction ID cannot name both live and certified authority."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    _repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="recovery-live-certificate-collision",
    )
    certified = external_recovery._CertifiedExternalEpochMember(
        transaction_id=journal.transaction_id,
        parent_transaction_id=None,
        epoch_record=initial_writer_epoch_record(),
    )

    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="duplicate|live.*authority|authority.*live",
        ):
            with external_recovery._installed_writer_epoch_recovery_members(
                (certified,)
            ):
                pass
    finally:
        _unregister(journal)


def test_recovery_epoch_uses_post_fork_process_state(
    tmp_path: Path,
) -> None:
    """Recovery cannot retain pre-fork TLS, locks, or live-owner maps."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    # Import recovery before simulating the startup-child boundary.  The fork
    # hook deliberately replaces all three process-local authority objects.
    journals._after_fork_child()
    _repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="post-fork-live-certificate-collision",
    )
    certified = external_recovery._CertifiedExternalEpochMember(
        transaction_id=journal.transaction_id,
        parent_transaction_id=None,
        epoch_record=initial_writer_epoch_record(),
    )

    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="duplicate|live.*authority|authority.*live",
        ):
            with external_recovery._installed_writer_epoch_recovery_members(
                (certified,)
            ):
                pass
    finally:
        _unregister(journal)


def test_recovery_epoch_install_reenters_exact_reconstructed_owner_and_restores(
    tmp_path: Path,
) -> None:
    """Nested installs retain one exact durable owner and restore outer TLS."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    _repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="reentrant-reconstructed-owner",
    )
    _unregister(journal)
    first = journals.discover_mutation_journals(
        corpus,
        run_id="reentrant-reconstructed-owner",
    )[0]
    second = journals.discover_mutation_journals(
        corpus,
        run_id="reentrant-reconstructed-owner",
    )[0]
    sentinel = object()
    previous = {"preexisting-sentinel": sentinel}
    missing = object()
    prior = getattr(
        journals._LOCK_STATE,
        "writer_epoch_recovery_members",
        missing,
    )
    journals._LOCK_STATE.writer_epoch_recovery_members = previous
    try:
        with external_recovery._installed_writer_epoch_recovery_members(
            (first,)
        ):
            outer = journals._LOCK_STATE.writer_epoch_recovery_members
            assert outer[first.transaction_id] is first
            with external_recovery._installed_writer_epoch_recovery_members(
                (first,)
            ):
                assert (
                    journals._LOCK_STATE.writer_epoch_recovery_members[
                        first.transaction_id
                    ]
                    is first
                )
            with external_recovery._installed_writer_epoch_recovery_members(
                (second,)
            ):
                assert (
                    journals._LOCK_STATE.writer_epoch_recovery_members[
                        first.transaction_id
                    ]
                    is first
                )
                with external_recovery._shadow_journal(second) as shadow:
                    assert (
                        journals._LOCK_STATE.writer_epoch_recovery_members[
                            first.transaction_id
                        ]
                        is shadow
                    )
                assert (
                    journals._LOCK_STATE.writer_epoch_recovery_members[
                        first.transaction_id
                    ]
                    is first
                )
            assert journals._LOCK_STATE.writer_epoch_recovery_members is outer
        assert journals._LOCK_STATE.writer_epoch_recovery_members is previous
        assert previous["preexisting-sentinel"] is sentinel
    finally:
        if prior is missing:
            del journals._LOCK_STATE.writer_epoch_recovery_members
        else:
            journals._LOCK_STATE.writer_epoch_recovery_members = prior


@pytest.mark.parametrize(
    "changed_binding",
    ("parent", "epoch", "manifest", "logical_inode"),
)
def test_recovery_epoch_reentrant_install_rejects_changed_binding(
    tmp_path: Path,
    changed_binding: str,
) -> None:
    """Equivalent transaction IDs cannot hide any changed owner binding."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    if changed_binding in {"parent", "epoch"}:
        record = initial_writer_epoch_record()
        first: object = external_recovery._CertifiedExternalEpochMember(
            transaction_id="a" * 32,
            parent_transaction_id=None,
            epoch_record=record,
        )
        second_record = copy.deepcopy(record)
        parent = None
        if changed_binding == "parent":
            parent = "b" * 32
        else:
            second_record["generation"] = 1
        second: object = external_recovery._CertifiedExternalEpochMember(
            transaction_id="a" * 32,
            parent_transaction_id=parent,
            epoch_record=second_record,
        )
    else:
        _repo, corpus, journal = _begin_single(
            tmp_path,
            run_id=f"changed-reentrant-{changed_binding}",
        )
        _unregister(journal)
        first, second = (
            journals.discover_mutation_journals(
                corpus,
                run_id=f"changed-reentrant-{changed_binding}",
            )[0]
            for _attempt in range(2)
        )
        if changed_binding == "manifest":
            second._manifest["slice_id"] = "forged-slice"
        else:
            assert second._logical_journal_identity is not None
            device, inode = second._logical_journal_identity
            second._logical_journal_identity = (device, inode + 1)

    with external_recovery._installed_writer_epoch_recovery_members((first,)):
        outer = journals._LOCK_STATE.writer_epoch_recovery_members
        with pytest.raises(
            journals.MutationJournalError,
            match="conflicts|duplicate",
        ):
            with external_recovery._installed_writer_epoch_recovery_members(
                (second,)
            ):
                pass
        assert journals._LOCK_STATE.writer_epoch_recovery_members is outer


def test_shadow_journal_refuses_changed_reconstructed_owner_binding(
    tmp_path: Path,
) -> None:
    """Shadow substitution cannot cross a changed durable inode binding."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    _repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="shadow-changed-reconstructed-owner",
    )
    _unregister(journal)
    first, changed = (
        journals.discover_mutation_journals(
            corpus,
            run_id="shadow-changed-reconstructed-owner",
        )[0]
        for _attempt in range(2)
    )
    assert changed._logical_journal_identity is not None
    device, inode = changed._logical_journal_identity
    changed._logical_journal_identity = (device, inode + 1)
    installed = {first.transaction_id: changed}
    missing = object()
    prior = getattr(
        journals._LOCK_STATE,
        "writer_epoch_recovery_members",
        missing,
    )
    journals._LOCK_STATE.writer_epoch_recovery_members = installed
    try:
        with external_recovery._shadow_journal(first) as shadow:
            assert shadow is not first
            assert (
                journals._LOCK_STATE.writer_epoch_recovery_members[
                    first.transaction_id
                ]
                is changed
            )
        assert journals._LOCK_STATE.writer_epoch_recovery_members is installed
    finally:
        if prior is missing:
            del journals._LOCK_STATE.writer_epoch_recovery_members
        else:
            journals._LOCK_STATE.writer_epoch_recovery_members = prior


def test_recovery_epoch_reentrant_conflict_does_not_publish_partial_additions(
) -> None:
    """One conflicting overlap rejects every staged inner-family addition."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    record = initial_writer_epoch_record()
    outer_member = external_recovery._CertifiedExternalEpochMember(
        transaction_id="a" * 32,
        parent_transaction_id=None,
        epoch_record=record,
    )
    conflict = external_recovery._CertifiedExternalEpochMember(
        transaction_id=outer_member.transaction_id,
        parent_transaction_id="b" * 32,
        epoch_record=record,
    )
    staged = external_recovery._CertifiedExternalEpochMember(
        transaction_id="c" * 32,
        parent_transaction_id=None,
        epoch_record=record,
    )
    with external_recovery._installed_writer_epoch_recovery_members(
        (outer_member,)
    ):
        outer = journals._LOCK_STATE.writer_epoch_recovery_members
        with pytest.raises(journals.MutationJournalError, match="conflicts"):
            with external_recovery._installed_writer_epoch_recovery_members(
                (staged, conflict)
            ):
                pass
        assert journals._LOCK_STATE.writer_epoch_recovery_members is outer
        assert staged.transaction_id not in outer


def test_recovery_epoch_install_rejects_malformed_preexisting_tls() -> None:
    """A non-mapping outer inventory is unknown authority, never emptiness."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import initial_writer_epoch_record

    member = external_recovery._CertifiedExternalEpochMember(
        transaction_id="a" * 32,
        parent_transaction_id=None,
        epoch_record=initial_writer_epoch_record(),
    )
    malformed = object()
    missing = object()
    prior = getattr(
        journals._LOCK_STATE,
        "writer_epoch_recovery_members",
        missing,
    )
    journals._LOCK_STATE.writer_epoch_recovery_members = malformed
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="thread-local inventory is malformed",
        ):
            with external_recovery._installed_writer_epoch_recovery_members(
                (member,)
            ):
                pass
        assert journals._LOCK_STATE.writer_epoch_recovery_members is malformed
    finally:
        if prior is missing:
            del journals._LOCK_STATE.writer_epoch_recovery_members
        else:
            journals._LOCK_STATE.writer_epoch_recovery_members = prior


@pytest.mark.parametrize(
    "inventory_change",
    ("missing", "extra", "substituted"),
)
def test_startup_resume_rejects_changed_exact_inventory_before_source_mutation(
    tmp_path: Path,
    inventory_change: str,
) -> None:
    """Startup resume cannot lose, add, or substitute a discovered owner."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from .test_workflows_journal_recover import (
        _setup as setup_external_recovery,
        _tree_image,
    )

    state = setup_external_recovery(tmp_path)
    source = cast(Path, state["source"])
    corpus = state["corpus"]
    request = state["request"]
    startup_inventory = tuple(journals.discover_mutation_journals(
        corpus,
        run_id=request.run_id,
    ))
    root = next(
        member
        for member in startup_inventory
        if member.transaction_id == request.transaction_id
    )
    installed: dict[str, object] = {
        member.transaction_id: member for member in startup_inventory
    }
    supplied_inventory = startup_inventory
    if inventory_change == "missing":
        omitted = next(
            (
                member
                for member in startup_inventory
                if member.transaction_id != root.transaction_id
            ),
            root,
        )
        supplied_inventory = tuple(
            member for member in startup_inventory if member is not omitted
        )
    elif inventory_change == "extra":
        installed["f" * 32] = object()
    else:
        replacement_inventory = tuple(journals.discover_mutation_journals(
            corpus,
            run_id=request.run_id,
        ))
        replaced = next(
            (
                member
                for member in startup_inventory
                if member.transaction_id != root.transaction_id
            ),
            root,
        )
        replacement = next(
            member
            for member in replacement_inventory
            if member.transaction_id == replaced.transaction_id
        )
        installed[replaced.transaction_id] = replacement

    before = {
        path: record
        for path, record in _tree_image(source).items()
        if path != ".git" and not path.startswith(".git/")
    }
    missing_tls = object()
    prior = getattr(
        journals._LOCK_STATE,
        "writer_epoch_recovery_members",
        missing_tls,
    )
    journals._LOCK_STATE.writer_epoch_recovery_members = installed
    try:
        with journals.workspace_mutation_lock(source):
            with pytest.raises(
                journals.MutationJournalError,
                match="startup recovery (inventory|root)|exact installed",
            ):
                external_recovery._startup_resume_journal(
                    corpus,
                    source.resolve(),
                    request,
                    root,
                    supplied_inventory,
                )
    finally:
        if prior is missing_tls:
            del journals._LOCK_STATE.writer_epoch_recovery_members
        else:
            journals._LOCK_STATE.writer_epoch_recovery_members = prior
    after = {
        path: record
        for path, record in _tree_image(source).items()
        if path != ".git" and not path.startswith(".git/")
    }
    assert after == before


def test_terminal_proof_inventory_key_must_match_embedded_member(
    tmp_path: Path,
) -> None:
    """Proof-map keys are authority and cannot alias another member."""

    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="terminal-proof-key-binding",
    )
    lease = _arm_family(journal, identifier="terminal-proof-key-binding")
    _quiesce_family(lease)
    journal._manifest["state"] = "rolled_back"
    proof = journals._writer_epoch_manifest_proof(
        journal._manifest,
        terminal_state="rolled_back",
    )
    journal._manifest["writer_epoch_terminal_proofs"] = {
        "different-transaction": proof.to_record(),
    }

    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="terminal proof.*(identity|key|inventory)|malformed",
        ):
            journals._writer_epoch_terminal_proof_records(journal._manifest)
    finally:
        _unregister(journal)


def test_external_recovery_tombstone_reaper_blocks_armed_epoch_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows import _writer_epoch as epochs
    from crucible.workflows._journal_recovery import ExternalRecoveryRequest

    source, corpus, journal = _begin_single(
        tmp_path,
        run_id="armed-external-tombstone",
    )
    _arm_family(journal, identifier="armed-external-tombstone")
    journal._manifest["state"] = "external_recovery_verified"
    journal._write()
    epoch_record = cast(dict[str, Any], journal.read_writer_epoch_record())
    proof = {
        "schema": epochs.WRITER_TERMINAL_PROOF_SCHEMA,
        "transaction_id": journal.transaction_id,
        "parent_transaction_id": None,
        "terminal_state": "rolled_back",
        "epoch_record": copy.deepcopy(epoch_record),
        "family": copy.deepcopy(epoch_record["family"]),
        "family_digest": epoch_record["family_digest"],
        "terminal_manifest_digest": epochs._digest(
            {"terminal_manifest": journal._manifest}
        ),
    }
    proof["proof_digest"] = epochs._digest(proof)
    run_dir = journal.journal_dir.parent.parent
    tombstone_root = run_dir / "journal_recovery_tombstones"
    audit_root = run_dir / "journal_recovery"
    tombstone_root.mkdir()
    audit_root.mkdir()
    original = journal.journal_dir
    tombstone = tombstone_root / journal.transaction_id
    original.rename(tombstone)
    metadata = tombstone.lstat()
    request = ExternalRecoveryRequest(
        run_id=run_dir.name,
        transaction_id=journal.transaction_id,
        expected_diff_ref="sha256:" + ("a" * 64),
        removed_linked_worktrees=(str(tmp_path / "removed-peer"),),
        appended_stash_tip="b" * 40,
        reason="hostile writer epoch tombstone",
    ).normalized()
    (audit_root / f"{journal.transaction_id}.json").write_text(
        json.dumps(
            {
                "schema": external_recovery._ACCEPTANCE_SCHEMA,
                "stage": "complete",
                "request": request.to_json(),
                "source_root": str(source.resolve()),
                "parent_slice_id": journal.slice_id,
                "tombstone": str(tombstone),
                "journal_dir": str(original),
                "writer_epoch_manifest": copy.deepcopy(journal._manifest),
                "writer_epoch_terminal_proof": proof,
                "tombstone_identity": {
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mode": stat.S_IMODE(metadata.st_mode),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    journals._unregister_live(journal.transaction_id)
    delete_touches: list[str] = []

    def forbidden_tombstone_delete(*_args, **_kwargs):
        delete_touches.append("delete")
        raise AssertionError("external tombstone deleted before epoch audit")

    monkeypatch.setattr(
        external_recovery,
        "_remove_bound_directory_tree",
        forbidden_tombstone_delete,
    )
    with journals.workspace_mutation_lock(source):
        with pytest.raises(journals.MutationJournalError, match="writer epoch"):
            external_recovery.reap_external_recovery_tombstones(
                corpus,
                source,
            )
    assert delete_touches == []
    assert tombstone.is_dir()


def test_external_recovery_tombstone_reaper_accepts_exact_quiesced_proof_after_manifest_delete(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    source, corpus, journal, _original, tombstone, _audit, _event = (
        _publish_quiesced_external_tombstone(
        tmp_path,
        run_id="quiesced-external-tombstone",
        )
    )
    (tombstone / "manifest.json").unlink()

    with journals.workspace_mutation_lock(source):
        recovered = external_recovery.reap_external_recovery_tombstones(
            corpus,
            source,
        )

    assert recovered == (journal.slice_id,)
    assert not tombstone.exists()


def test_external_recovery_rejects_duplicate_epoch_inventory_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two audits cannot independently certify the same transaction family."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    (
        source,
        corpus,
        journal,
        _original,
        first_tombstone,
        _first_audit,
        first_event,
    ) = _publish_quiesced_external_tombstone(
        tmp_path,
        run_id="duplicate-external-a",
    )
    second_tombstone = _clone_external_tombstone_authority(
        corpus=corpus,
        journal=journal,
        first_tombstone=first_tombstone,
        first_event=first_event,
        run_id="duplicate-external-b",
    )
    delete_touches: list[str] = []

    def forbidden_delete(*_args: object, **_kwargs: object) -> None:
        delete_touches.append("delete")
        raise AssertionError("duplicate epoch inventory reached deletion")

    monkeypatch.setattr(
        external_recovery,
        "_remove_bound_directory_tree",
        forbidden_delete,
    )
    with journals.workspace_mutation_lock(source):
        with pytest.raises(
            journals.MutationJournalError,
            match="duplicate|overlapping",
        ):
            external_recovery.reap_external_recovery_tombstones(
                corpus,
                source,
            )

    assert delete_touches == []
    assert first_tombstone.is_dir()
    assert second_tombstone.is_dir()


def test_external_recovery_revalidates_global_inventory_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate inserted after inventory confirmation cannot reuse proof."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    (
        source,
        corpus,
        journal,
        _original,
        first_tombstone,
        _first_audit,
        first_event,
    ) = _publish_quiesced_external_tombstone(
        tmp_path,
        run_id="inventory-revalidation-a",
    )
    real_bound_reaper = (
        external_recovery._reap_external_recovery_tombstones_bound
    )
    second_tombstones: list[Path] = []
    delete_touches: list[str] = []

    def insert_after_inventory_confirmation(
        corpus_arg: Any,
        source_arg: str | Path,
        inventory: Any,
    ) -> tuple[str, ...]:
        second_tombstones.append(
            _clone_external_tombstone_authority(
                corpus=corpus,
                journal=journal,
                first_tombstone=first_tombstone,
                first_event=first_event,
                run_id="inventory-revalidation-b",
            )
        )
        return real_bound_reaper(corpus_arg, source_arg, inventory)

    def forbidden_delete(*_args: object, **_kwargs: object) -> None:
        delete_touches.append("delete")
        raise AssertionError("changed global inventory reached deletion")

    monkeypatch.setattr(
        external_recovery,
        "_reap_external_recovery_tombstones_bound",
        insert_after_inventory_confirmation,
    )
    monkeypatch.setattr(
        external_recovery,
        "_remove_bound_directory_tree",
        forbidden_delete,
    )
    with journals.workspace_mutation_lock(source):
        with pytest.raises(
            journals.MutationJournalError,
            match="inventory|duplicate|overlapping|changed",
        ):
            external_recovery.reap_external_recovery_tombstones(
                corpus,
                source,
            )

    assert len(second_tombstones) == 1
    assert delete_touches == []
    assert first_tombstone.is_dir()
    assert second_tombstones[0].is_dir()


@pytest.mark.parametrize("mutation_kind", ("replacement", "same_inode"))
def test_immediate_external_recovery_reauthenticates_audit_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    """An audit mutation at the delete boundary retains the tombstone."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals
    from .test_workflows_journal_recover import _setup as recovery_setup

    state = recovery_setup(tmp_path)
    source = cast(Path, state["source"])
    corpus = cast(Any, state["corpus"])
    request = cast(Any, state["request"])
    journal = cast(Any, state["journal"])
    audit = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery"
        / f"{request.transaction_id}.json"
    )
    tombstone = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery_tombstones"
        / request.transaction_id
    )
    real_remove = external_recovery._remove_bound_directory_tree
    mutations: list[tuple[int, int]] = []

    def mutate_before_guard(
        parent_fd: int,
        name: str,
        directory_fd: int,
        expected: os.stat_result,
        *,
        pre_remove_guard: Any = None,
        authorized_inventory: Any = None,
    ) -> None:
        if name == request.transaction_id and not mutations:
            mutations.append(
                _mutate_recovery_audit(audit, kind=mutation_kind)
            )
        real_remove(
            parent_fd,
            name,
            directory_fd,
            expected,
            pre_remove_guard=pre_remove_guard,
            authorized_inventory=authorized_inventory,
        )

    monkeypatch.setattr(
        external_recovery,
        "_remove_bound_directory_tree",
        mutate_before_guard,
    )
    with journals.workspace_mutation_lock(source):
        with pytest.raises(
            journals.MutationJournalError,
            match="audit.*(changed|content)|destructive commit",
        ):
            external_recovery.inspect_external_recovery(
                corpus,
                source,
                request,
                apply=True,
            )

    assert len(mutations) == 1
    before, after = mutations[0]
    assert (before == after) == (mutation_kind == "same_inode")
    assert tombstone.is_dir()
    assert (tombstone / "manifest.json").is_file()
    assert not journal.journal_dir.exists()


@pytest.mark.parametrize("mutation_kind", ("replacement", "same_inode"))
def test_external_reaper_reauthenticates_audit_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    """The second-pass reaper retains authority when its audit changes."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    source, corpus, journal, _original, tombstone, audit, _event = (
        _publish_quiesced_external_tombstone(
            tmp_path,
            run_id=f"reaper-audit-{mutation_kind}",
        )
    )
    real_remove = external_recovery._remove_bound_directory_tree
    mutations: list[tuple[int, int]] = []

    def mutate_before_guard(
        parent_fd: int,
        name: str,
        directory_fd: int,
        expected: os.stat_result,
        *,
        pre_remove_guard: Any = None,
        authorized_inventory: Any = None,
    ) -> None:
        if name == journal.transaction_id and not mutations:
            mutations.append(
                _mutate_recovery_audit(audit, kind=mutation_kind)
            )
        real_remove(
            parent_fd,
            name,
            directory_fd,
            expected,
            pre_remove_guard=pre_remove_guard,
            authorized_inventory=authorized_inventory,
        )

    monkeypatch.setattr(
        external_recovery,
        "_remove_bound_directory_tree",
        mutate_before_guard,
    )
    with journals.workspace_mutation_lock(source):
        with pytest.raises(
            journals.MutationJournalError,
            match="audit.*(changed|content)|destructive commit",
        ):
            external_recovery.reap_external_recovery_tombstones(
                corpus,
                source,
            )

    assert len(mutations) == 1
    before, after = mutations[0]
    assert (before == after) == (mutation_kind == "same_inode")
    assert tombstone.is_dir()
    assert (tombstone / "manifest.json").is_file()


def test_external_reaper_post_guard_mutation_retains_durable_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit corruption after the last guard cannot destroy every proof."""

    from crucible.workflows import _journal_recovery as external_recovery
    from crucible.workflows import _mutation_journal as journals

    source, corpus, journal, _original, tombstone, audit, event = (
        _publish_quiesced_external_tombstone(
            tmp_path,
            run_id="reaper-post-guard-audit-race",
        )
    )
    expected_audit_payload = audit.read_bytes()
    real_remove = external_recovery._remove_bound_directory_tree
    mutations: list[tuple[int, int]] = []

    def mutate_after_guard(
        parent_fd: int,
        name: str,
        directory_fd: int,
        expected: os.stat_result,
        *,
        pre_remove_guard: Any = None,
        authorized_inventory: Any = None,
    ) -> None:
        if name == journal.transaction_id and not mutations:
            assert pre_remove_guard is not None
            pre_remove_guard()
            mutations.append(
                _mutate_recovery_audit(audit, kind="same_inode")
            )
            real_remove(
                parent_fd,
                name,
                directory_fd,
                expected,
                pre_remove_guard=None,
                authorized_inventory=authorized_inventory,
            )
            return
        real_remove(
            parent_fd,
            name,
            directory_fd,
            expected,
            pre_remove_guard=pre_remove_guard,
            authorized_inventory=authorized_inventory,
        )

    monkeypatch.setattr(
        external_recovery,
        "_remove_bound_directory_tree",
        mutate_after_guard,
    )
    with journals.workspace_mutation_lock(source):
        with pytest.raises(
            journals.MutationJournalError,
            match="audit.*(changed|content)|destructive commit",
        ):
            external_recovery.reap_external_recovery_tombstones(
                corpus,
                source,
            )

    assert len(mutations) == 1
    before, after = mutations[0]
    assert before == after
    tombstone_manifest = tombstone / "manifest.json"
    tombstone_authoritative = False
    if tombstone_manifest.is_file():
        tombstone_authoritative = (
            json.loads(tombstone_manifest.read_bytes())
            == event["writer_epoch_manifest"]
        )
    if tombstone_authoritative:
        return

    receipt_path = (
        audit.parent.parent
        / "journal_recovery_receipts"
        / f"{journal.transaction_id}.json"
    )
    receipt_metadata = receipt_path.lstat()
    assert stat.S_ISREG(receipt_metadata.st_mode)
    assert receipt_metadata.st_nlink == 1
    receipt = json.loads(receipt_path.read_bytes())
    assert isinstance(receipt, dict)
    assert set(receipt) == {
        "schema",
        "run_id",
        "transaction_id",
        "source_root",
        "tombstone",
        "tombstone_identity",
        "audit_payload",
        "audit_sha256",
    }
    assert receipt["schema"] == (
        "crucible.external-git-journal-recovery-cleanup.v1"
    )
    assert receipt["run_id"] == journal.journal_dir.parent.parent.name
    assert receipt["transaction_id"] == journal.transaction_id
    assert receipt["source_root"] == str(source.resolve())
    assert receipt["tombstone"] == str(tombstone)
    assert receipt["tombstone_identity"] == event["tombstone_identity"]
    audit_payload = receipt["audit_payload"]
    assert isinstance(audit_payload, str)
    assert audit_payload.encode("utf-8") == expected_audit_payload
    assert receipt["audit_sha256"] == hashlib.sha256(
        expected_audit_payload
    ).hexdigest()
    embedded_event = json.loads(audit_payload)
    assert embedded_event == event
    assert isinstance(embedded_event, dict)
    proof = external_recovery._parse_external_writer_epoch_proof(
        embedded_event,
        transaction_id=journal.transaction_id,
    )
    expected_proof = external_recovery._parse_external_writer_epoch_proof(
        event,
        transaction_id=journal.transaction_id,
    )
    assert proof.to_record() == expected_proof.to_record()


def test_quiesced_root_cannot_cover_arbitrarily_missing_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, parent, child = _begin_family(
        tmp_path,
        run_id="sparse-root",
    )
    lease = _arm_family(parent, child, identifier="sparse-root")
    _quiesce_family(lease)
    journals._unregister_live(child.transaction_id)
    touched: list[str] = []

    def forbidden_source_read():
        touched.append("source")
        raise AssertionError("source opened under a forged sparse root")

    monkeypatch.setattr(parent, "_bound_workspace_cwd", forbidden_source_read)
    try:
        with pytest.raises(journals.MutationJournalError, match="writer epoch"):
            parent.prepare_git_capture()
        assert touched == []
    finally:
        journals._unregister_live(parent.transaction_id)


def test_reparent_rejects_every_generated_epoch(tmp_path: Path) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, journal = _begin_single(tmp_path, run_id="reparent")
    lease = _arm_family(journal, identifier="reparent")
    _quiesce_family(lease)
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="writer epoch|reparent",
        ):
            journal.reparent("future-parent")
        assert journal.parent_transaction_id is None
    finally:
        _unregister(journal)


_FRONTIERS = (
    "assert_exact_baseline",
    "assert_head_unchanged",
    "assert_index_unchanged",
    "assert_repository_metadata_unchanged",
    "authorized_scope_changed",
    "commit",
    "detected_scope_escapes",
    "lifecycle_guard_operation",
    "lifecycle_settlement_operation",
    "locally_changed_paths",
    "locally_changed_scope_paths",
    "prepare",
    "prepare_git_capture",
    "record_implementation",
    "record_scope_escapes",
    "restore",
    "restore_paths_from_snapshot",
    "retry_cleanup",
    "settle_or_restore_outcome",
    "validate_implementation",
)


def _invoke_frontier(name: str, journal: Any, corpus: Any) -> object:
    implementation = MagicMock(
        scope_escapes=(),
        files_modified=(),
        classification="success",
    )
    if name == "authorized_scope_changed":
        return journal._authorized_scope_changed()
    if name == "lifecycle_guard_operation":
        return journal._lifecycle_guard_operation()
    if name == "lifecycle_settlement_operation":
        return journal._lifecycle_settlement_operation()
    if name == "locally_changed_paths":
        return journal.locally_changed_paths(("a.py",))
    if name == "prepare":
        return journal.prepare(implementation, keep=True)
    if name == "record_implementation":
        return journal.record_implementation(implementation)
    if name == "record_scope_escapes":
        return journal.record_scope_escapes(("a.py",))
    if name == "restore_paths_from_snapshot":
        return journal.restore_paths_from_snapshot(("a.py",))
    if name == "settle_or_restore_outcome":
        return journal.settle_or_restore_outcome(corpus)
    if name == "validate_implementation":
        return journal.validate_implementation(implementation)
    return getattr(journal, name)()


@pytest.mark.parametrize("frontier", _FRONTIERS)
def test_armed_epoch_blocks_every_source_git_and_terminal_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frontier: str,
) -> None:
    """Each frontier blocks before its first source/Git observation."""

    from crucible.workflows import _mutation_journal as journals

    _repo, corpus, journal = _begin_single(
        tmp_path,
        run_id=f"frontier-{frontier}",
    )
    _arm_family(journal, identifier=f"frontier-{frontier}")
    touched: list[str] = []

    def forbidden_source_access(*_args, **_kwargs):
        touched.append("source")
        raise AssertionError(f"{frontier} touched source/Git before writer quiescence")

    monkeypatch.setattr(
        journal,
        "_bound_workspace_cwd",
        forbidden_source_access,
    )
    monkeypatch.setattr(
        journal,
        "_assert_git_entry_unchanged",
        forbidden_source_access,
    )
    blocked = False
    try:
        try:
            result = _invoke_frontier(frontier, journal, corpus)
        except journals.MutationJournalError as exc:
            assert "writer epoch" in str(exc).lower()
            blocked = True
        else:
            if frontier in {"restore", "retry_cleanup"}:
                blocked = result is False
            elif frontier == "settle_or_restore_outcome":
                blocked = getattr(result, "completed", True) is False
        assert blocked, f"{frontier} crossed an armed writer epoch"
        assert touched == []
    finally:
        _unregister(journal)


def test_lifecycle_family_marks_children_then_root_before_any_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, parent, child = _begin_family(
        tmp_path,
        run_id="family-finalize-order",
    )
    lease = _arm_family(parent, child, identifier="family-finalize-order")
    _quiesce_family(lease)
    family = {parent.transaction_id, child.transaction_id}
    for journal in (parent, child):
        journal._manifest["expected_lifecycle"] = {
            "settlement_ref": "sha256:" + ("b" * 64),
            "settlement_family": sorted(family),
        }
        journal._manifest["state"] = "prepared_keep"

    events: list[tuple[str, str]] = []

    def record_write(current):
        assert current._manifest["state"] == "committed"
        events.append(("mark", current.transaction_id))

    def record_cleanup(current, **_kwargs):
        events.append(("cleanup", current.transaction_id))
        journals._unregister_live(current.transaction_id)

    monkeypatch.setattr(
        journals.WorkspaceMutationJournal,
        "_write",
        record_write,
    )
    monkeypatch.setattr(
        journals.WorkspaceMutationJournal,
        "detected_scope_escapes",
        lambda _self: (),
    )
    for name in (
        "assert_head_unchanged",
        "assert_repository_metadata_unchanged",
        "assert_index_unchanged",
        "_assert_no_failed_descendant",
    ):
        monkeypatch.setattr(
            journals.WorkspaceMutationJournal,
            name,
            lambda _self, *args, **kwargs: None,
        )
    monkeypatch.setattr(
        journals,
        "_cleanup_terminal_mutation_journal",
        record_cleanup,
    )
    try:
        journals._commit_prevalidated_lifecycle_family((parent, child))
        assert events == [
            ("mark", child.transaction_id),
            ("mark", parent.transaction_id),
            ("cleanup", child.transaction_id),
            ("cleanup", parent.transaction_id),
        ]
    finally:
        _unregister(parent, child)


def test_lifecycle_family_rejects_cleanup_that_retains_process_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, parent, child = _begin_family(
        tmp_path,
        run_id="family-retained-owner",
    )
    lease = _arm_family(parent, child, identifier="family-retained-owner")
    _quiesce_family(lease)
    family = {parent.transaction_id, child.transaction_id}
    for journal in (parent, child):
        journal._manifest["expected_lifecycle"] = {
            "settlement_ref": "sha256:" + ("c" * 64),
            "settlement_family": sorted(family),
        }
        journal._manifest["state"] = "prepared_keep"
        journal._write()

    monkeypatch.setattr(
        journals,
        "_cleanup_terminal_mutation_journal",
        lambda _journal, **_kwargs: None,
    )
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="retained terminal owners",
        ) as captured:
            journals._commit_prevalidated_lifecycle_family((parent, child))
        assert all(
            transaction_id in str(captured.value)
            for transaction_id in family
        )
    finally:
        _unregister(parent, child)


def test_armed_epoch_blocks_multi_journal_lifecycle_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, corpus, parent, child = _begin_family(
        tmp_path,
        run_id="armed-lifecycle-prepare",
    )
    _arm_family(parent, child, identifier="armed-lifecycle-prepare")
    touched: list[str] = []

    def forbidden_source_access(*_args, **_kwargs):
        touched.append("source")
        raise AssertionError("lifecycle preparation crossed an armed epoch")

    monkeypatch.setattr(parent, "_bound_workspace_cwd", forbidden_source_access)
    monkeypatch.setattr(child, "_bound_workspace_cwd", forbidden_source_access)
    try:
        with pytest.raises(journals.MutationJournalError, match="writer epoch"):
            journals.prepare_mutation_lifecycle_settlement(
                corpus,
                (parent, child),
                kind="landed",
                reason="must remain blocked",
            )
        assert touched == []
    finally:
        _unregister(parent, child)


def test_terminal_retry_audits_epoch_before_cleanup_receipt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="terminal-retry-armed",
    )
    _arm_family(journal, identifier="terminal-retry-armed")
    journal._manifest["state"] = "rolled_back"
    journal._write()
    receipt_touches: list[str] = []

    def forbidden_receipt(*_args, **_kwargs):
        receipt_touches.append("receipt")
        raise AssertionError("terminal receipt published before epoch audit")

    monkeypatch.setattr(
        journals,
        "_ensure_terminal_cleanup_receipt",
        forbidden_receipt,
    )

    assert journal.retry_cleanup() is False
    assert receipt_touches == []
    assert journal.journal_dir.is_dir()
    _unregister(journal)


def test_terminal_disposition_audits_epoch_before_git_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="terminal-disposition-armed",
    )
    _arm_family(journal, identifier="terminal-disposition-armed")
    git_touches: list[str] = []

    def forbidden_git_proof(*_args, **_kwargs):
        git_touches.append("git")
        raise AssertionError("terminal disposition read Git before epoch audit")

    monkeypatch.setattr(
        journals,
        "_terminal_assert_git_authority",
        forbidden_git_proof,
    )
    try:
        with pytest.raises(journals.MutationJournalError, match="writer epoch"):
            journals._prove_terminal_cleanup_disposition(
                corpus,
                MagicMock(),
                journal,
            )
        assert git_touches == []
    finally:
        _unregister(journal)


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "O_PATH"),
    reason="POSIX descriptor-bound inaccessible-root recovery",
)
def test_armed_epoch_blocks_inaccessible_root_repair_before_chmod(
    tmp_path: Path,
) -> None:
    repo, _corpus, journal = _begin_single(
        tmp_path,
        run_id="armed-root-access",
    )
    baseline_mode = stat.S_IMODE(repo.stat().st_mode)
    lease = _arm_family(journal, identifier="armed-root-access")
    try:
        repo.chmod(0)

        assert journal.restore() is False
        assert stat.S_IMODE(repo.stat().st_mode) == 0

        _quiesce_family(lease)
        assert journal.restore()
        assert stat.S_IMODE(repo.stat().st_mode) == baseline_mode
    finally:
        if repo.exists():
            repo.chmod(baseline_mode)
        if journal.cleanup_pending:
            _unregister(journal)


def test_recovery_audits_epoch_before_terminal_receipt_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus, journal = _begin_single(
        tmp_path,
        run_id="recovery-before-receipt",
    )
    _arm_family(journal, identifier="recovery-before-receipt")
    _simulate_process_death(journal)
    original_discover = journals.discover_mutation_journals
    receipt_touches: list[str] = []

    def discover_with_receipt(*args, **kwargs):
        discovered = original_discover(*args, **kwargs)
        kwargs["_terminal_receipts"].append(object())
        return discovered

    def forbidden_receipt_cleanup(*_args, **_kwargs):
        receipt_touches.append("receipt")
        raise AssertionError("receipt cleanup ran before epoch audit")

    monkeypatch.setattr(
        journals,
        "discover_mutation_journals",
        discover_with_receipt,
    )
    monkeypatch.setattr(
        journals,
        "_resume_terminal_cleanup_receipt",
        forbidden_receipt_cleanup,
    )

    report = journals.recover_mutation_journals(corpus, repo)

    assert receipt_touches == []
    assert report.failures
    assert any("writer epoch" in failure.lower() for failure in report.failures)


def test_recovery_terminal_family_cleanup_is_children_first_root_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus, parent, child = _begin_family(
        tmp_path,
        run_id="recovery-cleanup-order",
    )
    lease = _arm_family(parent, child, identifier="recovery-cleanup-order")
    _quiesce_family(lease)
    for journal in (child, parent):
        journal._manifest["state"] = "rolled_back"
        journal._write()
    _simulate_process_death(child)
    _simulate_process_death(parent)
    cleanup_order: list[str] = []

    def record_cleanup(current, **_kwargs):
        cleanup_order.append(current.transaction_id)

    monkeypatch.setattr(
        journals,
        "_cleanup_terminal_mutation_journal",
        record_cleanup,
    )
    monkeypatch.setattr(
        journals.WorkspaceMutationJournal,
        "_acknowledge_resolved_descendants",
        lambda _self, _resolved: None,
    )

    journals.recover_mutation_journals(corpus, repo)

    assert cleanup_order[:2] == [child.transaction_id, parent.transaction_id]
