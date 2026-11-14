# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Late adversarial checks for UIR-01 cleanup authority.

This module deliberately reuses the comparatively expensive external-recovery
fixture from its owning test module.  Keeping the regression here avoids
editing another agent's production or test ownership while the final audit is
in progress.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from assay.corpus.store import Corpus

from crucible.workflows import _backends
from crucible.workflows._journal_recovery import (
    ExternalRecoveryRequest,
    inspect_external_recovery,
    reap_external_recovery_tombstones,
)
from crucible.workflows._mutation_journal import (
    MutationJournalError,
    WorkspaceMutationJournal,
    recover_mutation_journals,
    workspace_mutation_lock,
)
from crucible.workflows._provider_settlement import (
    ProviderSettlementCleanupError,
    ProviderSettlementExitStack,
)
from crucible.workflows._writer_epoch import (
    initial_writer_epoch_record,
    inspect_writer_epoch_family,
)
from crucible.workflows._build_verify import (
    _RunnerSettlementError,
    run_supervised_process,
)


def _journal_recovery_helpers() -> ModuleType:
    path = Path(__file__).with_name("test_workflows_journal_recover.py")
    spec = importlib.util.spec_from_file_location(
        "_uir01_journal_recovery_helpers",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leave_external_tombstone_before_control_delete(
    state: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Crash after rename/receipt publication but before control deletion."""

    from crucible.workflows import _journal_recovery

    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    original = _journal_recovery._remove_bound_regular_entry

    def stop_before_control_delete(directory_fd, name, **kwargs):
        if name == "external_recovery.json":
            raise RuntimeError("simulated death before control deletion")
        return original(directory_fd, name, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_regular_entry",
        stop_before_control_delete,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="before control deletion"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_regular_entry",
        original,
    )
    run = Path(corpus.spec_dir) / "runs" / request.run_id
    tombstone = (
        run / "journal_recovery_tombstones" / request.transaction_id
    )
    audit = run / "journal_recovery" / f"{request.transaction_id}.json"
    assert tombstone.is_dir()
    assert (tombstone / "external_recovery.json").is_file()
    return tombstone, audit


def _leave_source_restored_for_startup_attestation(
    state: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash after source restoration so startup owns the exact family map."""

    from crucible.workflows import _journal_recovery

    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    original = _journal_recovery._write_event
    stopped = False

    def stop_after_source_restore(current, event):
        nonlocal stopped
        original(current, event)
        if event.get("stage") == "source_restored" and not stopped:
            stopped = True
            raise RuntimeError("stop before startup attestation")

    monkeypatch.setattr(
        _journal_recovery,
        "_write_event",
        stop_after_source_restore,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="startup attestation"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_write_event", original)
    assert stopped
    assert (source / "a.txt").read_bytes() == b"baseline a.txt\n"
    assert not (source / "new.txt").exists()


def _startup_attestation_locations(
    state: dict[str, object],
) -> tuple[Path, Path]:
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)
    tombstone = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery_tombstones"
        / request.transaction_id
    )
    return journal.journal_dir, tombstone


def _run_failed_startup_attestation(
    state: dict[str, object],
) -> object:
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    before = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source).parts
    }
    with workspace_mutation_lock(source):
        report = recover_mutation_journals(corpus, source)
    after = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source).parts
    }
    assert report.recovered == ()
    assert report.failures
    assert after == before
    assert (source / "a.txt").read_bytes() == b"baseline a.txt\n"
    assert not (source / "new.txt").exists()
    return report


def test_startup_attestation_rejects_wrong_root_fd_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real directory descriptor cannot stand in for the proof root."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    _leave_source_restored_for_startup_attestation(state, monkeypatch)
    source = state["source"]
    assert isinstance(source, Path)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._prevalidate_installed_writer_epoch_family
    injected = False

    def use_wrong_root(family, event, _journal_fd):
        nonlocal injected
        wrong_fd = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            injected = True
            return original(family, event, wrong_fd)
        finally:
            os.close(wrong_fd)

    monkeypatch.setattr(
        _journal_recovery,
        "_prevalidate_installed_writer_epoch_family",
        use_wrong_root,
    )
    report = _run_failed_startup_attestation(state)
    journal, tombstone = _startup_attestation_locations(state)
    assert injected
    assert any("exact proof root" in failure for failure in report.failures)
    assert journal.is_dir()
    assert not tombstone.exists()


def test_startup_attestation_rejects_tls_member_substitution_and_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen owner objects must remain installed after prevalidation."""

    from crucible.workflows import _journal_recovery, _mutation_journal

    helpers = _journal_recovery_helpers()
    for mode in ("substitution", "removal"):
        state = helpers._setup(tmp_path / mode)
        _leave_source_restored_for_startup_attestation(state, monkeypatch)
        original = _journal_recovery._prevalidate_installed_writer_epoch_family
        injected = False

        def change_tls(family, event, journal_fd, *, _mode=mode):
            nonlocal injected
            attestation = original(family, event, journal_fd)
            installed = getattr(
                _mutation_journal._LOCK_STATE,
                "writer_epoch_recovery_members",
            )
            transaction_id = attestation.members[-1][0]
            if _mode == "substitution":
                installed[transaction_id] = object()
            else:
                del installed[transaction_id]
            injected = True
            return attestation

        monkeypatch.setattr(
            _journal_recovery,
            "_prevalidate_installed_writer_epoch_family",
            change_tls,
        )
        report = _run_failed_startup_attestation(state)
        journal, tombstone = _startup_attestation_locations(state)
        assert injected
        assert any(
            "owner changed after rename" in failure
            for failure in report.failures
        )
        assert not journal.exists()
        assert tombstone.is_dir()
        assert (tombstone / "external_recovery.json").is_file()
        monkeypatch.setattr(
            _journal_recovery,
            "_prevalidate_installed_writer_epoch_family",
            original,
        )


def test_startup_attestation_rejects_same_inode_control_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared event bytes cannot change in place after they are frozen."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    _leave_source_restored_for_startup_attestation(state, monkeypatch)
    from crucible.workflows import _journal_recovery

    original = _journal_recovery._prevalidate_installed_writer_epoch_family
    injected = False

    def rewrite_control(family, event, journal_fd):
        nonlocal injected
        attestation = original(family, event, journal_fd)
        root = next(
            member for member in family
            if member.parent_transaction_id is None
        )
        control = root.journal_dir / "external_recovery.json"
        before = control.stat()
        request = event.get("request")
        assert isinstance(request, dict)
        request["reason"] = f"{request['reason']} [forged after attestation]"
        payload = _journal_recovery._encode_recovery_event(event)
        with control.open("r+b", buffering=0) as stream:
            stream.seek(0)
            stream.write(payload)
            stream.truncate()
            os.fsync(stream.fileno())
        after = control.stat()
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        injected = True
        return attestation

    monkeypatch.setattr(
        _journal_recovery,
        "_prevalidate_installed_writer_epoch_family",
        rewrite_control,
    )
    report = _run_failed_startup_attestation(state)
    journal, tombstone = _startup_attestation_locations(state)
    assert injected
    assert any(
        "artifact" in failure or "attestation" in failure
        for failure in report.failures
    )
    assert journal.is_dir()
    assert not tombstone.exists()


def test_startup_attestation_rejects_noncontrol_inventory_and_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-control journal byte remains bound after prevalidation."""

    from crucible.workflows import _journal_recovery

    helpers = _journal_recovery_helpers()
    for mode in ("inventory", "manifest"):
        state = helpers._setup(tmp_path / mode)
        _leave_source_restored_for_startup_attestation(state, monkeypatch)
        original = _journal_recovery._prevalidate_installed_writer_epoch_family
        injected = False

        def drift_noncontrol(family, event, journal_fd, *, _mode=mode):
            nonlocal injected
            attestation = original(family, event, journal_fd)
            root = next(
                member for member in family
                if member.parent_transaction_id is None
            )
            if _mode == "inventory":
                intruder = root.journal_dir / "post-attestation-intruder"
                intruder.write_bytes(b"unbound\n")
            else:
                manifest = root.journal_dir / "manifest.json"
                payload = manifest.read_bytes()
                needle = b'"external_recovery_verified"'
                replacement = b'"external_recovery_verifieX"'
                assert len(needle) == len(replacement)
                assert needle in payload
                before = manifest.stat()
                with manifest.open("r+b", buffering=0) as stream:
                    stream.seek(0)
                    stream.write(payload.replace(needle, replacement, 1))
                    stream.truncate()
                    os.fsync(stream.fileno())
                after = manifest.stat()
                assert (after.st_dev, after.st_ino) == (
                    before.st_dev,
                    before.st_ino,
                )
            injected = True
            return attestation

        monkeypatch.setattr(
            _journal_recovery,
            "_prevalidate_installed_writer_epoch_family",
            drift_noncontrol,
        )
        report = _run_failed_startup_attestation(state)
        journal, tombstone = _startup_attestation_locations(state)
        assert injected
        assert any(
            "artifact" in failure or "manifest" in failure
            for failure in report.failures
        )
        assert journal.is_dir()
        assert not tombstone.exists()
        monkeypatch.setattr(
            _journal_recovery,
            "_prevalidate_installed_writer_epoch_family",
            original,
        )


class _MemoryEpochStorage:
    """Minimal durable storage for provider-settlement adversaries."""

    transaction_id = "final-provider-adversary"
    parent_transaction_id = None
    active = True

    def __init__(self) -> None:
        self.record = initial_writer_epoch_record()

    def read_writer_epoch_record(self) -> object:
        return copy.deepcopy(self.record)

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        if self.record != expected:
            raise RuntimeError("writer epoch compare-and-write conflict")
        self.record = copy.deepcopy(replacement)
        return copy.deepcopy(self.record)


def test_external_cleanup_receipt_repairs_audit_after_delete_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The independent receipt must close deletion-to-audit crash recovery."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)

    from crucible.workflows import _journal_recovery

    audit = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery"
        / f"{request.transaction_id}.json"
    )
    receipt = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery_receipts"
        / f"{request.transaction_id}.json"
    )
    original = _journal_recovery._remove_bound_directory_tree

    def crash_after_delete(parent_fd, name, directory_fd, expected, **kwargs):
        result = original(parent_fd, name, directory_fd, expected, **kwargs)
        if name == request.transaction_id:
            # Model an audit race followed by process death after the journal
            # tombstone is gone but before in-process reconciliation runs.
            audit.write_text("{}\n", encoding="utf-8")
            raise RuntimeError("simulated death after journal deletion")
        return result

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        crash_after_delete,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="death after journal deletion"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        original,
    )

    assert receipt.is_file()
    assert not journal.journal_dir.exists()
    assert json.loads(audit.read_text(encoding="utf-8")) == {}

    with workspace_mutation_lock(source):
        report = recover_mutation_journals(corpus, source)

    assert report.failures == ()
    repaired = json.loads(audit.read_text(encoding="utf-8"))
    assert repaired["schema"] == "crucible.external-git-journal-recovery.v2"
    assert repaired["request"]["transaction_id"] == request.transaction_id


def test_external_cleanup_receipt_accepts_near_ceiling_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every admissible event has a writable and readable receipt envelope."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    _tombstone, audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )

    from crucible.workflows import _journal_recovery

    event = json.loads(audit.read_bytes())
    event["receipt_boundary_padding"] = ""
    base_payload = _journal_recovery._encode_recovery_event(event)
    target_size = _journal_recovery._MAX_RECOVERY_EVENT_BYTES - 128
    assert len(base_payload) < target_size
    event["receipt_boundary_padding"] = "x" * (
        target_size - len(base_payload)
    )
    audit_payload = _journal_recovery._encode_recovery_event(event)
    assert len(audit_payload) == target_size

    record = _journal_recovery._external_cleanup_receipt_record(
        event=event,
        audit_payload=audit_payload,
    )
    encoded_receipt = (
        json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert len(encoded_receipt) <= (
        _journal_recovery._MAX_RECOVERY_CLEANUP_RECEIPT_BYTES
    )

    receipt_root = tmp_path / "near-ceiling-receipts"
    receipt_root.mkdir()
    receipt_fd = os.open(
        receipt_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    receipt_name = f"{request.transaction_id}.json"
    logical_run = Path(corpus.spec_dir) / "runs" / request.run_id
    try:
        _journal_recovery._write_exclusive_bound_json(
            receipt_fd,
            receipt_name,
            record,
        )
        with _journal_recovery._bound_recovery_audit(
            receipt_fd,
            receipt_name,
            max_bytes=(
                _journal_recovery._MAX_RECOVERY_CLEANUP_RECEIPT_BYTES
            ),
        ) as (observed, _fd, _opened, _digest):
            parsed_event, parsed_payload = (
                _journal_recovery._parse_external_cleanup_receipt(
                    observed,
                    logical_run=logical_run,
                    receipt_name=receipt_name,
                )
            )
    finally:
        os.close(receipt_fd)

    assert parsed_event == event
    assert parsed_payload == audit_payload

    noncanonical_payload = json.dumps(
        event,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert json.loads(noncanonical_payload) == event
    assert noncanonical_payload != audit_payload
    forged_record = copy.deepcopy(record)
    forged_record["audit_payload"] = noncanonical_payload.decode("utf-8")
    forged_record["audit_sha256"] = hashlib.sha256(
        noncanonical_payload
    ).hexdigest()
    with pytest.raises(MutationJournalError, match="canonical"):
        _journal_recovery._parse_external_cleanup_receipt(
            forged_record,
            logical_run=logical_run,
            receipt_name=receipt_name,
        )


def test_external_reaper_refuses_unbound_tombstone_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deletion may consume only the manifest-bound journal inventory."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, _audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )
    foreign = tombstone / "not-authorized-by-terminal-manifest.txt"
    foreign.write_text("must not be deleted\n", encoding="utf-8")

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="inventory|unexpected|changed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert foreign.read_text(encoding="utf-8") == "must not be deleted\n"


def test_external_reaper_rechecks_inventory_at_delete_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign node racing after proof must not be consumed by cleanup."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    tombstone, _audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )

    from crucible.workflows import _journal_recovery

    original = _journal_recovery._remove_bound_directory_tree
    foreign = tombstone / "raced-after-authority-proof.txt"

    def inject_after_proof(parent_fd, name, directory_fd, expected, **kwargs):
        if name == request.transaction_id:
            foreign.write_text("must survive\n", encoding="utf-8")
        return original(parent_fd, name, directory_fd, expected, **kwargs)

    monkeypatch.setattr(
        _journal_recovery,
        "_remove_bound_directory_tree",
        inject_after_proof,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="inventory|unauthorized|changed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert foreign.read_text(encoding="utf-8") == "must survive\n"


def test_external_recovery_resumes_after_tombstone_prepare_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable pre-rename stage must remain resumable after process death."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)

    from crucible.workflows import _journal_recovery

    original = _journal_recovery._write_event

    def crash_after_prepare(current, event):
        original(current, event)
        if event.get("stage") == "tombstone_prepared":
            raise RuntimeError("simulated death after tombstone prepare")

    monkeypatch.setattr(_journal_recovery, "_write_event", crash_after_prepare)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="death after tombstone prepare"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "_write_event", original)

    assert journal.journal_dir.is_dir()
    durable = json.loads(
        (journal.journal_dir / "external_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert durable["stage"] == "tombstone_prepared"

    with workspace_mutation_lock(source):
        report = inspect_external_recovery(corpus, source, request, apply=True)

    assert report.status == "recovered"


def test_external_recovery_resumes_control_published_before_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared control publication may precede its mirrored audit on crash."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)

    from crucible.workflows import _journal_recovery

    original = _journal_recovery.write_atomic_bytes
    crashed = [False]

    def crash_before_prepared_audit(path, payload):
        target = Path(path)
        try:
            record = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            record = {}
        if (
            not crashed[0]
            and target.parent.resolve().name == "journal_recovery"
            and record.get("stage") == "tombstone_prepared"
        ):
            crashed[0] = True
            raise RuntimeError("simulated death before prepared audit")
        return original(path, payload)

    monkeypatch.setattr(
        _journal_recovery,
        "write_atomic_bytes",
        crash_before_prepared_audit,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="before prepared audit"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(_journal_recovery, "write_atomic_bytes", original)

    assert crashed == [True]
    control = json.loads(
        (journal.journal_dir / "external_recovery.json").read_text(
            encoding="utf-8"
        )
    )
    audit_path = (
        Path(corpus.spec_dir)
        / "runs"
        / request.run_id
        / "journal_recovery"
        / f"{request.transaction_id}.json"
    )
    mirrored = json.loads(audit_path.read_text(encoding="utf-8"))
    assert control["stage"] == "tombstone_prepared"
    assert mirrored["stage"] != "tombstone_prepared"

    with workspace_mutation_lock(source):
        report = inspect_external_recovery(corpus, source, request, apply=True)

    assert report.status == "recovered"


def test_external_reaper_resumes_after_rename_before_complete_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prepared audit/control pair is restart authority after rename."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    journal = state["journal"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    assert isinstance(journal, WorkspaceMutationJournal)

    original_rename = os.rename
    crashed = [False]

    def crash_after_tombstone_rename(
        source_name,
        destination_name,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        result = original_rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if (
            not crashed[0]
            and str(destination_name) == request.transaction_id
            and dst_dir_fd is not None
            and Path(f"/proc/self/fd/{dst_dir_fd}").resolve().name
            == "journal_recovery_tombstones"
        ):
            crashed[0] = True
            raise RuntimeError("simulated death after tombstone rename")
        return result

    monkeypatch.setattr(os, "rename", crash_after_tombstone_rename)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="after tombstone rename"):
            inspect_external_recovery(corpus, source, request, apply=True)
    monkeypatch.setattr(os, "rename", original_rename)

    run = Path(corpus.spec_dir) / "runs" / request.run_id
    tombstone = (
        run / "journal_recovery_tombstones" / request.transaction_id
    )
    audit = run / "journal_recovery" / f"{request.transaction_id}.json"
    receipt = (
        run / "journal_recovery_receipts" / f"{request.transaction_id}.json"
    )
    assert crashed == [True]
    assert not journal.journal_dir.exists()
    assert tombstone.is_dir()
    assert json.loads(audit.read_text(encoding="utf-8"))["stage"] == (
        "tombstone_prepared"
    )
    assert (tombstone / "external_recovery.json").is_file()
    assert not receipt.exists()

    with workspace_mutation_lock(source):
        recovered = reap_external_recovery_tombstones(corpus, source)

    assert recovered == ("s0",)
    assert not tombstone.exists()
    assert json.loads(audit.read_text(encoding="utf-8"))["stage"] == "complete"
    assert receipt.is_file()


def test_external_reaper_resumes_after_rename_before_control_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The permanent receipt resumes a renamed tombstone with its control."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    tombstone, _audit = _leave_external_tombstone_before_control_delete(
        state,
        monkeypatch,
    )

    with workspace_mutation_lock(source):
        recovered = reap_external_recovery_tombstones(corpus, source)

    assert recovered == ("s0",)
    assert not tombstone.exists()


def test_external_reaper_resumes_after_control_delete_before_tree_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receipt authorizes the frozen subset after control removal."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    tombstone, _audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )
    assert not (tombstone / "external_recovery.json").exists()

    with workspace_mutation_lock(source):
        recovered = reap_external_recovery_tombstones(corpus, source)

    assert recovered == ("s0",)
    assert not tombstone.exists()


def test_external_reaper_refuses_same_byte_control_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact bytes do not authorize deleting a replacement control inode."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, _audit = _leave_external_tombstone_before_control_delete(
        state,
        monkeypatch,
    )
    control = tombstone / "external_recovery.json"
    payload = control.read_bytes()
    original_metadata = control.stat()
    parked = tmp_path / "parked-original-external-recovery.json"
    control.rename(parked)
    control.write_bytes(payload)
    replacement_metadata = control.stat()
    assert (replacement_metadata.st_dev, replacement_metadata.st_ino) != (
        original_metadata.st_dev,
        original_metadata.st_ino,
    )

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="tombstone control identity changed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert control.read_bytes() == payload
    assert parked.read_bytes() == payload


def test_external_reaper_refuses_in_place_control_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-inode control mutation cannot consume the receipt authority."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, _audit = _leave_external_tombstone_before_control_delete(
        state,
        monkeypatch,
    )
    control = tombstone / "external_recovery.json"
    metadata = control.stat()
    payload = control.read_bytes()
    control.write_bytes(payload + b"\n")
    changed = control.stat()
    assert (changed.st_dev, changed.st_ino) == (
        metadata.st_dev,
        metadata.st_ino,
    )

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="control|audit|inventory|changed|malformed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert control.read_bytes() == payload + b"\n"


def test_external_reaper_resumes_bound_control_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the exact control rename remains receipt-resumable."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, _audit = _leave_external_tombstone_before_control_delete(
        state,
        monkeypatch,
    )

    original_unlink = os.unlink
    crashed = [False]
    quarantined: list[Path] = []

    def crash_before_control_unlink(path, *args, dir_fd=None, **kwargs):
        if (
            not crashed[0]
            and isinstance(path, str)
            and path.startswith(".crucible-cleanup-external-control-")
            and dir_fd is not None
        ):
            crashed[0] = True
            quarantined.append(Path(f"/proc/self/fd/{dir_fd}").resolve() / path)
            raise RuntimeError("simulated death after control quarantine")
        return original_unlink(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(os, "unlink", crash_before_control_unlink)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="after control quarantine"):
            reap_external_recovery_tombstones(corpus, source)
    monkeypatch.setattr(os, "unlink", original_unlink)

    assert crashed == [True]
    assert len(quarantined) == 1
    assert quarantined[0].is_file()
    assert not (tombstone / "external_recovery.json").exists()

    with workspace_mutation_lock(source):
        recovered = reap_external_recovery_tombstones(corpus, source)

    assert recovered == ("s0",)
    assert not tombstone.exists()


def test_external_reaper_retains_unbound_control_quarantine_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash before control rename cannot turn an empty name into proof."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, _audit = _leave_external_tombstone_before_control_delete(
        state,
        monkeypatch,
    )

    original_rename = os.rename
    crashed = [False]
    reservations: list[Path] = []

    def crash_after_control_reservation(
        source_name,
        destination_name,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        if (
            not crashed[0]
            and isinstance(destination_name, str)
            and destination_name.startswith(
                ".crucible-cleanup-external-control-"
            )
            and dst_dir_fd is not None
        ):
            crashed[0] = True
            reservations.append(
                Path(f"/proc/self/fd/{dst_dir_fd}").resolve()
                / destination_name
            )
            raise RuntimeError("simulated death after control reservation")
        return original_rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", crash_after_control_reservation)
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="after control reservation"):
            reap_external_recovery_tombstones(corpus, source)
    monkeypatch.setattr(os, "rename", original_rename)

    assert crashed == [True]
    assert len(reservations) == 1
    reservation = reservations[0]
    assert reservation.is_file()
    assert reservation.stat().st_size == 0
    residue_identity = (reservation.stat().st_dev, reservation.stat().st_ino)

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="inventory|unauthorized|changed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert any(
        (metadata.st_dev, metadata.st_ino) == residue_identity
        for candidate in tombstone.rglob("*")
        if (metadata := candidate.lstat())
    )


def test_external_reaper_resumes_after_partial_tree_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen inventory admits only an exact deletion-only subset."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    tombstone, _audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )
    before_files = sum(candidate.is_file() for candidate in tombstone.rglob("*"))

    from crucible.workflows import _mutation_journal

    original = _mutation_journal._remove_bound_regular_entry
    crashed = [False]

    def crash_after_one_file(directory_fd, name, **kwargs):
        directory = Path(f"/proc/self/fd/{directory_fd}").resolve()
        result = original(directory_fd, name, **kwargs)
        if (
            not crashed[0]
            and (directory == tombstone or tombstone in directory.parents)
        ):
            crashed[0] = True
            raise RuntimeError("simulated death after partial tree deletion")
        return result

    monkeypatch.setattr(
        _mutation_journal,
        "_remove_bound_regular_entry",
        crash_after_one_file,
    )
    with workspace_mutation_lock(source):
        with pytest.raises(RuntimeError, match="partial tree deletion"):
            reap_external_recovery_tombstones(corpus, source)
    monkeypatch.setattr(
        _mutation_journal,
        "_remove_bound_regular_entry",
        original,
    )

    assert crashed == [True]
    assert tombstone.is_dir()
    assert sum(candidate.is_file() for candidate in tombstone.rglob("*")) < (
        before_files
    )

    with workspace_mutation_lock(source):
        recovered = reap_external_recovery_tombstones(corpus, source)

    assert recovered == ("s0",)
    assert not tombstone.exists()


@pytest.mark.parametrize("reservation_kind", ["file", "directory"])
def test_external_reaper_refuses_unproven_empty_cleanup_reservation(
    reservation_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private-looking empty node is not proof it was cleanup-created."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    request = state["request"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    assert isinstance(request, ExternalRecoveryRequest)
    tombstone, _audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )
    reservation = tombstone / (".crucible-cleanup-" + "a" * 64)
    if reservation_kind == "file":
        reservation.touch(mode=0o600)
    else:
        reservation.mkdir(mode=0o700)

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="inventory|unauthorized|changed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert reservation.exists()


@pytest.mark.parametrize("reservation_kind", ["file", "directory"])
def test_external_reaper_retains_unbound_crash_cleanup_reservation(
    reservation_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-rename crash residue remains visible without durable identity."""

    helpers = _journal_recovery_helpers()
    state = helpers._setup(tmp_path)
    source = state["source"]
    corpus = state["corpus"]
    assert isinstance(source, Path)
    assert isinstance(corpus, Corpus)
    tombstone, _audit = helpers._leave_completed_recovery_tombstone(
        state,
        monkeypatch,
    )

    original_rename = os.rename
    crashed = [False]
    crash_residue: list[Path] = []

    def crash_after_reservation(
        source_name,
        destination_name,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        if (
            not crashed[0]
            and isinstance(destination_name, str)
            and destination_name.startswith(".crucible-cleanup-")
            and src_dir_fd is not None
        ):
            metadata = os.stat(
                source_name,
                dir_fd=src_dir_fd,
                follow_symlinks=False,
            )
            expected_kind = (
                stat.S_ISREG(metadata.st_mode)
                if reservation_kind == "file"
                else stat.S_ISDIR(metadata.st_mode)
            )
            if expected_kind:
                crashed[0] = True
                crash_residue.append(
                    Path(f"/proc/self/fd/{dst_dir_fd}").resolve()
                    / destination_name
                )
                raise RuntimeError("simulated death after cleanup reservation")
        return original_rename(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", crash_after_reservation)
    with workspace_mutation_lock(source):
        with pytest.raises(
            RuntimeError,
            match="death after cleanup reservation",
        ):
            reap_external_recovery_tombstones(corpus, source)
    monkeypatch.setattr(os, "rename", original_rename)

    assert crashed == [True]
    assert len(crash_residue) == 1
    reservation = crash_residue[0]
    assert reservation.is_file() if reservation_kind == "file" else reservation.is_dir()
    residue_metadata = reservation.lstat()
    residue_identity = (residue_metadata.st_dev, residue_metadata.st_ino)

    with workspace_mutation_lock(source):
        with pytest.raises(
            MutationJournalError,
            match="inventory|unauthorized|changed",
        ):
            reap_external_recovery_tombstones(corpus, source)

    assert tombstone.is_dir()
    assert any(
        (metadata.st_dev, metadata.st_ino) == residue_identity
        for candidate in tombstone.rglob("*")
        if (metadata := candidate.lstat())
    )


def test_unbindable_writer_backend_retains_provider_epoch_and_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend construction without close proof must poison pool settlement."""

    storage = _MemoryEpochStorage()
    callbacks: list[str] = []
    constructed: list[object] = []

    class _RaisingCloseDescriptor:
        name = "raising-close-descriptor"

        @property
        def close(self):
            raise RuntimeError("close authority descriptor failed")

        def attach_toolset(self, _toolset, *, context_budget=None) -> None:
            del context_budget

    class _WriteToolSet:
        read_only_names = frozenset({"read_source"})
        write_names = frozenset({"write_source"})

    def resolve(_role, **_kwargs):
        backend = _RaisingCloseDescriptor()
        constructed.append(backend)
        return backend

    monkeypatch.setattr(_backends, "resolve_for_role", resolve)
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    @contextmanager
    def pool_context():
        try:
            yield pool
        finally:
            pool.close()

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        with ProviderSettlementExitStack() as owner:
            owner.callback(callbacks.append, "journal-restore")
            owner.arm_provider_generation((storage,))
            active_pool = owner.enter_provider_context(pool_context())
            active_pool.get(
                "implementer",
                toolset=_WriteToolSet(),
                capability="workspace_write",
            )

    assert constructed
    assert captured.value.owner is owner
    assert callbacks == []
    assert inspect_writer_epoch_family((storage,)).state == "armed"


def test_raising_attach_descriptor_retains_provider_epoch_and_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned backend is cleanup-owned before any later descriptor read."""

    storage = _MemoryEpochStorage()
    callbacks: list[str] = []
    constructed: list[object] = []

    class _RaisingAttachDescriptor:
        name = "raising-attach-descriptor"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("provider descendant remains live")

        @property
        def attach_toolset(self):
            raise RuntimeError("attach authority descriptor failed")

    class _WriteToolSet:
        read_only_names = frozenset({"read_source"})
        write_names = frozenset({"write_source"})

    def resolve(_role, **_kwargs):
        backend = _RaisingAttachDescriptor()
        constructed.append(backend)
        return backend

    monkeypatch.setattr(_backends, "resolve_for_role", resolve)
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    @contextmanager
    def pool_context():
        try:
            yield pool
        finally:
            pool.close()

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        with ProviderSettlementExitStack() as owner:
            owner.callback(callbacks.append, "journal-restore")
            owner.arm_provider_generation((storage,))
            active_pool = owner.enter_provider_context(pool_context())
            active_pool.get(
                "implementer",
                toolset=_WriteToolSet(),
                capability="workspace_write",
            )

    assert constructed
    assert constructed[0].close_calls == 1
    assert captured.value.owner is owner
    assert callbacks == []
    assert inspect_writer_epoch_family((storage,)).state == "armed"


@pytest.mark.parametrize(
    ("constructor_name", "write_names"),
    [
        ("new_exclusive", frozenset()),
        ("new_exclusive_writer", frozenset({"write_source"})),
    ],
)
def test_exclusive_setup_retains_backend_until_positive_transfer(
    constructor_name: str,
    write_names: frozenset[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusive ownership transfers only after every setup access succeeds."""

    class _RaisingAttachDescriptor:
        name = "exclusive-raising-attach-descriptor"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("exclusive provider descendant remains live")

        @property
        def attach_toolset(self):
            raise RuntimeError("exclusive attach descriptor failed")

    class _ClassifiedToolSet:
        read_only_names = frozenset({"read_source"})

        def __init__(self, writes: frozenset[str]) -> None:
            self.write_names = writes

    backend = _RaisingAttachDescriptor()
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
    )

    constructor = getattr(pool, constructor_name)
    with pytest.raises(_backends.BackendCleanupError) as captured:
        constructor(
            "implementer",
            toolset=_ClassifiedToolSet(write_names),
            log_dir=tmp_path,
            cwd=str(tmp_path),
        )

    assert backend.close_calls == 1
    assert pool._provisional_backends[id(backend)] is backend  # noqa: SLF001
    captured.value.retry_cleanup()
    assert backend.close_calls == 2
    assert pool._provisional_backends == {}  # noqa: SLF001
    assert pool._cleanup_authorities == {}  # noqa: SLF001
    assert pool._cleanup_errors == {}  # noqa: SLF001


@pytest.mark.parametrize(
    ("constructor_name", "write_names"),
    [
        ("get", frozenset({"write_source"})),
        ("new_exclusive", frozenset()),
        ("new_exclusive_writer", frozenset({"write_source"})),
    ],
)
@pytest.mark.parametrize("failure_stage", ["attach_descriptor", "ledger_wrap"])
def test_backend_setup_owner_blocks_settlement_until_explicit_retry(
    constructor_name: str,
    write_names: frozenset[str],
    failure_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every post-resolution setup frontier retains the same retry owner."""

    storage = _MemoryEpochStorage()
    callbacks: list[str] = []

    class _SetupBackend:
        name = "post-resolution-setup-adversary"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("provider descendant remains live")

        @property
        def attach_toolset(self):
            if failure_stage == "attach_descriptor":
                raise RuntimeError("attach descriptor failed")
            return self._attach

        def _attach(self, _toolset, *, context_budget=None) -> None:
            del context_budget

    class _ClassifiedToolSet:
        read_only_names = frozenset({"read_source"})

        def __init__(self, writes: frozenset[str]) -> None:
            self.write_names = writes

    class _DevelopmentLedger:
        def require_role(self, _role: str) -> None:
            return None

        def wrap(self, _role: str, backend: object) -> object:
            if failure_stage == "ledger_wrap":
                raise RuntimeError("development ledger wrap failed")
            return backend

    backend = _SetupBackend()
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda _role, **_kwargs: backend,
    )
    pool = _backends._BackendPool(
        object(),
        log_dir=tmp_path,
        cwd=str(tmp_path),
        development_budget_ledger=_DevelopmentLedger(),  # type: ignore[arg-type]
    )

    @contextmanager
    def pool_context():
        try:
            yield pool
        finally:
            pool.close()

    with pytest.raises(ProviderSettlementCleanupError) as captured:
        with ProviderSettlementExitStack() as owner:
            owner.callback(callbacks.append, "journal-restore")
            owner.arm_provider_generation((storage,))
            active_pool = owner.enter_provider_context(pool_context())
            toolset = _ClassifiedToolSet(write_names)
            if constructor_name == "get":
                active_pool.get(
                    "implementer",
                    toolset=toolset,
                    capability="workspace_write",
                )
            else:
                getattr(active_pool, constructor_name)(
                    "implementer",
                    toolset=toolset,
                    log_dir=tmp_path,
                    cwd=str(tmp_path),
                )

    assert backend.close_calls == 1
    assert pool._provisional_backends[id(backend)] is backend  # noqa: SLF001
    assert callbacks == []
    assert inspect_writer_epoch_family((storage,)).state == "armed"

    captured.value.retry_cleanup()

    assert backend.close_calls == 2
    assert pool._provisional_backends == {}  # noqa: SLF001
    assert callbacks == ["journal-restore"]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert owner.cleanup_complete


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd exploit")
def test_target_cannot_duplicate_controller_settlement_socket(
    tmp_path: Path,
) -> None:
    """A target cannot duplicate the supervisor's controller-bound socket."""

    marker = tmp_path / "late-writer.txt"
    forged_receipt = (
        json.dumps(
            {
                "schema_version": "crucible.process-supervisor.v1",
                "returncode": 0,
                "settled": True,
                "error": None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + chr(10)
    )
    target = textwrap.dedent(
        f"""\
        import ctypes
        import os
        import signal
        import socket
        import time
        from pathlib import Path

        supervisor = os.getppid()
        receipt = {forged_receipt!r}.encode("ascii")
        forged = False
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        pidfd = libc.syscall(434, supervisor, 0)
        if pidfd >= 0:
            try:
                for descriptor_number in range(1024):
                    descriptor = libc.syscall(
                        438,
                        pidfd,
                        descriptor_number,
                        0,
                    )
                    if descriptor < 0:
                        continue
                    candidate = None
                    try:
                        candidate = socket.socket(fileno=descriptor)
                        if candidate.getsockopt(
                            socket.SOL_SOCKET,
                            socket.SO_TYPE,
                        ) != socket.SOCK_SEQPACKET:
                            continue
                        candidate.send(receipt)
                        forged = True
                        break
                    except OSError:
                        pass
                    finally:
                        if candidate is not None:
                            candidate.detach()
                        os.close(descriptor)
            finally:
                os.close(pidfd)

        if not forged:
            os._exit(125)

        child = os.fork()
        if child == 0:
            os.setsid()
            for descriptor in (0, 1, 2):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            null = os.open("/dev/null", os.O_RDWR)
            os.dup2(null, 0)
            os.dup2(null, 1)
            os.dup2(null, 2)
            if null > 2:
                os.close(null)
            time.sleep(0.5)
            Path({str(marker)!r}).write_text("late mutation\\n")
            os._exit(0)

        os.kill(supervisor, signal.SIGKILL)
        time.sleep(10)
        """
    )

    result = run_supervised_process(
        [sys.executable, "-I", "-c", target],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        timeout=5.0,
    )
    assert result.returncode == 125
    deadline = time.monotonic() + 1.5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.025)
    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor contract")
def test_nonzero_supervisor_exit_cannot_publish_settlement(
    tmp_path: Path,
) -> None:
    """Controller rc, not merely a packet, gates settlement authority."""

    marker = tmp_path / "detached-after-supervisor-death.txt"
    target = textwrap.dedent(
        f"""\
        import os
        import signal
        import time
        from pathlib import Path

        supervisor = os.getppid()
        child = os.fork()
        if child == 0:
            os.setsid()
            for descriptor in (0, 1, 2):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            null = os.open("/dev/null", os.O_RDWR)
            os.dup2(null, 0)
            os.dup2(null, 1)
            os.dup2(null, 2)
            if null > 2:
                os.close(null)
            time.sleep(0.5)
            Path({str(marker)!r}).write_text("survived supervisor death\\n")
            os._exit(0)
        os.kill(supervisor, signal.SIGKILL)
        time.sleep(10)
        """
    )

    with pytest.raises(
        _RunnerSettlementError,
        match=r"supervisor exited without trusted settlement \(returncode=-9\)",
    ):
        run_supervised_process(
            [sys.executable, "-I", "-c", target],
            cwd=tmp_path,
            env={"PATH": os.environ["PATH"]},
            capture_output=True,
            timeout=5.0,
        )

    deadline = time.monotonic() + 1.5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.025)
    assert marker.read_text(encoding="utf-8") == (
        "survived supervisor death\n"
    )
