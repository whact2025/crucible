# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crash recovery for durable hosted-delete quarantine receipts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess

import pytest

from .test_workflows_tier_mutation_transaction import (
    _setup,
    _simulate_process_death,
)


def _begin_delete_journal(tmp_path: Path):
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "pending-delete" / "mutations"
        ),
        slice_id="pending-delete",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    return journals, repo, corpus, journal


def _create_quarantine(repo: Path, receipt: dict[str, object]) -> Path:
    quarantine = repo / str(receipt["quarantine_relative"])
    quarantine.mkdir(mode=0o700)
    return quarantine


def _bind_to_child(journals, journal, receipt, quarantine, child):
    identity = journals._pending_delete_process_identity(child.pid)
    assert identity is not None
    descriptor = os.open(
        quarantine,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        metadata = os.fstat(descriptor)
        return journal.bind_pending_delete_quarantine(
            receipt,
            quarantine_device=metadata.st_dev,
            quarantine_inode=metadata.st_ino,
            quarantine_mode=metadata.st_mode,
            quarantine_mount_id=journals._pending_delete_mount_id(descriptor),
            child_pid=child.pid,
            child_start_ticks=identity[1],
            boot_id=journals._pending_delete_boot_id(),
        )
    finally:
        os.close(descriptor)


def test_live_owner_resolver_and_reserved_receipt_cas(tmp_path: Path) -> None:
    journals, repo, _corpus, journal = _begin_delete_journal(tmp_path)

    root_descriptor = os.open(
        repo,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    def worker() -> dict[str, object]:
        resolved = journals.resolve_live_mutation_journal(
            journal.journal_dir,
            journal.transaction_id,
        )
        assert resolved is journal
        receipt = journal.arm_pending_delete_quarantine(
            token="1" * 32,
            target_relative="a.py",
            workspace_root_descriptor=root_descriptor,
        )
        assert receipt["state"] == "reserved"
        assert receipt["quarantine_relative"] == (
            ".a.py.crucible-delete-" + "1" * 32 + ".quarantine"
        )
        assert receipt["boot_id"] == journals._pending_delete_boot_id()
        assert len(receipt["parent_chain"]) == 1
        assert receipt["parent_chain"][0][2] == os.stat(repo).st_mode & 0o170000
        assert journal.arm_pending_delete_quarantine(
            token="1" * 32,
            target_relative="a.py",
            workspace_root_descriptor=root_descriptor,
        ) == receipt
        with pytest.raises(
            journals.MutationJournalError,
            match="already armed",
        ):
            journal.arm_pending_delete_quarantine(
                token="8" * 32,
                target_relative="a.py",
                workspace_root_descriptor=root_descriptor,
            )
        stale = dict(receipt)
        stale["token"] = "9" * 32
        with pytest.raises(
            journals.MutationJournalError,
            match="does not match its target and token",
        ):
            journal.clear_pending_delete_quarantine(
                stale,
                workspace_root_descriptor=root_descriptor,
            )
        malformed_chain = dict(receipt)
        malformed_chain["parent_chain"] = []
        with pytest.raises(
            journals.MutationJournalError,
            match="parent chain has a malformed length",
        ):
            journal.clear_pending_delete_quarantine(
                malformed_chain,
                workspace_root_descriptor=root_descriptor,
            )
        journal.clear_pending_delete_quarantine(
            receipt,
            workspace_root_descriptor=root_descriptor,
        )
        journal.clear_pending_delete_quarantine(
            receipt,
            workspace_root_descriptor=root_descriptor,
        )
        return receipt

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result(timeout=10)
    finally:
        os.close(root_descriptor)

    assert journal._manifest["pending_delete_quarantines"] == []


def test_parent_chain_close_effect_never_retries_reused_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    target = repo / "nested" / "parent" / "a.py"
    target.parent.mkdir(parents=True)
    (repo / "a.py").rename(target)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "chain-close" / "mutations"
        ),
        slice_id="chain-close",
        scope=("nested/parent/a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    with journals.workspace_mutation_lock(repo):
        receipt = journal.arm_pending_delete_quarantine(
            token="b" * 32,
            target_relative="nested/parent/a.py",
        )
    parsed = journals._parse_pending_delete_quarantine(
        receipt,
        manifest=journal._manifest,
    )
    sentinel = tmp_path / "journal-close-sentinel"
    sentinel.mkdir()
    real_dup = journals.os.dup
    real_close = journals.os.close
    duplicated_root: int | None = None
    reused_descriptor: int | None = None
    close_attempts = 0

    def capture_dup(descriptor: int) -> int:
        nonlocal duplicated_root
        opened = real_dup(descriptor)
        if duplicated_root is None:
            duplicated_root = opened
        return opened

    def close_then_error(descriptor: int) -> None:
        nonlocal close_attempts, reused_descriptor
        if descriptor == duplicated_root:
            close_attempts += 1
            if close_attempts > 1:
                pytest.fail("detached journal parent fd was retried")
            real_close(descriptor)
            reused_descriptor = os.open(
                sentinel,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            assert reused_descriptor == descriptor
            raise OSError("injected after journal parent close")
        real_close(descriptor)

    with journals.workspace_mutation_lock(repo) as workspace_authority:
        monkeypatch.setattr(journals.os, "dup", capture_dup)
        monkeypatch.setattr(journals.os, "close", close_then_error)
        try:
            with pytest.raises(
                journals.MutationJournalError,
                match="descriptor close is indeterminate",
            ):
                with journals._open_pending_delete_parent(
                    repo,
                    parsed,
                    root_descriptor=workspace_authority.root_descriptor,
                ):
                    pytest.fail(
                        "parent context yielded after indeterminate close"
                    )
        finally:
            monkeypatch.setattr(journals.os, "dup", real_dup)
            monkeypatch.setattr(journals.os, "close", real_close)

    assert close_attempts == 1
    assert reused_descriptor == duplicated_root
    assert reused_descriptor is not None
    os.fstat(reused_descriptor)
    assert journal._manifest["pending_delete_quarantines"] == [receipt]

    real_close(reused_descriptor)
    with journals.workspace_mutation_lock(repo):
        journal.clear_pending_delete_quarantine(receipt)


def test_recovery_clears_reserved_empty_quarantine(tmp_path: Path) -> None:
    journals, repo, corpus, journal = _begin_delete_journal(tmp_path)

    with journals.workspace_mutation_lock(repo):
        receipt = journal.arm_pending_delete_quarantine(
            token="2" * 32,
            target_relative="a.py",
        )
        quarantine = _create_quarantine(repo, receipt)
    _simulate_process_death(journal)

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.failures == ()
    assert report.recovered == ("pending-delete",)
    assert not quarantine.exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "# baseline a.py\n"


def test_recovery_restores_bound_payload_before_journal_rollback(
    tmp_path: Path,
) -> None:
    journals, repo, corpus, journal = _begin_delete_journal(tmp_path)
    child = subprocess.Popen(["/bin/sh", "-c", "exec sleep 60"])
    try:
        with journals.workspace_mutation_lock(repo):
            receipt = journal.arm_pending_delete_quarantine(
                token="3" * 32,
                target_relative="a.py",
            )
            quarantine = _create_quarantine(repo, receipt)
            reserved = receipt
            receipt = _bind_to_child(
                journals,
                journal,
                receipt,
                quarantine,
                child,
            )
            assert journal.bind_pending_delete_quarantine(
                reserved,
                quarantine_device=int(receipt["quarantine_device"]),
                quarantine_inode=int(receipt["quarantine_inode"]),
                quarantine_mode=int(receipt["quarantine_mode"]),
                quarantine_mount_id=int(receipt["quarantine_mount_id"]),
                child_pid=int(receipt["child_pid"]),
                child_start_ticks=int(receipt["child_start_ticks"]),
                boot_id=str(receipt["boot_id"]),
            ) == receipt
        child.terminate()
        child.wait(timeout=10)
        with journals.workspace_mutation_lock(repo):
            assert journal.bind_pending_delete_quarantine(
                reserved,
                quarantine_device=int(receipt["quarantine_device"]),
                quarantine_inode=int(receipt["quarantine_inode"]),
                quarantine_mode=int(receipt["quarantine_mode"]),
                quarantine_mount_id=int(receipt["quarantine_mount_id"]),
                child_pid=int(receipt["child_pid"]),
                child_start_ticks=int(receipt["child_start_ticks"]),
                boot_id=str(receipt["boot_id"]),
            ) == receipt
            os.rename(repo / "a.py", quarantine / "payload")
        _simulate_process_death(journal)

        report = journals.recover_mutation_journals(corpus, repo)

        assert report.failures == ()
        assert report.recovered == ("pending-delete",)
        assert not quarantine.exists()
        assert (repo / "a.py").read_text(encoding="utf-8") == (
            "# baseline a.py\n"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_recovery_preserves_public_collision_and_quarantined_payload(
    tmp_path: Path,
) -> None:
    journals, repo, corpus, journal = _begin_delete_journal(tmp_path)
    child = subprocess.Popen(["/bin/sh", "-c", "exec sleep 60"])
    try:
        with journals.workspace_mutation_lock(repo):
            receipt = journal.arm_pending_delete_quarantine(
                token="4" * 32,
                target_relative="a.py",
            )
            quarantine = _create_quarantine(repo, receipt)
            _bind_to_child(
                journals,
                journal,
                receipt,
                quarantine,
                child,
            )
            os.rename(repo / "a.py", quarantine / "payload")
            (repo / "a.py").write_text("foreign bytes\n", encoding="utf-8")
        child.terminate()
        child.wait(timeout=10)
        _simulate_process_death(journal)

        report = journals.recover_mutation_journals(corpus, repo)

        assert report.recovered == ()
        assert report.failures
        assert (repo / "a.py").read_text(encoding="utf-8") == "foreign bytes\n"
        assert (quarantine / "payload").read_text(encoding="utf-8") == (
            "# baseline a.py\n"
        )
        manifest = journal.journal_dir / "manifest.json"
        assert "pending_delete_quarantines" in manifest.read_text(
            encoding="utf-8"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_recovery_waits_for_exact_bound_child_exit(tmp_path: Path) -> None:
    journals, repo, corpus, journal = _begin_delete_journal(tmp_path)
    child = subprocess.Popen(["/bin/sh", "-c", "exec sleep 60"])
    try:
        with journals.workspace_mutation_lock(repo):
            receipt = journal.arm_pending_delete_quarantine(
                token="5" * 32,
                target_relative="a.py",
            )
            quarantine = _create_quarantine(repo, receipt)
            _bind_to_child(
                journals,
                journal,
                receipt,
                quarantine,
                child,
            )
        _simulate_process_death(journal)

        blocked = journals.recover_mutation_journals(corpus, repo)

        assert blocked.recovered == ()
        assert any("child is still live" in failure for failure in blocked.failures)
        assert quarantine.is_dir()
        assert (repo / "a.py").is_file()

        child.terminate()
        child.wait(timeout=10)
        recovered = journals.recover_mutation_journals(corpus, repo)

        assert recovered.failures == ()
        assert recovered.recovered == ("pending-delete",)
        assert not quarantine.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_recovery_is_idempotent_after_payload_was_already_restored(
    tmp_path: Path,
) -> None:
    journals, repo, corpus, journal = _begin_delete_journal(tmp_path)
    child = subprocess.Popen(["/bin/sh", "-c", "exec sleep 60"])
    try:
        with journals.workspace_mutation_lock(repo):
            receipt = journal.arm_pending_delete_quarantine(
                token="6" * 32,
                target_relative="a.py",
            )
            quarantine = _create_quarantine(repo, receipt)
            _bind_to_child(
                journals,
                journal,
                receipt,
                quarantine,
                child,
            )
            os.rename(repo / "a.py", quarantine / "payload")
            # Simulate a restart that completed the no-replace restore but
            # died before removing the now-empty qdir or clearing its receipt.
            os.rename(quarantine / "payload", repo / "a.py")
        child.terminate()
        child.wait(timeout=10)
        _simulate_process_death(journal)

        report = journals.recover_mutation_journals(corpus, repo)

        assert report.failures == ()
        assert report.recovered == ("pending-delete",)
        assert not quarantine.exists()
        assert (repo / "a.py").read_text(encoding="utf-8") == (
            "# baseline a.py\n"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_global_preflight_preserves_earlier_qdir_when_later_is_tampered(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "delete-family" / "mutations",
        slice_id="delete-parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=parent.journal_dir / "nested_mutations",
        slice_id="delete-child",
        scope=("b.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    with journals.workspace_mutation_lock(repo):
        parent_receipt = parent.arm_pending_delete_quarantine(
            token="7" * 32,
            target_relative="a.py",
        )
        child_receipt = child.arm_pending_delete_quarantine(
            token="8" * 32,
            target_relative="b.py",
        )
        parent_qdir = _create_quarantine(repo, parent_receipt)
        child_qdir = _create_quarantine(repo, child_receipt)
        (child_qdir / "foreign").write_text("unowned\n", encoding="utf-8")
    _simulate_process_death(child)
    _simulate_process_death(parent)

    report = journals.recover_mutation_journals(corpus, repo)

    assert report.recovered == ()
    assert report.failures
    assert parent_qdir.is_dir()
    assert child_qdir.is_dir()
    assert (child_qdir / "foreign").read_text(encoding="utf-8") == "unowned\n"


def test_restart_recovery_rejects_renamed_parent_with_fresh_empty_chain(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    target = repo / "nested" / "parent" / "a.py"
    target.parent.mkdir(parents=True)
    (repo / "a.py").rename(target)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "renamed-delete" / "mutations"
        ),
        slice_id="renamed-delete",
        scope=("nested/parent/a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = subprocess.Popen(["/bin/sh", "-c", "exec sleep 60"])
    try:
        with journals.workspace_mutation_lock(repo):
            receipt = journal.arm_pending_delete_quarantine(
                token="9" * 32,
                target_relative="nested/parent/a.py",
            )
            quarantine = _create_quarantine(repo, receipt)
            _bind_to_child(journals, journal, receipt, quarantine, child)
            os.rename(target, quarantine / "payload")
        child.terminate()
        child.wait(timeout=10)

        retained = repo / "retained-nested"
        (repo / "nested").rename(retained)
        (repo / "nested" / "parent").mkdir(parents=True)
        retained_quarantine = retained / "parent" / quarantine.name
        retained_payload = retained_quarantine / "payload"
        _simulate_process_death(journal)

        report = journals.recover_mutation_journals(corpus, repo)

        assert report.recovered == ()
        assert any("parent chain differs" in item for item in report.failures)
        assert not (repo / "nested" / "parent" / "a.py").exists()
        assert retained_payload.read_text(encoding="utf-8") == (
            "# baseline a.py\n"
        )
        assert journal.journal_dir.joinpath("manifest.json").is_file()
        assert "pending_delete_quarantines" in journal.journal_dir.joinpath(
            "manifest.json"
        ).read_text(encoding="utf-8")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_armed_writer_epoch_blocks_pending_delete_recovery_before_mutation(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        arm_writer_epoch_family,
        capture_writer_epoch_owner,
    )

    journals, repo, corpus, journal = _begin_delete_journal(tmp_path)
    child = subprocess.Popen(["/bin/sh", "-c", "exec sleep 60"])
    try:
        with journals.workspace_mutation_lock(repo):
            receipt = journal.arm_pending_delete_quarantine(
                token="a" * 32,
                target_relative="a.py",
            )
            quarantine = _create_quarantine(repo, receipt)
            receipt = _bind_to_child(
                journals,
                journal,
                receipt,
                quarantine,
                child,
            )
            os.rename(repo / "a.py", quarantine / "payload")
            arm_writer_epoch_family(
                (journal,),
                owner=capture_writer_epoch_owner(
                    "test-controller",
                    "pending-delete",
                ),
                boundary=WriterEpochBoundary(
                    "test-controller",
                    "pending-delete",
                    "synchronous_callback",
                    ("controller",),
                    (),
                ),
            )
        child.terminate()
        child.wait(timeout=10)
        _simulate_process_death(journal)

        manifest_before = journal.journal_dir.joinpath(
            "manifest.json"
        ).read_bytes()
        # Exercise the locked implementation directly: the public recovery
        # entry point has an earlier defense-in-depth epoch gate as well.
        with journals.workspace_mutation_lock(repo):
            report = journals._recover_mutation_journals_locked(corpus, repo)

        assert report.recovered == ()
        assert any("writer epoch recovery audit failed" in item for item in report.failures)
        assert not (repo / "a.py").exists()
        assert (quarantine / "payload").read_text(encoding="utf-8") == (
            "# baseline a.py\n"
        )
        assert journal.journal_dir.joinpath("manifest.json").read_bytes() == (
            manifest_before
        )
        assert receipt["state"] == "bound"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
