# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adversarial registry-directory authority regressions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .test_workflows_tier_mutation_transaction import _setup


def _replace_registry_with_external_symlink(
    registry: Path,
    parked: Path,
    external: Path,
) -> None:
    registry.rename(parked)
    registry.symlink_to(external, target_is_directory=True)


def _restore_registry_path(registry: Path, parked: Path) -> None:
    registry.unlink()
    parked.rename(registry)


def _replace_registry_with_external_directory(
    registry: Path,
    parked: Path,
    external: Path,
) -> None:
    registry.rename(parked)
    external.rename(registry)


def _restore_registry_directory(
    registry: Path,
    parked: Path,
    external: Path,
) -> None:
    registry.rename(external)
    parked.rename(registry)


def test_registry_clear_holds_directory_across_external_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus = _setup(tmp_path)
    external = tmp_path / "outside-registry"
    external.mkdir()
    victim = external / ".tmp_victim"
    victim.write_text("outside must survive\n", encoding="utf-8")
    parked = tmp_path / "parked-registry"

    with journals.workspace_mutation_lock(repo):
        with journals._bound_workspace_registry_directory(repo) as (
            registry,
            _registry_fd,
        ):
            pass
        original_open = journals._open_authority_directory_at
        swapped = False

        def swap_after_registry_open(parent_fd, name, *, label):
            nonlocal swapped
            descriptor, metadata = original_open(parent_fd, name, label=label)
            if name == journals._REGISTRY_DIRECTORY and not swapped:
                swapped = True
                _replace_registry_with_external_symlink(
                    registry,
                    parked,
                    external,
                )
            return descriptor, metadata

        monkeypatch.setattr(
            journals,
            "_open_authority_directory_at",
            swap_after_registry_open,
        )
        try:
            with pytest.raises(
                journals.MutationJournalError,
                match="registry changed while held",
            ):
                journals._assert_registry_clear(repo)
            assert swapped
            assert victim.read_text(encoding="utf-8") == "outside must survive\n"
        finally:
            monkeypatch.setattr(
                journals,
                "_open_authority_directory_at",
                original_open,
            )
            _restore_registry_path(registry, parked)


def test_initial_registry_creation_never_publishes_a_racing_external_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real-directory winner at the final name is preserved and rejected."""

    from crucible.workflows import _mutation_journal as journals

    repo, _corpus = _setup(tmp_path)
    external = tmp_path / "outside-initial-registry"
    external.mkdir(mode=0o750)
    victim = external / "victim"
    victim.write_bytes(b"outside registry authority\n")
    external_before = external.lstat()
    victim_before = victim.lstat()
    original_rename = journals._rename_bound_noreplace
    published_path: Path | None = None

    def collide_before_noreplace(
        directory_fd: int,
        source_name: str,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal published_path
        if (
            published_path is None
            and destination_name == journals._REGISTRY_DIRECTORY
        ):
            published_path = (
                Path(journals._authority_descriptor_cwd(directory_fd))
                / destination_name
            ).resolve()
            external.rename(published_path)
        original_rename(
            directory_fd,
            source_name,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(
        journals,
        "_rename_bound_noreplace",
        collide_before_noreplace,
    )
    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="appeared before publication",
        ):
            with journals._bound_workspace_registry_directory(repo):
                pytest.fail("a colliding registry must never be yielded")

    assert published_path is not None
    published_after = published_path.lstat()
    published_victim = published_path / "victim"
    victim_after = published_victim.lstat()
    assert (
        published_after.st_dev,
        published_after.st_ino,
        published_after.st_mode,
    ) == (
        external_before.st_dev,
        external_before.st_ino,
        external_before.st_mode,
    )
    assert (
        victim_after.st_dev,
        victim_after.st_ino,
        victim_after.st_mode,
        victim_after.st_size,
    ) == (
        victim_before.st_dev,
        victim_before.st_ino,
        victim_before.st_mode,
        victim_before.st_size,
    )
    assert published_victim.read_bytes() == b"outside registry authority\n"
    assert not list(published_path.parent.glob(".crucible-restore-dir-*"))


def test_begin_retains_one_registry_inode_across_creation_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real-directory swap after preflight is rejected before capture."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    registry = journals._registry_directory(repo)
    parked = tmp_path / "parked-begin-registry"
    external = tmp_path / "outside-begin-registry"
    external.mkdir()
    victim = external / "victim"
    victim.write_bytes(b"outside begin authority\n")
    original_clear = journals._assert_registry_clear
    swapped = False

    def clear_then_swap(*args, **kwargs):
        nonlocal swapped
        result = original_clear(*args, **kwargs)
        if kwargs.get("_registry_binding") is not None and not swapped:
            swapped = True
            _replace_registry_with_external_directory(
                registry,
                parked,
                external,
            )
        return result

    monkeypatch.setattr(journals, "_assert_registry_clear", clear_then_swap)
    try:
        with pytest.raises(
            journals.MutationJournalError,
            match="registry changed while held",
        ):
            journals.WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=(
                    corpus.spec_dir / "runs" / "registry-begin-swap" / "mutations"
                ),
                slice_id="registry-begin-swap",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )
        assert swapped
        assert (registry / "victim").read_bytes() == b"outside begin authority\n"
        assert list(registry.iterdir()) == [registry / "victim"]
    finally:
        monkeypatch.setattr(journals, "_assert_registry_clear", original_clear)
        if swapped:
            _restore_registry_directory(registry, parked, external)


def test_begin_retains_retry_owner_when_registry_changes_on_context_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-publication swap cannot orphan a live marker and journal."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    registry = journals._registry_directory(repo)
    parked = tmp_path / "parked-post-publication-registry"
    external = tmp_path / "outside-post-publication-registry"
    external.mkdir()
    victim = external / "victim"
    victim.write_bytes(b"outside post-publication authority\n")
    original_write = journals._write_bound_registry_marker_at
    swapped = False

    def publish_then_swap(*args, **kwargs):
        nonlocal swapped
        result = original_write(*args, **kwargs)
        if not swapped:
            swapped = True
            _replace_registry_with_external_directory(
                registry,
                parked,
                external,
            )
        return result

    monkeypatch.setattr(
        journals,
        "_write_bound_registry_marker_at",
        publish_then_swap,
    )
    retained = None
    try:
        with pytest.raises(
            journals.MutationJournalCleanupError,
            match="registry changed while held",
        ) as caught:
            journals.WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=(
                    corpus.spec_dir
                    / "runs"
                    / "registry-post-publication-swap"
                    / "mutations"
                ),
                slice_id="registry-post-publication-swap",
                scope=("a.py",),
                sequence=1,
                spec=corpus.latest_spec(),
            )
        retained = caught.value.owner
        assert swapped
        assert retained.active
        assert retained.journal_dir.is_dir()
        assert (registry / "victim").read_bytes() == (
            b"outside post-publication authority\n"
        )
        assert not list(registry.glob("*.json"))
        assert (
            parked / f"{retained.transaction_id}.json"
        ).is_file()
    finally:
        monkeypatch.setattr(
            journals,
            "_write_bound_registry_marker_at",
            original_write,
        )
        if swapped:
            _restore_registry_directory(registry, parked, external)
        if retained is not None:
            assert retained.restore()


def test_registry_topology_restore_never_deletes_external_swap(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "registry-swap" / "mutations"
        ),
        slice_id="registry-swap",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    registry = journals._registry_directory(repo)
    parked = tmp_path / "parked-live-registry"
    external = tmp_path / "outside-live-registry"
    external.mkdir()
    victim = external / "victim"
    victim.write_text("outside must survive\n", encoding="utf-8")
    _replace_registry_with_external_symlink(registry, parked, external)
    try:
        with journals.workspace_mutation_lock(repo):
            with pytest.raises(journals.MutationJournalError):
                journal._restore_registry_topology()
        assert victim.read_text(encoding="utf-8") == "outside must survive\n"
    finally:
        _restore_registry_path(registry, parked)
        assert journal.restore()


def test_restarted_journal_rejects_real_registry_replacement(
    tmp_path: Path,
) -> None:
    """Persisted registry identity prevents restart cleanup of external bytes."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "registry-restart-swap" / "mutations"
        ),
        slice_id="registry-restart-swap",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    journals._unregister_live(journal.transaction_id)
    restarted = next(
        candidate
        for candidate in journals.discover_mutation_journals(
            corpus,
            workspace=repo,
        )
        if candidate.transaction_id == journal.transaction_id
    )

    registry = journals._registry_directory(repo)
    parked = tmp_path / "parked-restart-registry"
    external = tmp_path / "outside-restart-registry"
    external.mkdir()
    victim = external / "victim"
    victim.write_bytes(b"outside restart authority\n")
    _replace_registry_with_external_directory(registry, parked, external)
    try:
        with journals.workspace_mutation_lock(repo):
            with pytest.raises(
                journals.MutationJournalError,
                match="changed from its manifest identity",
            ):
                restarted._restore_registry_topology()
        assert (registry / "victim").read_bytes() == b"outside restart authority\n"
        assert list(registry.iterdir()) == [registry / "victim"]
    finally:
        _restore_registry_directory(registry, parked, external)
        assert journal.restore()


def test_registry_clear_retries_private_temp_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus = _setup(tmp_path)
    registry = journals._registry_directory(repo)
    temporary = registry / ".tmp_candidate_123_12345678"
    temporary.write_bytes(b"unpublished marker bytes\n")
    original_unlink = journals.os.unlink
    interrupted = False

    def interrupt_after_quarantine(path, *args, **kwargs):
        nonlocal interrupted
        if (
            not interrupted
            and isinstance(path, str)
            and path.startswith(".crucible-cleanup-")
        ):
            interrupted = True
            raise OSError("injected process-death window")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(journals.os, "unlink", interrupt_after_quarantine)
    with pytest.raises(
        journals.MutationJournalError,
        match="process-death window",
    ):
        journals._assert_registry_clear(repo)
    assert interrupted
    assert not temporary.exists()
    assert len(list(registry.glob(".crucible-cleanup-*"))) == 1

    monkeypatch.setattr(journals.os, "unlink", original_unlink)
    journals._assert_registry_clear(repo)
    assert list(registry.iterdir()) == []


def test_registry_marker_publication_never_replaces_racing_final_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final-name collision preserves the hostile inode and marker bytes."""

    from crucible.workflows import _mutation_journal as journals

    repo, _corpus = _setup(tmp_path)
    transaction_id = "a" * 32
    marker_name = f"{transaction_id}.json"
    intended = b'{"transaction_id":"' + transaction_id.encode() + b'"}\n'
    hostile = b"hostile final-name collision\n"
    original_link = journals._link_bound_unnamed_noreplace
    inserted: tuple[int, int] | None = None

    def collide_before_noreplace(
        temporary_fd: int,
        parent_fd: int,
        final_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal inserted
        if inserted is None and final_name == marker_name:
            descriptor = os.open(
                final_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                os.write(descriptor, hostile)
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
                inserted = (metadata.st_dev, metadata.st_ino)
            finally:
                os.close(descriptor)
        original_link(
            temporary_fd,
            parent_fd,
            final_name,
            label=label,
        )

    monkeypatch.setattr(
        journals,
        "_link_bound_unnamed_noreplace",
        collide_before_noreplace,
    )
    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="target appeared",
        ):
            journals._write_bound_registry_marker(
                repo,
                transaction_id,
                intended,
                expected_baseline_entries=(),
            )

    registry = journals._registry_directory(repo)
    marker = registry / marker_name
    observed = marker.lstat()
    assert inserted == (observed.st_dev, observed.st_ino)
    assert marker.read_bytes() == hostile
    assert not list(registry.glob(".crucible-publish-*"))


def test_registry_marker_restore_never_replaces_racing_final_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback retains its journal when a hostile marker wins publication."""

    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    journal = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "registry-restore-race" / "mutations"
        ),
        slice_id="registry-restore-race",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    marker = journal._registry_marker()
    assert marker is not None
    marker.unlink()
    hostile = b"hostile restored-marker collision\n"
    original_link = journals._link_bound_unnamed_noreplace
    inserted: tuple[int, int] | None = None

    def collide_before_noreplace(
        temporary_fd: int,
        parent_fd: int,
        final_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal inserted
        if inserted is None and final_name == marker.name:
            descriptor = os.open(
                final_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                os.write(descriptor, hostile)
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
                inserted = (metadata.st_dev, metadata.st_ino)
            finally:
                os.close(descriptor)
        original_link(
            temporary_fd,
            parent_fd,
            final_name,
            label=label,
        )

    monkeypatch.setattr(
        journals,
        "_link_bound_unnamed_noreplace",
        collide_before_noreplace,
    )
    with journals.workspace_mutation_lock(repo):
        with pytest.raises(
            journals.MutationJournalError,
            match="target appeared",
        ):
            journal._restore_registry_marker()

    observed = marker.lstat()
    assert journal.active
    assert inserted == (observed.st_dev, observed.st_ino)
    assert marker.read_bytes() == hostile

    monkeypatch.setattr(
        journals,
        "_link_bound_unnamed_noreplace",
        original_link,
    )
    marker.unlink()
    assert journal.restore()
