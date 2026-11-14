# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed coverage for scoped active-parent inode handoff."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import errno
import os
from pathlib import Path
import stat
import subprocess

import pytest

from .test_workflows_tier_mutation_transaction import _setup


def _begin_family(
    tmp_path: Path,
    *,
    parent_scope: tuple[str, ...] = ("a.py",),
    child_scope: tuple[str, ...] = ("b.py",),
):
    from crucible.workflows import _mutation_journal as journals

    tmp_path.mkdir(parents=True, exist_ok=True)
    repo, corpus = _setup(tmp_path)
    run = corpus.spec_dir / "runs" / "active-parent-handoff"
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="handoff-parent",
        scope=parent_scope,
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "child_mutations",
        slice_id="handoff-child",
        scope=child_scope,
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    return repo, corpus, parent, child


def _atomic_replace(target: Path, payload: bytes) -> tuple[int, int]:
    stage = target.with_name(f".{target.name}.handoff-stage")
    mode = stat.S_IMODE(target.stat().st_mode)
    stage.write_bytes(payload)
    stage.chmod(mode)
    os.replace(stage, target)
    current = target.stat()
    return current.st_dev, current.st_ino


def test_no_handoff_restore_preserves_legacy_instrumentation_call_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    _repo, _corpus, parent, child = _begin_family(tmp_path)
    restore_calls: list[tuple[str, bool | None, bool]] = []
    real_restore = journals.WorkspaceMutationJournal.restore

    def observe_restore(
        self,
        *,
        full: bool | None = None,
        _defer_terminal_cleanup: bool = False,
    ) -> bool:
        restore_calls.append((self.slice_id, full, _defer_terminal_cleanup))
        return real_restore(
            self,
            full=full,
            _defer_terminal_cleanup=_defer_terminal_cleanup,
        )

    monkeypatch.setattr(
        journals.WorkspaceMutationJournal,
        "restore",
        observe_restore,
    )

    assert child.restore(full=False)
    assert restore_calls
    assert parent.restore(full=True)


@pytest.mark.parametrize("grant_parent_path", (False, True))
def test_scoped_restore_retains_only_explicit_changed_parent_postimage(
    tmp_path: Path,
    grant_parent_path: bool,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    original = (repo / "a.py").stat()
    replacement_identity = _atomic_replace(
        repo / "a.py",
        b"# accepted parent replacement\n",
    )
    assert replacement_identity != (original.st_dev, original.st_ino)

    handoff = (
        child.capture_active_parent_postimage_handoff(("a.py",))
        if grant_parent_path
        else None
    )
    restored = child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    assert restored is grant_parent_path
    assert (repo / "a.py").read_bytes() == b"# accepted parent replacement\n"
    current = (repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == replacement_identity

    if restored:
        assert parent.restore(full=True)
    else:
        journals._unregister_live(child.transaction_id)
        journals._unregister_live(parent.transaction_id)


def test_scoped_restore_retires_linked_peer_guard_before_handoff_reproof(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    tmp_path.mkdir(parents=True, exist_ok=True)
    repo, corpus = _setup(tmp_path)
    linked_peer = tmp_path / "linked-peer"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "worktree",
            "add",
            "-q",
            "--detach",
            str(linked_peer),
            "HEAD",
        ],
        check=True,
        capture_output=True,
    )
    run = corpus.spec_dir / "runs" / "linked-peer-handoff"
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="linked-peer-handoff-parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "child_mutations",
        slice_id="linked-peer-handoff-child",
        scope=("b.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    nested_ids = {
        nested.transaction_id
        for nested in child._nested_journals_children_first()
    }
    assert nested_ids

    replacement_identity = _atomic_replace(
        repo / "a.py",
        b"# accepted parent replacement with linked peer\n",
    )
    handoff = child.capture_active_parent_postimage_handoff(("a.py",))

    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    current = (repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == replacement_identity
    assert (repo / "a.py").read_bytes() == (
        b"# accepted parent replacement with linked peer\n"
    )
    assert nested_ids.isdisjoint(journals._LIVE_JOURNAL_OBJECTS)
    terminal_proof_ids = set(
        parent._manifest.get("writer_epoch_terminal_proofs", {})
    )
    assert nested_ids.issubset(terminal_proof_ids)
    assert parent.restore(full=True)


@pytest.mark.parametrize(
    "unsafe_case",
    (
        "outside_parent_scope",
        "child_scope_overlap",
        "failed_family",
        "inactive_parent",
        "mode_drift",
        "xattr_drift",
        "hardlink",
        "duplicate_paths",
        "ambiguous_lineage",
    ),
)
def test_active_parent_handoff_rejects_unsafe_authority(
    tmp_path: Path,
    unsafe_case: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    parent_scope = (
        ("b.py",) if unsafe_case == "outside_parent_scope" else ("a.py", "b.py")
    )
    child_scope = ("a.py",) if unsafe_case == "child_scope_overlap" else ("b.py",)
    repo, _corpus, parent, child = _begin_family(
        tmp_path,
        parent_scope=parent_scope,
        child_scope=child_scope,
    )
    payload = b"# untrusted parent replacement\n"
    replacement_identity = _atomic_replace(repo / "a.py", payload)
    if unsafe_case == "mode_drift":
        (repo / "a.py").chmod(0o600)
    elif unsafe_case == "xattr_drift":
        try:
            os.setxattr(repo / "a.py", "user.crucible_probe", b"changed")
        except OSError as exc:
            if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip("filesystem does not support user xattrs")
            raise
    elif unsafe_case == "hardlink":
        os.link(repo / "a.py", repo / "a.py.alias")
    elif unsafe_case == "failed_family":
        journals._mark_live_family_failed(parent.transaction_id)
    elif unsafe_case == "inactive_parent":
        journals._unregister_live(parent.transaction_id)
    elif unsafe_case == "ambiguous_lineage":
        child._manifest["registry_family_parents"][child.transaction_id] = None
        child._write()

    with pytest.raises(journals.MutationJournalError):
        child.capture_active_parent_postimage_handoff(
            ("a.py", "a.py") if unsafe_case == "duplicate_paths" else ("a.py",)
        )
    assert child.active
    assert (repo / "a.py").read_bytes() == payload
    current = (repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == replacement_identity

    journals._unregister_live(child.transaction_id)
    if unsafe_case != "inactive_parent":
        journals._unregister_live(parent.transaction_id)


@pytest.mark.parametrize("replacement", ("deleted", "symlink"))
def test_active_parent_handoff_retains_reviewed_nonregular_postimage(
    tmp_path: Path,
    replacement: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    (repo / "a.py").unlink()
    if replacement == "symlink":
        (repo / "a.py").symlink_to("b.py")

    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    if replacement == "deleted":
        assert not (repo / "a.py").exists()
    else:
        assert (repo / "a.py").is_symlink()

    assert parent.restore(full=True)


def test_active_parent_handoff_retains_byte_identical_inode_replacement(
    tmp_path: Path,
) -> None:
    repo, _corpus, parent, child = _begin_family(tmp_path)
    baseline = (repo / "a.py").read_bytes()
    baseline_identity = (repo / "a.py").stat()
    replacement_identity = _atomic_replace(repo / "a.py", baseline)
    assert replacement_identity != (
        baseline_identity.st_dev,
        baseline_identity.st_ino,
    )

    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    current = (repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == replacement_identity
    assert (repo / "a.py").read_bytes() == baseline
    assert parent.restore(full=True)


def test_active_parent_handoff_retains_reviewed_executable_bit(
    tmp_path: Path,
) -> None:
    repo, _corpus, parent, child = _begin_family(tmp_path)
    baseline_mode = stat.S_IMODE((repo / "a.py").stat().st_mode)
    accepted_mode = baseline_mode | 0o111
    (repo / "a.py").chmod(accepted_mode)

    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    assert stat.S_IMODE((repo / "a.py").stat().st_mode) == accepted_mode
    assert parent.restore(full=True)


def test_active_parent_handoff_retains_reviewed_new_file(
    tmp_path: Path,
) -> None:
    repo, _corpus, parent, child = _begin_family(
        tmp_path,
        parent_scope=("new.py",),
    )
    (repo / "new.py").write_bytes(b"# reviewed new file\n")
    (repo / "new.py").chmod(0o644)
    new_identity = (repo / "new.py").stat()

    handoff = child.capture_active_parent_postimage_handoff(("new.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    current = (repo / "new.py").stat()
    assert (current.st_dev, current.st_ino) == (
        new_identity.st_dev,
        new_identity.st_ino,
    )
    assert (repo / "new.py").read_bytes() == b"# reviewed new file\n"
    assert parent.restore(full=True)


def test_active_parent_handoff_retains_reviewed_rename(
    tmp_path: Path,
) -> None:
    repo, _corpus, parent, child = _begin_family(
        tmp_path,
        parent_scope=("a.py", "renamed.py"),
    )
    os.replace(repo / "a.py", repo / "renamed.py")
    renamed_identity = (repo / "renamed.py").stat()

    handoff = child.capture_active_parent_postimage_handoff(
        ("a.py", "renamed.py")
    )
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    assert not (repo / "a.py").exists()
    current = (repo / "renamed.py").stat()
    assert (current.st_dev, current.st_ino) == (
        renamed_identity.st_dev,
        renamed_identity.st_ino,
    )
    assert parent.restore(full=True)


@pytest.mark.parametrize("unsafe_new_path", ("directory", "mode", "xattr"))
def test_active_parent_handoff_rejects_unreviewed_new_path_state(
    tmp_path: Path,
    unsafe_new_path: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(
        tmp_path,
        parent_scope=("new.py",),
    )
    if unsafe_new_path == "directory":
        (repo / "new.py").mkdir()
    else:
        (repo / "new.py").write_bytes(b"# unreviewed metadata\n")
        if unsafe_new_path == "mode":
            (repo / "new.py").chmod(0o600)
        else:
            try:
                os.setxattr(
                    repo / "new.py",
                    "user.crucible_probe",
                    b"changed",
                )
            except OSError as exc:
                if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                    pytest.skip("filesystem does not support user xattrs")
                raise

    with pytest.raises(journals.MutationJournalError):
        child.capture_active_parent_postimage_handoff(("new.py",))
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_rejects_parent_from_another_checkout(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    parent_repo, parent_corpus = _setup(first)
    child_repo, child_corpus = _setup(second)
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=parent_repo,
        journal_root=(parent_corpus.spec_dir / "runs" / "wrong-cwd" / "mutations"),
        slice_id="wrong-cwd-parent",
        scope=("a.py",),
        sequence=1,
        spec=parent_corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=child_repo,
        journal_root=(child_corpus.spec_dir / "runs" / "wrong-cwd" / "mutations"),
        slice_id="wrong-cwd-child",
        scope=("b.py",),
        sequence=2,
        spec=child_corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    replacement_identity = _atomic_replace(
        child_repo / "a.py",
        b"# wrong-checkout replacement\n",
    )

    with pytest.raises(journals.MutationJournalError):
        child.capture_active_parent_postimage_handoff(("a.py",))
    current = (child_repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == replacement_identity

    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


@pytest.mark.parametrize("substitution_style", ("atomic", "in_place"))
def test_active_parent_handoff_rejects_post_proof_substitution(
    tmp_path: Path,
    substitution_style: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    _atomic_replace(repo / "a.py", b"# reviewed and gated postimage\n")
    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    if substitution_style == "atomic":
        substituted_identity = _atomic_replace(
            repo / "a.py",
            b"# substituted after proof capture\n",
        )
    else:
        (repo / "a.py").write_bytes(b"# substituted after proof capture\n")
        substituted = (repo / "a.py").stat()
        substituted_identity = substituted.st_dev, substituted.st_ino

    assert not child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
        _defer_terminal_cleanup=True,
    )
    current = (repo / "a.py").stat()
    assert (current.st_dev, current.st_ino) == substituted_identity
    assert (repo / "a.py").read_bytes() == b"# substituted after proof capture\n"

    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_retains_explicit_missing_to_missing(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    (repo / "a.py").unlink()
    run = corpus.spec_dir / "runs" / "missing-parent-handoff"
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="missing-parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "child_mutations",
        slice_id="missing-child",
        scope=("b.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )

    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    assert not (repo / "a.py").exists()
    child.reprove_consumed_active_parent_postimage_handoff(handoff)
    assert parent.restore(full=True)


def test_active_parent_handoff_retains_nested_path_under_captured_parent(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    (repo / "nested").mkdir()
    run = corpus.spec_dir / "runs" / "nested-parent-handoff"
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="nested-parent",
        scope=("nested/new.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "child_mutations",
        slice_id="nested-child",
        scope=("b.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    (repo / "nested" / "new.py").write_bytes(b"# nested candidate\n")
    (repo / "nested" / "new.py").chmod(0o644)

    handoff = child.capture_active_parent_postimage_handoff(("nested/new.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    assert (repo / "nested" / "new.py").read_bytes() == b"# nested candidate\n"
    assert parent.restore(full=True)


@pytest.mark.parametrize("topology_change", ("created_parent", "removed_parent"))
def test_active_parent_handoff_rejects_changed_nested_parent_topology(
    tmp_path: Path,
    topology_change: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    if topology_change == "removed_parent":
        (repo / "nested").mkdir()
    run = corpus.spec_dir / "runs" / "changed-nested-parent"
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="changed-nested-parent",
        scope=("nested/new.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "child_mutations",
        slice_id="changed-nested-child",
        scope=("b.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    if topology_change == "created_parent":
        (repo / "nested").mkdir()
        (repo / "nested" / "new.py").write_bytes(b"# candidate\n")
    else:
        (repo / "nested").rmdir()

    with pytest.raises(
        journals.MutationJournalError,
        match="directory topology changed|directory that did not exist",
    ):
        child.capture_active_parent_postimage_handoff(("nested/new.py",))
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


@pytest.mark.parametrize("mode_change", ("partial_exec", "nonexec_drift"))
def test_active_parent_handoff_rejects_noncanonical_executable_transition(
    tmp_path: Path,
    mode_change: str,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    baseline_mode = stat.S_IMODE((repo / "a.py").stat().st_mode)
    if mode_change == "partial_exec":
        changed_mode = (baseline_mode & ~0o111) | 0o011
    else:
        changed_mode = (baseline_mode ^ 0o020) | 0o111
    (repo / "a.py").chmod(changed_mode)

    with pytest.raises(
        journals.MutationJournalError,
        match="permission drift",
    ):
        child.capture_active_parent_postimage_handoff(("a.py",))
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_rejects_partial_rename_grant(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(
        tmp_path,
        parent_scope=("a.py", "renamed.py"),
    )
    os.replace(repo / "a.py", repo / "renamed.py")

    with pytest.raises(
        journals.MutationJournalError,
        match="explicit source and destination grants",
    ):
        child.capture_active_parent_postimage_handoff(("renamed.py",))
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_is_one_shot_and_inactive_replay_fails(
    tmp_path: Path,
) -> None:
    repo, _corpus, parent, child = _begin_family(tmp_path)
    _atomic_replace(repo / "a.py", b"# accepted once\n")
    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(
            lambda _index: child.restore(
                full=False,
                active_parent_postimage_handoff=handoff,
            ),
            range(2),
        ))
    assert sorted(results) == [False, True]
    assert not child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    assert parent.restore(full=True)


def test_active_parent_handoff_rejects_full_restore_with_capability(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    _atomic_replace(repo / "a.py", b"# accepted proof\n")
    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    assert not child.restore(
        full=True,
        active_parent_postimage_handoff=handoff,
    )
    assert child.active
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_rejects_forged_capability(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    _atomic_replace(repo / "a.py", b"# accepted proof\n")
    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    forged = replace(handoff)
    assert forged is not handoff
    assert not child.restore(
        full=False,
        active_parent_postimage_handoff=forged,
    )
    assert child.active
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_rejects_foreign_sibling_capability(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    run = corpus.spec_dir / "runs" / "sibling-parent-handoff"
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="sibling-parent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    first = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "first_mutations",
        slice_id="first-sibling",
        scope=("b.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    second = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "second_mutations",
        slice_id="second-sibling",
        scope=("c.py",),
        sequence=3,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    _atomic_replace(repo / "a.py", b"# sibling proof\n")
    handoff = first.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None

    assert not second.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    journals._unregister_live(second.transaction_id)
    journals._unregister_live(first.transaction_id)
    journals._unregister_live(parent.transaction_id)


def test_active_parent_handoff_rejects_foreign_checkout_capability(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    first_repo, _first_corpus, first_parent, first_child = _begin_family(
        tmp_path / "first"
    )
    _second_repo, _second_corpus, second_parent, second_child = _begin_family(
        tmp_path / "second"
    )
    _atomic_replace(first_repo / "a.py", b"# foreign checkout proof\n")
    handoff = first_child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None

    assert not second_child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    for journal in (
        second_child,
        second_parent,
        first_child,
        first_parent,
    ):
        journals._unregister_live(journal.transaction_id)


def test_active_parent_handoff_rejects_grandparent_only_path_grant(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, corpus = _setup(tmp_path)
    run = corpus.spec_dir / "runs" / "grandparent-handoff"
    grandparent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "grandparent_mutations",
        slice_id="grandparent",
        scope=("a.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    parent = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "parent_mutations",
        slice_id="parent",
        scope=("c.py",),
        sequence=2,
        spec=corpus.latest_spec(),
        parent_transaction_id=grandparent.transaction_id,
    )
    child = journals.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=run / "child_mutations",
        slice_id="child",
        scope=("b.py",),
        sequence=3,
        spec=corpus.latest_spec(),
        parent_transaction_id=parent.transaction_id,
    )
    _atomic_replace(repo / "a.py", b"# grandparent-owned postimage\n")

    with pytest.raises(
        journals.MutationJournalError,
        match="outside immediate-parent scope",
    ):
        child.capture_active_parent_postimage_handoff(("a.py",))
    journals._unregister_live(child.transaction_id)
    journals._unregister_live(parent.transaction_id)
    journals._unregister_live(grandparent.transaction_id)


def test_consumed_active_parent_handoff_reproof_rejects_substitution(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _mutation_journal as journals

    repo, _corpus, parent, child = _begin_family(tmp_path)
    _atomic_replace(repo / "a.py", b"# accepted and consumed\n")
    handoff = child.capture_active_parent_postimage_handoff(("a.py",))
    assert handoff is not None
    assert child.restore(
        full=False,
        active_parent_postimage_handoff=handoff,
    )
    child.reprove_consumed_active_parent_postimage_handoff(handoff)
    (repo / "a.py").write_bytes(b"# substituted after cleanup\n")

    with pytest.raises(
        journals.MutationJournalError,
        match="differs from its accepted proof",
    ):
        child.reprove_consumed_active_parent_postimage_handoff(handoff)
    assert parent.restore(full=True)
