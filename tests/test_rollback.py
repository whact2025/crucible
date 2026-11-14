# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._rollback.rollback_slice``.

Each test fixtures a fresh tmpdir git repo, mocks Corpus.get_artifact to
return synthetic diff bytes, and asserts the working-tree state after
rollback. No live Corpus dependency.

Coverage:
- Modified tracked file is reverted to HEAD state.
- Untracked file in ``impl.files_modified`` is deleted (covers the
  ``creates_paths`` case that ``git diff HEAD`` doesn't capture).
- Untracked directory in ``impl.files_modified`` is removed iff empty.
- Untracked file outside ``impl.files_modified`` is left alone (rollback
  is per-slice, not per-cwd).
- Empty diff_ref + empty files_modified is a no-op success.
- Unresolvable diff_ref returns False without raising.
- Malformed diff bytes that ``git apply -R`` rejects → False without
  raising.
"""

from __future__ import annotations

import os
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)


def _commit(root: Path, msg: str = "step") -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-qm", msg], cwd=str(root), check=True)


def _capture_diff(root: Path) -> bytes:
    return subprocess.run(
        ["git", "diff", "HEAD"], cwd=str(root),
        check=True, capture_output=True,
    ).stdout


def _mock_corpus(diff_bytes: bytes) -> MagicMock:
    corpus = MagicMock()
    corpus.get_artifact.return_value = diff_bytes
    return corpus


def _impl(
    slice_id: str,
    files_modified: tuple[str, ...],
    *,
    scope_escapes: tuple[str, ...] = (),
) -> MagicMock:
    impl = MagicMock()
    impl.slice_id = slice_id
    impl.diff_ref = "sha256:fake"
    impl.files_modified = files_modified
    impl.scope_escapes = scope_escapes
    return impl


def test_rollback_reverts_modified_tracked_file(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path, "initial")
    (tmp_path / "foo.txt").write_text("modified by slice")
    diff = _capture_diff(tmp_path)
    ok = rollback_slice(_mock_corpus(diff), _impl("s0", ("foo.txt",)), tmp_path)
    assert ok is True
    assert (tmp_path / "foo.txt").read_text() == "original"


def test_rollback_deletes_untracked_file_in_scope(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    # Slice creates a new file; git diff HEAD doesn't see it.
    (tmp_path / "created.txt").write_text("new content")
    diff = _capture_diff(tmp_path)
    assert diff == b"", "untracked file shouldn't appear in git diff HEAD"
    ok = rollback_slice(
        _mock_corpus(diff), _impl("s0", ("created.txt",)), tmp_path,
    )
    assert ok is True
    assert not (tmp_path / "created.txt").exists()


def test_rollback_leaves_untracked_file_outside_scope(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    # Two untracked files; only one is in the slice's scope.
    (tmp_path / "in_scope.txt").write_text("slice wrote this")
    (tmp_path / "out_of_scope.txt").write_text("someone else wrote this")
    ok = rollback_slice(
        _mock_corpus(b""), _impl("s0", ("in_scope.txt",)), tmp_path,
    )
    assert ok is True
    assert not (tmp_path / "in_scope.txt").exists()
    assert (tmp_path / "out_of_scope.txt").read_text() == "someone else wrote this"


def test_rollback_removes_empty_untracked_dir(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    (tmp_path / "subdir").mkdir()
    ok = rollback_slice(
        _mock_corpus(b""), _impl("s0", ("subdir",)), tmp_path,
    )
    assert ok is True
    assert not (tmp_path / "subdir").exists()


def test_rollback_leaves_non_empty_untracked_dir(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "leftover.txt").write_text("something else")
    ok = rollback_slice(
        _mock_corpus(b""), _impl("s0", ("subdir",)), tmp_path,
    )
    # Module refuses to recurse: dir + content remain, function still True.
    assert ok is True
    assert (tmp_path / "subdir" / "leftover.txt").exists()


def test_rollback_empty_diff_no_files_modified_is_noop(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    ok = rollback_slice(_mock_corpus(b""), _impl("s0", ()), tmp_path)
    assert ok is True


def test_rollback_unresolvable_diff_ref_returns_false(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    corpus = MagicMock()
    corpus.get_artifact.side_effect = FileNotFoundError("not in corpus")
    ok = rollback_slice(corpus, _impl("s0", ("anchor.txt",)), tmp_path)
    assert ok is False


def test_rollback_malformed_diff_returns_false(tmp_path: Path) -> None:
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    # Bytes that look diff-like but aren't applicable.
    malformed = b"--- a/nonexistent.txt\n+++ b/nonexistent.txt\n@@ -1 +1 @@\n-x\n+y\n"
    ok = rollback_slice(
        _mock_corpus(malformed), _impl("s0", ()), tmp_path,
    )
    assert ok is False


def test_rollback_combined_modify_and_create(tmp_path: Path) -> None:
    """End-to-end: slice both modifies an existing file AND creates a new one."""
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "existing.txt").write_text("original")
    _commit(tmp_path, "initial")
    (tmp_path / "existing.txt").write_text("modified")
    (tmp_path / "new.txt").write_text("created")
    diff = _capture_diff(tmp_path)
    ok = rollback_slice(
        _mock_corpus(diff),
        _impl("s0", ("existing.txt", "new.txt")),
        tmp_path,
    )
    assert ok is True
    assert (tmp_path / "existing.txt").read_text() == "original"
    assert not (tmp_path / "new.txt").exists()


# ----------------------------------------------------------------------
# Scope-escape cleanup (Layer 2 of implementer-time scope enforcement)
# ----------------------------------------------------------------------


def test_rollback_reverts_scope_escape_modification_on_tracked_file(tmp_path: Path) -> None:
    """A tracked file outside files_in_scope that the implementer
    modified anyway must be reverted to HEAD via ``git checkout``.
    The in-scope diff_ref doesn't include it (scope-filtered capture),
    so without this branch the orphan would persist."""
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "in_scope.txt").write_text("in_scope_v1")
    (tmp_path / "out_of_scope.txt").write_text("out_of_scope_v1")
    _commit(tmp_path, "initial")
    # In-scope edit + out-of-scope edit (the implementer's mistake).
    (tmp_path / "in_scope.txt").write_text("in_scope_v2")
    (tmp_path / "out_of_scope.txt").write_text("out_of_scope_TAINTED")
    # Diff capture is scope-filtered: only in_scope.txt is recorded.
    scoped_diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "in_scope.txt"],
        cwd=str(tmp_path), check=True, capture_output=True,
    ).stdout

    ok = rollback_slice(
        _mock_corpus(scoped_diff),
        _impl(
            "s0", ("in_scope.txt",),
            scope_escapes=("out_of_scope.txt",),
        ),
        tmp_path,
    )
    assert ok is True
    assert (tmp_path / "in_scope.txt").read_text() == "in_scope_v1"
    # The orphan: this is the structural fix — out_of_scope.txt must
    # be reverted in the same run, not left for the next-run reaper.
    assert (tmp_path / "out_of_scope.txt").read_text() == "out_of_scope_v1"


def test_rollback_deletes_scope_escape_untracked_file(tmp_path: Path) -> None:
    """An untracked file the implementer created outside its scope
    gets unlinked. Mirrors the in-scope untracked-deletion path."""
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    (tmp_path / "rogue_new.txt").write_text("created out of scope")

    ok = rollback_slice(
        _mock_corpus(b""),
        _impl("s0", (), scope_escapes=("rogue_new.txt",)),
        tmp_path,
    )
    assert ok is True
    assert not (tmp_path / "rogue_new.txt").exists()


def test_rollback_handles_mixed_scope_escapes(tmp_path: Path) -> None:
    """Mix of tracked-modified + untracked-created escapes get the
    right treatment per-path."""
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "existing.h").write_text("original_header")
    _commit(tmp_path, "initial")
    (tmp_path / "existing.h").write_text("touched_out_of_scope")
    (tmp_path / "created.cpp").write_text("created_out_of_scope")

    ok = rollback_slice(
        _mock_corpus(b""),
        _impl(
            "s0", (),
            scope_escapes=("existing.h", "created.cpp"),
        ),
        tmp_path,
    )
    assert ok is True
    assert (tmp_path / "existing.h").read_text() == "original_header"
    assert not (tmp_path / "created.cpp").exists()


def test_rollback_scope_escape_missing_untracked_is_silent_noop(tmp_path: Path) -> None:
    """If a scope_escape path is recorded but no longer on disk (e.g.
    a prior cleanup step beat us to it), we skip it without erroring.
    Rollback is best-effort cleanup, not a strict accounting check."""
    from crucible.workflows._rollback import rollback_slice
    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    # Don't create the escape file at all.

    ok = rollback_slice(
        _mock_corpus(b""),
        _impl("s0", (), scope_escapes=("phantom_escape.txt",)),
        tmp_path,
    )
    assert ok is True


def test_rollback_scope_escape_descriptor_restore_failure_logged_continues(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    """Descriptor-relative restore is best-effort on the scope-escape
    revert path. A failure logs a WARNING and the rollback loop
    continues with the rest — return value stays True because the
    slice has already failed and per-path restore failures are the
    lesser-evil dirt scenario."""
    import logging
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("v1_original")
    _commit(tmp_path, "initial")
    # Implementer "wrote" this out-of-scope.
    (tmp_path / "tracked.txt").write_text("v2_out_of_scope")

    def failed_restore(*args, **kwargs):
        raise _rollback.MutationJournalError(
            "fake: descriptor restore intentionally failed for test"
        )

    monkeypatch.setattr(_rollback, "_restore_head_entry_at_parent", failed_restore)

    with caplog.at_level(logging.WARNING, logger="crucible.workflows._rollback"):
        ok = _rollback.rollback_slice(
            _mock_corpus(b""),
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
        )
    # Slice rollback still returns True (best-effort) — the captured
    # diff revert is the load-bearing one; scope-escape per-path
    # failures don't fail the whole rollback.
    assert ok is True
    # The file was NOT reverted because the mocked descriptor restore failed.
    assert (tmp_path / "tracked.txt").read_text() == "v2_out_of_scope"
    # The WARNING fired so an operator can see it in postmortem.
    assert any(
        "failed to revert scope_escape" in r.message
        and "tracked.txt" in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_revert_scope_escapes_rejects_parent_traversal(tmp_path: Path) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "anchor.txt").write_text("anchor")
    _commit(repo, "initial")
    victim = tmp_path / "victim.txt"
    victim.write_text("must survive")
    impl = _impl("s0", (), scope_escapes=("../victim.txt",))

    assert revert_scope_escapes(impl, repo) == ([], [])
    assert victim.read_text() == "must survive"
    with pytest.raises(MutationJournalError, match="escapes checkout"):
        revert_scope_escapes(impl, repo, strict=True)
    assert victim.read_text() == "must survive"


@pytest.mark.parametrize("literal_path", (":(glob)*", "*.txt", ":leading"))
def test_revert_scope_escape_treats_untracked_git_pathspec_as_literal(
    tmp_path: Path,
    literal_path: str,
) -> None:
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("head\n")
    _commit(tmp_path, "initial")
    unrelated.write_text("operator worktree bytes\n")
    rogue = tmp_path / literal_path
    rogue.write_text("rogue\n")

    reverted, deleted = revert_scope_escapes(
        _impl("s0", (), scope_escapes=(literal_path,)),
        tmp_path,
        strict=True,
    )

    assert reverted == []
    assert deleted == [literal_path]
    assert not rogue.exists()
    assert unrelated.read_text() == "operator worktree bytes\n"


def test_revert_scope_escape_restores_literal_git_pathspec_only(
    tmp_path: Path,
) -> None:
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    literal_path = ":(glob)*"
    target = tmp_path / literal_path
    unrelated = tmp_path / "unrelated.txt"
    target.write_text("target head\n")
    unrelated.write_text("unrelated head\n")
    _commit(tmp_path, "initial")
    target.write_text("target rogue\n")
    unrelated.write_text("operator worktree bytes\n")

    reverted, deleted = revert_scope_escapes(
        _impl("s0", (), scope_escapes=(literal_path,)),
        tmp_path,
        strict=True,
    )

    assert reverted == [literal_path]
    assert deleted == []
    assert target.read_text() == "target head\n"
    assert unrelated.read_text() == "operator worktree bytes\n"


def test_revert_scope_escape_ignores_replace_ref_for_head_tree(
    tmp_path: Path,
) -> None:
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("immutable original HEAD\n")
    _commit(tmp_path, "original")
    original_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    target.write_text("ATTACKER replacement tree\n")
    _commit(tmp_path, "attacker replacement")
    attacker_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "reset", "--hard", original_commit],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "replace", original_commit, attacker_commit],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    # Ordinary Git object lookup is now attacker-substituted, proving the
    # replacement ref is live rather than merely present on disk.
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "ATTACKER replacement tree\n"
    target.write_text("dirty failed-slice bytes\n")

    reverted, deleted = revert_scope_escapes(
        _impl("s0", (), scope_escapes=("tracked.txt",)),
        tmp_path,
        strict=True,
    )

    assert reverted == ["tracked.txt"]
    assert deleted == []
    assert target.read_text() == "immutable original HEAD\n"


def test_revert_scope_escape_rejects_initialized_dirty_submodule(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    child = tmp_path / "child"
    parent = tmp_path / "parent"
    child.mkdir()
    parent.mkdir()
    _init_git_repo(child)
    tracked_child = child / "tracked.txt"
    tracked_child.write_text("child head\n")
    _commit(child, "child initial")
    _init_git_repo(parent)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "../child",
            "sub",
        ],
        cwd=parent,
        check=True,
        capture_output=True,
    )
    _commit(parent, "parent initial")
    initialized_child = parent / "sub" / "tracked.txt"
    initialized_child.write_text("dirty child bytes\n")

    impl = _impl("s0", (), scope_escapes=("sub",))
    assert revert_scope_escapes(impl, parent) == ([], [])
    assert initialized_child.read_text() == "dirty child bytes\n"
    with pytest.raises(MutationJournalError, match="gitlink/submodule"):
        revert_scope_escapes(impl, parent, strict=True)
    assert initialized_child.read_text() == "dirty child bytes\n"


@pytest.mark.parametrize(
    "hidden_index_flag",
    (None, "--assume-unchanged", "--skip-worktree"),
)
def test_revert_scope_escape_verifies_successful_restore_postcondition(
    tmp_path: Path,
    monkeypatch,
    hidden_index_flag: str | None,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    if hidden_index_flag is not None:
        subprocess.run(
            ["git", "update-index", hidden_index_flag, "tracked.txt"],
            cwd=tmp_path,
            check=True,
        )
    target.write_text("rogue\n")
    monkeypatch.setattr(
        _rollback,
        "_restore_head_entry_at_parent",
        lambda *args, **kwargs: None,
    )
    impl = _impl("s0", (), scope_escapes=("tracked.txt",))

    assert _rollback.revert_scope_escapes(impl, tmp_path) == ([], [])
    assert target.read_text() == "rogue\n"
    with pytest.raises(
        _rollback.MutationJournalError,
        match="remains different from HEAD",
    ):
        _rollback.revert_scope_escapes(impl, tmp_path, strict=True)
    assert target.read_text() == "rogue\n"


def test_revert_scope_escape_root_swap_keeps_replacement_victim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    repo = tmp_path / "repo"
    replacement = tmp_path / "replacement"
    held_repo = tmp_path / "held-repo"
    repo.mkdir()
    replacement.mkdir()
    _init_git_repo(repo)
    (repo / "anchor.txt").write_text("anchor")
    _commit(repo, "initial")
    original_rogue = repo / "rogue.txt"
    original_rogue.write_text("delete from original checkout")
    replacement_victim = replacement / "rogue.txt"
    replacement_victim.write_text("replacement must survive")
    real_run = subprocess.run
    swapped = False

    def swap_root_after_classification(cmd, *args, **kwargs):
        nonlocal swapped
        result = real_run(cmd, *args, **kwargs)
        if isinstance(cmd, list) and "ls-files" in cmd and not swapped:
            repo.rename(held_repo)
            replacement.rename(repo)
            swapped = True
        return result

    monkeypatch.setattr(
        _rollback.subprocess,
        "run",
        swap_root_after_classification,
    )

    reverted, deleted = _rollback.revert_scope_escapes(
        _impl("s0", (), scope_escapes=("rogue.txt",)),
        repo,
        strict=True,
    )

    assert reverted == []
    assert deleted == ["rogue.txt"]
    assert not (held_repo / "rogue.txt").exists()
    assert (repo / "rogue.txt").read_text() == "replacement must survive"


def test_revert_scope_escape_normalizes_batch_before_root_swap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    repo = tmp_path / "repo"
    replacement = tmp_path / "replacement"
    held_repo = tmp_path / "held-repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    replacement.mkdir()
    outside.mkdir()
    _init_git_repo(repo)
    (repo / "anchor.txt").write_text("anchor")
    _commit(repo, "initial")
    (repo / "first.txt").write_text("first original escape")
    (repo / "dir").mkdir()
    (repo / "dir" / "second.txt").write_text("second original escape")
    (replacement / "first.txt").write_text("replacement first victim")
    (outside / "second.txt").write_text("replacement symlink victim")
    (replacement / "dir").symlink_to(outside, target_is_directory=True)
    real_run = subprocess.run
    swapped = False

    def swap_root_after_first_classification(cmd, *args, **kwargs):
        nonlocal swapped
        result = real_run(cmd, *args, **kwargs)
        if isinstance(cmd, list) and "ls-files" in cmd and not swapped:
            repo.rename(held_repo)
            replacement.rename(repo)
            swapped = True
        return result

    monkeypatch.setattr(
        _rollback.subprocess,
        "run",
        swap_root_after_first_classification,
    )

    reverted, deleted = _rollback.revert_scope_escapes(
        _impl(
            "s0",
            (),
            scope_escapes=("first.txt", "dir/second.txt"),
        ),
        repo,
        strict=True,
    )

    assert reverted == []
    assert deleted == ["first.txt", "dir/second.txt"]
    assert not (held_repo / "first.txt").exists()
    assert not (held_repo / "dir" / "second.txt").exists()
    assert (repo / "first.txt").read_text() == "replacement first victim"
    assert (outside / "second.txt").read_text() == "replacement symlink victim"


def test_embedded_nul_paths_typed_fail_or_remain_best_effort(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes, rollback_slice

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    legitimate = tmp_path / "created.txt"
    legitimate.write_text("slice-created")
    nul_path = "rogue\0victim"
    scope_impl = _impl("s0", (), scope_escapes=(nul_path,))

    assert revert_scope_escapes(scope_impl, tmp_path) == ([], [])
    with pytest.raises(MutationJournalError, match="embedded NUL"):
        revert_scope_escapes(scope_impl, tmp_path, strict=True)

    ok = rollback_slice(
        _mock_corpus(b""),
        _impl("s0", (nul_path, "created.txt")),
        tmp_path,
    )
    assert ok is True
    assert not legitimate.exists()


def test_revert_scope_escape_fatal_git_classification_never_deletes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    rogue = tmp_path / "rogue.txt"
    rogue.write_text("must survive a classification failure")
    real_run = subprocess.run

    def fatal_ls_files(cmd, *args, **kwargs):
        if isinstance(cmd, list) and "ls-files" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=128,
                stdout=b"",
                stderr=b"fatal: simulated index failure",
            )
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(_rollback.subprocess, "run", fatal_ls_files)
    impl = _impl("s0", (), scope_escapes=("rogue.txt",))

    assert _rollback.revert_scope_escapes(impl, tmp_path) == ([], [])
    assert rogue.read_text() == "must survive a classification failure"
    with pytest.raises(
        _rollback.MutationJournalError,
        match="git ls-files exited 128",
    ):
        _rollback.revert_scope_escapes(impl, tmp_path, strict=True)
    assert rogue.read_text() == "must survive a classification failure"


def test_revert_scope_escape_deletes_dangling_symlink(tmp_path: Path) -> None:
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    dangling = tmp_path / "dangling"
    dangling.symlink_to("missing-target")
    assert not dangling.exists()
    assert dangling.is_symlink()

    reverted, deleted = revert_scope_escapes(
        _impl("s0", (), scope_escapes=("dangling",)),
        tmp_path,
        strict=True,
    )

    assert reverted == []
    assert deleted == ["dangling"]
    assert not dangling.is_symlink()


def test_revert_scope_escape_parent_swap_cannot_delete_outside(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    _init_git_repo(repo)
    (repo / "anchor.txt").write_text("anchor")
    _commit(repo, "initial")
    parent = repo / "parent"
    parent.mkdir()
    (parent / "rogue.txt").write_text("original in-repo entry")
    outside_victim = outside / "rogue.txt"
    outside_victim.write_text("outside must survive")
    held_parent = repo / "held-parent"
    real_run = subprocess.run
    swapped = False

    def swap_parent_before_classification(cmd, *args, **kwargs):
        nonlocal swapped
        if isinstance(cmd, list) and "ls-files" in cmd and not swapped:
            parent.rename(held_parent)
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(
        _rollback.subprocess,
        "run",
        swap_parent_before_classification,
    )

    with pytest.raises(
        _rollback.MutationJournalError,
        match="cannot safely open parent",
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("parent/rogue.txt",)),
            repo,
            strict=True,
        )

    assert outside_victim.read_text() == "outside must survive"
    assert (held_parent / "rogue.txt").read_text() == "original in-repo entry"


def test_revert_scope_escape_detects_parent_replacement_during_unlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    parent = tmp_path / "d"
    parent.mkdir()
    (parent / "rogue.txt").write_text("old untracked bytes")
    displaced = tmp_path / "displaced-d"
    real_unlink = _rollback.os.unlink
    swapped = False

    def replace_parent_during_unlink(name, *args, **kwargs):
        nonlocal swapped
        if name == "rogue.txt" and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            (parent / "rogue.txt").write_text("replacement must survive")
        return real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(_rollback.os, "unlink", replace_parent_during_unlink)
    monkeypatch.setattr(
        _rollback.os,
        "supports_dir_fd",
        {*os.supports_dir_fd, replace_parent_during_unlink},
    )
    with pytest.raises(
        _rollback.MutationJournalError,
        match="parent of workspace path .* changed identity",
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("d/rogue.txt",)),
            tmp_path,
            strict=True,
        )

    assert swapped
    assert (parent / "rogue.txt").read_text() == "replacement must survive"
    assert not (displaced / "rogue.txt").exists()


def test_revert_scope_escape_detects_entry_replacement_before_unlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    parent = tmp_path / "d"
    parent.mkdir()
    victim = parent / "rogue.txt"
    victim.write_text("classified bytes")
    displaced = parent / "classified-saved"
    real_assert_parent = _rollback._assert_workspace_parent_binding
    path_checks = 0

    def replace_entry_after_parent_revalidation(root_fd, path, parent_fd):
        nonlocal path_checks
        real_assert_parent(root_fd, path, parent_fd)
        if path == "d/rogue.txt":
            path_checks += 1
            if path_checks == 2:
                victim.rename(displaced)
                victim.write_text("replacement must survive")

    monkeypatch.setattr(
        _rollback,
        "_assert_workspace_parent_binding",
        replace_entry_after_parent_revalidation,
    )
    with pytest.raises(
        _rollback.MutationJournalError,
        match="workspace path .* changed identity",
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("d/rogue.txt",)),
            tmp_path,
            strict=True,
        )

    assert path_checks == 2
    assert victim.read_text() == "replacement must survive"
    assert displaced.read_text() == "classified bytes"


def test_revert_scope_escape_detects_parent_replacement_during_rmdir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    parent = tmp_path / "d"
    parent.mkdir()
    (parent / "empty").mkdir()
    displaced = tmp_path / "displaced-d"
    real_rmdir = _rollback.os.rmdir
    swapped = False

    def replace_parent_during_rmdir(name, *args, **kwargs):
        nonlocal swapped
        if name == "empty" and not swapped:
            swapped = True
            parent.rename(displaced)
            parent.mkdir()
            (parent / "empty").mkdir()
        return real_rmdir(name, *args, **kwargs)

    monkeypatch.setattr(_rollback.os, "rmdir", replace_parent_during_rmdir)
    monkeypatch.setattr(
        _rollback.os,
        "supports_dir_fd",
        {*os.supports_dir_fd, replace_parent_during_rmdir},
    )
    with pytest.raises(
        _rollback.MutationJournalError,
        match="parent of workspace path .* changed identity",
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("d/empty",)),
            tmp_path,
            strict=True,
        )

    assert swapped
    assert (parent / "empty").is_dir()
    assert not (displaced / "empty").exists()


def test_revert_scope_escape_real_parent_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path, "initial")
    parent = tmp_path / "dir"
    replacement = tmp_path / "replacement"
    held_parent = tmp_path / "held-dir"
    parent.mkdir()
    replacement.mkdir()
    original = parent / "rogue.txt"
    victim = replacement / "rogue.txt"
    original.write_text("original rogue must remain on failed cleanup")
    victim.write_text("replacement VICTIM must survive")
    real_run = subprocess.run
    swapped = False

    def replace_parent_after_classification(cmd, *args, **kwargs):
        nonlocal swapped
        result = real_run(cmd, *args, **kwargs)
        if isinstance(cmd, list) and "ls-files" in cmd and not swapped:
            parent.rename(held_parent)
            replacement.rename(parent)
            swapped = True
        return result

    monkeypatch.setattr(
        _rollback.subprocess,
        "run",
        replace_parent_after_classification,
    )

    with pytest.raises(
        _rollback.MutationJournalError,
        match="changed identity after classification",
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("dir/rogue.txt",)),
            tmp_path,
            strict=True,
        )

    assert (held_parent / "rogue.txt").read_text() == (
        "original rogue must remain on failed cleanup"
    )
    assert (parent / "rogue.txt").read_text() == "replacement VICTIM must survive"


def test_tracked_restore_parent_replacement_cannot_overwrite_victim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    parent = tmp_path / "dir"
    replacement = tmp_path / "replacement"
    held_parent = tmp_path / "held-dir"
    parent.mkdir()
    replacement.mkdir()
    original = parent / "tracked.txt"
    original.write_text("HEAD bytes\n")
    _commit(tmp_path, "initial")
    original.write_text("dirty original bytes\n")
    victim = replacement / "tracked.txt"
    victim.write_text("replacement VICTIM must survive\n")
    real_run = subprocess.run
    swapped = False

    def replace_parent_after_blob_read(cmd, *args, **kwargs):
        nonlocal swapped
        result = real_run(cmd, *args, **kwargs)
        if isinstance(cmd, list) and "cat-file" in cmd and not swapped:
            parent.rename(held_parent)
            replacement.rename(parent)
            swapped = True
        return result

    monkeypatch.setattr(
        _rollback.subprocess,
        "run",
        replace_parent_after_blob_read,
    )

    with pytest.raises(
        _rollback.MutationJournalError,
        match="changed identity after classification",
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("dir/tracked.txt",)),
            tmp_path,
            strict=True,
        )

    # The descriptor-bound original was restored, while the path replacement
    # was never opened or overwritten by the cleanup operation.
    assert (held_parent / "tracked.txt").read_text() == "HEAD bytes\n"
    assert (parent / "tracked.txt").read_text() == (
        "replacement VICTIM must survive\n"
    )


def test_tree_verifier_parent_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    parent = tmp_path / "dir"
    replacement = tmp_path / "replacement"
    displaced = tmp_path / "displaced"
    parent.mkdir()
    replacement.mkdir()
    (parent / "tracked.txt").write_text("base bytes\n")
    (replacement / "tracked.txt").write_text("attacker bytes\n")
    _commit(tmp_path, "initial")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "dir/tracked.txt")
    real_read = _rollback.os.read
    swapped = False

    def swap_parent_after_read(fd, size):
        nonlocal swapped
        payload = real_read(fd, size)
        if payload and not swapped:
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True
        return payload

    monkeypatch.setattr(_rollback.os, "read", swap_parent_after_read)
    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="parent of workspace path .* changed identity",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "dir/tracked.txt",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert (parent / "tracked.txt").read_text() == "attacker bytes\n"
    assert (displaced / "tracked.txt").read_text() == "base bytes\n"


def test_tree_verifier_final_entry_replacement_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    replacement = tmp_path / "replacement.txt"
    displaced = tmp_path / "displaced.txt"
    target.write_text("base bytes\n")
    _commit(tmp_path, "initial")
    replacement.write_text("attacker bytes\n")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "tracked.txt")
    real_read = _rollback.os.read
    swapped = False

    def swap_entry_after_read(fd, size):
        nonlocal swapped
        payload = real_read(fd, size)
        if payload and not swapped:
            target.rename(displaced)
            replacement.rename(target)
            swapped = True
        return payload

    monkeypatch.setattr(_rollback.os, "read", swap_entry_after_read)
    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="workspace path .* changed .*verification",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "tracked.txt",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert target.read_text() == "attacker bytes\n"
    assert displaced.read_text() == "base bytes\n"


def test_tree_verifier_same_inode_write_after_digest_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("base bytes\n")
    _commit(tmp_path, "initial")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "tracked.txt")
    initial_inode = target.stat().st_ino
    real_read = _rollback.os.read
    mutated = False

    def mutate_same_inode_at_eof(fd, size):
        nonlocal mutated
        payload = real_read(fd, size)
        if not payload and not mutated:
            mutated = True
            target.write_text("evil bytes\n")
            assert target.stat().st_ino == initial_inode
        return payload

    monkeypatch.setattr(_rollback.os, "read", mutate_same_inode_at_eof)
    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="workspace path .* changed during verification",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "tracked.txt",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert target.read_text() == "evil bytes\n"
    assert target.stat().st_ino == initial_inode


def test_tree_verifier_same_inode_write_during_binding_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("base bytes\n")
    _commit(tmp_path, "initial")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "tracked.txt")
    initial_inode = target.stat().st_ino
    real_binding = _rollback._assert_workspace_entry_binding
    mutated = False

    def mutate_same_inode_before_binding(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            target.write_text("evil bytes\n")
            assert target.stat().st_ino == initial_inode
        return real_binding(*args, **kwargs)

    monkeypatch.setattr(
        _rollback,
        "_assert_workspace_entry_binding",
        mutate_same_inode_before_binding,
    )
    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="workspace path .* changed after verification",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "tracked.txt",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert target.read_text() == "evil bytes\n"
    assert target.stat().st_ino == initial_inode


def test_tree_verifier_rejects_hardlinked_regular_file(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("base bytes\n")
    _commit(tmp_path, "initial")
    alias = tmp_path / "untracked-alias.txt"
    os.link(target, alias)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "tracked.txt")

    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="workspace path .* is hard-linked",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "tracked.txt",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert target.stat().st_ino == alias.stat().st_ino
    assert target.read_text() == "base bytes\n"


def test_tree_verifier_rejects_hardlinked_symlink(
    tmp_path: Path,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked-link"
    target.symlink_to("HEAD-target")
    _commit(tmp_path, "initial")
    alias = tmp_path / "untracked-link-alias"
    os.link(target, alias, follow_symlinks=False)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "tracked-link")

    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="workspace path .* is hard-linked",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "tracked-link",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert target.lstat().st_ino == alias.lstat().st_ino
    assert target.readlink() == Path("HEAD-target")


def test_tree_verifier_missing_parent_appearance_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor\n")
    _commit(tmp_path, "initial")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    entry = _rollback._git_tree_entry(root_fd, base, "new/file.txt")
    assert entry is None
    real_parent = _rollback._open_workspace_parent_if_present
    inserted = False

    @contextmanager
    def insert_missing_parent(root_descriptor, path):
        nonlocal inserted
        with real_parent(root_descriptor, path) as binding:
            if binding is None and not inserted:
                inserted = True
                (tmp_path / "new").mkdir()
                (tmp_path / "new" / "file.txt").write_text(
                    "attacker bytes\n"
                )
            yield binding

    monkeypatch.setattr(
        _rollback,
        "_open_workspace_parent_if_present",
        insert_missing_parent,
    )
    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="parent of workspace path .* appeared",
        ):
            _rollback._verify_worktree_path_matches_tree(
                root_fd,
                "new/file.txt",
                entry,
                tree_label=f"base {base}",
            )
    finally:
        _rollback.os.close(root_fd)

    assert (tmp_path / "new" / "file.txt").read_text() == "attacker bytes\n"


def test_absent_tracked_symlink_no_replace_preserves_inserted_victim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    link = tmp_path / "tracked-link"
    link.symlink_to("HEAD-target")
    _commit(tmp_path, "initial")
    link.unlink()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    parent_context = _rollback._open_workspace_parent(root_fd, "tracked-link")
    parent_fd, name = parent_context.__enter__()
    metadata, held_fd = _rollback._snapshot_workspace_entry(
        parent_fd,
        name,
        "tracked-link",
    )
    head_entry = _rollback._git_head_entry(root_fd, "tracked-link")
    real_symlink = _rollback.os.symlink
    inserted = False

    def insert_victim_before_no_replace(target, link_name, *args, **kwargs):
        nonlocal inserted
        if link_name == "tracked-link" and not inserted:
            inserted = True
            real_symlink("REPLACEMENT-VICTIM", link_name, *args, **kwargs)
        return real_symlink(target, link_name, *args, **kwargs)

    monkeypatch.setattr(_rollback.os, "symlink", insert_victim_before_no_replace)

    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="appeared before no-replace symlink install",
        ):
            _rollback._restore_head_entry_at_parent(
                root_fd,
                parent_fd,
                name,
                "tracked-link",
                head_entry,
                metadata,
                held_fd,
            )
    finally:
        parent_context.__exit__(None, None, None)
        _rollback.os.close(root_fd)

    assert link.is_symlink()
    assert link.readlink() == Path("REPLACEMENT-VICTIM")


def test_absent_tracked_regular_no_replace_preserves_inserted_victim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _rollback

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("HEAD bytes\n")
    _commit(tmp_path, "initial")
    target.unlink()
    _cwd, root_fd = _rollback._open_checkout_root(tmp_path)
    parent_context = _rollback._open_workspace_parent(root_fd, "tracked.txt")
    parent_fd, name = parent_context.__enter__()
    metadata, held_fd = _rollback._snapshot_workspace_entry(
        parent_fd,
        name,
        "tracked.txt",
    )
    head_entry = _rollback._git_head_entry(root_fd, "tracked.txt")
    real_open = _rollback.os.open
    inserted = False

    def insert_victim_before_no_replace(path, flags, *args, **kwargs):
        nonlocal inserted
        if (
            path == "tracked.txt"
            and flags & _rollback.os.O_CREAT
            and flags & _rollback.os.O_EXCL
            and not inserted
        ):
            inserted = True
            victim_fd = real_open(path, flags, *args, **kwargs)
            try:
                _rollback.os.write(victim_fd, b"REPLACEMENT-VICTIM\n")
            finally:
                _rollback.os.close(victim_fd)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(_rollback.os, "open", insert_victim_before_no_replace)

    try:
        with pytest.raises(
            _rollback.MutationJournalError,
            match="appeared before no-replace file install",
        ):
            _rollback._restore_head_entry_at_parent(
                root_fd,
                parent_fd,
                name,
                "tracked.txt",
                head_entry,
                metadata,
                held_fd,
            )
    finally:
        parent_context.__exit__(None, None, None)
        _rollback.os.close(root_fd)

    assert target.read_text() == "REPLACEMENT-VICTIM\n"


def test_rollback_slice_rejects_unsafe_files_modified_paths(
    tmp_path: Path,
) -> None:
    from crucible.workflows._rollback import rollback_slice

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    _init_git_repo(repo)
    (repo / "anchor.txt").write_text("anchor")
    _commit(repo, "initial")
    victim = tmp_path / "victim.txt"
    victim.write_text("parent victim")
    outside_victim = outside / "victim.txt"
    outside_victim.write_text("symlink-parent victim")
    (repo / "alias").symlink_to(outside, target_is_directory=True)
    git_config = (repo / ".git" / "config").read_bytes()
    legitimate = repo / "created.txt"
    legitimate.write_text("slice-created")

    ok = rollback_slice(
        _mock_corpus(b""),
        _impl(
            "s0",
            (
                "../victim.txt",
                str(victim),
                ".git/config",
                "alias/victim.txt",
                "created.txt",
            ),
        ),
        repo,
    )

    assert ok is True
    assert victim.read_text() == "parent victim"
    assert outside_victim.read_text() == "symlink-parent victim"
    assert (repo / ".git" / "config").read_bytes() == git_config
    assert not legitimate.exists()


def test_revert_tracked_scope_escape_preserves_staged_index(tmp_path: Path) -> None:
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    target.write_text("staged\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--binary", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout
    target.write_text("rogue worktree\n")

    reverted, deleted = revert_scope_escapes(
        _impl("s0", (), scope_escapes=("tracked.txt",)),
        tmp_path,
        strict=True,
    )

    assert reverted == ["tracked.txt"]
    assert deleted == []
    assert target.read_text() == "head\n"
    assert subprocess.run(
        ["git", "diff", "--cached", "--binary", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout == staged


def test_revert_tracked_scope_escape_rejects_staged_deletion(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    subprocess.run(
        ["git", "rm", "--cached", "tracked.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    target.write_text("dirty worktree\n")

    with pytest.raises(
        MutationJournalError,
        match="HEAD contains the path but the index does not",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_text() == "dirty worktree\n"


def test_revert_scope_escape_rejects_repository_filter_materialization(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    subprocess.run(
        ["git", "config", "filter.demo.clean", "sed s/WORKTREE/CANONICAL/"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "filter.demo.smudge", "sed s/CANONICAL/WORKTREE/"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".gitattributes").write_text("tracked.txt filter=demo\n")
    target = tmp_path / "tracked.txt"
    target.write_text("WORKTREE\n")
    _commit(tmp_path, "filtered initial")
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "CANONICAL\n"
    subprocess.run(
        ["git", "config", "filter.demo.clean", "false"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "filter.demo.smudge", "false"],
        cwd=tmp_path,
        check=True,
    )
    target.write_text("DIRTY\n")

    with pytest.raises(
        MutationJournalError,
        match="transforming Git attribute filter",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_text() == "DIRTY\n"


def test_revert_scope_escape_rejects_global_filter_materialization(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    global_config = tmp_path / "operator-global-config"
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.demo.clean",
            "sed s/WORKTREE/CANONICAL/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.demo.smudge",
            "sed s/CANONICAL/WORKTREE/",
        ],
        check=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / ".gitattributes").write_text("tracked.txt filter=demo\n")
    target = repo / "tracked.txt"
    target.write_text("WORKTREE\n")
    operator_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(global_config),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo,
        env=operator_env,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "globally filtered initial"],
        cwd=repo,
        env=operator_env,
        check=True,
    )
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=repo,
        env=operator_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "CANONICAL\n"
    target.write_text("DIRTY\n")

    with pytest.raises(
        MutationJournalError,
        match="transforming Git attribute filter",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            repo,
            strict=True,
        )

    assert target.read_text() == "DIRTY\n"


def test_revert_scope_escape_rejects_local_autocrlf_materialization(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"],
        cwd=tmp_path,
        check=True,
    )
    target = tmp_path / "tracked.txt"
    target.write_bytes(b"base\r\n")
    _commit(tmp_path, "autocrlf initial")
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout == b"base\n"
    target.write_bytes(b"dirty\r\n")

    with pytest.raises(
        MutationJournalError,
        match="core.autocrlf='true'",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_bytes() == b"dirty\r\n"


def test_revert_scope_escape_rejects_ambient_autocrlf_materialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    operator_home = tmp_path / "operator-home"
    operator_home.mkdir()
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(operator_home / ".gitconfig"),
            "core.autocrlf",
            "true",
        ],
        check=True,
    )
    monkeypatch.setenv("HOME", str(operator_home))
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    target = repo / "tracked.txt"
    target.write_bytes(b"base\r\n")
    _commit(repo, "ambient autocrlf initial")
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout == b"base\n"
    target.write_bytes(b"dirty\r\n")

    with pytest.raises(
        MutationJournalError,
        match="core.autocrlf='true'.*config-ambient",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            repo,
            strict=True,
        )

    assert target.read_bytes() == b"dirty\r\n"


def test_revert_scope_escape_rejects_masked_global_autocrlf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    operator_home = tmp_path / "operator-home"
    operator_home.mkdir()
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(operator_home / ".gitconfig"),
            "core.autocrlf",
            "true",
        ],
        check=True,
    )
    monkeypatch.setenv("HOME", str(operator_home))
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    target = repo / "tracked.txt"
    target.write_bytes(b"base\r\n")
    _commit(repo, "global autocrlf initial")
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout == b"base\n"
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"],
        cwd=repo,
        check=True,
    )
    target.write_bytes(b"dirty\r\n")

    with pytest.raises(
        MutationJournalError,
        match="core.autocrlf='true'.*config-ambient",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            repo,
            strict=True,
        )

    assert target.read_bytes() == b"dirty\r\n"


def test_revert_scope_escape_rejects_fifo_attributes_file_before_query(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    attributes_fifo = tmp_path / "attrs.fifo"
    os.mkfifo(attributes_fifo)
    subprocess.run(
        ["git", "config", "core.attributesFile", str(attributes_fifo)],
        cwd=tmp_path,
        check=True,
    )
    target.write_text("dirty\n")

    with pytest.raises(
        MutationJournalError,
        match="core.attributesfile=.*config-local",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_text() == "dirty\n"
    assert stat.S_ISFIFO(attributes_fifo.lstat().st_mode)


def test_revert_scope_escape_rejects_config_include_before_query(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    include_fifo = tmp_path / "include.fifo"
    os.mkfifo(include_fifo)
    subprocess.run(
        ["git", "config", "include.path", str(include_fifo)],
        cwd=tmp_path,
        check=True,
    )
    target.write_text("dirty\n")

    with pytest.raises(
        MutationJournalError,
        match="config include/includeIf directives are unsupported",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_text() == "dirty\n"
    assert stat.S_ISFIFO(include_fifo.lstat().st_mode)


@pytest.mark.parametrize(
    ("source_kind", "error_pattern"),
    (
        ("fifo", "not a regular file"),
        ("symlink", "not a regular file"),
        ("oversized", "exceeds"),
    ),
)
def test_revert_scope_escape_rejects_unsafe_default_attribute_source(
    tmp_path: Path,
    monkeypatch,
    source_kind: str,
    error_pattern: str,
) -> None:
    from crucible.workflows import _rollback

    operator_home = tmp_path / "operator-home"
    attribute_root = operator_home / ".config" / "git"
    attribute_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(operator_home))
    # Git gives XDG_CONFIG_HOME precedence over HOME for this default file.
    # Exercise the HOME fallback independently of the runner's ambient env.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    target = repo / "tracked.txt"
    target.write_text("head\n")
    _commit(repo, "initial")
    source = attribute_root / "attributes"
    if source_kind == "fifo":
        os.mkfifo(source)
    elif source_kind == "symlink":
        payload = attribute_root / "payload"
        payload.write_text("*.txt text\n")
        source.symlink_to(payload.name)
    else:
        with source.open("wb") as stream:
            stream.truncate(_rollback._ATTRIBUTE_SOURCE_MAX_BYTES + 1)
    target.write_text("dirty\n")

    with pytest.raises(
        _rollback.MutationJournalError,
        match=error_pattern,
    ):
        _rollback.revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            repo,
            strict=True,
        )

    assert target.read_text() == "dirty\n"


def test_revert_scope_escape_rejects_ambient_git_config_redirect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    redirected = tmp_path / "redirected-global-config"
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(redirected),
            "core.autocrlf",
            "true",
        ],
        check=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    operator_env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(redirected),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    target = repo / "tracked.txt"
    target.write_bytes(b"base\r\n")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo,
        env=operator_env,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "redirected autocrlf initial"],
        cwd=repo,
        env=operator_env,
        check=True,
    )
    target.write_bytes(b"dirty\r\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(redirected))

    with pytest.raises(
        MutationJournalError,
        match="ambient Git redirect environment is unsupported",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            repo,
            strict=True,
        )

    assert target.read_bytes() == b"dirty\r\n"


def test_revert_scope_escape_global_filter_cannot_be_masked_or_executed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    operator_home = tmp_path / "operator-home"
    operator_home.mkdir()
    global_attributes = operator_home / "global-attributes"
    global_attributes.write_text("tracked.txt filter=demo\n")
    global_config = operator_home / ".gitconfig"
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "core.attributesFile",
            str(global_attributes),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.demo.clean",
            "sed s/WORKTREE/CANONICAL/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.demo.smudge",
            "sed s/CANONICAL/WORKTREE/",
        ],
        check=True,
    )
    monkeypatch.setenv("HOME", str(operator_home))
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    target = repo / "tracked.txt"
    target.write_text("WORKTREE\n")
    _commit(repo, "globally attributed initial")
    assert subprocess.run(
        ["git", "show", "HEAD:tracked.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "CANONICAL\n"

    # Failed work attempts to mask the baseline global transform with a new
    # highest-precedence worktree rule. Cleanup must still inspect ambient
    # index/HEAD authority, and check-attr must never execute the driver.
    (repo / ".gitattributes").write_text("tracked.txt -filter\n")
    target.write_text("DIRTY\n")
    marker = tmp_path / "FILTER_DRIVER_EXECUTED"
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.demo.clean",
            f"touch {marker}",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.demo.smudge",
            f"touch {marker}",
        ],
        check=True,
    )

    with pytest.raises(
        MutationJournalError,
        match="transforming Git attribute|core.attributesFile",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            repo,
            strict=True,
        )

    assert target.read_text() == "DIRTY\n"
    assert not marker.exists()


@pytest.mark.parametrize(
    "attribute",
    (
        "text",
        "text eol=crlf",
        "crlf",
        "ident",
        "working-tree-encoding=UTF-16",
    ),
)
def test_revert_scope_escape_rejects_builtin_worktree_transform(
    tmp_path: Path,
    attribute: str,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    (tmp_path / ".gitattributes").write_text(
        f"tracked.txt {attribute}\n"
    )
    target.write_text("dirty\n")

    with pytest.raises(
        MutationJournalError,
        match="transforming Git attribute",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_text() == "dirty\n"


def test_revert_scope_escape_checks_transform_from_immutable_head(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import MutationJournalError
    from crucible.workflows._rollback import revert_scope_escapes

    _init_git_repo(tmp_path)
    (tmp_path / ".gitattributes").write_text("tracked.txt filter=demo\n")
    target = tmp_path / "tracked.txt"
    target.write_text("head\n")
    _commit(tmp_path, "initial")
    # Failed work suppresses the transform in both worktree and index. The
    # immutable HEAD attribute source must still prevent raw blob restore.
    (tmp_path / ".gitattributes").write_text("tracked.txt -filter\n")
    subprocess.run(
        ["git", "add", ".gitattributes"],
        cwd=tmp_path,
        check=True,
    )
    target.write_text("dirty\n")

    with pytest.raises(
        MutationJournalError,
        match="transforming Git attribute filter.*head",
    ):
        revert_scope_escapes(
            _impl("s0", (), scope_escapes=("tracked.txt",)),
            tmp_path,
            strict=True,
        )

    assert target.read_text() == "dirty\n"


def test_revert_scope_escapes_cleans_orphan_keeps_in_scope(
    tmp_path: Path,
) -> None:
    """Standalone helper used by the breach loop: it must delete the
    out-of-scope orphan (the duplicate-symbol hazard a glob build would
    sweep in) WITHOUT touching the slice's legitimate in-scope edits,
    which the breach loop carries cumulatively across rounds."""
    from crucible.workflows._rollback import revert_scope_escapes
    _init_git_repo(tmp_path)
    (tmp_path / "test_pool.cpp").write_text("// committed test\n")
    _commit(tmp_path, "initial")
    # In-scope edit the Defender legitimately made this round — NOT a
    # scope escape, so it must survive.
    (tmp_path / "test_pool.cpp").write_text("// edited in-scope\n")
    # Out-of-scope orphan the Defender left behind (untracked backup the
    # build would glob in → multiple-definition link error).
    (tmp_path / "test_pool_orig.cpp").write_text("// stray backup\n")

    reverted, deleted = revert_scope_escapes(
        _impl("breach_defender_r0", ("test_pool.cpp",),
              scope_escapes=("test_pool_orig.cpp",)),
        tmp_path,
    )

    assert deleted == ["test_pool_orig.cpp"]
    assert reverted == []
    # Orphan gone; in-scope edit preserved verbatim.
    assert not (tmp_path / "test_pool_orig.cpp").exists()
    assert (tmp_path / "test_pool.cpp").read_text() == "// edited in-scope\n"
