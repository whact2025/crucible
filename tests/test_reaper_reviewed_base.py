# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""A reviewed source pin preserves historical bytes without reaccepting work."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path

import pytest

from assay import ContentHash
from assay.canonical import hash_record
from assay.types.dag import TierDAG
from crucible.workflows import _reaper
from crucible.workflows._mutation_journal import RecoveryReport
from crucible.workflows._partition import attach_resume_identity
from tests.test_reaper import _capture_diff, _commit, _real_impl, _real_setup


def _head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()


def _retained_files(root: Path) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _reviewed_baseline(
    tmp_path: Path, *, scope_escape: bool = False, multi_path: bool = False,
):
    repo, corpus, tier, slice_, old_base = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("historical implementation\n")
    if multi_path:
        slice_ = dataclasses.replace(
            slice_, directive=dataclasses.replace(
                slice_.directive, creates_paths=("new.txt",), blast_radius={"max_files": 2},
            ),
        )
        tier = dataclasses.replace(tier, slices=(slice_,))
        previous = corpus.latest_spec()
        corpus.put_spec(dataclasses.replace(
            previous, version=previous.version + 1,
            parent=ContentHash(hash_record(previous)),
            root=previous.root or ContentHash(hash_record(previous)),
            tier_dag=TierDAG(tiers=(tier,)),
        ))
        (repo / "new.txt").write_text("committed new baseline file\n")
        subprocess.run(["git", "add", "new.txt"], cwd=repo, check=True)
    impl = _real_impl(
        corpus, diff=_capture_diff(repo), classification="hard_fail_max_turns",
        base_commit=old_base, timestamp="2026-08-06T00:00:00+00:00",
        scope_escapes=("escape.txt",) if scope_escape else (),
        files_modified=("foo.txt", "new.txt") if multi_path else ("foo.txt",),
    )
    impl = attach_resume_identity(corpus, corpus.latest_spec(), tier, slice_, impl)
    corpus.commit(implementations=(impl,))
    if scope_escape:
        (repo / "escape.txt").write_text("historical escape still needs authority\n")
    _commit(repo, "explicitly reviewed successor source baseline")
    reviewed_base = _head(repo)
    replacement = dataclasses.replace(
        slice_, id="successor", directive=dataclasses.replace(
            slice_.directive, slice_id="successor", intent="new requirements",
        ),
    )
    previous = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        previous, version=previous.version + 1,
        parent=ContentHash(hash_record(previous)),
        project_context=dataclasses.replace(previous.project_context, base_commit=reviewed_base),
        tier_dag=TierDAG(tiers=(dataclasses.replace(tier, slices=(replacement,)),)),
    ))
    return repo, corpus, old_base, reviewed_base


def test_reviewed_baseline_in_fresh_independent_clone_preserves_old_records(tmp_path):
    repo, corpus, old_base, reviewed_base = _reviewed_baseline(tmp_path)
    fresh = tmp_path / "private-clone"
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", str(repo), str(fresh)],
        check=True, capture_output=True,
    )
    before = _retained_files(corpus.spec_dir)
    spec_before = corpus.latest_spec()

    report = _reaper.reap_orphaned_dirt(corpus, fresh)

    assert reviewed_base != old_base
    assert report.failures == ()
    assert not report.reaped_anything
    assert report.cleanup_settlement_ref is None
    assert corpus.latest_spec() == spec_before
    assert (fresh / "foo.txt").read_text() == "historical implementation\n"
    assert _head(fresh) == reviewed_base
    assert report.legacy_mutations_skipped
    for path, digest in before.items():
        assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("change", [
    "bytes", "staged-bytes", "assume-unchanged", "mode", "deletion", "symlink", "directory",
])
def test_reviewed_base_never_protects_changed_worktree_entries(tmp_path, change):
    repo, corpus, _old_base, _reviewed_base = _reviewed_baseline(tmp_path)
    target = repo / "foo.txt"
    if change in {"bytes", "staged-bytes", "assume-unchanged"}:
        target.write_text("unaccepted candidate bytes\n")
        if change == "staged-bytes":
            subprocess.run(["git", "add", "foo.txt"], cwd=repo, check=True)
        elif change == "assume-unchanged":
            subprocess.run(
                ["git", "update-index", "--assume-unchanged", "foo.txt"],
                cwd=repo, check=True,
            )
    elif change == "mode":
        target.chmod(0o755)
    elif change == "deletion":
        target.unlink()
    elif change == "symlink":
        target.unlink()
        target.symlink_to("elsewhere")
    else:
        target.unlink()
        target.mkdir()
    spec_before = corpus.latest_spec()

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert report.failures
    assert not report.reaped_anything
    assert report.scope_escapes_deleted == ()
    assert corpus.latest_spec() == spec_before
    if change in {"bytes", "staged-bytes", "assume-unchanged"}:
        assert target.read_text() == "unaccepted candidate bytes\n"
    elif change == "mode":
        assert target.stat().st_mode & 0o100
    elif change == "deletion":
        assert not target.exists()
    elif change == "symlink":
        assert target.is_symlink()
    else:
        assert target.is_dir()


def test_clean_unreviewed_head_is_not_baseline_authority(tmp_path):
    repo, corpus, _old_base, reviewed_base = _reviewed_baseline(tmp_path)
    (repo / "unrelated.txt").write_text("same old path bytes, different HEAD\n")
    _commit(repo, "not the declared source pin")
    assert _head(repo) != reviewed_base
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo)

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert any("mutation may still be present" in item for item in report.failures)
    assert not report.reaped_anything


def test_committed_scope_escape_is_not_protected_by_new_baseline(tmp_path):
    repo, corpus, _old_base, reviewed_base = _reviewed_baseline(tmp_path, scope_escape=True)
    assert _head(repo) == reviewed_base

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert any("mutation may still be present" in item for item in report.failures)
    assert report.scope_escapes_deleted == ()
    assert (repo / "escape.txt").read_text() == "historical escape still needs authority\n"


def test_old_base_absence_proof_remains_valid(tmp_path):
    repo, corpus, _old_base, _reviewed_base = _reviewed_baseline(tmp_path)
    (repo / "foo.txt").write_text("original\n")

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert not report.reaped_anything
    assert (repo / "foo.txt").read_text() == "original\n"


def test_old_absence_and_reviewed_baseline_protection_are_per_path(tmp_path):
    repo, corpus, _old_base, _reviewed_base = _reviewed_baseline(tmp_path, multi_path=True)
    (repo / "foo.txt").write_text("original\n")

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert not report.reaped_anything
    assert (repo / "foo.txt").read_text() == "original\n"
    assert (repo / "new.txt").read_text() == "committed new baseline file\n"


def test_unsettled_journal_still_blocks_before_reviewed_baseline(tmp_path, monkeypatch):
    repo, corpus, _old_base, _reviewed_base = _reviewed_baseline(tmp_path)
    monkeypatch.setattr(
        _reaper, "_recover_mutation_journals_locked",
        lambda *args, **kwargs: RecoveryReport(
            recovered=(), preserved=(), failures=("unsettled exact journal",),
        ),
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert report.failures == ("unsettled exact journal",)
    assert not report.reaped_anything
    assert (repo / "foo.txt").read_text() == "historical implementation\n"


def test_head_change_during_baseline_comparison_fails_closed(tmp_path, monkeypatch):
    repo, corpus, old_base, reviewed_base = _reviewed_baseline(tmp_path)
    values = iter((reviewed_base, old_base))
    monkeypatch.setattr(_reaper, "_head", lambda *args: next(values))

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert any("HEAD changed during reviewed baseline proof" in item for item in report.failures)
    assert not report.reaped_anything
    assert (repo / "foo.txt").read_text() == "historical implementation\n"
