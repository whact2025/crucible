# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``_compose_dependency_diffs`` persistent-workspace mode.

The DeepSWE in-container model bind-mounts a SINGLE host workspace at
``/app`` across every tier, so landed slices' writes (and breach-round
mutations) already persist on disk. Re-applying a prior tier's stored
diff there double-applies and hard-fails — the tier-handoff bug the
adaptix pilot surfaced 2026-06-13. ``persistent_workspace=True`` skips
re-application while still accumulating ``provided_files``; the
fresh-worktree products (swebench / kit / IsaacSim) keep re-applying.

Hermetic: a real throwaway git repo + a deliberately non-applying diff
blob + fake corpus / tier_dag / partition. No backends, no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.workflows import _partition
from crucible.workflows import tier as T


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ws"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "foo.py").write_text("real line 1\nreal line 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


# A valid unified diff (so _diff_file_paths extracts foo.py) whose
# context matches NEITHER the on-disk before- nor after-state, so it
# fails both `git apply` and `git apply --reverse` — exactly the drift
# the in-container breach round produces.
_NON_APPLYING_DIFF = (
    b"diff --git a/foo.py b/foo.py\n"
    b"index 1111111..2222222 100644\n"
    b"--- a/foo.py\n"
    b"+++ b/foo.py\n"
    b"@@ -1,2 +1,3 @@\n"
    b" drifted context A\n"
    b" drifted context B\n"
    b"+added line\n"
)


def _fake_tier_dag():
    """t0 (no deps) ← t1 (depends_on t0). One slice each."""
    t0 = SimpleNamespace(
        tier_id="t0", depends_on=[],
        slices=[SimpleNamespace(id="s0")],
    )
    t1 = SimpleNamespace(
        tier_id="t1", depends_on=["t0"],
        slices=[SimpleNamespace(id="s1")],
    )
    return SimpleNamespace(tiers=[t0, t1])


def _wire(monkeypatch):
    """Fake corpus.get_artifact + partition so t0/s0 carries the
    non-applying diff_ref and t1/s1 carries nothing."""
    corpus = SimpleNamespace(
        get_artifact=lambda ref: _NON_APPLYING_DIFF if ref == "ref0" else b"",
    )

    def _fake_partition(_corpus, tier):
        if tier.tier_id == "t0":
            return {"s0": SimpleNamespace(diff_ref="ref0")}, ()
        return {}, ()

    monkeypatch.setattr(_partition, "partition_for_resume", _fake_partition)
    return corpus


def test_persistent_workspace_skips_reapplication(tmp_path, monkeypatch):
    """With persistent_workspace=True the non-applying dependency diff is
    NOT re-applied (no raise), and foo.py is still reported as provided."""
    repo = _make_repo(tmp_path)
    corpus = _wire(monkeypatch)

    applied, provided = T._compose_dependency_diffs(
        corpus, _fake_tier_dag(), "t1", str(repo),
        persistent_workspace=True,
    )
    assert applied == []                 # nothing re-applied on disk
    assert provided == ["foo.py"]        # still accounted for review
    # The workspace file is untouched (no double-apply mangling).
    assert (repo / "foo.py").read_text() == "real line 1\nreal line 2\n"


def test_fresh_worktree_still_raises_on_bad_diff(tmp_path, monkeypatch):
    """Default (fresh-worktree) mode re-applies and hard-fails on a diff
    that doesn't apply — the swebench/kit invariant is preserved."""
    repo = _make_repo(tmp_path)
    corpus = _wire(monkeypatch)

    with pytest.raises(RuntimeError, match="failed to apply landed diff"):
        T._compose_dependency_diffs(
            corpus, _fake_tier_dag(), "t1", str(repo),
            persistent_workspace=False,
        )
