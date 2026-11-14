# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for ``_compose_dependency_diffs`` composing CUMULATIVE
per-tier diffs that touch the SAME file across tiers.

Per-tier diffs are captured as ``git diff [--binary] HEAD`` scoped to a
slice's files, and nothing is committed between tiers in the compose
model — so each landed diff is the full base→tier delta. Two real
failure modes the fabric-usd ``fabric_render_ovrtx_fabricschema_conformance``
run hit (2026-06-24):

  1. A file CREATED by an earlier tier re-appears as a ``new file`` in a
     later tier's diff (HEAD lacks it until a commit) → stacking the two
     collides ("already exists in working directory").
  2. A binary artifact captured without ``--binary`` carries no literal /
     full index → "cannot apply binary patch ... without full index line".

The fix resets every file a diff touches back to its base state, then
plain-applies the cumulative diff (later tier wins for shared files), and
assay captures with ``--binary`` so binary artifacts round-trip. These
tests pin both behaviors. Hermetic: real throwaway git repo + fake corpus
/ tier_dag / partition. No backends, no network.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.workflows import _partition
from crucible.workflows import tier as T


@pytest.fixture(autouse=True)
def _hermetic_git_env(monkeypatch, tmp_path):
    """Neutralize global/system git config for these tests — both our
    capture helpers AND tier.py's subprocess git calls (which inherit
    ``os.environ``) must see vanilla git, independent of the host's
    ``~/.gitconfig`` or any state a sibling test leaked in a full-suite
    run. Per-repo identity is still set locally in ``_make_repo``."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _bin(n: int, salt: int) -> bytes:
    """Deterministic pseudo-binary payload (no RNG → reproducible)."""
    return b"\x89PNG\r\n\x1a\n" + bytes((i * 31 + salt) % 256 for i in range(n))


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True
    ).stdout


def _make_repo(tmp_path: Path, seed: dict[str, str] | None = None) -> Path:
    repo = tmp_path / "ws"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n")
    for rel, content in (seed or {}).items():
        (repo / rel).write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _capture(repo: Path, scope: tuple[str, ...]) -> bytes:
    """Replicate assay's per-tier capture: intent-to-add new files so they
    show as new-file diffs, then ``git diff --binary HEAD`` (cumulative vs
    base), scoped — mirroring ``assay.primitives.implement``."""
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard",
                     "--", *scope).decode().split()
    if untracked:
        subprocess.run(["git", "add", "-N", "--", *untracked],
                       cwd=repo, check=False, capture_output=True)
    blob = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *scope],
        cwd=repo, check=False, capture_output=True,
    ).stdout
    if untracked:
        subprocess.run(["git", "reset", "-q", "HEAD", "--", *untracked],
                       cwd=repo, check=False, capture_output=True)
    return blob


def _dag():
    """t0 (no deps) ← t1 (depends_on t0). One slice each."""
    t0 = SimpleNamespace(tier_id="t0", depends_on=[],
                         slices=[SimpleNamespace(id="s0")])
    t1 = SimpleNamespace(tier_id="t1", depends_on=["t0"],
                         slices=[SimpleNamespace(id="s1")])
    return SimpleNamespace(tiers=[t0, t1])


def _wire(monkeypatch, blob_t0: bytes, blob_t1: bytes):
    corpus = SimpleNamespace(
        get_artifact=lambda ref: {"ref0": blob_t0, "ref1": blob_t1}.get(ref, b""),
    )

    def _fake_partition(_corpus, tier):
        if tier.tier_id == "t0":
            return {"s0": SimpleNamespace(diff_ref="ref0")}, ()
        if tier.tier_id == "t1":
            return {"s1": SimpleNamespace(diff_ref="ref1")}, ()
        return {}, ()

    monkeypatch.setattr(_partition, "partition_for_resume", _fake_partition)
    return corpus


def test_compose_new_file_created_then_modified_across_tiers(tmp_path, monkeypatch):
    """t0 CREATES module.rs (v1); t1 EDITS it (cumulative v2). Both diffs
    are `new file` vs the uncommitted base, so naive stacking collides.
    Compose must land t1's cumulative version."""
    # t0's diff: create module.rs v1
    src = _make_repo(tmp_path / "a")
    (src / "module.rs").write_text("// v1\nfn a() {}\n")
    blob_t0 = _capture(src, ("module.rs",))
    # t1's diff: cumulative v2 (same clean base, so also a new-file diff)
    src2 = _make_repo(tmp_path / "b")
    (src2 / "module.rs").write_text("// v1\nfn a() {}\n// v2\nfn b() {}\n")
    blob_t1 = _capture(src2, ("module.rs",))

    repo = _make_repo(tmp_path / "c")
    corpus = _wire(monkeypatch, blob_t0, blob_t1)
    applied, _ = T._compose_dependency_diffs(corpus, _dag(), "t1", str(repo))

    assert set(applied) == {"s0", "s1"}
    assert (repo / "module.rs").read_text() == "// v1\nfn a() {}\n// v2\nfn b() {}\n"


def test_compose_base_file_modified_by_two_tiers(tmp_path, monkeypatch):
    """A base-committed file edited by both tiers (cumulative). Resetting to
    base + plain-applying the later cumulative diff lands both edits — the
    behavior the old --3way path provided, preserved."""
    seed = {"f.rs": "l1\nl2\nl3\nl4\nl5\n"}
    a = _make_repo(tmp_path / "a", seed)
    (a / "f.rs").write_text("l1X\nl2\nl3\nl4\nl5\n")
    blob_t0 = _capture(a, ("f.rs",))
    b = _make_repo(tmp_path / "b", seed)
    (b / "f.rs").write_text("l1X\nl2\nl3\nl4\nl5Y\n")  # cumulative
    blob_t1 = _capture(b, ("f.rs",))

    repo = _make_repo(tmp_path / "c", seed)
    corpus = _wire(monkeypatch, blob_t0, blob_t1)
    T._compose_dependency_diffs(corpus, _dag(), "t1", str(repo))
    assert (repo / "f.rs").read_text() == "l1X\nl2\nl3\nl4\nl5Y\n"


def test_compose_binary_artifact_round_trips(tmp_path, monkeypatch):
    """A binary artifact created by t0 and replaced by t1 composes — the
    `--binary` capture carries the literal so plain apply reconstructs it
    (the prior path emitted only 'Binary files differ' → unappliable)."""
    png1 = _bin(600, salt=11)
    png2 = _bin(900, salt=97)
    a = _make_repo(tmp_path / "a")
    (a / "art.png").write_bytes(png1)
    blob_t0 = _capture(a, ("art.png",))
    b = _make_repo(tmp_path / "b")
    (b / "art.png").write_bytes(png2)
    blob_t1 = _capture(b, ("art.png",))
    assert b"GIT binary patch" in blob_t0  # capture carried the literal

    repo = _make_repo(tmp_path / "c")
    corpus = _wire(monkeypatch, blob_t0, blob_t1)
    T._compose_dependency_diffs(corpus, _dag(), "t1", str(repo))
    assert (repo / "art.png").read_bytes() == png2
