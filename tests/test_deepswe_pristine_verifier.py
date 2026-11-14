# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from crucible.products.deepswe.pristine_verifier import (
    CandidateApplyStatus,
    CandidateCaptureStatus,
    PristineReplayStatus,
    VerifierPatchStatus,
    candidate_from_image_patch,
    capture_source_candidate,
    prepare_candidate_replay,
    prove_image_replay,
    verify_hidden_model_patch,
)


def _git(repo: Path, *args: str, input: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        input=input,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "authoring"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "bin").mkdir()
    (repo / "src" / "old.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "src" / "keep.py").write_text("KEEP = 1\n", encoding="utf-8")
    (repo / "tests" / "test_old.py").write_text(
        "def test_old(): pass\n", encoding="utf-8"
    )
    tool = repo / "bin" / "tool"
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD").decode().strip()


def _render_patch(repo: Path, base: str) -> bytes:
    _git(repo, "reset", "--soft", base)
    try:
        _git(repo, "add", "-A", "--", ".")
        return _git(repo, "diff", "--cached", "--binary")
    finally:
        _git(repo, "reset", "-q")


def test_capture_replay_and_verifier_artifact_are_byte_bound(
    tmp_path: Path,
):
    repo, base = _repo(tmp_path)
    pristine = tmp_path / "pristine"
    _git(tmp_path, "clone", "-q", "--no-hardlinks", str(repo), str(pristine))

    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    os.rename(repo / "src" / "old.py", repo / "src" / "renamed.py")
    (repo / "src" / "new.bin").write_bytes(b"\x00\x01\xffsource\x00")
    tool = repo / "bin" / "tool"
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    # A model-authored visible repository test is part of the canonical patch.
    (repo / "tests" / "test_old.py").write_text(
        "def test_new_contract(): pass\n", encoding="utf-8"
    )
    # Simulate Tier's provenance-bound breach journal: the pipeline creates a
    # temporary artifact and restores the absent pre-breach state before
    # candidate capture.
    (repo / "tests" / "breach_round.py").write_text(
        "raise AssertionError('pipeline only')\n", encoding="utf-8"
    )
    (repo / "tests" / "breach_round.py").unlink()

    candidate = capture_source_candidate(repo, expected_base_commit=base)

    assert candidate.status == CandidateCaptureStatus.CAPTURED
    assert candidate.patch
    assert candidate.diff_ref == f"sha256:{candidate.sha256}"
    assert candidate.diff_bytes == len(candidate.patch)
    assert candidate.raw_diff_bytes == candidate.diff_bytes
    assert candidate.stripped_protected_paths == ()
    assert b"tests/test_old.py" in candidate.patch
    assert b"tests/breach_round.py" not in candidate.patch
    assert b"src/new.bin" in candidate.patch
    assert b"src/renamed.py" in candidate.patch

    applied = prepare_candidate_replay(
        pristine,
        candidate,
        expected_base_commit=base,
    )
    replay = prove_image_replay(
        candidate,
        applied,
        _render_patch(pristine, base),
        expected_base_commit=base,
    )
    assert replay.status == PristineReplayStatus.PASSED
    assert replay.observed_sha256 == candidate.sha256
    assert not (pristine / "tests" / "breach_round.py").exists()
    assert "test_new_contract" in (
        pristine / "tests" / "test_old.py"
    ).read_text(encoding="utf-8")

    # This is the exact capture sequence used by upstream /tests/test.sh.
    _git(pristine, "add", "-A", "--", ".")
    upstream_patch = _git(pristine, "diff", "--cached", "--binary")
    assert upstream_patch == candidate.patch
    artifact = tmp_path / "logs" / "artifacts" / "model.patch"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(upstream_patch)
    proof = verify_hidden_model_patch(artifact, candidate)
    assert proof.status == VerifierPatchStatus.MATCHED
    assert proof.observed_sha256 == candidate.sha256


def test_same_image_rendering_requires_a_fresh_exact_replay(tmp_path: Path):
    repo, base = _repo(tmp_path)
    pristine = tmp_path / "pristine"
    _git(tmp_path, "clone", "-q", "--no-hardlinks", str(repo), str(pristine))
    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    candidate = capture_source_candidate(repo, expected_base_commit=base)
    applied = prepare_candidate_replay(
        pristine, candidate, expected_base_commit=base
    )
    replay = prove_image_replay(
        candidate,
        applied,
        _render_patch(pristine, base),
        expected_base_commit=base,
    )
    rebound = candidate_from_image_patch(
        candidate.patch,
        files_modified=candidate.files_modified,
        raw_diff_bytes=candidate.raw_diff_bytes,
    )
    fresh = tmp_path / "fresh"
    _git(tmp_path, "clone", "-q", "--no-hardlinks", str(repo), str(fresh))
    rebound_apply = prepare_candidate_replay(
        fresh, rebound, expected_base_commit=base
    )
    rebound_replay = prove_image_replay(
        rebound,
        rebound_apply,
        _render_patch(fresh, base),
        expected_base_commit=base,
    )

    assert rebound.patch == candidate.patch
    # The prior proof is not rebound or mutated into authority for new bytes.
    assert replay.candidate_sha256 == candidate.sha256
    assert rebound_replay.candidate_sha256 == rebound.sha256
    assert rebound_replay.observed_sha256 == rebound.sha256
    assert rebound_replay.status == PristineReplayStatus.PASSED


def test_arbitrary_rendered_bytes_cannot_borrow_prior_replay(tmp_path: Path):
    repo, base = _repo(tmp_path)
    pristine = tmp_path / "pristine"
    fresh = tmp_path / "fresh"
    _git(tmp_path, "clone", "-q", "--no-hardlinks", str(repo), str(pristine))
    _git(tmp_path, "clone", "-q", "--no-hardlinks", str(repo), str(fresh))
    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    candidate = capture_source_candidate(repo, expected_base_commit=base)
    prior_apply = prepare_candidate_replay(
        pristine, candidate, expected_base_commit=base
    )
    prior = prove_image_replay(
        candidate,
        prior_apply,
        _render_patch(pristine, base),
        expected_base_commit=base,
    )
    rendered = candidate_from_image_patch(
        b"not a Git patch\n",
        files_modified=("src/keep.py",),
        raw_diff_bytes=len(b"not a Git patch\n"),
    )
    rendered_apply = prepare_candidate_replay(
        fresh, rendered, expected_base_commit=base
    )
    proof = prove_image_replay(
        rendered,
        rendered_apply,
        candidate.patch,
        expected_base_commit=base,
    )

    assert prior.status == PristineReplayStatus.PASSED
    assert prior.candidate_sha256 == candidate.sha256
    assert rendered.sha256 != prior.candidate_sha256
    assert rendered_apply.status == CandidateApplyStatus.PATCH_APPLY_FAILED
    assert proof.status == PristineReplayStatus.PATCH_APPLY_FAILED
    assert proof.candidate_sha256 == rendered.sha256


def test_initial_candidate_capture_rejects_oversize_before_persistence(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.products.deepswe import pristine_verifier as pv

    repo, base = _repo(tmp_path)
    observed: dict = {}

    def _oversize(**kwargs):
        observed.update(kwargs)
        return b"x" * 9, ("huge.bin",), base

    monkeypatch.setattr(
        pv,
        "capture_workspace_delta",
        _oversize,
    )

    candidate = pv.capture_source_candidate(
        repo,
        expected_base_commit=base,
        max_bytes=8,
    )

    assert candidate.status == CandidateCaptureStatus.FAILED
    assert candidate.patch == b""
    assert "patch byte limit" in (candidate.error or "")
    assert observed["max_diff_bytes"] == 8


def test_rendered_candidate_rejects_absolute_patch_limit(tmp_path: Path):
    repo, base = _repo(tmp_path)
    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    candidate = capture_source_candidate(repo, expected_base_commit=base)

    with pytest.raises(ValueError, match="patch byte limit"):
        candidate_from_image_patch(
            b"x" * 9,
            files_modified=candidate.files_modified,
            raw_diff_bytes=9,
            max_bytes=8,
        )


def test_cross_boundary_rename_is_part_of_canonical_patch(
    tmp_path: Path,
):
    repo, base = _repo(tmp_path)
    os.rename(repo / "src" / "old.py", repo / "tests" / "test_moved.py")

    candidate = capture_source_candidate(repo, expected_base_commit=base)

    assert candidate.status == CandidateCaptureStatus.CAPTURED
    assert candidate.patch
    assert candidate.stripped_protected_paths == ()
    assert b"src/old.py" in candidate.patch
    assert b"tests/test_moved.py" in candidate.patch


def test_test_named_production_paths_are_never_filtered(tmp_path: Path):
    repo, base = _repo(tmp_path)
    production_paths = (
        "bandit/core/test_set.py",
        "mobly/base_test.py",
        "lib/runners/browser_test_runner.js",
    )
    for relative in production_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("production = true\n", encoding="utf-8")

    candidate = capture_source_candidate(repo, expected_base_commit=base)

    assert candidate.status == CandidateCaptureStatus.CAPTURED
    assert candidate.stripped_protected_paths == ()
    assert set(production_paths) <= set(candidate.files_modified)
    for relative in production_paths:
        assert relative.encode() in candidate.patch


def test_pristine_replay_fails_closed_on_dirty_or_wrong_base(tmp_path: Path):
    repo, base = _repo(tmp_path)
    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    candidate = capture_source_candidate(repo, expected_base_commit=base)
    assert candidate.status == CandidateCaptureStatus.CAPTURED

    dirty = tmp_path / "dirty"
    _git(tmp_path, "clone", "-q", "--no-hardlinks", str(repo), str(dirty))
    # clone copied HEAD, not the dirty source delta; make its own unrelated dirt.
    (dirty / "src" / "old.py").write_text("DIRTY = 1\n", encoding="utf-8")

    report = prepare_candidate_replay(
        dirty, candidate, expected_base_commit=base
    )
    assert report.status == CandidateApplyStatus.BASE_INVALID


def test_verifier_patch_mismatch_is_typed_and_preserved(tmp_path: Path):
    repo, base = _repo(tmp_path)
    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    candidate = capture_source_candidate(repo, expected_base_commit=base)
    artifact = tmp_path / "model.patch"
    altered = bytearray(candidate.patch)
    altered[-1] = ord("X") if altered[-1] != ord("X") else ord("Y")
    artifact.write_bytes(altered)

    report = verify_hidden_model_patch(artifact, candidate)

    assert report.status == VerifierPatchStatus.MISMATCH
    assert report.observed_sha256 != candidate.sha256
    assert report.observed_bytes == len(candidate.patch)


def _captured_candidate(tmp_path: Path):
    repo, base = _repo(tmp_path)
    (repo / "src" / "keep.py").write_text("KEEP = 2\n", encoding="utf-8")
    return capture_source_candidate(repo, expected_base_commit=base)


def test_verifier_patch_leaf_symlink_is_never_followed(tmp_path: Path):
    candidate = _captured_candidate(tmp_path)
    forged = tmp_path / "forged.patch"
    forged.write_bytes(candidate.patch)
    artifact = tmp_path / "logs" / "artifacts" / "model.patch"
    artifact.parent.mkdir(parents=True)
    artifact.symlink_to(forged)

    report = verify_hidden_model_patch(artifact, candidate)

    assert report.status == VerifierPatchStatus.UNREADABLE
    assert report.observed_sha256 is None
    assert "unsafe" in (report.error or "")


def test_verifier_patch_parent_symlink_is_never_followed(tmp_path: Path):
    candidate = _captured_candidate(tmp_path)
    forged_parent = tmp_path / "forged-artifacts"
    forged_parent.mkdir()
    (forged_parent / "model.patch").write_bytes(candidate.patch)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "artifacts").symlink_to(forged_parent, target_is_directory=True)

    report = verify_hidden_model_patch(
        logs / "artifacts" / "model.patch",
        candidate,
    )

    assert report.status == VerifierPatchStatus.UNREADABLE
    assert report.observed_sha256 is None


def test_verifier_patch_nonregular_leaf_is_rejected(tmp_path: Path):
    candidate = _captured_candidate(tmp_path)
    artifact = tmp_path / "logs" / "artifacts" / "model.patch"
    artifact.mkdir(parents=True)

    report = verify_hidden_model_patch(artifact, candidate)

    assert report.status == VerifierPatchStatus.UNREADABLE
    assert "regular file" in (report.error or "")


def test_verifier_patch_oversize_is_rejected_before_comparison(tmp_path: Path):
    candidate = _captured_candidate(tmp_path)
    artifact = tmp_path / "logs" / "artifacts" / "model.patch"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(candidate.patch + b"forged suffix")

    report = verify_hidden_model_patch(artifact, candidate)

    assert report.status == VerifierPatchStatus.UNREADABLE
    assert report.observed_sha256 is None
    assert report.observed_bytes is None
    assert "oversize" in (report.error or "")
