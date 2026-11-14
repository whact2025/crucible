# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    Implementation,
    ImplementationDirective,
    LifecycleEvent,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from assay.corpus.store import Corpus
from assay.primitives.implement import capture_workspace_delta

from crucible.workflows import _gate_evidence, _partition, reconcile
from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows._gate_evidence import (
    HANDOFF_VERIFICATION_ARTIFACT,
    persist_verification_evidence,
)
from crucible.workflows._atomic_persistence import persist_and_attach
from crucible.workflows.handoff import main as handoff_main
from crucible.workflows.reconcile import main as reconcile_main
from crucible.workflows._run import exec_ctx_now


_REAL_VERIFICATION_MATCHES = (
    _gate_evidence.verification_matches_implementation
)


@pytest.fixture(autouse=True)
def _isolate_reconcile_postimage_tests_from_handoff_gate(monkeypatch):
    """Keep legacy postimage tests focused on their declared invariant.

    Those tests use ``handoff --skip-verify`` only as a convenient diff
    recorder.  Their synthetic candidates stand in for already-verified
    records; the explicit verification-authority matrix below restores the
    real matcher and covers the new fail-closed behavior.
    """

    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        lambda *args, **kwargs: True,
    )


_SPEC = """
schema_version: 1
project_context:
  project: reconcile-test
  repo_url: https://example.invalid/reconcile.git
  base_commit: "0000000"
  branch: main
  submodule_path: null
  runners: {}
intent: Reconcile completed implementation evidence.
constraints: []
done_criteria: []
"""


def _setup(
    tmp_path: Path,
    *,
    creates_paths: tuple[str, ...] = (),
    base_commit_length: int | None = None,
    initial_files: dict[str, str] | None = None,
) -> tuple[Path, Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "r@invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Reconcile"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("seed\n")
    for relative, content in (initial_files or {}).items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)

    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    declared_base = (
        base_commit[:base_commit_length]
        if base_commit_length is not None
        else base_commit
    )
    base = Spec.from_yaml(
        _SPEC.replace('"0000000"', f'"{declared_base}"'),
    )
    directive = ImplementationDirective(
        slice_id="s0",
        intent=(
            f"create {', '.join(creates_paths)}"
            if creates_paths
            else "update seed"
        ),
        files_in_scope=() if creates_paths else ("seed.txt",),
        creates_paths=creates_paths,
        verify_mode="skip",
        blast_radius={"max_files": 1},
    )
    planned = base.with_tier_dag(TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(Slice(id="s0", directive=directive),),
    ),)))
    spec_dir = tmp_path / "corpus"
    corpus = Corpus(spec_dir)
    corpus.put_spec(base)
    corpus.put_spec(planned)
    return repo, spec_dir, corpus


def _attach_tier_manifest(
    corpus: Corpus,
    *,
    publish_lifecycle: bool,
    passed: bool = True,
) -> Implementation:
    """Attach a real Tier aggregate proof to the latest Handoff candidate."""

    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    tier = spec.tier_dag.tiers[0]
    assert len(tier.slices) == 1
    slice_ = tier.slices[0]
    implementation = next(
        corpus.get_record(ref, as_type=Implementation)
        for ref in reversed(spec.implementation_refs)
        if corpus.get_record(ref, as_type=Implementation).slice_id
        == slice_.id
    )
    outcome = (
        BuildVerifyOutcome(
            "success",
            build=RunnerResult(
                role="build",
                runner="tests.fake_build",
                returncode=0,
                values={},
                raw_output=b"{}",
                stderr=b"",
                duration_s=0.1,
            ),
            verify=None,
        )
        if passed
        else None
    )
    evidence_ref = persist_verification_evidence(
        corpus,
        tier_id=tier.tier_id,
        slice_id=slice_.id,
        spec_execution_authority_hash=(
            _partition.spec_execution_structure_hash(spec)
        ),
        candidate_diff_ref=implementation.diff_ref,
        base_commit=implementation.base_commit,
        files_modified=implementation.files_modified,
        outcome=outcome,
        skipped_by_policy=not passed,
        build_timeout_s=30.0,
        verify_timeout_s=30.0,
    )
    with_evidence = replace(
        implementation,
        artifacts={
            **(implementation.artifacts or {}),
            _gate_evidence.TIER_VERIFICATION_ARTIFACT: evidence_ref,
        },
    )
    manifest_ref = _gate_evidence.persist_tier_verification_manifest(
        corpus,
        tier_id=tier.tier_id,
        spec_execution_authority_hash=(
            _partition.spec_execution_structure_hash(spec)
        ),
        project_base_commit=spec.project_context.base_commit,
        candidate_diff_ref=implementation.diff_ref,
        base_commit=implementation.base_commit,
        files_modified=implementation.files_modified,
        gate="verify",
        obligation_slice_ids=(slice_.id,),
        current_implementations=(with_evidence,),
        dependency_implementations=(),
        verification_evidence_refs=(evidence_ref,),
    )
    final = replace(
        with_evidence,
        artifacts={
            **(with_evidence.artifacts or {}),
            _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT: manifest_ref,
        },
    )
    if publish_lifecycle:
        corpus.commit(
            implementations=(final,),
            lifecycle_event_factory=lambda version: LifecycleEvent(
                kind="landed",
                reason=f"tier {tier.tier_id}: complete",
                affected_versions=(version,),
                execution=exec_ctx_now("tier"),
            ),
        )
    else:
        corpus.commit(implementations=(final,))
    return final


def test_reconcile_atomically_lands_complete_plan_and_is_idempotent(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\ncomplete\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    latest = corpus.latest_spec()
    assert latest is not None
    assert summary["committed_version"] == latest.version
    event = corpus.get_record(
        latest.lifecycle_event_refs[-1], as_type=LifecycleEvent,
    )
    assert event.kind == "landed"
    assert event.affected_versions == (latest.version,)
    assert latest.verdict_refs[-1] == summary["validation"]["verdict_ref"]

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["status"] == "no_work"
    assert corpus.latest_spec().version == latest.version


def test_reconcile_does_not_land_missing_slice_evidence(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["eligible_to_land"] is False
    assert summary["tiers"][0]["pending_slices"] == ["s0"]
    assert corpus.latest_spec().lifecycle_event_refs == ()


@pytest.mark.parametrize(
    "evidence_state",
    ("waived", "unavailable", "missing", "stale"),
)
def test_reconcile_does_not_promote_unverified_handoff_candidate(
    tmp_path: Path,
    monkeypatch,
    capsys,
    evidence_state: str,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\ncomplete\n")
    handoff_args = [
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
    ]
    if evidence_state != "unavailable":
        handoff_args.append("--skip-verify")
    assert handoff_main(handoff_args) == 0
    capsys.readouterr()

    spec = corpus.latest_spec()
    assert spec is not None
    implementation = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    if evidence_state == "missing":
        artifacts = dict(implementation.artifacts or {})
        artifacts.pop(HANDOFF_VERIFICATION_ARTIFACT, None)
        corpus.commit(implementations=(replace(
            implementation,
            artifacts=artifacts,
        ),))
    elif evidence_state == "stale":
        stale_candidate_ref = corpus.put_artifact(
            b"stale candidate",
            kind="diffs",
            name_hint="stale.diff",
        )
        stale_evidence_ref = persist_verification_evidence(
            corpus,
            tier_id="t0",
            slice_id="s0",
            spec_execution_authority_hash=(
                _partition.spec_execution_structure_hash(spec)
            ),
            candidate_diff_ref=stale_candidate_ref,
            base_commit=implementation.base_commit,
            files_modified=implementation.files_modified,
            outcome=BuildVerifyOutcome(
                "success",
                build=RunnerResult(
                    role="build",
                    runner="tests.fake_build",
                    returncode=0,
                    values={},
                    raw_output=b'{"values":{}}',
                    stderr=b"",
                    duration_s=0.1,
                ),
                verify=None,
            ),
            skipped_by_policy=False,
            build_timeout_s=30.0,
            verify_timeout_s=30.0,
            name_hint="stale_reconcile_evidence",
        )
        corpus.commit(implementations=(replace(
            implementation,
            artifacts={
                **(implementation.artifacts or {}),
                HANDOFF_VERIFICATION_ARTIFACT: stale_evidence_ref,
            },
        ),))

    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        _REAL_VERIFICATION_MATCHES,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["eligible_to_land"] is False
    assert summary["tiers"][0]["pending_slices"] == []
    assert summary["tiers"][0]["verified_slices"] == []
    assert summary["tiers"][0]["verification_pending_slices"] == ["s0"]
    assert corpus.latest_spec().lifecycle_event_refs == ()


def test_reconcile_accepts_exact_required_pass_handoff_evidence(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\ncomplete\n")
    monkeypatch.setattr(
        "crucible.workflows.handoff.build_and_verify",
        lambda *args, **kwargs: BuildVerifyOutcome(
            "success",
            build=RunnerResult(
                role="build",
                runner="tests.fake_build",
                returncode=0,
                values={},
                raw_output=b'{}',
                stderr=b"",
                duration_s=0.1,
            ),
            verify=None,
        ),
    )
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
    ]) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        _REAL_VERIFICATION_MATCHES,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    assert summary["eligible_to_land"] is True
    assert summary["tiers"][0]["verified_slices"] == ["s0"]
    assert summary["tiers"][0]["verification_pending_slices"] == []
    assert corpus.latest_spec().lifecycle_event_refs


def test_reconcile_does_not_synthesize_missing_tier_lifecycle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\ncomplete\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    _attach_tier_manifest(corpus, publish_lifecycle=False)
    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        _REAL_VERIFICATION_MATCHES,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    row = summary["tiers"][0]

    assert summary["status"] == "incomplete"
    assert summary["eligible_to_land"] is False
    assert row["tier_manifest_verified"] is True
    assert row["verified_slices"] == []
    assert row["verification_pending_slices"] == ["s0"]
    assert row["lifecycle_accepted"] is False
    assert row["lifecycle_reason"] == "no tier lifecycle event is attached"
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.lifecycle_event_refs == ()


def test_reconcile_consumes_manifest_only_with_exact_tier_lifecycle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\ncomplete\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    _attach_tier_manifest(corpus, publish_lifecycle=True)
    before = corpus.latest_spec()
    assert before is not None
    before_lifecycle_refs = before.lifecycle_event_refs
    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        _REAL_VERIFICATION_MATCHES,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    row = summary["tiers"][0]

    assert summary["status"] == "no_work"
    assert summary["eligible_to_land"] is True
    assert row["tier_manifest_verified"] is True
    assert row["verified_slices"] == ["s0"]
    assert row["verification_pending_slices"] == []
    assert row["lifecycle_accepted"] is True
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.lifecycle_event_refs == before_lifecycle_refs


def test_reconcile_rejects_waived_tier_manifest_with_lifecycle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\ncomplete\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    _attach_tier_manifest(
        corpus,
        publish_lifecycle=True,
        passed=False,
    )
    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        _REAL_VERIFICATION_MATCHES,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: (),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    row = summary["tiers"][0]

    assert summary["status"] == "incomplete"
    assert summary["eligible_to_land"] is False
    assert row["tier_manifest_verified"] is False
    assert row["verified_slices"] == []
    assert row["verification_pending_slices"] == ["s0"]
    assert row["lifecycle_accepted"] is True


@pytest.mark.parametrize("base_commit_length", [None, 8])
def test_reconcile_allows_create_path_from_current_bound_success(
    tmp_path: Path, capsys, base_commit_length: int | None,
) -> None:
    repo, spec_dir, corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
        base_commit_length=base_commit_length,
    )
    (repo / "generated.txt").write_text("generated\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    assert summary["validation"]["decision"] == "pass"
    assert corpus.latest_spec().lifecycle_event_refs


def test_reconcile_does_not_exempt_success_bound_to_stale_plan(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    (repo / "generated.txt").write_text("generated\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()

    implemented_spec = corpus.latest_spec()
    assert implemented_spec is not None
    assert implemented_spec.tier_dag is not None
    old_tier = implemented_spec.tier_dag.tiers[0]
    old_slice = old_tier.slices[0]
    changed_directive = replace(
        old_slice.directive,
        intent="create generated.txt under a changed current plan",
    )
    changed_tier = replace(
        old_tier,
        slices=(replace(old_slice, directive=changed_directive),),
    )
    changed_spec = implemented_spec.with_tier_dag(
        TierDAG(tiers=(changed_tier,)),
    )
    corpus.put_spec(changed_spec)
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["eligible_to_land"] is False
    assert summary["tiers"][0]["pending_slices"] == ["s0"]
    assert summary["validation"]["decision"] == "reject"


def test_reconcile_does_not_launder_unmodified_preexisting_create_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt", "seed.txt"),
    )
    (repo / "generated.txt").write_text("generated\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["targets"][0]["files_modified"] == ["generated.txt"]
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"
    assert summary["validation"]["finding_count"] == 1


def test_reconcile_does_not_launder_modified_preexisting_create_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("seed.txt",),
    )
    (repo / "seed.txt").write_text("seed\nmodified\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["targets"][0]["files_modified"] == ["seed.txt"]
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"
    assert summary["validation"]["finding_count"] == 1


def test_reconcile_rejects_deleted_post_implementation_create_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    created = repo / "generated.txt"
    created.write_text("generated\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    created.unlink()
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"
    assert summary["validation"]["finding_count"] == 1


def test_reconcile_create_authority_requires_current_head_match(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    (repo / "generated.txt").write_text("generated\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    subprocess.run(["git", "add", "generated.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "advance checkout"],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


def test_reconcile_rejects_replaced_create_path_bytes(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    created = repo / "generated.txt"
    created.write_bytes(b"implementation postimage\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    created.write_bytes(b"tampered_______postimage\n")
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"
    assert summary["validation"]["finding_count"] == 1


@pytest.mark.parametrize("replacement", ["executable", "directory", "symlink"])
def test_reconcile_rejects_create_path_mode_or_type_replacement(
    tmp_path: Path, monkeypatch, capsys, replacement: str,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    created = repo / "generated.txt"
    created.write_text("implementation postimage\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    if replacement == "executable":
        created.chmod(created.stat().st_mode | stat.S_IXUSR)
    elif replacement == "directory":
        created.unlink()
        created.mkdir()
    else:
        created.unlink()
        created.symlink_to("seed.txt")
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


@pytest.mark.parametrize("created_kind", ["executable", "broken_symlink"])
def test_reconcile_accepts_evidenced_create_path_mode_or_type(
    tmp_path: Path, capsys, created_kind: str,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    created = repo / "generated.txt"
    if created_kind == "executable":
        created.write_text("#!/bin/sh\nexit 0\n")
        created.chmod(created.stat().st_mode | stat.S_IXUSR)
    else:
        created.symlink_to("target-that-does-not-exist")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--all",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    assert summary["validation"]["decision"] == "pass"


@pytest.mark.parametrize("restore_creator_postimage", [False, True])
def test_reconcile_uses_latest_legitimate_downstream_postimage(
    tmp_path: Path, capsys, restore_creator_postimage: bool,
) -> None:
    repo, spec_dir, corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    planned = corpus.latest_spec()
    assert planned is not None
    assert planned.tier_dag is not None
    creator_tier = planned.tier_dag.tiers[0]
    downstream = Slice(
        id="s1",
        directive=ImplementationDirective(
            slice_id="s1",
            intent="append the downstream edit to generated.txt",
            files_in_scope=("generated.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    corpus.put_spec(planned.with_tier_dag(TierDAG(tiers=(
        creator_tier,
        Tier(
            tier_id="t1",
            depends_on=(creator_tier.tier_id,),
            slices=(downstream,),
        ),
    ))))

    created = repo / "generated.txt"
    created.write_text("creator postimage\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s0",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    created.write_text("creator postimage\ndownstream postimage\n")
    assert handoff_main([
        "--spec", str(spec_dir),
        "--slice", "s1",
        "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    if restore_creator_postimage:
        created.write_text("creator postimage\n")

    rc = reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ])
    summary = json.loads(capsys.readouterr().out)
    if restore_creator_postimage:
        assert rc == 1
        assert summary["status"] == "incomplete"
        assert summary["validation"]["decision"] == "reject"
    else:
        assert rc == 0
        assert summary["status"] == "landed"
        assert summary["validation"]["decision"] == "pass"


@pytest.mark.parametrize("drift", [False, True])
def test_reconcile_compares_binary_postimage_bytes(
    tmp_path: Path, monkeypatch, capsys, drift: bool,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.bin",),
    )
    binary = repo / "generated.bin"
    binary.write_bytes(bytes(range(256)) * 4)
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    if drift:
        binary.write_bytes(bytes(reversed(range(256))) * 4)
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    rc = reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ])
    summary = json.loads(capsys.readouterr().out)
    assert rc == (1 if drift else 0)
    assert summary["status"] == ("incomplete" if drift else "landed")


def test_workspace_postimage_comparison_never_executes_clean_filter(
    tmp_path: Path,
) -> None:
    from crucible.workflows.reconcile import (
        _GitPathState,
        _workspace_matches_postimage,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    payload = b"physical postimage\n"
    (repo / "payload.txt").write_bytes(payload)
    (repo / ".gitattributes").write_text(
        "payload.txt filter=hostile\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "filter-executed"
    filter_script = tmp_path / "clean-filter.sh"
    filter_script.write_text(
        "#!/bin/sh\ntouch \"$1\"\ncat\n",
        encoding="utf-8",
    )
    filter_script.chmod(0o755)
    subprocess.run(
        [
            "git", "config", "filter.hostile.clean",
            f"{filter_script} {sentinel}",
        ],
        cwd=repo,
        check=True,
    )
    object_id = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=repo,
        input=payload,
        capture_output=True,
        check=True,
    ).stdout.decode().strip()

    assert _workspace_matches_postimage(
        repo,
        "payload.txt",
        _GitPathState(
            mode="100644",
            object_id=object_id,
            content=payload,
        ),
    )
    assert not sentinel.exists()

    (repo / "payload.txt").write_bytes(b"different physical bytes\n")
    assert not _workspace_matches_postimage(
        repo,
        "payload.txt",
        _GitPathState(
            mode="100644",
            object_id=object_id,
            content=payload,
        ),
    )
    assert not sentinel.exists()


@pytest.mark.parametrize("reappear", [False, True])
def test_reconcile_verifies_planned_deletion_and_reappearance(
    tmp_path: Path, monkeypatch, capsys, reappear: bool,
) -> None:
    repo, spec_dir, _corpus = _setup(tmp_path)
    (repo / "seed.txt").unlink()
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    if reappear:
        (repo / "seed.txt").write_text("unrecorded reappearance\n")
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    rc = reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ])
    summary = json.loads(capsys.readouterr().out)
    assert rc == (1 if reappear else 0)
    assert summary["status"] == ("incomplete" if reappear else "landed")


def test_reconcile_allows_verified_recursive_scope_root_deletion(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo, spec_dir, corpus = _setup(
        tmp_path,
        initial_files={
            "obsolete/first.txt": "first\n",
            "obsolete/nested/second.txt": "second\n",
        },
    )
    planned = corpus.latest_spec()
    assert planned is not None
    assert planned.tier_dag is not None
    tier = planned.tier_dag.tiers[0]
    slice_ = tier.slices[0]
    recursive_delete = replace(
        slice_.directive,
        intent="delete the obsolete directory recursively",
        files_in_scope=("obsolete",),
        blast_radius={"max_files": 2},
    )
    corpus.put_spec(planned.with_tier_dag(TierDAG(tiers=(replace(
        tier,
        slices=(replace(slice_, directive=recursive_delete),),
    ),))))
    (repo / "obsolete" / "first.txt").unlink()
    (repo / "obsolete" / "nested" / "second.txt").unlink()
    (repo / "obsolete" / "nested").rmdir()
    (repo / "obsolete").rmdir()
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert set(handoff["targets"][0]["files_modified"]) == {
        "obsolete/first.txt",
        "obsolete/nested/second.txt",
    }
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)

    assert summary["status"] == "landed"
    assert summary["validation"]["finding_count"] == 0
    assert summary["workspace_evidence"] == {
        "passed": True,
        "expected_path_count": 2,
        "verified_path_count": 2,
        "required_create_path_count": 0,
        "verified_create_path_count": 0,
        "error_count": 0,
    }


def test_reconcile_rejects_files_in_scope_byte_drift(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(tmp_path)
    seed = repo / "seed.txt"
    seed.write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    seed.write_text("seed\ntampered___\n")
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


def test_reconcile_rejects_files_in_scope_head_base_drift(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "advance edit checkout"],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


@pytest.mark.parametrize("tamper_untouched_child", [False, True])
def test_reconcile_expands_recursive_directory_scope(
    tmp_path: Path, monkeypatch, capsys, tamper_untouched_child: bool,
) -> None:
    repo, spec_dir, corpus = _setup(
        tmp_path,
        initial_files={
            "src/changed.txt": "base changed\n",
            "src/untouched.txt": "base untouched\n",
        },
    )
    planned = corpus.latest_spec()
    assert planned is not None
    assert planned.tier_dag is not None
    tier = planned.tier_dag.tiers[0]
    slice_ = tier.slices[0]
    directory_directive = replace(
        slice_.directive,
        intent="update files recursively under src/",
        files_in_scope=("src/",),
    )
    corpus.put_spec(planned.with_tier_dag(TierDAG(tiers=(replace(
        tier,
        slices=(replace(slice_, directive=directory_directive),),
    ),))))
    (repo / "src" / "changed.txt").write_text("implemented change\n")
    (repo / "src" / "created.txt").write_text("implemented create\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert set(handoff["targets"][0]["files_modified"]) == {
        "src/changed.txt",
        "src/created.txt",
    }
    if tamper_untouched_child:
        (repo / "src" / "untouched.txt").write_text("unrecorded drift\n")
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    rc = reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ])
    summary = json.loads(capsys.readouterr().out)
    assert rc == (1 if tamper_untouched_child else 0)
    assert summary["status"] == (
        "incomplete" if tamper_untouched_child else "landed"
    )


def test_reconcile_rejects_incomparable_path_writers(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    planned = corpus.latest_spec()
    assert planned is not None
    assert planned.tier_dag is not None
    creator = planned.tier_dag.tiers[0]

    def edit_slice(slice_id: str) -> Slice:
        return Slice(
            id=slice_id,
            directive=ImplementationDirective(
                slice_id=slice_id,
                intent=f"edit generated.txt from {slice_id}",
                files_in_scope=("generated.txt",),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )

    corpus.put_spec(planned.with_tier_dag(TierDAG(tiers=(
        creator,
        Tier(
            tier_id="left",
            depends_on=(creator.tier_id,),
            slices=(edit_slice("left-slice"),),
        ),
        Tier(
            tier_id="right",
            depends_on=(creator.tier_id,),
            slices=(edit_slice("right-slice"),),
        ),
    ))))
    generated = repo / "generated.txt"
    for slice_id, content in (
        ("s0", "creator\n"),
        ("left-slice", "creator\nleft\n"),
        ("right-slice", "creator\nleft\nright\n"),
    ):
        generated.write_text(content)
        assert handoff_main([
            "--spec", str(spec_dir), "--slice", slice_id,
            "--cwd", str(repo), "--skip-verify",
        ]) == 0
        capsys.readouterr()
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


def test_reconcile_accepts_in_scope_git_rename_evidence(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    planned = corpus.latest_spec()
    assert planned is not None
    assert planned.tier_dag is not None
    tier = planned.tier_dag.tiers[0]
    slice_ = tier.slices[0]
    rename_directive = replace(
        slice_.directive,
        intent="rename seed.txt to renamed.txt",
        creates_paths=("renamed.txt",),
    )
    corpus.put_spec(planned.with_tier_dag(TierDAG(tiers=(replace(
        tier,
        slices=(replace(slice_, directive=rename_directive),),
    ),))))
    subprocess.run(
        ["git", "mv", "seed.txt", "renamed.txt"],
        cwd=repo,
        check=True,
    )
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    handoff = json.loads(capsys.readouterr().out)
    assert handoff["targets"][0]["files_modified"] == ["renamed.txt"]
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    assert summary["workspace_evidence"]["passed"] is True


def test_reconcile_compares_git_cleaned_worktree_content(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(
        tmp_path,
        creates_paths=("generated.txt",),
    )
    subprocess.run(
        ["git", "config", "core.autocrlf", "true"],
        cwd=repo,
        check=True,
    )
    (repo / "generated.txt").write_bytes(b"line one\r\nline two\r\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    assert summary["workspace_evidence"]["passed"] is True


def test_reconcile_rejects_diff_files_modified_disagreement(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()

    spec = corpus.latest_spec()
    assert spec is not None
    assert spec.tier_dag is not None
    original = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    inconsistent = replace(original, files_modified=())
    tier = spec.tier_dag.tiers[0]
    inconsistent = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        tier.slices[0],
        inconsistent,
    )
    persist_and_attach(
        corpus,
        inconsistent,
        kind="implementations",
        attachment="implementation",
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


@pytest.mark.parametrize(
    "tamper",
    ["corrupt_diff", "outside_scope", "scope_escape"],
)
def test_reconcile_rejects_invalid_or_out_of_scope_diff_evidence(
    tmp_path: Path, monkeypatch, capsys, tamper: str,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()

    spec = corpus.latest_spec()
    assert spec is not None
    assert spec.tier_dag is not None
    original = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    if tamper == "scope_escape":
        inconsistent = replace(
            original,
            scope_escapes=("../escaped",),
        )
    elif tamper == "corrupt_diff":
        diff_ref = corpus.put_artifact(
            b"not a unified diff\n",
            kind="diffs",
            name_hint="corrupt.diff",
        )
        inconsistent = replace(original, diff_ref=diff_ref)
    else:
        (repo / "outside.txt").write_text("outside scope\n")
        diff_bytes, files_modified, base_commit = capture_workspace_delta(
            cwd=str(repo),
            path_scope=("outside.txt",),
        )
        diff_ref = corpus.put_artifact(
            diff_bytes,
            kind="diffs",
            name_hint="outside.diff",
        )
        inconsistent = replace(
            original,
            base_commit=base_commit,
            diff_ref=diff_ref,
            files_modified=files_modified,
        )
    tier = spec.tier_dag.tiers[0]
    inconsistent = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        tier.slices[0],
        inconsistent,
    )
    persist_and_attach(
        corpus,
        inconsistent,
        kind="implementations",
        attachment="implementation",
    )
    monkeypatch.setattr(
        "crucible.workflows.reconcile.available_rules",
        lambda: ("cited_path_exists",),
    )

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "incomplete"
    assert summary["validation"]["decision"] == "reject"


def test_reconcile_ignores_outside_dirt_but_preserves_real_index(
    tmp_path: Path, capsys,
) -> None:
    repo, spec_dir, _corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    (repo / "outside.txt").write_text("unrelated dirt\n")
    index_path = Path(subprocess.check_output(
        ["git", "rev-parse", "--git-path", "index"],
        cwd=repo,
        text=True,
    ).strip())
    if not index_path.is_absolute():
        index_path = repo / index_path
    index_before = index_path.read_bytes()

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "landed"
    assert index_path.read_bytes() == index_before


def test_reconcile_holds_workspace_lock_through_validate_and_commit(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    import crucible.workflows.reconcile as reconcile_workflow

    repo, spec_dir, _corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    state = {"locked": False, "validated": False, "committed": False}

    class ObservedLock:
        def __init__(self) -> None:
            self.entered = False

        def __enter__(self):
            assert state["locked"] is False
            state["locked"] = True
            self.entered = True

        def retry_cleanup(self) -> None:
            if self.entered:
                self.entered = False
                state["locked"] = False

    def observed_lock(_corpus, _cwd):
        return ObservedLock()

    original_validate = reconcile_workflow.validate
    original_commit = Corpus.commit

    def observed_validate(*args, **kwargs):
        assert state["locked"] is True
        state["validated"] = True
        return original_validate(*args, **kwargs)

    def observed_commit(self, *args, **kwargs):
        assert state["locked"] is True
        state["committed"] = True
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(
        reconcile_workflow,
        "recovery_workspace_mutation_lock",
        observed_lock,
    )
    monkeypatch.setattr(reconcile_workflow, "validate", observed_validate)
    monkeypatch.setattr(Corpus, "commit", observed_commit)

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    capsys.readouterr()
    assert state == {"locked": False, "validated": True, "committed": True}


def test_reconcile_cas_rejects_concurrent_spec_change(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    original_commit = Corpus.commit
    injected = False

    def racing_commit(self, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            latest = self.latest_spec()
            assert latest is not None
            assert latest.tier_dag is not None
            pending = Slice(
                id="concurrent-slice",
                directive=ImplementationDirective(
                    slice_id="concurrent-slice",
                    intent="concurrently edit seed.txt",
                    files_in_scope=("seed.txt",),
                    creates_paths=(),
                    verify_mode="skip",
                    blast_radius={"max_files": 1},
                ),
            )
            self.put_spec(latest.with_tier_dag(TierDAG(tiers=(
                *latest.tier_dag.tiers,
                Tier(
                    tier_id="concurrent-tier",
                    depends_on=(latest.tier_dag.tiers[-1].tier_id,),
                    slices=(pending,),
                ),
            ))))
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(Corpus, "commit", racing_commit)

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "concurrent_spec_change"
    assert summary["lifecycle_ref"] is None
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.tier_dag is not None
    assert latest.tier_dag.tiers[-1].tier_id == "concurrent-tier"
    assert latest.lifecycle_event_refs == ()
    reconcile_run = sorted(spec_dir.joinpath("runs").glob("*reconcile*"))[-1]
    outcome = json.loads((reconcile_run / "outcome.json").read_text())
    assert outcome["terminal_class"] == "retryable_failure"


def test_reconcile_no_work_rechecks_latest_spec_atomically(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "landed"

    original_commit = Corpus.commit
    injected = False

    def racing_no_work_commit(self, *args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            latest = self.latest_spec()
            assert latest is not None
            assert latest.tier_dag is not None
            pending = Slice(
                id="late-slice",
                directive=ImplementationDirective(
                    slice_id="late-slice",
                    intent="late concurrent edit of seed.txt",
                    files_in_scope=("seed.txt",),
                    creates_paths=(),
                    verify_mode="skip",
                    blast_radius={"max_files": 1},
                ),
            )
            self.put_spec(latest.with_tier_dag(TierDAG(tiers=(
                *latest.tier_dag.tiers,
                Tier(
                    tier_id="late-tier",
                    depends_on=(latest.tier_dag.tiers[-1].tier_id,),
                    slices=(pending,),
                ),
            ))))
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(Corpus, "commit", racing_no_work_commit)

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "concurrent_spec_change"
    assert summary["publication_error"] is not None
    assert corpus.latest_spec().tier_dag.tiers[-1].tier_id == "late-tier"


def test_reconcile_parent_cas_uses_persisted_raw_spec_digest(
    tmp_path: Path,
) -> None:
    import hashlib

    from assay import ContentHash, PlanPolicy
    from assay.canonical import hash_record, parse, serialize
    from crucible.workflows import reconcile as reconcile_workflow

    root = replace(
        Spec.from_yaml(_SPEC),
        plan_policy=PlanPolicy(),
    )
    legacy_payload = parse(serialize(root))
    del legacy_payload["plan_policy"][
        "max_verification_only_empty_scope_slices"
    ]
    legacy_blob = serialize(legacy_payload)
    legacy_digest = "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    legacy_corpus = Corpus(tmp_path / "legacy-corpus")
    versions = Path(legacy_corpus.spec_dir) / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    (versions / f"v0_{legacy_digest.split(':', 1)[1][:12]}.json").write_bytes(
        legacy_blob
    )

    hydrated = legacy_corpus.latest_spec()
    assert hydrated == root
    assert str(hash_record(hydrated)) != legacy_digest
    candidate = replace(
        hydrated,
        version=hydrated.version + 1,
        parent=ContentHash(legacy_digest),
    )
    transform = reconcile_workflow._expected_spec_transform(
        hydrated,
        legacy_digest,
    )

    assert transform(candidate) is candidate
    with pytest.raises(
        reconcile_workflow._SpecSnapshotChanged,
        match="latest Spec changed",
    ):
        transform(replace(
            candidate,
            parent=ContentHash(hash_record(hydrated)),
        ))


def test_reconcile_no_work_rejects_persisted_digest_drift(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()
    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "landed"
    before = corpus.latest_spec()
    assert before is not None
    before_refs = (before.lifecycle_event_refs, before.verdict_refs)
    original_get = Corpus.get_spec_with_digest
    original_commit = Corpus.commit
    no_work_commit_returned = False
    drift_returned = False

    def observe_no_work_commit(self, *args, **kwargs):
        nonlocal no_work_commit_returned
        result = original_commit(self, *args, **kwargs)
        if self.spec_dir == corpus.spec_dir and not args and not kwargs:
            no_work_commit_returned = True
        return result

    def drifting_digest(self, version):
        nonlocal drift_returned
        observed, digest = original_get(self, version)
        if (
            self.spec_dir == corpus.spec_dir
            and version == before.version
            and no_work_commit_returned
            and not drift_returned
        ):
            drift_returned = True
            return observed, "sha256:" + "f" * 64
        return observed, digest

    monkeypatch.setattr(Corpus, "commit", observe_no_work_commit)
    monkeypatch.setattr(Corpus, "get_spec_with_digest", drifting_digest)

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(repo),
    ]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "concurrent_spec_change"
    assert summary["publication_error"] is not None
    after = corpus.latest_spec()
    assert after is not None
    assert (after.lifecycle_event_refs, after.verdict_refs) == before_refs
    assert no_work_commit_returned is True
    assert drift_returned is True


@pytest.mark.parametrize("submodule_path", ["../repo", "linked-repo"])
def test_reconcile_rejects_submodule_workspace_escape(
    tmp_path: Path, capsys, submodule_path: str,
) -> None:
    repo, spec_dir, corpus = _setup(tmp_path)
    (repo / "seed.txt").write_text("seed\nimplemented\n")
    assert handoff_main([
        "--spec", str(spec_dir), "--all", "--cwd", str(repo),
        "--skip-verify",
    ]) == 0
    capsys.readouterr()

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.tier_dag is not None
    bumped = latest.with_tier_dag(latest.tier_dag)
    corpus.put_spec(replace(
        bumped,
        project_context=replace(
            bumped.project_context,
            submodule_path=submodule_path,
        ),
    ))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if submodule_path == "linked-repo":
        (workspace / submodule_path).symlink_to(repo, target_is_directory=True)

    assert reconcile_main([
        "--spec", str(spec_dir), "--cwd", str(workspace),
    ]) == 2
    assert corpus.latest_spec().lifecycle_event_refs == ()


def test_reconcile_postimage_ignores_repository_replace_refs(tmp_path: Path) -> None:
    repo, _spec_dir, corpus = _setup(tmp_path)
    spec = corpus.latest_spec()
    assert spec is not None
    base_commit = spec.project_context.base_commit

    (repo / "seed.txt").write_text("replacement\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "replacement candidate"],
        cwd=repo,
        check=True,
    )
    replacement_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    subprocess.run(
        ["git", "replace", base_commit, replacement_commit],
        cwd=repo,
        check=True,
    )
    assert subprocess.check_output(
        ["git", "show", f"{base_commit}:seed.txt"], cwd=repo
    ) == b"replacement\n"

    reconstructed = reconcile._implementation_postimages(
        repo,
        base_commit,
        None,
        {"seed.txt"},
    )

    assert reconstructed is not None
    postimages, touched, renames = reconstructed
    assert postimages["seed.txt"].content == b"seed\n"
    assert touched == frozenset()
    assert renames == frozenset()
