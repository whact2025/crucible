# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from assay import (
    Implementation,
    ImplementationDirective,
    LifecycleEvent,
    ProjectContext,
    RuleContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
    Verdict,
)
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.primitives.implement import capture_workspace_delta

from crucible.rules.cited_path_exists import cited_path_exists
from crucible.rules.scope_covers_intent import scope_covers_intent
from crucible.workflows import _partition
from crucible.workflows._run import exec_ctx_now
from crucible.workflows._workspace_evidence import (
    reconstruct_plan_workspace_evidence,
)
from crucible.workflows.tier import tier_workspace_matches_plan_implementations


def _setup_chained_create_and_edit(
    tmp_path: Path,
) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "workspace-evidence@invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Workspace Evidence"],
        cwd=repo,
        check=True,
    )
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "seed.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    create = ImplementationDirective(
        slice_id="create-x",
        intent="Create x.txt.",
        files_in_scope=(),
        creates_paths=("x.txt",),
        verify_mode="skip",
        blast_radius={"max_files": 1},
    )
    edit = ImplementationDirective(
        slice_id="edit-x",
        intent="Edit the x.txt created by the predecessor.",
        files_in_scope=("x.txt",),
        creates_paths=(),
        verify_mode="skip",
        blast_radius={"max_files": 1},
    )
    dag = TierDAG(tiers=(
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(Slice(id="create-x", directive=create),),
        ),
        Tier(
            tier_id="t1",
            depends_on=("t0",),
            slices=(Slice(id="edit-x", directive=edit),),
        ),
    ))
    base = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="workspace-evidence-test",
            repo_url="https://example.invalid/workspace-evidence.git",
            base_commit=base_commit,
            branch="main",
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Prove a retained create followed by a retained edit.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(base)
    corpus.put_spec(base.with_tier_dag(dag))
    return repo, corpus


def _setup_recursive_deletion(tmp_path: Path) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "workspace-evidence@invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Workspace Evidence"],
        cwd=repo,
        check=True,
    )
    obsolete = repo / "obsolete"
    obsolete.mkdir()
    obsolete.joinpath("first.txt").write_text("first\n", encoding="utf-8")
    obsolete.joinpath("nested").mkdir()
    obsolete.joinpath("nested", "second.txt").write_text(
        "second\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()

    directive = ImplementationDirective(
        slice_id="delete-obsolete",
        intent="Delete the obsolete directory recursively.",
        files_in_scope=("obsolete",),
        creates_paths=(),
        verify_mode="skip",
        blast_radius={"max_files": 2},
    )
    dag = TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(Slice(id="delete-obsolete", directive=directive),),
    ),))
    base = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="workspace-evidence-test",
            repo_url="https://example.invalid/workspace-evidence.git",
            base_commit=base_commit,
            branch="main",
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Prove a recursive directory deletion.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(base)
    corpus.put_spec(base.with_tier_dag(dag))
    return repo, corpus


def _land_current_workspace(
    repo: Path,
    corpus: Corpus,
    *,
    tier_id: str,
    publish_lifecycle: bool = True,
    implementation_recipe: str = "test_workspace_evidence",
    slice_id: str | None = None,
    classification: str = "success",
    scope_escapes: tuple[str, ...] = (),
) -> None:
    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    tier = next(item for item in spec.tier_dag.tiers if item.tier_id == tier_id)
    slice_ = (
        tier.slices[0]
        if slice_id is None
        else next(item for item in tier.slices if item.id == slice_id)
    )
    scope = (
        *slice_.directive.files_in_scope,
        *slice_.directive.creates_paths,
    )
    diff, files, base_commit = capture_workspace_delta(
        cwd=str(repo),
        path_scope=scope,
    )
    implementation = Implementation(
        slice_id=slice_.id,
        base_commit=base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint=f"{tier_id}_{slice_.id}.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint=f"{tier_id}_{slice_.id}.json",
        ),
        files_modified=files,
        classification=classification,
        execution=exec_ctx_now(implementation_recipe),
        scope_escapes=scope_escapes,
    )
    implementation = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        slice_,
        implementation,
    )
    corpus.commit(implementations=(implementation,))
    if not publish_lifecycle:
        return
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=f"tier {tier_id}: complete",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )


def _attach_tier_manifest(
    corpus: Corpus,
    *,
    tier_id: str,
    publish_lifecycle: bool,
    passed: bool = True,
) -> None:
    """Attach one real Tier manifest to the latest single-slice candidate."""

    from dataclasses import replace

    from crucible.workflows import _gate_evidence
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    tier = next(item for item in spec.tier_dag.tiers if item.tier_id == tier_id)
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
    evidence_ref = _gate_evidence.persist_verification_evidence(
        corpus,
        tier_id=tier_id,
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
        tier_id=tier_id,
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
                reason=f"tier {tier_id}: complete",
                affected_versions=(version,),
                execution=exec_ctx_now("tier"),
            ),
        )
    else:
        corpus.commit(implementations=(final,))


def test_handoff_without_required_pass_evidence_is_retained_but_not_accepted(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)

    (repo / "x.txt").write_text("created by handoff\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        implementation_recipe="handoff",
    )

    spec = corpus.latest_spec()
    assert spec is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    row = evidence.rows[0]

    # The exact candidate bytes remain usable as retained workspace evidence,
    # but a missing/waived gate cannot suppress Architect's deterministic
    # replay by joining the accepted prefix.
    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ()
    assert row["landed_slices"] == ["create-x"]
    assert row["verified_slices"] == []
    assert row["verification_pending_slices"] == ["create-x"]
    assert row["verification_complete"] is False
    assert row["lifecycle_accepted"] is True


def test_handoff_with_exact_required_pass_evidence_can_join_accepted_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _gate_evidence

    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by handoff\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        implementation_recipe="handoff",
    )
    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        lambda *args, **kwargs: True,
    )

    spec = corpus.latest_spec()
    assert spec is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    row = evidence.rows[0]

    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ("t0",)
    assert row["verified_slices"] == ["create-x"]
    assert row["verification_pending_slices"] == []
    assert row["verification_complete"] is True


def test_passed_tier_manifest_still_requires_exact_tier_lifecycle(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by handoff\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        publish_lifecycle=False,
        implementation_recipe="handoff",
    )
    _attach_tier_manifest(
        corpus,
        tier_id="t0",
        publish_lifecycle=False,
    )

    spec = corpus.latest_spec()
    assert spec is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    row = evidence.rows[0]

    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ()
    assert row["verified_slices"] == ["create-x"]
    assert row["verification_pending_slices"] == []
    assert row["tier_manifest_obligations"] == ["create-x"]
    assert row["tier_manifest_verified"] is True
    assert row["lifecycle_accepted"] is False
    assert row["lifecycle_reason"] == "no tier lifecycle event is attached"


def test_passed_tier_manifest_with_exact_lifecycle_joins_accepted_prefix(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by handoff\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        publish_lifecycle=False,
        implementation_recipe="handoff",
    )
    _attach_tier_manifest(
        corpus,
        tier_id="t0",
        publish_lifecycle=True,
    )

    spec = corpus.latest_spec()
    assert spec is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    row = evidence.rows[0]

    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ("t0",)
    assert row["tier_manifest_verified"] is True
    assert row["verification_complete"] is True
    assert row["lifecycle_accepted"] is True


def test_unpublished_validation_draft_preserves_accepted_prefix(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by handoff\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        publish_lifecycle=False,
        implementation_recipe="handoff",
    )
    _attach_tier_manifest(
        corpus,
        tier_id="t0",
        publish_lifecycle=True,
    )
    persisted = corpus.latest_spec()
    assert persisted is not None
    staged = persisted.with_appended_refs()
    assert corpus.latest_spec() == persisted

    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=staged,
        source_root=repo,
    )

    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ("t0",)
    assert evidence.rows[0]["lifecycle_accepted"] is True


def test_waived_tier_manifest_never_joins_accepted_prefix(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by handoff\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        publish_lifecycle=False,
        implementation_recipe="handoff",
    )
    _attach_tier_manifest(
        corpus,
        tier_id="t0",
        publish_lifecycle=True,
        passed=False,
    )

    spec = corpus.latest_spec()
    assert spec is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    row = evidence.rows[0]

    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ()
    assert row["tier_manifest_verified"] is False
    assert row["verification_pending_slices"] == ["create-x"]
    assert row["lifecycle_accepted"] is True


def test_landed_prefix_uses_dependency_composed_final_postimage(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)

    (repo / "x.txt").write_text("created by t0\n", encoding="utf-8")
    _land_current_workspace(repo, corpus, tier_id="t0")
    (repo / "x.txt").write_text(
        "created by t0\nedited by t1\n",
        encoding="utf-8",
    )
    _land_current_workspace(repo, corpus, tier_id="t1")

    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    create_key = ("t0", "create-x", "x.txt")

    assert evidence.errors == ()
    assert evidence.workspace_evidence_passed is True
    assert evidence.all_slices_complete is True
    assert evidence.retained_tier_ids == ("t0", "t1")
    assert evidence.accepted_tier_ids == ("t0", "t1")
    assert evidence.required_create_paths == frozenset({create_key})
    assert evidence.context.verified_create_paths == frozenset({create_key})
    assert evidence.context.expected_plan_surface_paths == frozenset({"x.txt"})
    assert evidence.context.verified_plan_surface_paths == frozenset({"x.txt"})
    assert not [
        finding
        for finding in cited_path_exists(spec, evidence.context)
        if finding.category == "overwrite_collision"
    ]

    t0, t1 = spec.tier_dag.topo_order()
    assert not tier_workspace_matches_plan_implementations(
        corpus=corpus,
        spec=spec,
        tier=t0,
        cwd=str(repo),
    )
    assert tier_workspace_matches_plan_implementations(
        corpus=corpus,
        spec=spec,
        tier=t1,
        cwd=str(repo),
    )

    (repo / "x.txt").write_text("tampered\n", encoding="utf-8")
    tampered = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )

    assert tampered.workspace_evidence_passed is False
    assert tampered.retained_tier_ids == ("t0", "t1")
    assert tampered.accepted_tier_ids == ("t0", "t1")
    assert tampered.context.verified_create_paths == frozenset()
    assert tampered.context.expected_plan_surface_paths == frozenset({"x.txt"})
    assert tampered.context.verified_plan_surface_paths == frozenset()
    assert any(
        finding.category == "implementation_postimage_mismatch"
        for finding in cited_path_exists(spec, tampered.context)
    )


def test_recursive_scope_root_inherits_verified_descendant_deletions(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_recursive_deletion(tmp_path)
    (repo / "obsolete" / "first.txt").unlink()
    (repo / "obsolete" / "nested" / "second.txt").unlink()
    (repo / "obsolete" / "nested").rmdir()
    (repo / "obsolete").rmdir()
    _land_current_workspace(repo, corpus, tier_id="t0")
    spec = corpus.latest_spec()
    assert spec is not None

    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    descendants = frozenset({
        "obsolete/first.txt",
        "obsolete/nested/second.txt",
    })

    assert evidence.workspace_evidence_passed is True
    assert evidence.context.expected_plan_surface_paths == descendants
    assert evidence.context.verified_plan_surface_paths == descendants
    assert evidence.context.verified_recursive_scope_paths == frozenset({
        ("t0", "delete-obsolete", "obsolete"),
    })
    assert evidence.context.planned_deletion_paths == descendants
    assert cited_path_exists(spec, evidence.context) == ()


def test_superseded_retained_predecessor_preserves_later_landed_evidence(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)

    (repo / "x.txt").write_text("created by t0\n", encoding="utf-8")
    _land_current_workspace(repo, corpus, tier_id="t0")
    (repo / "x.txt").write_text(
        "created by t0\nedited by t1\n",
        encoding="utf-8",
    )
    _land_current_workspace(repo, corpus, tier_id="t1")

    # A handoff/review correction supersedes t0's exact retained record, but
    # has not yet published the replacement t0 lifecycle.  The historical t1
    # landing remains sound evidence for its unchanged exact record.  It must
    # not become accepted ahead of t0, but neither should it make the retained
    # dependency-composed workspace look corrupt.
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        publish_lifecycle=False,
    )

    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    rows = {row["tier_id"]: row for row in evidence.rows}

    assert evidence.errors == ()
    assert evidence.workspace_evidence_passed is True
    assert evidence.all_slices_complete is True
    assert evidence.retained_tier_ids == ("t0", "t1")
    assert evidence.accepted_tier_ids == ()
    assert evidence.context.expected_plan_surface_paths == frozenset({"x.txt"})
    assert evidence.context.verified_plan_surface_paths == frozenset({"x.txt"})
    assert rows["t0"]["lifecycle_accepted"] is False
    assert rows["t0"]["lifecycle_reason"] == (
        "landed lifecycle does not bind the latest slice records"
    )
    assert rows["t1"]["lifecycle_accepted"] is True
    assert rows["t1"]["lifecycle_reason"] == ""


def test_manual_pending_create_remains_an_overwrite_collision(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("manual unbound output\n", encoding="utf-8")
    spec = corpus.latest_spec()
    assert spec is not None

    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
    )
    findings = cited_path_exists(spec, evidence.context)

    assert evidence.retained_tier_ids == ()
    assert evidence.accepted_tier_ids == ()
    assert evidence.context.verified_create_paths == frozenset()
    assert any(
        finding.category == "overwrite_collision"
        and finding.id == "creates_path_already_exists__create-x__x.txt"
        for finding in findings
    )


def test_admission_tier_requires_pending_create_to_remain_base_absent(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by t0\n", encoding="utf-8")
    _land_current_workspace(repo, corpus, tier_id="t0")
    spec = corpus.latest_spec()
    assert spec is not None
    admission = Tier(
        tier_id="t1",
        depends_on=("t0",),
        slices=(Slice(
            id="pending-create",
            directive=ImplementationDirective(
                slice_id="pending-create",
                intent="Create a new output.",
                files_in_scope=(),
                creates_paths=("new.py",),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        ),),
    )

    clean = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
        admission_tier=admission,
    )
    assert clean.workspace_evidence_passed is True
    assert clean.context.expected_plan_surface_paths == frozenset({
        "new.py",
        "x.txt",
    })

    (repo / "new.py").write_text("unrecorded = True\n", encoding="utf-8")
    dirty = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
        admission_tier=admission,
    )
    assert dirty.workspace_evidence_passed is False
    assert "new.py" in dirty.context.expected_plan_surface_paths
    assert "new.py" not in dirty.context.verified_plan_surface_paths


def test_admission_tier_expands_recursive_pending_scope(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_chained_create_and_edit(tmp_path)
    (repo / "x.txt").write_text("created by t0\n", encoding="utf-8")
    _land_current_workspace(repo, corpus, tier_id="t0")
    spec = corpus.latest_spec()
    assert spec is not None
    admission = Tier(
        tier_id="t1",
        depends_on=("t0",),
        slices=(Slice(
            id="pending-directory",
            directive=ImplementationDirective(
                slice_id="pending-directory",
                intent="Edit files recursively.",
                files_in_scope=("pending_dir/",),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 2},
            ),
        ),),
    )

    clean = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
        admission_tier=admission,
    )
    assert clean.workspace_evidence_passed is True

    pending_dir = repo / "pending_dir"
    pending_dir.mkdir()
    pending_dir.joinpath("nested.py").write_text(
        "unrecorded = True\n",
        encoding="utf-8",
    )
    dirty = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
        admission_tier=admission,
    )
    assert dirty.workspace_evidence_passed is False
    assert "pending_dir/nested.py" in (
        dirty.context.expected_plan_surface_paths
    )
    assert "pending_dir/nested.py" not in (
        dirty.context.verified_plan_surface_paths
    )


# ----------------------------------------------------------------------
# Retained siblings of a failed tier (resume after a partial tier)
# ----------------------------------------------------------------------


def _slice(
    slice_id: str,
    *,
    intent: str,
    files_in_scope: tuple[str, ...] = (),
    creates_paths: tuple[str, ...] = (),
) -> Slice:
    return Slice(
        id=slice_id,
        directive=ImplementationDirective(
            slice_id=slice_id,
            intent=intent,
            files_in_scope=files_in_scope,
            creates_paths=creates_paths,
            verify_mode="skip",
            blast_radius={
                "max_files": max(1, len(files_in_scope) + len(creates_paths)),
            },
        ),
    )


def _setup_plan(
    tmp_path: Path,
    *,
    tiers: tuple[Tier, ...],
    base_files: dict[str, str],
) -> tuple[Path, Corpus]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "workspace-evidence@invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Workspace Evidence"],
        cwd=repo,
        check=True,
    )
    for name, content in base_files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    base = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="workspace-evidence-test",
            repo_url="https://example.invalid/workspace-evidence.git",
            base_commit=base_commit,
            branch="main",
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Prove the retained state of a partially completed tier.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(base)
    if tiers:
        corpus.put_spec(base.with_tier_dag(TierDAG(tiers=tiers)))
    return repo, corpus


def _abandon_tier(corpus: Corpus, *, tier_id: str) -> None:
    """Publish the lifecycle event Tier writes when a slice fails."""

    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="abandoned",
            reason=f"tier {tier_id}: slice_failed",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )


def _evidence(
    repo: Path,
    corpus: Corpus,
    *,
    admission_tier_id: str | None = None,
):
    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    admission_tier = (
        None
        if admission_tier_id is None
        else next(
            item
            for item in spec.tier_dag.tiers
            if item.tier_id == admission_tier_id
        )
    )
    evidence = reconstruct_plan_workspace_evidence(
        corpus=corpus,
        spec=spec,
        source_root=repo,
        admission_tier=admission_tier,
    )
    return spec, evidence, {row["tier_id"]: row for row in evidence.rows}


def _blocking_ids(spec: Spec, context) -> list[str]:
    return sorted(
        finding.id
        for finding in cited_path_exists(spec, context)
        if finding.severity in {"Critical", "High"}
    )


def _latest_implementation(corpus: Corpus, slice_id: str) -> Implementation:
    spec = corpus.latest_spec()
    assert spec is not None
    return next(
        implementation
        for implementation in (
            corpus.get_record(ref, as_type=Implementation)
            for ref in reversed(spec.implementation_refs)
        )
        if implementation.slice_id == slice_id
    )


def _partial_tier_plan() -> tuple[Tier, ...]:
    return (
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(
                _slice(
                    "keep",
                    intent="Create kept.txt, edit edit.txt, delete drop.txt.",
                    files_in_scope=("edit.txt", "drop.txt"),
                    creates_paths=("kept.txt",),
                ),
                _slice(
                    "fail",
                    intent="Create failed.txt.",
                    creates_paths=("failed.txt",),
                ),
            ),
        ),
        Tier(
            tier_id="t1",
            depends_on=("t0",),
            slices=(
                _slice(
                    "follow",
                    intent="Extend the kept output.",
                    files_in_scope=("kept.txt",),
                ),
            ),
        ),
    )


_PARTIAL_TIER_BASE_FILES = {
    "edit.txt": "base edit\n",
    "drop.txt": "base drop\n",
}


def _run_partial_tier(
    repo: Path,
    corpus: Corpus,
    *,
    record_failed: bool = True,
    roll_back_failed: bool = True,
    keep_scope_escapes: tuple[str, ...] = (),
) -> None:
    """Leave ``t0`` as Tier does after ``keep`` landed and ``fail`` failed."""

    (repo / "kept.txt").write_text("created by keep\n", encoding="utf-8")
    (repo / "edit.txt").write_text("edited by keep\n", encoding="utf-8")
    (repo / "drop.txt").unlink()
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        slice_id="keep",
        publish_lifecycle=False,
        scope_escapes=keep_scope_escapes,
    )
    (repo / "failed.txt").write_text("partial output\n", encoding="utf-8")
    if record_failed:
        _land_current_workspace(
            repo,
            corpus,
            tier_id="t0",
            slice_id="fail",
            classification="verify_failed",
            publish_lifecycle=False,
        )
    if roll_back_failed:
        (repo / "failed.txt").unlink()
    _abandon_tier(corpus, tier_id="t0")


def _prefix_then_partial_plan() -> tuple[Tier, ...]:
    return (
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(
                _slice(
                    "create-x",
                    intent="Create x.txt.",
                    creates_paths=("x.txt",),
                ),
            ),
        ),
        Tier(
            tier_id="t1",
            depends_on=("t0",),
            slices=(
                _slice(
                    "edit-x",
                    intent="Edit the x.txt created by the predecessor.",
                    files_in_scope=("x.txt",),
                ),
                _slice(
                    "create-y",
                    intent="Create y.txt.",
                    creates_paths=("y.txt",),
                ),
            ),
        ),
        Tier(
            tier_id="t2",
            depends_on=("t0",),
            slices=(
                _slice(
                    "create-z",
                    intent="Create z.txt.",
                    creates_paths=("z.txt",),
                ),
            ),
        ),
    )


def _run_prefix_then_partial(repo: Path, corpus: Corpus) -> None:
    """Land ``t0``; in ``t1`` retain ``edit-x`` while ``create-y`` failed."""

    (repo / "x.txt").write_text("created by t0\n", encoding="utf-8")
    _land_current_workspace(repo, corpus, tier_id="t0")
    (repo / "x.txt").write_text(
        "created by t0\nedited by t1\n",
        encoding="utf-8",
    )
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t1",
        slice_id="edit-x",
        publish_lifecycle=False,
    )
    _abandon_tier(corpus, tier_id="t1")


class _ResumeBackend:
    """Implementer stub: ``s0`` creates its file, ``s1`` can be made to fail."""

    name = "resume-create-authority-stub"
    model = "stub-model"

    def __init__(self, repo: Path, *, success) -> None:
        self.repo = repo
        self.success = success
        self.fail_s1 = True
        self.calls: list[str] = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        is_s0 = "slice `s0`" in invocation.user
        self.calls.append("s0" if is_s0 else "s1")
        if is_s0:
            (self.repo / "new_a.py").write_text("created = True\n")
        elif not self.fail_s1:
            (self.repo / "b.py").write_text("# completed s1\n")
        return self.success()


@pytest.mark.timeout(300)
def test_architect_resume_completes_tier_that_retained_a_sibling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The real Tier leaves a landed sibling; the real Architect resumes it."""

    from crucible.workflows import architect as arch
    from crucible.workflows._plan_provenance import bind_plan_contract
    from crucible.workflows.tier import run_tier
    from tests.test_workflows_architect_reaccept import _fixed_fingerprint
    from tests.test_workflows_tier_mutation_transaction import (
        _patch_backend,
        _success,
    )

    dag = TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(
            _slice("s0", intent="Create new_a.py.", creates_paths=("new_a.py",)),
            _slice("s1", intent="Edit b.py.", files_in_scope=("b.py",)),
        ),
    ),))
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=(),
        base_files={"b.py": "# baseline b.py\n"},
    )
    base = corpus.latest_spec()
    assert base is not None and base.tier_dag is None
    fingerprint = _fixed_fingerprint()
    planned_ref = corpus.put_record(
        LifecycleEvent(
            kind="planned",
            reason="Retained reviewed plan",
            affected_versions=(base.version + 1,),
            execution=exec_ctx_now("plan"),
        ),
        kind="lifecycle_events",
    )
    corpus.put_spec(bind_plan_contract(
        corpus,
        base.with_appended_refs(
            tier_dag=dag,
            lifecycle_event_refs=(planned_ref,),
        ),
        fingerprint=fingerprint,
    ))

    backend = _ResumeBackend(repo, success=_success)
    _patch_backend(monkeypatch, backend)

    assert run_tier(
        spec=corpus.spec_dir,
        tier_id="t0",
        cwd=str(repo),
        max_concurrent_slices=1,
        build_retries=0,
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 1
    failed = corpus.latest_spec()
    assert failed is not None and failed.tier_dag == dag
    assert [
        (implementation.slice_id, implementation.classification)
        for implementation in (
            corpus.get_record(ref, as_type=Implementation)
            for ref in failed.implementation_refs
        )
    ] == [("s0", "success"), ("s1", "soft_fail_no_changes")]
    assert backend.calls == ["s0", "s1"]
    assert (repo / "new_a.py").read_text() == "created = True\n"
    assert (repo / "b.py").read_text() == "# baseline b.py\n"

    @contextmanager
    def pool(*_args, **_kwargs):
        class Pool:
            def get(self, role, **_kwargs):
                return backend

        yield Pool()

    backend.calls.clear()
    backend.fail_s1 = False
    monkeypatch.setattr(arch, "backend_pool", pool)
    monkeypatch.setattr(
        arch,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: fingerprint,
    )

    returncode = arch.run_architect(
        spec=corpus.spec_dir,
        cwd=str(repo),
        skip_build_verify=True,
    )

    # Only the failed sibling is implemented again; the plan keeps its
    # authority and the tier lands.
    resumed = corpus.latest_spec()
    assert resumed is not None and resumed.tier_dag == dag
    assert returncode == 0
    assert backend.calls == ["s1"]
    assert [
        corpus.get_record(ref, as_type=LifecycleEvent).kind
        for ref in resumed.lifecycle_event_refs
    ][-1] == "landed"
    assert (repo / "new_a.py").read_text() == "created = True\n"
    assert (repo / "b.py").read_text() == "# completed s1\n"
    (architect_run,) = corpus.spec_dir.glob("runs/*_architect_*")
    summary = json.loads((architect_run / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert not (architect_run / "validate_findings.json").exists()


def test_retained_sibling_of_failed_tier_is_proven_and_authorized(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_partial_tier_plan(),
        base_files=_PARTIAL_TIER_BASE_FILES,
    )
    _run_partial_tier(repo, corpus)

    spec, evidence, rows = _evidence(repo, corpus)
    create_key = ("t0", "keep", "kept.txt")
    surface = frozenset({"kept.txt", "edit.txt", "drop.txt"})

    # The tier is incomplete, so nothing joins the retained prefix and the
    # prefix bindings Tier admission compares stay empty.
    assert evidence.retained_tier_ids == ()
    assert evidence.accepted_tier_ids == ()
    assert evidence.all_slices_complete is False
    assert evidence.selected_implementation_hashes == ()
    assert evidence.errors == ()
    assert evidence.workspace_evidence_passed is True
    assert evidence.required_create_paths == frozenset({create_key})
    assert evidence.context.successful_plan_bound_slices == frozenset({
        ("t0", "keep"),
    })
    assert evidence.context.verified_create_paths == frozenset({create_key})
    assert evidence.context.expected_plan_surface_paths == surface
    assert evidence.context.verified_plan_surface_paths == surface
    assert evidence.context.planned_deletion_paths == frozenset({"drop.txt"})
    assert rows["t0"]["retained_prefix"] is False
    assert rows["t0"]["pending_slices"] == ["fail"]
    assert rows["t0"]["boundary_retained_slices"] == ["keep"]
    assert rows["t0"]["boundary_retained_proven"] is True
    assert rows["t0"]["boundary_retained_implementation_hashes"] == [
        ["keep", hash_record(_latest_implementation(corpus, "keep"))],
    ]
    assert rows["t0"]["boundary_retained_unproven"] == []
    assert rows["t1"]["boundary_retained_slices"] == []
    assert rows["t1"]["boundary_retained_proven"] is False
    assert rows["t1"]["boundary_retained_implementation_hashes"] == []
    assert cited_path_exists(spec, evidence.context) == ()


@pytest.mark.parametrize("record_failed", [True, False])
def test_failed_or_unrecorded_sibling_output_is_never_excused(
    tmp_path: Path,
    record_failed: bool,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_partial_tier_plan(),
        base_files=_PARTIAL_TIER_BASE_FILES,
    )
    _run_partial_tier(
        repo,
        corpus,
        record_failed=record_failed,
        roll_back_failed=False,
    )

    spec, evidence, rows = _evidence(repo, corpus)

    # The retained sibling is proven; what the failed sibling left behind
    # is the only collision and stays blocking.
    assert _blocking_ids(spec, evidence.context) == [
        "creates_path_already_exists__fail__failed.txt",
    ]
    assert evidence.context.successful_plan_bound_slices == frozenset({
        ("t0", "keep"),
    })
    assert evidence.context.verified_create_paths == frozenset({
        ("t0", "keep", "kept.txt"),
    })
    assert rows["t0"]["boundary_retained_slices"] == ["keep"]
    assert rows["t0"]["boundary_retained_proven"] is True


@pytest.mark.parametrize("state", ["tampered", "reset", "record_error"])
def test_unproven_retained_sibling_changes_nothing(
    tmp_path: Path,
    state: str,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_partial_tier_plan(),
        base_files=_PARTIAL_TIER_BASE_FILES,
    )
    _run_partial_tier(
        repo,
        corpus,
        keep_scope_escapes=(
            ("outside.txt",) if state == "record_error" else ()
        ),
    )
    if state == "tampered":
        (repo / "edit.txt").write_text("tampered\n", encoding="utf-8")
    elif state == "reset":
        for name, content in _PARTIAL_TIER_BASE_FILES.items():
            (repo / name).write_text(content, encoding="utf-8")
        (repo / "kept.txt").unlink()

    spec, evidence, rows = _evidence(repo, corpus)
    mismatch = "workspace does not match the composed postimage"
    release_findings = [
        "creates_path_already_exists__keep__kept.txt",
        "missing_files_in_scope__keep__drop.txt",
    ]

    # Without a complete proof the result is the retained-prefix proof:
    # no authority for the sibling and no finding the prefix does not have.
    # An error of the rejected proof stays out of the result as well.
    assert evidence.errors == ()
    assert evidence.context.postimage_evidence_errors == ()
    assert evidence.workspace_evidence_passed is True
    assert evidence.selected_implementation_hashes == ()
    assert evidence.required_create_paths == frozenset()
    assert evidence.context.successful_plan_bound_slices == frozenset()
    assert evidence.context.verified_create_paths == frozenset()
    assert evidence.context.expected_plan_surface_paths == frozenset()
    if state == "tampered":
        assert _blocking_ids(spec, evidence.context) == release_findings
        unproven = [f"edit.txt: {mismatch}"]
    elif state == "reset":
        assert cited_path_exists(spec, evidence.context) == ()
        unproven = [
            f"drop.txt: {mismatch}",
            f"edit.txt: {mismatch}",
            f"kept.txt: {mismatch}",
            "t0/keep: declared create 'kept.txt' is not verified",
        ]
    else:
        # The record itself cannot be proven, so none of its writes is
        # attributed to it: the error comes first, then what follows from it.
        assert _blocking_ids(spec, evidence.context) == release_findings
        unproven = [
            "t0/keep: successful Implementation carries scope escapes "
            "['outside.txt']",
            f"drop.txt: {mismatch}",
            f"edit.txt: {mismatch}",
            f"kept.txt: {mismatch}",
            "t0/keep: declared create 'kept.txt' is not verified",
        ]
    # The row keeps why the sibling stayed without authority.
    assert rows["t0"]["boundary_retained_slices"] == ["keep"]
    assert rows["t0"]["boundary_retained_proven"] is False
    assert rows["t0"]["boundary_retained_implementation_hashes"] == []
    assert rows["t0"]["boundary_retained_unproven"] == unproven


def test_retained_siblings_obtain_authority_only_together(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=(Tier(
            tier_id="t0",
            depends_on=(),
            slices=(
                _slice("keep-a", intent="Create a.txt.", creates_paths=("a.txt",)),
                _slice("keep-b", intent="Create b.txt.", creates_paths=("b.txt",)),
                _slice(
                    "fail",
                    intent="Create failed.txt.",
                    creates_paths=("failed.txt",),
                ),
            ),
        ),),
        base_files={"seed.txt": "seed\n"},
    )
    for slice_id, name in (("keep-a", "a.txt"), ("keep-b", "b.txt")):
        (repo / name).write_text(f"created by {slice_id}\n", encoding="utf-8")
        _land_current_workspace(
            repo,
            corpus,
            tier_id="t0",
            slice_id=slice_id,
            publish_lifecycle=False,
        )
    (repo / "failed.txt").write_text("partial output\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t0",
        slice_id="fail",
        classification="verify_failed",
        publish_lifecycle=False,
    )
    (repo / "failed.txt").unlink()
    _abandon_tier(corpus, tier_id="t0")

    spec, evidence, rows = _evidence(repo, corpus)

    assert evidence.workspace_evidence_passed is True
    assert evidence.context.successful_plan_bound_slices == frozenset({
        ("t0", "keep-a"),
        ("t0", "keep-b"),
    })
    assert evidence.context.verified_create_paths == frozenset({
        ("t0", "keep-a", "a.txt"),
        ("t0", "keep-b", "b.txt"),
    })
    assert rows["t0"]["boundary_retained_slices"] == ["keep-a", "keep-b"]
    assert rows["t0"]["boundary_retained_proven"] is True
    assert rows["t0"]["boundary_retained_implementation_hashes"] == [
        ["keep-a", hash_record(_latest_implementation(corpus, "keep-a"))],
        ["keep-b", hash_record(_latest_implementation(corpus, "keep-b"))],
    ]
    assert cited_path_exists(spec, evidence.context) == ()

    # One sibling's output no longer matches its record.  The untouched
    # sibling does not keep its authority: the siblings are proven as one
    # state or not at all.
    (repo / "b.txt").write_text("tampered\n", encoding="utf-8")
    spec, evidence, rows = _evidence(repo, corpus)

    assert evidence.context.successful_plan_bound_slices == frozenset()
    assert evidence.context.verified_create_paths == frozenset()
    assert rows["t0"]["boundary_retained_slices"] == ["keep-a", "keep-b"]
    assert rows["t0"]["boundary_retained_proven"] is False
    assert rows["t0"]["boundary_retained_implementation_hashes"] == []
    assert rows["t0"]["boundary_retained_unproven"] == [
        "b.txt: workspace does not match the composed postimage",
        "t0/keep-b: declared create 'b.txt' is not verified",
    ]
    assert _blocking_ids(spec, evidence.context) == [
        "creates_path_already_exists__keep-a__a.txt",
        "creates_path_already_exists__keep-b__b.txt",
    ]


def test_retained_sibling_final_writer_supersedes_prefix_postimage(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_prefix_then_partial_plan(),
        base_files={"seed.txt": "seed\n"},
    )
    _run_prefix_then_partial(repo, corpus)

    spec, evidence, rows = _evidence(repo, corpus)

    assert evidence.errors == ()
    assert evidence.workspace_evidence_passed is True
    assert evidence.retained_tier_ids == ("t0",)
    assert evidence.accepted_tier_ids == ("t0",)
    assert evidence.selected_implementation_hashes == ((
        "t0",
        "create-x",
        hash_record(_latest_implementation(corpus, "create-x")),
    ),)
    assert rows["t0"]["boundary_retained_slices"] == []
    assert rows["t1"]["boundary_retained_slices"] == ["edit-x"]
    assert rows["t1"]["boundary_retained_proven"] is True
    assert rows["t1"]["boundary_retained_implementation_hashes"] == [
        ["edit-x", hash_record(_latest_implementation(corpus, "edit-x"))],
    ]
    assert evidence.context.verified_create_paths == frozenset({
        ("t0", "create-x", "x.txt"),
    })
    assert evidence.context.expected_plan_surface_paths == frozenset({"x.txt"})
    assert evidence.context.verified_plan_surface_paths == frozenset({"x.txt"})
    assert cited_path_exists(spec, evidence.context) == ()


def test_architect_skips_accepted_tier_rewritten_by_proven_retained_sibling(
    tmp_path: Path,
    caplog,
) -> None:
    from crucible.workflows import architect as arch

    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_prefix_then_partial_plan(),
        base_files={"seed.txt": "seed\n"},
    )
    _run_prefix_then_partial(repo, corpus)
    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    accepted, partial, _untouched = spec.tier_dag.topo_order()
    assert (accepted.tier_id, partial.tier_id) == ("t0", "t1")

    def skip_is_safe(tier: Tier) -> bool:
        return arch._completed_tier_skip_is_safe(
            corpus=corpus,
            spec=spec,
            tier=tier,
            cwd=str(repo),
        )

    # x.txt no longer holds t0's own postimage: the retained sibling of t1
    # rewrote it.  The proven final writer keeps t0's skip available, and
    # the partial tier itself gains no acceptance from the same proof.
    assert skip_is_safe(accepted) is True
    assert skip_is_safe(partial) is False

    caplog.clear()
    (repo / "x.txt").write_text("tampered\n", encoding="utf-8")

    assert skip_is_safe(accepted) is False
    assert (
        "retained-prefix proof did not authorize tier t0" in caplog.text
    )


def test_retained_slice_after_prefix_gap_gets_no_authority(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_prefix_then_partial_plan(),
        base_files={"seed.txt": "seed\n"},
    )
    (repo / "y.txt").write_text("created by t1\n", encoding="utf-8")
    _land_current_workspace(
        repo,
        corpus,
        tier_id="t1",
        slice_id="create-y",
        publish_lifecycle=False,
    )

    spec, evidence, rows = _evidence(repo, corpus)

    # t0 never ran, so t1's record is not part of a dependency-composed
    # state: only the first incomplete tier is eligible.
    assert evidence.retained_tier_ids == ()
    assert evidence.context.successful_plan_bound_slices == frozenset()
    assert evidence.context.verified_create_paths == frozenset()
    assert _blocking_ids(spec, evidence.context) == [
        "creates_path_already_exists__create-y__y.txt",
    ]
    assert [row["boundary_retained_slices"] for row in evidence.rows] == [
        [],
        [],
        [],
    ]


@pytest.mark.parametrize(
    ("plan", "admission_tier_id", "boundary_tier_id", "expected", "verified"),
    [
        # The boundary tier itself: its retained sibling stays unselected,
        # so its writes are unrecorded dirt for the admission surface.
        (
            "partial",
            "t0",
            "t0",
            {"edit.txt", "drop.txt", "kept.txt", "failed.txt"},
            {"failed.txt"},
        ),
        ("prefix", "t1", "t1", {"x.txt", "y.txt"}, {"y.txt"}),
        # A tier that does not depend on the partial tier: its accepted
        # dependency output was changed by a tier outside its closure.
        ("prefix", "t2", "t1", {"x.txt", "z.txt"}, {"z.txt"}),
    ],
)
def test_tier_admission_receives_the_retained_prefix_only(
    tmp_path: Path,
    plan: str,
    admission_tier_id: str,
    boundary_tier_id: str,
    expected: set[str],
    verified: set[str],
) -> None:
    if plan == "partial":
        repo, corpus = _setup_plan(
            tmp_path,
            tiers=_partial_tier_plan(),
            base_files=_PARTIAL_TIER_BASE_FILES,
        )
        _run_partial_tier(repo, corpus)
        prefix_slices: frozenset[tuple[str, str]] = frozenset()
    else:
        repo, corpus = _setup_plan(
            tmp_path,
            tiers=_prefix_then_partial_plan(),
            base_files={"seed.txt": "seed\n"},
        )
        _run_prefix_then_partial(repo, corpus)
        prefix_slices = frozenset({("t0", "create-x")})

    resume = _evidence(repo, corpus)[1]
    _, admission, rows = _evidence(
        repo,
        corpus,
        admission_tier_id=admission_tier_id,
    )

    # Tier's persistent-workspace admission keeps the retained-prefix proof:
    # the retained sibling is not selected and its writes do not verify.
    assert admission.workspace_evidence_passed is False
    assert admission.errors == ()
    assert admission.context.successful_plan_bound_slices == prefix_slices
    assert admission.context.verified_create_paths == frozenset()
    assert admission.context.expected_plan_surface_paths == frozenset(expected)
    assert admission.context.verified_plan_surface_paths == frozenset(verified)
    assert [
        identity[:2] for identity in admission.selected_implementation_hashes
    ] == sorted(prefix_slices)
    assert admission.selected_implementation_hashes == (
        resume.selected_implementation_hashes
    )
    # The same state is proven for Architect's resume gate.
    assert resume.workspace_evidence_passed is True
    assert resume.context.verified_plan_surface_paths == (
        resume.context.expected_plan_surface_paths
    )
    boundary = rows[boundary_tier_id]
    assert boundary["boundary_retained_slices"] != []
    assert boundary["boundary_retained_proven"] is False
    assert boundary["boundary_retained_implementation_hashes"] == []
    assert boundary["boundary_retained_unproven"] == [
        "not attempted: Tier admission receives the retained prefix only",
    ]


def _create_then_name_plan() -> tuple[Tier, ...]:
    """``t0`` creates a module that the intent of ``t1`` names."""

    return (
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(
                _slice(
                    "creator",
                    intent="Create the new module.",
                    creates_paths=("pkg/new.py",),
                ),
            ),
        ),
        Tier(
            tier_id="t1",
            depends_on=("t0",),
            slices=(
                _slice(
                    "use-new",
                    intent="Then import from pkg/new.py in the seed module.",
                    files_in_scope=("pkg/seed.py",),
                ),
            ),
        ),
    )


_NAMED_CREATE_FINDING = "scope_omits_referenced_path__use-new__pkg/new.py"


def test_retained_create_named_by_later_intent_keeps_plan_time_verdict(
    tmp_path: Path,
) -> None:
    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_create_then_name_plan(),
        base_files={"pkg/seed.py": "seed = True\n"},
    )
    spec = corpus.latest_spec()
    assert spec is not None

    assert scope_covers_intent(spec, RuleContext(workspace_root=repo)) == ()

    (repo / "pkg" / "new.py").write_text("new = True\n", encoding="utf-8")
    _land_current_workspace(repo, corpus, tier_id="t0")
    spec, evidence, _rows = _evidence(repo, corpus)

    assert evidence.context.verified_create_paths == frozenset({
        ("t0", "creator", "pkg/new.py"),
    })
    assert scope_covers_intent(spec, evidence.context) == ()

    (repo / "pkg" / "new.py").write_text("tampered = True\n", encoding="utf-8")
    spec, evidence, _rows = _evidence(repo, corpus)

    assert evidence.context.verified_create_paths == frozenset()
    assert [
        finding.id for finding in scope_covers_intent(spec, evidence.context)
    ] == [_NAMED_CREATE_FINDING]


@pytest.mark.parametrize("alter_created_file", [False, True])
def test_reconcile_lands_plan_whose_later_intent_names_a_proven_create(
    tmp_path: Path,
    monkeypatch,
    capsys,
    alter_created_file: bool,
) -> None:
    """Reconcile hands its own post-implementation context to every rule."""

    from crucible.workflows import _gate_evidence
    from crucible.workflows.handoff import main as handoff_main
    from crucible.workflows.reconcile import main as reconcile_main

    repo, corpus = _setup_plan(
        tmp_path,
        tiers=_create_then_name_plan(),
        base_files={"pkg/seed.py": "seed = True\n"},
    )
    # Handoff records the two diffs; as in the Reconcile tests the waived
    # records stand in for verified ones.
    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        lambda *args, **kwargs: True,
    )
    created = repo / "pkg" / "new.py"
    for slice_id, path, content in (
        ("creator", created, "new = True\n"),
        ("use-new", repo / "pkg" / "seed.py", "from pkg.new import new\n"),
    ):
        path.write_text(content, encoding="utf-8")
        assert handoff_main([
            "--spec", str(corpus.spec_dir),
            "--slice", slice_id,
            "--cwd", str(repo),
            "--skip-verify",
        ]) == 0
    capsys.readouterr()
    if alter_created_file:
        created.write_text("tampered = True\n", encoding="utf-8")

    returncode = reconcile_main([
        "--spec", str(corpus.spec_dir), "--cwd", str(repo),
    ])

    summary = json.loads(capsys.readouterr().out)
    latest = corpus.latest_spec()
    assert latest is not None
    finding_ids = [
        finding.id
        for finding in corpus.get_record(
            latest.verdict_refs[-1],
            as_type=Verdict,
        ).findings
    ]
    lifecycle_kinds = [
        corpus.get_record(ref, as_type=LifecycleEvent).kind
        for ref in latest.lifecycle_event_refs
    ]
    if alter_created_file:
        # Without the proof the created file is an existing path again.
        assert returncode == 1
        assert summary["status"] == "incomplete"
        assert summary["validation"]["decision"] == "reject"
        assert _NAMED_CREATE_FINDING in finding_ids
        assert "landed" not in lifecycle_kinds
    else:
        assert returncode == 0
        assert summary["status"] == "landed"
        assert summary["validation"]["decision"] == "pass"
        assert _NAMED_CREATE_FINDING not in finding_ids
        assert lifecycle_kinds[-1] == "landed"
