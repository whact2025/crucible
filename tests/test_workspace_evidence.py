# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

from assay import (
    Implementation,
    ImplementationDirective,
    LifecycleEvent,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from assay.corpus.store import Corpus
from assay.primitives.implement import capture_workspace_delta

from crucible.rules.cited_path_exists import cited_path_exists
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
) -> None:
    spec = corpus.latest_spec()
    assert spec is not None and spec.tier_dag is not None
    tier = next(item for item in spec.tier_dag.tiers if item.tier_id == tier_id)
    slice_ = tier.slices[0]
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
        classification="success",
        execution=exec_ctx_now(implementation_recipe),
        scope_escapes=(),
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
