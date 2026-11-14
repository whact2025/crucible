# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused safety, resume, and parallel-budget contracts for SWE-bench."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from assay import (
    ExecutionContext,
    Implementation,
    ImplementationDirective,
    LifecycleEvent,
    Measurement,
    ReviewCoverage,
    Slice,
    Tier,
    TierDAG,
    Verdict,
    hash_record,
)
from assay.corpus.store import Corpus

from crucible.common.cost import CostAccumulator
from crucible.common.benchmark_runtime import task_execution_lease
from crucible.common.storage_quota import (
    StorageQuotaBackendUnavailable,
    StorageQuotaCleanupError,
    StorageQuotaExceeded,
)
from crucible.products.swebench.dataset import SwebenchInstance
from crucible.constraints import (
    AuthoringInstructionConstraint,
    LanguagesConstraint,
)
from crucible.products.swebench.spec_synthesis import (
    SWEBENCH_AUTHORING_POLICY_ID,
    SWEBENCH_PLANNER_CONSTRAINTS,
    synthesize_starter_spec,
)
from crucible.workflows import _outcome, swebench as wf


def _prepare_swebench_parent(path: Path) -> Path:
    path.mkdir(parents=True)
    _outcome.begin_run(path, workflow="swebench")
    return path


def _instance(iid: str = "org__repo-1") -> SwebenchInstance:
    return SwebenchInstance(
        instance_id=iid,
        repo="org/repo",
        base_commit="a" * 40,
        patch="",
        test_patch="",
        problem_statement=f"fix {iid}",
        fail_to_pass=(),
        pass_to_pass=(),
        version="1.0",
    )


def _passed_pristine_report(
    patch: str,
    inst: SwebenchInstance,
    image_digest: str,
) -> wf.PristineVerificationReport:
    patch_bytes = patch.encode("utf-8")
    return wf.PristineVerificationReport(
        schema="crucible.swebench.pristine-verification.v1",
        status=wf.PristineVerificationStatus.PASSED,
        reason="candidate matched the clean visible-test baseline",
        patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
        patch_bytes=len(patch_bytes),
        base_commit=inst.base_commit,
        image_digest=image_digest,
        apply_check_rc=0,
        apply_rc=0,
        runner_rc=0,
        candidate_rc=0,
        comparison_mode="baseline_match",
        preexisting_failures=0,
        new_failures=0,
        worsened_failures=0,
        duration_s=0.1,
        cleanup_status="complete",
        raw_output_tail="",
    )


def _stale_cleanup_dag() -> TierDAG:
    return TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(Slice(
            id="source",
            directive=ImplementationDirective(
                slice_id="source",
                intent=(
                    "Fix the source behavior and remove the now-unused import."
                ),
                files_in_scope=("pkg/core.py",),
                creates_paths=(),
                verify_mode="tests",
                blast_radius={"max_files": 1},
            ),
        ),),
    ),))


def _seed_legacy_managed_plan(
    corpus: Corpus,
    inst: SwebenchInstance,
    *,
    extra_constraints: tuple[object, ...] = (),
):
    starter = synthesize_starter_spec(inst)
    legacy_managed = tuple(
        replace(
            constraint,
            policy_owner=None,
            policy_rule_id=None,
            policy_revision=0,
        )
        for constraint in SWEBENCH_PLANNER_CONSTRAINTS[:2]
    )
    corpus.put_spec(replace(
        starter,
        constraints=(legacy_managed[0], *extra_constraints, legacy_managed[1]),
    ))
    execution = wf.exec_ctx_now("test")
    return corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="planned",
            reason="accepted legacy SWE-bench plan",
            affected_versions=(version,),
            execution=execution,
        ),
        transform=lambda candidate: replace(
            candidate,
            tier_dag=_stale_cleanup_dag(),
        ),
    ).spec


def test_authoring_metadata_excludes_hidden_evaluation_fields():
    digest = "test/image@sha256:" + "a" * 64
    metadata = wf._authoring_metadata(
        _instance(), container_id="sandbox-1", image_digest=digest,
    )
    assert metadata == {
        "instance_id": "org__repo-1",
        "repo": "org/repo",
        "version": "1.0",
        "container_id": "sandbox-1",
        "image_digest": digest,
    }
    assert not wf._HIDDEN_EVALUATION_FIELDS.intersection(metadata)


def test_authoring_metadata_requires_container():
    with pytest.raises(Exception, match="live container"):
        wf._authoring_metadata(_instance(), container_id="", image_digest="")


def test_checkpoint_binds_filter_normalizer_and_evaluator_policy(
    monkeypatch,
):
    versions: dict[str, str] = {}
    seen: set[str] = set()

    def _digest(path):
        name = Path(path).name
        seen.add(name)
        return f"sha256:{versions.get(name, 'stable')}"

    monkeypatch.setattr(wf, "file_digest", _digest)
    monkeypatch.setattr(
        wf, "file_set_fingerprint", lambda *args, **kwargs: "sha256:rubrics",
    )
    monkeypatch.setattr(
        wf,
        "image_verification_policy_identity",
        lambda: {"schema": "test-policy.v1"},
    )

    def _fingerprint() -> str:
        return wf._swe_checkpoint_fingerprint(
            _instance(),
            routing_source_sha256=None,
            max_turns=10,
            child_budget_usd=10.0,
            include_breach=True,
            include_review=True,
            enable_build_verify=True,
            evaluate_inline=True,
            eval_timeout=60,
            routed_backend="stub",
            routed_model="model",
            image_digest="test/image@sha256:" + "a" * 64,
        )

    baseline = _fingerprint()
    policy_modules = {
        "plan_normalization.py",
        "paths.py",
        "evaluator_compat.py",
        "swebench_pytest.py",
        "pristine_verifier.py",
        "instance_container.py",
        "repo_workspace.py",
    }
    assert policy_modules <= seen
    for module in policy_modules:
        versions[module] = "changed"
        assert _fingerprint() != baseline
        versions.clear()


def test_build_verify_refuses_host_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(wf, "_docker_available", lambda: False)
    with pytest.raises(
        wf._InstanceContainerSetupError,
        match="host execution fallback is disabled",
    ):
        with wf._instance_env_lifecycle(
            _instance(), tmp_path, enabled=True, log_prefix="test",
        ):
            pytest.fail("unreachable")


def test_authoring_requires_kernel_quota_before_starting_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.common.storage_quota import StorageQuotaBackendUnavailable
    from crucible.products.swebench.instance_container import InstanceContainerSpec

    teardown: list[tuple[str, str | None]] = []
    monkeypatch.setattr(wf, "_docker_available", lambda: True)
    monkeypatch.setattr(
        wf,
        "_resolve_container_spec_impl",
        lambda inst: InstanceContainerSpec(image="repo/image:tag", install_command="true"),
    )
    monkeypatch.setattr(wf, "_remove_owned_stopped_container", lambda **kwargs: None)
    monkeypatch.setattr(
        wf,
        "_setup_instance_container_impl",
        lambda workspace, container_name, spec, *, labels: "container-id",
    )
    monkeypatch.setattr(
        wf,
        "_container_immutable_image",
        lambda container_id: "repo/image@sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        wf,
        "_teardown_instance_container",
        lambda container_id, image=None: teardown.append((container_id, image)),
    )

    class _UnavailableQuota:
        def __init__(self, paths, **kwargs):
            assert kwargs["require_kernel"] is True
            raise StorageQuotaBackendUnavailable("kernel quota required")

    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _UnavailableQuota)

    with pytest.raises(StorageQuotaBackendUnavailable, match="required"):
        with wf._instance_env_lifecycle(
            _instance(),
            tmp_path,
            enabled=True,
            log_prefix="test",
            scope=str(tmp_path),
        ):
            pytest.fail("unreachable")

    assert teardown == []


def test_authoring_setup_and_teardown_are_inside_kernel_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.products.swebench.instance_container import InstanceContainerSpec

    order: list[str] = []
    monkeypatch.setattr(wf, "_docker_available", lambda: True)
    monkeypatch.setattr(
        wf,
        "_resolve_container_spec_impl",
        lambda inst: InstanceContainerSpec(image="repo/image:tag", install_command="true"),
    )
    monkeypatch.setattr(wf, "_remove_owned_stopped_container", lambda **kwargs: None)

    class _Quota:
        def __init__(self, paths, **kwargs):
            assert paths == (tmp_path,)
            assert kwargs["require_kernel"] is True
            order.append("quota_init")

        def __enter__(self):
            order.append("quota_enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            order.append("quota_exit")

    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _Quota)

    def setup(workspace, container_name, spec, *, labels):
        order.append("container_setup")
        return "container-id"

    monkeypatch.setattr(wf, "_setup_instance_container_impl", setup)
    monkeypatch.setattr(
        wf,
        "_container_immutable_image",
        lambda container_id: "repo/image@sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        wf,
        "_teardown_instance_container",
        lambda container_id, image=None: order.append("container_teardown"),
    )

    with wf._instance_env_lifecycle(
        _instance(),
        tmp_path,
        enabled=True,
        log_prefix="test",
        scope=str(tmp_path),
    ):
        order.append("yield")

    assert order == [
        "quota_init",
        "quota_enter",
        "container_setup",
        "yield",
        "container_teardown",
        "quota_exit",
    ]


def test_generated_workspace_is_removed_before_final_quota_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.products.swebench.instance_container import InstanceContainerSpec

    workspace = tmp_path / "generated-checkout"
    workspace.mkdir()
    order: list[str] = []

    @contextlib.contextmanager
    def _workspace(*args, **kwargs):
        order.append("workspace_enter")
        try:
            yield workspace
        finally:
            workspace.rmdir()
            order.append("workspace_delete")

    monkeypatch.setattr(wf, "repo_workspace", _workspace)
    monkeypatch.setattr(wf, "_docker_available", lambda: True)
    monkeypatch.setattr(
        wf,
        "_resolve_container_spec_impl",
        lambda inst: InstanceContainerSpec(image="repo/image:tag", install_command="true"),
    )
    monkeypatch.setattr(wf, "_remove_owned_stopped_container", lambda **kwargs: None)

    class _Quota:
        def __init__(self, paths, **kwargs):
            assert paths == (workspace,)
            assert kwargs["require_kernel"] is True
            order.append("quota_init")

        def __enter__(self):
            order.append("quota_enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            order.append("quota_initial_release")

        def retire_kernel_lease(self):
            assert not workspace.exists()
            order.append("quota_final_retirement")

    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _Quota)
    monkeypatch.setattr(
        wf,
        "_setup_instance_container_impl",
        lambda *args, **kwargs: order.append("container_setup") or "container-id",
    )
    monkeypatch.setattr(
        wf,
        "_container_immutable_image",
        lambda container_id: "repo/image@sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        wf,
        "_teardown_instance_container",
        lambda *args, **kwargs: order.append("container_teardown"),
    )

    with wf._repo_workspace_with_quota_retirement(
        _instance(), cache_dir=None,
    ) as (managed_workspace, retirements):
        with wf._instance_env_lifecycle(
            _instance(),
            managed_workspace,
            enabled=True,
            log_prefix="test",
            scope=str(tmp_path),
            quota_retirements=retirements,
        ):
            order.append("yield")

    assert order == [
        "workspace_enter",
        "quota_init",
        "quota_enter",
        "container_setup",
        "yield",
        "container_teardown",
        "quota_initial_release",
        "workspace_delete",
        "quota_final_retirement",
    ]


def test_authoring_image_resolution_failure_tears_down_inside_kernel_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.products.swebench.instance_container import InstanceContainerSpec

    order: list[str] = []
    monkeypatch.setattr(wf, "_docker_available", lambda: True)
    monkeypatch.setattr(
        wf,
        "_resolve_container_spec_impl",
        lambda inst: InstanceContainerSpec(image="repo/image:tag", install_command="true"),
    )
    monkeypatch.setattr(wf, "_remove_owned_stopped_container", lambda **kwargs: None)

    class _Quota:
        def __init__(self, paths, **kwargs):
            pass

        def __enter__(self):
            order.append("quota_enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            order.append("quota_exit")

    monkeypatch.setattr(wf, "DirectoryGrowthQuota", _Quota)
    monkeypatch.setattr(
        wf,
        "_setup_instance_container_impl",
        lambda *args, **kwargs: "container-id",
    )

    def fail_image(container_id):
        order.append("image_resolution")
        raise RuntimeError("image resolution failed")

    monkeypatch.setattr(wf, "_container_immutable_image", fail_image)
    monkeypatch.setattr(
        wf,
        "_teardown_instance_container",
        lambda container_id, image=None: order.append("container_teardown"),
    )

    with pytest.raises(RuntimeError, match="image resolution failed"):
        with wf._instance_env_lifecycle(
            _instance(),
            tmp_path,
            enabled=True,
            log_prefix="test",
            scope=str(tmp_path),
        ):
            pytest.fail("unreachable")

    assert order == [
        "quota_enter",
        "image_resolution",
        "container_teardown",
        "quota_exit",
    ]


def test_worker_preserves_completed_outcome_when_quota_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.workflows import architect

    inst = _instance()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "root"
    digest = "test/image@sha256:" + "a" * 64

    @contextlib.contextmanager
    def _workspace(*args, **kwargs):
        yield workspace

    @contextlib.contextmanager
    def _lifecycle(*args, **kwargs):
        try:
            yield {"container_id": "container", "image_digest": digest}
        finally:
            raise StorageQuotaCleanupError("release helper unavailable")

    @contextlib.contextmanager
    def _child_operation(*args, **kwargs):
        yield

    monkeypatch.setattr(wf, "repo_workspace", _workspace)
    monkeypatch.setattr(wf, "_instance_env_lifecycle", _lifecycle)
    monkeypatch.setattr(wf._outcome, "child_operation", _child_operation)
    monkeypatch.setattr(architect, "run_architect", lambda **kwargs: 0)
    monkeypatch.setattr(
        wf,
        "_pool_checkpoint_fingerprint",
        lambda payload, *, image_digest: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        wf,
        "_architect_cost_for_operation",
        lambda *args: (1.75, False),
    )
    monkeypatch.setattr(
        wf,
        "_architect_usage_for_operation",
        lambda *args: (123, 45, 3, False),
    )
    monkeypatch.setattr(
        wf,
        "_verify_submitted_candidate",
        lambda instance, outcome, **kwargs: replace(
            outcome,
            pristine_verification=_passed_pristine_report(
                outcome.model_patch,
                instance,
                kwargs["image_digest"],
            ),
        ),
    )
    monkeypatch.setattr(
        wf,
        "_spec_to_outcome",
        lambda *args, **kwargs: wf.InstanceOutcome(
            instance_id=inst.instance_id,
            classification="success",
            model_patch="diff --git a/x.py b/x.py\n",
            cost_usd=0.0,
            diff_bytes=27,
            files_modified=("x.py",),
        ),
    )
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(root),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "parent"),
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        routed_backend="stub",
        routed_model="model",
    )

    outcome = wf._pool_worker_unleased(payload)

    assert outcome.classification == "hard_fail_storage_quota_cleanup"
    assert outcome.model_patch == "diff --git a/x.py b/x.py\n"
    assert outcome.cost_usd == 1.75
    assert outcome.input_tokens == 123
    assert outcome.output_tokens == 45
    assert outcome.planning_backend_calls == 3
    prior = wf._existing_predictions_by_id(
        root, (inst.instance_id,),
    )[inst.instance_id]
    assert prior.classification == "hard_fail_storage_quota_cleanup"
    assert prior.model_patch == outcome.model_patch
    assert prior.cost_usd == outcome.cost_usd


@pytest.mark.parametrize(
    ("error_factory", "classification"),
    (
        (
            lambda: StorageQuotaBackendUnavailable("helper unavailable"),
            "hard_fail_storage_quota_backend",
        ),
        (
            lambda: StorageQuotaExceeded("growth exceeded"),
            "hard_fail_storage_quota",
        ),
    ),
)
def test_worker_distinguishes_quota_backend_failure_from_growth_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_factory,
    classification: str,
):
    inst = _instance()
    workspace = tmp_path / classification
    workspace.mkdir()

    @contextlib.contextmanager
    def _workspace(*args, **kwargs):
        yield workspace

    @contextlib.contextmanager
    def _failed_lifecycle(*args, **kwargs):
        raise error_factory()
        yield  # pragma: no cover - makes this a contextmanager generator

    monkeypatch.setattr(wf, "repo_workspace", _workspace)
    monkeypatch.setattr(wf, "_instance_env_lifecycle", _failed_lifecycle)
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(tmp_path / "root"),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "parent"),
    )

    outcome = wf._pool_worker_unleased(payload)

    assert outcome.classification == classification
    assert outcome.cost_usd == 0.0
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0


def test_terminal_prediction_resumes_from_per_instance_corpus(tmp_path: Path):
    inst = _instance()
    corpus_dir = tmp_path / "instances" / inst.instance_id / "corpus"
    corpus = Corpus(corpus_dir)
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    patch = "diff --git a/x b/x\n"
    image_digest = "test/image@sha256:" + "a" * 64
    outcome = wf.InstanceOutcome(
        instance_id=inst.instance_id,
        classification="success",
        model_patch=patch,
        cost_usd=2.0,
        diff_bytes=len(patch.encode("utf-8")),
        files_modified=("x",),
        eval_status="resolved",
        pristine_verification=_passed_pristine_report(
            patch, inst, image_digest,
        ),
    )
    wf._persist_surfaced_prediction(
        corpus,
        inst,
        outcome,
        backend_name="codex",
        model="gpt-test",
        image_digest=image_digest,
    )

    found = wf._existing_predictions_by_id(
        tmp_path, (inst.instance_id, "missing"),
    )
    assert set(found) == {inst.instance_id}
    assert found[inst.instance_id].model_patch == outcome.model_patch
    assert found[inst.instance_id].model == "gpt-test"
    assert found[inst.instance_id].classification == "success"
    assert found[inst.instance_id].candidate_capture is not None
    assert found[inst.instance_id].candidate_capture.diff_ref.startswith(
        "sha256:"
    )
    assert found[inst.instance_id].candidate_eligibility.status == "eligible"
    assert found[inst.instance_id].submission_decision.submitted is True
    assert (
        found[inst.instance_id].submission_decision.submitted_patch_ref
        == found[inst.instance_id].candidate_capture.diff_ref
    )

    spec = corpus.latest_spec()
    marker = corpus.get_record(
        spec.implementation_refs[-1], as_type=Implementation,
    )
    transcript = json.loads(corpus.get_artifact(marker.transcript_ref))
    assert transcript["schema_version"] == "crucible.swebench.prediction.v5"
    assert transcript["pristine_verification"]["status"] == "passed"
    assert found[inst.instance_id].pristine_verification.status == "passed"
    assert transcript["candidate_capture"]["diff_ref"] == str(marker.diff_ref)
    assert transcript["candidate_eligibility"] == {
        "blockers": [],
        "status": "eligible",
    }
    assert transcript["submission_decision"]["submitted"] is True


def test_suppressed_candidate_is_persisted_separately_from_empty_submission(
    tmp_path: Path,
):
    inst = _instance()
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    patch = b"diff --git a/x b/x\n"
    candidate_ref = corpus.put_artifact(
        patch, kind="diffs", name_hint="candidate.diff",
    )
    outcome = wf.InstanceOutcome(
        instance_id=inst.instance_id,
        classification="architect_rc_1",
        model_patch="",
        cost_usd=2.0,
        diff_bytes=0,
        files_modified=(),
        candidate_capture=wf.PredictionCandidateCapture(
            source_patch=patch,
            diff_ref=str(candidate_ref),
            sha256=str(candidate_ref).removeprefix("sha256:"),
            diff_bytes=len(patch),
            files_modified=("x",),
            origin=wf.CandidateOrigin.CORPUS_FALLBACK,
            capture_status=wf.CandidateCaptureStatus.SOURCE_FILTERED,
            producer_slice_id="breach_defender_r2",
            producer_classification="success",
        ),
        candidate_eligibility=wf.PredictionCandidateEligibility(
            status=wf.CandidateEligibilityStatus.INELIGIBLE,
            blockers=(wf.CandidateEligibilityBlocker.ARCHITECT_FAILED,),
        ),
        submission_decision=wf.PredictionSubmissionDecision(
            policy=wf.PredictionSubmissionPolicy.VERIFIED_SUCCESS_ONLY,
            submitted=False,
            override_requested=False,
            override_used=False,
        ),
    )

    wf._persist_surfaced_prediction(
        corpus, inst, outcome, backend_name="codex", model="gpt-test",
    )

    spec = corpus.latest_spec()
    marker = corpus.get_record(
        spec.implementation_refs[-1], as_type=Implementation,
    )
    assert corpus.get_artifact(marker.diff_ref) == b""
    transcript = json.loads(corpus.get_artifact(marker.transcript_ref))
    assert transcript["candidate_capture"]["diff_ref"] == str(candidate_ref)
    assert corpus.get_artifact(
        transcript["candidate_capture"]["diff_ref"]
    ) == patch
    assert transcript["candidate_eligibility"] == {
        "blockers": ["architect_failed"],
        "status": "ineligible",
    }
    assert transcript["submission_decision"] == {
        "override_requested": False,
        "override_used": False,
        "policy": "verified_success_only",
        "submitted": False,
        "submitted_patch_ref": None,
    }

    resumed = wf._existing_predictions_by_id(
        tmp_path, (inst.instance_id,),
    )[inst.instance_id]
    assert resumed.model_patch == ""
    assert resumed.candidate_capture.source_patch == patch
    assert resumed.submission_decision.submitted is False


def test_legacy_v3_marker_without_capture_metadata_uses_safe_defaults(
    tmp_path: Path,
):
    inst = _instance()
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    diff_ref = corpus.put_artifact(
        b"diff --git a/x b/x\n", kind="diffs", name_hint="legacy.diff",
    )
    transcript_ref = corpus.put_artifact(
        json.dumps({
            "schema_version": "crucible.swebench.prediction.v3",
            "instance_id": inst.instance_id,
            "eval_status": "resolved",
            "cost_usd": 1.0,
            "cost_incomplete": False,
            "checkpoint_fingerprint": "sha256:legacy",
            "image_digest": "test/image@sha256:" + "a" * 64,
            "assigned_budget_usd": None,
            # Historical v3 marker: no prediction_capture field.
        }).encode(),
        kind="transcripts",
        name_hint="legacy.json",
    )
    marker = Implementation(
        slice_id=f"swebench/{inst.instance_id}",
        base_commit=inst.base_commit,
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=("x",),
        scope_escapes=(),
        classification="success",
        execution=ExecutionContext(
            timestamp="2026-01-01T00:00:00+00:00",
            machine="test",
            cwd=str(tmp_path),
            toolchain={},
            backend="stub",
            model="legacy-model",
            recipe="test",
            recipe_commit=None,
            gpu_model=None,
            run_id=None,
        ),
        cost_usd=0.0,
    )
    corpus.commit(implementations=(marker,))

    resumed = wf._existing_predictions_by_id(
        tmp_path, (inst.instance_id,),
    )[inst.instance_id]

    assert resumed.stripped_test_paths == ()
    assert resumed.stripped_test_diff_bytes == 0
    assert resumed.candidate_capture is not None
    assert resumed.candidate_eligibility.status == "eligible"
    assert resumed.submission_decision.policy == "legacy_v3"
    assert resumed.raw_workspace_diff_bytes == 0
    assert resumed.diff_provenance == "none"


def test_hidden_eval_commit_is_itself_resume_authority_after_crash(
    tmp_path: Path,
    monkeypatch,
):
    """A crash after the oracle commit cannot cause a second oracle run."""

    from crucible.products.swebench.eval import EvalReport, InstanceEvalReport

    inst = _instance()
    corpus = Corpus(
        tmp_path / "instances" / inst.instance_id / "corpus"
    )
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    digest = "test/image@sha256:" + "a" * 64
    fingerprint = "sha256:" + "b" * 64
    patch = "diff --git a/x b/x\n"
    outcome = wf.InstanceOutcome(
        instance_id=inst.instance_id,
        classification="success",
        model_patch=patch,
        cost_usd=1.25,
        diff_bytes=len(patch.encode("utf-8")),
        files_modified=("x",),
        pristine_verification=_passed_pristine_report(patch, inst, digest),
    )
    report = InstanceEvalReport(
        instance_id=inst.instance_id,
        resolved=True,
        patch_applied=True,
        raw_report={"resolved": True},
    )
    monkeypatch.setattr(
        wf,
        "run_evaluation",
        lambda *args, **kwargs: EvalReport(
            run_id=f"inline_{inst.instance_id}",
            model="model",
            output_dir=tmp_path / "harness",
            per_instance=(report,),
        ),
    )
    monkeypatch.setattr(wf, "_stamp_inline_reclassification", lambda *a, **k: None)

    def _marker(result):
        finalized = wf._replace_outcome_eval_status(outcome, True, result)
        finalized = wf.dataclasses.replace(
            finalized,
            producer_model="model",
            checkpoint_fingerprint=fingerprint,
            image_digest=digest,
        )
        return wf._build_surfaced_prediction_marker(
            corpus,
            inst,
            finalized,
            backend_name="stub",
            model="model",
            checkpoint_fingerprint=fingerprint,
            image_digest=digest,
        )

    evaluated = wf._evaluate_prediction_inline(
        inst,
        outcome,
        "model",
        instance_corpus=corpus,
        instance_log_dir=tmp_path / "logs",
        terminal_marker_factory=_marker,
    )
    assert evaluated is report

    # Simulate death here: no later _persist_surfaced_prediction call occurs.
    latest = corpus.latest_spec()
    assert latest.version == 1
    assert len(latest.measurement_refs) == 2
    assert len(latest.implementation_refs) == 1
    resumed = wf._existing_predictions_by_id(
        tmp_path,
        (inst.instance_id,),
    )[inst.instance_id]
    assert resumed.model_patch == outcome.model_patch
    assert resumed.eval_status == "resolved"
    assert resumed.checkpoint_fingerprint == fingerprint


def test_terminal_persistence_failure_propagates_through_task_lease(
    tmp_path: Path,
    monkeypatch,
):
    """An unpublished terminal cannot be surfaced as worker success."""

    state = {"leased": False}

    @contextlib.contextmanager
    def _lease(*args, **kwargs):
        state["leased"] = True
        try:
            yield tmp_path / "lease"
        finally:
            state["leased"] = False

    def _publication_failure(payload):
        assert state["leased"] is True
        raise wf._TerminalPersistenceError("disk full")

    monkeypatch.setattr(wf, "task_execution_lease", _lease)
    monkeypatch.setattr(wf, "_pool_worker_unleased", _publication_failure)
    payload = wf._PoolPayload(
        instance=_instance(),
        workspace_root_spec_dir=str(tmp_path),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "parent"),
    )

    with pytest.raises(wf._TerminalPersistenceError, match="disk full"):
        wf._pool_worker(payload)
    assert state["leased"] is False


def test_pool_checkpoint_binds_exact_routing_source_bytes(tmp_path: Path):
    payload = wf._PoolPayload(
        instance=_instance(),
        workspace_root_spec_dir=str(tmp_path),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "parent"),
        routing_fingerprint="sha256:" + "1" * 64,
        routing_source_sha256="sha256:" + "2" * 64,
    )

    baseline = wf._pool_checkpoint_fingerprint(
        payload,
        image_digest="image@sha256:" + "3" * 64,
    )
    changed = wf._pool_checkpoint_fingerprint(
        replace(payload, routing_source_sha256="sha256:" + "4" * 64),
        image_digest="image@sha256:" + "3" * 64,
    )

    assert changed != baseline


def test_worker_rechecks_matching_checkpoint_with_original_assigned_budget(
    tmp_path: Path, monkeypatch,
):
    inst = _instance()
    root = tmp_path / "root"
    corpus = Corpus(root / "instances" / inst.instance_id / "corpus")
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    digest = "test/image@sha256:" + "a" * 64
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(root),
        backend_config_path=None,
        max_turns=5,
        budget_usd=1.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "unused-parent"),
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        eval_timeout=10,
        routed_backend="stub",
        routed_model="model",
    )
    fingerprint = wf._pool_checkpoint_fingerprint(
        payload, image_digest=digest,
    )
    prior = wf.InstanceOutcome(
        instance_id=inst.instance_id,
        classification="success",
        model_patch="prior patch",
        cost_usd=2.0,
        diff_bytes=11,
        files_modified=("x",),
        eval_status="resolved",
        stripped_test_paths=("tests/test_x.py",),
        stripped_test_diff_bytes=321,
        raw_workspace_diff_bytes=654,
        diff_provenance="workspace+corpus_fallback",
        pristine_verification=_passed_pristine_report(
            "prior patch", inst, digest,
        ),
    )
    wf._persist_surfaced_prediction(
        corpus,
        inst,
        prior,
        backend_name="stub",
        model="old-model",
        checkpoint_fingerprint=fingerprint,
        image_digest=digest,
        assigned_budget_usd=payload.budget_usd,
    )
    monkeypatch.setattr(
        wf,
        "_pool_worker_unleased",
        lambda payload: pytest.fail("matching checkpoint must not execute"),
    )
    verified: list[str] = []
    monkeypatch.setattr(wf, "verify_immutable_image", verified.append)

    # A parallel restart may offer a different transient fair share. Reuse is
    # validated against the budget persisted with the completed attempt.
    resumed = wf._pool_worker(
        wf.dataclasses.replace(payload, budget_usd=0.5)
    )

    assert resumed.resumed is True
    assert resumed.model_patch == "prior patch"
    assert resumed.producer_model == "old-model"
    assert resumed.cost_usd == 0.0
    assert resumed.stripped_test_paths == ("tests/test_x.py",)
    assert resumed.stripped_test_diff_bytes == 321
    assert resumed.raw_workspace_diff_bytes == 654
    assert resumed.diff_provenance == "workspace+corpus_fallback"
    assert verified == [digest]


def test_worker_reports_busy_cross_process_task_lease(tmp_path: Path):
    inst = _instance()
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(tmp_path),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "unused-parent"),
    )
    with task_execution_lease(
        tmp_path, namespace="swebench", task_id=inst.instance_id,
    ):
        outcome = wf._pool_worker(payload)
    assert outcome.classification == "hard_fail_task_lease_busy"


def test_worker_target_reruns_reusable_test_only_outcome(
    tmp_path: Path,
    monkeypatch,
):
    inst = _instance()
    prior = wf._PriorPrediction(
        model_patch="",
        model="old-model",
        classification="soft_fail_test_only_changes",
        eval_status="skipped_empty",
        cost_usd=0.0,
        cost_incomplete=False,
        checkpoint_fingerprint="sha256:matching",
        image_digest="test/image@sha256:" + "a" * 64,
        assigned_budget_usd=10.0,
    )
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(tmp_path),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "parent"),
        rerun_status=("soft_fail_test_only_changes",),
    )
    monkeypatch.setattr(
        wf,
        "_existing_predictions_by_id",
        lambda *args: {inst.instance_id: prior},
    )
    monkeypatch.setattr(
        wf,
        "_pool_checkpoint_fingerprint",
        lambda *args, **kwargs: prior.checkpoint_fingerprint,
    )
    sentinel = wf.InstanceOutcome(
        instance_id=inst.instance_id,
        classification="success",
        model_patch="new patch",
        cost_usd=0.0,
        diff_bytes=9,
        files_modified=("x.py",),
    )
    calls: list[wf._PoolPayload] = []

    def _rerun(candidate):
        calls.append(candidate)
        return sentinel

    monkeypatch.setattr(wf, "_pool_worker_unleased", _rerun)

    assert wf._pool_worker_with_routing(payload) is sentinel
    assert calls == [payload]


def test_stale_managed_policy_migrates_atomically_and_idempotently(
    tmp_path: Path,
):
    inst = _instance()
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    operator = AuthoringInstructionConstraint(
        "Preserve this operator-owned instruction.",
    )
    language = LanguagesConstraint(languages=("python",))
    _seed_legacy_managed_plan(
        corpus,
        inst,
        extra_constraints=(operator, language),
    )
    execution = wf.exec_ctx_now("test")
    diff_ref = corpus.put_artifact(
        b"diff --git a/pkg/core.py b/pkg/core.py\n",
        kind="diffs",
        name_hint="legacy.diff",
    )
    transcript_ref = corpus.put_artifact(
        b"[]", kind="transcripts", name_hint="legacy.json",
    )
    raw_ref = corpus.put_artifact(
        b"ok", kind="runner_outputs", name_hint="legacy.txt",
    )
    implementation = Implementation(
        slice_id="source",
        base_commit=inst.base_commit,
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=("pkg/core.py",),
        scope_escapes=(),
        classification="success",
        execution=execution,
    )
    verdict = Verdict(
        rubric="test",
        decision="pass",
        findings=(),
        artifact_ref=diff_ref,
        artifact_commit=inst.base_commit,
        reasoning="legacy evidence",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name=None,
        transcript_ref=None,
        cost_usd=0.0,
        execution=execution,
        kind="review",
    )
    measurement = Measurement(
        bench_id="legacy/dev_verify",
        commit=inst.base_commit,
        config_hash="legacy-config",
        values={"passed": 1.0},
        raw_output_ref=raw_ref,
        execution=execution,
    )
    corpus.commit(
        implementations=(implementation,),
        verdicts=(verdict,),
        measurements=(measurement,),
    )
    preimage = corpus.latest_spec()
    assert preimage is not None
    preimage_hash = hash_record(preimage)
    preimage_plan_policy_hash = hash_record(preimage.plan_policy)
    preimage_refs = (
        preimage.implementation_refs,
        preimage.verdict_refs,
        preimage.measurement_refs,
        preimage.lifecycle_event_refs,
    )

    preparation = wf._prepare_instance_corpus(
        corpus,
        synthesize_starter_spec(inst),
        instance_id=inst.instance_id,
    )

    assert preparation == wf._InstanceCorpusPreparation(
        had_lineage=True,
        requires_replan=True,
    )
    migrated = corpus.latest_spec()
    assert migrated is not None
    assert migrated.parent == preimage_hash
    assert migrated.tier_dag is None
    assert migrated.plan_policy == preimage.plan_policy
    assert hash_record(migrated.plan_policy) == preimage_plan_policy_hash
    assert migrated.constraints == (
        *SWEBENCH_PLANNER_CONSTRAINTS,
        operator,
        language,
    )
    assert migrated.implementation_refs == preimage_refs[0]
    assert migrated.verdict_refs == preimage_refs[1]
    assert migrated.measurement_refs == preimage_refs[2]
    assert migrated.lifecycle_event_refs[:-1] == preimage_refs[3]
    event = corpus.get_record(
        migrated.lifecycle_event_refs[-1],
        as_type=LifecycleEvent,
    )
    assert event.kind == "authoring_policy_migrated"
    assert event.affected_versions == (migrated.version,)
    reason = json.loads(event.reason)
    assert reason == {
        "constraints_after": hash_record(migrated.constraints),
        "constraints_before": hash_record(preimage.constraints),
        "instance_id": inst.instance_id,
        "managed_after": hash_record(SWEBENCH_PLANNER_CONSTRAINTS),
        "managed_before": hash_record((
            preimage.constraints[0],
            preimage.constraints[-1],
        )),
        "plan_policy": preimage_plan_policy_hash,
        "policy": SWEBENCH_AUTHORING_POLICY_ID,
        "retired_tier_dag": hash_record(preimage.tier_dag),
        "schema": "crucible.swebench.authoring-policy-migration.v1",
        "source_spec": preimage_hash,
    }
    # Historical manifests and their hash-bound records remain readable.
    assert hash_record(corpus.get_spec(preimage.version)) == preimage_hash
    assert corpus.get_artifact(diff_ref).startswith(b"diff --git")

    version = migrated.version
    second = wf._prepare_instance_corpus(
        corpus,
        synthesize_starter_spec(inst),
        instance_id=inst.instance_id,
    )
    assert second == wf._InstanceCorpusPreparation(
        had_lineage=True,
        requires_replan=False,
    )
    assert corpus.latest_spec().version == version


def test_policy_migration_rebases_over_concurrent_record_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    inst = _instance()
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    _seed_legacy_managed_plan(corpus, inst)
    concurrent_event = LifecycleEvent(
        kind="concurrent_evidence",
        reason="attached between migration read and commit",
        affected_versions=(),
        execution=wf.exec_ctx_now("test"),
    )
    original_commit = corpus.commit
    injected_refs = []

    def _commit_after_concurrent_write(**kwargs):
        result = Corpus(corpus.spec_dir).commit(
            lifecycle_events=(concurrent_event,),
        )
        injected_refs.extend(result.refs.lifecycle_event_refs)
        return original_commit(**kwargs)

    monkeypatch.setattr(corpus, "commit", _commit_after_concurrent_write)

    preparation = wf._prepare_instance_corpus(
        corpus,
        synthesize_starter_spec(inst),
        instance_id=inst.instance_id,
    )

    assert preparation.requires_replan is True
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.tier_dag is None
    assert injected_refs[0] in latest.lifecycle_event_refs
    assert corpus.get_record(
        injected_refs[0], as_type=LifecycleEvent,
    ).kind == "concurrent_evidence"
    assert corpus.get_record(
        latest.lifecycle_event_refs[-1], as_type=LifecycleEvent,
    ).kind == "authoring_policy_migrated"


def test_policy_migration_preserves_old_resume_artifact_but_vetoes_reuse(
    tmp_path: Path,
):
    from crucible.workflows._partition import (
        attach_resume_identity,
        partition_for_resume,
    )

    inst = _instance()
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    planned = _seed_legacy_managed_plan(corpus, inst)
    tier = planned.tier_dag.tiers[0]
    slice_ = tier.slices[0]
    implementation = Implementation(
        slice_id=slice_.id,
        base_commit=inst.base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            b"diff --git a/pkg/core.py b/pkg/core.py\n",
            kind="diffs",
            name_hint="old-plan.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="old-plan.json",
        ),
        files_modified=("pkg/core.py",),
        scope_escapes=(),
        classification="success",
        execution=wf.exec_ctx_now("test"),
    )
    bound = attach_resume_identity(
        corpus, planned, tier, slice_, implementation,
    )
    resume_ref = bound.artifacts["resume_identity"]
    resume_bytes = corpus.get_artifact(resume_ref)
    corpus.commit(implementations=(bound,))
    before_migration = corpus.latest_spec()
    landed, pending = partition_for_resume(
        corpus, tier, spec=before_migration,
    )
    assert landed == {slice_.id: bound}
    assert pending == ()

    wf._prepare_instance_corpus(
        corpus,
        synthesize_starter_spec(inst),
        instance_id=inst.instance_id,
    )
    execution = wf.exec_ctx_now("test")
    replanned = corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="planned",
            reason="same DAG accepted under the current authoring policy",
            affected_versions=(version,),
            execution=execution,
        ),
        transform=lambda candidate: replace(
            candidate,
            tier_dag=_stale_cleanup_dag(),
        ),
    ).spec

    landed, pending = partition_for_resume(
        corpus,
        replanned.tier_dag.tiers[0],
        spec=replanned,
    )
    assert landed == {}
    assert pending == replanned.tier_dag.tiers[0].slices
    assert corpus.get_artifact(resume_ref) == resume_bytes
    assert hash_record(bound) in replanned.implementation_refs


def test_worker_forces_replan_after_managed_policy_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.workflows import architect

    inst = _instance()
    root = tmp_path / "root"
    corpus_dir = root / "instances" / inst.instance_id / "corpus"
    corpus = Corpus(corpus_dir)
    _seed_legacy_managed_plan(corpus, inst)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    digest = "test/image@sha256:" + "a" * 64

    @contextlib.contextmanager
    def _workspace(*args, **kwargs):
        yield workspace, []

    @contextlib.contextmanager
    def _lifecycle(*args, **kwargs):
        yield {"container_id": "container", "image_digest": digest}

    @contextlib.contextmanager
    def _child_operation(*args, **kwargs):
        yield

    captured = {}

    def _architect(**kwargs):
        captured.update(kwargs)
        live = Corpus(kwargs["spec"]).latest_spec()
        assert live is not None
        assert live.tier_dag is None
        assert live.constraints == SWEBENCH_PLANNER_CONSTRAINTS
        return 0

    monkeypatch.setattr(
        wf, "_repo_workspace_with_quota_retirement", _workspace,
    )
    monkeypatch.setattr(wf, "_instance_env_lifecycle", _lifecycle)
    monkeypatch.setattr(wf._outcome, "child_operation", _child_operation)
    monkeypatch.setattr(architect, "run_architect", _architect)
    monkeypatch.setattr(
        wf,
        "_pool_checkpoint_fingerprint",
        lambda *args, **kwargs: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        wf, "_architect_cost_for_operation", lambda *args: (0.0, False),
    )
    monkeypatch.setattr(
        wf,
        "_architect_usage_for_operation",
        lambda *args: (0, 0, 0, False),
    )
    monkeypatch.setattr(
        wf,
        "_spec_to_outcome",
        lambda *args, **kwargs: wf.InstanceOutcome(
            instance_id=inst.instance_id,
            classification="success",
            model_patch="",
            cost_usd=0.0,
            diff_bytes=0,
            files_modified=(),
        ),
    )
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(root),
        backend_config_path=None,
        max_turns=1,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(tmp_path / "parent"),
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        routed_backend="stub",
        routed_model="model",
    )

    wf._pool_worker_unleased(payload)

    assert captured["replan"] is True
    assert captured["restart"] is True
    assert captured["plan_candidate_transform"] is not None


def test_existing_lineage_is_validated_and_signals_restart(tmp_path: Path):
    inst = _instance()
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    starter = synthesize_starter_spec(inst)
    assert wf._prepare_instance_corpus(
        corpus, starter, instance_id=inst.instance_id,
    ) == wf._InstanceCorpusPreparation(False, False)
    assert wf._prepare_instance_corpus(
        corpus, starter, instance_id=inst.instance_id,
    ) == wf._InstanceCorpusPreparation(True, False)

    changed = synthesize_starter_spec(
        SwebenchInstance(
            **{**inst.__dict__, "problem_statement": "different task"},
        ),
    )
    with pytest.raises(ValueError, match="conflicts with existing Corpus"):
        wf._prepare_instance_corpus(
            corpus, changed, instance_id=inst.instance_id,
        )


def test_retry_rollup_excludes_prior_attempt_cost_and_patch(tmp_path: Path):
    import subprocess

    repo = tmp_path / "workspace"
    repo.mkdir()

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True,
        ).stdout

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "old.py").write_text("OLD = 0\n")
    (repo / "new.py").write_text("NEW = 0\n")
    git("add", ".")
    git("commit", "-qm", "base")
    inst = replace(_instance(), base_commit=git("rev-parse", "HEAD").decode().strip())

    (repo / "old.py").write_text("OLD = 1\n")
    prior_patch = git("diff", "--binary", "HEAD").decode()
    git("checkout", "--", ".")
    (repo / "new.py").write_text("NEW = 1\n")
    current_patch = git("diff", "--binary", "HEAD")
    git("checkout", "--", ".")

    image_digest = "test/image@sha256:" + "a" * 64
    corpus = Corpus(tmp_path / "instances" / inst.instance_id / "corpus")
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    prior = wf.InstanceOutcome(
        instance_id=inst.instance_id,
        classification="success",
        model_patch=prior_patch,
        cost_usd=9.0,
        diff_bytes=len(prior_patch.encode()),
        files_modified=("old.py",),
        pristine_verification=_passed_pristine_report(
            prior_patch, inst, image_digest,
        ),
    )
    wf._persist_surfaced_prediction(
        corpus,
        inst,
        prior,
        backend_name="stub",
        model="old",
        image_digest=image_digest,
    )
    base = corpus.latest_spec()
    assert base is not None
    start = len(base.implementation_refs)

    current_diff = corpus.put_artifact(
        current_patch, kind="diffs", name_hint="current.diff",
    )
    current = Implementation(
        slice_id="s0",
        base_commit=inst.base_commit,
        result_commit=None,
        diff_ref=current_diff,
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="current.json",
        ),
        files_modified=("new.py",),
        scope_escapes=(),
        classification="success",
        execution=ExecutionContext(
            timestamp="2026-01-01T00:00:00+00:00",
            machine="test", cwd=str(tmp_path), toolchain={},
            backend="stub", model="new", recipe="test",
            recipe_commit=None, gpu_model=None, run_id=None,
        ),
        cost_usd=1.25,
    )
    corpus.put_record(current, kind="implementations")
    corpus.put_spec(base.with_implementation(current))

    outcome = wf._spec_to_outcome(
        corpus, inst, 0, workspace=repo,
        implementation_start_index=start,
        verdict_start_index=len(base.verdict_refs),
    )
    assert outcome.cost_usd == pytest.approx(1.25)
    assert outcome.model_patch == current_patch.decode()
    assert outcome.files_modified == ("new.py",)

    assert outcome.candidate_capture is not None
    assert outcome.candidate_capture.origin == wf.CandidateOrigin.CORPUS_FALLBACK


@pytest.mark.parametrize("planner_count,task_budget", [(1, None), (3, 10.0)])
def test_pool_payload_execution_policy_is_forwarded_to_architect(
    tmp_path: Path, monkeypatch, planner_count: int, task_budget: float | None,
):
    inst = _instance()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    @contextlib.contextmanager
    def _workspace(*args, **kwargs):
        yield workspace

    @contextlib.contextmanager
    def _sandbox(*args, **kwargs):
        yield {
            "container_id": "test-container",
            "image_digest": "test/image@sha256:" + "a" * 64,
        }

    seen: list[dict] = []
    monkeypatch.setattr(wf, "repo_workspace", _workspace)
    monkeypatch.setattr(wf, "_instance_env_lifecycle", _sandbox)
    monkeypatch.setattr(
        "crucible.workflows.architect.run_architect",
        lambda **kwargs: seen.append(kwargs) or 2,
    )
    parent_run = _prepare_swebench_parent(tmp_path / "runs" / "swebench-parent")
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(tmp_path / "root"),
        backend_config_path=None,
        max_turns=1,
        budget_usd=task_budget,
        planner_count=planner_count,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(parent_run),
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        restart=True,
    )
    wf._pool_worker(payload)
    assert len(seen) == 1
    assert seen[0]["restart"] is True
    assert seen[0]["budget_usd"] == task_budget
    assert seen[0]["planner_count"] == planner_count
    policy = seen[0]["planning_policy"]
    assert policy.planner_count == planner_count
    assert policy.planning_budget_share == (0.40 if planner_count > 1 else None)
    assert policy.execution_fingerprint == wf._pool_checkpoint_fingerprint(
        payload, image_digest="test/image@sha256:" + "a" * 64,
    )


class _Routing:
    class _Route:
        backend = "stub"
        model = "stub-model"

    def for_role(self, role):
        return self._Route()


class _Handle:
    def __init__(self):
        self.events: list[dict] = []

    def write_event(self, event):
        self.events.append(event)


def test_parallel_budget_drains_and_accounts_all_inflight_results(
    tmp_path: Path, monkeypatch,
):
    instances = tuple(_instance(f"org__repo-{i}") for i in range(3))
    launched: list[str] = []
    reservations: list[tuple[float | None, int]] = []

    def _worker(payload):
        launched.append(payload.instance.instance_id)
        reservations.append((payload.budget_usd, payload.planner_count))
        return wf.InstanceOutcome(
            instance_id=payload.instance.instance_id,
            classification="success",
            model_patch=f"patch-{payload.instance.instance_id}",
            cost_usd=0.75,
            diff_bytes=5,
            files_modified=("x",),
        )

    monkeypatch.setattr(wf, "_pool_worker", _worker)
    monkeypatch.setattr(wf, "_docker_available", lambda: True)
    monkeypatch.setattr(wf, "_check_disk_headroom", lambda: None)
    args = argparse.Namespace(
        enable_build_verify=False,
        max_concurrent_instances=2,
        backend_config=None,
        spec=tmp_path / "spec",
        max_turns=1,
        planner_count=3,
        budget_usd_per_instance=None,
        budget_usd=1.0,
        repo_cache_dir=None,
        include_breach=False,
        include_review=False,
        evaluate_inline=False,
        eval_timeout=1,
        rerun=False,
    )
    run_dir = tmp_path / "run"
    predictions = run_dir / "predictions.jsonl"
    handle = _Handle()
    costs = CostAccumulator()
    outcomes, spent, halted, accounting_incomplete = (
        wf._run_implement_parallel(
        instances,
        args=args,
        run_dir=run_dir,
        handle=handle,
        predictions_path=predictions,
        routing=_Routing(),
        costs=costs,
        )
    )

    # Both initially in-flight workers are retained and charged even though
    # their combined cost crosses the cap; only the third is never launched.
    assert len(launched) == 2
    assert reservations == [(0.5, 3), (0.5, 3)]
    assert len(outcomes) == 2
    assert len(handle.events) == 2
    assert spent == pytest.approx(1.5)
    assert costs.total_cost_usd == pytest.approx(1.5)
    assert halted is True
    assert accounting_incomplete is False
    lines = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert {line["instance_id"] for line in lines} == set(launched)


def test_parallel_cost_incomplete_stops_new_scheduling(
    tmp_path: Path, monkeypatch,
):
    instances = tuple(_instance(f"org__repo-{i}") for i in range(3))
    launched: list[str] = []

    def _worker(payload):
        launched.append(payload.instance.instance_id)
        return wf.InstanceOutcome(
            instance_id=payload.instance.instance_id,
            classification="success",
            model_patch="patch",
            cost_usd=0.25,
            diff_bytes=5,
            files_modified=("x",),
            cost_incomplete=True,
        )

    monkeypatch.setattr(wf, "_pool_worker", _worker)
    monkeypatch.setattr(wf, "_docker_available", lambda: True)
    monkeypatch.setattr(wf, "_check_disk_headroom", lambda: None)
    args = argparse.Namespace(
        enable_build_verify=False,
        max_concurrent_instances=2,
        backend_config=None,
        spec=tmp_path / "spec",
        max_turns=1,
        planner_count=3,
        budget_usd_per_instance=None,
        budget_usd=10.0,
        repo_cache_dir=None,
        include_breach=False,
        include_review=False,
        evaluate_inline=False,
        eval_timeout=1,
        rerun=False,
    )
    handle = _Handle()

    outcomes, spent, budget_halted, accounting_incomplete = (
        wf._run_implement_parallel(
            instances,
            args=args,
            run_dir=tmp_path / "run",
            handle=handle,
            predictions_path=tmp_path / "run" / "predictions.jsonl",
            routing=_Routing(),
            costs=CostAccumulator(),
        )
    )

    assert len(launched) == 2  # already in flight; the third never starts
    assert len(outcomes) == 2
    assert spent == pytest.approx(0.5)
    assert budget_halted is False
    assert accounting_incomplete is True
