# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import replace

import pytest
import yaml

from assay import (
    ImplementationDirective,
    PlanPolicy,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from assay.corpus.store import Corpus
from assay.canonical import hash_record
from assay.types.artifacts import LifecycleEvent

import crucible.rubrics as rubric_module
import crucible.workflows._plan_provenance as provenance_module
from crucible.rubrics import load
from crucible.products.swebench.plan_normalization import (
    AuditedSwebenchPlanTransform,
)
from crucible.workflows._plan_provenance import (
    _callable_identity,
    accepted_plan_contract_state,
    bind_plan_contract,
    plan_contract_fingerprint,
)
from crucible.workflows._run import exec_ctx_now


def _spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="demo",
            repo_url="https://example.invalid/demo.git",
            base_commit="a" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Keep the first declaration for each key.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )


def _dag(intent: str = "Implement the declared precedence rule.") -> TierDAG:
    return TierDAG(tiers=(Tier(
        tier_id="t0",
        depends_on=(),
        slices=(Slice(
            id="s0",
            directive=ImplementationDirective(
                slice_id="s0",
                intent=intent,
                files_in_scope=("src/config.py",),
                creates_paths=(),
                verify_mode="tests",
                blast_radius={"max_files": 1},
                requirement_refs=(
                    "Keep the first declaration for each key.",
                ),
            ),
        ),),
    ),))


def _fingerprint(spec: Spec) -> str:
    return plan_contract_fingerprint(
        spec,
        critic_rubric=load("deepswe_plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_transform=None,
    )


def _transform_one(spec):
    return spec


def _transform_two(spec):
    return replace(spec, tier_dag=spec.tier_dag)


def _accepted_staged(corpus: Corpus, spec: Spec) -> Spec:
    event = LifecycleEvent(
        kind="planned",
        reason="accepted test plan",
        affected_versions=(spec.version + 1,),
        execution=exec_ctx_now("plan"),
    )
    event_ref = corpus.put_record(event, kind="lifecycle_events")
    return spec.with_appended_refs(
        tier_dag=_dag(),
        lifecycle_event_refs=(event_ref,),
    )


def test_plan_contract_fingerprint_tracks_root_semantics() -> None:
    spec = _spec()

    original = _fingerprint(spec)
    changed = _fingerprint(replace(
        spec,
        intent="Keep the last declaration for each key.",
    ))

    assert original.startswith("sha256:")
    assert changed != original


def test_plan_contract_fingerprint_tracks_fresh_judge_policy() -> None:
    spec = _spec()
    baseline = _fingerprint(replace(
        spec,
        plan_policy=PlanPolicy(
            fresh_judge_reassessment_min_prior_rounds=None,
        ),
    ))
    changed = _fingerprint(replace(
        spec,
        plan_policy=PlanPolicy(
            fresh_judge_reassessment_min_prior_rounds=2,
        ),
    ))

    assert changed != baseline
    assert {
        key
        for key in baseline.component_digests
        if baseline.component_digests[key] != changed.component_digests[key]
    } == {"authority.plan_policy"}


def test_bound_accepted_plan_is_fresh_only_for_matching_contract(
    tmp_path,
) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    staged = _accepted_staged(corpus, spec)
    fingerprint = _fingerprint(spec)
    bound = bind_plan_contract(
        corpus,
        staged,
        fingerprint=fingerprint,
    )

    fresh = accepted_plan_contract_state(
        corpus,
        bound,
        expected_fingerprint=fingerprint,
    )
    stale = accepted_plan_contract_state(
        corpus,
        bound,
        expected_fingerprint="sha256:" + "f" * 64,
    )

    assert fresh.status == "fresh"
    assert fresh.actual_fingerprint == fingerprint
    assert stale.status == "stale"
    assert "contract changed" in (stale.reason or "")


def test_bound_plan_is_stale_when_executable_dag_is_replaced(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    fingerprint = _fingerprint(spec)
    bound = bind_plan_contract(
        corpus,
        _accepted_staged(corpus, spec),
        fingerprint=fingerprint,
    )
    swapped = replace(
        bound,
        tier_dag=_dag("Implement a different externally visible rule."),
    )

    state = accepted_plan_contract_state(
        corpus,
        swapped,
        expected_fingerprint=fingerprint,
    )

    assert state.status == "stale"
    assert "tier_dag differs" in (state.reason or "")


def test_plan_binding_is_idempotent_and_rebinds_latest_repair(
    tmp_path,
) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    fingerprint = _fingerprint(spec)
    bound = bind_plan_contract(
        corpus,
        _accepted_staged(corpus, spec),
        fingerprint=fingerprint,
    )

    assert bind_plan_contract(
        corpus, bound, fingerprint=fingerprint,
    ) is bound

    repair_event = LifecycleEvent(
        kind="planned",
        reason="accepted validation repair",
        affected_versions=(bound.version + 1,),
        execution=exec_ctx_now("plan"),
    )
    repair_ref = corpus.put_record(repair_event, kind="lifecycle_events")
    repaired = bound.with_appended_refs(
        lifecycle_event_refs=(repair_ref,),
        tier_dag=_dag("Implement the validation-repaired rule."),
    )
    rebound = bind_plan_contract(
        corpus,
        repaired,
        fingerprint=fingerprint,
    )

    assert len(rebound.lifecycle_event_refs) == len(repaired.lifecycle_event_refs) + 1
    assert accepted_plan_contract_state(
        corpus,
        rebound,
        expected_fingerprint=fingerprint,
    ).status == "fresh"


def test_old_binding_cannot_authorize_a_later_planned_event(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    fingerprint = _fingerprint(spec)
    bound = bind_plan_contract(
        corpus,
        _accepted_staged(corpus, spec),
        fingerprint=fingerprint,
    )
    old_binding_ref = bound.lifecycle_event_refs[-1]
    later_event = LifecycleEvent(
        kind="planned",
        reason="different accepted plan",
        affected_versions=(bound.version + 1,),
        execution=exec_ctx_now("plan"),
    )
    later_ref = corpus.put_record(later_event, kind="lifecycle_events")
    forged = replace(
        bound,
        lifecycle_event_refs=(
            *bound.lifecycle_event_refs,
            later_ref,
            old_binding_ref,
        ),
    )

    state = accepted_plan_contract_state(
        corpus,
        forged,
        expected_fingerprint=fingerprint,
    )

    assert state.status == "stale"
    assert "different plan terminal" in (state.reason or "")


def test_callable_object_transform_identity_binds_class_and_call(tmp_path) -> None:
    identity = _callable_identity(
        AuditedSwebenchPlanTransform(Corpus(tmp_path)),
    )

    assert identity["kind"] == "callable_object"
    assert identity["qualname"] == "AuditedSwebenchPlanTransform"
    assert identity["module_digest"].startswith("sha256:")
    assert identity["class_digest"].startswith("sha256:")
    assert identity["call_digest"].startswith("sha256:")


def test_candidate_transform_without_source_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="source identity is unavailable"):
        _callable_identity(len)


def test_workspace_head_must_match_spec_base_and_ignores_dirt(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "base"], check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    spec = replace(
        _spec(),
        project_context=replace(_spec().project_context, base_commit=head),
    )

    clean = plan_contract_fingerprint(
        spec,
        critic_rubric=load("deepswe_plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_transform=None,
        cwd=repo,
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    dirty = plan_contract_fingerprint(
        spec,
        critic_rubric=load("deepswe_plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_transform=None,
        cwd=repo,
    )
    assert dirty == clean

    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "next"], check=True,
    )
    with pytest.raises(ValueError, match="does not match Spec base_commit"):
        plan_contract_fingerprint(
            spec,
            critic_rubric=load("deepswe_plan_critic"),
            judge_rubric=load("plan_judge"),
            candidate_transform=None,
            cwd=repo,
        )


def test_legacy_accepted_plan_fails_closed_as_stale(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    staged = _accepted_staged(corpus, spec)

    state = accepted_plan_contract_state(
        corpus,
        staged,
        expected_fingerprint=_fingerprint(spec),
    )

    assert state.status == "stale"
    assert state.reason == "accepted plan predates provenance binding"


def test_latest_nonaccepted_plan_terminal_disables_reuse(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    staged = _accepted_staged(corpus, spec)
    rejected = LifecycleEvent(
        kind="plan_rejected",
        reason="later review rejected the candidate",
        affected_versions=(staged.version,),
        execution=exec_ctx_now("plan"),
    )
    rejected_ref = corpus.put_record(rejected, kind="lifecycle_events")
    staged = replace(
        staged,
        lifecycle_event_refs=(*staged.lifecycle_event_refs, rejected_ref),
    )

    state = accepted_plan_contract_state(
        corpus,
        staged,
        expected_fingerprint=_fingerprint(spec),
    )

    assert state.status == "absent"


def test_v3_binding_persists_scoped_component_digests(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    fingerprint = _fingerprint(spec)
    bound = bind_plan_contract(
        corpus,
        _accepted_staged(corpus, spec),
        fingerprint=fingerprint,
    )

    event = corpus.get_record(
        bound.lifecycle_event_refs[-1], as_type=LifecycleEvent,
    )
    payload = provenance_module._bound_payload(event)

    assert payload is not None
    assert payload["schema_version"] == "crucible.plan-contract.v3"
    assert payload["component_digests"] == dict(
        fingerprint.component_digests
    )
    assert set(payload["component_digests"]) >= {
        "authority.intent",
        "authority.workspace",
        "planner.candidate_transform",
        "review.critic.rubric",
        "review.critic.contract",
        "review.judge.rubric",
        "review.judge.contract",
        "review.runtime_abi",
        "fingerprint.algorithm",
    }


def test_v3_stale_reason_names_changed_authority_component(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    bound = bind_plan_contract(
        corpus,
        _accepted_staged(corpus, spec),
        fingerprint=_fingerprint(spec),
    )
    changed = replace(
        spec,
        intent="Keep the last declaration for each key.",
    )

    state = accepted_plan_contract_state(
        corpus,
        bound,
        expected_fingerprint=_fingerprint(changed),
    )

    assert state.status == "stale"
    assert state.changed_components == ("authority.intent",)
    assert "authority.intent" in (state.reason or "")


def test_unselected_manifest_policy_extension_does_not_change_plan_identity(
    tmp_path,
    monkeypatch,
) -> None:
    spec = _spec()
    critic = load("deepswe_plan_critic")
    judge = load("plan_judge")
    baseline = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=None,
    )
    manifest = yaml.safe_load(
        rubric_module._MANIFEST_PATH.read_text(encoding="utf-8")
    )
    # This optional policy is executable for the unselected reviewer, but it
    # is neither selected planning authority nor a plan-review runtime ABI.
    manifest["rubrics"]["impl_breaker"][
        "finding_required_source_sets"
    ] = [["artifact"]]
    changed_manifest = tmp_path / "manifest.yaml"
    changed_manifest.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8",
    )
    monkeypatch.setattr(rubric_module, "_MANIFEST_PATH", changed_manifest)

    changed = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=None,
    )

    assert changed == baseline
    assert changed.component_digests == baseline.component_digests


@pytest.mark.parametrize("selected", ["critic", "judge"])
def test_selected_rubric_prompt_change_invalidates_only_loaded_rubric(
    selected,
) -> None:
    spec = _spec()
    critic = load("deepswe_plan_critic")
    judge = load("plan_judge")
    baseline = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=None,
    )
    if selected == "critic":
        critic = replace(critic, system_prompt=critic.system_prompt + "\nRevision.")
    else:
        judge = replace(judge, system_prompt=judge.system_prompt + "\nRevision.")

    changed = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=None,
    )

    differing = {
        key
        for key in baseline.component_digests
        if baseline.component_digests[key] != changed.component_digests[key]
    }
    assert changed != baseline
    assert differing == {f"review.{selected}.rubric"}


@pytest.mark.parametrize(
    ("selected", "severity_floor"),
    [("critic", "High"), ("judge", "Medium")],
)
def test_selected_active_contract_policy_change_invalidates_semantics(
    tmp_path,
    monkeypatch,
    selected,
    severity_floor,
) -> None:
    spec = _spec()
    critic = load("deepswe_plan_critic")
    judge = load("plan_judge")
    baseline = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=None,
    )
    selected_name = critic.name if selected == "critic" else judge.name
    manifest = yaml.safe_load(
        rubric_module._MANIFEST_PATH.read_text(encoding="utf-8")
    )
    manifest["rubrics"][selected_name]["severity_floor"] = severity_floor
    changed_manifest = tmp_path / f"{selected}.yaml"
    changed_manifest.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8",
    )
    monkeypatch.setattr(rubric_module, "_MANIFEST_PATH", changed_manifest)

    changed = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=None,
    )

    differing = {
        key
        for key in baseline.component_digests
        if baseline.component_digests[key] != changed.component_digests[key]
    }
    assert changed != baseline
    assert differing == {f"review.{selected}.contract"}


@pytest.mark.parametrize("abi_owner", ["rubric", "adapter"])
def test_plan_review_abi_revision_invalidates_runtime_component(
    monkeypatch,
    abi_owner,
) -> None:
    spec = _spec()
    baseline = _fingerprint(spec)
    if abi_owner == "rubric":
        monkeypatch.setattr(
            rubric_module,
            "PLAN_REVIEW_RUBRIC_RUNTIME_ABI",
            rubric_module.PLAN_REVIEW_RUBRIC_RUNTIME_ABI + ".next",
        )
    else:
        monkeypatch.setattr(
            provenance_module.review_module,
            "PLAN_REVIEW_RUNTIME_ABI",
            provenance_module.review_module.PLAN_REVIEW_RUNTIME_ABI + ".next",
        )

    changed = _fingerprint(spec)

    assert changed != baseline
    assert (
        changed.component_digests["review.runtime_abi"]
        != baseline.component_digests["review.runtime_abi"]
    )


def test_fingerprint_algorithm_revision_invalidates_algorithm_component(
    monkeypatch,
) -> None:
    spec = _spec()
    baseline = _fingerprint(spec)
    monkeypatch.setattr(
        provenance_module,
        "PLAN_CONTRACT_FINGERPRINT_ALGORITHM",
        provenance_module.PLAN_CONTRACT_FINGERPRINT_ALGORITHM + ".next",
    )

    changed = _fingerprint(spec)

    assert changed != baseline
    assert (
        changed.component_digests["fingerprint.algorithm"]
        != baseline.component_digests["fingerprint.algorithm"]
    )


def test_candidate_transform_change_invalidates_planner_component() -> None:
    spec = _spec()
    critic = load("deepswe_plan_critic")
    judge = load("plan_judge")
    first = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=_transform_one,
    )
    second = plan_contract_fingerprint(
        spec,
        critic_rubric=critic,
        judge_rubric=judge,
        candidate_transform=_transform_two,
    )

    assert first != second
    assert (
        first.component_digests["planner.candidate_transform"]
        != second.component_digests["planner.candidate_transform"]
    )


def test_legacy_v2_binding_is_parsed_but_never_silently_migrated(
    tmp_path,
) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    staged = _accepted_staged(corpus, spec)
    planned_ref = staged.lifecycle_event_refs[-1]
    payload = {
        "fingerprint": "sha256:" + "1" * 64,
        "planned_event_ref": str(planned_ref),
        "schema_version": "crucible.plan-contract.v2",
        "tier_dag_digest": hash_record(staged.tier_dag),
    }
    event = LifecycleEvent(
        kind="plan_contract_bound",
        reason=(
            "accepted plan contract: "
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        ),
        affected_versions=(staged.version,),
        execution=exec_ctx_now("plan_contract"),
    )
    ref = corpus.put_record(event, kind="lifecycle_events")
    legacy = replace(
        staged,
        lifecycle_event_refs=(*staged.lifecycle_event_refs, ref),
    )

    state = accepted_plan_contract_state(
        corpus,
        legacy,
        expected_fingerprint=_fingerprint(spec),
    )

    assert state.status == "stale"
    assert state.actual_fingerprint == payload["fingerprint"]
    assert "legacy v2" in (state.reason or "")
    assert "explicit reacceptance" in (state.reason or "")
    assert legacy.lifecycle_event_refs[-1] == ref
    with pytest.raises(ValueError, match="explicit plan reacceptance"):
        bind_plan_contract(
            corpus,
            legacy,
            fingerprint=_fingerprint(spec),
        )


def test_plan_provenance_selection_has_no_product_name_branching() -> None:
    source = "\n".join((
        inspect.getsource(provenance_module.plan_contract_component_digests),
        inspect.getsource(provenance_module._selected_rubric_component_digests),
        inspect.getsource(rubric_module.plan_review_contract_semantics),
    )).lower()

    assert "deepswe" not in source
    assert "swebench" not in source
    assert "adaptix" not in source
