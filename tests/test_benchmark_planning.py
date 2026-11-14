# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Benchmark generation provenance cannot be inferred from an old accepted DAG."""

from __future__ import annotations

import json
import inspect
from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from pathlib import Path

import pytest

from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.primitives.plan import _serialize_tier_dag
from assay.types.artifacts import LifecycleEvent

from crucible.workflows import _benchmark_planning as planning
from crucible.workflows._run import exec_ctx_now
from tests.test_plan_selection import _dag, _spec


assay_plan = import_module("assay.primitives.plan")


def _stage(corpus, base, *, kind="planned"):
    event = LifecycleEvent(
        kind=kind,
        reason="fresh deterministic test outcome",
        affected_versions=(base.version + 1,),
        execution=exec_ctx_now("plan"),
    )
    ref = corpus.put_record(event, kind="lifecycle_events")
    return base.with_appended_refs(tier_dag=_dag(), lifecycle_event_refs=(ref,))


def _corpus(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    base = _spec()
    corpus.put_spec(base)
    return corpus, base


def _competition(corpus, base, staged, policy, *, mode="select", change=None):
    candidate_ref = corpus.put_artifact(
        _serialize_tier_dag(staged.tier_dag), kind="plan_candidates",
    )
    report = {
        "schema_version": "crucible.plan-competition.v1",
        "planner_count": policy.planner_count,
        "status": "accepted",
        "accounting_complete": True,
        "source_identity": {"spec": str(hash_record(base))},
        "implementation_sources": {
            name: policy.identity()["planning_recipe"]["modules"]["workflows/" + name]
            for name in (
                "_plan_competition.py", "_plan_selection.py", "_plan_source_snapshot.py",
            )
        },
        "candidates": [
            {
                "id": f"planner-{index}", "accepted": True, "status": "accepted",
                "candidate_ref": str(candidate_ref),
                "digest": str(hash_record(staged.tier_dag)),
            }
            for index in range(policy.planner_count)
        ],
        "selected_id": "planner-0" if mode == "select" else None,
        "comparison": {"mode": mode, "selected_id": "planner-0" if mode == "select" else None},
        "synthesis": {
            "accepted": True, "status": "accepted", "candidate_ref": str(candidate_ref),
        },
    }
    if change is not None:
        change(report)
    report_ref = corpus.put_artifact(json.dumps(report).encode(), kind="plan_competitions")
    event = LifecycleEvent(
        kind="plan_competition",
        reason=json.dumps({"report_ref": str(report_ref), "source_spec": str(hash_record(base))}),
        affected_versions=(staged.version,),
        execution=exec_ctx_now("plan_competition"),
    )
    ref = corpus.put_record(event, kind="lifecycle_events")
    return replace(staged, lifecycle_event_refs=(*staged.lifecycle_event_refs, ref))


@pytest.mark.parametrize("kwargs", [
    {"planner_count": 0}, {"planner_count": 9}, {"planner_count": True},
    {"planner_count": 3.0}, {"planning_budget_share": 0},
    {"planning_budget_share": 1.01}, {"planning_budget_share": float("nan")},
    {"planning_budget_share": float("inf")}, {"planning_budget_share": True},
    {"execution_fingerprint": "sha256:" + "A" * 64},
    {"execution_fingerprint": "1" * 64},
])
def test_invalid_policy_is_rejected_before_execution(kwargs):
    with pytest.raises(ValueError):
        planning.BenchmarkPlanningPolicy(**kwargs)


def test_identity_is_frozen_per_execution_and_returned_as_independent_copies(monkeypatch):
    policy = planning.BenchmarkPlanningPolicy(execution_fingerprint="sha256:" + "1" * 64)
    original = policy.identity()
    exposed = policy.identity()
    exposed["planning_recipe"]["modules"].clear()
    assert policy.identity() == original
    with pytest.raises(FrozenInstanceError):
        policy.planner_count = 1
    monkeypatch.setattr(planning, "_digest_file", lambda path: "sha256:" + "2" * 64)
    assert policy.identity() == original
    assert planning.BenchmarkPlanningPolicy(
        execution_fingerprint=policy.execution_fingerprint,
    ).identity() != original
    semantic = planning.benchmark_planning_identity(3)
    assert "execution_fingerprint" not in semantic
    assert semantic["planning_budget_share"] == 0.40
    assert planning.benchmark_planning_identity(1, planning_budget_share=None)["mode"] == "single_track"


def test_checkpoint_recipe_preserves_hash_callback_and_covers_planning_authority():
    root = Path(planning.__file__).resolve().parents[1]
    visited = []

    def digest(path):
        assert path.is_file()
        visited.append(path)
        return {"digest": path.name}

    recipe = planning.planning_recipe_identity(root, digest_file=digest)
    modules = recipe["modules"]
    assert modules["workflows/_benchmark_planning.py"] == {"digest": "_benchmark_planning.py"}
    assert {
        "workflows/architect.py", "workflows/plan.py", "workflows/_development_budget.py",
        "workflows/_plan_competition.py", "workflows/_plan_selection.py",
        "workflows/_plan_source_snapshot.py", "workflows/_plan_provenance.py",
        "workflows/_plan_validation.py", "workflows/_readonly_scheduler.py",
        "workflows/_atomic_persistence.py", "common/read_only_sources.py", "common/cost.py",
    } <= modules.keys()
    assert "rules/read_only_references.py" in recipe["rules"]
    assert {
        "primitives/plan.py", "primitives/review.py", "primitives/validate.py",
        "canonical/serialize.py", "canonical/registry.py", "corpus/store.py",
        "types/spec.py", "types/dag.py", "types/findings.py", "types/policy.py",
    } <= recipe["assay"]["modules"].keys()
    assert recipe["assay"]["modules"]["primitives/plan.py"] == {"digest": "plan.py"}
    assert len(set(visited)) == len(visited)


def test_actual_imported_assay_source_changes_checkpoint_recipe_and_new_policy(tmp_path, monkeypatch):
    root = Path(planning.__file__).resolve().parents[1]
    imported_plan = Path(inspect.getsourcefile(assay_plan)).resolve(strict=True)
    original_digest = planning._digest_file
    before_bytes = imported_plan.read_bytes()
    before_recipe = planning.planning_recipe_identity(root, digest_file=original_digest)
    policy = planning.BenchmarkPlanningPolicy(planner_count=1)
    frozen = policy.identity()
    corpus, base = _corpus(tmp_path)
    bound = planning.bind_benchmark_planning_policy(corpus, _stage(corpus, base), policy)

    def changed_digest(path):
        return "sha256:" + "7" * 64 if path == imported_plan else original_digest(path)

    monkeypatch.setattr(planning, "_digest_file", changed_digest)
    after_recipe = planning.planning_recipe_identity(root, digest_file=changed_digest)
    assert after_recipe["modules"] == before_recipe["modules"]
    assert after_recipe["rules"] == before_recipe["rules"]
    assert before_recipe["assay"]["modules"]["primitives/plan.py"] == original_digest(imported_plan)
    assert after_recipe["assay"]["modules"]["primitives/plan.py"] == "sha256:" + "7" * 64
    assert hash_record(after_recipe) != hash_record(before_recipe)
    assert policy.identity() == frozen
    assert not planning.benchmark_plan_matches_policy(
        corpus, bound, planning.BenchmarkPlanningPolicy(planner_count=1),
    )
    with pytest.raises(ValueError, match="source recipe changed during execution"):
        planning.bind_benchmark_planning_policy(corpus, _stage(corpus, base), policy)
    assert not planning.benchmark_plan_matches_policy(corpus, bound, policy)
    with pytest.raises(ValueError, match="source recipe changed during execution"):
        planning.bind_benchmark_planning_policy(corpus, bound, policy)
    assert corpus.latest_spec() == base
    assert imported_plan.read_bytes() == before_bytes


@pytest.mark.parametrize("missing", ["package", "planner"])
def test_imported_assay_recipe_fails_closed_without_actual_source(monkeypatch, missing):
    original_source = planning.inspect.getsourcefile

    def missing_source(module):
        if module is (planning.assay if missing == "package" else assay_plan):
            return None
        return original_source(module)

    monkeypatch.setattr(planning.inspect, "getsourcefile", missing_source)
    with pytest.raises(ValueError, match="Assay.*source.*unavailable"):
        planning.BenchmarkPlanningPolicy()


def test_old_published_single_track_plan_cannot_acquire_new_policy(tmp_path):
    corpus, base = _corpus(tmp_path)
    accepted = _stage(corpus, base)
    corpus.put_spec(accepted)
    for count in (1, 3):
        policy = planning.BenchmarkPlanningPolicy(planner_count=count)
        assert not planning.benchmark_plan_matches_policy(corpus, accepted, policy)
        with pytest.raises(ValueError, match="fresh staged"):
            planning.bind_benchmark_planning_policy(corpus, accepted, policy)
    assert corpus.latest_spec() == accepted


@pytest.mark.parametrize("count,mode", [(1, None), (3, "select"), (3, "synthesize")])
def test_fresh_acceptance_stages_one_exact_idempotent_stamp_without_publication(tmp_path, count, mode):
    corpus, base = _corpus(tmp_path)
    policy = planning.BenchmarkPlanningPolicy(planner_count=count)
    staged = _stage(corpus, base)
    if mode is not None:
        staged = _competition(corpus, base, staged, policy, mode=mode)
    bound = planning.bind_benchmark_planning_policy(corpus, staged, policy)
    assert corpus.latest_spec() == base
    assert bound.version == staged.version
    assert len(bound.lifecycle_event_refs) == len(staged.lifecycle_event_refs) + 1
    assert planning.bind_benchmark_planning_policy(corpus, bound, policy) is bound
    assert planning.benchmark_plan_matches_policy(corpus, bound, policy)
    event = corpus.get_record(bound.lifecycle_event_refs[-1], as_type=LifecycleEvent)
    payload = json.loads(event.reason)
    assert payload["planned_event_ref"] == str(staged.lifecycle_event_refs[0])
    assert event.affected_versions == (bound.version,)
    corpus.put_spec(bound)
    later = bound.with_appended_refs()
    assert planning.benchmark_plan_matches_policy(corpus, later, policy)


def test_any_generation_identity_change_requires_fresh_planning(tmp_path, monkeypatch):
    corpus, base = _corpus(tmp_path)
    policy = planning.BenchmarkPlanningPolicy(planner_count=1)
    bound = planning.bind_benchmark_planning_policy(corpus, _stage(corpus, base), policy)
    alternatives = [
        planning.BenchmarkPlanningPolicy(planner_count=3),
        planning.BenchmarkPlanningPolicy(planner_count=1, planning_budget_share=None),
        planning.BenchmarkPlanningPolicy(planner_count=1, planning_budget_share=0.30),
        planning.BenchmarkPlanningPolicy(planner_count=1, execution_fingerprint="sha256:" + "3" * 64),
    ]
    monkeypatch.setattr(planning, "_digest_file", lambda path: "sha256:" + "4" * 64)
    alternatives.append(planning.BenchmarkPlanningPolicy(planner_count=1))
    for changed in alternatives:
        assert not planning.benchmark_plan_matches_policy(corpus, bound, changed)
        with pytest.raises(ValueError, match="fresh accepted|source recipe changed"):
            planning.bind_benchmark_planning_policy(corpus, bound, changed)
    assert not planning.benchmark_plan_matches_policy(corpus, bound, policy)


def test_stamps_bind_terminal_dag_version_and_current_success(tmp_path):
    corpus, base = _corpus(tmp_path)
    policy = planning.BenchmarkPlanningPolicy(planner_count=1)
    staged = _stage(corpus, base)
    with pytest.raises(ValueError, match="fresh staged"):
        planning.bind_benchmark_planning_policy(corpus, replace(
            staged, parent="sha256:" + "e" * 64,
        ), policy)
    bound = planning.bind_benchmark_planning_policy(corpus, staged, policy)
    assert not planning.benchmark_plan_matches_policy(corpus, replace(bound, tier_dag=_dag("other")), policy)
    stamp = corpus.get_record(bound.lifecycle_event_refs[-1], as_type=LifecycleEvent)
    bad_ref = corpus.put_record(replace(stamp, affected_versions=(99,)), kind="lifecycle_events")
    assert not planning.benchmark_plan_matches_policy(corpus, replace(
        bound, lifecycle_event_refs=(*bound.lifecycle_event_refs[:-1], bad_ref),
    ), policy)
    for kind in ("planned", "plan_rejected", "plan_inconclusive"):
        later = _stage(corpus, bound, kind=kind)
        # Even moving an old stamp after a new planned event cannot rebind it.
        later = replace(later, lifecycle_event_refs=(*later.lifecycle_event_refs, bound.lifecycle_event_refs[-1]))
        assert not planning.benchmark_plan_matches_policy(corpus, later, policy)
    missing = replace(bound, lifecycle_event_refs=(*bound.lifecycle_event_refs, "sha256:" + "f" * 64))
    assert not planning.benchmark_plan_matches_policy(corpus, missing, policy)


@pytest.mark.parametrize("change", [
    lambda r: r.update(schema_version="crucible.plan-competition.v0"),
    lambda r: r.update(planner_count=1),
    lambda r: r.update(status="plan_competition_inconclusive"),
    lambda r: r.update(accounting_complete=False),
    lambda r: r.update(candidates=r["candidates"][:1]),
    lambda r: r["candidates"][1].update(id="planner-0"),
    lambda r: r.update(selected_id="missing"),
    lambda r: r["candidates"][0].update(digest="sha256:" + "a" * 64),
    lambda r: r["candidates"][0].update(accepted=False),
    lambda r: r["implementation_sources"].update({"_plan_competition.py": "sha256:" + "b" * 64}),
    lambda r: r["source_identity"].update(spec="sha256:" + "c" * 64),
])
def test_competition_binding_rejects_wrong_or_incomplete_generation_proof(tmp_path, change):
    corpus, base = _corpus(tmp_path)
    policy = planning.BenchmarkPlanningPolicy()
    staged = _competition(corpus, base, _stage(corpus, base), policy, change=change)
    with pytest.raises(ValueError):
        planning.bind_benchmark_planning_policy(corpus, staged, policy)
    assert corpus.latest_spec() == base


def test_competition_requires_current_report_and_exact_final_candidate(tmp_path):
    corpus, base = _corpus(tmp_path)
    policy = planning.BenchmarkPlanningPolicy()
    staged = _stage(corpus, base)
    with pytest.raises(ValueError, match="current competition evidence"):
        planning.bind_benchmark_planning_policy(corpus, staged, policy)
    wrong_ref = corpus.put_artifact(_serialize_tier_dag(_dag("other")), kind="plan_candidates")
    for mode in ("select", "synthesize"):
        altered = _competition(
            corpus, base, staged, policy, mode=mode,
            change=lambda report, mode=mode: (
                report["candidates"][0] if mode == "select" else report["synthesis"]
            ).update(candidate_ref=str(wrong_ref)),
        )
        with pytest.raises(ValueError, match="final candidate differs"):
            planning.bind_benchmark_planning_policy(corpus, altered, policy)
    valid = _competition(corpus, base, staged, policy)
    with pytest.raises(ValueError, match="current competition evidence"):
        planning.bind_benchmark_planning_policy(corpus, replace(
            valid, lifecycle_event_refs=tuple(reversed(valid.lifecycle_event_refs)),
        ), policy)
    event = corpus.get_record(valid.lifecycle_event_refs[-1], as_type=LifecycleEvent)
    old_ref = corpus.put_record(replace(event, affected_versions=(99,)), kind="lifecycle_events")
    with pytest.raises(ValueError, match="current competition evidence"):
        planning.bind_benchmark_planning_policy(corpus, replace(
            valid, lifecycle_event_refs=(*staged.lifecycle_event_refs, old_ref),
        ), policy)
    with pytest.raises(ValueError, match="single-track"):
        planning.bind_benchmark_planning_policy(
            corpus, valid, planning.BenchmarkPlanningPolicy(planner_count=1),
        )
