# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Benchmark competition spends planning authority, preserving task funds."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from assay import PlanOutcome
from assay.corpus.runs import RunResult
from assay.corpus.store import ConcurrentSpecError

from crucible.common.cost import CostAccumulator
from crucible.workflows import architect, plan
from crucible.workflows._benchmark_planning import (
    BenchmarkPlanningPolicy,
    benchmark_plan_matches_policy,
)
from crucible.workflows._development_budget import (
    DevelopmentBudgetPolicy,
    RoleAllocation,
)
from crucible.workflows._run import open_run_with_capture
from tests.test_plan_competition import Backend, Pool, _repo
from tests.test_benchmark_planning import _stage


def _development_policy(planning_cap):
    task = 10.0
    return DevelopmentBudgetPolicy(
        task_budget_usd=task,
        planning_review_cap_usd=planning_cap,
        implementation_verifier_reserve_usd=task - planning_cap - 1.0,
        contingency_usd=1.0,
        release_unused_planning_authority=True,
        roles=tuple(
            RoleAllocation(
                role=role, group="planning_review", max_calls=calls,
                max_output_tokens=1_000_000, max_wall_seconds=300,
                max_cost_usd=planning_cap,
            )
            for role, calls in (
                ("architect_planner", 3), ("architect_critic", 12),
                ("architect_judge", 15),
            )
        ) + (RoleAllocation(
            role="implementer", group="implementation_verifier", max_calls=12,
            max_output_tokens=1_000_000, max_wall_seconds=300, max_cost_usd=task,
        ),),
    )


class _LedgerPool(Pool):
    def __init__(self, ledger, **kwargs):
        super().__init__(**kwargs)
        self.ledger = ledger

    def new_exclusive(self, role, **kwargs):
        backend = super().new_exclusive(role, **kwargs)
        return self.ledger.wrap(role, backend) if self.ledger is not None else backend


def _install_pool(monkeypatch, *, defect=None):
    pools = []

    @contextmanager
    def backend_pool(*args, development_budget_ledger=None, **kwargs):
        pool = _LedgerPool(development_budget_ledger, synthesis=True, defect=defect)
        pools.append(pool)
        yield pool

    monkeypatch.setattr(architect, "backend_pool", backend_pool)
    return pools


def _tier_boundary(monkeypatch, corpus, root):
    calls = []

    def tier(**kwargs):
        calls.append(kwargs)
        # Stop at the first real implementation boundary; no code is built.
        with open_run_with_capture(corpus, "tier", provenance_root=root) as handle:
            handle.write_summary({"status": "abandoned", "costs": CostAccumulator().to_dict()})
            handle.finalize_elapsed(RunResult(status="abandoned", cost_usd=0.0))
        return 1

    monkeypatch.setattr(architect, "run_tier", tier)
    return calls


@pytest.mark.parametrize("ledger_cap", [None, 4.0, 2.0])
def test_competition_partitions_only_planning_funds_and_keeps_synthesis_reachable(
    tmp_path, monkeypatch, ledger_cap,
):
    _, root, corpus, _ = _repo(tmp_path)
    pools = _install_pool(monkeypatch)
    tier_calls = _tier_boundary(monkeypatch, corpus, root)
    policy = BenchmarkPlanningPolicy(execution_fingerprint="sha256:" + "a" * 64)
    reservations = []
    raw_invoke = Backend.invoke

    def observed_invoke(backend, invocation):
        if backend.pool.ledger is not None:
            reservations.append(backend.pool.ledger.to_summary()["group_reserved_usd"]["planning_review"])
        return raw_invoke(backend, invocation)

    monkeypatch.setattr(Backend, "invoke", observed_invoke)
    assert architect.run_architect(
        spec=corpus.spec_dir, cwd=str(root), budget_usd=10,
        planning_policy=policy, enable_source_tools=True,
        development_budget_policy=(
            _development_policy(ledger_cap) if ledger_cap is not None else None
        ),
    ) == 1
    pool, = pools
    run, = corpus.spec_dir.glob("runs/*_architect_*")
    report = json.loads((run / "plan_competition.json").read_text())
    cap = min(4.0, ledger_cap) if ledger_cap is not None else 4.0
    assert report["budget_usd"] == cap
    drafts = [backend for backend in pool.backends if backend.label in pool.finished]
    assert len(drafts) == 3
    assert [draft.invocations[0].budget_usd for draft in drafts] == pytest.approx([cap / 10] * 3)
    assert [name for name, _ in pool.order[-2:]] == ["synthesis_critic", "synthesis_judge"]
    assert report["cost_usd"] == pytest.approx(0.15)
    assert len(tier_calls) == 1
    assert tier_calls[0]["budget_usd"] == pytest.approx(9.85)
    assert benchmark_plan_matches_policy(corpus, corpus.latest_spec(), policy)
    if pool.ledger is not None:
        assert max(reservations) <= cap
        assert pool.ledger.group_dispatch_capacity("planning_review") == 0
        assert pool.ledger.group_dispatch_capacity("implementation_verifier") == pytest.approx(9.85)


def test_unknown_draft_spend_cannot_open_review_or_implementation_reserve(tmp_path, monkeypatch):
    _, root, corpus, _ = _repo(tmp_path)
    pools = _install_pool(monkeypatch, defect="missing_pricing")
    tier_calls = _tier_boundary(monkeypatch, corpus, root)
    assert architect.run_architect(
        spec=corpus.spec_dir, cwd=str(root), budget_usd=10,
        planning_policy=BenchmarkPlanningPolicy(execution_fingerprint="sha256:" + "b" * 64),
        development_budget_policy=_development_policy(4.0),
        enable_source_tools=True,
    ) == 1
    pool, = pools
    assert not tier_calls
    assert corpus.latest_spec().tier_dag is None
    assert all(backend.label in pool.finished for backend in pool.backends if backend.invocations)
    assert pool.ledger.to_summary()["group_spend_usd"]["implementation_verifier"] == 0
    assert all(backend.closed.is_set() for backend in pool.backends if backend.invocations)


def test_benchmark_replans_unbound_old_dag_then_reuses_its_exact_new_policy(tmp_path, monkeypatch):
    _, root, corpus, _ = _repo(tmp_path)
    original_pool = Pool()

    @contextmanager
    def previous_plan(*args, **kwargs):
        yield original_pool

    monkeypatch.setattr(plan, "backend_pool", previous_plan)
    assert plan.main([
        "--spec", str(corpus.spec_dir), "--cwd", str(root),
        "--budget-usd", "10", "--enable-source-tools",
    ]) == 0
    original = corpus.latest_spec()
    policy = BenchmarkPlanningPolicy(execution_fingerprint="sha256:" + "c" * 64)
    assert not benchmark_plan_matches_policy(corpus, original, policy)
    pools = _install_pool(monkeypatch)
    _tier_boundary(monkeypatch, corpus, root)
    kwargs = dict(
        spec=corpus.spec_dir, cwd=str(root), budget_usd=10,
        planning_policy=policy, enable_source_tools=True,
    )
    assert architect.run_architect(**kwargs) == 1
    assert len(pools[0].order) == 12
    accepted = corpus.latest_spec()
    assert benchmark_plan_matches_policy(corpus, accepted, policy)
    assert architect.run_architect(**kwargs) == 1
    assert len(pools) == 2
    assert pools[1].order == []


def test_competition_cannot_fall_back_to_single_track_validation_repair(tmp_path):
    with pytest.raises(ValueError, match="competition repair requires replan=True"):
        architect.run_architect(
            spec=tmp_path, cwd=str(tmp_path), budget_usd=10,
            planning_policy=BenchmarkPlanningPolicy(), replan_on_validate_fail=True,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("planner_count", [1, 3])
def test_benchmark_reacceptance_stops_before_any_run_or_provider(tmp_path, planner_count):
    with pytest.raises(ValueError, match="digest reacceptance is unsupported"):
        architect.run_architect(
            spec=tmp_path, cwd=str(tmp_path), budget_usd=10, planner_count=planner_count,
            planning_policy=BenchmarkPlanningPolicy(planner_count=planner_count),
            reaccept_plan_digest="sha256:" + "d" * 64,
        )
    assert not list(tmp_path.iterdir())


def test_single_track_benchmark_stamp_cannot_rebase_after_concurrent_advance(tmp_path, monkeypatch):
    _, root, corpus, _ = _repo(tmp_path)
    _install_pool(monkeypatch)
    tier_calls = _tier_boundary(monkeypatch, corpus, root)

    def single_plan(spec, **kwargs):
        staged = _stage(corpus, spec)
        return PlanOutcome(
            spec=staged, candidate_dag=staged.tier_dag, candidate_ref=None,
            decision="accepted", status="accepted", rounds_used=1,
            final_critic_decision="pass", final_judge_decision="pass",
            reason="zero-cost deterministic fixture", cost_usd=0.0,
        )

    monkeypatch.setattr(architect, "run_plan", single_plan)
    publish = architect.commit_staged_spec

    def concurrent_advance(*args, **kwargs):
        # The competing commit advances the manifest without replacing its DAG.
        corpus.commit(transform=lambda candidate: replace(
            candidate, intent=candidate.intent + " Concurrent operator clarification.",
        ))
        assert kwargs["require_exact_base"] is True
        return publish(*args, **kwargs)

    monkeypatch.setattr(architect, "commit_staged_spec", concurrent_advance)
    with pytest.raises(ConcurrentSpecError):
        architect.run_architect(
            spec=corpus.spec_dir, cwd=str(root), budget_usd=10, planner_count=1,
            planning_policy=BenchmarkPlanningPolicy(planner_count=1, planning_budget_share=None),
        )
    assert not tier_calls
    assert corpus.latest_spec().tier_dag is None
