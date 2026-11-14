# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Independent planning joins costs and authority before publishing one DAG."""

from __future__ import annotations

import json
import hashlib
import subprocess
import threading
from types import SimpleNamespace
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    InvocationResult,
    Message,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.primitives.plan import _serialize_tier_dag

from crucible.common.cost import CostAccumulator
from crucible.rubrics import load
from crucible.workflows import architect, plan
from crucible.workflows import _plan_competition as competition
from crucible.workflows import _plan_provenance as provenance
from tests.test_plan_selection import _Backend as ComparisonBackend, _dag, _spec


COMPLETE = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)


def _repo(tmp_path, *, submodule=False):
    workspace = tmp_path / "workspace"
    root = workspace / "project" if submodule else workspace
    (root / "src").mkdir(parents=True)
    (root / "src/thing.py").write_text("value = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "src/thing.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Planner Test",
            "-c",
            "user.email=planner@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    spec = _spec()
    spec = replace(
        spec,
        project_context=replace(
            spec.project_context,
            base_commit=head,
            submodule_path="project" if submodule else None,
        ),
    )
    corpus = Corpus(workspace / ".crucible/specs/project/goal")
    corpus.put_spec(spec)
    return workspace, root, corpus, spec


def _frames(user):
    frames = {}
    for line in user.splitlines():
        try:
            value = json.loads(line.strip())
        except (ValueError, TypeError):
            continue
        if type(value) is dict and "source_id" in value and "schema_version" in value:
            frames[value["source_id"]] = value
    return frames


def _pass_review(invocation, *, critic):
    response = {
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "The complete candidate preserves the root requirement.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [],
    }
    frames = _frames(invocation.user)
    response["coverage"]["sources_reviewed"] = [
        "artifact",
        "context:spec_intent",
        "context:spec_done_criteria",
    ]

    def evidence(source_id):
        anchor = frames[source_id]["anchors"][0]
        return {
            "source_id": source_id,
            "kind": "source_quote",
            "location": anchor["location"],
            "excerpt": anchor["text"],
        }

    response["criterion_results"] = [
        {
            "criterion_id": criterion,
            "outcome": "satisfied",
            "headline": i == 0,
            "reasoning": "The cited complete candidate satisfies the root contract.",
            "evidence": [
                evidence("artifact"),
                *(
                    [evidence("context:spec_intent")]
                    if criterion == "intent_coverage"
                    else [evidence("context:spec_done_criteria")]
                    if criterion == "done_criteria_coverage"
                    else []
                ),
            ],
        }
        for i, criterion in enumerate(
            load("plan_critic" if critic else "plan_judge").required_criteria
        )
    ]
    return json.dumps(response)


class Pool:
    def __init__(
        self,
        *,
        synthesis=False,
        exhaust=False,
        defect=None,
        mutate=None,
        reference=None,
        cancel_event=None,
        cancel_at=None,
    ):
        self.synthesis, self.exhaust, self.defect, self.mutate = (
            synthesis,
            exhaust,
            defect,
            mutate,
        )
        self.backends = []
        self.barrier = threading.Barrier(3)
        self.finished = {f"plan_{i}": threading.Event() for i in (1, 2, 3)}
        self.order = []
        self.lock = threading.Lock()
        self.reference = reference
        self.cancel_event, self.cancel_at = cancel_event, cancel_at
        self.close_attempts = {}

    def get(self, role, **kwargs):
        return self.new_exclusive(
            role, log_dir=Path("unused") / role, cwd="unused", **kwargs
        )

    def new_exclusive(self, role, *, log_dir, cwd, toolset=None, **kwargs):
        backend = Backend(self, role, Path(log_dir).name, Path(cwd), toolset)
        self.backends.append(backend)
        return backend


class Backend:
    name = "competition-fixture"
    model = "competition-model"
    capability = "text"

    def __init__(self, pool, role, label, cwd, toolset):
        self.pool, self.role, self.label = pool, role, label
        self.cwd, self.toolset = cwd, toolset
        self.invocations = []
        self.closed = threading.Event()

    def invoke(self, invocation):
        self.invocations.append(invocation)
        draft = self.label in self.pool.finished
        if draft:
            assert self.cwd.name == "plan_source_snapshot"
            assert (self.cwd / "src/thing.py").read_text() == "value = 1\n"
            assert not (self.cwd / ".crucible").exists()
            assert invocation.response_schema is not None
            self.pool.barrier.wait(timeout=10)
            # Prove all three were concurrent and retain reverse completion
            # with distinct exact pricing segments.
            if self.label != "plan_3":
                self.pool.finished[f"plan_{int(self.label[-1]) + 1}"].wait(10)
            text = _serialize_tier_dag(_dag(self.label)).decode()
            if self.pool.reference:
                payload = json.loads(text)
                payload["tiers"][0]["slices"][0]["directive"][
                    "read_only_references"
                ] = [self.pool.reference]
                text = json.dumps(payload)
            if self.pool.defect == "invalid" and self.label == "plan_2":
                text = "not JSON"
            if self.pool.defect == "invalid_reference" and self.label == "plan_2":
                payload = json.loads(text)
                payload["tiers"][0]["slices"][0]["directive"][
                    "read_only_references"
                ] = [
                    {
                        "reference_id": "phantom",
                        "path": "missing_source.py",
                        "sha256": "sha256:" + "0" * 64,
                        "start_line": 1,
                        "end_line": 1,
                    }
                ]
                text = json.dumps(payload)
            if self.pool.defect == "unknown_usage" and self.label == "plan_2":
                complete = replace(COMPLETE, cost_complete=False)
            else:
                complete = COMPLETE
            cost = 1.0 if self.pool.exhaust else 0.01 * int(self.label[-1])
        elif self.label == "comparison":
            changes = (
                {
                    "mode": "synthesize",
                    "selected_id": None,
                    "selected_digest": None,
                    "synthesized_dag": json.loads(
                        _serialize_tier_dag(_dag("combined"))
                    ),
                }
                if self.pool.synthesis
                else {}
            )
            response = ComparisonBackend(changes).invoke(invocation)
            text, complete = response.final_message, COMPLETE
            cost = 1.0 if self.pool.exhaust else 0.01
            if self.pool.mutate:
                self.pool.mutate()
        else:
            text = _pass_review(invocation, critic="critic" in self.role)
            complete = COMPLETE
            cost = (
                (1.5 if self.label.startswith("synthesis") else 0.5)
                if self.pool.exhaust
                else 0.01
            )
        index = int(self.label[-1]) if draft else 5
        usage = Usage(
            input_tokens=10 + index,
            output_tokens=index,
            cost_usd=cost,
            cache_read_tokens=1,
        )
        result = InvocationResult(
            final_message=text,
            transcript=(Message(role="assistant", content=text),),
            usage=usage,
            stop_reason="end_turn",
            error=None,
            completeness=complete,
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(
                    BackendExecutionEvent(sequence=0, kind="request", name="invoke"),
                    BackendExecutionEvent(sequence=1, kind="terminal", name="complete"),
                ),
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=(
                    UsagePricingSegment(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_tokens=0,
                        cache_read_tokens=1,
                        cost_usd=cost,
                    ),
                ),
                complete=complete.cost_complete,
            ),
        )
        if self.pool.cancel_at == self.label:
            self.pool.cancel_event.set()
            assert self.closed.wait(10), "active review must be closed on cancellation"
        if self.pool.defect == "missing_pricing" and self.label == "plan_2":
            result = replace(result, pricing_evidence=None)
        with self.pool.lock:
            self.pool.order.append((self.label, result))
        if draft:
            self.pool.finished[self.label].set()
        return result

    def close(self):
        attempts = self.pool.close_attempts.get(self.label, 0) + 1
        self.pool.close_attempts[self.label] = attempts
        if (
            self.pool.defect == "double_cleanup"
            and self.label in {"plan_1_critic", "plan_1_judge"}
            and attempts == 1
        ):
            raise RuntimeError("retained provider cleanup")
        self.closed.set()
        if self.pool.defect == "cleanup" and self.label == "plan_1_critic":
            raise RuntimeError("retained provider cleanup")


def _run(tmp_path, **kwargs):
    workspace, root, corpus, spec = _repo(tmp_path)
    pool = Pool(**kwargs)
    ledger = CostAccumulator()
    outcome = competition.run_plan_competition(
        spec,
        pool=pool,
        corpus=corpus,
        cwd=str(root),
        run_dir=corpus.spec_dir / "runs/competition",
        planner_role="plan_planner",
        critic_role="plan_critic",
        judge_role="plan_judge",
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        budget_usd=10,
        enable_source_tools=True,
        wrap_backend=lambda role, backend, _tools: plan._CompletedPlanCallBackend(
            backend, role=role, costs=ledger
        ),
        call_ledger=ledger,
        cancel_event=pool.cancel_event,
    )
    report = json.loads(
        (corpus.spec_dir / "runs/competition/plan_competition.json").read_text()
    )
    return pool, ledger, corpus, spec, outcome, report


def test_three_independent_drafts_select_one_without_publishing_and_reconcile_reverse_pricing(
    tmp_path,
):
    pool, ledger, corpus, spec, outcome, report = _run(tmp_path)
    assert outcome.accepted, (outcome.status, outcome.reason, report)
    assert outcome.spec.tier_dag == _dag("plan_1")
    assert corpus.latest_spec() == spec
    assert report["planner_count"] == 3
    assert [name for name, _ in pool.order[:3]] == ["plan_3", "plan_2", "plan_1"]
    assert (
        len([backend for backend in pool.backends if backend.label in pool.finished])
        == 3
    )
    assert all(backend.closed.is_set() for backend in pool.backends)
    assert outcome.cost_usd == pytest.approx(0.13)
    assert outcome.pricing_evidence == ledger.pricing_evidence
    architect._require_plan_outcome_pricing_reconciliation(
        outcome=outcome,
        usage_before=CostAccumulator().to_dict(),
        planning_usage=ledger,
        role="competition",
    )
    assert len(report["candidates"]) == 3
    assert len(report["calls"]) == 10


def test_synthesized_plan_gets_fresh_critic_judge_and_deterministic_gates(tmp_path):
    pool, _, _, _, outcome, report = _run(tmp_path, synthesis=True)
    assert outcome.accepted, (outcome.status, outcome.reason, report)
    assert outcome.spec.tier_dag == _dag("combined")
    assert [name for name, _ in pool.order[-2:]] == [
        "synthesis_critic",
        "synthesis_judge",
    ]
    assert report["synthesis"]["accepted"] is True
    assert len(report["synthesis"]["verdict_refs"]) >= 3


@pytest.mark.parametrize(
    "defect,expected",
    [("invalid", True), ("invalid_reference", True), ("unknown_usage", False)],
)
def test_failed_draft_never_hides_spend_or_borrows_other_branch_acceptance(
    tmp_path, defect, expected
):
    pool, _, corpus, spec, outcome, report = _run(tmp_path, defect=defect)
    assert outcome.accepted is expected
    assert outcome.cost_usd >= 0.06
    assert corpus.latest_spec() == spec
    if not expected:
        assert not any(name == "comparison" for name, _ in pool.order)
        assert outcome.accounting_debt == ("cost_complete",)
    assert all(backend.closed.is_set() for backend in pool.backends)
    assert report["calls"]


def test_comparison_source_change_blocks_an_otherwise_accepted_winner(tmp_path):
    workspace, root, corpus, spec = _repo(tmp_path)
    pool = Pool(mutate=lambda: (root / "src/thing.py").write_text("changed = True\n"))
    outcome = competition.run_plan_competition(
        spec,
        pool=pool,
        corpus=corpus,
        cwd=str(root),
        run_dir=corpus.spec_dir / "runs/competition",
        planner_role="plan_planner",
        critic_role="plan_critic",
        judge_role="plan_judge",
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        budget_usd=10,
    )
    assert not outcome.accepted and outcome.status == "validate_blocked"
    assert corpus.latest_spec() == spec and outcome.spec.tier_dag is None


def test_cleanup_after_paid_review_retains_receipt_and_no_publishable_outcome(tmp_path):
    workspace, root, corpus, spec = _repo(tmp_path)
    pool = Pool(defect="cleanup")
    with pytest.raises(RuntimeError, match="cleanup"):
        competition.run_plan_competition(
            spec,
            pool=pool,
            corpus=corpus,
            cwd=str(root),
            run_dir=corpus.spec_dir / "runs/competition",
            planner_role="plan_planner",
            critic_role="plan_critic",
            judge_role="plan_judge",
            critic_rubric=load("plan_critic"),
            judge_rubric=load("plan_judge"),
            budget_usd=10,
        )
    report = json.loads(
        (corpus.spec_dir / "runs/competition/plan_competition.json").read_text()
    )
    assert report["cost_usd"] == pytest.approx(0.08)
    assert len(report["calls"]) == 5
    assert corpus.latest_spec() == spec
    assert not any(name == "comparison" for name, _ in pool.order)


def test_default_plan_cli_uses_three_planners_and_resolves_submodule_once(
    tmp_path, monkeypatch
):
    workspace, root, corpus, spec = _repo(tmp_path, submodule=True)
    pool = Pool()

    @contextmanager
    def backend_pool(*args, **kwargs):
        yield pool

    monkeypatch.setattr(plan, "backend_pool", backend_pool)
    assert (
        plan.main(
            [
                "--spec",
                str(corpus.spec_dir),
                "--cwd",
                str(workspace),
                "--budget-usd",
                "10",
                "--enable-source-tools",
            ]
        )
        == 0
    )
    assert corpus.latest_spec().tier_dag == _dag("plan_1")
    assert len([b for b in pool.backends if b.label in pool.finished]) == 3
    assert all(
        backend.closed.is_set() for backend in pool.backends if backend.invocations
    )
    (run,) = corpus.spec_dir.glob("runs/*_plan_*")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["costs"]["rollup_incomplete"] is False
    assert summary["cost_usd_is_lower_bound"] is False


def test_default_architect_accepts_fresh_synthesis_with_one_total_budget_before_tier_dispatch(
    tmp_path, monkeypatch
):
    workspace, root, corpus, spec = _repo(tmp_path, submodule=True)
    pool = Pool(synthesis=True, exhaust=True)

    @contextmanager
    def backend_pool(*args, **kwargs):
        yield pool

    monkeypatch.setattr(architect, "backend_pool", backend_pool)
    monkeypatch.setattr(
        architect, "run_tier", lambda **kw: pytest.fail("no tier budget remains")
    )
    assert (
        architect.run_architect(
            spec=corpus.spec_dir,
            cwd=str(root),
            budget_usd=10,
            enable_source_tools=True,
        )
        == 1
    )
    assert corpus.latest_spec().tier_dag == _dag("combined")
    (run,) = corpus.spec_dir.glob("runs/*_architect_*")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["authoritative_spent_usd"] == pytest.approx(10)
    assert summary["budget_remaining_usd"] == 0
    assert summary["tiers_executed"] == 0
    assert len([b for b in pool.backends if b.label in pool.finished]) == 3


@pytest.mark.parametrize(
    "component", ["plan_competition", "plan_selection", "plan_source_snapshot"]
)
def test_new_acceptance_component_changes_invalidate_reuse(
    tmp_path, monkeypatch, component
):
    _, root, _, spec = _repo(tmp_path)
    kwargs = dict(
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        candidate_transform=None,
        cwd=root,
    )
    before = provenance.plan_contract_fingerprint(spec, **kwargs)
    original = provenance._file_digest

    def changed(path, *, label):
        return (
            "sha256:" + "f" * 64
            if Path(path).name == f"_{component}.py"
            else original(path, label=label)
        )

    monkeypatch.setattr(provenance, "_file_digest", changed)
    after = provenance.plan_contract_fingerprint(spec, **kwargs)
    assert before != after
    assert (
        after.component_digests[f"planner.{component}"]
        != before.component_digests[f"planner.{component}"]
    )


@pytest.mark.parametrize(
    "private_path",
    [
        ".crucible/specs/other-project/other-goal/retained.py",
        ".CRUCIBLE/specs/other-project/other-goal/retained.py",
    ],
)
def test_private_state_citations_cannot_reenter_drafts_through_live_source_gates(
    tmp_path, private_path
):
    _, root, corpus, spec = _repo(tmp_path)
    private = root / private_path
    private.parent.mkdir(parents=True)
    private.write_text("secret_sibling_reasoning = True\n")
    pool = Pool(
        reference={
            "reference_id": "sibling",
            "path": private_path,
            "sha256": "sha256:" + hashlib.sha256(private.read_bytes()).hexdigest(),
            "start_line": 1,
            "end_line": 1,
        }
    )
    outcome = competition.run_plan_competition(
        spec,
        pool=pool,
        corpus=corpus,
        cwd=str(root),
        run_dir=corpus.spec_dir / "runs/competition",
        planner_role="plan_planner",
        critic_role="plan_critic",
        judge_role="plan_judge",
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        budget_usd=10,
    )
    assert not outcome.accepted
    assert outcome.spec.tier_dag is None and corpus.latest_spec() == spec
    assert len(pool.order) == 3  # Failed before paid Critic/Judge/comparison.


@pytest.mark.parametrize(
    "cancel_at", ["plan_1_critic", "comparison", "synthesis_critic"]
)
def test_cancellation_admits_no_later_provider_calls_and_joins_all_leases(
    tmp_path, cancel_at
):
    cancel = threading.Event()
    pool, ledger, corpus, spec, outcome, report = _run(
        tmp_path,
        synthesis=True,
        cancel_event=cancel,
        cancel_at=cancel_at,
    )
    assert not outcome.accepted and outcome.status == "cancelled"
    assert corpus.latest_spec() == spec
    labels = [label for label, _ in pool.order]
    assert labels[-1] == cancel_at
    assert "plan_1_judge" not in labels if cancel_at == "plan_1_critic" else True
    assert "synthesis_judge" not in labels if cancel_at == "synthesis_critic" else True
    assert all(backend.closed.is_set() for backend in pool.backends)
    assert not outcome.accounting_debt and not ledger.to_dict()["rollup_incomplete"]
    assert report["status"] == "cancelled" and report["cost_usd"] > 0


def test_simultaneous_review_cleanup_failures_keep_one_retry_owner_for_every_lease(
    tmp_path,
):
    from crucible.workflows._backends import BackendCleanupError

    _, root, corpus, spec = _repo(tmp_path)
    pool = Pool(defect="double_cleanup")
    with pytest.raises(BackendCleanupError) as failure:
        competition.run_plan_competition(
            spec,
            pool=pool,
            corpus=corpus,
            cwd=str(root),
            run_dir=corpus.spec_dir / "runs/competition",
            planner_role="plan_planner",
            critic_role="plan_critic",
            judge_role="plan_judge",
            critic_rubric=load("plan_critic"),
            judge_rubric=load("plan_judge"),
            budget_usd=10,
        )
    assert {label for label, _ in failure.value.failures} == {
        "plan_1_critic",
        "plan_1_judge",
    }
    failure.value.retry_cleanup()
    assert all(backend.closed.is_set() for backend in pool.backends)
    assert corpus.latest_spec() == spec
    report = json.loads(
        (corpus.spec_dir / "runs/competition/plan_competition.json").read_text()
    )
    assert report["status"] == "cleanup_failed" and report["cost_usd"] == pytest.approx(
        0.08
    )


def test_missing_pricing_stops_after_drafts_before_any_paid_review(tmp_path):
    pool, ledger, corpus, spec, outcome, report = _run(
        tmp_path, defect="missing_pricing"
    )
    assert not outcome.accepted and outcome.status == "budget_accounting_incomplete"
    assert len(pool.order) == 3 and outcome.cost_usd == pytest.approx(0.06)
    assert outcome.accounting_debt and ledger.to_dict()["rollup_incomplete"]
    assert corpus.latest_spec() == spec


@pytest.mark.parametrize("workflow", ["plan", "architect"])
def test_final_source_change_inside_publication_owner_cannot_publish_a_competition_dag(
    tmp_path, monkeypatch, workflow
):
    workspace, root, corpus, spec = _repo(tmp_path)
    module = plan if workflow == "plan" else architect
    pool = Pool()

    @contextmanager
    def backend_pool(*args, **kwargs):
        yield pool

    monkeypatch.setattr(module, "backend_pool", backend_pool)
    if workflow == "plan":
        original_validate = plan.validate

        def changing_validate(*args, **kwargs):
            verdict = original_validate(*args, **kwargs)
            if kwargs["ruleset"].name == "plan_gate":
                (root / "src/thing.py").write_text("changed_after_validation = True\n")
            return verdict

        monkeypatch.setattr(plan, "validate", changing_validate)
        assert (
            plan.main(
                [
                    "--spec",
                    str(corpus.spec_dir),
                    "--cwd",
                    str(workspace),
                    "--budget-usd",
                    "10",
                ]
            )
            == 1
        )
    else:
        original_validate = architect._assay_validate

        def changing_validate(*args, **kwargs):
            verdict = original_validate(*args, **kwargs)
            if kwargs["ruleset"].name == "architect_gate":
                (root / "src/thing.py").write_text("changed_after_validation = True\n")
            return verdict

        monkeypatch.setattr(architect, "_assay_validate", changing_validate)
        monkeypatch.setattr(
            architect,
            "run_tier",
            lambda **kw: pytest.fail("source mismatch cannot reach a tier"),
        )
        assert (
            architect.run_architect(spec=corpus.spec_dir, cwd=str(root), budget_usd=10)
            == 2
        )
    assert corpus.latest_spec() == spec
    (run,) = corpus.spec_dir.glob(f"runs/*_{workflow}_*")
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "validate_blocked"


def test_prompt_wrappers_cannot_hide_a_reused_provider_session(tmp_path):
    _, root, corpus, spec = _repo(tmp_path)

    class SharedPool(Pool):
        def __init__(self):
            super().__init__()
            self.shared = None
            self.barrier = SimpleNamespace(wait=lambda **kw: None)

        def new_exclusive(self, role, *, log_dir, cwd, toolset=None, **kwargs):
            if Path(log_dir).parent.name == "independent_planners":
                with self.lock:
                    if self.shared is None:
                        self.shared = Backend(self, role, "plan_3", Path(cwd), toolset)
                        self.backends.append(self.shared)
                    return self.shared
            return super().new_exclusive(
                role, log_dir=log_dir, cwd=cwd, toolset=toolset, **kwargs
            )

    pool = SharedPool()
    outcome = competition.run_plan_competition(
        spec,
        pool=pool,
        corpus=corpus,
        cwd=str(root),
        run_dir=corpus.spec_dir / "runs/competition",
        planner_role="plan_planner",
        critic_role="plan_critic",
        judge_role="plan_judge",
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        budget_usd=10,
    )
    assert (
        not outcome.accepted and outcome.status == "plan_competition_isolation_failed"
    )
    assert len(pool.order) <= 1
    assert pool.shared.closed.is_set() and corpus.latest_spec() == spec
