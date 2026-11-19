# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Cost-accounting regression tests for crucible.workflows.architect.

The pre-fix architect.py constructed a ``CostAccumulator`` but never
called ``record()`` on it: planner/critic/judge cost ran through
``run_plan`` (no ``costs=`` kwarg), per-tier implementer/breach/review
cost lived in tier's own accumulator, and the breach subprocess's
cost lived in its own accumulator. The architect ``summary.json``
always reported ``total_usd: 0`` regardless of real spend.

Fix: carry typed PlanOutcome cost through one workflow-wide ledger and use
call-side planning usage plus child Tier summaries as token authority. The
corpus-backed ``verdict_refs`` / ``implementation_refs`` walk supplies role
attribution and discrepancy diagnostics without double-counting final records
or losing discarded retries.
"""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import (
    ExecutionCompleteness,
    ExecutionContext,
    Implementation,
    ImplementationDirective,
    Invocation,
    LifecycleEvent,
    PlanOutcome,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.corpus.runs import RunResult
from assay.primitives.implement import capture_workspace_delta
from assay.types.findings import ReviewCoverage, Verdict
from assay.types.invocation import Usage
from crucible.common.cost import CostAccumulator
from crucible.tools._dispatch_budget import (
    BoundedReadOnlyToolSet,
    PLANNER_TOOL_CONTROL_ALLOWANCE_BYTES,
    ToolDispatchBudgetExceeded,
)
from crucible.workflows._run import FailureTerminal, open_run_with_capture
from crucible.workflows._run_policy import OrchestrationPolicy
from crucible.workflows._development_budget import (
    DevelopmentBudgetLedger,
    DevelopmentBudgetPolicy,
    RoleAllocation,
)
from crucible.workflows.architect import (
    _PlanningAccountingIncomplete,
    _PlanningUsageBackend,
    _authoritative_cost_summary,
    _budget_enforcement_summary,
    _candidate_validation_repair_count,
    _first_persisted_mutation_metrics,
    _record_spec_delta_into_costs,
    _remaining_budget,
    _require_plan_outcome_pricing_reconciliation,
    _retain_incomplete_planner_candidate,
    _terminal_status,
)
from crucible.workflows import architect as arch
from crucible.workflows import _mutation_journal, _partition
from crucible.workflows._run import exec_ctx_now


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)


def _exact_pricing_evidence(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    cost_usd: float,
) -> UsagePricingEvidence:
    return UsagePricingEvidence(
        segments=(UsagePricingSegment(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost_usd,
        ),),
        complete=True,
    )


@pytest.fixture(autouse=True)
def _isolate_cost_tests_from_plan_provenance(monkeypatch):
    """These unit tests use non-Git protocol doubles, not real workspaces."""

    monkeypatch.setattr(
        arch,
        "plan_contract_fingerprint",
        lambda *args, **kwargs: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(
        arch,
        "bind_plan_contract",
        lambda corpus, staged_spec, *, fingerprint: staged_spec,
    )


def _make_corpus(
    tmp_path: Path,
    *,
    runner_parameters: dict[str, object] | None = None,
) -> Corpus:
    spec_dir = tmp_path / "corpus"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="x", base_commit="a" * 40,
            branch=None, submodule_path=None, runners={},
            runner_parameters=runner_parameters or {},
        ),
        execution=None, intent="", constraints=(),
        done_criteria=(), plan_policy=None,
    )
    corpus.put_spec(spec)
    return corpus


def _make_impl(
    corpus: Corpus,
    *,
    slice_id: str,
    cost: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    files_modified: tuple[str, ...] = (),
    timestamp: str = "2026-05-23T00:00:00+00:00",
) -> Implementation:
    return Implementation(
        slice_id=slice_id, base_commit="a" * 40, result_commit=None,
        diff_ref=corpus.put_artifact(b"", kind="diffs", name_hint=slice_id),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint=slice_id,
        ),
        files_modified=files_modified,
        scope_escapes=(),
        classification="success",
        execution=ExecutionContext(
            timestamp=timestamp,
            machine="t", cwd="/", toolchain={},
            backend="codex", model="gpt-5.5",
            recipe="architect", recipe_commit=None,
            gpu_model=None, run_id=None,
        ),
        cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _make_final_workspace_binding(corpus: Corpus) -> Implementation:
    binding = _make_impl(
        corpus,
        slice_id="final-binding",
        cost=0.0,
        input_tokens=0,
        output_tokens=0,
    )
    return replace(
        binding,
        execution=replace(
            binding.execution,
            backend=None,
            model=None,
            recipe="tier_final_workspace_binding",
        ),
        artifacts={
            "tier_final_workspace_binding": binding.transcript_ref,
        },
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=UsagePricingEvidence(segments=(), complete=True),
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )


def _make_verdict(corpus: Corpus, *, rubric: str, kind, cost: float) -> Verdict:
    return Verdict(
        rubric=rubric, decision="pass", findings=(),
        artifact_ref=corpus.put_artifact(
            b"x", kind="reviewed_artifacts", name_hint=rubric,
        ),
        artifact_commit=None,
        reasoning="", coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=1,
            artifact_chars_reviewed=1,
        ), backend_name="codex",
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint=f"{rubric}.json",
        ),
        cost_usd=cost, execution=None, kind=kind,
    )


def test_candidate_validation_metric_counts_the_single_repair_authority(
    tmp_path: Path,
):
    corpus = _make_corpus(tmp_path)
    baseline = corpus.latest_spec()
    verdict = _make_verdict(
        corpus,
        rubric="candidate_validation",
        kind="validate",
        cost=0.0,
    )
    verdict_ref = corpus.put_record(verdict, kind="verdicts")
    current = baseline.with_appended_refs(verdict_refs=(verdict_ref,))

    assert _candidate_validation_repair_count(
        corpus=corpus,
        spec=current,
        baseline_verdict_count=len(baseline.verdict_refs),
    ) == 1


def test_first_mutation_metric_uses_first_changed_implementation(
    tmp_path: Path,
):
    corpus = _make_corpus(tmp_path)
    baseline = corpus.latest_spec()
    no_change = _make_impl(
        corpus,
        slice_id="noop",
        cost=0.2,
        timestamp="2026-05-23T00:00:02+00:00",
    )
    mutation = _make_impl(
        corpus,
        slice_id="mutate",
        cost=0.3,
        files_modified=("src/thing.py",),
        timestamp="2026-05-23T00:00:05+00:00",
    )
    refs = tuple(
        corpus.put_record(record, kind="implementations")
        for record in (no_change, mutation)
    )
    current = baseline.with_appended_refs(implementation_refs=refs)

    assert _first_persisted_mutation_metrics(
        corpus=corpus,
        prior_spec=baseline,
        current_spec=current,
        architect_started_at="2026-05-23T00:00:00+00:00",
        spent_before_tier_usd=2.0,
    ) == pytest.approx((5.0, 2.5))


def test_delta_walk_records_only_new_refs(tmp_path: Path):
    """Architect must not double-count records that existed before
    its run. Pre-existing implementation/verdict refs (from a prior
    ``crucible plan`` invocation, say) should be excluded by the
    delta walk."""
    corpus = _make_corpus(tmp_path)
    spec = corpus.latest_spec()

    # Pre-existing records — should NOT be in the delta.
    prior_impl = _make_impl(corpus, slice_id="prior", cost=99.0)
    prior_impl_ref = corpus.put_record(prior_impl, kind="implementations")
    prior_verdict = _make_verdict(corpus, rubric="prior_critic", kind="critic", cost=99.0)
    prior_verdict_ref = corpus.put_record(prior_verdict, kind="verdicts")
    spec = spec.with_appended_refs(
        implementation_refs=(prior_impl_ref,),
        verdict_refs=(prior_verdict_ref,),
    )
    corpus.put_spec(spec)

    prior_spec = corpus.latest_spec()
    pre_verdict_ref_count = len(prior_spec.verdict_refs)
    pre_implementation_ref_count = len(prior_spec.implementation_refs)

    # Simulate architect adding new records this run.
    new_impl = _make_impl(corpus, slice_id="new", cost=0.40)
    new_impl_ref = corpus.put_record(new_impl, kind="implementations")
    new_verdict = _make_verdict(
        corpus, rubric="new_critic", kind="critic", cost=0.30,
    )
    new_verdict_ref = corpus.put_record(new_verdict, kind="verdicts")
    new_spec = prior_spec.with_appended_refs(
        implementation_refs=(new_impl_ref,),
        verdict_refs=(new_verdict_ref,),
    )
    corpus.put_spec(new_spec)

    costs = CostAccumulator()
    _record_spec_delta_into_costs(
        costs,
        corpus=corpus,
        prior_spec=prior_spec,
        current_spec=corpus.latest_spec(),
        pre_verdict_ref_count=pre_verdict_ref_count,
        pre_implementation_ref_count=pre_implementation_ref_count,
    )

    # Delta is implementer 0.40 + critic 0.30 = 0.70. The $99 prior
    # records must be ignored.
    assert costs.total_cost_usd == pytest.approx(0.70)


def test_delta_walk_accepts_exact_zero_deterministic_implementation(
    tmp_path: Path,
):
    """A no-transport record must not invalidate retained pricing evidence."""

    corpus = _make_corpus(tmp_path)
    baseline = corpus.latest_spec()
    priced_evidence = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.40,
    )
    priced = replace(
        _make_impl(
            corpus,
            slice_id="priced",
            cost=0.40,
            input_tokens=100,
            output_tokens=20,
        ),
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=priced_evidence,
        cache_creation_tokens=10,
        cache_read_tokens=40,
    )
    deterministic_base = _make_impl(
        corpus,
        slice_id="defender-not-required",
        cost=0.0,
    )
    deterministic = replace(
        deterministic_base,
        classification="success_pre_fix_attacks_passed",
        execution=replace(
            deterministic_base.execution,
            backend="breach_defender_not_required",
            model=None,
        ),
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=UsagePricingEvidence(segments=(), complete=True),
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    refs = tuple(
        corpus.put_record(record, kind="implementations")
        for record in (priced, deterministic)
    )
    current = baseline.with_appended_refs(implementation_refs=refs)
    corpus.put_spec(current)

    costs = CostAccumulator()
    _record_spec_delta_into_costs(
        costs,
        corpus=corpus,
        prior_spec=baseline,
        current_spec=current,
        pre_verdict_ref_count=len(baseline.verdict_refs),
        pre_implementation_ref_count=len(baseline.implementation_refs),
    )

    summary = costs.to_dict()
    assert summary["total_usd"] == pytest.approx(0.40)
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 20
    assert summary["by_role"]["implementer"]["calls"] == 2
    assert len(summary["pricing_evidence_by_call"]) == 2
    assert summary["pricing_evidence_by_call"][1]["pricing_evidence"] == {
        "segments": [],
        "complete": True,
    }
    assert summary["pricing_evidence"]["complete"] is True
    assert arch._pricing_summary_projection(
        summary,
        label="priced implementation plus deterministic no-transport record",
    ).valid


def test_delta_walk_excludes_final_binding_from_v2_provider_call_ledger(
    tmp_path: Path,
):
    corpus = _make_corpus(tmp_path)
    baseline = corpus.latest_spec()
    priced_evidence = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.40,
    )
    priced = replace(
        _make_impl(
            corpus,
            slice_id="priced",
            cost=0.40,
            input_tokens=100,
            output_tokens=20,
        ),
        execution_completeness=_COMPLETE_EXECUTION,
        pricing_evidence=priced_evidence,
        cache_creation_tokens=10,
        cache_read_tokens=40,
    )
    binding = _make_final_workspace_binding(corpus)
    refs = tuple(
        corpus.put_record(record, kind="implementations")
        for record in (priced, binding)
    )
    current = baseline.with_appended_refs(implementation_refs=refs)
    corpus.put_spec(current)

    costs = CostAccumulator()
    _record_spec_delta_into_costs(
        costs,
        corpus=corpus,
        prior_spec=baseline,
        current_spec=current,
        pre_verdict_ref_count=len(baseline.verdict_refs),
        pre_implementation_ref_count=len(baseline.implementation_refs),
    )

    summary = costs.to_dict()
    assert summary["total_usd"] == pytest.approx(0.40)
    assert summary["by_role"]["implementer"]["calls"] == 1
    assert len(summary["pricing_evidence_by_call"]) == 1
    assert summary["pricing_evidence"]["complete"] is True
    assert "unknown" not in summary["by_backend"]
    assert arch._pricing_summary_projection(
        summary,
        label="priced implementation excluding its final binding",
    ).valid


def test_delta_walk_excludes_final_binding_without_activating_v2(
    tmp_path: Path,
):
    corpus = _make_corpus(tmp_path)
    baseline = corpus.latest_spec()
    legacy = _make_impl(
        corpus,
        slice_id="legacy-paid",
        cost=0.40,
        input_tokens=100,
        output_tokens=20,
    )
    binding = _make_final_workspace_binding(corpus)
    refs = tuple(
        corpus.put_record(record, kind="implementations")
        for record in (legacy, binding)
    )
    current = baseline.with_appended_refs(implementation_refs=refs)
    corpus.put_spec(current)

    costs = CostAccumulator()
    _record_spec_delta_into_costs(
        costs,
        corpus=corpus,
        prior_spec=baseline,
        current_spec=current,
        pre_verdict_ref_count=len(baseline.verdict_refs),
        pre_implementation_ref_count=len(baseline.implementation_refs),
    )

    summary = costs.to_dict()
    assert summary["total_usd"] == pytest.approx(0.40)
    assert summary["by_role"]["implementer"]["calls"] == 1
    assert summary["rollup_incomplete"] is False
    assert "pricing_evidence_schema" not in summary
    assert "pricing_evidence" not in summary
    assert "pricing_evidence_by_call" not in summary
    assert arch._pricing_summary_projection(
        summary,
        label="legacy implementation excluding its final binding",
    ).active is False


@pytest.mark.parametrize(
    "tamper",
    [
        "nonzero_cost",
        "nonzero_tokens",
        "incomplete_evidence",
        "artifact_without_recipe",
        "recipe_without_artifact",
        "terminated",
        "non_success",
        "scope_escape",
    ],
)
def test_delta_walk_rejects_malformed_final_binding_claim(
    tmp_path: Path,
    tamper: str,
):
    corpus = _make_corpus(tmp_path)
    baseline = corpus.latest_spec()
    binding = _make_final_workspace_binding(corpus)
    if tamper == "nonzero_cost":
        binding = replace(
            binding,
            cost_usd=0.01,
            pricing_evidence=_exact_pricing_evidence(
                input_tokens=0,
                output_tokens=0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_usd=0.01,
            ),
        )
    elif tamper == "nonzero_tokens":
        binding = replace(
            binding,
            input_tokens=1,
            pricing_evidence=_exact_pricing_evidence(
                input_tokens=1,
                output_tokens=0,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_usd=0.0,
            ),
        )
    elif tamper == "incomplete_evidence":
        binding = replace(
            binding,
            pricing_evidence=UsagePricingEvidence(
                segments=(),
                complete=False,
            ),
        )
    elif tamper == "artifact_without_recipe":
        binding = replace(
            binding,
            execution=replace(binding.execution, recipe="implement"),
        )
    elif tamper == "recipe_without_artifact":
        binding = replace(binding, artifacts={})
    elif tamper == "terminated":
        binding = replace(binding, stop_reason="max_turns")
    elif tamper == "non_success":
        binding = replace(binding, classification="hard_fail_incomplete")
    else:
        binding = replace(binding, scope_escapes=("outside.py",))
    binding_ref = corpus.put_record(binding, kind="implementations")
    current = baseline.with_appended_refs(
        implementation_refs=(binding_ref,),
    )
    corpus.put_spec(current)

    with pytest.raises(
        ValueError,
        match="invalid tier_final_workspace_binding accounting envelope",
    ):
        _record_spec_delta_into_costs(
            CostAccumulator(),
            corpus=corpus,
            prior_spec=baseline,
            current_spec=current,
            pre_verdict_ref_count=len(baseline.verdict_refs),
            pre_implementation_ref_count=len(baseline.implementation_refs),
        )


def test_delta_walk_attributes_per_role(tmp_path: Path):
    """The ``role`` field on each record() call should differentiate
    implementer vs critic vs review so the ``by_role`` rollup is
    informative — operators inspecting summary.json want to see
    where the dollars went."""
    corpus = _make_corpus(tmp_path)
    baseline_spec = corpus.latest_spec()
    baseline_v = len(baseline_spec.verdict_refs)
    baseline_i = len(baseline_spec.implementation_refs)

    impl = _make_impl(corpus, slice_id="s0", cost=0.50)
    impl_ref = corpus.put_record(impl, kind="implementations")
    verdict_refs = []
    for rubric, kind, cost in (
        ("plan_critic", "critic", 0.20),
        ("plan_judge", "judge", 0.30),
        ("diff_scope", "review", 0.10),
    ):
        v = _make_verdict(corpus, rubric=rubric, kind=kind, cost=cost)
        verdict_refs.append(corpus.put_record(v, kind="verdicts"))
    spec = baseline_spec.with_appended_refs(
        implementation_refs=(impl_ref,), verdict_refs=verdict_refs,
    )
    corpus.put_spec(spec)

    costs = CostAccumulator()
    _record_spec_delta_into_costs(
        costs,
        corpus=corpus,
        prior_spec=baseline_spec,
        current_spec=corpus.latest_spec(),
        pre_verdict_ref_count=baseline_v,
        pre_implementation_ref_count=baseline_i,
    )

    rollup = costs.to_dict()
    assert rollup["total_usd"] == pytest.approx(1.10)
    assert "implementer" in rollup["by_role"]
    # Rubric name as role — already encodes the phase by convention
    # (``plan_critic`` / ``plan_judge`` / ``diff_scope`` / etc.),
    # avoids the ``critic_plan_critic`` stutter.
    assert "plan_critic" in rollup["by_role"]
    assert "plan_judge" in rollup["by_role"]
    assert "diff_scope" in rollup["by_role"]
    assert rollup["by_backend"]["codex"] == pytest.approx(1.10)
    assert "unknown" not in rollup["by_backend"]


def test_delta_walk_asserts_baseline_prefix_unchanged(tmp_path: Path):
    """Sanity-assert: if a concurrent writer reordered or truncated
    the prior portion of spec.verdict_refs / implementation_refs,
    our baseline-indexed slice would silently misalign. The helper
    asserts the prefix matches ``prior_spec`` and fails loudly.
    """
    corpus = _make_corpus(tmp_path)

    # Seed baseline with one verdict so the prefix is non-empty;
    # without that the assert checks ``() == ()`` and the drift
    # detector can't fire.
    spec = corpus.latest_spec()
    seed = _make_verdict(
        corpus, rubric="seeded", kind="critic", cost=0.0,
    )
    corpus.put_record(seed, kind="verdicts")
    spec = spec.with_verdict(seed)
    corpus.put_spec(spec)

    prior_spec = corpus.latest_spec()
    pre_verdict_ref_count = len(prior_spec.verdict_refs)
    pre_implementation_ref_count = len(prior_spec.implementation_refs)

    # Drift the baseline portion: replace the seeded verdict ref
    # with a bogus one. The baseline-indexed slice would now
    # silently misalign without the prefix assert.
    bogus_ref = "0" * 64
    drifted = prior_spec._next_version(
        verdict_refs=(bogus_ref,),  # was (seed_ref,)
    )

    costs = CostAccumulator()
    with pytest.raises(AssertionError, match="prefix drifted"):
        _record_spec_delta_into_costs(
            costs,
            corpus=corpus,
            prior_spec=prior_spec,
            current_spec=drifted,
            pre_verdict_ref_count=pre_verdict_ref_count,
            pre_implementation_ref_count=pre_implementation_ref_count,
        )


def test_delta_walk_skips_assertion_when_prior_spec_none(tmp_path: Path):
    """The prior_spec assert is skipped when None — call sites that
    can't / don't capture a baseline spec (test fixtures, ad-hoc
    rollup helpers) can still use the helper."""
    corpus = _make_corpus(tmp_path)
    current = corpus.latest_spec()
    impl = _make_impl(corpus, slice_id="s0", cost=0.10)
    corpus.put_record(impl, kind="implementations")
    current = current.with_implementation(impl)
    corpus.put_spec(current)

    costs = CostAccumulator()
    _record_spec_delta_into_costs(
        costs,
        corpus=corpus,
        prior_spec=None,
        current_spec=corpus.latest_spec(),
        pre_verdict_ref_count=0,
        pre_implementation_ref_count=0,
    )
    assert costs.total_cost_usd == pytest.approx(0.10)


def test_delta_walk_tolerates_missing_corpus_record(tmp_path: Path, caplog):
    """If a verdict ref doesn't load (corpus inconsistency), the
    helper must skip it with a warning rather than raising — a
    partial cost rollup is better than a crashed summary write."""
    import logging

    corpus = _make_corpus(tmp_path)
    spec = corpus.latest_spec()
    baseline_v = len(spec.verdict_refs)
    baseline_i = len(spec.implementation_refs)

    # The strict Corpus refuses to publish a missing reference. Simulate
    # post-commit disk loss instead: publish a valid verdict, then remove its
    # content-addressed blob before the rollup reads it.
    verdict = _make_verdict(
        corpus, rubric="deleted_after_commit", kind="critic", cost=1.0,
    )
    committed = corpus.commit(verdicts=(verdict,))
    verdict_ref = committed.refs.verdict_refs[0]
    verdict_path = (
        corpus.spec_dir / "artifacts" / "verdicts"
        / f"{verdict_ref.split(':', 1)[1]}.bin"
    )
    verdict_path.unlink()

    costs = CostAccumulator()
    with caplog.at_level(logging.WARNING):
        _record_spec_delta_into_costs(
            costs,
            corpus=corpus,
            prior_spec=spec,
            current_spec=committed.spec,
            pre_verdict_ref_count=baseline_v,
            pre_implementation_ref_count=baseline_i,
        )

    # Helper must log + continue; total stays at 0.
    assert costs.total_cost_usd == 0.0
    assert any(
        "could not load verdict" in r.message
        for r in caplog.records
    )


def test_authoritative_summary_includes_unpersisted_planner_spend():
    costs = CostAccumulator()
    costs.record(
        role="plan_critic",
        usage=Usage(cost_usd=2.0),
        backend_name="codex",
    )

    summary = _authoritative_cost_summary(
        costs,
        budget_usd=6.0,
        spent_usd=5.0,
    )

    assert summary["total_usd"] == pytest.approx(5.0)
    assert summary["record_rollup_usd"] == pytest.approx(2.0)
    assert summary["unattributed_usd"] == pytest.approx(3.0)
    assert summary["remaining_usd"] == pytest.approx(1.0)


def test_planning_usage_tracks_tokens_and_logical_calls_separately_from_usd():
    class _Backend:
        name = "openai_compat"
        model = "nvidia/zai-org/glm-5.2"

        def invoke(self, invocation):
            return SimpleNamespace(
                usage=Usage(
                    input_tokens=123,
                    output_tokens=45,
                    cost_usd=0.0,
                ),
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    planning_usage = CostAccumulator()
    backend = _PlanningUsageBackend(
        _Backend(), role="architect_planner", usage=planning_usage,
    )
    backend.invoke(None)
    backend.invoke(None)

    summary = _authoritative_cost_summary(
        CostAccumulator(),
        budget_usd=None,
        spent_usd=0.0,
        planning_usage=planning_usage,
    )

    assert summary["total_usd"] == 0.0
    assert summary["total_input_tokens"] == 246
    assert summary["total_output_tokens"] == 90
    assert summary["planning_backend_calls"] == 2
    assert summary["planning_usage"]["by_role"][
        "architect_planner"
    ]["calls"] == 2
    assert "total_cache_creation_tokens" not in summary
    assert "pricing_evidence" not in summary
    assert "pricing_evidence_by_call" not in summary


def test_planning_backend_preserves_invocation_pricing_evidence():
    evidence = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
    )

    class _Backend:
        name = "codex"
        model = "gpt-5.5"

        def invoke(self, invocation):
            return SimpleNamespace(
                usage=evidence.usage,
                pricing_evidence=evidence,
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    planning_usage = CostAccumulator()
    backend = _PlanningUsageBackend(
        _Backend(),
        role="architect_planner",
        usage=planning_usage,
    )

    backend.invoke(None)

    summary = planning_usage.to_dict()
    assert summary["total_cache_creation_tokens"] == 10
    assert summary["total_cache_read_tokens"] == 40
    assert summary["pricing_evidence"]["complete"] is True
    assert summary["pricing_evidence_schema"] == (
        "crucible.usage-pricing-ledger.v1"
    )
    assert summary["pricing_evidence_by_call"][0]["role"] == (
        "architect_planner"
    )


def test_incomplete_planner_candidate_is_diagnostic_only(tmp_path):
    corpus = _make_corpus(tmp_path)
    candidate = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "contract",
                "directive": {
                    "slice_id": "contract",
                    "intent": "produce a diagnostic contract",
                    "requirement_refs": [],
                    "files_in_scope": [],
                    "creates_paths": ["native/contract.json"],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            }],
        }],
    }, separators=(",", ":"))
    failure = _PlanningAccountingIncomplete(
        "architect_planner",
        ("cost_complete", "token_usage_complete"),
        result=SimpleNamespace(final_message=candidate),
    )

    ref = _retain_incomplete_planner_candidate(corpus, failure)

    assert ref.startswith("sha256:")
    assert corpus.get_artifact(ref) == candidate.encode("utf-8")
    assert corpus.latest_spec().tier_dag is None


def test_plan_outcome_pricing_must_reconcile_backend_call_delta():
    backend_evidence = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
    )
    outcome_evidence = _exact_pricing_evidence(
        input_tokens=101,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
    )
    planning_usage = CostAccumulator()
    before = planning_usage.to_dict()
    planning_usage.record(
        role="architect_planner",
        usage=backend_evidence.usage,
        backend_name="codex",
        model="gpt-5.5",
        pricing_evidence=backend_evidence,
    )
    matching_outcome = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
        pricing_evidence=backend_evidence,
        execution_completeness=_COMPLETE_EXECUTION,
    )
    _require_plan_outcome_pricing_reconciliation(
        outcome=matching_outcome,
        usage_before=before,
        planning_usage=planning_usage,
        role="architect_planning",
    )
    outcome = SimpleNamespace(
        input_tokens=101,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
        pricing_evidence=outcome_evidence,
        execution_completeness=_COMPLETE_EXECUTION,
    )

    with pytest.raises(RuntimeError, match="incomplete aggregate accounting"):
        _require_plan_outcome_pricing_reconciliation(
            outcome=outcome,
            usage_before=before,
            planning_usage=planning_usage,
            role="architect_planning",
        )

    debt = planning_usage.to_dict()["accounting_debt_by_role"]
    assert debt["architect_planning"] == [
        "cost_complete",
        "token_usage_complete",
    ]


def test_plan_outcome_retry_pricing_reconciles_two_physical_calls():
    first = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.1,
    )
    second = _exact_pricing_evidence(
        input_tokens=80,
        output_tokens=15,
        cache_creation_tokens=5,
        cache_read_tokens=25,
        cost_usd=0.2,
    )
    aggregate = UsagePricingEvidence(
        segments=(*first.segments, *second.segments),
        complete=True,
    )
    aggregate_usage = aggregate.usage
    planning_usage = CostAccumulator()
    before = planning_usage.to_dict()
    for evidence in (first, second):
        planning_usage.record(
            role="architect_critic",
            usage=evidence.usage,
            backend_name="codex",
            model="gpt-5.5",
            pricing_evidence=evidence,
        )
    outcome = SimpleNamespace(
        input_tokens=aggregate_usage.input_tokens,
        output_tokens=aggregate_usage.output_tokens,
        cache_creation_tokens=aggregate_usage.cache_creation_tokens,
        cache_read_tokens=aggregate_usage.cache_read_tokens,
        cost_usd=aggregate_usage.cost_usd,
        pricing_evidence=aggregate,
        execution_completeness=_COMPLETE_EXECUTION,
    )

    _require_plan_outcome_pricing_reconciliation(
        outcome=outcome,
        usage_before=before,
        planning_usage=planning_usage,
        role="architect_planning",
    )

    summary = planning_usage.to_dict()
    assert len(summary["pricing_evidence_by_call"]) == 2
    assert summary["pricing_evidence"]["segments"] == [
        {
            "input_tokens": segment.input_tokens,
            "output_tokens": segment.output_tokens,
            "cache_creation_tokens": segment.cache_creation_tokens,
            "cache_read_tokens": segment.cache_read_tokens,
            "cost_usd": segment.cost_usd,
        }
        for segment in aggregate.segments
    ]


def test_plan_outcome_pricing_reconciles_exact_zero_pretransport_denial():
    paid = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
    )
    denied = UsagePricingEvidence(segments=(), complete=True)
    planning_usage = CostAccumulator()
    before = planning_usage.to_dict()
    planning_usage.record(
        role="architect_planner",
        usage=paid.usage,
        backend_name="codex",
        model="gpt-5.5",
        pricing_evidence=paid,
    )
    planning_usage.record(
        role="architect_critic",
        usage=Usage(),
        backend_name="codex",
        model="gpt-5.5",
        pricing_evidence=denied,
    )
    outcome = SimpleNamespace(
        input_tokens=paid.usage.input_tokens,
        output_tokens=paid.usage.output_tokens,
        cache_creation_tokens=paid.usage.cache_creation_tokens,
        cache_read_tokens=paid.usage.cache_read_tokens,
        cost_usd=paid.usage.cost_usd,
        pricing_evidence=paid,
        execution_completeness=_COMPLETE_EXECUTION,
    )

    _require_plan_outcome_pricing_reconciliation(
        outcome=outcome,
        usage_before=before,
        planning_usage=planning_usage,
        role="architect_planning",
    )

    summary = planning_usage.to_dict()
    assert len(summary["pricing_evidence_by_call"]) == 2
    denied_call = summary["pricing_evidence_by_call"][1]
    assert denied_call["pricing_evidence"]["segments"] == []
    assert denied_call["pricing_evidence"]["complete"] is True
    assert summary["accounting_debt"] is False


def test_authoritative_summary_combines_ordered_planning_and_tier_pricing():
    planning_evidence = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.25,
    )
    tier_evidence = _exact_pricing_evidence(
        input_tokens=60,
        output_tokens=15,
        cache_creation_tokens=5,
        cache_read_tokens=20,
        cost_usd=0.15,
    )
    planning = CostAccumulator()
    planning.record(
        role="architect_planner",
        usage=planning_evidence.usage,
        backend_name="codex",
        model="gpt-5.5",
        pricing_evidence=planning_evidence,
    )
    tier = CostAccumulator()
    tier.record(
        role="implementer",
        usage=tier_evidence.usage,
        backend_name="claude",
        model="claude-opus-4-1",
        pricing_evidence=tier_evidence,
    )
    tier_summary = tier.to_dict()

    summary = _authoritative_cost_summary(
        CostAccumulator(),
        budget_usd=1.0,
        spent_usd=0.4,
        planning_usage=planning,
        tier_input_tokens=60,
        tier_output_tokens=15,
        tier_cache_creation_tokens=5,
        tier_cache_read_tokens=20,
        tier_pricing_evidence_by_call=tuple(
            tier_summary["pricing_evidence_by_call"]
        ),
        tier_pricing_evidence_active=True,
    )

    assert summary["total_input_tokens"] == 160
    assert summary["total_cache_creation_tokens"] == 15
    assert summary["total_cache_read_tokens"] == 60
    assert summary["pricing_evidence"]["complete"] is True
    assert summary["pricing_evidence_schema"] == (
        "crucible.usage-pricing-ledger.v1"
    )
    assert [
        call["role"] for call in summary["pricing_evidence_by_call"]
    ] == ["architect_planner", "implementer"]
    assert summary["by_role"]["architect_planner"]["calls"] == 1
    assert summary["by_role"]["implementer"]["calls"] == 1
    assert summary["record_rollup_by_role"] == {}
    assert arch._pricing_summary_projection(
        summary,
        label="test authoritative summary",
    ).valid


@pytest.mark.parametrize(
    "tamper",
    [
        "total_usd",
        "role_cost",
        "backend_cost",
        "model_cost",
        "role_input",
        "role_route",
        "empty_model",
    ],
)
def test_extended_pricing_projection_rejects_raw_projection_drift(tamper):
    evidence = _exact_pricing_evidence(
        input_tokens=100,
        output_tokens=20,
        cache_creation_tokens=10,
        cache_read_tokens=40,
        cost_usd=0.123456789,
    )
    costs = CostAccumulator()
    costs.record(
        role="architect_planner",
        usage=evidence.usage,
        backend_name="codex",
        model="gpt-5.5",
        pricing_evidence=evidence,
    )
    summary = json.loads(json.dumps(costs.to_dict()))
    if tamper == "total_usd":
        summary["total_usd"] += 0.0000004
    elif tamper == "role_cost":
        summary["by_role"]["architect_planner"]["cost_usd"] += 0.0000004
    elif tamper == "backend_cost":
        summary["by_backend"]["codex"] += 0.0000004
    elif tamper == "model_cost":
        summary["by_model"]["gpt-5.5"] += 0.0000004
    elif tamper == "role_input":
        summary["by_role"]["architect_planner"]["input_tokens"] += 1
    elif tamper == "role_route":
        summary["by_role"]["architect_planner"]["backends"].append(
            "forged"
        )
    else:
        summary["pricing_evidence_by_call"][0]["model"] = ""

    projection = arch._pricing_summary_projection(
        summary,
        label="tampered pricing",
    )

    assert projection.active
    assert not projection.valid


def test_persisted_review_tokens_are_not_double_counted_by_planning_tracker():
    persisted = CostAccumulator()
    persisted.record(
        role="plan_critic",
        usage=Usage(input_tokens=100, output_tokens=20),
        backend_name="openai_compat",
    )
    call_side = CostAccumulator()
    call_side.record(
        role="architect_critic",
        usage=Usage(input_tokens=100, output_tokens=20),
        backend_name="openai_compat",
    )
    call_side.record(
        role="architect_planner",
        usage=Usage(input_tokens=50, output_tokens=10),
        backend_name="openai_compat",
    )

    summary = _authoritative_cost_summary(
        persisted,
        budget_usd=None,
        spent_usd=0.0,
        planning_usage=call_side,
    )

    assert summary["total_input_tokens"] == 150
    assert summary["total_output_tokens"] == 30
    assert summary["record_rollup_input_tokens"] == 100
    assert summary["record_rollup_output_tokens"] == 20
    assert summary["token_authority"]["planning_input_tokens"] == 150
    assert summary["planning_backend_calls"] == 2


def test_authoritative_summary_adds_child_tier_tokens_not_record_tokens():
    records = CostAccumulator()
    records.record(
        role="implementer",
        usage=Usage(input_tokens=100, output_tokens=20),
        backend_name="openai_compat",
    )
    planning = CostAccumulator()
    planning.record(
        role="architect_planner",
        usage=Usage(input_tokens=50, output_tokens=10),
        backend_name="openai_compat",
    )

    summary = _authoritative_cost_summary(
        records,
        budget_usd=None,
        spent_usd=0.0,
        planning_usage=planning,
        tier_input_tokens=300,
        tier_output_tokens=60,
    )

    assert summary["total_input_tokens"] == 350
    assert summary["total_output_tokens"] == 70
    assert summary["record_rollup_input_tokens"] == 100
    assert summary["record_rollup_output_tokens"] == 20
    assert summary["token_authority"]["tier_input_tokens"] == 300


def test_run_architect_preserves_usage_before_untyped_transport_exception(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)

    class _StatusError(RuntimeError):
        status_code = 500

    class _Backend:
        name = "openai_compat"
        model = "nvidia/zai-org/glm-5.2"

        def __init__(self, role: str) -> None:
            self.role = role

        def invoke(self, invocation):
            if self.role == "architect_critic":
                raise _StatusError("provider returned HTTP 500")
            return SimpleNamespace(
                usage=Usage(input_tokens=100, output_tokens=20),
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    class _Pool:
        def get(self, role, **kwargs):
            return _Backend(role)

    @contextmanager
    def _pool(*args, **kwargs):
        yield _Pool()

    def _run_plan(spec, **kwargs):
        kwargs["planner_backend"].invoke(None)
        kwargs["critic_backend"].invoke(None)
        raise AssertionError("unreachable")

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(arch, "run_plan", _run_plan)
    checkpoints: list[arch.ArchitectCheckpoint] = []

    def _checkpoint(checkpoint: arch.ArchitectCheckpoint):
        checkpoints.append(checkpoint)
        return {"example.product": {"terminal": True}}

    with pytest.raises(_StatusError):
        arch.run_architect(
            planner_count=1,
            spec=Path(corpus.spec_dir),
            cwd=str(tmp_path),
            skip_validate=True,
            checkpoint_hook=_checkpoint,
            enable_source_tools=True,
            planner_tool_call_limit=2,
            planner_tool_result_bytes=10,
        )

    summary = _architect_summary(corpus)
    partial = summary["partial_summary"]
    assert partial["status"] == "planning_exception"
    assert partial["transport_failure"] is True
    assert partial["costs"]["total_input_tokens"] == 100
    assert partial["costs"]["total_output_tokens"] == 20
    # One successful Planner call plus the throwing Critic call.
    assert partial["costs"]["planning_backend_calls"] == 2
    assert partial["costs"]["rollup_incomplete"] is True
    assert partial["budget_remaining_usd"] == 0.0
    assert partial["costs"]["remaining_usd"] == 0.0
    assert partial["tiers_executed"] == 0
    assert partial["tiers_failed"] == 0
    assert partial["tier_results"] == []
    assert partial["checkpoint_extensions"] == {
        "example.product": {"terminal": True},
    }
    assert len(checkpoints) == 1
    assert checkpoints[0].kind == "final"
    event_paths = [
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/events.jsonl")
        if "_architect_" in path.parent.name
    ]
    events = [
        json.loads(line)
        for line in event_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    terminal = [
        event
        for event in events
        if event.get("phase") == "planner_tool_budget"
        and event.get("status") == "terminal"
    ]
    assert len(terminal) == 1
    assert terminal[0]["terminal_reason"] == "planning_exception"
    assert terminal[0]["calls_used"] == 0
    assert terminal[0]["result_bytes_used"] == 0
    assert terminal[0]["exhausted_by"] is None
    assert terminal[0]["dispatch_in_flight"] is False


def test_remaining_budget_never_refreshes_or_goes_negative():
    assert _remaining_budget(None, 99.0) is None
    assert _remaining_budget(10.0, 3.25) == pytest.approx(6.75)
    assert _remaining_budget(10.0, 12.0) == 0.0


def test_finite_usd_budget_is_reported_as_soft_not_hard():
    summary = _budget_enforcement_summary(3.0)

    assert summary == {
        "classification": "soft_ceiling",
        "hard_cap_guaranteed": False,
        "pretransport_zero_remaining_rejected": True,
        "worst_case_call_reservation_usd": None,
        "overshoot_possible": True,
    }


def test_architect_attaches_one_shared_planner_dispatch_budget(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    attachments: dict[str, dict] = {}

    class _SourceTools:
        read_only_names = {"read"}
        write_names: set[str] = set()

        def to_anthropic_tools(self):
            return [{
                "name": "read",
                "description": "read",
                "input_schema": {"type": "object"},
            }]

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            }]

        def dispatch(self, name, arguments):
            return "result"

    class _Backend:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

    class _Pool:
        def get(self, role, **kwargs):
            attachments[role] = kwargs
            return _Backend()

    @contextmanager
    def _pool(*args, **kwargs):
        yield _Pool()

    def _budget_outcome(spec, **kwargs):
        prompt_toolset = kwargs["prompt_toolset"]
        attachments["prompt"] = {"toolset": prompt_toolset}
        admitted = prompt_toolset.dispatch("read", {})
        assert admitted.startswith("result")
        assert "CALL LIMIT EXHAUSTED" in admitted
        with pytest.raises(
            ToolDispatchBudgetExceeded,
            match="call limit exhausted",
        ):
            prompt_toolset.dispatch("read", {})
        return PlanOutcome(
            spec=spec,
            candidate_dag=None,
            candidate_ref=None,
            decision="inconclusive",
            status="budget_exhausted",
            rounds_used=0,
            final_critic_decision=None,
            final_judge_decision=None,
            reason="stop",
            cost_usd=0.0,
        )

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "build_source_tools", lambda ctx: _SourceTools())
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(arch, "run_plan", _budget_outcome)

    policy = DevelopmentBudgetPolicy(
        task_budget_usd=1.0,
        planning_review_cap_usd=0.4,
        implementation_verifier_reserve_usd=0.5,
        contingency_usd=0.1,
        roles=(
            RoleAllocation(
                role="architect_planner",
                group="planning_review",
                max_calls=1,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.4,
            ),
        ),
    )
    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        enable_source_tools=True,
        planner_tool_call_limit=1,
        planner_tool_result_bytes=11,
        skip_validate=True,
        development_budget_policy=policy,
    )

    assert rc == 1
    planner = attachments["architect_planner"]
    assert isinstance(planner["toolset"], BoundedReadOnlyToolSet)
    assert planner["context_budget"] == (
        11 + PLANNER_TOOL_CONTROL_ALLOWANCE_BYTES
    )
    assert attachments["prompt"]["toolset"] is planner["toolset"]
    event_paths = [
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/events.jsonl")
        if "_architect_" in path.parent.name
    ]
    assert len(event_paths) == 1
    development_metrics = _architect_summary(corpus)[
        "development_budget"
    ]["autonomous_development_metrics"]
    assert development_metrics["source_calls_dispatched"] == 1
    assert development_metrics["source_calls_denied"] == 1
    events = [
        json.loads(line)
        for line in event_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    terminal = [
        event
        for event in events
        if event.get("phase") == "planner_tool_budget"
        and event.get("status") == "terminal"
    ]
    assert terminal == [{
        "phase": "planner_tool_budget",
        "status": "terminal",
        "terminal_reason": "planning_budget_exhausted",
        "max_calls": 1,
        "call_tranches": [1],
        "active_call_limit": 1,
        "active_tranche_index": 0,
        "invocations_started": 0,
        "denied_calls": 1,
        "max_result_bytes": 11,
        "adapter_context_budget_bytes": (
            11 + PLANNER_TOOL_CONTROL_ALLOWANCE_BYTES
        ),
        "control_allowance_bytes": PLANNER_TOOL_CONTROL_ALLOWANCE_BYTES,
        "calls_used": 1,
        "result_bytes_used": 6,
        "exhausted_by": "call_count",
        "dispatch_in_flight": False,
        "enforcement": "workflow_dispatch_boundary",
    }]


@pytest.mark.parametrize(
    (
        "caller_max_turns",
        "expected_synthesis_max_turns",
        "expected_synthesis_error",
    ),
    [
        pytest.param(90, 2, None, id="structured-output-headroom"),
        pytest.param(1, 1, "max_turns", id="lower-caller-cap"),
    ],
)
def test_planner_tranches_allow_one_plain_synthesis_then_terminal(
    caller_max_turns,
    expected_synthesis_max_turns,
    expected_synthesis_error,
):
    class _SourceTools:
        read_only_names = {"read"}
        write_names: set[str] = set()

        def to_anthropic_tools(self):
            return [{
                "name": "read",
                "description": "read",
                "input_schema": {"type": "object"},
            }]

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            }]

        def dispatch(self, name, arguments):
            return "unused"

    class _Backend:
        name = "codex"
        model = "gpt-test"

        def __init__(self, *, minimum_turns=1):
            self.calls = 0
            self.invocations = []
            self.minimum_turns = minimum_turns

        def invoke(self, invocation):
            self.calls += 1
            self.invocations.append(invocation)
            return SimpleNamespace(
                usage=Usage(),
                error=(
                    None
                    if invocation.max_turns >= self.minimum_turns
                    else SimpleNamespace(kind="max_turns")
                ),
                completeness=_COMPLETE_EXECUTION,
            )

    raw = _Backend()
    # Claude structured output can require one turn to emit the schema call
    # and a second turn to consume its result and terminate. Model the failed
    # one-turn boundary so this regression test fails if the cap returns to 1.
    synthesis = _Backend(minimum_turns=2)
    tool_budget = BoundedReadOnlyToolSet(
        _SourceTools(),
        max_calls=2,
        max_result_bytes=10,
        call_tranches=(1, 1),
    )
    backend = _PlanningUsageBackend(
        raw,
        role="architect_planner",
        usage=CostAccumulator(),
        tool_budget=tool_budget,
        synthesis_backend_factory=lambda: synthesis,
    )
    invocation = Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=caller_max_turns,
        budget_usd=None,
    )

    assert backend.invoke(invocation).error is None
    assert backend.invoke(invocation).error is None
    synthesis_result = backend.invoke(invocation)
    if expected_synthesis_error is None:
        assert synthesis_result.error is None
    else:
        assert synthesis_result.error is not None
        assert synthesis_result.error.kind == expected_synthesis_error
    terminal = backend.invoke(invocation)

    assert terminal.error is not None
    assert terminal.error.kind == "budget_exhausted"
    assert terminal.completeness.token_usage_complete is True
    assert terminal.completeness.cost_complete is True
    assert raw.calls == 2
    assert synthesis.calls == 1
    assert "PLANNER SOURCE TOOL BUDGET CONTRACT" in (
        raw.invocations[0].system
    )
    assert "1 source dispatch call(s) currently available" in (
        raw.invocations[0].system
    )
    assert "Every source call counts, including each call emitted in a " in (
        raw.invocations[0].system
    )
    assert "parallel batch" in raw.invocations[0].system
    assert "2 source dispatch call(s) currently available" in (
        raw.invocations[1].system
    )
    assert (
        synthesis.invocations[0].max_turns
        == expected_synthesis_max_turns
    )
    assert synthesis.invocations[0].tools is None
    assert "SOURCE AUTHORITY TERMINAL OVERRIDE" in (
        synthesis.invocations[0].system
    )


def test_planner_terminal_synthesis_limit_matches_bounded_plan_attempts():
    policy = SimpleNamespace(
        max_critique_rounds=4,
        inflation_compaction_retries=1,
    )

    assert arch._planner_terminal_synthesis_limit(
        policy,
        (24, 4, 4),
    ) == 4
    assert arch._planner_terminal_synthesis_limit(None, (24, 4, 4)) == 1
    assert arch._planner_terminal_synthesis_limit(policy, None) == 1


def test_planner_allows_bounded_revisions_after_source_authority_closes():
    class _SourceTools:
        read_only_names = {"read"}
        write_names: set[str] = set()

        @staticmethod
        def to_anthropic_tools():
            return [{
                "name": "read",
                "description": "read",
                "input_schema": {"type": "object"},
            }]

        @staticmethod
        def to_openai_tools():
            return [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            }]

        @staticmethod
        def dispatch(name, arguments):
            return "unused"

    class _Backend:
        name = "codex"
        model = "gpt-test"

        def __init__(self):
            self.invocations = []

        def invoke(self, invocation):
            self.invocations.append(invocation)
            return SimpleNamespace(
                usage=Usage(),
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    raw = _Backend()
    synthesis = _Backend()
    tool_budget = BoundedReadOnlyToolSet(
        _SourceTools(),
        max_calls=3,
        max_result_bytes=1_000,
        call_tranches=(1, 1, 1),
    )
    backend = _PlanningUsageBackend(
        raw,
        role="architect_planner",
        usage=CostAccumulator(),
        tool_budget=tool_budget,
        synthesis_backend_factory=lambda: synthesis,
        terminal_synthesis_limit=4,
    )
    invocation = Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=90,
        budget_usd=None,
    )

    for _ in range(3):
        assert backend.invoke(invocation).error is None
    source_closed_snapshot = tool_budget.snapshot()
    for _ in range(4):
        assert backend.invoke(invocation).error is None
    terminal = backend.invoke(invocation)

    assert terminal.error is not None
    assert terminal.error.kind == "budget_exhausted"
    assert len(raw.invocations) == 3
    assert len(synthesis.invocations) == 4
    assert all(item.tools is None for item in synthesis.invocations)
    assert all(
        "SOURCE AUTHORITY TERMINAL OVERRIDE" in item.system
        for item in synthesis.invocations
    )
    assert [
        f"attempt {index} of 4" in item.user
        for index, item in enumerate(synthesis.invocations, start=1)
    ] == [True, True, True, True]
    assert tool_budget.begin_invocation() is False
    assert tool_budget.snapshot().calls_used == source_closed_snapshot.calls_used


@pytest.mark.parametrize("limit", (True, False, 0, -1, 1.5, "4", None))
def test_planner_rejects_invalid_terminal_synthesis_limit(limit):
    with pytest.raises(
        ValueError,
        match="terminal_synthesis_limit must be a positive integer",
    ):
        _PlanningUsageBackend(
            SimpleNamespace(),
            role="architect_planner",
            usage=CostAccumulator(),
            terminal_synthesis_limit=limit,
        )


def test_child_consumed_discovery_tranche_preserves_repairs_then_latches():
    class _SourceTools:
        read_only_names = {"read"}
        write_names: set[str] = set()

        def to_anthropic_tools(self):
            return [{
                "name": "read",
                "description": "read",
                "input_schema": {"type": "object"},
            }]

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            }]

        def dispatch(self, name, arguments):
            return f"grounded:{arguments['path']}"

    class _Backend:
        name = "codex"
        model = "gpt-test"

        def __init__(self):
            self.invocations = []

        def invoke(self, invocation):
            self.invocations.append(invocation)
            return SimpleNamespace(
                usage=Usage(),
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    source_tools = _SourceTools()
    raw = _Backend()
    synthesis = _Backend()
    tool_budget = BoundedReadOnlyToolSet(
        source_tools,
        max_calls=3,
        max_result_bytes=1_000,
        call_tranches=(1, 1, 1),
    )
    child_tools = tool_budget.share(source_tools)
    backend = _PlanningUsageBackend(
        raw,
        role="architect_planner",
        usage=CostAccumulator(),
        tool_budget=tool_budget,
        synthesis_backend_factory=lambda: synthesis,
    )
    invocation = Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=5,
        budget_usd=None,
    )

    assert backend.claim_initial_source_invocation() is True
    assert child_tools.dispatch("read", {"path": "src/child.py"}).startswith(
        "grounded:"
    )

    # The parent does not misclassify a child-consumed discovery tranche as
    # terminal exhaustion and does not borrow either correction tranche.
    assert backend.invoke(invocation).error is None
    assert raw.invocations == []
    assert len(synthesis.invocations) == 1
    assert synthesis.invocations[0].max_turns == 2
    assert synthesis.invocations[0].tools is None
    assert "SOURCE DISCOVERY TRANCHE CONSUMED" in (
        synthesis.invocations[0].system
    )
    first = tool_budget.snapshot()
    assert first.invocations_started == 1
    assert first.active_tranche_index == 0

    # Later workflow invocations can still claim the two repair tranches.
    assert backend.invoke(invocation).error is None
    assert backend.invoke(invocation).error is None
    assert len(raw.invocations) == 2

    # Once the terminal text-only fallback starts, both the planner wrapper
    # and every shared ToolSet stay closed even though calls 2 and 3 were not
    # physically dispatched.
    assert backend.invoke(invocation).error is None
    assert synthesis.invocations[1].max_turns == 2
    assert synthesis.invocations[1].tools is None
    assert "SOURCE AUTHORITY TERMINAL OVERRIDE" in (
        synthesis.invocations[1].system
    )
    terminal_snapshot = tool_budget.snapshot()
    assert terminal_snapshot.invocations_started == 4
    terminal = backend.invoke(invocation)
    assert terminal.error is not None
    assert terminal.error.kind == "budget_exhausted"
    assert len(raw.invocations) == 2
    assert len(synthesis.invocations) == 2
    assert tool_budget.snapshot() == terminal_snapshot
    with pytest.raises(
        ToolDispatchBudgetExceeded,
        match="permanently closed",
    ):
        child_tools.dispatch("read", {"path": "src/must-not-run.py"})


def test_planner_repair_and_synthesis_receive_cached_grounded_excerpts():
    class _SourceTools:
        read_only_names = {"read"}
        write_names: set[str] = set()

        def to_anthropic_tools(self):
            return [{
                "name": "read",
                "description": "read",
                "input_schema": {"type": "object"},
            }]

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            }]

        def dispatch(self, name, arguments):
            return "def grounded_symbol(): pass"

    class _Backend:
        name = "codex"
        model = "gpt-test"

        def __init__(self):
            self.invocations = []

        def invoke(self, invocation):
            self.invocations.append(invocation)
            return SimpleNamespace(
                usage=Usage(),
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    raw = _Backend()
    synthesis = _Backend()
    tool_budget = BoundedReadOnlyToolSet(
        _SourceTools(),
        max_calls=2,
        max_result_bytes=1_000,
        call_tranches=(1, 1),
    )
    backend = _PlanningUsageBackend(
        raw,
        role="architect_planner",
        usage=CostAccumulator(),
        tool_budget=tool_budget,
        synthesis_backend_factory=lambda: synthesis,
    )
    invocation = Invocation(
        system="system",
        user="user",
        tools=None,
        max_turns=5,
        budget_usd=None,
    )

    backend.invoke(invocation)
    tool_budget.dispatch("read", {"path": "src/grounded.py"})
    backend.invoke(invocation)
    backend.invoke(invocation)

    assert "src/grounded.py" in raw.invocations[1].user
    assert "grounded_symbol" in raw.invocations[1].user
    assert "untrusted repository data" in raw.invocations[1].user
    assert "src/grounded.py" in synthesis.invocations[0].user
    assert "SOURCE AUTHORITY TERMINAL OVERRIDE" in (
        synthesis.invocations[0].system
    )


def test_planner_budget_snapshot_survives_backend_setup_exception(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)

    class _SourceTools:
        read_only_names = {"read"}
        write_names: set[str] = set()

        def to_anthropic_tools(self):
            return [{
                "name": "read",
                "description": "read",
                "input_schema": {"type": "object"},
            }]

        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            }]

        def dispatch(self, name, arguments):
            return "result"

    class _Pool:
        def get(self, role, **kwargs):
            raise RuntimeError("backend setup failed")

    @contextmanager
    def _pool(*args, **kwargs):
        yield _Pool()

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "build_source_tools", lambda ctx: _SourceTools())

    with pytest.raises(RuntimeError, match="backend setup failed"):
        arch.run_architect(
            planner_count=1,
            spec=Path(corpus.spec_dir),
            cwd=str(tmp_path),
            enable_source_tools=True,
            planner_tool_call_limit=2,
            planner_tool_result_bytes=10,
            skip_validate=True,
        )

    event_path = next(
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/events.jsonl")
        if "_architect_" in path.parent.name
    )
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    terminal = [
        event
        for event in events
        if event.get("phase") == "planner_tool_budget"
        and event.get("status") == "terminal"
    ]
    assert len(terminal) == 1
    assert terminal[0]["terminal_reason"] == "architect_scope_exit"
    assert terminal[0]["calls_used"] == 0
    assert terminal[0]["result_bytes_used"] == 0
    assert terminal[0]["dispatch_in_flight"] is False


def test_deliberate_stop_is_explicitly_partial_but_not_a_failure():
    assert _terminal_status(
        failed_tiers=[],
        accounting_incomplete=False,
        budget_exhausted=False,
        stopped_early=True,
    ) == "partial_stopped"


def test_budget_status_takes_precedence_over_generic_tier_failure():
    assert _terminal_status(
        failed_tiers=[{"tier_rc": 1, "status": "budget_exhausted"}],
        accounting_incomplete=False,
        budget_exhausted=True,
        stopped_early=False,
    ) == "budget_exhausted"


def test_retryable_tier_failure_is_not_collapsed_to_permanent_partial():
    assert _terminal_status(
        failed_tiers=[{
            "tier_rc": 1,
            "status": "all_slices_transport_failed",
            "terminal_class": "retryable_failure",
        }],
        accounting_incomplete=False,
        budget_exhausted=False,
        stopped_early=False,
    ) == "retryable_tier_failure"

    assert _terminal_status(
        failed_tiers=[{
            "tier_rc": 1,
            "status": "slice_failed",
            "terminal_class": "permanent_failure",
        }],
        accounting_incomplete=False,
        budget_exhausted=False,
        stopped_early=False,
    ) == "partial_failed"


def _failed_tier(status: str, terminal_class: str | None) -> dict:
    result: dict = {"tier_rc": 1, "status": status}
    if terminal_class is not None:
        result["terminal_class"] = terminal_class
    return result


@pytest.mark.parametrize(
    ("failed_tiers", "expected"),
    [
        pytest.param(
            [_failed_tier("slice_inconclusive", "inconclusive")],
            "inconclusive_tier_failure",
            id="inconclusive",
        ),
        pytest.param(
            [_failed_tier("cancelled", "cancelled")],
            "cancelled",
            id="cancelled",
        ),
        pytest.param(
            [
                _failed_tier("all_slices_transport_failed", "retryable_failure"),
                _failed_tier("review_inconclusive", "inconclusive"),
            ],
            "inconclusive_tier_failure",
            id="retryable-and-inconclusive",
        ),
        pytest.param(
            [
                _failed_tier("cancelled", "cancelled"),
                _failed_tier("all_slices_transport_failed", "retryable_failure"),
            ],
            "inconclusive_tier_failure",
            id="cancelled-and-retryable",
        ),
        pytest.param(
            [
                _failed_tier("all_slices_transport_failed", "retryable_failure"),
                _failed_tier("all_slices_transport_failed", "retryable_failure"),
            ],
            "retryable_tier_failure",
            id="all-retryable",
        ),
        pytest.param(
            [_failed_tier("final_binding_failed", "contract_error")],
            "partial_failed",
            id="contract-error",
        ),
        pytest.param(
            [_failed_tier("slice_failed", "permanent_failure")],
            "partial_failed",
            id="permanent",
        ),
        pytest.param(
            [_failed_tier("budget_exhausted", "budget_exhausted")],
            "partial_failed",
            id="budget-class-without-the-budget-flag",
        ),
        pytest.param(
            [
                _failed_tier("breach_inconclusive", "inconclusive"),
                _failed_tier("slice_failed", "permanent_failure"),
            ],
            "partial_failed",
            id="inconclusive-and-permanent",
        ),
        pytest.param(
            [_failed_tier("tier_exception", None)],
            "partial_failed",
            id="no-typed-class",
        ),
    ],
)
def test_unjudged_tier_failure_is_not_collapsed_to_permanent_partial(
    failed_tiers: list[dict],
    expected: str,
):
    """A walk stopped only by tiers nobody ruled on is not a judged failure.

    One judged, contract-error or untyped tier keeps ``partial_failed``.
    """

    assert _terminal_status(
        failed_tiers=failed_tiers,
        accounting_incomplete=False,
        budget_exhausted=False,
        stopped_early=False,
    ) == expected


@pytest.mark.parametrize(
    ("accounting_incomplete", "budget_exhausted", "expected"),
    [
        (True, False, "budget_accounting_incomplete"),
        (True, True, "budget_accounting_incomplete"),
        (False, True, "budget_exhausted"),
    ],
)
def test_accounting_and_budget_precede_an_unjudged_tier_failure(
    accounting_incomplete: bool,
    budget_exhausted: bool,
    expected: str,
):
    assert _terminal_status(
        failed_tiers=[_failed_tier("slice_inconclusive", "inconclusive")],
        accounting_incomplete=accounting_incomplete,
        budget_exhausted=budget_exhausted,
        stopped_early=False,
    ) == expected


def _three_tier_dag() -> TierDAG:
    tiers = []
    for index in range(3):
        slice_id = f"s{index}"
        tiers.append(Tier(
            tier_id=f"t{index}",
            depends_on=(() if index == 0 else (f"t{index - 1}",)),
            slices=(Slice(
                id=slice_id,
                directive=ImplementationDirective(
                    slice_id=slice_id,
                    intent=f"implement {slice_id}",
                    files_in_scope=(f"{slice_id}.py",),
                    creates_paths=(),
                    verify_mode="skip",
                    blast_radius={},
                ),
            ),),
        ))
    return TierDAG(tiers=tuple(tiers))


def _install_architect_stubs(
    monkeypatch,
    *,
    dag: TierDAG,
    plan_cost: float,
    tier_costs: list[float],
):
    calls: list[dict] = []

    class _Pool:
        def get(self, *args, **kwargs):
            return object()

    @contextmanager
    def _pool(*args, **kwargs):
        yield _Pool()

    def _run_plan(spec, **kwargs):
        planned = spec.with_tier_dag(dag)
        return PlanOutcome(
            spec=planned,
            candidate_dag=dag,
            candidate_ref=None,
            decision="accepted",
            status="accepted",
            rounds_used=1,
            final_critic_decision="pass",
            final_judge_decision="pass",
            reason="test",
            cost_usd=plan_cost,
        )

    def _run_tier(**kwargs):
        calls.append(kwargs)
        cost = tier_costs[len(calls) - 1]
        corpus = Corpus(kwargs["spec"])
        current = corpus.latest_spec()
        impl = _make_impl(
            corpus,
            slice_id=kwargs["tier_id"],
            cost=cost,
        )
        corpus.put_record(impl, kind="implementations")
        corpus.put_spec(current.with_implementation(impl))
        with open_run_with_capture(corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": {
                    "total_usd": cost,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status="complete", cost_usd=cost))
        return 0

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(arch, "run_plan", _run_plan)
    monkeypatch.setattr(arch, "run_tier", _run_tier)
    return calls


def test_final_planned_dag_rejects_unknown_slice_runner_overlay_before_tier(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(
        tmp_path,
        runner_parameters={"per_slice": {"unknown": {}}},
    )
    calls = _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def reject_plan_binding(*args, **kwargs):
        pytest.fail("unsafe final runner routing must not bind a plan")

    monkeypatch.setattr(arch, "bind_plan_contract", reject_plan_binding)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        skip_validate=True,
    )

    assert rc == 2
    assert calls == []
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.tier_dag is None
    summary = _architect_summary(corpus)
    assert summary["status"] == "setup_error"
    assert summary["failure_kind"] == "unsafe_runner_parameters"
    assert summary["tiers_executed"] == 0


def _install_protected_tail_architect_stubs(
    monkeypatch,
    *,
    fail_tier: str | None = None,
):
    """Install a two-tier cascade that spends both mandatory review roles."""

    dag = TierDAG(tiers=_three_tier_dag().tiers[:2])
    policy = DevelopmentBudgetPolicy(
        task_budget_usd=1.0,
        planning_review_cap_usd=0.4,
        implementation_verifier_reserve_usd=0.5,
        contingency_usd=0.1,
        roles=tuple(
            RoleAllocation(
                role=role,
                group="implementation_verifier",
                max_calls=2,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=0.1,
                protected_tail_usd=0.05,
            )
            for role in ("diff_scope", "tier_completeness")
        ),
    )
    state = {
        "tier_calls": [],
        "review_calls": [],
        "ledgers": [],
    }

    class _RecordingLedger(DevelopmentBudgetLedger):
        def __init__(self, budget_policy):
            super().__init__(budget_policy)
            self.completion_snapshots = []
            state["ledgers"].append(self)

        def complete_protected_tail_role(self, role: str) -> None:
            calls = self.to_summary()["autonomous_development_metrics"][
                "calls_by_role"
            ]
            self.completion_snapshots.append({
                "role": role,
                "tier_calls": tuple(state["tier_calls"]),
                "calls_by_role": dict(calls),
            })
            super().complete_protected_tail_role(role)

    class _ReviewBackend:
        name = "review-stub"
        model = "stub-model"
        features = SimpleNamespace(usd_limit="post_hoc")

        def __init__(self, tier_id: str, role: str) -> None:
            self.tier_id = tier_id
            self.role = role

        def invoke(self, invocation):
            state["review_calls"].append((
                self.tier_id,
                self.role,
                invocation.budget_usd,
            ))
            return SimpleNamespace(
                usage=Usage(
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.01,
                ),
                completeness=_COMPLETE_EXECUTION,
            )

    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[],
    )
    monkeypatch.setattr(arch, "DevelopmentBudgetLedger", _RecordingLedger)

    def _run_reviewing_tier(**kwargs):
        tier_id = kwargs["tier_id"]
        state["tier_calls"].append(tier_id)
        ledger = kwargs["development_budget_ledger"]
        assert ledger is state["ledgers"][0]
        for role in ("diff_scope", "tier_completeness"):
            result = ledger.wrap(
                role,
                _ReviewBackend(tier_id, role),
            ).invoke(Invocation(
                system=role,
                user=tier_id,
                tools=None,
                max_turns=1,
                budget_usd=0.05,
            ))
            assert result.usage.cost_usd == pytest.approx(0.01)

        corpus = Corpus(kwargs["spec"])
        current = corpus.latest_spec()
        implementation = _make_impl(
            corpus,
            slice_id=tier_id,
            cost=0.02,
            input_tokens=2,
            output_tokens=2,
        )
        corpus.put_record(implementation, kind="implementations")
        corpus.put_spec(current.with_implementation(implementation))
        failed = tier_id == fail_tier
        child_status = "review_failed" if failed else "complete"
        with open_run_with_capture(corpus, "tier") as child:
            child.write_summary({
                "tier_id": tier_id,
                "status": child_status,
                "costs": {
                    "total_usd": 0.02,
                    "total_input_tokens": 2,
                    "total_output_tokens": 2,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status=child_status, cost_usd=0.02))
        return 3 if failed else 0

    monkeypatch.setattr(arch, "run_tier", _run_reviewing_tier)
    return policy, state


def test_protected_review_tail_releases_only_after_complete_two_tier_dag(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    policy, state = _install_protected_tail_architect_stubs(monkeypatch)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        skip_validate=True,
        development_budget_policy=policy,
    )

    assert rc == 0
    assert state["tier_calls"] == ["t0", "t1"]
    assert [
        (tier_id, role)
        for tier_id, role, _budget in state["review_calls"]
    ] == [
        ("t0", "diff_scope"),
        ("t0", "tier_completeness"),
        ("t1", "diff_scope"),
        ("t1", "tier_completeness"),
    ]
    ledger = state["ledgers"][0]
    assert ledger.completion_snapshots == [
        {
            "role": "diff_scope",
            "tier_calls": ("t0", "t1"),
            "calls_by_role": {
                "diff_scope": 2,
                "tier_completeness": 2,
            },
        },
        {
            "role": "tier_completeness",
            "tier_calls": ("t0", "t1"),
            "calls_by_role": {
                "diff_scope": 2,
                "tier_completeness": 2,
            },
        },
    ]
    summary = _architect_summary(corpus)
    assert summary["status"] == "complete"
    tail = summary["development_budget"][
        "implementation_verifier_authority"
    ]["protected_tail"]
    assert tail["outstanding_usd"] == 0.0
    for owner in tail["owners"].values():
        assert owner["terminal_completed"] is True
        assert owner["released_unused_usd"] == 0.03


@pytest.mark.parametrize(
    ("stop_after_tier", "fail_tier", "expected_status", "expected_calls"),
    [
        (0, None, "partial_stopped", 1),
        (None, "t1", "partial_failed", 2),
    ],
)
def test_incomplete_architect_walk_keeps_review_tail_protected(
    tmp_path: Path,
    monkeypatch,
    stop_after_tier: int | None,
    fail_tier: str | None,
    expected_status: str,
    expected_calls: int,
):
    corpus = _make_corpus(tmp_path)
    policy, state = _install_protected_tail_architect_stubs(
        monkeypatch,
        fail_tier=fail_tier,
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        stop_after_tier=stop_after_tier,
        skip_validate=True,
        development_budget_policy=policy,
    )

    assert rc == (1 if fail_tier is not None else 0)
    assert state["tier_calls"] == ["t0", "t1"][:expected_calls]
    assert len(state["review_calls"]) == expected_calls * 2
    ledger = state["ledgers"][0]
    assert ledger.completion_snapshots == []
    summary = _architect_summary(corpus)
    assert summary["status"] == expected_status
    tail = summary["development_budget"][
        "implementation_verifier_authority"
    ]["protected_tail"]
    assert tail["outstanding_usd"] == pytest.approx(
        0.08 if expected_calls == 1 else 0.06
    )
    for owner in tail["owners"].values():
        assert owner["terminal_completed"] is False
        assert owner["released_unused_usd"] == 0.0


def test_architect_forwards_build_retries_to_every_tier(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    calls = _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[0.0, 0.0, 0.0],
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        skip_validate=True,
        build_retries=0,
    )

    assert rc == 0
    assert [call["build_retries"] for call in calls] == [0, 0, 0]


def _architect_summary(corpus: Corpus) -> dict:
    summaries = [
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/summary.json")
        if "_architect_" in path.parent.name
    ]
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def _architect_result(corpus: Corpus) -> dict:
    results = [
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/result.json")
        if "_architect_" in path.parent.name
    ]
    assert len(results) == 1
    return json.loads(results[0].read_text(encoding="utf-8"))


def test_architect_preserves_retryable_tier_terminal_end_to_end(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    calls = _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _transport_failure_tier(**kwargs):
        calls.append(kwargs)
        child_corpus = Corpus(kwargs["spec"])
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "all_slices_transport_failed",
                "token_rollup_incomplete": False,
                "costs": {
                    "total_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(
                status="all_slices_transport_failed",
                cost_usd=0.0,
            ))
        return 1

    monkeypatch.setattr(arch, "run_tier", _transport_failure_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        skip_validate=True,
    )

    assert rc == 1
    assert [call["tier_id"] for call in calls] == ["t0"]
    summary = _architect_summary(corpus)
    assert summary["status"] == "retryable_tier_failure"
    assert summary["tiers_executed"] == 1
    assert summary["tiers_failed"] == 1
    assert summary["tier_results"][0]["status"] == (
        "all_slices_transport_failed"
    )
    assert summary["tier_results"][0]["terminal_class"] == (
        "retryable_failure"
    )
    result = _architect_result(corpus)
    assert result["status"] == "retryable_tier_failure"
    outcome_path = next(
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/outcome.json")
        if "_architect_" in path.parent.name
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["detail_status"] == "retryable_tier_failure"
    assert outcome["terminal_class"] == "retryable_failure"


@pytest.mark.parametrize(
    ("tier_status", "tier_rc", "tier_class", "expected_status", "expected_class"),
    [
        (
            "slice_inconclusive", 1, "inconclusive",
            "inconclusive_tier_failure", "inconclusive",
        ),
        (
            "review_inconclusive", 3, "inconclusive",
            "inconclusive_tier_failure", "inconclusive",
        ),
        (
            "breach_inconclusive", 1, "inconclusive",
            "inconclusive_tier_failure", "inconclusive",
        ),
        ("cancelled", 1, "cancelled", "cancelled", "cancelled"),
        # Guards: a judged tier keeps the permanent bucket.
        (
            "slice_failed", 1, "permanent_failure",
            "partial_failed", "permanent_failure",
        ),
        (
            "review_failed", 3, "permanent_failure",
            "partial_failed", "permanent_failure",
        ),
        (
            "breach_failed", 1, "permanent_failure",
            "partial_failed", "permanent_failure",
        ),
    ],
)
def test_architect_preserves_unjudged_tier_terminal_end_to_end(
    tmp_path: Path,
    monkeypatch,
    tier_status: str,
    tier_rc: int,
    tier_class: str,
    expected_status: str,
    expected_class: str,
):
    """A tier nobody ruled on stops the walk without a permanent label."""

    corpus = _make_corpus(tmp_path)
    calls = _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _unjudged_tier(**kwargs):
        calls.append(kwargs)
        child_corpus = Corpus(kwargs["spec"])
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": tier_status,
                "token_rollup_incomplete": False,
                "costs": {
                    "total_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status=tier_status, cost_usd=0.0))
        return tier_rc

    monkeypatch.setattr(arch, "run_tier", _unjudged_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        skip_validate=True,
    )

    assert rc == 1
    # The walk stops: no later tier gains execution authority.
    assert [call["tier_id"] for call in calls] == ["t0"]
    summary = _architect_summary(corpus)
    assert summary["status"] == expected_status
    assert summary["tiers_executed"] == 1
    assert summary["tiers_failed"] == 1
    assert summary["tier_results"][0]["status"] == tier_status
    assert summary["tier_results"][0]["terminal_class"] == tier_class
    assert summary["costs"]["rollup_incomplete"] is False
    assert _architect_result(corpus)["status"] == expected_status
    outcome_path = next(
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/outcome.json")
        if "_architect_" in path.parent.name
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["detail_status"] == expected_status
    assert outcome["terminal_class"] == expected_class


def test_architect_planner_accounting_debt_stops_all_later_paid_roles(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    calls = {"planner": 0, "critic": 0, "tier": 0}

    class _DebtPlanner:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

        def invoke(self, _invocation):
            calls["planner"] += 1
            return SimpleNamespace(
                final_message=json.dumps({
                    "tiers": [{
                        "tier_id": "t0",
                        "depends_on": [],
                        "slices": [{
                            "id": "diagnostic",
                            "directive": {
                                "slice_id": "diagnostic",
                                "intent": "retain diagnostics only",
                                "requirement_refs": [],
                                "files_in_scope": [],
                                "creates_paths": ["native/diagnostic.json"],
                                "verify_mode": "tests",
                                "blast_radius": {"max_files": 1},
                            },
                        }],
                    }],
                }, separators=(",", ":")),
                usage=Usage(
                    input_tokens=10,
                    output_tokens=4,
                    cost_usd=0.25,
                ),
                error=None,
                completeness=ExecutionCompleteness(
                    terminal_complete=True,
                    trace_complete=True,
                ),
            )

    class _ForbiddenBackend:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

        def invoke(self, _invocation):
            calls["critic"] += 1
            raise AssertionError(
                "critic/judge must not run after planner accounting debt"
            )

    class _Pool:
        def get(self, role, **_kwargs):
            if role == "architect_planner":
                return _DebtPlanner()
            return _ForbiddenBackend()

    @contextmanager
    def _pool(*_args, **_kwargs):
        yield _Pool()

    def _run_plan(spec, **kwargs):
        kwargs["planner_backend"].invoke(None)
        kwargs["critic_backend"].invoke(None)
        raise AssertionError("unreachable")

    def _run_tier(**_kwargs):
        calls["tier"] += 1
        raise AssertionError(
            "Tier must not run after planner accounting debt"
        )

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(arch, "run_plan", _run_plan)
    monkeypatch.setattr(arch, "run_tier", _run_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        skip_validate=True,
    )

    assert rc == 1
    assert calls == {"planner": 1, "critic": 0, "tier": 0}
    summary = _architect_summary(corpus)
    assert summary["status"] == "incomplete_accounting"
    assert summary["decision"] == "inconclusive"
    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == (
        "architect_planner_accounting_incomplete"
    )
    assert summary["authoritative_spent_usd"] == pytest.approx(0.25)
    assert summary["budget_remaining_usd"] == 0.0
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["accounting_debt_by_role"][
        "architect_planner"
    ] == ["cost_complete", "token_usage_complete"]
    assert summary["candidate_ref"].startswith("sha256:")
    assert corpus.get_artifact(summary["candidate_ref"])
    assert corpus.latest_spec().tier_dag is None
    assert _architect_result(corpus)["status"] == "incomplete_accounting"


def test_architect_orchestration_debt_is_role_attributed_and_revokes_budget(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    planner_toolsets = []

    class _Backend:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

    class _Pool:
        def get(self, role, **kwargs):
            if role == "architect_planner" and "toolset" in kwargs:
                planner_toolsets.append(kwargs["toolset"])
            return _Backend()

    @contextmanager
    def _pool(*_args, **_kwargs):
        yield _Pool()

    child = SimpleNamespace(
        role="repository_mapper",
        route_role="architect_planner",
        backend_name="codex",
        model="gpt-test",
        result=SimpleNamespace(
            usage=Usage(
                input_tokens=12,
                output_tokens=5,
                cost_usd=0.4,
            ),
        ),
        token_usage_complete=False,
        cost_complete=False,
        error_kind="IncompleteEvidence",
    )
    fanout = SimpleNamespace(
        children=(child,),
        usage=child.result.usage,
        successful=False,
        status="failed",
        trace=SimpleNamespace(events=()),
        joined_text="",
        to_summary=lambda: {
            "schema_version": "crucible.read-only-fanout-result.v1",
            "status": "failed",
            "child_count": 1,
        },
    )

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    def _run_read_only_fanout(*args, **kwargs):
        del args, kwargs
        assert len(planner_toolsets) == 1
        assert planner_toolsets[0].snapshot().invocations_started == 1
        return fanout

    monkeypatch.setattr(
        arch,
        "run_read_only_fanout",
        _run_read_only_fanout,
    )
    monkeypatch.setattr(
        arch,
        "run_plan",
        lambda *args, **kwargs: pytest.fail(
            "parent planner must not run after child accounting debt"
        ),
    )

    policy = OrchestrationPolicy(
        mode="read_only",
        child_roles=("repository_mapper",),
        max_children=1,
        max_concurrency=1,
    )
    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        enable_source_tools=True,
        planner_tool_call_limit=8,
        planner_tool_result_bytes=4096,
        replan=True,
        skip_validate=True,
        orchestration_policy=policy,
        read_only_tasks=(
            arch.ArchitectReadOnlyTask(
                child_id="map",
                role="repository_mapper",
                instruction="map source",
            ),
        ),
    )

    assert rc == 1
    assert planner_toolsets[0].snapshot().invocations_started == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == pytest.approx(0.4)
    assert summary["budget_remaining_usd"] == 0.0
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt"] is True
    expected_debt = ["cost_complete", "token_usage_complete"]
    assert summary["costs"]["accounting_debt_by_role"][
        "architect_planner"
    ] == expected_debt
    assert summary["costs"]["planning_usage"][
        "accounting_debt_by_role"
    ]["architect_planner"] == expected_debt


@pytest.mark.parametrize("enable_source_tools", [False, True])
def test_architect_orchestration_requires_task_wide_dispatch_ledger_before_pool(
    tmp_path: Path,
    monkeypatch,
    enable_source_tools: bool,
):
    corpus = _make_corpus(tmp_path)
    pool_entered = False

    @contextmanager
    def forbidden_pool(*args, **kwargs):
        nonlocal pool_entered
        pool_entered = True
        pytest.fail("invalid orchestration reached backend pool")
        yield

    monkeypatch.setattr(arch, "backend_pool", forbidden_pool)
    policy = OrchestrationPolicy(
        mode="read_only",
        child_roles=("repository_mapper",),
        max_children=1,
        max_concurrency=1,
    )

    with pytest.raises(
        ValueError,
        match="task-wide hard ledger",
    ):
        arch.run_architect(
            planner_count=1,
            spec=Path(corpus.spec_dir),
            cwd=str(tmp_path),
            budget_usd=1.0,
            enable_source_tools=enable_source_tools,
            replan=True,
            skip_validate=True,
            orchestration_policy=policy,
            read_only_tasks=(
                arch.ArchitectReadOnlyTask(
                    child_id="map",
                    role="repository_mapper",
                    instruction="map source",
                ),
            ),
        )

    assert pool_entered is False


def test_architect_validation_repair_accounting_debt_is_handled_terminal(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    dag = _three_tier_dag()
    planner_calls = 0

    class _Planner:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

        def invoke(self, _invocation):
            nonlocal planner_calls
            planner_calls += 1
            return SimpleNamespace(
                usage=Usage(cost_usd=0.2),
                error=None,
                completeness=ExecutionCompleteness(
                    terminal_complete=True,
                    trace_complete=True,
                ),
            )

    class _TextBackend:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

    class _Pool:
        def get(self, role, **_kwargs):
            return _Planner() if role == "architect_planner" else _TextBackend()

    @contextmanager
    def _pool(*_args, **_kwargs):
        yield _Pool()

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(
        arch,
        "run_plan",
        lambda spec, **_kwargs: PlanOutcome(
            spec=spec.with_tier_dag(dag),
            candidate_dag=dag,
            candidate_ref=None,
            decision="accepted",
            status="accepted",
            rounds_used=1,
            final_critic_decision="pass",
            final_judge_decision="pass",
            reason="candidate",
            cost_usd=0.0,
        ),
    )

    def _validation_gate(**kwargs):
        kwargs["planner_backend"].invoke(None)
        raise AssertionError("unreachable")

    monkeypatch.setattr(arch, "_validation_gate", _validation_gate)
    monkeypatch.setattr(
        arch,
        "run_tier",
        lambda **_kwargs: pytest.fail(
            "Tier must not run after validation accounting debt"
        ),
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
    )

    assert rc == 1
    assert planner_calls == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "incomplete_accounting"
    assert summary["failure_phase"] == "architect_validation"
    assert summary["budget_remaining_usd"] == 0.0
    assert summary["costs"]["remaining_usd"] == 0.0
    assert _architect_result(corpus)["status"] == "incomplete_accounting"


def test_architect_skips_exact_lifecycle_landed_tier_before_backend_dispatch(
    tmp_path: Path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Architect Test"],
        check=True,
    )
    state_path = repo / "s0.py"
    state_path.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "s0.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "base"],
        check=True,
    )
    corpus = _make_corpus(tmp_path)
    base_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corpus.commit(transform=lambda candidate: replace(
        candidate,
        project_context=replace(
            candidate.project_context,
            base_commit=base_commit,
        ),
    ))
    dag = _three_tier_dag()
    planned = corpus.latest_spec().with_tier_dag(dag)
    corpus.put_spec(planned)
    tier = dag.topo_order()[0]
    slice_ = tier.slices[0]
    accepted_bytes = b"accepted tier state\n"
    state_path.write_bytes(accepted_bytes)
    accepted_diff, accepted_files, accepted_base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("s0.py",),
    )
    implementation = replace(
        _make_impl(
            corpus,
            slice_id=slice_.id,
            cost=0.0,
        ),
        base_commit=accepted_base,
        diff_ref=corpus.put_artifact(
            accepted_diff,
            kind="diffs",
            name_hint=slice_.id,
        ),
        files_modified=accepted_files,
    )
    implementation = _partition.attach_resume_identity(
        corpus,
        planned,
        tier,
        slice_,
        implementation,
    )
    corpus.commit(implementations=(implementation,))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=f"tier {tier.tier_id}: complete",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )
    journal = _mutation_journal.WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "crashed_rerun" / "phase1_mutations"
        ),
        slice_id=slice_.id,
        scope=("s0.py",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    state_path.write_bytes(b"crashed unaccepted dirt\n")
    _mutation_journal._unregister_live(journal.transaction_id)
    calls = _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[0.0, 0.0, 0.0],
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(repo),
        stop_after_tier=0,
        budget_usd=0.0,
        replan=True,
        skip_validate=True,
    )

    assert rc == 0
    assert calls == []  # run_tier/backend cascade was never entered
    assert state_path.read_bytes() == accepted_bytes
    assert not journal.journal_dir.exists()
    summary = _architect_summary(corpus)
    assert summary["tier_results"] == [{
        "tier_id": "t0",
        "operation_id": "tier-0",
        "slice_count": 1,
        "tier_rc": 0,
        "status": "already_landed",
        "terminal_class": "success",
        "run_dir": None,
        "skipped": True,
    }]

    # A lifecycle record alone cannot authorize the fast path: even a
    # one-byte drift in the declared slice scope must dispatch the tier.
    state_path.write_bytes(accepted_bytes + b"x")
    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(repo),
        stop_after_tier=0,
        budget_usd=1.0,
        replan=True,
        skip_validate=True,
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["tier_id"] == "t0"


@pytest.mark.parametrize(
    ("discovery_result", "warning_fragment"),
    [
        ("remaining", "active mutation journals remain"),
        ("invalid", "completed-tier skip settlement failed"),
    ],
)
def test_architect_completed_tier_skip_fails_closed_on_central_inventory(
    tmp_path: Path,
    monkeypatch,
    caplog,
    discovery_result: str,
    warning_fragment: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    corpus = _make_corpus(tmp_path)
    spec = corpus.latest_spec()
    assert spec is not None
    tier = _three_tier_dag().topo_order()[0]
    monkeypatch.setattr(
        _mutation_journal,
        "recover_mutation_journals",
        lambda *_args, **_kwargs: _mutation_journal.RecoveryReport((), (), ()),
    )

    if discovery_result == "remaining":
        remaining = SimpleNamespace(journal_dir=tmp_path / "active-journal")
        inventories = iter(((), (remaining,)))
        monkeypatch.setattr(
            _mutation_journal,
            "discover_mutation_journals",
            lambda *_args, **_kwargs: next(inventories),
        )
    else:
        def reject_inventory(*_args, **_kwargs):
            raise _mutation_journal.MutationJournalError(
                "unsafe central journal inventory"
            )

        monkeypatch.setattr(
            _mutation_journal,
            "discover_mutation_journals",
            reject_inventory,
        )

    monkeypatch.setattr(
        arch._workspace_evidence,
        "reconstruct_plan_workspace_evidence",
        lambda **_kwargs: pytest.fail(
            "retained-prefix evidence must not run after inventory rejection"
        ),
    )

    assert arch._completed_tier_skip_is_safe(
        corpus=corpus,
        spec=spec,
        tier=tier,
        cwd=str(repo),
    ) is False
    assert warning_fragment in caplog.text


def test_architect_checkpoint_hook_observes_tiers_and_publishes_only_final(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    dag = _three_tier_dag()
    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[0.0, 0.0, 0.0],
    )
    checkpoints: list[arch.ArchitectCheckpoint] = []

    def _checkpoint(checkpoint: arch.ArchitectCheckpoint):
        checkpoints.append(checkpoint)
        if checkpoint.kind == "tier":
            return None
        return {
            "example.product": {
                "schema_version": "example.checkpoint.v1",
                "observed_operations": [
                    item.operation_id for item in checkpoints
                ],
            },
        }

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        stop_after_tier=0,
        skip_validate=True,
        checkpoint_hook=_checkpoint,
    )

    assert rc == 0
    summary = _architect_summary(corpus)
    assert "checkpoint_extensions" not in summary["tier_results"][0]
    assert summary["checkpoint_extensions"] == {
        "example.product": {
            "schema_version": "example.checkpoint.v1",
            "observed_operations": ["tier-0", "architect-final"],
        },
    }
    assert [item.operation_id for item in checkpoints] == [
        "tier-0",
        "architect-final",
    ]
    assert [item.kind for item in checkpoints] == ["tier", "final"]
    assert (
        checkpoints[0].child_run_dir
        == summary["tier_results"][0]["run_dir"]
    )
    assert checkpoints[0].tiers_executed == 1
    assert checkpoints[1].child_run_dir is None
    assert checkpoints[1].tiers_executed == 1


def test_architect_treats_deepswe_metadata_as_opaque_without_hook(
    tmp_path: Path,
    monkeypatch,
):
    """Recognizable runner metadata must not reactivate product behavior."""

    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[0.0, 0.0, 0.0],
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        stop_after_tier=0,
        skip_validate=True,
        extra_parameters={
            "deepswe_metadata": {
                "task_id": "task",
                "ext_id": "ext",
                "base_commit": "a" * 40,
                "max_patch_bytes": 1024,
            },
        },
    )

    assert rc == 0
    summary = _architect_summary(corpus)
    assert "checkpoint_extensions" not in summary
    assert "mutation_journal_settlement" not in summary
    assert all(
        "mutation_journal_settlement" not in tier
        for tier in summary["tier_results"]
    )


def test_architect_tier_checkpoint_return_is_recorded_as_terminal_error(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[0.0, 0.0, 0.0],
    )

    def _checkpoint(checkpoint: arch.ArchitectCheckpoint):
        if checkpoint.kind == "tier":
            return {"example.product": {"invalid": "tier payload"}}
        return {"example.product": {"final": True}}

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        stop_after_tier=0,
        skip_validate=True,
        checkpoint_hook=_checkpoint,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "checkpoint_failed"
    extensions = summary["checkpoint_extensions"]
    assert extensions["example.product"] == {"final": True}
    assert extensions["checkpoint_errors"] == [{
        "kind": "tier",
        "operation_id": "tier-0",
        "error_kind": "TypeError",
        "error_message": (
            "architect checkpoint hook must return None for tier checkpoints"
        ),
    }]


def test_architect_setup_terminal_finalizes_checkpoint_once(tmp_path: Path):
    spec_dir = tmp_path / "missing-corpus"
    checkpoints: list[arch.ArchitectCheckpoint] = []

    def _checkpoint(checkpoint: arch.ArchitectCheckpoint):
        checkpoints.append(checkpoint)
        return {"example.product": {"terminal": True}}

    rc = arch.run_architect(
        planner_count=1,
        spec=spec_dir,
        cwd=str(tmp_path),
        checkpoint_hook=_checkpoint,
    )

    assert rc == 2
    summary = _architect_summary(Corpus(spec_dir))
    assert summary["status"] == "setup_error"
    assert summary["tiers_executed"] == 0
    assert summary["tiers_failed"] == 0
    assert summary["tier_results"] == []
    assert summary["checkpoint_extensions"] == {
        "example.product": {"terminal": True},
    }
    assert len(checkpoints) == 1
    assert checkpoints[0].kind == "final"


def test_run_architect_passes_only_remaining_budget_to_tiers(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    dag = _three_tier_dag()
    calls = _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=3.0,
        tier_costs=[2.0, 3.0],
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=8.0,
        skip_validate=True,
    )

    # Plan consumes $3, tier 0 gets $5 and consumes $2, tier 1 gets only
    # the remaining $3. Tier 2 is never dispatched after the cap is spent.
    assert rc == 1
    assert [call["budget_usd"] for call in calls] == [5.0, 3.0]
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_exhausted"
    assert summary["authoritative_spent_usd"] == pytest.approx(8.0)
    assert summary["budget_remaining_usd"] == 0.0
    assert summary["tiers_executed"] == 2


def test_run_architect_stop_after_tier_is_partial_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    calls = _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=3.0,
        tier_costs=[2.0],
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=10.0,
        stop_after_tier=0,
        skip_validate=True,
    )

    assert rc == 0
    assert [call["budget_usd"] for call in calls] == [7.0]
    summary = _architect_summary(corpus)
    assert summary["status"] == "partial_stopped"
    assert summary["stopped_early"] is True
    assert summary["tiers_executed"] == 1
    assert summary["tiers_failed"] == 0
    assert summary["authoritative_spent_usd"] == pytest.approx(5.0)
    assert summary["budget_remaining_usd"] == pytest.approx(5.0)


def test_architect_child_summary_retains_discarded_retry_tokens(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _token_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        final_impl = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=0.0,
            input_tokens=100,
            output_tokens=20,
        )
        child_corpus.put_record(final_impl, kind="implementations")
        child_corpus.put_spec(current.with_implementation(final_impl))
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": {
                    "total_usd": 0.0,
                    # Includes 200/40 tokens from retries that were discarded
                    # before the final 100/20-token Implementation record.
                    "total_input_tokens": 300,
                    "total_output_tokens": 60,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status="complete", cost_usd=0.0))
        return 0

    monkeypatch.setattr(arch, "run_tier", _token_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        stop_after_tier=0,
        skip_validate=True,
    )

    assert rc == 0
    summary = _architect_summary(corpus)
    costs = summary["costs"]
    assert costs["total_input_tokens"] == 300
    assert costs["total_output_tokens"] == 60
    assert costs["record_rollup_input_tokens"] == 100
    assert costs["record_rollup_output_tokens"] == 20
    assert costs["by_role"]["implementer"]["input_tokens"] == 100
    tier_result = summary["tier_results"][0]
    assert tier_result["input_tokens"] == 300
    assert tier_result["output_tokens"] == 60
    assert tier_result["record_input_tokens"] == 100
    assert tier_result["record_output_tokens"] == 20
    assert tier_result["token_source"] == "child_summary"
    assert tier_result["token_record_delta"] == {
        "input_tokens": 200,
        "output_tokens": 40,
    }
    assert tier_result["token_rollup_incomplete"] is False


@pytest.mark.parametrize(
    ("tamper", "expected_rc", "expected_status", "expected_error_kind"),
    [
        (None, 0, "partial_stopped", None),
        (
            "cache_total",
            1,
            "budget_accounting_incomplete",
            "ChildPricingEvidenceMismatch",
        ),
        (
            "outcome_cost",
            1,
            "budget_accounting_incomplete",
            "ChildCostAuthorityMismatch",
        ),
    ],
)
def test_architect_preserves_and_reconciles_child_tier_pricing_ledger(
    tmp_path: Path,
    monkeypatch,
    tamper: str | None,
    expected_rc: int,
    expected_status: str,
    expected_error_kind: str | None,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )
    evidence = _exact_pricing_evidence(
        input_tokens=10,
        output_tokens=5,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        cost_usd=0.1,
    )

    def _priced_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=0.1,
            input_tokens=10,
            output_tokens=5,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        child_costs = CostAccumulator()
        child_costs.record(
            role="implementer",
            usage=evidence.usage,
            backend_name="codex",
            model="gpt-5.5",
            pricing_evidence=evidence,
        )
        cost_summary = child_costs.to_dict()
        if tamper == "cache_total":
            cost_summary["total_cache_read_tokens"] = 4
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": cost_summary,
            })
            child.finalize(RunResult(
                status="complete",
                cost_usd=(0.1000004 if tamper == "outcome_cost" else 0.1),
            ))
        return 0

    monkeypatch.setattr(arch, "run_tier", _priced_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        stop_after_tier=0,
        skip_validate=True,
    )

    assert rc == expected_rc
    summary = _architect_summary(corpus)
    assert summary["status"] == expected_status
    costs = summary["costs"]
    assert costs["total_cache_creation_tokens"] == 2
    assert costs["pricing_evidence"]["complete"] is (tamper is None)
    assert costs["pricing_rollup_incomplete"] is (tamper is not None)
    assert costs["pricing_evidence_by_call"][0]["role"] == "implementer"
    if expected_error_kind is not None:
        assert summary["tier_results"][0][
            "cost_reconciliation_error_kind"
        ] == expected_error_kind


@pytest.mark.parametrize(
    ("tamper", "expected_error_kind", "expected_total", "known_calls"),
    [
        (None, None, 0.1, 1),
        (
            "partial_cache_total",
            "ChildPricingEvidenceMismatch",
            0.1,
            1,
        ),
        (
            "outer_cost",
            "ChildCostAuthorityMismatch",
            0.2,
            1,
        ),
        (
            "malformed_top_level_costs",
            "ChildCostSummaryMalformed",
            0.1,
            0,
        ),
    ],
)
def test_architect_reconciles_exact_partial_pricing_but_preserves_outer_debt(
    tmp_path: Path,
    monkeypatch,
    tamper: str | None,
    expected_error_kind: str | None,
    expected_total: float,
    known_calls: int,
):
    """R7-shaped cleanup terminals retain calls but never restore authority."""

    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )
    evidence = _exact_pricing_evidence(
        input_tokens=10,
        output_tokens=5,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        cost_usd=0.1,
    )
    tier_calls: list[str] = []

    class ProviderSettlementCleanupError(RuntimeError):
        pass

    def _cleanup_tier(**kwargs):
        tier_calls.append(kwargs["tier_id"])
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = replace(
            _make_impl(
                child_corpus,
                slice_id=kwargs["tier_id"],
                cost=0.1,
                input_tokens=10,
                output_tokens=5,
            ),
            execution_completeness=_COMPLETE_EXECUTION,
            pricing_evidence=evidence,
            cache_creation_tokens=2,
            cache_read_tokens=3,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        child_costs = CostAccumulator()
        child_costs.record(
            role="implementer",
            usage=evidence.usage,
            backend_name="codex",
            model="gpt-5.5",
            pricing_evidence=evidence,
        )
        partial_costs = child_costs.to_dict()
        if tamper == "partial_cache_total":
            partial_costs["total_cache_read_tokens"] = 4

        try:
            with open_run_with_capture(
                child_corpus,
                "tier",
                failure_cost_usd=lambda: 0.1,
                failure_terminal=lambda _exc: FailureTerminal(
                    status="incomplete_accounting",
                    summary_fields={
                        "cost_usd_is_lower_bound": True,
                        "accounting_debt": True,
                        "accounting_debt_by_gate": {
                            "tier_cleanup": [
                                "cost_complete",
                                "token_usage_complete",
                            ],
                        },
                        "accounting_debt_by_role": {},
                        "cleanup_pending": True,
                        "provider_cleanup_pending": True,
                        "mutation_cleanup_pending": True,
                        "journal_cleanup_pending": True,
                        "workspace_cleanup_pending": True,
                    },
                ),
            ) as child:
                child.write_summary({
                    "tier_id": kwargs["tier_id"],
                    "status": "incomplete",
                    "token_rollup_incomplete": False,
                    "costs": partial_costs,
                })
                raise ProviderSettlementCleanupError(
                    "landed settlement retained a terminal owner"
                )
        except ProviderSettlementCleanupError:
            summary_paths = list(
                (Path(child_corpus.spec_dir) / "runs").glob(
                    "*_tier_*/summary.json"
                )
            )
            assert len(summary_paths) == 1
            if tamper in {"outer_cost", "malformed_top_level_costs"}:
                failure_summary = json.loads(
                    summary_paths[0].read_text(encoding="utf-8")
                )
                if tamper == "outer_cost":
                    # The immutable typed outcome remains $0.10.
                    failure_summary["cost_usd"] = 0.2
                else:
                    # Presence advertises top-level authority. It may not
                    # downgrade to the otherwise-valid partial ledger.
                    failure_summary["costs"] = None
                summary_paths[0].write_text(
                    json.dumps(failure_summary),
                    encoding="utf-8",
                )
            raise

    monkeypatch.setattr(arch, "run_tier", _cleanup_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        skip_validate=True,
    )

    assert rc == 1
    assert tier_calls == ["t0"]
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == pytest.approx(
        expected_total
    )
    assert summary["budget_remaining_usd"] == 0.0
    costs = summary["costs"]
    assert costs["rollup_incomplete"] is True
    assert costs["accounting_debt"] is True
    assert len(costs["pricing_evidence_by_call"]) == known_calls
    assert costs["pricing_rollup_incomplete"] is (
        expected_error_kind is not None
    )
    assert costs["pricing_evidence"]["complete"] is (
        expected_error_kind is None
    )
    tier_result = summary["tier_results"][0]
    assert tier_result["cost_rollup_incomplete"] is True
    assert tier_result["budget_remaining_usd"] == 0.0
    assert tier_result["cost_usd_is_lower_bound"] is True
    assert tier_result["accounting_debt"] is True
    assert tier_result["provider_cleanup_pending"] is True
    assert tier_result["mutation_cleanup_pending"] is True
    assert tier_result["journal_cleanup_pending"] is True
    if tamper == "malformed_top_level_costs":
        assert "cost_authority_source" not in tier_result
    else:
        assert tier_result["cost_authority_source"] == "partial_summary"
    if expected_error_kind is None:
        assert tier_result["input_tokens"] == 10
        assert tier_result["output_tokens"] == 5
        assert tier_result["token_rollup_incomplete"] is False
        assert tier_result["pricing_evidence_complete"] is True
        assert "cost_reconciliation_error_kind" not in tier_result
    else:
        assert tier_result["cost_reconciliation_error_kind"] == (
            expected_error_kind
        )


@pytest.mark.parametrize(
    ("child_status", "terminal_class"),
    [("complete", "succeeded"), ("no_work", "no_work")],
)
def test_architect_rejects_partial_costs_from_crafted_success_terminal(
    tmp_path: Path,
    monkeypatch,
    child_status: str,
    terminal_class: str,
):
    """A boundary error cannot turn success/no-work into an exception."""

    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )
    evidence = _exact_pricing_evidence(
        input_tokens=10,
        output_tokens=5,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        cost_usd=0.1,
    )
    tier_calls: list[str] = []
    error_message = "crafted exception after a successful child terminal"

    def _crafted_success_tier(**kwargs):
        tier_calls.append(kwargs["tier_id"])
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = replace(
            _make_impl(
                child_corpus,
                slice_id=kwargs["tier_id"],
                cost=0.1,
                input_tokens=10,
                output_tokens=5,
            ),
            execution_completeness=_COMPLETE_EXECUTION,
            pricing_evidence=evidence,
            cache_creation_tokens=2,
            cache_read_tokens=3,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        child_costs = CostAccumulator()
        child_costs.record(
            role="implementer",
            usage=evidence.usage,
            backend_name="codex",
            model="gpt-5.5",
            pricing_evidence=evidence,
        )
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "workflow": "tier",
                "status": child_status,
                "cost_usd": 0.1,
                "error_kind": "RuntimeError",
                "error_message": error_message,
                "partial_summary": {
                    "status": "incomplete",
                    "costs": child_costs.to_dict(),
                },
            })
            child.finalize(RunResult(
                status=child_status,
                cost_usd=0.1,
                error_kind="RuntimeError",
                error_message=error_message,
            ))
        raise RuntimeError(error_message)

    monkeypatch.setattr(arch, "run_tier", _crafted_success_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=1.0,
        skip_validate=True,
    )

    assert rc == 1
    assert tier_calls == ["t0"]
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == pytest.approx(0.1)
    costs = summary["costs"]
    assert costs["rollup_incomplete"] is True
    assert costs["accounting_debt"] is True
    assert costs["pricing_rollup_incomplete"] is True
    assert costs["pricing_evidence_by_call"] == []
    tier_result = summary["tier_results"][0]
    assert tier_result["terminal_class"] == terminal_class
    assert tier_result["cost_authority_source"] == "partial_summary"
    assert tier_result["cost_rollup_incomplete"] is True
    assert tier_result["cost_reconciliation_error_kind"] == (
        "ChildCostSummaryMalformed"
    )
    assert tier_result["cost_reconciliation_error_message"] == (
        "success/no-work typed child terminal cannot supply partial costs"
    )


def test_architect_trusts_handled_zero_cost_tier_setup_failure(
    tmp_path: Path,
    monkeypatch,
):
    """A pre-invocation child failure must not poison budget accounting."""

    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _setup_failure_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "setup_error",
                "error_kind": "MutationJournalError",
                "error_message": "unsupported checkout transform",
                "token_rollup_incomplete": False,
                "costs": {
                    "total_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(
                status="setup_error",
                cost_usd=0.0,
                error_kind="MutationJournalError",
                error_message="unsupported checkout transform",
            ))
        return 2

    monkeypatch.setattr(arch, "run_tier", _setup_failure_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=5.0,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "partial_failed"
    assert summary["authoritative_spent_usd"] == 0.0
    assert summary["costs"]["rollup_incomplete"] is False
    assert summary["costs"]["token_rollup_incomplete"] is False
    tier_result = summary["tier_results"][0]
    assert tier_result["status"] == "setup_error"
    assert tier_result["tier_rc"] == 2
    assert tier_result["error_kind"] == "MutationJournalError"
    assert tier_result["error_message"] == "unsupported checkout transform"
    assert tier_result["cost_usd"] == 0.0
    assert tier_result["token_source"] == "child_summary"
    assert tier_result["token_rollup_incomplete"] is False
    assert tier_result["cost_rollup_incomplete"] is False


def test_architect_retains_finalized_journal_error_and_returned_child_usage(
    tmp_path: Path,
    monkeypatch,
):
    """A rollback-boundary exception cannot replace a typed Tier terminal."""

    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _journal_failure_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "slice_failed",
                "failure_phase": "initial_mutation_validation",
                "failure_slice_id": "s0",
                "error_kind": "MutationJournalError",
                "error_message": "post-call repository attestation failed",
                "token_rollup_incomplete": False,
                "costs": {
                    "total_usd": 1.234567,
                    "total_input_tokens": 123,
                    "total_output_tokens": 45,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(
                status="slice_failed",
                cost_usd=1.234567,
                error_kind="MutationJournalError",
                error_message="post-call repository attestation failed",
            ))
        raise RuntimeError("secondary rollback callback failed")

    monkeypatch.setattr(arch, "run_tier", _journal_failure_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=5.0,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "partial_failed"
    assert summary["authoritative_spent_usd"] == pytest.approx(1.234567)
    assert summary["costs"]["total_input_tokens"] == 123
    assert summary["costs"]["total_output_tokens"] == 45
    assert summary["costs"]["rollup_incomplete"] is False
    tier_result = summary["tier_results"][0]
    assert tier_result["status"] == "slice_failed"
    assert tier_result["error_kind"] == "MutationJournalError"
    assert (
        tier_result["error_message"]
        == "post-call repository attestation failed"
    )
    assert tier_result["boundary_exception_kind"] == "RuntimeError"
    assert (
        tier_result["boundary_exception_message"]
        == "secondary rollback callback failed"
    )
    assert tier_result["cost_usd"] == pytest.approx(1.234567)
    assert tier_result["input_tokens"] == 123
    assert tier_result["output_tokens"] == 45
    assert tier_result["token_source"] == "child_summary"
    assert tier_result["token_rollup_incomplete"] is False
    assert tier_result["cost_rollup_incomplete"] is False


def test_architect_uses_raw_typed_child_cost_across_fractional_rollover(
    tmp_path: Path,
    monkeypatch,
):
    """A rounded child summary must not understate task-wide authority."""

    corpus = _make_corpus(tmp_path)
    dag = TierDAG(tiers=_three_tier_dag().tiers[:1])
    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.19014024,
        tier_costs=[],
    )

    def _fractional_tier(**kwargs):
        exact_cost = 0.00834538
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=exact_cost,
            input_tokens=10,
            output_tokens=4,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": {
                    "total_usd": round(exact_cost, 6),
                    "total_input_tokens": 10,
                    "total_output_tokens": 4,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status="complete", cost_usd=exact_cost))
        return 0

    monkeypatch.setattr(arch, "run_tier", _fractional_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 0
    summary = _architect_summary(corpus)
    assert summary["tier_results"][0]["cost_usd"] == 0.008345
    assert summary["authoritative_spent_usd"] == 0.198486
    assert summary["costs"]["total_usd"] == 0.198486
    assert _architect_result(corpus)["cost_usd"] == pytest.approx(0.19848562)


def test_architect_fails_closed_on_typed_child_summary_cost_mismatch(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    dag = TierDAG(tiers=_three_tier_dag().tiers[:1])
    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[],
    )

    def _mismatched_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": {
                    "total_usd": 0.25,
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status="complete", cost_usd=0.5))
        return 0

    monkeypatch.setattr(arch, "run_tier", _mismatched_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    tier = summary["tier_results"][0]
    assert tier["cost_reconciliation_error_kind"] == (
        "ChildCostAuthorityMismatch"
    )
    assert tier["cost_rollup_incomplete"] is True
    assert summary["authoritative_spent_usd"] == 0.5


@pytest.mark.parametrize("invalid_total", ["missing", True, "0.5"])
def test_architect_requires_numeric_child_summary_cost_authority(
    tmp_path: Path,
    monkeypatch,
    invalid_total: object,
):
    corpus = _make_corpus(tmp_path)
    dag = TierDAG(tiers=_three_tier_dag().tiers[:1])
    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[],
    )

    def _invalid_summary_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=0.25,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        with open_run_with_capture(child_corpus, "tier") as child:
            costs = {
                "total_input_tokens": 1,
                "total_output_tokens": 1,
                "rollup_incomplete": False,
                "token_rollup_incomplete": False,
            }
            if invalid_total != "missing":
                costs["total_usd"] = invalid_total
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": costs,
            })
            child.finalize(RunResult(status="complete", cost_usd=0.5))
        return 0

    monkeypatch.setattr(arch, "run_tier", _invalid_summary_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == 0.5
    tier = summary["tier_results"][0]
    assert tier["cost_rollup_incomplete"] is True
    assert tier["cost_reconciliation_error_kind"] == (
        "ChildCostAuthorityMissing"
    )


@pytest.mark.parametrize(
    ("summary_total", "record_cost", "expected_error_kind"),
    [
        (0.5, 0.75, "RecordCostAuthorityMismatch"),
        (None, 0.75, "ChildCostAuthorityMissing"),
        (0.5, 0.50000038, "RecordCostAuthorityMismatch"),
    ],
)
def test_architect_fails_closed_when_record_cost_exceeds_child_authority(
    tmp_path: Path,
    monkeypatch,
    summary_total: float | None,
    record_cost: float,
    expected_error_kind: str,
):
    corpus = _make_corpus(tmp_path)
    dag = TierDAG(tiers=_three_tier_dag().tiers[:1])
    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[],
    )

    def _underreported_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=record_cost,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        with open_run_with_capture(child_corpus, "tier") as child:
            costs = {
                "total_input_tokens": 1,
                "total_output_tokens": 1,
                "rollup_incomplete": False,
                "token_rollup_incomplete": False,
            }
            if summary_total is not None:
                costs["total_usd"] = summary_total
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": costs,
            })
            child.finalize(RunResult(status="complete", cost_usd=0.5))
        return 0

    monkeypatch.setattr(arch, "run_tier", _underreported_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    tier = summary["tier_results"][0]
    assert tier["cost_reconciliation_error_kind"] == expected_error_kind
    assert tier["cost_rollup_incomplete"] is True
    assert summary["authoritative_spent_usd"] == round(record_cost, 6)
    assert _architect_result(corpus)["cost_usd"] == pytest.approx(record_cost)


def test_architect_accepts_typed_child_cost_above_retained_records(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    dag = TierDAG(tiers=_three_tier_dag().tiers[:1])
    _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[],
    )

    def _discarded_call_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=0.5,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": {
                    "total_usd": 0.75,
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status="complete", cost_usd=0.75))
        return 0

    monkeypatch.setattr(arch, "run_tier", _discarded_call_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 0
    summary = _architect_summary(corpus)
    assert summary["status"] == "complete"
    assert summary["authoritative_spent_usd"] == 0.75
    assert summary["tier_results"][0]["cost_rollup_incomplete"] is False


def test_architect_retains_typed_cost_when_child_path_recovery_fails(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    tier_calls: list[str] = []
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _recovered_tier(**kwargs):
        tier_calls.append(kwargs["tier_id"])
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=0.25,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "complete",
                "costs": {
                    "total_usd": 0.75,
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "rollup_incomplete": False,
                    "token_rollup_incomplete": False,
                },
            })
            child.finalize(RunResult(status="complete", cost_usd=0.75))
        return 0

    monkeypatch.setattr(arch, "run_tier", _recovered_tier)
    monkeypatch.setattr(
        arch._outcome,
        "child_run_dir",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("saga-bound child path unavailable")
        ),
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 1
    assert tier_calls == ["t0"]
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == 0.75
    tier = summary["tier_results"][0]
    assert tier["cost_rollup_incomplete"] is True
    assert tier["cost_reconciliation_error_kind"] == (
        "ChildEvidenceBindingError"
    )
    assert _architect_result(corpus)["cost_usd"] == pytest.approx(0.75)


def test_architect_fails_closed_without_typed_child_outcome(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    tier_calls: list[str] = []
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _unbound_tier(**kwargs):
        tier_calls.append(kwargs["tier_id"])
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        implementation = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=0.25,
        )
        child_corpus.put_record(implementation, kind="implementations")
        child_corpus.put_spec(current.with_implementation(implementation))
        return 0

    monkeypatch.setattr(arch, "run_tier", _unbound_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 1
    assert tier_calls == ["t0"]
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == 0.25
    tier = summary["tier_results"][0]
    assert tier["terminal_class"] == "contract_error"
    assert tier["cost_rollup_incomplete"] is True
    assert tier["cost_reconciliation_error_kind"] == (
        "ChildEvidenceBindingError"
    )


def test_run_architect_plan_budget_exhaustion_returns_one(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _budget_outcome(spec, **_kwargs):
        return PlanOutcome(
            spec=spec,
            candidate_dag=None,
            candidate_ref=None,
            decision="inconclusive",
            status="budget_exhausted",
            rounds_used=0,
            final_critic_decision=None,
            final_judge_decision=None,
            reason="budget spent before planning",
            cost_usd=2.0,
        )

    monkeypatch.setattr(arch, "run_plan", _budget_outcome)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=2.0,
        skip_validate=True,
    )

    assert rc == 1
    assert _architect_summary(corpus)["status"] == "budget_exhausted"
    assert _architect_result(corpus)["status"] == "budget_exhausted"


def test_accepted_plan_soft_overshoot_preserves_implementation_reserve(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    dag = _three_tier_dag()
    provider_calls: list[tuple[str, Invocation]] = []
    tier_calls: list[dict] = []

    class _PaidBackend:
        name = "codex"
        model = "gpt-test"
        features = SimpleNamespace(usd_limit="post_hoc")

        def __init__(self, role: str) -> None:
            self.role = role

        def invoke(self, invocation):
            provider_calls.append((self.role, invocation))
            return SimpleNamespace(
                usage=Usage(
                    input_tokens=10,
                    output_tokens=4,
                    cost_usd=6.0,
                ),
                error=None,
                completeness=_COMPLETE_EXECUTION,
            )

    class _Pool:
        def __init__(self, ledger) -> None:
            self.ledger = ledger

        def get(self, role, **_kwargs):
            return self.ledger.wrap(role, _PaidBackend(role))

    @contextmanager
    def _pool(*_args, development_budget_ledger=None, **_kwargs):
        assert development_budget_ledger is not None
        yield _Pool(development_budget_ledger)

    def _accepted_outcome(spec, **kwargs):
        result = kwargs["judge_backend"].invoke(Invocation(
            system="judge",
            user="accept",
            tools=None,
            max_turns=1,
            budget_usd=4.0,
        ))
        assert result.usage.cost_usd == 6.0
        return PlanOutcome(
            spec=spec.with_tier_dag(dag),
            candidate_dag=dag,
            candidate_ref=None,
            decision="accepted",
            status="accepted",
            rounds_used=1,
            final_critic_decision="pass",
            final_judge_decision="pass",
            reason="accepted after soft provider overshoot",
            cost_usd=6.0,
        )

    policy = DevelopmentBudgetPolicy(
        task_budget_usd=10.0,
        planning_review_cap_usd=4.0,
        implementation_verifier_reserve_usd=5.0,
        contingency_usd=1.0,
        roles=tuple(
            RoleAllocation(
                role=role,
                group=(
                    "implementation_verifier"
                    if role == "implementer"
                    else "planning_review"
                ),
                max_calls=1,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=5.0 if role == "implementer" else 4.0,
            )
            for role in (
                "architect_planner",
                "architect_critic",
                "architect_judge",
                "implementer",
            )
        ),
    )

    monkeypatch.setattr(arch, "backend_pool", _pool)
    monkeypatch.setattr(arch, "load_rubric", lambda name: name)
    monkeypatch.setattr(arch, "run_plan", _accepted_outcome)
    monkeypatch.setattr(
        arch,
        "bind_plan_contract",
        lambda *_args, **_kwargs: pytest.fail(
            "reserve-invalid plan reached authorization"
        ),
    )
    monkeypatch.setattr(
        arch,
        "run_tier",
        lambda **kwargs: tier_calls.append(kwargs),
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=10.0,
        skip_validate=True,
        development_budget_policy=policy,
    )

    assert rc == 1
    assert [role for role, _invocation in provider_calls] == [
        "architect_judge"
    ]
    assert provider_calls[0][1].budget_usd == 4.0
    assert tier_calls == []
    assert corpus.latest_spec().tier_dag is None
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_exhausted"
    assert summary["failure_phase"] == "architect_budget_integrity"
    assert summary["failure_kind"] == "implementation_reserve_eroded"
    assert summary["budget_integrity_reason"] == (
        "planning_review_cap_exceeded"
    )
    assert summary["tiers_executed"] == 0
    assert summary["tier_results"] == []
    assert summary["authoritative_spent_usd"] == 6.0
    assert summary["budget_remaining_usd"] == 4.0
    assert summary["costs"]["accounting_debt"] is False
    assert summary["development_budget"]["authority_revoked"] is True
    assert summary["development_budget"]["group_spend_usd"][
        "planning_review"
    ] == 6.0
    assert summary["development_budget"]["group_remaining_usd"][
        "implementation_verifier"
    ] == 6.0
    assert _architect_result(corpus)["status"] == "budget_exhausted"


@pytest.mark.parametrize(
    ("status", "decision", "terminal_class"),
    [
        ("inflation_veto", "rejected", "permanent_failure"),
        ("judge_protocol_failure", "inconclusive", "inconclusive"),
        # The planner backend was lost: nobody ruled on a plan.
        ("planner_infrastructure_failure", "inconclusive", "inconclusive"),
    ],
)
def test_plan_terminal_publishes_zero_tier_inventory_and_final_checkpoint(
    tmp_path: Path,
    monkeypatch,
    status: str,
    decision: str,
    terminal_class: str,
):
    corpus = _make_corpus(tmp_path)
    dag = _three_tier_dag()
    calls = _install_architect_stubs(
        monkeypatch,
        dag=dag,
        plan_cost=0.0,
        tier_costs=[],
    )

    def _terminal_outcome(spec, **_kwargs):
        return PlanOutcome(
            spec=spec.with_tier_dag(dag),
            candidate_dag=dag,
            candidate_ref=None,
            decision=decision,
            status=status,
            rounds_used=2,
            final_critic_decision="revise",
            final_judge_decision=decision,
            reason="planning terminal",
            cost_usd=0.0,
        )

    checkpoints: list[arch.ArchitectCheckpoint] = []

    def _checkpoint(checkpoint: arch.ArchitectCheckpoint):
        checkpoints.append(checkpoint)
        return {"example.product": {"terminal": True}}

    monkeypatch.setattr(arch, "run_plan", _terminal_outcome)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        skip_validate=True,
        checkpoint_hook=_checkpoint,
    )

    assert rc == 2
    assert calls == []
    summary = _architect_summary(corpus)
    assert summary["status"] == status
    assert summary["decision"] == decision
    assert summary["failure_phase"] == "architect_planning"
    assert summary["failure_kind"] == status
    assert summary["tiers_executed"] == 0
    assert summary["tiers_failed"] == 0
    assert summary["tier_results"] == []
    assert summary["checkpoint_extensions"] == {
        "example.product": {"terminal": True},
    }
    assert len(checkpoints) == 1
    assert checkpoints[0].kind == "final"
    assert checkpoints[0].tier_id == "__architect_final__"
    assert checkpoints[0].operation_id == "architect-final"
    assert checkpoints[0].tiers_executed == 0

    outcome_path = next(
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/outcome.json")
        if "_architect_" in path.parent.name
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["detail_status"] == status
    assert outcome["terminal_class"] == terminal_class


def test_final_checkpoint_rejects_core_field_collision(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    checkpoints = arch._ArchitectCheckpointDispatcher(
        lambda _checkpoint: {"status": "shadowed"},
    )

    result = arch._zero_tier_terminal_fields(
        corpus=corpus,
        cwd=str(tmp_path),
        checkpoints=checkpoints,
    )

    assert result["tiers_executed"] == 0
    assert result["tiers_failed"] == 0
    assert result["tier_results"] == []
    assert result["checkpoint_extensions"]["checkpoint_errors"] == [{
        "kind": "final",
        "operation_id": "architect-final",
        "error_kind": "ValueError",
        "error_message": (
            "architect checkpoint hook returned reserved core fields: status"
        ),
    }]


def test_run_architect_validation_budget_exhaustion_returns_one(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=1.0,
        tier_costs=[],
    )

    monkeypatch.setattr(
        arch,
        "_validation_gate",
        lambda **kwargs: (
            kwargs["planned"],
            "budget_exhausted",
            2.0,
        ),
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=3.0,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_exhausted"
    assert summary["authoritative_spent_usd"] == pytest.approx(3.0)
    assert _architect_result(corpus)["status"] == "budget_exhausted"


def test_run_architect_propagates_child_budget_status_at_exact_cap(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )
    pricing_evidence = _exact_pricing_evidence(
        input_tokens=29_919,
        output_tokens=9_741,
        cache_creation_tokens=17_217,
        cache_read_tokens=12_698,
        cost_usd=0.35750025,
    )
    child_cost = pricing_evidence.usage.cost_usd

    def _budget_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        current = child_corpus.latest_spec()
        impl = _make_impl(
            child_corpus,
            slice_id=kwargs["tier_id"],
            cost=child_cost,
            input_tokens=29_919,
            output_tokens=9_741,
        )
        impl = replace(
            impl,
            execution=replace(
                impl.execution,
                backend="claude",
                model="claude-opus-5",
            ),
            execution_completeness=_COMPLETE_EXECUTION,
            pricing_evidence=pricing_evidence,
            cache_creation_tokens=17_217,
            cache_read_tokens=12_698,
        )
        child_corpus.put_record(impl, kind="implementations")
        child_corpus.put_spec(current.with_implementation(impl))
        child_costs = CostAccumulator()
        child_costs.record(
            role="implementer",
            usage=impl,
            backend_name="claude",
            model="claude-opus-5",
        )
        with open_run_with_capture(child_corpus, "tier") as child:
            child.write_summary({
                "tier_id": kwargs["tier_id"],
                "status": "budget_exhausted",
                "costs": child_costs.to_dict(),
            })
            child.finalize(RunResult(
                status="budget_exhausted",
                cost_usd=child_cost,
            ))
        return 1

    monkeypatch.setattr(arch, "run_tier", _budget_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=child_cost,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_exhausted"
    assert summary["authoritative_spent_usd"] == pytest.approx(child_cost)
    costs = summary["costs"]
    assert costs["pricing_evidence"]["complete"] is True
    pricing_calls = costs["pricing_evidence_by_call"]
    assert len(pricing_calls) == 1
    assert pricing_calls[0]["usage"]["cost_usd"] == pytest.approx(
        child_cost
    )
    assert pricing_calls[0]["pricing_evidence"]["complete"] is True
    assert costs["pricing_rollup_incomplete"] is False
    assert costs["accounting_debt"] is False
    assert costs["rollup_incomplete"] is False
    assert costs["unloaded_refs"] == 0
    tier_result = summary["tier_results"][0]
    assert tier_result["status"] == "budget_exhausted"
    assert tier_result["cost_usd"] == pytest.approx(child_cost)
    assert tier_result["pricing_evidence_complete"] is True
    assert tier_result["cost_rollup_incomplete"] is False
    assert "cost_reconciliation_error_kind" not in tier_result
    assert tier_result["run_dir"]
    assert _architect_result(corpus)["status"] == "budget_exhausted"


def test_run_architect_reconciles_and_finalizes_after_child_exception(
    tmp_path: Path,
    monkeypatch,
):
    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=0.0,
        tier_costs=[],
    )

    def _raising_tier(**kwargs):
        child_corpus = Corpus(kwargs["spec"])
        with open_run_with_capture(child_corpus, "tier"):
            current = child_corpus.latest_spec()
            impl = _make_impl(
                child_corpus,
                slice_id=kwargs["tier_id"],
                cost=1.25,
            )
            child_corpus.put_record(impl, kind="implementations")
            child_corpus.put_spec(current.with_implementation(impl))
            raise RuntimeError("tier exploded after spending")

    monkeypatch.setattr(arch, "run_tier", _raising_tier)

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=5.0,
        skip_validate=True,
    )

    assert rc == 1
    summary = _architect_summary(corpus)
    assert summary["status"] == "budget_accounting_incomplete"
    assert summary["authoritative_spent_usd"] == pytest.approx(1.25)
    assert summary["budget_remaining_usd"] == 0.0
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["tiers_executed"] == 1
    tier_result = summary["tier_results"][0]
    assert tier_result["status"] == "failed"
    assert tier_result["tier_rc"] == 1
    assert tier_result["error_kind"] == "RuntimeError"
    assert tier_result["cost_usd"] == pytest.approx(1.25)
    assert tier_result["cost_rollup_incomplete"] is True
    assert tier_result["budget_remaining_usd"] == 0.0
    assert _architect_result(corpus)["status"] == "budget_accounting_incomplete"


def test_run_architect_repair_replan_without_a_ruling_keeps_its_status(
    tmp_path: Path,
    monkeypatch,
):
    """A repair re-plan nobody ruled on is published under its own status.

    The validation gate hands the re-plan's typed terminal to the run: the
    run exits 2, and summary, ``result.json`` and ``outcome.json`` carry
    ``planner_infrastructure_failure`` with the validation phase.
    """

    corpus = _make_corpus(tmp_path)
    _install_architect_stubs(
        monkeypatch,
        dag=_three_tier_dag(),
        plan_cost=1.0,
        tier_costs=[],
    )

    monkeypatch.setattr(
        arch,
        "_validation_gate",
        lambda **kwargs: (
            kwargs["planned"],
            "planner_infrastructure_failure",
            0.5,
        ),
    )

    rc = arch.run_architect(
        planner_count=1,
        spec=Path(corpus.spec_dir),
        cwd=str(tmp_path),
        budget_usd=3.0,
    )

    assert rc == 2
    summary = _architect_summary(corpus)
    assert summary["status"] == "planner_infrastructure_failure"
    assert summary["failure_phase"] == "architect_validation"
    assert summary["tiers_executed"] == 0
    assert _architect_result(corpus)["status"] == (
        "planner_infrastructure_failure"
    )
    outcome_path = next(
        path
        for path in (Path(corpus.spec_dir) / "runs").glob("*/outcome.json")
        if "_architect_" in path.parent.name
    )
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert outcome["detail_status"] == "planner_infrastructure_failure"
    assert outcome["terminal_class"] == "inconclusive"
