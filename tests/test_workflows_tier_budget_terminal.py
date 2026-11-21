# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Budget and child-terminal contracts for the tier breach boundary."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BudgetExhausted,
    ContentHash,
    ExecutionCompleteness,
    ExecutionContext,
    Implementation,
    Invocation,
    InvocationResult,
    LifecycleEvent,
    ProjectContext,
    ReviewCoverage,
    Spec,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
    Verdict,
)
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

from crucible.common.policies import BreachPolicy, ReviewPolicy
from crucible.workflows._build_verify import BuildVerifyOutcome, RunnerResult
from crucible.workflows._atomic_persistence import persist_and_attach
from crucible.workflows._run import open_run_with_capture


class _OneShotImplementer:
    name = "stub-tier-budget"
    model = "stub-model"

    def __init__(
        self,
        repo: Path,
        *,
        accounting_debt_on_calls: tuple[int, ...] = (),
    ):
        self.repo = repo
        self.calls = 0
        self.accounting_debt_on_calls = frozenset(accounting_debt_on_calls)

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation) -> InvocationResult:
        self.calls += 1
        target = self.repo / f"file_{self.calls - 1}.py"
        target.write_text(f"# implementation {self.calls}\n")
        accounting_complete = self.calls not in self.accounting_debt_on_calls
        return InvocationResult(
            final_message="done",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(
                token_usage_complete=accounting_complete,
                cost_complete=accounting_complete,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),
                ),
            ),
        )


def _final_b_snapshot_evidence() -> UsagePricingEvidence:
    return UsagePricingEvidence(
        segments=(
            UsagePricingSegment(8_627, 2, 4_921, 3_704, 0.03266825),
            UsagePricingSegment(8_627, 2, 4_921, 3_704, 0.03266825),
            UsagePricingSegment(8_627, 2, 4_921, 3_704, 0.03266825),
            UsagePricingSegment(12_735, 2, 4_108, 8_625, 0.0300475),
            UsagePricingSegment(12_735, 2, 4_108, 8_625, 0.0300475),
            UsagePricingSegment(12_735, 2, 4_108, 8_625, 0.0300475),
            UsagePricingSegment(17_973, 3, 5_238, 12_733, 0.039189),
            UsagePricingSegment(17_973, 3, 5_238, 12_733, 0.039189),
            UsagePricingSegment(17_973, 3, 5_238, 12_733, 0.039189),
        ),
        complete=False,
    )


def test_complete_pricing_requires_every_accumulated_call():
    """A later exact call cannot repair an earlier pricing-evidence gap."""

    from crucible.common.cost import CostAccumulator
    from crucible.workflows.tier import _costs_have_complete_pricing

    costs = CostAccumulator(max_cost_usd=1.0)
    first = Usage(10, 2, 0.1, 0, 0)
    second = Usage(8, 1, 0.05, 0, 0)
    for usage, complete in ((first, False), (second, True)):
        costs.record(
            role="implementer",
            usage=usage,
            backend_name="stub-tier-budget",
            model="stub-model",
            pricing_evidence=UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=usage.cache_creation_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cost_usd=usage.cost_usd,
                ),),
                complete=complete,
            ),
        )

    assert costs.to_dict()["accounting_debt"] is False
    assert costs.pricing_evidence is not None
    assert costs.pricing_evidence.complete is False
    assert _costs_have_complete_pricing(costs) is False


class _FinalBSequenceImplementer(_OneShotImplementer):
    """Reproduce the exact three implementation calls from final arm B."""

    model = "claude-opus-5"
    _USAGES = (
        Usage(5_635_703, 51_629, 4.883054, 134_590, 5_500_983),
        Usage(917_756, 9_467, 1.03540175, 59_065, 858_641),
        Usage(39_335, 1_730, 0.14497975, 14_267, 25_062),
    )

    def invoke(self, invocation) -> InvocationResult:  # noqa: ARG002
        self.calls += 1
        if self.calls > len(self._USAGES):
            raise AssertionError("unexpected implementer call")
        target = self.repo / "file_0.py"
        target.write_text(f"# implementation {self.calls}\n")
        usage = self._USAGES[self.calls - 1]
        terminal = self.calls == len(self._USAGES)
        evidence = (
            _final_b_snapshot_evidence()
            if terminal
            else UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=usage.cache_creation_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cost_usd=usage.cost_usd,
                ),),
                complete=True,
            )
        )
        return InvocationResult(
            final_message="budget exhausted" if terminal else "done",
            transcript=(),
            usage=usage,
            stop_reason="budget_exhausted" if terminal else "end_turn",
            error=(
                BudgetExhausted(
                    spent_usd=usage.cost_usd,
                    limit_usd=0.08154425,
                    message="implementer repair authority exhausted",
                )
                if terminal
                else None
            ),
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(BackendExecutionEvent(
                    sequence=0,
                    kind="terminal",
                    name="budget_exhausted" if terminal else "end_turn",
                ),),
            ),
            pricing_evidence=evidence,
        )


class _FullyPricedFinalBSequenceImplementer(_FinalBSequenceImplementer):
    """Make final B's budget terminal exactly priced and reconcilable."""

    def invoke(self, invocation) -> InvocationResult:
        result = super().invoke(invocation)
        if self.calls != len(self._USAGES):
            return result
        usage = result.usage
        return replace(
            result,
            pricing_evidence=UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=usage.cache_creation_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cost_usd=usage.cost_usd,
                ),),
                complete=True,
            ),
        )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tier-budget@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tier Budget"],
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _spec(base_commit: str, *, slice_count: int = 1) -> Spec:
    slices = tuple(
        Slice(
            id=f"s{index}",
            directive=ImplementationDirective(
                slice_id=f"s{index}",
                intent=f"make scoped change {index}",
                files_in_scope=(f"file_{index}.py",),
                creates_paths=(f"file_{index}.py",),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )
        for index in range(slice_count)
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_budget_terminal",
            repo_url="https://example.invalid/tier-budget.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            # Breach is enabled in these boundary tests, so the Spec must
            # satisfy the same trusted observation-runner preflight as a real
            # invocation even though the child workflow itself is stubbed.
            runners={
                "verify": "crucible.common.runners.lang.python",
            },
        ),
        execution=None,
        intent="exercise tier child terminal behavior",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(
            tiers=(Tier(
                tier_id="t0",
                slices=slices,
                depends_on=(),
            ),),
        ),
    )


def _setup(
    tmp_path: Path,
    monkeypatch,
    *,
    slice_count: int = 1,
    accounting_debt_on_calls: tuple[int, ...] = (),
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base_commit = _init_repo(repo)
    spec_dir = tmp_path / "corpus" / "spec1"
    spec_dir.mkdir(parents=True)
    Corpus(spec_dir).put_spec(_spec(base_commit, slice_count=slice_count))
    backend = _OneShotImplementer(
        repo,
        accounting_debt_on_calls=accounting_debt_on_calls,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kwargs: backend,
    )
    return repo, spec_dir, backend


def test_breach_record_delta_accepts_exact_zero_deterministic_record(
    tmp_path: Path,
):
    from crucible.workflows.tier import _breach_record_delta_usage

    spec_dir = tmp_path / "corpus"
    spec_dir.mkdir()
    corpus = Corpus(spec_dir)
    corpus.put_spec(_spec("a" * 40))
    before = corpus.latest_spec()
    assert before is not None
    priced_segment = UsagePricingSegment(
        input_tokens=10,
        output_tokens=2,
        cache_creation_tokens=3,
        cache_read_tokens=4,
        cost_usd=0.10,
    )

    def implementation(
        slice_id: str,
        *,
        classification: str,
        usage: Usage,
        evidence: UsagePricingEvidence,
        backend: str,
    ) -> Implementation:
        return Implementation(
            slice_id=slice_id,
            base_commit=before.project_context.base_commit,
            result_commit=None,
            diff_ref=corpus.put_artifact(
                b"", kind="diffs", name_hint=f"{slice_id}.diff"
            ),
            transcript_ref=corpus.put_artifact(
                b"{}", kind="transcripts", name_hint=f"{slice_id}.json"
            ),
            files_modified=(),
            scope_escapes=(),
            classification=classification,
            execution=ExecutionContext(
                timestamp="2026-08-29T00:00:00Z",
                machine="test",
                cwd=str(tmp_path),
                toolchain={},
                backend=backend,
                model=None,
                recipe="breach",
                recipe_commit=None,
                gpu_model=None,
                run_id=None,
            ),
            cost_usd=usage.cost_usd,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=True,
            ),
            pricing_evidence=evidence,
        )

    priced = implementation(
        "priced",
        classification="success",
        usage=Usage(
            input_tokens=priced_segment.input_tokens,
            output_tokens=priced_segment.output_tokens,
            cache_creation_tokens=priced_segment.cache_creation_tokens,
            cache_read_tokens=priced_segment.cache_read_tokens,
            cost_usd=priced_segment.cost_usd,
        ),
        evidence=UsagePricingEvidence(
            segments=(priced_segment,),
            complete=True,
        ),
        backend="codex",
    )
    deterministic = implementation(
        "defender-not-required",
        classification="success_pre_fix_attacks_passed",
        usage=Usage(),
        evidence=UsagePricingEvidence(segments=(), complete=True),
        backend="breach_defender_not_required",
    )
    refs = tuple(
        corpus.put_record(record, kind="implementations")
        for record in (priced, deterministic)
    )
    after = before.with_appended_refs(implementation_refs=refs)

    usage, pricing, unloaded, debt = _breach_record_delta_usage(
        corpus,
        before,
        after,
    )

    assert usage == Usage(
        input_tokens=10,
        output_tokens=2,
        cache_creation_tokens=3,
        cache_read_tokens=4,
        cost_usd=0.10,
    )
    assert pricing == UsagePricingEvidence(
        segments=(priced_segment,),
        complete=True,
    )
    assert unloaded == 0
    assert debt == ()


def _run_tier(
    repo: Path,
    spec_dir: Path,
    *,
    budget_usd: float,
    breach_policy: BreachPolicy | None = None,
    review_policy: ReviewPolicy | None = None,
    skip_build_verify: bool = True,
    build_retries: int = 2,
    extra_parameters: dict | None = None,
) -> int:
    from crucible.workflows.tier import run_tier

    return run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=budget_usd,
        skip_build_verify=skip_build_verify,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        breach_policy=(
            breach_policy
            if breach_policy is not None
            else BreachPolicy(enabled=True, max_rounds=1)
        ),
        review_policy=(
            review_policy
            if review_policy is not None
            else ReviewPolicy(rubrics=())
        ),
        build_retries=build_retries,
        extra_parameters=extra_parameters,
    )


def _tier_run_dir(spec_dir: Path) -> Path:
    return sorted((spec_dir / "runs").glob("*tier*"))[-1]


def _tier_documents(spec_dir: Path) -> tuple[dict, dict]:
    run_dir = _tier_run_dir(spec_dir)
    return (
        json.loads((run_dir / "summary.json").read_text()),
        json.loads((run_dir / "result.json").read_text()),
    )


def _pricing_segment(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
    cost_usd: float,
) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cost_usd": cost_usd,
    }


def _pricing_call(*, role: str, backend: str, model: str, segment: dict) -> dict:
    return {
        "role": role,
        "backend": backend,
        "model": model,
        "usage": dict(segment),
        "pricing_evidence": {
            "segments": [dict(segment)],
            "complete": True,
        },
    }


def _priced_child_costs(*, total_input_tokens: int = 20) -> dict:
    breaker = _pricing_segment(
        input_tokens=8,
        output_tokens=2,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        cost_usd=0.08,
    )
    judge = _pricing_segment(
        input_tokens=12,
        output_tokens=4,
        cache_creation_tokens=1,
        cache_read_tokens=2,
        cost_usd=0.12,
    )
    return {
        "pricing_evidence_schema": "crucible.usage-pricing-ledger.v1",
        "total_usd": 0.20,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": 6,
        "total_cache_creation_tokens": 3,
        "total_cache_read_tokens": 5,
        "by_role": {
            "impl_breaker": {
                "cost_usd": 0.08,
                "input_tokens": 8,
                "output_tokens": 2,
                "calls": 1,
                "backends": ["codex"],
                "models": ["gpt-5.4"],
                "cache_creation_tokens": 2,
                "cache_read_tokens": 3,
            },
            "breach_judge": {
                "cost_usd": 0.12,
                "input_tokens": 12,
                "output_tokens": 4,
                "calls": 1,
                "backends": ["kimi"],
                "models": ["kimi-k2.7-turbo"],
                "cache_creation_tokens": 1,
                "cache_read_tokens": 2,
            },
        },
        "by_backend": {"codex": 0.08, "kimi": 0.12},
        "by_model": {"gpt-5.4": 0.08, "kimi-k2.7-turbo": 0.12},
        "accounting_debt": False,
        "accounting_debt_by_role": {},
        "unloaded_refs": 0,
        "rollup_incomplete": False,
        "pricing_evidence": {
            "segments": [breaker, judge],
            "complete": True,
        },
        "pricing_evidence_by_call": [
            _pricing_call(
                role="impl_breaker",
                backend="codex",
                model="gpt-5.4",
                segment=breaker,
            ),
            _pricing_call(
                role="breach_judge",
                backend="kimi",
                model="kimi-k2.7-turbo",
                segment=judge,
            ),
        ],
    }


def test_child_breach_exception_finalizes_tier_with_reconciled_cost(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)

    def _raising_breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        issue_text = argv[argv.index("--issue-text") + 1]
        assert issue_text == "exercise tier child terminal behavior"
        assert "--authority-file" in argv
        assert "--authority-digest" in argv
        assert "Active tier `t0`" not in issue_text
        assert "make one scoped change" not in issue_text
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        latest = child_corpus.latest_spec()
        assert latest is not None
        diff_ref = child_corpus.put_artifact(
            b"child partial diff", kind="diffs", name_hint="child.diff"
        )
        transcript_ref = child_corpus.put_artifact(
            b"child transcript", kind="transcripts", name_hint="child.json"
        )
        paid_attempt = Implementation(
            slice_id="breach/paid-attempt",
            base_commit=latest.project_context.base_commit,
            result_commit=None,
            diff_ref=diff_ref,
            transcript_ref=transcript_ref,
            files_modified=(),
            scope_escapes=(),
            classification="hard_fail_child_exception",
            execution=ExecutionContext(
                timestamp="2026-06-28T00:00:00Z",
                machine="test",
                cwd=str(repo),
                toolchain={},
                backend="stub",
                model="stub-model",
                recipe="breach",
                recipe_commit=None,
                gpu_model=None,
                run_id=None,
            ),
            cost_usd=0.25,
            input_tokens=13,
            output_tokens=5,
            pricing_evidence=UsagePricingEvidence(
                segments=(
                    UsagePricingSegment(
                        input_tokens=13,
                        output_tokens=5,
                        cache_creation_tokens=3,
                        cache_read_tokens=4,
                        cost_usd=0.25,
                    ),
                ),
                complete=False,
            ),
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=False,
                trace_complete=False,
            ),
        )
        with open_run_with_capture(child_corpus, "breach"):
            persist_and_attach(
                child_corpus,
                paid_attempt,
                kind="implementations",
                attachment="implementation",
            )
            raise RuntimeError("synthetic child failure")

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _raising_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "breach_exception"
    assert summary["phase_3_breach"]["status"] == "failed"
    assert summary["phase_3_breach"]["error"] == {
        "kind": "RuntimeError",
        "message": "synthetic child failure",
    }
    reconciliation = summary["phase_3_breach"]["cost_reconciliation"]
    assert reconciliation["summary_usd"] == 0.0
    assert reconciliation["result_usd"] == 0.0
    assert reconciliation["attached_records_usd"] == pytest.approx(0.25)
    assert reconciliation["reconciled_usd"] == pytest.approx(0.25)
    assert reconciliation["summary_input_tokens"] is None
    assert reconciliation["attached_records_input_tokens"] == 13
    assert reconciliation["reconciled_input_tokens"] == 13
    assert reconciliation["reconciled_output_tokens"] == 5
    assert reconciliation["token_source"] == "attached_record_fallback"
    assert reconciliation["reconciled_cache_creation_tokens"] == 3
    assert reconciliation["reconciled_cache_read_tokens"] == 4
    assert reconciliation["pricing_source"] == "attached_record_fallback"
    assert reconciliation["pricing_ledger_valid"] is False
    assert reconciliation["pricing_evidence_complete"] is False
    assert reconciliation["pricing_errors"] == ["child_pricing_ledger_unavailable"]
    assert reconciliation["accounting_debt"] == [
        "cost_complete",
        "token_usage_complete",
    ]
    assert summary["costs"]["total_usd"] == pytest.approx(0.26)
    # Initial implementer call (1/1) plus the persisted breach fallback.
    assert summary["costs"]["total_input_tokens"] == 14
    assert summary["costs"]["total_output_tokens"] == 6
    assert summary["costs"]["total_cache_creation_tokens"] == 3
    assert summary["costs"]["total_cache_read_tokens"] == 4
    assert summary["costs"]["pricing_evidence"]["complete"] is False
    assert summary["costs"]["pricing_evidence"]["segments"] == [
        {
            "input_tokens": 13,
            "output_tokens": 5,
            "cache_creation_tokens": 3,
            "cache_read_tokens": 4,
            "cost_usd": 0.25,
        }
    ]
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["remaining_usd"] == 0.0
    assert result["status"] == "breach_exception"
    assert result["cost_usd"] == pytest.approx(0.26)
    assert result["error_kind"] == "RuntimeError"


def test_child_exact_cap_budget_status_propagates_to_tier(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)

    def _budget_breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        with open_run_with_capture(child_corpus, "breach") as child:
            child.write_summary({
                "status": "budget_exhausted",
                "spent_usd": 0.20,
                "costs": {
                    "total_usd": 0.20,
                    "total_input_tokens": 101,
                    "total_output_tokens": 23,
                },
                "rounds": [],
            })
            child.finalize(RunResult(
                status="budget_exhausted",
                cost_usd=0.20,
            ))
        return 1

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _budget_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=0.21) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "budget_exhausted"
    assert summary["phase_3_breach"]["status"] == "budget_exhausted"
    assert summary["phase_3_breach"]["cost_reconciliation"][
        "reconciled_usd"
    ] == pytest.approx(0.20)
    assert summary["phase_3_breach"]["cost_reconciliation"][
        "token_source"
    ] == "child_summary"
    assert summary["costs"]["total_usd"] == pytest.approx(0.21)
    # The child summary is authoritative even though no final breach record
    # carries these discarded-attempt tokens.
    assert summary["costs"]["total_input_tokens"] == 102
    assert summary["costs"]["total_output_tokens"] == 24
    assert "pricing_evidence_schema" not in summary["costs"]
    assert "pricing_evidence" not in summary["costs"]
    assert "pricing_evidence_by_call" not in summary["costs"]
    assert "total_cache_creation_tokens" not in summary["costs"]
    assert "total_cache_read_tokens" not in summary["costs"]
    assert result["status"] == "budget_exhausted"
    assert result["cost_usd"] == pytest.approx(0.21)


def test_child_pricing_ledger_replays_calls_once_with_cache_and_order(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)

    def _priced_breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        with open_run_with_capture(child_corpus, "breach") as child:
            child.write_summary(
                {
                    "status": "budget_exhausted",
                    "spent_usd": 0.20,
                    "costs": _priced_child_costs(),
                    "rounds": [],
                }
            )
            child.finalize(
                RunResult(
                    status="budget_exhausted",
                    cost_usd=0.20,
                )
            )
        return 1

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _priced_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    reconciliation = summary["phase_3_breach"]["cost_reconciliation"]
    assert reconciliation["pricing_source"] == "child_call_ledger"
    assert reconciliation["pricing_ledger_valid"] is True
    assert reconciliation["pricing_evidence_complete"] is True
    assert reconciliation["pricing_errors"] == []
    assert reconciliation["reconciled_cache_creation_tokens"] == 3
    assert reconciliation["reconciled_cache_read_tokens"] == 5

    costs = summary["costs"]
    # 0.01/1/1 from Tier's implementer plus the two child calls exactly once.
    assert costs["total_usd"] == pytest.approx(0.21)
    assert costs["total_input_tokens"] == 21
    assert costs["total_output_tokens"] == 7
    assert costs["total_cache_creation_tokens"] == 3
    assert costs["total_cache_read_tokens"] == 5
    assert costs["pricing_evidence_schema"] == (
        "crucible.usage-pricing-ledger.v1"
    )
    assert "breach_subprocess" not in costs["by_role"]
    assert [item["role"] for item in costs["pricing_evidence_by_call"]] == [
        "implementer",
        "impl_breaker",
        "breach_judge",
    ]
    assert [item["backend"] for item in costs["pricing_evidence_by_call"]] == [
        "stub-tier-budget",
        "codex",
        "kimi",
    ]
    # The legacy implementer is an explicit gap once later calls provide
    # evidence; it cannot be hidden by the exact child ledger.
    assert costs["pricing_evidence_by_call"][0]["pricing_evidence"] == {
        "segments": [],
        "complete": False,
    }
    assert costs["pricing_evidence"]["complete"] is False
    assert [
        segment["input_tokens"] for segment in costs["pricing_evidence"]["segments"]
    ] == [8, 12]
    assert result["status"] == "budget_exhausted"


def test_child_pricing_mismatch_falls_back_once_and_stays_incomplete(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)

    def _mismatched_breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        mismatched_costs = _priced_child_costs(total_input_tokens=21)
        mismatched_costs["by_backend"]["codex"] = 0.09
        with open_run_with_capture(child_corpus, "breach") as child:
            child.write_summary(
                {
                    "status": "budget_exhausted",
                    "spent_usd": 0.20,
                    # The ledger and aggregate account for 20 input tokens, while
                    # the advertised child total claims 21. The backend cost
                    # projection is also inconsistent with those exact calls.
                    "costs": mismatched_costs,
                    "rounds": [],
                }
            )
            child.finalize(
                RunResult(
                    status="budget_exhausted",
                    cost_usd=0.20,
                )
            )
        return 1

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _mismatched_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    reconciliation = summary["phase_3_breach"]["cost_reconciliation"]
    assert reconciliation["pricing_source"] == "child_aggregate_fallback"
    assert reconciliation["pricing_ledger_valid"] is False
    assert reconciliation["pricing_evidence_complete"] is False
    assert reconciliation["pricing_errors"] == [
        "logical_call_ledger_total_mismatch",
        "by_backend_projection_mismatch",
        "aggregate_usage_mismatch",
    ]

    costs = summary["costs"]
    assert costs["total_usd"] == pytest.approx(0.21)
    assert costs["total_input_tokens"] == 22
    assert costs["total_output_tokens"] == 7
    assert costs["pricing_evidence_schema"] == (
        "crucible.usage-pricing-ledger.v1"
    )
    # The aggregate fallback is charged once; malformed child calls are not
    # also replayed into the parent accumulator.
    assert list(costs["by_role"]) == ["breach_subprocess", "implementer"]
    assert [item["role"] for item in costs["pricing_evidence_by_call"]] == [
        "implementer",
        "breach_subprocess",
    ]
    assert costs["pricing_evidence"]["complete"] is False
    assert [
        segment["input_tokens"] for segment in costs["pricing_evidence"]["segments"]
    ] == [8, 12]
    assert result["status"] == "budget_exhausted"


def test_child_unloaded_pricing_rollup_revokes_parent_spend_authority(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)

    def _unloaded_breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        incomplete_costs = _priced_child_costs()
        incomplete_costs["unloaded_refs"] = 2
        incomplete_costs["rollup_incomplete"] = True
        incomplete_costs["pricing_evidence"]["complete"] = False
        with open_run_with_capture(child_corpus, "breach") as child:
            child.write_summary({
                "status": "no_attacks",
                "spent_usd": 0.20,
                "costs": incomplete_costs,
                "rounds": [],
            })
            child.finalize(RunResult(
                status="no_attacks",
                cost_usd=0.20,
            ))
        return 0

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _unloaded_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "incomplete_accounting"
    reconciliation = summary["phase_3_breach"]["cost_reconciliation"]
    assert reconciliation["unloaded_refs"] == 2
    assert reconciliation["pricing_ledger_valid"] is False
    assert reconciliation["pricing_errors"] == [
        "child_rollup_incomplete_without_accounting_debt"
    ]
    assert reconciliation["accounting_debt"] == [
        "cost_complete",
        "token_usage_complete",
    ]
    costs = summary["costs"]
    assert costs["unloaded_refs"] == 2
    assert costs["rollup_incomplete"] is True
    assert costs["remaining_usd"] == 0.0
    assert costs["accounting_debt_by_role"] == {
        "breach_subprocess": [
            "cost_complete",
            "token_usage_complete",
        ]
    }
    assert costs["pricing_evidence"]["complete"] is False
    assert result["status"] == "incomplete_accounting"


def test_child_accounting_debt_propagates_without_budget_misclassification(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)

    def _accounting_debt_breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        with open_run_with_capture(child_corpus, "breach") as child:
            child.write_summary({
                "status": "incomplete_accounting",
                "spent_usd": 0.20,
                "costs": {
                    "total_usd": 0.20,
                    "total_input_tokens": 101,
                    "total_output_tokens": 23,
                    "accounting_debt": True,
                    "accounting_debt_by_role": {
                        "impl_breaker": [
                            "token_usage_complete",
                            "cost_complete",
                        ],
                    },
                    "rollup_incomplete": True,
                },
                "rounds": [],
            })
            child.finalize(RunResult(
                status="incomplete_accounting",
                cost_usd=0.20,
            ))
        return 3

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _accounting_debt_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "incomplete_accounting"
    assert summary["phase_3_breach"]["status"] == "incomplete_accounting"
    reconciliation = summary["phase_3_breach"]["cost_reconciliation"]
    assert reconciliation["accounting_debt"] == [
        "cost_complete",
        "token_usage_complete",
    ]
    assert reconciliation["cost_complete"] is False
    assert reconciliation["complete"] is False
    assert summary["costs"]["total_usd"] == pytest.approx(0.21)
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["accounting_debt_by_role"] == {
        "breach_subprocess": [
            "cost_complete",
            "token_usage_complete",
        ],
    }
    assert summary["costs"]["remaining_usd"] == 0.0
    assert result["status"] == "incomplete_accounting"
    assert result["cost_usd"] == pytest.approx(0.21)


def _publishing_breach(
    status: str,
    exit_code: int,
    calls: list[str],
    *,
    spent_usd: float = 0.0,
    tokens: tuple[int, int] = (0, 0),
):
    """A breach child that publishes one typed terminal and exits."""

    def _breach(
        argv: list[str],
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ) -> int:
        assert fresh_lineage is False
        assert development_budget_ledger is None
        calls.append(status)
        child_corpus = Corpus(Path(argv[argv.index("--spec") + 1]))
        with open_run_with_capture(child_corpus, "breach") as child:
            child.write_summary({
                "status": status,
                "spent_usd": spent_usd,
                "costs": {
                    "total_usd": spent_usd,
                    "total_input_tokens": tokens[0],
                    "total_output_tokens": tokens[1],
                },
                "rounds": [],
            })
            child.finalize(RunResult(status=status, cost_usd=spent_usd))
        return exit_code

    return _breach


def _lifecycle_kinds(spec_dir: Path) -> list[str]:
    corpus = Corpus(spec_dir)
    latest = corpus.latest_spec()
    assert latest is not None
    return [
        corpus.get_record(ref, as_type=LifecycleEvent).kind
        for ref in latest.lifecycle_event_refs
    ]


@pytest.mark.parametrize(
    ("child_status", "child_class"),
    [
        ("inconclusive_judge", "inconclusive"),
        ("inconclusive_breaker", "inconclusive"),
        # The attack-authority reviewer returned no usable admission, so no
        # attack reached the Defender or the Judge.
        ("inconclusive_authority", "inconclusive"),
        ("inconclusive_transport_failed", "inconclusive"),
        # A gate could not run: no evidence about the accepted bytes.
        ("inconclusive_gate_unavailable", "inconclusive"),
        # A retryable child is unruled as well; it is not a breach failure.
        ("concurrent_spec_change", "retryable_failure"),
        # Breach publishes no cancelled terminal of its own today; the hop
        # reads the class, so a stopped child is not filed as a failure.
        ("cancelled", "cancelled"),
    ],
)
def test_unruled_breach_child_ends_breach_inconclusive(
    tmp_path: Path,
    monkeypatch,
    child_status: str,
    child_class: str,
):
    """A breach loop that reached no ruling is not a failed breach."""

    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    breach_calls: list[str] = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _publishing_breach(child_status, 3, breach_calls),
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.review",
        lambda *args, **kwargs: pytest.fail("unruled breach reached review"),
    )

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=1.0,
        review_policy=ReviewPolicy(rubrics=("diff_scope",)),
    ) == 1
    assert backend.calls == 1
    assert breach_calls == [child_status]
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "breach_inconclusive"
    assert result["status"] == "breach_inconclusive"
    # The exact child terminal stays readable below the tier status.
    assert summary["phase_3_breach"]["status"] == child_status
    assert summary["phase_3_breach"]["terminal_class"] == child_class
    assert summary["phase_3_breach"]["exit_code"] == 3
    assert summary["phase_4_reviews"] == []
    outcome = json.loads(
        (_tier_run_dir(spec_dir) / "outcome.json").read_text()
    )
    assert outcome["detail_status"] == "breach_inconclusive"
    assert outcome["terminal_class"] == "inconclusive"
    # Nothing lands. The verified Phase 1 candidate stays as it was before
    # breach, exactly as for a failed breach.
    assert "landed" not in _lifecycle_kinds(spec_dir)
    assert (repo / "file_0.py").read_text() == "# implementation 1\n"


def test_unruled_breach_child_on_a_deferred_seed_ends_breach_inconclusive(
    tmp_path: Path,
    monkeypatch,
):
    """The benchmark repair loop was lost; the seed's own gate row is kept.

    A single-slice benchmark tier hands a slice that failed its gate to
    Breach as the repair seed. When that loop ends without a ruling, the tier
    reports the lost loop; the deterministic judgement of the seed stays in
    its Phase 1 row and the seed is restored.
    """

    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    from crucible.workflows import tier as tier_workflow

    def _failed_gate(*args, **kwargs):  # noqa: ARG001
        outcome = BuildVerifyOutcome(
            "dev_verify_failed",
            build=None,
            verify=None,
            dev_verify=RunnerResult(
                role="dev_verify",
                runner="tests.deferred_seed_gate",
                returncode=1,
                values={"tests_failed": 1.0},
                raw_output=json.dumps({
                    "raw_output": "one project-visible test failed",
                }).encode(),
                stderr=b"",
                duration_s=0.01,
            ),
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_deferred_seed_gate_stub",
            succeeded=False,
        )
        return outcome

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _failed_gate,
    )
    breach_calls: list[str] = []
    seeds: list[str] = []
    publish = _publishing_breach("inconclusive_judge", 3, breach_calls)

    def _breach(argv, *, fresh_lineage, development_budget_ledger):
        seeds.append((repo / "file_0.py").read_text())
        from crucible.workflows import _mutation_journal

        # A real child settles the writer epoch it joins on the deferred
        # seed's transaction before it returns.
        parent = (
            _mutation_journal._require_live_parent_transaction(  # noqa: SLF001
                argv[argv.index("--parent-mutation-transaction-id") + 1]
            )
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            _mutation_journal.live_writer_epoch_family(parent),
            phase="test_deferred_seed_breach_stub",
            succeeded=False,
        )
        return publish(
            argv,
            fresh_lineage=fresh_lineage,
            development_budget_ledger=development_budget_ledger,
        )

    monkeypatch.setattr(tier_workflow.breach_workflow, "main", _breach)

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=1.0,
        skip_build_verify=False,
        build_retries=0,
        extra_parameters={"deepswe_metadata": {}},
    ) == 1
    assert backend.calls == 1
    # The failing candidate was the Breach seed.
    assert seeds == ["# implementation 1\n"]
    assert breach_calls == ["inconclusive_judge"]
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "breach_inconclusive"
    assert result["status"] == "breach_inconclusive"
    assert summary["phase_1_slices"][0]["classification"] == (
        "dev_verify_failed"
    )
    assert summary["phase_3_breach"]["status"] == "inconclusive_judge"
    assert "landed" not in _lifecycle_kinds(spec_dir)
    # The unrepaired seed is restored.
    assert not (repo / "file_0.py").exists()


def test_unruled_breach_child_that_spent_the_budget_ends_budget_exhausted(
    tmp_path: Path,
    monkeypatch,
):
    """Budget keeps its precedence over the unruled-loop status."""

    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    breach_calls: list[str] = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _publishing_breach(
            "inconclusive_judge",
            3,
            breach_calls,
            spent_usd=0.20,
            tokens=(101, 23),
        ),
    )

    # The implementer spends 0.01 and the child the remaining 0.20.
    assert _run_tier(repo, spec_dir, budget_usd=0.21) == 1
    assert backend.calls == 1
    assert breach_calls == ["inconclusive_judge"]
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "budget_exhausted"
    assert result["status"] == "budget_exhausted"
    assert summary["costs"]["total_usd"] == pytest.approx(0.21)
    assert summary["costs"]["remaining_usd"] == 0.0
    # The child's own unruled terminal stays readable below the tier status.
    assert summary["phase_3_breach"]["status"] == "inconclusive_judge"
    assert summary["phase_3_breach"]["terminal_class"] == "inconclusive"
    assert summary["phase_3_breach"]["exit_code"] == 3
    outcome = json.loads(
        (_tier_run_dir(spec_dir) / "outcome.json").read_text()
    )
    assert outcome["detail_status"] == "budget_exhausted"
    assert outcome["terminal_class"] == "budget_exhausted"


@pytest.mark.parametrize(
    "child_status",
    [
        # Breach accepted the candidate and the final shipping gate ran and
        # did not pass it, or changed source or Git state.
        "inconclusive_post_strip_verify",
        # A deterministic gate changed source or Git state or wrote outside
        # the transaction scope, or the candidate was absent from the
        # workspace.
        "inconclusive_workspace_mismatch",
    ],
)
def test_breach_child_refused_on_deterministic_evidence_keeps_breach_failed(
    tmp_path: Path,
    monkeypatch,
    child_status: str,
):
    """Not every inconclusive breach terminal is a loop without a ruling.

    Breach files these two statuses under ``inconclusive``, but both follow
    deterministic evidence, not a lost reviewer. As a Phase 1 gate that
    changed source keeps ``slice_failed``, they keep ``breach_failed``.
    """

    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    breach_calls: list[str] = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _publishing_breach(child_status, 3, breach_calls),
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    assert breach_calls == [child_status]
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "breach_failed"
    assert result["status"] == "breach_failed"
    assert summary["phase_3_breach"]["status"] == child_status
    assert summary["phase_3_breach"]["terminal_class"] == "inconclusive"
    assert summary["phase_3_breach"]["exit_code"] == 3
    outcome = json.loads(
        (_tier_run_dir(spec_dir) / "outcome.json").read_text()
    )
    assert outcome["detail_status"] == "breach_failed"
    assert outcome["terminal_class"] == "permanent_failure"
    assert "landed" not in _lifecycle_kinds(spec_dir)
    assert (repo / "file_0.py").read_text() == "# implementation 1\n"


def test_unavailable_gate_in_a_benchmark_tier_is_never_a_breach_seed(
    tmp_path: Path,
    monkeypatch,
):
    """The benchmark shape that produces ``dev_verify_unavailable``.

    A single-slice benchmark tier defers a slice its gate judged failing to
    Breach. A gate that could not run judged nothing: the candidate is not a
    seed, the model gets no retry, and the tier ends ``slice_inconclusive``.
    """

    from tests.test_tier_build_progress_retry import _dev_verify_unavailable

    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    from crucible.workflows import tier as tier_workflow

    gate_calls: list[str] = []

    def _unavailable_gate(*args, **kwargs):  # noqa: ARG001
        gate_calls.append(kwargs.get("slice_id", "?"))
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_unavailable_benchmark_gate_stub",
            succeeded=False,
        )
        return _dev_verify_unavailable()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _unavailable_gate,
    )
    breach_seeds: list[str] = []

    def _unexpected_breach(argv, *, fresh_lineage, development_budget_ledger):  # noqa: ARG001
        from crucible.workflows import _mutation_journal

        breach_seeds.append((repo / "file_0.py").read_text())
        # Settle the writer epoch a real child joins on the seed transaction,
        # so that a regression fails on the assertions below and not in the
        # tier's own cleanup.
        parent = (
            _mutation_journal._require_live_parent_transaction(  # noqa: SLF001
                argv[argv.index("--parent-mutation-transaction-id") + 1]
            )
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            _mutation_journal.live_writer_epoch_family(parent),
            phase="test_unavailable_benchmark_gate_breach_stub",
            succeeded=False,
        )
        raise RuntimeError("an unjudged slice must not reach Breach")

    monkeypatch.setattr(tier_workflow.breach_workflow, "main", _unexpected_breach)

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=1.0,
        skip_build_verify=False,
        build_retries=2,
        extra_parameters={"deepswe_metadata": {}},
    ) == 1
    assert breach_seeds == []
    # No retry is keyed on the gate result: one gate run, one model call.
    assert len(gate_calls) == 1
    assert backend.calls == 1
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "slice_inconclusive"
    assert result["status"] == "slice_inconclusive"
    assert summary["phase_1_slices"][0]["classification"] == (
        "dev_verify_unavailable"
    )
    assert summary["phase_3_breach"] is None
    outcome = json.loads(
        (_tier_run_dir(spec_dir) / "outcome.json").read_text()
    )
    assert outcome["detail_status"] == "slice_inconclusive"
    assert outcome["terminal_class"] == "inconclusive"
    assert "landed" not in _lifecycle_kinds(spec_dir)
    # The slice transaction is restored.
    assert summary["rollback_applied"] is True
    assert not (repo / "file_0.py").exists()


@pytest.mark.parametrize(
    ("child_status", "child_exit", "tier_status", "tier_exit"),
    [
        ("max_rounds_exhausted", 1, "breach_failed", 1),
        ("persistence_veto", 1, "breach_failed", 1),
        ("incomplete", 3, "breach_failed", 1),
        ("reject", 2, "breach_reject", 2),
    ],
)
def test_breach_child_with_a_permanent_terminal_keeps_its_tier_status(
    tmp_path: Path,
    monkeypatch,
    child_status: str,
    child_exit: int,
    tier_status: str,
    tier_exit: int,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    breach_calls: list[str] = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _publishing_breach(child_status, child_exit, breach_calls),
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == tier_exit
    assert backend.calls == 1
    assert breach_calls == [child_status]
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == tier_status
    assert result["status"] == tier_status
    assert summary["phase_3_breach"]["status"] == child_status
    assert summary["phase_3_breach"]["terminal_class"] == "permanent_failure"
    outcome = json.loads(
        (_tier_run_dir(spec_dir) / "outcome.json").read_text()
    )
    assert outcome["detail_status"] == tier_status
    assert outcome["terminal_class"] == "permanent_failure"
    assert "landed" not in _lifecycle_kinds(spec_dir)
    assert (repo / "file_0.py").read_text() == "# implementation 1\n"


def test_implementation_accounting_debt_stops_before_next_slice(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(
        tmp_path,
        monkeypatch,
        slice_count=2,
        accounting_debt_on_calls=(1,),
    )
    breach_called = False

    def _unexpected_breach(argv: list[str]) -> int:  # noqa: ARG001
        nonlocal breach_called
        breach_called = True
        raise AssertionError("accounting debt must stop before breach")

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _unexpected_breach,
    )

    assert _run_tier(repo, spec_dir, budget_usd=1.0) == 1
    assert backend.calls == 1
    assert breach_called is False
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "incomplete_accounting"
    assert summary["costs"]["total_usd"] == pytest.approx(0.01)
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": [
            "cost_complete",
            "token_usage_complete",
        ],
    }
    assert summary["costs"]["remaining_usd"] == 0.0
    assert result["status"] == "incomplete_accounting"
    outcome = json.loads(
        (_tier_run_dir(spec_dir) / "outcome.json").read_text()
    )
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"


def test_final_promotion_rejects_single_slice_accounting_debt(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(
        tmp_path,
        monkeypatch,
        accounting_debt_on_calls=(1,),
    )
    (repo / "README.md").write_text("inherited staged dirt\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    (repo / "ambient.txt").write_text("inherited untracked dirt\n")
    inherited_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    downstream_calls: list[str] = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: downstream_calls.append("breach") or 0,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.review",
        lambda *args, **kwargs: (
            downstream_calls.append("review"),
            pytest.fail("accounting debt reached review"),
        )[1],
    )

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=1.0,
        breach_policy=BreachPolicy(enabled=True, max_rounds=1),
        review_policy=ReviewPolicy(rubrics=("diff_scope",)),
    ) == 1
    assert backend.calls == 1
    assert downstream_calls == []
    assert not (repo / "file_0.py").exists()
    assert (repo / "README.md").read_text() == "inherited staged dirt\n"
    assert (repo / "ambient.txt").read_text() == "inherited untracked dirt\n"
    assert subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout == inherited_status
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "incomplete_accounting"
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_incomplete_evidence"
    )
    assert summary["costs"]["total_usd"] == pytest.approx(0.01)
    assert summary["costs"]["remaining_usd"] == 0.0
    assert result["status"] == "incomplete_accounting"
    corpus = Corpus(spec_dir)
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_incomplete_evidence"
    assert implementation.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )


def test_final_b_budget_terminal_checkpoints_all_three_calls(
    tmp_path: Path,
    monkeypatch,
):
    """A partial SDK ledger cannot erase the returned terminal usage."""

    repo, spec_dir, _ = _setup(tmp_path, monkeypatch)
    backend = _FinalBSequenceImplementer(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )
    from crucible.workflows import tier as tier_workflow

    gate_calls = 0

    def _failed_gate(*args, **kwargs):  # noqa: ARG001
        nonlocal gate_calls
        gate_calls += 1
        outcome = BuildVerifyOutcome(
            "verify_failed",
            build=None,
            verify=RunnerResult(
                role="verify",
                runner="tests.final_b_gate",
                returncode=1,
                values={"tests_failed": 1.0},
                raw_output=json.dumps({
                    "raw_output": f"repairable failure {gate_calls}",
                }).encode(),
                stderr=b"repairable failure",
                duration_s=0.01,
            ),
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_final_b_gate_stub",
            succeeded=False,
        )
        return outcome

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _failed_gate,
    )
    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "main",
        lambda argv: pytest.fail("budget terminal reached breach"),
    )

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=11.04060975,
        skip_build_verify=False,
        build_retries=2,
        extra_parameters={"deepswe_metadata": {}},
    ) == 1
    assert backend.calls == 3
    assert gate_calls == 2
    assert not (repo / "file_0.py").exists()

    summary, result = _tier_documents(spec_dir)
    costs = summary["costs"]
    assert costs["total_usd"] == pytest.approx(6.0634355)
    assert costs["total_input_tokens"] == 6_592_794
    assert costs["total_output_tokens"] == 62_826
    assert costs["total_cache_creation_tokens"] == 207_922
    assert costs["total_cache_read_tokens"] == 6_384_686
    assert costs["by_role"]["implementer"]["calls"] == 3
    assert costs["accounting_debt"] is False
    assert costs["pricing_evidence"]["complete"] is False
    terminal_call = costs["pricing_evidence_by_call"][2]
    assert terminal_call["usage"] == {
        "input_tokens": 39_335,
        "output_tokens": 1_730,
        "cache_creation_tokens": 14_267,
        "cache_read_tokens": 25_062,
        "cost_usd": 0.14497975,
    }
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_budget_exhausted"
    )
    assert result["cost_usd"] == pytest.approx(6.0634355)
    corpus = Corpus(spec_dir)
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "hard_fail_budget_exhausted"
    assert implementation.cache_creation_tokens == 14_267
    assert implementation.cache_read_tokens == 25_062
    assert implementation.pricing_evidence == _final_b_snapshot_evidence()
    assert "build_retry_budget_terminal_restoration" not in (
        implementation.artifacts
    )
    assert "build_retry_budget_terminal_restorations" not in summary


def test_fully_priced_build_retry_budget_terminal_restores_prior_breach_seed(
    tmp_path: Path,
    monkeypatch,
):
    """A closed budget terminal cannot replace a prior rankable candidate."""

    repo, spec_dir, _ = _setup(tmp_path, monkeypatch)
    backend = _FullyPricedFinalBSequenceImplementer(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )
    from crucible.workflows import tier as tier_workflow

    gate_calls = 0

    def _failed_gate(*args, **kwargs):  # noqa: ARG001
        nonlocal gate_calls
        gate_calls += 1
        outcome = BuildVerifyOutcome(
            "verify_failed",
            build=None,
            verify=RunnerResult(
                role="verify",
                runner="tests.fully_priced_final_b_gate",
                returncode=1,
                values={"tests_failed": 1.0},
                raw_output=json.dumps({
                    "raw_output": f"repairable failure {gate_calls}",
                }).encode(),
                stderr=b"repairable failure",
                duration_s=0.01,
            ),
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_fully_priced_final_b_gate_stub",
            succeeded=False,
        )
        return outcome

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _failed_gate,
    )
    breach_seeds: list[str] = []

    def _observe_breach_seed(
        argv,
        *,
        fresh_lineage,
        development_budget_ledger,
    ):
        assert fresh_lineage is False
        assert development_budget_ledger is None
        breach_seeds.append((repo / "file_0.py").read_text())
        from crucible.workflows import _mutation_journal

        parent_id = argv[
            argv.index("--parent-mutation-transaction-id") + 1
        ]
        parent = (
            _mutation_journal._require_live_parent_transaction(  # noqa: SLF001
                parent_id
            )
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            _mutation_journal.live_writer_epoch_family(parent),
            phase="test_budget_terminal_breach_observation",
            succeeded=False,
        )
        raise RuntimeError("stop after observing restored breach seed")

    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "main",
        _observe_breach_seed,
    )

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=11.04060975,
        skip_build_verify=False,
        build_retries=2,
        extra_parameters={"deepswe_metadata": {}},
    ) == 1
    assert backend.calls == 3
    assert gate_calls == 2
    assert breach_seeds == ["# implementation 2\n"]
    assert not (repo / "file_0.py").exists()

    summary, result = _tier_documents(spec_dir)
    assert result["status"] == "breach_exception"
    assert summary["phase_1_slices"][0]["classification"] == (
        "verify_failed"
    )
    costs = summary["costs"]
    assert costs["by_role"]["implementer"]["calls"] == 3
    assert costs["total_usd"] == pytest.approx(6.0634355)
    assert costs["accounting_debt"] is False
    assert costs["pricing_evidence"]["complete"] is True

    restoration = summary[
        "build_retry_budget_terminal_restorations"
    ]
    assert len(restoration) == 1
    assert restoration[0]["retry_attempt"] == 2
    assert restoration[0]["terminal_classification"] == (
        "hard_fail_budget_exhausted"
    )
    assert restoration[0]["terminal_bytes_authorized"] is False

    corpus = Corpus(spec_dir)
    implementation = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == "verify_failed"
    # Assay canonicalizes natural completion to the legacy-clean ``None``;
    # Tier admits either None or the explicit ``end_turn`` spelling.
    assert implementation.stop_reason is None
    assert implementation.cost_usd == pytest.approx(1.03540175)
    evidence_ref = implementation.artifacts[
        "build_retry_budget_terminal_restoration"
    ]
    evidence = json.loads(corpus.get_artifact(evidence_ref))
    assert evidence["restored_to_prior_checkpoint"] is True
    assert evidence["terminal_bytes_authorized"] is False
    assert evidence["prior"]["diff_ref"] == str(implementation.diff_ref)
    assert evidence["terminal"]["classification"] == (
        "hard_fail_budget_exhausted"
    )
    assert evidence["terminal"]["cost_usd"] == pytest.approx(0.14497975)
    assert evidence["terminal"]["execution_complete"] is True
    assert evidence["terminal"]["pricing_evidence_complete"] is True
    assert evidence["terminal"]["accounting_debt"] == []
    assert evidence["terminal"]["diff_ref"] != str(implementation.diff_ref)
    prior_diff = corpus.get_artifact(implementation.diff_ref)
    terminal_diff = corpus.get_artifact(
        ContentHash(evidence["terminal"]["diff_ref"])
    )
    assert b"# implementation 2" in prior_diff
    assert b"# implementation 3" in terminal_diff


def test_review_accounting_debt_stops_before_next_paid_rubric(
    tmp_path: Path,
    monkeypatch,
):
    repo, spec_dir, backend = _setup(tmp_path, monkeypatch)
    review_calls: list[str] = []

    def _debt_review(artifact, rubric, *, corpus, **kwargs):  # noqa: ARG001
        review_calls.append(rubric.name)
        if isinstance(artifact, bytes):
            artifact_ref = corpus.put_artifact(
                artifact,
                kind="analysis_inputs",
                name_hint=f"{rubric.name}.bin",
            )
        else:
            artifact_ref = ContentHash(str(artifact))
        return Verdict(
            rubric=rubric.name,
            decision="pass",
            findings=(),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="semantically complete review with unknown accounting",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
            ),
            backend_name="stub-review",
            transcript_ref=None,
            cost_usd=0.02,
            execution=None,
            input_tokens=4,
            output_tokens=2,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=True,
                trace_complete=True,
            ),
        )

    monkeypatch.setattr("crucible.workflows.tier.review", _debt_review)

    assert _run_tier(
        repo,
        spec_dir,
        budget_usd=1.0,
        breach_policy=BreachPolicy(enabled=False),
        review_policy=ReviewPolicy(
            rubrics=("diff_scope", "tier_completeness"),
        ),
    ) == 1
    assert backend.calls == 1
    assert review_calls == ["diff_scope"]
    summary, result = _tier_documents(spec_dir)
    assert summary["status"] == "incomplete_accounting"
    assert [item["rubric"] for item in summary["phase_4_reviews"]] == [
        "diff_scope"
    ]
    assert summary["costs"]["total_usd"] == pytest.approx(0.03)
    assert summary["costs"]["accounting_debt_by_role"] == {
        "diff_scope": [
            "cost_complete",
            "token_usage_complete",
        ],
    }
    assert summary["costs"]["remaining_usd"] == 0.0
    assert result["status"] == "incomplete_accounting"


@pytest.mark.parametrize(
    (
        "implement_cost_usd",
        "expected_reason",
        "expected_remaining_usd",
        "expected_total_usd",
        "api_cost_usd",
        "expected_rc",
    ),
    (
        (
            0.375,
            "insufficient_optional_review_reservation",
            0.125,
            0.875,
            None,
            0,
        ),
        (0.5, "no_optional_review_budget", 0.0, 1.0, None, 0),
        (0.25, None, 0.0, 1.125, 0.375, 1),
    ),
)
def test_advisory_review_budget_softening_preserves_exact_cap(
    tmp_path: Path,
    monkeypatch,
    implement_cost_usd: float,
    expected_reason: str | None,
    expected_remaining_usd: float,
    expected_total_usd: float,
    api_cost_usd: float | None,
    expected_rc: int,
):
    from crucible.workflows import _partition
    from crucible.workflows._development_budget import (
        DevelopmentBudgetLedger,
        DevelopmentBudgetPolicy,
        RoleAllocation,
    )

    repo, spec_dir, _backend = _setup(tmp_path, monkeypatch)
    review_costs = {
        "diff_scope": 0.25,
        "tier_completeness": 0.25,
    }
    resolved_roles: list[str] = []
    review_calls: list[str] = []

    def _pricing(cost_usd: float) -> UsagePricingEvidence:
        return UsagePricingEvidence(
            segments=(UsagePricingSegment(
                input_tokens=1,
                output_tokens=1,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                cost_usd=cost_usd,
            ),),
            complete=True,
        )

    class _CostedBackend:
        name = "stub-advisory-shed"
        model = "stub-model"

        def __init__(self, role: str):
            self.role = role

        def attach_toolset(self, toolset, *, context_budget=None) -> None:
            pass

        def close(self) -> None:
            pass

        def invoke(self, invocation) -> InvocationResult:  # noqa: ARG002
            if self.role == "api_contract_fidelity" and api_cost_usd is None:
                raise AssertionError(
                    "underfunded advisory review crossed provider transport"
                )
            if self.role == "implementer":
                (repo / "file_0.py").write_text("# implementation\n")
                cost_usd = implement_cost_usd
            else:
                review_calls.append(self.role)
                cost_usd = (
                    api_cost_usd
                    if self.role == "api_contract_fidelity"
                    else review_costs[self.role]
                )
                assert cost_usd is not None
            return InvocationResult(
                final_message="done",
                transcript=(),
                usage=Usage(
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=cost_usd,
                ),
                stop_reason="end_turn",
                completeness=ExecutionCompleteness(
                    token_usage_complete=True,
                    cost_complete=True,
                    terminal_complete=True,
                    trace_complete=True,
                ),
                execution_trace=BackendExecutionTrace(
                    backend=self.name,
                    requested_model=self.model,
                    events=(BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),),
                ),
                pricing_evidence=_pricing(cost_usd),
            )

    backends: dict[str, _CostedBackend] = {}

    def _resolve(role, *args, **kwargs):  # noqa: ARG001
        resolved_roles.append(role)
        return backends.setdefault(role, _CostedBackend(role))

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        _resolve,
    )

    def _pass_review(artifact, rubric, *, backend, corpus, **kwargs):
        result = backend.invoke(Invocation(
            system="review",
            user=rubric.name,
            tools=None,
            max_turns=1,
            budget_usd=kwargs["budget_usd"],
        ))
        artifact_ref = (
            corpus.put_artifact(
                artifact,
                kind="analysis_inputs",
                name_hint=f"{rubric.name}.bin",
            )
            if isinstance(artifact, bytes)
            else ContentHash(str(artifact))
        )
        return Verdict(
            rubric=rubric.name,
            decision="pass",
            findings=(),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="mandatory reviewer passed",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
            ),
            backend_name=backend.name,
            transcript_ref=None,
            cost_usd=result.usage.cost_usd,
            execution=None,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            execution_completeness=result.completeness,
            pricing_evidence=result.pricing_evidence,
            cache_creation_tokens=result.usage.cache_creation_tokens,
            cache_read_tokens=result.usage.cache_read_tokens,
        )

    monkeypatch.setattr("crucible.workflows.tier.review", _pass_review)

    policy = DevelopmentBudgetPolicy(
        task_budget_usd=5.0,
        planning_review_cap_usd=2.0,
        implementation_verifier_reserve_usd=2.5,
        contingency_usd=0.5,
        roles=tuple(
            RoleAllocation(
                role=role,
                group="implementation_verifier",
                max_calls=1,
                max_output_tokens=100,
                max_wall_seconds=60.0,
                max_cost_usd=3.0,
                protected_tail_usd=(
                    0.25
                    if role in {"diff_scope", "tier_completeness"}
                    else 0.0
                ),
            )
            for role in (
                "implementer",
                "diff_scope",
                "tier_completeness",
                "api_contract_fidelity",
            )
        ),
    )
    ledger = DevelopmentBudgetLedger(policy)
    review_policy = ReviewPolicy(
        # Deliberately declare the advisory role first: Tier must still run
        # mandatory gates before optional diagnostics.
        rubrics=(
            "api_contract_fidelity",
            "tier_completeness",
            "diff_scope",
        ),
        advisory_rubrics=("api_contract_fidelity",),
    )

    from crucible.workflows.tier import run_tier

    assert run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=1.0,
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        breach_policy=BreachPolicy(enabled=False),
        review_policy=review_policy,
        development_budget_ledger=ledger,
    ) == expected_rc

    summary, result = _tier_documents(spec_dir)
    expected_status = "complete" if expected_rc == 0 else "budget_exhausted"
    assert summary["status"] == result["status"] == expected_status
    assert summary["costs"]["total_usd"] == pytest.approx(
        expected_total_usd
    )
    api_calls = ledger.to_summary()["autonomous_development_metrics"][
        "calls_by_role"
    ]["api_contract_fidelity"]
    if expected_reason is not None:
        assert review_calls == ["tier_completeness", "diff_scope"]
        assert "api_contract_fidelity" not in resolved_roles
        assert summary["phase_4_advisory_skips"] == [{
            "rubric": "api_contract_fidelity",
            "outcome": "skipped",
            "reason": expected_reason,
            "provider_dispatched": False,
            "cost_usd": 0.0,
            "remaining_tier_budget_usd": expected_remaining_usd,
            "available_reservation_usd": expected_remaining_usd,
            "minimum_useful_reservation_usd": 0.25,
            "review_attempt": 0,
        }]
        assert "api_contract_fidelity" not in summary["costs"]["by_role"]
        assert api_calls == 0
    else:
        assert review_calls == [
            "tier_completeness",
            "diff_scope",
            "api_contract_fidelity",
        ]
        assert "phase_4_advisory_skips" not in summary
        assert summary["phase_5_final_bindings"] == []
        assert summary["costs"]["by_role"][
            "api_contract_fidelity"
        ]["cost_usd"] == pytest.approx(api_cost_usd)
        assert api_calls == 1
        return

    corpus = Corpus(spec_dir)
    latest = corpus.latest_spec()
    assert latest is not None
    tier = latest.tier_dag.tiers[0]
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=latest,
    )
    assert pending == ()
    assert set(landed) == {"s0"}


@pytest.mark.parametrize("with_development_ledger", (False, True))
def test_uncapped_gating_failure_sheds_advisory_without_transport(
    with_development_ledger: bool,
):
    from crucible.common.cost import CostAccumulator
    from crucible.workflows._development_budget import (
        DevelopmentBudgetLedger,
        DevelopmentBudgetPolicy,
        RoleAllocation,
    )
    from crucible.workflows.tier import _advisory_review_skip_record

    policy = ReviewPolicy(
        rubrics=("diff_scope", "api_contract_fidelity"),
        advisory_rubrics=("api_contract_fidelity",),
    )
    ledger = None
    if with_development_ledger:
        ledger = DevelopmentBudgetLedger(DevelopmentBudgetPolicy(
            task_budget_usd=1.0,
            planning_review_cap_usd=0.4,
            implementation_verifier_reserve_usd=0.5,
            contingency_usd=0.1,
            roles=(
                RoleAllocation(
                    role="diff_scope",
                    group="implementation_verifier",
                    max_calls=1,
                    max_output_tokens=100,
                    max_wall_seconds=60.0,
                    max_cost_usd=0.6,
                    protected_tail_usd=0.1,
                ),
                RoleAllocation(
                    role="api_contract_fidelity",
                    group="implementation_verifier",
                    max_calls=1,
                    max_output_tokens=100,
                    max_wall_seconds=60.0,
                    max_cost_usd=0.6,
                ),
            ),
        ))

    record = _advisory_review_skip_record(
        policy=policy,
        ordered_rubrics=policy.rubrics,
        rubric_name="api_contract_fidelity",
        costs=CostAccumulator(),
        development_budget_ledger=ledger,
        gating_failure_pending=True,
    )

    assert record is not None
    assert record["reason"] == "gating_review_retry_pending"
    assert record["provider_dispatched"] is False
    assert record["cost_usd"] == 0.0
    assert "remaining_tier_budget_usd" not in record
    if with_development_ledger:
        assert record["available_reservation_usd"] == 0.5
        assert record["minimum_useful_reservation_usd"] == 0.1
    else:
        assert "available_reservation_usd" not in record
        assert "minimum_useful_reservation_usd" not in record
