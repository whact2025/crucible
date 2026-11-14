# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Plan failure envelopes retain only usage returned by completed calls."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from assay import (
    ExecutionCompleteness,
    ImplementationDirective,
    InvocationResult,
    PlanOutcome,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus

from crucible.workflows import _backends
from crucible.workflows import plan as plan_workflow


_COMPLETED_COST_USD = 1.086703
_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=False,
)


def _spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="demo",
            repo_url="https://example.invalid/demo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="implement the thing",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _accepted(spec: Spec) -> PlanOutcome:
    directive = ImplementationDirective(
        slice_id="s0",
        intent="implement the thing",
        requirement_refs=("implement the thing",),
        files_in_scope=("src/thing.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    dag = TierDAG(tiers=(
        Tier(
            tier_id="t0",
            depends_on=(),
            slices=(Slice(id="s0", directive=directive),),
        ),
    ))
    return PlanOutcome(
        spec=replace(spec, tier_dag=dag),
        candidate_dag=dag,
        candidate_ref=None,
        decision="accepted",
        status="accepted",
        rounds_used=1,
        final_critic_decision="pass",
        final_judge_decision="pass",
        reason="test outcome",
        cost_usd=_COMPLETED_COST_USD,
    )


def _completed_result() -> InvocationResult:
    usage = Usage(
        input_tokens=313607,
        output_tokens=23546,
        cost_usd=_COMPLETED_COST_USD,
        cache_read_tokens=263936,
    )
    return InvocationResult(
        final_message="{}",
        transcript=(),
        usage=usage,
        stop_reason="end_turn",
        completeness=_COMPLETE_EXECUTION,
        pricing_evidence=UsagePricingEvidence(
            segments=(
                UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=0,
                    cache_read_tokens=usage.cache_read_tokens,
                    cost_usd=usage.cost_usd,
                ),
            ),
            complete=True,
        ),
    )


class _Planner:
    name = "codex"
    model = "gpt-test"

    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.calls = 0
        self.interrupt_on_call = interrupt_on_call

    def invoke(self, _invocation):
        self.calls += 1
        if self.calls == self.interrupt_on_call:
            interruption = KeyboardInterrupt()
            # Deliberately untrusted: a raised call returned no typed usage.
            # Result-like exception metadata cannot prove whether the paid
            # provider call completed before cancellation.
            interruption.unreported_cost_usd = 99.0
            interruption.returned_result = _completed_result()
            raise interruption
        return _completed_result()


def _install_pool(
    monkeypatch: pytest.MonkeyPatch,
    planner: _Planner,
) -> None:
    class _Pool:
        def get(self, role: str, **_kwargs):
            if role == "plan_planner":
                return planner
            return object()

    @contextmanager
    def _pool(*_args, **_kwargs):
        yield _Pool()

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)


class _InterruptingClosePlanner(_Planner):
    def __init__(
        self,
        *,
        close_interrupt: KeyboardInterrupt,
        interrupt_on_call: int | None = None,
    ) -> None:
        super().__init__(interrupt_on_call=interrupt_on_call)
        self.close_interrupt = close_interrupt
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise self.close_interrupt
        self.closed = True


class _PassiveBackend:
    model = "passive-test"

    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0
        self.closed = False

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def _install_interrupting_real_pool(
    monkeypatch: pytest.MonkeyPatch,
    planner: _InterruptingClosePlanner,
) -> tuple[_PassiveBackend, _PassiveBackend]:
    critic = _PassiveBackend("critic")
    judge = _PassiveBackend("judge")
    routed = {
        "plan_planner": planner,
        "plan_critic": critic,
        "plan_judge": judge,
    }
    monkeypatch.setattr(
        _backends,
        "resolve_for_role",
        lambda role, **_kwargs: routed[role],
    )

    @contextmanager
    def _pool(*_args, **kwargs):
        pool = _backends._BackendPool(
            object(),
            log_dir=kwargs.get("log_dir"),
            cwd=kwargs.get("cwd"),
        )
        try:
            yield pool
        finally:
            pool.close()

    monkeypatch.setattr(plan_workflow, "backend_pool", _pool)
    return critic, judge


def _terminal_envelope(spec_dir: Path) -> tuple[dict, dict, dict]:
    (run_dir,) = (spec_dir / "runs").iterdir()
    return tuple(
        json.loads((run_dir / name).read_text())
        for name in ("summary.json", "result.json", "outcome.json")
    )


def _assert_pool_settled(
    planner: _InterruptingClosePlanner,
    critic: _PassiveBackend,
    judge: _PassiveBackend,
) -> None:
    assert planner.close_calls == 2
    assert planner.closed
    for backend in (critic, judge):
        assert backend.close_calls == 1
        assert backend.closed


def _assert_failed_completed_cost_envelope(spec_dir: Path) -> tuple[dict, dict, dict]:
    summary, result, outcome = _terminal_envelope(spec_dir)
    for terminal in (summary, result, outcome):
        assert terminal["cost_usd"] == pytest.approx(_COMPLETED_COST_USD)
    assert summary["status"] == "failed"
    assert result["status"] == "failed"
    assert outcome["detail_status"] == "failed"
    assert outcome["terminal_class"] == "permanent_failure"
    return summary, result, outcome


def test_plan_interrupt_retains_only_completed_backend_call_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_dir = tmp_path / "corpus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Corpus(spec_dir).put_spec(_spec())
    planner = _Planner(interrupt_on_call=2)
    _install_pool(monkeypatch, planner)

    def _interrupt_during_correction(_spec, **kwargs):
        kwargs["planner_backend"].invoke(object())
        kwargs["planner_backend"].invoke(object())
        raise AssertionError("the interrupted call must not return")

    monkeypatch.setattr(plan_workflow, "run_plan", _interrupt_during_correction)

    with pytest.raises(KeyboardInterrupt):
        plan_workflow.main([
            "--planner-count", "1",
            "--spec", str(spec_dir),
            "--cwd", str(workspace),
            "--no-validate-gate",
        ])

    assert planner.calls == 2
    summary, result, outcome = _terminal_envelope(spec_dir)
    for terminal in (summary, result, outcome):
        assert terminal["cost_usd"] == pytest.approx(_COMPLETED_COST_USD)
    assert summary["error_kind"] == "KeyboardInterrupt"
    assert result["error_kind"] == "KeyboardInterrupt"
    assert outcome["terminal_class"] == "permanent_failure"


def test_plan_commit_failure_retains_completed_plan_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_dir = tmp_path / "corpus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Corpus(spec_dir).put_spec(_spec())
    planner = _Planner()
    _install_pool(monkeypatch, planner)

    def _completed_plan(spec, **kwargs):
        kwargs["planner_backend"].invoke(object())
        return _accepted(spec)

    def _commit_failure(*_args, **_kwargs):
        raise RuntimeError("post-plan commit cleanup failed")

    monkeypatch.setattr(plan_workflow, "run_plan", _completed_plan)
    monkeypatch.setattr(
        plan_workflow,
        "bind_plan_contract",
        lambda _corpus, staged, *, fingerprint: staged,
    )
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )
    monkeypatch.setattr(plan_workflow, "commit_staged_spec", _commit_failure)

    with pytest.raises(RuntimeError, match="post-plan commit cleanup failed"):
        plan_workflow.main([
            "--planner-count", "1",
            "--spec", str(spec_dir),
            "--cwd", str(workspace),
            "--no-validate-gate",
        ])

    assert planner.calls == 1
    summary, result, outcome = _terminal_envelope(spec_dir)
    for terminal in (summary, result, outcome):
        assert terminal["cost_usd"] == pytest.approx(_COMPLETED_COST_USD)
    assert summary["error_kind"] == "RuntimeError"
    assert result["error_message"] == "post-plan commit cleanup failed"
    assert outcome["terminal_class"] == "permanent_failure"


def test_plan_close_interrupt_replaces_unpublished_success_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plan cannot publish success before its paid providers settle."""

    spec_dir = tmp_path / "corpus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Corpus(spec_dir).put_spec(_spec())
    close_interrupt = KeyboardInterrupt()
    planner = _InterruptingClosePlanner(
        close_interrupt=close_interrupt,
    )
    critic, judge = _install_interrupting_real_pool(monkeypatch, planner)

    def _completed_plan(spec, **kwargs):
        kwargs["planner_backend"].invoke(object())
        return _accepted(spec)

    monkeypatch.setattr(plan_workflow, "run_plan", _completed_plan)
    monkeypatch.setattr(
        plan_workflow,
        "bind_plan_contract",
        lambda _corpus, staged, *, fingerprint: staged,
    )
    monkeypatch.setattr(
        plan_workflow,
        "plan_contract_fingerprint",
        lambda *_args, **_kwargs: "sha256:" + "a" * 64,
    )

    commit_calls = 0

    def _commit(*_args, **_kwargs):
        nonlocal commit_calls
        commit_calls += 1
        return _args[1] if len(_args) > 1 else _kwargs["staged"]

    monkeypatch.setattr(
        plan_workflow,
        "commit_staged_spec",
        _commit,
    )

    with pytest.raises(_backends.BackendSettlementInterrupted) as caught:
        plan_workflow.main([
            "--planner-count", "1",
            "--spec", str(spec_dir),
            "--cwd", str(workspace),
            "--no-validate-gate",
        ])

    assert caught.value.interrupt is close_interrupt
    _assert_pool_settled(planner, critic, judge)
    assert commit_calls == 0
    summary, result, _outcome = _assert_failed_completed_cost_envelope(spec_dir)
    assert summary["error_kind"] == "BackendSettlementInterrupted"
    assert "partial_summary" not in summary
    assert result["error_kind"] == "BackendSettlementInterrupted"


@pytest.mark.parametrize("terminal_path", ("cancelled", "rejected"))
def test_plan_close_interrupt_replaces_branch_terminal_before_commit(
    terminal_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every explicit Plan terminal crosses provider settlement first."""

    spec_dir = tmp_path / "corpus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Corpus(spec_dir).put_spec(_spec())
    close_interrupt = KeyboardInterrupt()
    planner = _InterruptingClosePlanner(
        close_interrupt=close_interrupt,
    )
    critic, judge = _install_interrupting_real_pool(monkeypatch, planner)

    def _terminal_plan(spec, **kwargs):
        kwargs["planner_backend"].invoke(object())
        if terminal_path == "cancelled":
            raise plan_workflow.PlannerWorkflowCancelled("cancelled by test")
        return replace(
            _accepted(spec),
            decision="rejected",
            status="rejected",
        )

    commit_calls = 0

    def _commit(*_args, **_kwargs):
        nonlocal commit_calls
        commit_calls += 1
        return _kwargs["staged"]

    monkeypatch.setattr(plan_workflow, "run_plan", _terminal_plan)
    monkeypatch.setattr(plan_workflow, "commit_staged_spec", _commit)

    with pytest.raises(_backends.BackendSettlementInterrupted) as caught:
        plan_workflow.main([
            "--planner-count", "1",
            "--spec", str(spec_dir),
            "--cwd", str(workspace),
            "--no-validate-gate",
        ])

    assert caught.value.interrupt is close_interrupt
    _assert_pool_settled(planner, critic, judge)
    assert commit_calls == 0
    _assert_failed_completed_cost_envelope(spec_dir)


def test_plan_body_and_close_interrupt_preserve_completed_cost_after_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second SIGINT during unwind cannot erase cost or skip settlement."""

    spec_dir = tmp_path / "corpus"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Corpus(spec_dir).put_spec(_spec())
    close_interrupt = KeyboardInterrupt()
    planner = _InterruptingClosePlanner(
        close_interrupt=close_interrupt,
        interrupt_on_call=2,
    )
    critic, judge = _install_interrupting_real_pool(monkeypatch, planner)

    def _interrupt_during_correction(_spec, **kwargs):
        kwargs["planner_backend"].invoke(object())
        kwargs["planner_backend"].invoke(object())
        raise AssertionError("the interrupted call must not return")

    monkeypatch.setattr(plan_workflow, "run_plan", _interrupt_during_correction)

    with pytest.raises(_backends.BackendSettlementInterrupted) as caught:
        plan_workflow.main([
            "--planner-count", "1",
            "--spec", str(spec_dir),
            "--cwd", str(workspace),
            "--no-validate-gate",
        ])

    assert caught.value.interrupt is close_interrupt
    assert planner.calls == 2
    _assert_pool_settled(planner, critic, judge)
    summary, result, _outcome = _assert_failed_completed_cost_envelope(spec_dir)
    assert summary["error_kind"] == "BackendSettlementInterrupted"
    assert result["error_kind"] == "BackendSettlementInterrupted"
