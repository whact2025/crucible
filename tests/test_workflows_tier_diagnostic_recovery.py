# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Diagnostic max-turn accounting cannot authorize another recovery call."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    InvocationResult,
    Implementation,
    MaxTurnsExceeded,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from crucible.backends.features import feature_profile_for_route
from crucible.backends.limits import (
    InvocationLimitContractError,
    InvocationLimitIdentity,
    bind_invocation_limit_policy,
    invocation_limit_terminal,
)
from crucible.common.policies import BreachPolicy
from tests.test_workflows_tier_empty_diff_retry import _COMPLETE_EXECUTION, _run


class _SettledMaxTurnsBackend:
    name = "kimi"
    model = "kimi-k2.7-code-highspeed"

    def __init__(self, *, mutate, completeness):
        self.repo = None
        self.mutate = mutate
        self.completeness = completeness
        self.invocations = []
        self.recovery_path = "target.py"

    def attach_toolset(self, toolset, *, context_budget=None):
        pass

    def close(self):
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        initial = len(self.invocations) == 1
        if self.mutate or not initial:
            self.repo.joinpath("target.py" if initial else self.recovery_path).write_text(
                "value = partial\n" if initial else "value = recovered\n",
                encoding="utf-8",
            )
        usage = Usage(input_tokens=13, output_tokens=4, cost_usd=0.25)
        stop = "max_turns" if initial else "end_turn"
        return InvocationResult(
            final_message="" if initial else "DONE",
            transcript=(),
            usage=usage,
            stop_reason=stop,
            error=MaxTurnsExceeded(message="logical step limit") if initial else None,
            completeness=self.completeness if initial else _COMPLETE_EXECUTION,
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=self.model,
                events=(
                    BackendExecutionEvent(sequence=0, kind="request"),
                    BackendExecutionEvent(sequence=1, kind="terminal", name=stop),
                ),
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    cost_usd=usage.cost_usd,
                ),),
                complete=True,
            ),
        )


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("incomplete_field", ["terminal_complete", "trace_complete"])
def test_diagnostic_incomplete_execution_stops_before_recovery(
    tmp_path, monkeypatch, mutate, incomplete_field,
):
    """Use the real shared latch, not a stub that silently accepts recovery."""

    backend = _SettledMaxTurnsBackend(
        mutate=mutate,
        completeness=replace(_COMPLETE_EXECUTION, **{incomplete_field: False}),
    )
    states = []

    def resolve(role, **kwargs):
        states.append(kwargs["limit_state"])
        bind_invocation_limit_policy(
            backend,
            feature_profile_for_route(
                backend.name, model=backend.model,
                base_url="https://api.moonshot.ai/v1",
            ),
            kwargs["limit_policy"],
            route=f"{backend.name}/{backend.model} for role '{role}'",
            shared_state=kwargs["limit_state"],
            evidence_ledger=kwargs["limit_ledger"],
            identity=InvocationLimitIdentity(
                role=role, backend=backend.name, model=backend.model,
            ),
        )
        return backend

    from crucible.workflows import tier

    def unexpected_gate(*args, **kwargs):
        pytest.fail("a failed diagnostic cannot enter deterministic or paid gates")

    monkeypatch.setattr(tier._build_verify, "build_and_verify", unexpected_gate)
    monkeypatch.setattr(tier.breach_workflow, "main", unexpected_gate)
    monkeypatch.setattr(
        tier.breach_workflow, "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend,
        backend_resolver=resolve,
        inherited_dirty=True,
        skip_build_verify=False,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
        backend_limit_mode="diagnostic",
        backend_limit_deadline_seconds=1200,
        budget_usd=12,
        max_turns=90,
    )

    assert rc == 1
    assert summary["status"] == "slice_failed"
    assert len(backend.invocations) == len(states) == 1
    assert backend.invocations[0].max_turns == 90
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    assert summary["costs"]["total_usd"] == 0.25
    assert summary["costs"]["accounting_debt"] is False
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_max_turns"
    recovery = phase["incomplete_terminal_recovery"]
    if mutate:
        assert recovery == {
            "classification": "hard_fail_max_turns",
            "restored_paths": ["target.py"],
            "restored_to_pre_call_snapshot": True,
            "retry_attempted": False,
        }
    else:
        assert recovery is None

    terminal = invocation_limit_terminal(backend)
    assert states[0].revocation is terminal
    assert terminal.terminal_reason == "limit_evidence_incomplete"
    assert terminal.observed_usage.cost_complete is True
    assert terminal.observed_usage.token_usage_complete is True
    assert terminal.physical_attempts == 1
    with pytest.raises(InvocationLimitContractError, match="no recovery provider call"):
        states[0].require_call_authority()

    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    assert len(spec.implementation_refs) == 1
    implementation = corpus.get_record(spec.implementation_refs[0], as_type=Implementation)
    assert implementation.classification == "hard_fail_max_turns"
    assert implementation.stop_reason == "max_turns"
    assert implementation.input_tokens == 13
    assert implementation.output_tokens == 4
    assert implementation.cost_usd == 0.25
    assert implementation.accounting_debt == ()
    assert implementation.execution_completeness == backend.completeness
    assert implementation.pricing_evidence.complete is True
    if mutate:
        assert "incomplete_terminal_failed_diff" in implementation.artifacts
    settlement = tier.attest_mutation_journal_settlement(
        corpus=corpus, cwd=repo, tier_id="t0", operation_id="diagnostic-recovery-test",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []
    run_dir = next(path for path in (corpus.spec_dir / "runs").iterdir() if "tier" in path.name)
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["terminal_class"] != "success"
    ledger = json.loads((run_dir / "invocation-limits.json").read_text())
    assert len(ledger["records"]) == 1


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_legacy_or_complete_diagnostic_recovery_remains_available(
    tmp_path, monkeypatch, mutate, diagnostic,
):
    """Neither legacy incomplete evidence nor complete diagnostics is tightened."""

    completeness = (
        _COMPLETE_EXECUTION if diagnostic
        else replace(_COMPLETE_EXECUTION, terminal_complete=False)
    )
    backend = _SettledMaxTurnsBackend(mutate=mutate, completeness=completeness)
    if mutate:
        # The rollback remains exact while the recovery completes a declared
        # create, matching the established post-mutation recovery fixture.
        backend.recovery_path = "generated.py"
    limit_kwargs = (
        {"backend_limit_mode": "diagnostic", "backend_limit_deadline_seconds": 1200}
        if diagnostic else {}
    )
    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend, budget_usd=12, max_turns=90,
        creates_paths=("generated.py",) if mutate else (), **limit_kwargs,
    )
    assert rc == 0
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == (24 if mutate else 8)
    assert summary["costs"]["total_usd"] == 0.5
    assert summary["costs"]["accounting_debt"] is False
    assert repo.joinpath(backend.recovery_path).read_text() == "value = recovered\n"
    if mutate:
        assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert summary["phase_1_slices"][0]["classification"].startswith("success")


@pytest.mark.parametrize("completeness", [
    None,
    replace(_COMPLETE_EXECUTION, cost_complete=False),
    replace(_COMPLETE_EXECUTION, token_usage_complete=False),
])
def test_missing_or_unpriced_diagnostic_evidence_cannot_authorize_recovery(completeness):
    from types import SimpleNamespace
    from crucible.workflows.tier import _has_complete_diagnostic_recovery_evidence

    assert not _has_complete_diagnostic_recovery_evidence(
        SimpleNamespace(execution_completeness=completeness), limit_mode="diagnostic",
    )
