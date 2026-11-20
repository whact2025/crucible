# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Restored-path handoff cannot upgrade K3-series diagnostic recovery authority."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from assay import BackendExecutionEvent, LifecycleEvent, MaxTurnsExceeded, Message
from assay.corpus.store import Corpus
from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY
from crucible.backends.features import feature_profile_for_route
from crucible.backends.limits import (
    InvocationLimitContractError,
    InvocationLimitIdentity,
    bind_invocation_limit_policy,
    invocation_limit_terminal,
)
from crucible.workflows import tier
from tests.test_backend_limits_cost_bounds import _bound_result
from tests.test_workflow_diagnostic_cost import _DiagnosticWriter
from tests.test_workflows_tier_empty_diff_retry import (
    _contract_aware_build_verify_stub,
    _run,
    _successful_gate_outcome,
)


@pytest.mark.parametrize("known_read", [False, True])
@pytest.mark.parametrize("incomplete_terminal", [False, True])
def test_same_path_restore_cannot_override_k3_terminal_revocation(
    tmp_path, monkeypatch, known_read, incomplete_terminal,
):
    class Writer(_DiagnosticWriter):
        effort = "max"
        diagnostic_cost_upper_bound = True

        def invoke(self, invocation):
            result = super().invoke(invocation)
            initial = len(self.invocations) == 1
            self.repo.joinpath("target.py").write_text(
                "value = partial\n" if initial else "value = recovered\n",
            )
            if not initial:
                return result
            return replace(
                result, final_message="scoped work unfinished",
                transcript=(Message(role="assistant", content="scoped work unfinished"),),
                stop_reason="max_turns", error=MaxTurnsExceeded(message="fixture bounded terminal"),
                execution_trace=replace(result.execution_trace, events=(
                    BackendExecutionEvent(sequence=0, kind="request"),
                    BackendExecutionEvent(sequence=1, kind="terminal", name="max_turns"),
                )),
            )

    backend = Writer(known_read=known_read, incomplete_terminal=incomplete_terminal)

    def resolve(role, **kwargs):
        bind_invocation_limit_policy(
            backend,
            feature_profile_for_route("kimi", model="kimi-k3.1-pro", base_url="https://api.moonshot.ai/v1"),
            kwargs["limit_policy"], route=f"kimi/kimi-k3.1-pro for role '{role}'",
            shared_state=kwargs["limit_state"], evidence_ledger=kwargs["limit_ledger"],
            identity=InvocationLimitIdentity(role=role, backend="kimi", model="kimi-k3.1-pro"),
        )
        return backend

    monkeypatch.setattr(
        tier._build_verify, "build_and_verify",
        _contract_aware_build_verify_stub(lambda *args, **kwargs: _successful_gate_outcome()),
    )
    def run():
        return _run(
            tmp_path, monkeypatch, backend, backend_resolver=resolve,
            inherited_dirty=True, skip_build_verify=False, max_turns=90, budget_usd=12,
            diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
            backend_limit_mode="diagnostic", backend_limit_deadline_seconds=1200,
        )

    if known_read and not incomplete_terminal:
        # Complete counters cannot override the bound policy's clean-end-turn
        # requirement. The shared latch rejects before a second physical call.
        with pytest.raises(InvocationLimitContractError, match="no recovery provider call"):
            run()
    else:
        rc, summary, _repo = run()
        assert rc == 1
        assert summary["status"] != "diagnostic_completed"
        recovery = summary["phase_1_slices"][0]["incomplete_terminal_recovery"]
        assert recovery["restored_to_pre_call_snapshot"] is True
        assert recovery["retry_attempted"] is False
    assert len(backend.invocations) == 1
    repo = tmp_path / "repo"
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    terminal = invocation_limit_terminal(backend)
    assert terminal.revokes_diagnostic_authority
    assert not terminal.diagnostic_cost_authority
    assert terminal.physical_attempts == 1
    run_dir = next(path for path in (tmp_path / "corpus/runs").iterdir() if "_tier_" in path.name)
    ledger = json.loads((run_dir / "invocation-limits.json").read_text())
    assert len(ledger["records"]) == 1
    assert ledger["records"][0]["state"] == "terminal"
    assert ledger["records"][0]["terminal"] == terminal.to_record()
    _assert_not_landed_and_settled(tmp_path, repo)


def _assert_not_landed_and_settled(tmp_path, repo):
    corpus = Corpus(tmp_path / "corpus")
    lifecycle = [corpus.get_record(ref, as_type=LifecycleEvent)
                 for ref in corpus.latest_spec().lifecycle_event_refs]
    assert not any(event.kind == "landed" for event in lifecycle)
    settlement = tier.attest_mutation_journal_settlement(
        corpus=corpus, cwd=repo, tier_id="t0", operation_id="k3-same-path-compatibility",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []


@pytest.mark.parametrize("known_read", [False, True])
def test_clean_k3_scope_repair_rewrites_only_admitted_restored_paths(
    tmp_path, monkeypatch, known_read,
):
    class Writer(_DiagnosticWriter):
        effort = "max"
        diagnostic_cost_upper_bound = True

        def invoke(self, invocation):
            result = super().invoke(invocation)
            if len(self.invocations) == 1:
                self.repo.joinpath("outside.py").write_text("scope escape\n")
            else:
                self.repo.joinpath("target.py").write_text("value = recovered\n")
            return result

    backend = Writer(known_read=known_read)

    def resolve(role, **kwargs):
        bind_invocation_limit_policy(
            backend,
            feature_profile_for_route("kimi", model="kimi-k3.1-pro", base_url="https://api.moonshot.ai/v1"),
            kwargs["limit_policy"], route=f"kimi/kimi-k3.1-pro for role '{role}'",
            shared_state=kwargs["limit_state"], evidence_ledger=kwargs["limit_ledger"],
            identity=InvocationLimitIdentity(role=role, backend="kimi", model="kimi-k3.1-pro"),
        )
        return backend

    monkeypatch.setattr(
        tier._build_verify, "build_and_verify",
        _contract_aware_build_verify_stub(lambda *args, **kwargs: _successful_gate_outcome()),
    )
    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend, backend_resolver=resolve,
        inherited_dirty=True, committed_files={"outside.py": "original\n"},
        skip_build_verify=False, max_turns=90, budget_usd=12,
        diagnostic_cost_policy=DIAGNOSTIC_COST_POLICY,
        backend_limit_mode="diagnostic", backend_limit_deadline_seconds=1200,
    )
    assert rc == 1 and summary["status"] == "diagnostic_completed"
    assert summary["rankable"] is summary["promotable"] is False
    assert len(backend.invocations) == 2
    assert backend.invocations[1].budget_usd == pytest.approx(
        12 - _bound_result(known_read=known_read).cost_bounds.upper_bound_usd,
    )
    assert repo.joinpath("target.py").read_text() == "value = recovered\n"
    assert repo.joinpath("outside.py").read_text() == "original\n"
    run = next(path for path in (tmp_path / "corpus/runs").iterdir() if "_tier_" in path.name)
    ledger = json.loads((run / "invocation-limits.json").read_text())
    assert len(ledger["records"]) == 2
    for record in ledger["records"]:
        terminal = record["terminal"]["terminal"]
        assert terminal["terminal_reason"] == "diagnostic_cost_upper_bound"
        assert terminal["observed_usage"]["cost_complete"] is known_read
        assert terminal["rankable"] is False
    _assert_not_landed_and_settled(tmp_path, repo)
