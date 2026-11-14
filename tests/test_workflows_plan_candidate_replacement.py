# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Explicit replacement reviews retain accepted authority on failed promotion."""

from contextlib import contextmanager
from dataclasses import replace
import json

import pytest

from assay import (
    ExecutionCompleteness, Finding, Implementation, LifecycleEvent, ReviewCoverage, Verdict,
)
from assay.canonical import hash_record
from assay.corpus.store import ConcurrentSpecError, Corpus

from crucible.workflows import plan
from crucible.workflows._atomic_persistence import commit_staged_spec
from tests.test_plan_accounting import _result
from tests.test_workflows_plan_candidate_import import _candidate_payload, _spec


@pytest.fixture
def replacement(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "thing.py").write_text("value = 1\n")
    corpus = Corpus(tmp_path / "corpus")
    prior = _candidate_payload()
    prior["tiers"][0]["slices"][0]["directive"]["intent"] = "Retain the prior plan."
    old_dag = plan._strict_review_candidate_dag(prior)
    retained = corpus.put_record(LifecycleEvent(
        kind="planned", reason="Prior accepted plan remains historical authority.",
        affected_versions=(0,), execution=plan.exec_ctx_now("prior_review"),
    ), kind="lifecycle_events")
    old_implementation = corpus.put_record(Implementation(
        slice_id="s0", base_commit="0" * 40, result_commit=None,
        diff_ref=corpus.put_artifact(b"prior implementation only", kind="diffs"),
        transcript_ref=corpus.put_artifact(b"prior attempt", kind="transcripts"),
        files_modified=("src/thing.py",), scope_escapes=(), classification="success",
        execution=plan.exec_ctx_now("prior_implementation"),
    ), kind="implementations")
    base = replace(
        _spec(), tier_dag=old_dag, lifecycle_event_refs=(retained,),
        implementation_refs=(old_implementation,),
    )
    corpus.put_spec(base)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_candidate_payload()))
    args = [
        "--spec", str(corpus.spec_dir), "--cwd", str(workspace),
        "--review-candidate", str(candidate),
        "--replace-plan-digest", str(hash_record(old_dag)),
    ]
    return corpus, base, candidate, args


@pytest.mark.parametrize("change", ["missing", "wrong", "no_old_plan", "malformed"])
def test_replacement_authority_rejected_before_backend(replacement, monkeypatch, change):
    corpus, base, _candidate, args = replacement
    if change == "missing":
        args = args[:-2]
    elif change == "wrong":
        args[-1] = "sha256:" + "0" * 64
    elif change == "no_old_plan":
        base = corpus.commit(transform=lambda spec: replace(spec, tier_dag=None)).spec
    else:
        args[-1] = "not-a-digest"

    @contextmanager
    def unexpected(*_args, **_kwargs):
        pytest.fail("invalid replacement must not acquire a backend")
        yield

    monkeypatch.setattr(plan, "backend_pool", unexpected)
    if change == "malformed":
        with pytest.raises(SystemExit) as caught:
            plan.main(args)
        assert caught.value.code == 2
    else:
        assert plan.main(args) == 2
    assert corpus.latest_spec() == base


def test_replace_flag_cannot_enable_generated_refresh(replacement):
    _, _, _, args = replacement
    del args[4:6]
    with pytest.raises(SystemExit) as caught:
        plan._parse_args(args)
    assert caught.value.code == 2


def test_exact_base_cas_rejects_same_version_different_digest(replacement):
    corpus, base, _, _ = replacement
    # The reviewed base has the current version number but different content.
    # Version equality alone must not authorize publication against this Corpus.
    other_base = replace(base, intent="Different reviewed requirements.")
    with pytest.raises(ConcurrentSpecError, match="refusing to rebase"):
        commit_staged_spec(
            corpus, base=other_base, staged=other_base,
            structural_fields=("tier_dag",), require_exact_base=True,
        )
    assert corpus.latest_spec() == base


@pytest.mark.parametrize(
    "terminal,expected_rc,expected_roles",
    [
        ("accepted", 0, ["critic", "judge"]),
        ("critic_rejected", 2, ["critic", "judge"]),
        ("judge_rejected", 2, ["critic", "judge"]),
        ("pre_review_rejected", 2, []),
        ("late_accounting_debt", 0, ["critic", "judge"]),
    ],
)
def test_real_replacement_adjudication_and_late_debt(
    replacement, monkeypatch, terminal, expected_rc, expected_roles,
):
    corpus, base, candidate, args = replacement
    roles = []
    acquired = []
    replacement_dag = plan._read_review_candidate(candidate)[1]

    class Backend:
        name = "fixture"
        model = "fixture"

        def invoke(self, _invocation):
            return _result(complete=False)

    class Pool:
        def get(self, role, **_kwargs):
            acquired.append(role)
            assert role in {"plan_critic", "plan_judge"}
            return Backend()

    @contextmanager
    def pool(*_args, **_kwargs):
        yield Pool()

    def strict_review(artifact, *positional, **kwargs):
        role = kwargs["kind"]
        roles.append(role)
        assert corpus.latest_spec() == base
        cost = 0.0
        if terminal == "late_accounting_debt":
            # Simulate complete review-level metadata with an independently
            # incomplete actual transport ledger, detected only at terminal.
            backend = kwargs.get("backend") or positional[0]
            cost = backend.invoke(object()).usage.cost_usd
        decision = "revise" if terminal == f"{role}_rejected" else "pass"
        return Verdict(
            rubric=role, decision=decision,
            findings=(Finding(
                id="requires_correction", severity="High", category="scope",
                evidence=(), suggested_action="Review a corrected candidate.", source="llm",
            ),) if decision == "revise" else (),
            artifact_ref=corpus.put_artifact(artifact, kind="reviewed_artifacts"),
            artifact_commit=None, reasoning=f"{role} {decision}",
            coverage=ReviewCoverage(
                complete=True, sources_reviewed=("artifact",), omissions=(),
                artifact_chars_total=len(artifact), artifact_chars_reviewed=len(artifact),
            ),
            backend_name="fixture", transcript_ref=None, cost_usd=cost,
            execution=None, kind=role, cache_creation_tokens=0, cache_read_tokens=0,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True, cost_complete=True,
                terminal_complete=True, trace_complete=True,
            ),
        )

    monkeypatch.setattr(plan, "backend_pool", pool)
    monkeypatch.setattr(plan, "strict_review", strict_review)
    monkeypatch.setattr(plan, "available_rules", lambda: ())
    monkeypatch.setattr(plan, "plan_contract_fingerprint", lambda *_a, **_k: "sha256:" + "a" * 64)
    monkeypatch.setattr(plan, "bind_plan_contract", lambda _corpus, staged, **_kwargs: staged)
    if terminal == "pre_review_rejected":
        monkeypatch.setattr(plan, "build_pre_review_candidate_validator", lambda **_kwargs: lambda _dag: (
            Finding(
                id="scope_mismatch", severity="High", category="scope", evidence=(),
                suggested_action="Correct the scope before review.", source="deterministic",
            ),
        ))

    assert plan.main(args) == expected_rc
    assert roles == expected_roles
    assert acquired == ["plan_critic", "plan_judge"]
    latest = corpus.latest_spec()
    assert latest.version == base.version + 1
    assert latest.parent == hash_record(base)
    assert corpus.get_spec(base.version) == base
    assert latest.lifecycle_event_refs[:len(base.lifecycle_event_refs)] == base.lifecycle_event_refs
    assert latest.implementation_refs == base.implementation_refs
    assert latest.measurement_refs == base.measurement_refs
    assert latest.intent == base.intent
    assert latest.constraints == base.constraints
    assert latest.done_criteria == base.done_criteria
    assert latest.project_context == base.project_context
    accepted = terminal in {"accepted", "late_accounting_debt"}
    assert latest.tier_dag == (replacement_dag if accepted else base.tier_dag)
    events = [corpus.get_record(ref, as_type=LifecycleEvent) for ref in latest.lifecycle_event_refs[1:]]
    assert events[0].kind == ("planned" if accepted else "plan_rejected")
    source = json.loads(events[-1].reason.removeprefix("fixed plan candidate source: "))
    assert source["base_spec_digest"] == hash_record(base)
    assert source["replaces_tier_dag_digest"] == hash_record(base.tier_dag)
    assert source["canonical_tier_dag_digest"] == hash_record(replacement_dag)
    assert corpus.get_artifact(source["raw_candidate_ref"]) == candidate.read_bytes()
    (run,) = (corpus.spec_dir / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["planner_calls"] == 0
    assert summary["status"] != "complete"
    if terminal == "late_accounting_debt":
        assert summary["costs"]["accounting_debt"] is True
        assert summary["cost_usd_is_lower_bound"] is True
        assert summary["costs"]["total_usd"] == pytest.approx(0.2)
        # Published history survives the late hold; no rollback to the old DAG.
        assert corpus.latest_spec() == latest
