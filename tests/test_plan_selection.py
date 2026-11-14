# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hash-bound competition selection is evidence, never plan publication."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from assay import (
    ExecutionCompleteness, Invocation, InvocationResult, Message, ProjectContext,
    ReviewCoverage, Spec, Usage, UsagePricingEvidence, UsagePricingSegment, Verdict,
)
from assay.canonical import hash_record, serialize
from assay.corpus.store import Corpus
from assay.primitives.plan import _serialize_tier_dag

from crucible.workflows._plan_selection import (
    PlanSelectionCandidate, select_plan_candidate,
)
from crucible.workflows.plan import _decode_review_candidate


def _dag(name: str = "s0"):
    return _decode_review_candidate(json.dumps({
        "tiers": [{
            "tier_id": "t0", "depends_on": [], "slices": [{
                "id": name, "directive": {
                    "slice_id": name, "intent": "implement the thing",
                    "requirement_refs": ["implement the thing"],
                    "files_in_scope": ["src/thing.py"], "creates_paths": [],
                    "verify_mode": "tests", "blast_radius": {"max_files": 1, "languages": ["python"]},
                },
            }],
        }],
    }).encode())


def _spec() -> Spec:
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="demo", repo_url="https://example.invalid/demo.git",
            base_commit="0" * 40, branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="implement the thing", constraints=(),
        done_criteria=(), plan_policy=None, depends_on=(), tier_dag=None,
        implementation_refs=(), verdict_refs=(), measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _candidate(
    corpus: Corpus, name: str, *, accepted: bool = True,
) -> PlanSelectionCandidate:
    dag = _dag(name)
    artifact_ref = corpus.put_artifact(_serialize_tier_dag(dag), kind="plan_candidates")
    refs = []
    for role in ("critic", "judge", "validate"):
        verdict = Verdict(
            rubric=f"plan_{role}", decision="pass" if accepted else "reject",
            findings=(), artifact_ref=artifact_ref, artifact_commit="0" * 40,
            reasoning=f"{name} {role} evidence", coverage=ReviewCoverage(
                complete=True, sources_reviewed=("artifact",), omissions=(),
            ),
            backend_name=None if role == "validate" else "stub",
            transcript_ref=None, cost_usd=0.0, execution=None, kind=role,
            input_tokens=1 if role != "validate" else 0,
            output_tokens=1 if role != "validate" else 0,
            cache_creation_tokens=0 if role != "validate" else None,
            cache_read_tokens=0 if role != "validate" else None,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True, cost_complete=True,
                terminal_complete=True, trace_complete=False,
            ) if role != "validate" else None,
        )
        refs.append(corpus.put_record(verdict, kind="verdicts"))
    return PlanSelectionCandidate(name, dag, accepted, tuple(refs))


class _Backend:
    name = "selector-stub"
    model = "selector-model"
    capability = "text"

    def __init__(
        self, *changes: dict[str, Any], raw_replacement: str | None = None,
        omit_evidence_prefix: str | None = None,
    ) -> None:
        self.changes = changes
        self.raw_replacement = raw_replacement
        self.omit_evidence_prefix = omit_evidence_prefix
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        envelope, _ = json.JSONDecoder().raw_decode(
            invocation.user[invocation.user.index("{"):]
        )
        source = envelope.get("text")
        if source is None:
            source = "".join(item["text"] for item in envelope["anchors"])
        packet = json.loads(source)
        candidate = packet["bundle"]["candidates"][0]
        selection = {
            "schema_version": "crucible.plan-selection.v1",
            "source_spec_hash": packet["bundle"]["source_spec_hash"],
            "bundle_ref": packet["bundle_ref"], "mode": "select",
            "selected_id": candidate["id"], "selected_digest": candidate["digest"],
            "synthesized_dag": None, "reason": "This plan covers the root goal.",
        }
        change = self.changes[min(len(self.invocations), len(self.changes)) - 1] if self.changes else {}
        selection.update(change)
        mode = selection["mode"]
        category = {"select": "selected", "synthesize": "synthesized", "none": "no_selection"}[mode]
        evidence = [{
            "source_id": "artifact", "kind": "source_quote",
            "location": "json:$.bundle.spec.intent",
            "excerpt": packet["bundle"]["spec"]["intent"],
        }]
        for index, item in enumerate(packet["bundle"]["candidates"]):
            prefix = f"json:$.bundle.candidates[{index}]"
            evidence.append({
                "source_id": "artifact", "kind": "source_quote",
                "location": prefix + ".dag.tiers[0].slices[0].directive.intent",
                "excerpt": item["dag"]["tiers"][0]["slices"][0]["directive"]["intent"],
            })
            for role in ("critic", "judge"):
                review_index = max(
                    i for i, entry in enumerate(item["reviews"])
                    if entry["verdict"]["kind"] == role
                )
                evidence.append({
                    "source_id": "artifact", "kind": "source_quote",
                    "location": prefix + f".reviews[{review_index}].verdict.reasoning",
                    "excerpt": item["reviews"][review_index]["verdict"]["reasoning"],
                })
        if self.omit_evidence_prefix is not None:
            evidence = [
                item for item in evidence
                if not item["location"].startswith(self.omit_evidence_prefix)
            ]
        payload = {
            "schema_version": "assay.review.v2",
            "decision": "inconclusive" if mode == "none" else "pass",
            "reasoning": "Compared both complete plans and their review evidence.",
            "coverage": {
                "complete": True, "sources_reviewed": ["artifact"], "omissions": [],
            },
            "criterion_results": [{
                "criterion_id": "plan_choice",
                "headline": True,
                "outcome": "inconclusive" if mode == "none" else "satisfied",
                "reasoning": "Compared the hash-bound plan bundle.",
                "evidence": evidence,
            }],
            "findings": [{
                "id": "plan_selection", "criterion_id": "plan_choice",
                "severity": "Low", "category": category, "confidence": 1.0,
                "disposition": "addressed", "targets": [], "attack_spec": None,
                "evidence": evidence,
                "proposal": {
                    "action": "Choose the complete source-grounded plan.",
                    "replacement": self.raw_replacement or json.dumps(selection),
                },
            }],
        }
        text = json.dumps(payload)
        sequence = len(self.invocations)
        usage = Usage(input_tokens=10 * sequence, output_tokens=sequence, cost_usd=.01 * sequence)
        return InvocationResult(
            final_message=text,
            transcript=(Message(role="assistant", content=text),),
            usage=usage, stop_reason="end_turn", error=None,
            completeness=ExecutionCompleteness(
                token_usage_complete=True, cost_complete=True,
                terminal_complete=True, trace_complete=False,
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                    cache_creation_tokens=0, cache_read_tokens=0, cost_usd=usage.cost_usd,
                ),), complete=True,
            ),
        )


@pytest.mark.parametrize("same_dag", [False, True])
def test_exact_selection_retains_all_evidence_without_publishing_spec(tmp_path, same_dag) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    candidates = (_candidate(corpus, "a"), _candidate(corpus, "b"))
    if same_dag:
        candidates = (candidates[0], replace(candidates[0], id="b"))
    backend = _Backend()

    result = select_plan_candidate(spec, candidates, backend=backend, corpus=corpus)

    assert result.error is None
    assert result.mode == "select" and result.selected_id == "a"
    assert result.candidate_dag == candidates[0].dag
    assert len(backend.invocations) == 1
    assert hash_record(corpus.latest_spec()) == hash_record(spec)
    bundle = json.loads(corpus.get_artifact(result.bundle_ref))
    assert bundle["source_spec_hash"] == hash_record(spec)
    assert [item["digest"] for item in bundle["candidates"]] == [hash_record(item.dag) for item in candidates]
    assert bundle["candidates"][1]["reviews"][1]["verdict"]["reasoning"] == (
        "a judge evidence" if same_dag else "b judge evidence"
    )
    persisted = json.loads(corpus.get_artifact(result.selection_ref))
    assert persisted["verdict_ref"] == result.verdict_ref


@pytest.mark.parametrize("wrong", [
    {"selected_digest": "sha256:" + "f" * 64},
    {"selected_id": "unknown"},
    {"source_spec_hash": "sha256:" + "f" * 64},
    {"bundle_ref": "sha256:" + "f" * 64},
    {"extra": "untrusted"},
    {"synthesized_dag": {"tiers": []}},
    {"mode": "none"},
])
def test_invalid_selection_is_repaired_with_exact_aggregate_usage(tmp_path, wrong) -> None:
    corpus = Corpus(tmp_path)
    backend = _Backend(wrong, {})

    result = select_plan_candidate(
        _spec(), (_candidate(corpus, "a"),), backend=backend, corpus=corpus,
        budget_usd=1,
    )

    assert result.error is None and result.mode == "select"
    assert len(backend.invocations) == 2
    assert result.cost_usd == pytest.approx(.03)
    assert result.verdict.input_tokens == 30
    assert result.verdict.output_tokens == 3
    assert result.verdict.pricing_evidence is not None
    assert result.verdict.pricing_evidence.complete
    assert len(result.verdict.pricing_evidence.segments) == 2


def test_rejected_branch_cannot_be_selected_and_diagnostics_survive(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    backend = _Backend()

    result = select_plan_candidate(
        _spec(), (_candidate(corpus, "a", accepted=False),),
        backend=backend, corpus=corpus,
    )

    assert result.mode == "none" and result.candidate_dag is None
    assert result.error is not None
    assert result.verdict.decision == "inconclusive"
    assert len(backend.invocations) == 3
    assert result.cost_usd == pytest.approx(.06)
    assert "review_retry_manifest" in result.verdict.artifacts


def test_prose_cannot_grant_selection_authority(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    backend = _Backend(raw_replacement="Choose a because it is the best complete plan.")

    result = select_plan_candidate(
        _spec(), (_candidate(corpus, "a"),), backend=backend, corpus=corpus,
    )

    assert result.mode == "none" and result.candidate_dag is None
    assert result.error is not None and len(backend.invocations) == 3


@pytest.mark.parametrize("missing", [
    "json:$.bundle.spec", "json:$.bundle.candidates[1].dag",
    "json:$.bundle.candidates[0].reviews[0]",
])
def test_comparison_must_cite_root_all_plans_and_review_evidence(tmp_path, missing) -> None:
    corpus = Corpus(tmp_path)
    backend = _Backend(omit_evidence_prefix=missing)

    result = select_plan_candidate(
        replace(_spec(), intent="implement the thing; preserve café output"),
        (_candidate(corpus, "a"), _candidate(corpus, "b")),
        backend=backend, corpus=corpus,
    )

    assert result.mode == "none" and result.candidate_dag is None
    assert result.error is not None and len(backend.invocations) == 3


def test_duplicate_candidate_ids_are_rejected_before_selection(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    candidate = _candidate(corpus, "a")
    backend = _Backend()

    with pytest.raises(ValueError, match="ids must be unique"):
        select_plan_candidate(
            _spec(), (candidate, candidate), backend=backend, corpus=corpus,
        )
    assert not backend.invocations


def test_synthesis_is_full_typed_proposal_and_never_published(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    spec = _spec()
    corpus.put_spec(spec)
    synthesized = _dag("combined")
    backend = _Backend({
        "mode": "synthesize", "selected_id": None, "selected_digest": None,
        "synthesized_dag": json.loads(_serialize_tier_dag(synthesized)),
    })

    result = select_plan_candidate(
        spec, (_candidate(corpus, "a", accepted=False),),
        backend=backend, corpus=corpus,
    )

    assert result.mode == "synthesize" and result.error is None
    assert result.candidate_dag == synthesized and result.selected_id is None
    assert corpus.latest_spec().tier_dag is None


def test_partial_synthesis_fails_closed_after_bounded_repair(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    backend = _Backend({
        "mode": "synthesize", "selected_id": None, "selected_digest": None,
        "synthesized_dag": {"patch": "change a slice"},
    })

    result = select_plan_candidate(
        _spec(), (_candidate(corpus, "a"),), backend=backend, corpus=corpus,
    )

    assert result.mode == "none" and result.candidate_dag is None
    assert result.error is not None and len(backend.invocations) == 3


def test_typed_abstention_is_a_valid_diagnostic_result(tmp_path) -> None:
    corpus = Corpus(tmp_path)
    backend = _Backend({
        "mode": "none", "selected_id": None, "selected_digest": None,
        "synthesized_dag": None,
    })

    result = select_plan_candidate(
        _spec(), (_candidate(corpus, "a"),), backend=backend, corpus=corpus,
    )

    assert result.mode == "none" and result.error is None
    assert result.candidate_dag is None and len(backend.invocations) == 1


@pytest.mark.parametrize("problem", [
    "missing", "wrong-dag", "latest-rejected", "latest-inconclusive",
    "incomplete-execution",
])
def test_missing_or_mismatched_current_review_evidence_blocks_paid_selection(tmp_path, problem) -> None:
    corpus = Corpus(tmp_path)
    candidate = _candidate(corpus, "a")
    if problem == "missing":
        candidate = replace(candidate, verdict_refs=())
    elif problem == "wrong-dag":
        candidate = replace(candidate, dag=_dag("another"))
    elif problem in {"latest-rejected", "latest-inconclusive"}:
        judge = corpus.get_record(candidate.verdict_refs[1], as_type=Verdict)
        decision = "reject" if problem == "latest-rejected" else "inconclusive"
        ref = corpus.put_record(replace(judge, decision=decision), kind="verdicts")
        candidate = replace(candidate, verdict_refs=(*candidate.verdict_refs, ref))
    else:
        judge = corpus.get_record(candidate.verdict_refs[1], as_type=Verdict)
        ref = corpus.put_record(
            replace(judge, execution_completeness=None), kind="verdicts",
        )
        candidate = replace(candidate, verdict_refs=(*candidate.verdict_refs, ref))
    backend = _Backend()

    result = select_plan_candidate(_spec(), (candidate,), backend=backend, corpus=corpus)

    assert result.mode == "none" and result.error is not None
    assert result.candidate_dag is None and not backend.invocations
    assert result.cost_usd == 0
    assert corpus.get_artifact(result.selection_ref)
