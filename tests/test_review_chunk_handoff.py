# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded review coverage across Crucible's typed handoff adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    Evidence,
    ExecutionCompleteness,
    Finding,
    Invocation,
    InvocationResult,
    Message,
    ReviewCoverage,
    Usage,
    Verdict,
)
from assay.corpus.store import Corpus
from assay.primitives.review import (
    MAX_CHUNK_RESPONSE_RETRY_BUDGET,
    _MAX_ARTIFACT_CHARS,
)

from crucible.rubrics import load
from crucible.workflows._analysis_handoff import (
    handoff_artifact,
    parse_review_handoff,
)
from crucible.workflows._review import (
    _context_prior_findings,
    _prior_target_alias,
    review,
)


class _ChunkProtocolBackend:
    name = "chunk-protocol-stub"
    model = "chunk-protocol-model"
    capability = "text"

    def __init__(self) -> None:
        self.invocations: list[Invocation] = []
        self.prior_targets_emitted: list[str] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        request = json.loads(invocation.user)
        if request["schema_version"] == "assay.review.chunk.v1":
            payload = {
                key: request[key]
                for key in (
                    "schema_version",
                    "source_id",
                    "source_sha256",
                    "source_chars",
                    "chunk_index",
                    "chunk_count",
                    "char_start",
                    "char_end",
                    "chunk_sha256",
                    "evidence_addressable",
                )
            }
            findings = []
            if (
                request["source_id"] == "artifact"
                and "stable metric" in request["chunk_text"]
            ):
                findings = [{
                    "id": "current-metric-observation",
                    "criterion_id": "noise",
                    "severity": "Low",
                    "confidence": 1.0,
                    "disposition": "confirmed",
                    "category": "supported",
                    "targets": [],
                    "evidence": [{
                        "source_id": "artifact",
                        "kind": "current_state",
                        "location": "artifact",
                        "excerpt": "stable metric",
                    }],
                    "proposal": {
                        "action": "retain the supported observation",
                        "replacement": None,
                    },
                    "attack_spec": None,
                }]
            payload.update({"summary": "No local ambiguity.", "findings": findings})
        else:
            assert request["schema_version"] == "assay.review.reduce.v1"
            prior_targets = request["required_prior_targets"]
            aliases_by_target = {
                item["finding_ref"]: item["alias"]
                for item in request.get("prior_target_aliases", [])
            }
            sources_by_target = {
                item["finding_ref"]: item["source_id"]
                for item in request.get("prior_target_sources", [])
            }
            emitted_prior_targets = [
                aliases_by_target.get(target, target)
                for target in prior_targets
            ]
            self.prior_targets_emitted = emitted_prior_targets
            report_targets = request["required_targets"]
            payload = {
                "schema_version": "assay.review.v2",
                "decision": "pass",
                "reasoning": "All mapped source ranges are unambiguous.",
                "coverage": {
                    "complete": True,
                    "sources_reviewed": request["required_sources"],
                    "omissions": [],
                },
                "findings": ([{
                    "id": "prior-adjudication",
                    "criterion_id": "noise",
                    "severity": "Low",
                    "confidence": 1.0,
                    "disposition": "addressed",
                    "category": "supported",
                    "targets": [*emitted_prior_targets, *report_targets],
                    "evidence": ([
                        {
                            "source_id": sources_by_target.get(target, target),
                            "kind": "prior_finding_ref",
                            "location": "finding_ref",
                            "excerpt": target,
                        }
                        for target in prior_targets
                    ] + ([{
                        "source_id": "artifact",
                        "kind": "current_state",
                        "location": "artifact",
                        "excerpt": "stable metric",
                    }] if report_targets else [])),
                    "proposal": {
                        "action": "retain the supported observation",
                        "replacement": None,
                    },
                    "attack_spec": None,
                }] if prior_targets else []),
            }
        final = json.dumps(payload)
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.001),
            stop_reason="end_turn",
            error=None,
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=self.model,
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),
                ),
            ),
        )


class _GoalRevisionChunkBackend(_ChunkProtocolBackend):
    """Close mapped goal challenges with one canonical goal revision."""

    def invoke(self, invocation: Invocation) -> InvocationResult:
        result = super().invoke(invocation)
        request = json.loads(invocation.user)
        payload = json.loads(result.final_message)
        reducing = request["schema_version"] == "assay.review.reduce.v1"
        for finding in payload["findings"]:
            finding["criterion_id"] = "red_coverage"
            finding["category"] = (
                "canonical_revision" if reducing else "address_ambiguity"
            )
            if reducing:
                finding["proposal"]["replacement"] = json.dumps({
                    "intent": "Preserve the stable metric.",
                    "constraints": [],
                    "done_criteria": [],
                })
        final = json.dumps(payload)
        return dataclasses.replace(
            result,
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
        )


class _OneShotPassBackend:
    name = "one-shot-pass-stub"
    model = "one-shot-pass-model"
    capability = "text"

    def __init__(self) -> None:
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        final = json.dumps(
            {
                "schema_version": "assay.review.v2",
                "decision": "pass",
                "reasoning": "The supplied artifact is complete.",
                "coverage": {
                    "complete": True,
                    "sources_reviewed": ["artifact"],
                    "omissions": [],
                },
                "findings": [],
            }
        )
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.001),
            stop_reason="end_turn",
            error=None,
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=self.model,
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),
                ),
            ),
        )


def _diff_scope_context() -> dict[str, str]:
    return {
        "files_in_scope": "src/lib.rs\nCargo.lock",
        "creates_paths": "(none)",
        "dependency_provided_files": "(none)",
        "directive_intent": (
            "Add the observation reader; its behavior-neutral rustfmt reflows "
            "and mechanical lockfile update are part of this increment."
        ),
    }


@pytest.mark.parametrize("dynamic_chunk_switch", [False, True])
def test_diff_scope_maps_receive_exact_required_context(
    tmp_path, monkeypatch, dynamic_chunk_switch,
):
    context = _diff_scope_context()
    required_context = dict(context)
    context["unrelated_note"] = "Not part of the directive context selection."
    if dynamic_chunk_switch:
        # Raw sources fit; JSON escaping in the rendered prompt triggers the
        # late switch. The mapper envelope remains its ordinary 200K bound.
        render_limit = 10_000
        monkeypatch.setitem(review.__globals__, "_MAX_ARTIFACT_CHARS", render_limit)
        assay_review = review.__globals__["_assay_review"]
        monkeypatch.setitem(
            assay_review.__globals__, "_MAX_ARTIFACT_CHARS", render_limit,
        )
        context["directive_intent"] += "\x00" * 1_800
        required_context["directive_intent"] = context["directive_intent"]
        artifact = b"diff --git a/src/lib.rs b/src/lib.rs\n" + b"A" * 1_000
        assert len(artifact) + sum(map(len, context.values())) < render_limit
    else:
        artifact = b"diff --git a/src/lib.rs b/src/lib.rs\n" + (
            b"A" * _MAX_ARTIFACT_CHARS
        )

    class ContextCheckingBackend(_ChunkProtocolBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            request = json.loads(invocation.user)
            if request["schema_version"] == "assay.review.chunk.v1":
                if request["source_id"] == "artifact":
                    companions = request["map_context_sources"]
                    assert companions == [
                        {
                            "source_id": f"context:{key}",
                            "source_sha256": (
                                "sha256:" + hashlib.sha256(text.encode()).hexdigest()
                            ),
                            "source_chars": len(text),
                            "text": text,
                        }
                        for key, text in required_context.items()
                    ]
                else:
                    assert "map_context_sources" not in request
            return super().invoke(invocation)

    backend = ContextCheckingBackend()
    corpus = Corpus(tmp_path / "corpus")
    verdict = review(
        artifact, load("diff_scope"), backend=backend, corpus=corpus,
        context=context,
    )

    assert verdict.decision == "pass"
    assert verdict.coverage.complete
    assert verdict.coverage.artifact_chars_reviewed == len(artifact)
    requests = [json.loads(item.user) for item in backend.invocations]
    maps = [
        item for item in requests
        if item["schema_version"] == "assay.review.chunk.v1"
    ]
    assert {item["source_id"] for item in maps} >= {
        "artifact", *(f"context:{key}" for key in required_context),
    }
    assert set(verdict.coverage.sources_reviewed) == {
        "artifact", *(f"context:{key}" for key in required_context),
    }
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_source_manifest"],
    ))
    assert manifest["map_context_sources"] == [
        {
            "source_id": f"context:{key}",
            "source_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            "source_chars": len(text),
        }
        for key, text in required_context.items()
    ]


def test_diff_scope_single_prompt_remains_single_prompt(tmp_path):
    context = _diff_scope_context()

    class DiffScopePassBackend(_OneShotPassBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            payload = json.loads(result.final_message)
            payload["coverage"]["sources_reviewed"] = [
                "artifact", *(f"context:{key}" for key in context),
            ]
            final = json.dumps(payload)
            return dataclasses.replace(
                result, final_message=final,
                transcript=(
                    Message(role="user", content=invocation.user),
                    Message(role="assistant", content=final),
                ),
            )

    backend = DiffScopePassBackend()
    verdict = review(
        b"diff --git a/src/lib.rs b/src/lib.rs\n", load("diff_scope"),
        backend=backend, corpus=Corpus(tmp_path / "corpus"), context=context,
    )
    assert verdict.decision == "pass"
    assert len(backend.invocations) == 1
    assert "map_context_sources" not in backend.invocations[0].user
    assert "review_source_manifest" not in verdict.artifacts


def test_other_rubrics_do_not_repeat_context_in_maps(tmp_path):
    backend = _ChunkProtocolBackend()
    verdict = review(
        b"A" * (_MAX_ARTIFACT_CHARS + 1), load("spec_clarity"),
        backend=backend, corpus=Corpus(tmp_path / "corpus"),
        context={"operator_note": "Do not auto-select optional context."},
    )
    assert verdict.decision == "pass"
    assert all(
        "map_context_sources" not in json.loads(invocation.user)
        for invocation in backend.invocations
    )


@pytest.mark.parametrize("rubric_name", ["diff_scope", "spec_clarity"])
def test_map_context_selection_is_adapter_owned(tmp_path, rubric_name):
    backend = _ChunkProtocolBackend()
    with pytest.raises(TypeError, match="map_context_source_ids is adapter-owned"):
        review(
            b"diff", load(rubric_name), backend=backend,
            corpus=Corpus(tmp_path / "corpus"), context=_diff_scope_context(),
            map_context_source_ids=(),
        )
    assert backend.invocations == []


class _InvalidChunkProtocolBackend(_ChunkProtocolBackend):
    """Inject one strict map or reducer protocol violation."""

    def __init__(self, failure_stage: str) -> None:
        super().__init__()
        self.failure_stage = failure_stage

    def invoke(self, invocation: Invocation) -> InvocationResult:
        result = super().invoke(invocation)
        request = json.loads(invocation.user)
        schema = request["schema_version"]
        payload = json.loads(result.final_message)
        if (
            self.failure_stage == "chunk"
            and schema == "assay.review.chunk.v1"
        ):
            payload["chunk_sha256"] = "sha256:" + "0" * 64
        elif (
            self.failure_stage == "reducer"
            and schema == "assay.review.reduce.v1"
        ):
            payload["schema_version"] = "assay.review.invalid.v1"
        else:
            return result
        final = json.dumps(payload)
        return dataclasses.replace(
            result,
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
        )


def test_response_retry_budget_rejects_above_assay_bound_before_invocation(
    tmp_path,
):
    backend = _OneShotPassBackend()

    with pytest.raises(
        ValueError,
        match=rf"integer in \[0, {MAX_CHUNK_RESPONSE_RETRY_BUDGET}\]",
    ):
        review(
            b"bounded artifact",
            load("spec_clarity"),
            backend=backend,
            corpus=Corpus(tmp_path / "corpus"),
            response_retry_budget=MAX_CHUNK_RESPONSE_RETRY_BUDGET + 1,
        )

    assert backend.invocations == []


class _DynamicJudgeChunkBackend(_ChunkProtocolBackend):
    """Adjudicate the exact material-prior ledger in terminal reduction."""

    name = "openai_compat"
    model = "nvidia/zai-org/glm-5.2"

    def __init__(self) -> None:
        super().__init__()
        self.reduce_request: dict[str, object] | None = None

    def invoke(self, invocation: Invocation) -> InvocationResult:
        request = json.loads(invocation.user)
        result = super().invoke(invocation)
        if request["schema_version"] != "assay.review.reduce.v1":
            return result

        self.reduce_request = request
        (target,) = request["required_prior_targets"]
        (alias_record,) = request["prior_target_aliases"]
        (source_record,) = request["prior_target_sources"]
        assert alias_record["finding_ref"] == target
        assert source_record == {
            "finding_ref": target,
            "source_id": "context:prior_critic_findings",
        }
        (report_target,) = request["required_targets"]
        artifact_evidence = {
            "source_id": "artifact",
            "kind": "current_state",
            "location": "artifact",
            "excerpt": "stable metric",
        }
        payload = {
            "schema_version": "assay.review.v2",
            "decision": "revise",
            "reasoning": "The active blocker remains repairable.",
            "coverage": {
                "complete": True,
                "sources_reviewed": request["required_sources"],
                "omissions": [],
            },
            "findings": [{
                "id": "active-blocker-adjudication",
                "criterion_id": "executability",
                "severity": "High",
                "confidence": 1.0,
                "disposition": "confirmed",
                "category": "other",
                "targets": [alias_record["alias"], report_target],
                "evidence": [
                    artifact_evidence,
                    {
                        "source_id": "context:prior_critic_findings",
                        "kind": "source_quote",
                        "location": "active frontier",
                        "excerpt": "active-blocker",
                    },
                ],
                "proposal": {
                    "action": "Repair the active blocker before implementation.",
                    "replacement": None,
                },
                "attack_spec": None,
            }],
            "criterion_results": [
                {
                    "criterion_id": criterion,
                    "outcome": (
                        "violated"
                        if criterion == "executability"
                        else "satisfied"
                    ),
                    "reasoning": "The current TierDAG was checked directly.",
                    "evidence": [artifact_evidence],
                    "headline": criterion == "executability",
                }
                for criterion in (
                    "intent_coverage",
                    "executability",
                    "critic_disposition",
                    "contradiction",
                )
            ],
        }
        final = json.dumps(payload)
        return dataclasses.replace(
            result,
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
        )


def test_compact_handoff_prompts_original_artifact_and_retains_prior_sources(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    artifact = b"CURRENT_DIFF_ONLY"
    artifact_ref = corpus.put_artifact(artifact, kind="test_inputs")
    prior = Verdict(
        rubric="spec_clarity",
        decision="revise",
        findings=(
            Finding(
                id="prior-finding",
                criterion_id="intent_clarity",
                severity="Medium",
                confidence=1.0,
                disposition="confirmed",
                category="ambiguity",
                targets=(),
                evidence=(
                    Evidence(
                        source_id="artifact",
                        kind="current_state",
                        location="artifact",
                        excerpt="old intent",
                    ),
                ),
                proposal=None,
                attack_spec=None,
                suggested_action=None,
                source="llm",
            ),
        ),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="PRIVATE_PRIOR_REASONING",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.decode()),
            artifact_chars_reviewed=len(artifact.decode()),
        ),
        backend_name="prior-stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )
    handoff = handoff_artifact(artifact, (("defender", prior),))
    parsed = parse_review_handoff(handoff)
    assert parsed is not None
    assert parsed.prior_findings[0]["source_id"] in parsed.source_texts
    backend = _OneShotPassBackend()

    verdict = review(
        handoff,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        compact_handoff_prompt=True,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 1
    prompt = backend.invocations[0].user
    assert "CURRENT_DIFF_ONLY" in prompt
    assert "crucible.review-input.v2" not in prompt
    assert "PRIVATE_PRIOR_REASONING" not in prompt


def test_compact_handoff_rejects_untyped_artifact(tmp_path):
    with pytest.raises(
        ValueError,
        match="compact_handoff_prompt requires a typed review handoff",
    ):
        review(
            b"ordinary artifact",
            load("spec_clarity"),
            backend=_OneShotPassBackend(),
            corpus=Corpus(tmp_path / "corpus"),
            compact_handoff_prompt=True,
        )


def test_chunked_handoff_keeps_original_artifact_as_evidence(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    artifact = ("clear requirement\n" + "A" * _MAX_ARTIFACT_CHARS).encode()
    artifact_ref = corpus.put_artifact(artifact, kind="test_inputs")
    prior = Verdict(
        rubric="spec_clarity",
        decision="pass",
        findings=(),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="No finding.",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.decode()),
            artifact_chars_reviewed=len(artifact.decode()),
        ),
        backend_name="prior-stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )
    handoff = handoff_artifact(artifact, [("red", prior)])
    backend = _ChunkProtocolBackend()

    verdict = review(
        handoff,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "pass"
    assert verdict.coverage.sources_reviewed == ("artifact",)
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    sources = {item["source_id"]: item for item in manifest["sources"]}
    assert sources["artifact"]["source_sha256"] == (
        "sha256:" + hashlib.sha256(artifact).hexdigest()
    )
    assert sources["artifact"]["evidence_addressable"] is True
    assert sources["artifact"]["terminal_required"] is True
    assert sources["harness:prompt_envelope"]["evidence_addressable"] is False
    assert sources["harness:prompt_envelope"]["terminal_required"] is False
    assert all(
        chunk["status"] == "reviewed"
        for source in sources.values()
        for chunk in source["chunks"]
    )
    assert len(backend.invocations) >= 4


def test_chunked_review_degrades_anchor_only_to_chunk_ledger(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    artifact = ("clear requirement\n" + "A" * _MAX_ARTIFACT_CHARS).encode()
    backend = _ChunkProtocolBackend()

    verdict = review(
        artifact,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        review_anchor_policy="anchors_only",
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) >= 3
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" not in invocation.system
        for invocation in backend.invocations
    )
    schemas = {
        json.loads(invocation.user)["schema_version"]
        for invocation in backend.invocations
    }
    assert schemas == {"assay.review.chunk.v1", "assay.review.reduce.v1"}
    obligations = json.loads(corpus.get_artifact(
        verdict.artifacts["review_obligations"]
    ))
    assert obligations["obligations"]["anchor_policy_resolution"] == {
        "requested": "anchors_only",
        "effective": "anchors",
        "reason": "chunk_protocol",
    }


def test_plan_judge_dynamic_chunk_fallback_keeps_material_prior_ledger(
    tmp_path,
    monkeypatch,
):
    """A post-render chunk switch must retain the Judge's exact alias join."""

    render_limit = 10_000
    assay_review = review.__globals__["_assay_review"]
    monkeypatch.setitem(
        review.__globals__, "_MAX_ARTIFACT_CHARS", render_limit
    )
    monkeypatch.setitem(
        assay_review.__globals__, "_MAX_ARTIFACT_CHARS", render_limit
    )
    artifact = json.dumps({
        "tiers": [{
            "tier_id": "t0",
            "depends_on": [],
            "slices": [{
                "id": "stable-metric",
                "directive": {
                    "slice_id": "stable-metric",
                    "intent": "Preserve the stable metric.",
                    "files_in_scope": ["src/metric.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            }],
        }],
    }, indent=2).encode()
    context = {
        "spec_intent": "Preserve the stable metric.",
        "spec_constraints": "[]",
        "spec_done_criteria": "The stable metric remains available.",
        "prior_critic_findings": json.dumps([{
            "round": 0,
            "decision": "revise",
            "source": "critic",
            "findings": [{
                "id": "active-blocker",
                "criterion_id": "executability",
                "severity": "High",
                "confidence": 1.0,
                "disposition": "confirmed",
                "category": "other",
                "targets": [],
                "evidence": [],
                "proposal": None,
                "attack_spec": None,
            }],
        }]),
        "prior_judge_findings": "[]",
    }
    retained_chars = len(artifact.decode()) + sum(map(len, context.values()))
    assert retained_chars < render_limit
    (prior,) = _context_prior_findings(dict(context))
    prior_ref = prior["ref"]
    alias = _prior_target_alias(prior_ref)
    backend = _DynamicJudgeChunkBackend()
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        artifact,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=context,
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert backend.reduce_request is not None
    assert backend.reduce_request["required_prior_targets"] == [prior_ref]
    assert backend.reduce_request["prior_target_sources"] == [{
        "finding_ref": prior_ref,
        "source_id": "context:prior_critic_findings",
    }]
    assert backend.reduce_request["prior_target_aliases"] == [{
        "alias": alias,
        "finding_ref": prior_ref,
    }]
    assert set(backend.reduce_request["required_sources"]) == {
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    }
    assert sum(
        target == prior_ref
        for finding in verdict.findings
        for target in finding.targets
    ) == 1
    rendered = json.loads(corpus.get_artifact(verdict.rendered_prompt_ref))
    assert rendered["protocol"] == "bounded_chunk_map_terminal_reduce"
    assert rendered["response_repair_degradation"]["reason"] == (
        "rendered_prompt_requires_chunk_protocol"
    )


@pytest.mark.parametrize(
    ("failure_stage", "finding_id"),
    [
        ("chunk", "chunk_schema_failure"),
        ("reducer", "reducer_schema_failure"),
    ],
)
def test_chunk_protocol_validation_failure_uses_only_ledger_local_retry(
    tmp_path,
    failure_stage,
    finding_id,
):
    corpus = Corpus(tmp_path / failure_stage)
    backend = _InvalidChunkProtocolBackend(failure_stage)
    artifact = ("clear requirement\n" + "A" * _MAX_ARTIFACT_CHARS).encode()

    verdict = review(
        artifact,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == finding_id
    requests = [json.loads(item.user) for item in backend.invocations]
    chunk_coordinates = [
        (
            item["source_id"],
            item["chunk_index"],
            item["char_start"],
            item["char_end"],
        )
        for item in requests
        if item["schema_version"] == "assay.review.chunk.v1"
    ]
    if failure_stage == "chunk":
        assert len(chunk_coordinates) == 3
        assert len(set(chunk_coordinates)) == 1
        assert [
            item.get("response_repair", {}).get("attempt")
            for item in requests
            if item["schema_version"] == "assay.review.chunk.v1"
        ] == [None, 1, 2]
    else:
        assert len(chunk_coordinates) == len(set(chunk_coordinates))
    assert sum(
        item["schema_version"] == "assay.review.reduce.v1"
        for item in requests
    ) <= 1
    assert verdict.cost_usd == pytest.approx(0.001 * len(backend.invocations))
    assert verdict.accounting_debt == ()

    retry = json.loads(
        corpus.get_artifact(verdict.artifacts["review_retry_manifest"])
    )
    assert retry["terminal_reason"] == (
        "chunk_protocol_outer_retry_disabled"
    )
    assert len(retry["attempts"]) == 1
    assert retry["attempts"][0]["validation_error"]
    assert retry["total_cost_usd"] == pytest.approx(verdict.cost_usd)
    telemetry = json.loads(
        corpus.get_artifact(verdict.artifacts["review_call_telemetry"])
    )
    assert telemetry["terminal_reason"] == (
        "chunk_protocol_outer_retry_disabled"
    )
    assert len(telemetry["attempts"]) == 1
    assert telemetry["total_cost_usd"] == pytest.approx(verdict.cost_usd)
    assert telemetry["accounting_debt"] == []


def test_chunk_protocol_repairs_one_invalid_map_without_replaying_prefix(
    tmp_path,
):
    class _InvalidOnceBackend(_InvalidChunkProtocolBackend):
        def __init__(self) -> None:
            super().__init__("chunk")
            self.invalidated = False

        def invoke(self, invocation: Invocation) -> InvocationResult:
            if not self.invalidated:
                self.invalidated = True
                return super().invoke(invocation)
            return _ChunkProtocolBackend.invoke(self, invocation)

    corpus = Corpus(tmp_path / "repair")
    backend = _InvalidOnceBackend()
    artifact = ("clear requirement\n" + "A" * _MAX_ARTIFACT_CHARS).encode()

    verdict = review(
        artifact,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    requests = [json.loads(item.user) for item in backend.invocations]
    chunk_requests = [
        item
        for item in requests
        if item["schema_version"] == "assay.review.chunk.v1"
    ]
    assert chunk_requests[0]["chunk_index"] == 0
    assert chunk_requests[1]["chunk_index"] == 0
    assert chunk_requests[1]["response_repair"]["attempt"] == 1
    assert chunk_requests[0]["chunk_sha256"] == chunk_requests[1][
        "chunk_sha256"
    ]
    assert all(
        "response_repair" not in item
        for item in chunk_requests[2:]
    )


def test_rendered_prompt_chunking_repairs_current_map_without_replaying_prefix(
    tmp_path,
):
    class _InvalidSecondChunkOnceBackend(_ChunkProtocolBackend):
        def __init__(self) -> None:
            super().__init__()
            self.invalidated = False

        def invoke(self, invocation: Invocation) -> InvocationResult:
            result = super().invoke(invocation)
            request = json.loads(invocation.user)
            if (
                not self.invalidated
                and request["schema_version"] == "assay.review.chunk.v1"
                and request["chunk_index"] == 1
            ):
                self.invalidated = True
                payload = json.loads(result.final_message)
                payload["chunk_sha256"] = "sha256:" + "0" * 64
                final = json.dumps(payload)
                return dataclasses.replace(
                    result,
                    final_message=final,
                    transcript=(
                        Message(role="user", content=invocation.user),
                        Message(role="assistant", content=final),
                    ),
                )
            return result

    # The retained source is only 40K raw characters, while JSON renders every
    # NUL as six characters. Assay therefore dynamically selects chunking only
    # after rendering its trust frame; Crucible's raw-source estimate cannot
    # predict the adaptive 31,809-character first map boundary.
    artifact = ("\0" * 40_000).encode()
    corpus = Corpus(tmp_path / "rendered-overflow")
    backend = _InvalidSecondChunkOnceBackend()

    verdict = review(
        artifact,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    requests = [json.loads(item.user) for item in backend.invocations]
    chunk_requests = [
        item
        for item in requests
        if item["schema_version"] == "assay.review.chunk.v1"
    ]
    chunk_indices = [item["chunk_index"] for item in chunk_requests]
    assert chunk_indices.count(0) == 1
    assert chunk_indices.count(1) == 2
    repaired = [item for item in chunk_requests if item["chunk_index"] == 1]
    assert "response_repair" not in repaired[0]
    assert repaired[1]["response_repair"]["attempt"] == 1
    assert sum(
        item["schema_version"] == "assay.review.reduce.v1"
        for item in requests
    ) == 1
    manifest = json.loads(
        corpus.get_artifact(verdict.artifacts["review_source_manifest"])
    )
    assert manifest["map_retry_policy"] == {
        "scope": "review",
        "authorized_retries": 2,
        "consumed_retries": 1,
    }


def test_rendered_prompt_chunking_does_not_replay_after_reducer_failure(
    tmp_path,
):
    artifact = ("\0" * 40_000).encode()
    corpus = Corpus(tmp_path / "rendered-reducer-failure")
    backend = _InvalidChunkProtocolBackend("reducer")

    verdict = review(
        artifact,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "reducer_schema_failure"
    requests = [json.loads(item.user) for item in backend.invocations]
    chunk_coordinates = [
        (
            item["source_id"],
            item["chunk_index"],
            item["char_start"],
            item["char_end"],
        )
        for item in requests
        if item["schema_version"] == "assay.review.chunk.v1"
    ]
    assert len(chunk_coordinates) == len(set(chunk_coordinates))
    assert sum(
        item["schema_version"] == "assay.review.reduce.v1"
        for item in requests
    ) == 1
    retry = json.loads(
        corpus.get_artifact(verdict.artifacts["review_retry_manifest"])
    )
    assert retry["terminal_reason"] == "chunk_protocol_outer_retry_disabled"
    assert len(retry["attempts"]) == 1


def test_rendered_prompt_chunking_uses_full_material_prior_targets(
    tmp_path,
):
    corpus = Corpus(tmp_path / "dynamic-prior-targets")
    artifact = ("stable metric\n" + "\0" * 40_000).encode()
    artifact_ref = corpus.put_artifact(artifact, kind="test_inputs")
    prior = Verdict(
        rubric="goal_red",
        decision="revise",
        findings=(Finding(
            id="same-id",
            criterion_id="intent_clarity",
            severity="High",
            confidence=0.9,
            disposition="confirmed",
            category="ambiguity",
            targets=(),
            evidence=(Evidence(
                source_id="artifact",
                kind="metric_excerpt",
                location="artifact",
                excerpt="stable metric",
            ),),
            proposal=None,
            attack_spec=None,
            suggested_action=None,
            source="llm",
        ),),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="Goal ambiguity needs adjudication.",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.decode()),
            artifact_chars_reviewed=len(artifact.decode()),
        ),
        backend_name="prior-stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )
    handoff = handoff_artifact(artifact, [("red", prior)])
    backend = _GoalRevisionChunkBackend()

    verdict = review(
        handoff,
        load("goal_blue"),
        backend=backend,
        corpus=corpus,
        compact_handoff_prompt=True,
        review_anchor_policy="anchors",
    )

    assert verdict.decision == "pass"
    requests = [json.loads(item.user) for item in backend.invocations]
    reducer = next(
        item
        for item in requests
        if item["schema_version"] == "assay.review.reduce.v1"
    )
    (prior_ref,) = reducer["required_prior_targets"]
    assert prior_ref.startswith("prior:red:0:sha256-")
    assert all(
        not target.startswith("harness-obligation:prior-")
        for finding in verdict.findings
        for target in finding.targets
    )
    assert prior_ref in verdict.findings[0].targets
    obligations = json.loads(
        corpus.get_artifact(verdict.artifacts["review_obligations"])
    )["obligations"]
    (alias_record,) = obligations["material_prior_aliases"]
    assert alias_record["finding_ref"] == prior_ref
    assert reducer["prior_target_aliases"] == [{
        "alias": alias_record["alias"],
        "finding_ref": prior_ref,
    }]
    assert backend.prior_targets_emitted == [alias_record["alias"]]
    reducer_invocation = next(
        item
        for item in backend.invocations
        if json.loads(item.user)["schema_version"]
        == "assay.review.reduce.v1"
    )
    assert "sole entry in exactly one" in reducer_invocation.system


@pytest.mark.parametrize("compact_handoff_prompt", [False, True])
def test_chunked_closure_preserves_qualified_prior_ref_and_evidence(
    tmp_path,
    compact_handoff_prompt,
):
    corpus = Corpus(tmp_path / f"corpus-{compact_handoff_prompt}")
    artifact = ("stable metric\n" + "A" * _MAX_ARTIFACT_CHARS).encode()
    artifact_ref = corpus.put_artifact(artifact, kind="test_inputs")
    prior = Verdict(
        rubric="goal_red",
        decision="revise",
        findings=(Finding(
            id="same-id",
            criterion_id="intent_clarity",
            severity="High",
            confidence=0.9,
            disposition="confirmed",
            category="ambiguity",
            targets=(),
            evidence=(Evidence(
                source_id="artifact",
                kind="metric_excerpt",
                location="artifact",
                excerpt="stable metric",
            ),),
            proposal=None,
            attack_spec=None,
            suggested_action=None,
            source="llm",
        ),),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="Goal ambiguity needs adjudication.",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
            artifact_chars_total=len(artifact.decode()),
            artifact_chars_reviewed=len(artifact.decode()),
        ),
        backend_name="prior-stub",
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
    )
    handoff = handoff_artifact(artifact, [("red", prior)])
    backend = _GoalRevisionChunkBackend()

    verdict = review(
        handoff,
        load("goal_blue"),
        backend=backend,
        corpus=corpus,
        compact_handoff_prompt=compact_handoff_prompt,
    )

    assert verdict.decision == "pass"
    (finding,) = verdict.findings
    (prior_ref,) = (
        target for target in finding.targets if target.startswith("prior:")
    )
    assert prior_ref.startswith("prior:red:0:sha256-")
    assert {item.source_id for item in finding.evidence} == {
        "artifact", prior_ref,
    }
    if compact_handoff_prompt:
        chunk_requests = [
            json.loads(invocation.user)
            for invocation in backend.invocations
            if json.loads(invocation.user)["schema_version"] == "assay.review.chunk.v1"
        ]
        assert {request["source_id"] for request in chunk_requests} >= {
            "artifact",
            prior_ref,
        }
        assert all(
            "crucible.review-input.v2" not in request["chunk_text"]
            for request in chunk_requests
        )
