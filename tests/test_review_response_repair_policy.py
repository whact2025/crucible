# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crucible rollout policy for Assay's deterministic review repair."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from assay import (
    AttackSpec,
    BackendExecutionEvent,
    BackendExecutionTrace,
    Evidence,
    ExecutionCompleteness,
    Finding,
    Invocation,
    InvocationResult,
    Message,
    Proposal,
    ReviewCoverage,
    Usage,
    Verdict,
)
from assay.corpus.store import Corpus
from assay.primitives.review import ReviewResponseRepairPolicy

from crucible.rubrics import load
from crucible.workflows._analysis_handoff import (
    handoff_artifact,
    parse_review_handoff,
)
from crucible.workflows._review import (
    _anchor_only_catalog_is_bounded,
    _assay_response_repair_policy,
    _effective_source_anchor_catalog_available,
    _expand_material_prior_aliases,
    _obligations_rubric,
    _plan_anchor_only_repair,
    _material_prior_target_repair_ledger,
    _review_obligations_envelope,
    review,
)


_ARTIFACT = b"The retry limit is ambiguous."
_JSON_ARTIFACT_TEXT = (
    '{"phase":"post_fix","records":[{"status":"passed"}]}'
)
_ANCHOR_RE = re.compile(r'"location":"(anchor:sha256:[0-9a-f]{64})"')
_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="review-response-repair-policy-stub",
    requested_model="review-response-repair-policy-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="end_turn",
        ),
    ),
)


@pytest.fixture
def assay_review_module():
    return importlib.import_module("assay.primitives.review")


def _anchor_limit_context():
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )
    return intent, {
        "spec_intent": intent,
        "spec_constraints": "C" * 2_500,
        "spec_done_criteria": "D" * 500,
    }


@dataclass
class _AnchorResponseBackend:
    model: str
    supports_review_evidence_anchors: bool = False
    invalid_first_contract: bool = False
    name: str = "repair-policy-stub"
    capability: str = "text"
    invocations: list[Invocation] = field(default_factory=list)

    def invoke(self, invocation: Invocation) -> InvocationResult:
        attempt = len(self.invocations)
        self.invocations.append(invocation)
        match = _ANCHOR_RE.search(invocation.user)
        location = (
            match.group(1)
            if match is not None
            else "anchor:sha256:" + "0" * 64
        )
        response = json.dumps({
            "schema_version": "assay.review.v2",
            "decision": (
                "pass"
                if self.invalid_first_contract and attempt == 0
                else "revise"
            ),
            "reasoning": "The limit needs a precise value.",
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": [{
                "id": "ambiguous-limit",
                "criterion_id": "intent_clarity",
                "severity": "High",
                "confidence": 0.95,
                "disposition": "confirmed",
                "category": "ambiguity",
                "targets": [],
                "evidence": [{
                    "source_id": "artifact",
                    "kind": "source_quote",
                    "location": location,
                    "excerpt": "model placeholder",
                }],
                "proposal": {
                    "action": "State the exact retry limit.",
                    "replacement": None,
                },
                "attack_spec": None,
            }],
        })
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


@dataclass
class _JsonQuoteEscapedResponseBackend:
    """Claude-like backend that adds one quote-escape layer to evidence."""

    model: str = "claude-sonnet-4-5"
    name: str = "json-quote-escaped-stub"
    capability: str = "text"
    supports_review_evidence_anchors: bool = False
    invocations: list[Invocation] = field(default_factory=list)

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        escaped_excerpt = _JSON_ARTIFACT_TEXT.replace('"', r'\"')
        response = json.dumps({
            "schema_version": "assay.review.v2",
            "decision": "revise",
            "reasoning": "The record lacks a required outcome definition.",
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": [{
                "id": "missing-outcome-definition",
                "criterion_id": "intent_clarity",
                "severity": "High",
                "confidence": 0.99,
                "disposition": "confirmed",
                "category": "ambiguity",
                "targets": [],
                "evidence": [{
                    "source_id": "artifact",
                    "kind": "source_quote",
                    "location": "post-fix record",
                    "excerpt": escaped_excerpt,
                }],
                "proposal": {
                    "action": "Define the required post-fix outcome.",
                    "replacement": None,
                },
                "attack_spec": None,
            }],
        })
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


@dataclass
class _BreachJudgeJsonQuoteEscapedBackend:
    """R5-shaped Judge: three Medium closures and one escaped JSON quote."""

    prior_refs: tuple[str, ...]
    model: str = "claude-sonnet-4-5"
    name: str = "breach-judge-json-quote-escaped-stub"
    capability: str = "text"
    supports_review_evidence_anchors: bool = False
    invocations: list[Invocation] = field(default_factory=list)

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        escaped_excerpt = _JSON_ARTIFACT_TEXT.replace('"', r'\"')
        findings = [{
            "id": f"attack-{index}-survived",
            "criterion_id": "attack_execution",
            "severity": "Medium",
            "confidence": 0.99,
            "disposition": "addressed",
            "category": "surviving_test",
            "targets": [prior_ref],
            "evidence": [{
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "post-fix records",
                "excerpt": escaped_excerpt,
            }],
            "proposal": {
                "action": "Retain the passing attack as a regression test.",
                "replacement": None,
            },
            "attack_spec": None,
        } for index, prior_ref in enumerate(self.prior_refs)]
        criterion_results = [{
            "criterion_id": criterion_id,
            "outcome": "satisfied",
            "reasoning": "The complete attack ledger supplies this result.",
            "evidence": [{
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "attack ledger",
                "excerpt": _JSON_ARTIFACT_TEXT,
            }],
            "headline": index == 0,
        } for index, criterion_id in enumerate(
            load("breach_judge").required_criteria
        )]
        response = json.dumps({
            "schema_version": "assay.review.v2",
            "decision": "pass",
            "reasoning": "Every admitted attack has a passing post-fix result.",
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": findings,
            "criterion_results": criterion_results,
        })
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(input_tokens=100, output_tokens=100, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


@dataclass
class _PlanCitationTypoBackend:
    model: str = "gpt-5.6-sol"
    name: str = "plan-citation-typo-stub"
    capability: str = "text"
    expect_anchor_retry: bool = True
    valid_first_response: bool = False
    mistyped_anchor_first_response: bool = False
    whitespace_normalized_first_response: bool = False
    json_path_then_whitespace_anchor_retry: bool = False
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    invocations: list[Invocation] = field(default_factory=list)

    @staticmethod
    def _anchor(invocation: Invocation, source_id: str) -> str:
        for line in invocation.user.splitlines():
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if frame.get("source_id") == source_id and frame.get("anchors"):
                return frame["anchors"][0]["location"]
        raise AssertionError(f"missing deterministic anchors for {source_id}")

    @staticmethod
    def _criterion_evidence(
        invocation: Invocation,
        source_id: str,
    ) -> dict[str, str]:
        for line in invocation.user.splitlines():
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(frame, dict) or frame.get("source_id") != source_id:
                continue
            if (
                frame.get("schema_version") == "assay.untrusted-source.v1"
                and isinstance(frame.get("text"), str)
                and frame["text"]
            ):
                excerpt = frame["text"][:8]
                return {
                    "source_id": source_id,
                    "kind": "source_quote",
                    "location": f"chars:0-{len(excerpt)}",
                    "excerpt": excerpt,
                }
            if frame.get("schema_version") == (
                "assay.untrusted-source.anchored.v1"
            ):
                anchor = next(
                    (
                        item
                        for item in frame.get("anchors", ())
                        if isinstance(item, dict)
                        and isinstance(item.get("location"), str)
                        and isinstance(item.get("text"), str)
                        and item["text"]
                    ),
                    None,
                )
                if anchor is not None:
                    return {
                        "source_id": source_id,
                        "kind": "source_quote",
                        "location": anchor["location"],
                        "excerpt": anchor["text"][:8],
                    }
        raise AssertionError(f"missing retained source for {source_id}")

    @classmethod
    def _criterion_results(cls, invocation: Invocation) -> list[dict]:
        artifact = cls._criterion_evidence(invocation, "artifact")
        results = []
        for criterion_id in load("plan_critic").required_criteria:
            evidence = [dict(artifact)]
            if criterion_id == "intent_coverage":
                evidence.append(cls._criterion_evidence(
                    invocation,
                    "context:spec_intent",
                ))
            elif criterion_id == "done_criteria_coverage":
                evidence.append(cls._criterion_evidence(
                    invocation,
                    "context:spec_done_criteria",
                ))
            results.append({
                "criterion_id": criterion_id,
                "outcome": (
                    "violated"
                    if criterion_id == "prerequisites"
                    else "satisfied"
                ),
                "reasoning": "Fixture-backed criterion adjudication.",
                "evidence": evidence,
                "headline": criterion_id == "prerequisites",
            })
        return results

    def invoke(self, invocation: Invocation) -> InvocationResult:
        attempt = len(self.invocations)
        self.invocations.append(invocation)
        if attempt == 0 and self.mistyped_anchor_first_response:
            intent = (
                "The Spec-bound runner, not workflow CLI arguments,\n"
                "owns the fixed targets."
            )
            evidence = [{
                "source_id": "context:spec_intent",
                "kind": "requirement",
                "location": "anchor:sha256:" + "0" * 64,
                "excerpt": intent,
            }]
        elif attempt == 0 and self.json_path_then_whitespace_anchor_retry:
            evidence = [{
                "source_id": "context:spec_intent",
                "kind": "requirement",
                "location": "json:$.missing",
                "excerpt": (
                    "The Spec-bound runner, not workflow CLI arguments,\n"
                    "owns the fixed targets."
                ),
            }]
        elif (
            attempt > 0
            and self.json_path_then_whitespace_anchor_retry
        ):
            evidence = [{
                "source_id": "context:spec_intent",
                "kind": "requirement",
                "location": "runner requirement",
                "excerpt": (
                    "The Spec-bound runner, not workflow CLI arguments, "
                    "owns the fixed targets."
                ),
            }]
        elif attempt == 0 and self.whitespace_normalized_first_response:
            evidence = [{
                "source_id": "context:spec_intent",
                "kind": "requirement",
                "location": "runner requirement",
                "excerpt": (
                    "The Spec-bound runner, not workflow CLI arguments, "
                    "owns the fixed targets."
                ),
            }]
        elif attempt == 0 and not self.valid_first_response:
            evidence = [{
                "source_id": "context:spec_intent",
                "kind": "requirement",
                "location": "runner requirement",
                "excerpt": (
                    "The Spec-bound runner, not workflow CLI arguments,\n"
                    "own the fixed targets."
                ),
            }]
        else:
            if not self.expect_anchor_retry:
                intent = (
                    "The Spec-bound runner, not workflow CLI arguments,\n"
                    "owns the fixed targets."
                )
                evidence = [{
                    "source_id": "context:spec_intent",
                    "kind": "requirement",
                    "location": f"chars:0-{len(intent)}",
                    "excerpt": intent,
                }]
            else:
                evidence = [{
                    "source_id": "context:spec_intent",
                    "kind": "requirement",
                    "location": self._anchor(
                        invocation,
                        "context:spec_intent",
                    ),
                    "excerpt": "anchor placeholder",
                }]
        response = json.dumps({
            "schema_version": "assay.review.v2",
            "decision": "revise",
            "reasoning": "Target registration precedes target source creation.",
            "coverage": {
                "complete": True,
                "sources_reviewed": [
                    "artifact",
                    "context:spec_intent",
                    "context:spec_constraints",
                    "context:spec_done_criteria",
                ],
                "omissions": [],
            },
            "findings": [{
                "id": "runner-target-order",
                "criterion_id": "prerequisites",
                "severity": "High",
                "confidence": 0.99,
                "disposition": "confirmed",
                "category": "missing_prereq",
                "targets": [],
                "evidence": evidence,
                "proposal": {
                    "action": "Create each target before registering it.",
                    "replacement": None,
                },
                "attack_spec": None,
            }],
            "criterion_results": self._criterion_results(invocation),
        })
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(
                input_tokens=10,
                output_tokens=20,
                cost_usd=0.0,
                cache_creation_tokens=self.cache_creation_tokens,
                cache_read_tokens=self.cache_read_tokens,
            ),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


def test_chunk_policy_disables_unavailable_source_reference_modes():
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")

    policy = _assay_response_repair_policy(
        backend=backend,
        rollout="auto",
        chunk_protocol=True,
    )

    assert policy is not None
    assert policy.repair_unique_exact_excerpt_char_spans is True
    assert (
        policy.repair_unresolved_source_anchor_from_unique_exact_excerpt
        is False
    )
    assert (
        policy.repair_unique_whitespace_normalized_excerpt_char_spans
        is False
    )
    assert (
        policy.repair_unique_json_quote_escaped_excerpt_char_spans
        is False
    )
    assert policy.derive_decision_from_criterion_results is True
    assert policy.materialize_source_anchors is False
    assert policy.materialize_json_paths is False


def test_explicit_anchor_policy_is_backend_neutral():
    backend = _AnchorResponseBackend(model="gpt-5.6-sol")

    policy = _assay_response_repair_policy(
        backend=backend,
        rollout="anchors",
        chunk_protocol=False,
    )

    assert policy is not None
    assert policy.materialize_source_anchors is True
    assert policy.materialize_json_paths is True
    assert (
        policy.repair_unresolved_source_anchor_from_unique_exact_excerpt
        is True
    )
    assert (
        policy.repair_unique_whitespace_normalized_excerpt_char_spans
        is True
    )
    assert (
        policy.repair_unique_json_quote_escaped_excerpt_char_spans
        is False
    )


def test_deterministic_source_evidence_repair_is_minimal_and_backend_neutral():
    backend = _AnchorResponseBackend(model="claude-sonnet-4-5")

    policy = _assay_response_repair_policy(
        backend=backend,
        rollout="auto",
        chunk_protocol=False,
        deterministic_source_evidence_repair=True,
    )

    assert policy is not None
    assert policy.repair_unique_exact_excerpt_char_spans is True
    assert (
        policy.repair_unique_json_quote_escaped_excerpt_char_spans
        is True
    )
    assert (
        policy.repair_unresolved_source_anchor_from_unique_exact_excerpt
        is False
    )
    assert (
        policy.repair_unique_whitespace_normalized_excerpt_char_spans
        is False
    )
    assert policy.materialize_source_anchors is False
    assert policy.materialize_json_paths is False
    assert policy.require_source_anchors is False
    assert policy.derive_decision_from_criterion_results is False


def test_deterministic_source_evidence_repair_respects_off_and_chunk_modes():
    backend = _AnchorResponseBackend(model="claude-sonnet-4-5")

    for rollout, chunk_protocol in (("off", False), ("auto", True)):
        assert _assay_response_repair_policy(
            backend=backend,
            rollout=rollout,
            chunk_protocol=chunk_protocol,
            deterministic_source_evidence_repair=True,
        ) is None


@pytest.mark.parametrize(
    ("effective_value", "expected"),
    ((True, True), (False, False), (None, False), ("true", False)),
)
def test_effective_anchor_catalog_capability_is_harness_owned(
    tmp_path,
    effective_value,
    expected,
):
    corpus = Corpus(tmp_path / "corpus")
    policy = {}
    if effective_value is not None:
        policy["materialize_source_anchors"] = effective_value
    rubric_ref = corpus.put_artifact(
        json.dumps({
            "response_repair_policy": policy,
            "response_repair_capabilities": {
                "source_anchor_catalog_available": effective_value is True,
                "source_anchor_count": 1 if effective_value is True else 0,
            },
        }).encode(),
        kind="rubrics",
        name_hint="effective-policy.json",
    )

    assert _effective_source_anchor_catalog_available(
        corpus=corpus,
        verdict=SimpleNamespace(rubric_ref=rubric_ref),
    ) is expected


def test_effective_anchor_catalog_capability_fails_closed(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    malformed_ref = corpus.put_artifact(
        b"{not-json",
        kind="rubrics",
        name_hint="malformed.json",
    )

    assert not _effective_source_anchor_catalog_available(
        corpus=corpus,
        verdict=SimpleNamespace(rubric_ref=None),
    )
    assert not _effective_source_anchor_catalog_available(
        corpus=corpus,
        verdict=SimpleNamespace(rubric_ref=malformed_ref),
    )


@pytest.mark.parametrize(
    "capabilities",
    (
        None,
        {},
        {"source_anchor_catalog_available": True},
        {"source_anchor_catalog_available": True, "source_anchor_count": 0},
        {"source_anchor_catalog_available": True, "source_anchor_count": True},
        {"source_anchor_catalog_available": False, "source_anchor_count": 1},
    ),
)
def test_effective_anchor_catalog_capability_requires_rendered_nonempty_catalog(
    tmp_path,
    capabilities,
):
    corpus = Corpus(tmp_path / "corpus")
    payload = {
        "response_repair_policy": {"materialize_source_anchors": True},
    }
    if capabilities is not None:
        payload["response_repair_capabilities"] = capabilities
    rubric_ref = corpus.put_artifact(
        json.dumps(payload).encode(),
        kind="rubrics",
        name_hint="effective-policy.json",
    )

    assert not _effective_source_anchor_catalog_available(
        corpus=corpus,
        verdict=SimpleNamespace(rubric_ref=rubric_ref),
    )


def test_non_glm_plan_citation_failure_retries_with_exact_anchor(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend()
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "Edits remain in the declared scope.",
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    assert "ANCHOR-ONLY" in backend.invocations[1].system
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "context:spec_intent"
    assert evidence.excerpt == intent
    assert evidence.location == f"chars:0-{len(intent)}"
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert len(manifest["attempts"]) == 2
    assert manifest["terminal_reason"] == "repaired"


def test_plan_exact_unique_excerpt_repairs_mistyped_anchor_without_retry(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(
        mistyped_anchor_first_response=True,
    )
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "Edits remain in the declared scope.",
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "context:spec_intent"
    assert evidence.location == f"chars:0-{len(intent)}"
    assert evidence.excerpt == intent
    assert "review_retry_manifest" not in verdict.artifacts
    telemetry = json.loads(corpus.get_artifact(
        verdict.artifacts["review_call_telemetry"]
    ))
    assert telemetry["schema_version"] == (
        "crucible.review-call-telemetry.v1"
    )
    assert telemetry["terminal_reason"] == "complete_first_attempt"
    assert len(telemetry["attempts"]) == 1
    call = telemetry["attempts"][0]
    assert call["repair_cause"] is None
    assert call["rendered_prompt_bytes"] > 0
    assert call["raw_response_bytes"] > 0
    assert call["source_payload_bytes"] > 0
    assert call["source_payload_count"] >= 2
    assert call["physical_turn_usage"] is None
    assert call["usage_scope"] == "logical_call_aggregate"
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["actions"][0]["operation"] == (
        "repair_unresolved_source_anchor_from_unique_exact_excerpt"
    )


def test_plan_unique_whitespace_normalized_excerpt_repairs_without_retry(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(
        whitespace_normalized_first_response=True,
    )
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "Edits remain in the declared scope.",
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "context:spec_intent"
    assert evidence.location == f"chars:0-{len(intent)}"
    assert evidence.excerpt == intent
    assert "review_retry_manifest" not in verdict.artifacts
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["actions"][0]["operation"] == (
        "repair_unique_whitespace_normalized_excerpt_char_span"
    )


def test_required_anchor_plan_does_not_repair_reflowed_quote(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(
        whitespace_normalized_first_response=True,
    )
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "Edits remain in the declared scope.",
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors_only",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    assert "ANCHOR-ONLY" in backend.invocations[1].system
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert manifest["terminal_reason"] == "repaired"
    first_rubric = json.loads(corpus.get_artifact(
        manifest["attempts"][0]["rubric_ref"]
    ))
    assert first_rubric["response_repair_policy"]["require_source_anchors"] is True
    assert first_rubric["response_repair_policy"][
        "repair_unique_whitespace_normalized_excerpt_char_spans"
    ] is False


def test_dynamic_anchor_only_retry_does_not_repair_reflowed_quote(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(
        json_path_then_whitespace_anchor_retry=True,
    )
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "Edits remain in the declared scope.",
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors",
        response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert len(backend.invocations) == 2
    assert "ANCHOR-ONLY" in backend.invocations[1].system
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    second_rubric = json.loads(corpus.get_artifact(
        manifest["attempts"][1]["rubric_ref"]
    ))
    policy = second_rubric["response_repair_policy"]
    assert policy["require_source_anchors"] is True
    assert policy[
        "repair_unique_whitespace_normalized_excerpt_char_spans"
    ] is False
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert all(
        action["operation"]
        != "repair_unique_whitespace_normalized_excerpt_char_span"
        for action in repair["actions"]
    )


def test_plan_citation_retry_does_not_require_absent_effective_anchors(
    tmp_path,
    monkeypatch,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(expect_anchor_retry=False)
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )
    monkeypatch.setitem(
        review.__globals__,
        "_effective_source_anchor_catalog_available",
        lambda **_kwargs: False,
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "Edits remain in the declared scope.",
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    assert "ANCHOR-ONLY" not in backend.invocations[1].system
    evidence = verdict.findings[0].evidence[0]
    assert evidence.location == f"chars:0-{len(intent)}"
    assert evidence.excerpt == intent
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert manifest["terminal_reason"] == "repaired"


def test_real_assay_degradation_suppresses_anchor_only_retry(
    tmp_path,
    monkeypatch,
    assay_review_module,
):
    # Keep the optional-anchor form above the envelope while leaving enough
    # room for the anchor-degraded repair prompt after plan_critic gained its
    # structured criterion contract.
    monkeypatch.setattr(assay_review_module, "_MAX_ARTIFACT_CHARS", 19_000)
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(expect_anchor_retry=False)
    intent, context = _anchor_limit_context()

    verdict = review(
        b"x",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context=context,
        review_anchor_policy="anchors",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" not in invocation.system
        for invocation in backend.invocations
    )
    assert "anchor:sha256:" not in backend.invocations[1].user
    evidence = verdict.findings[0].evidence[0]
    assert evidence.location == f"chars:0-{len(intent)}"
    assert evidence.excerpt == intent
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert manifest["terminal_reason"] == "repaired"
    assert len(manifest["attempts"]) == 2
    first_rubric = json.loads(corpus.get_artifact(
        manifest["attempts"][0]["rubric_ref"]
    ))
    assert (
        first_rubric["response_repair_policy"][
            "materialize_source_anchors"
        ]
        is False
    )
    degradation = first_rubric["response_repair_degradation"]
    assert degradation["reason"] == "anchor_render_limit_exceeded"
    assert degradation["requested_anchor_render_chars"] > 19_000
    assert degradation["effective_rendered_chars"] <= 19_000
    assert degradation["anchor_render_limit_chars"] == 19_000
    assert (
        degradation["requested_response_repair_policy"][
            "materialize_source_anchors"
        ]
        is True
    )
    assert (
        degradation["effective_response_repair_policy"][
            "materialize_source_anchors"
        ]
        is False
    )


def test_anchor_only_repair_growth_fails_before_second_backend_call(
    tmp_path,
    monkeypatch,
    assay_review_module,
):
    monkeypatch.setattr(assay_review_module, "_MAX_ARTIFACT_CHARS", 20_000)
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(
        cache_creation_tokens=3,
        cache_read_tokens=4,
    )
    _intent, context = _anchor_limit_context()

    verdict = review(
        b"x",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context=context,
        review_anchor_policy="anchors",
        response_retry_budget=1,
        budget_usd=2.0,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "review_prompt_envelope_failure"
    assert len(backend.invocations) == 1
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert manifest["terminal_reason"] == "deterministic_inconclusive"
    assert len(manifest["attempts"]) == 2
    second = manifest["attempts"][1]
    assert second["cost_usd"] == 0.0
    assert second["input_tokens"] == 0
    assert second["output_tokens"] == 0
    assert second["cache_creation_tokens"] == 0
    assert second["cache_read_tokens"] == 0
    assert manifest["total_cache_creation_tokens"] == 3
    assert manifest["total_cache_read_tokens"] == 4
    assert verdict.cache_creation_tokens == 3
    assert verdict.cache_read_tokens == 4
    telemetry = json.loads(corpus.get_artifact(
        verdict.artifacts["review_call_telemetry"]
    ))
    assert telemetry["total_cache_creation_tokens"] == 3
    assert telemetry["total_cache_read_tokens"] == 4
    assert [
        attempt["cache_creation_tokens"]
        for attempt in telemetry["attempts"]
    ] == [3, 0]
    assert [
        attempt["cache_read_tokens"]
        for attempt in telemetry["attempts"]
    ] == [4, 0]
    second_rubric = json.loads(corpus.get_artifact(second["rubric_ref"]))
    degradation = second_rubric["response_repair_degradation"]
    assert degradation["reason"] == "required_anchor_render_limit_exceeded"
    assert second_rubric["response_repair_policy"][
        "require_source_anchors"
    ] is True
    rendered = json.loads(corpus.get_artifact(second["rendered_prompt_ref"]))
    assert len(rendered["system"]) + len(rendered["user"]) > 20_000
    assert json.loads(corpus.get_artifact(second["transcript_ref"])) == []


def test_plan_review_downgrades_to_anchors_only_after_json_failure():
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")

    policy = _assay_response_repair_policy(
        backend=backend,
        rollout="auto",
        chunk_protocol=False,
    )

    assert policy is not None
    assert policy.materialize_source_anchors is True
    assert policy.materialize_json_paths is True
    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name="plan_judge",
        validation_error=(
            "invalid review_v2 response: unresolved deterministic JSON path"
        ),
        response_repair_policy=policy,
    ) is True


def test_anchor_only_policy_is_required_from_first_attempt():
    backend = _AnchorResponseBackend(
        model="kimi-k3.1-pro", name="kimi"
    )

    policy = _assay_response_repair_policy(
        backend=backend,
        rollout="anchors_only",
        chunk_protocol=False,
    )

    assert policy is not None
    assert policy.materialize_source_anchors is True
    assert policy.require_source_anchors is True
    assert policy.materialize_json_paths is False
    assert (
        policy.repair_unresolved_source_anchor_from_unique_exact_excerpt
        is False
    )
    assert (
        policy.repair_unique_whitespace_normalized_excerpt_char_spans
        is False
    )
    chunk_policy = _assay_response_repair_policy(
        backend=backend,
        rollout="anchors_only",
        chunk_protocol=True,
    )
    assert chunk_policy is not None
    assert chunk_policy.materialize_source_anchors is False
    assert chunk_policy.require_source_anchors is False
    assert chunk_policy.materialize_json_paths is False
    assert (
        chunk_policy.repair_unique_whitespace_normalized_excerpt_char_spans
        is False
    )


def test_anchor_only_catalog_has_a_conservative_retained_source_bound(
    monkeypatch,
):
    monkeypatch.setattr(
        "crucible.workflows._review._ANCHOR_ONLY_RETAINED_SOURCE_CHARS",
        12,
    )

    assert _anchor_only_catalog_is_bounded(
        prompt_artifact_text="1234",
        context={"spec_intent": "5678"},
        evidence_sources={},
    ) is True
    assert _anchor_only_catalog_is_bounded(
        prompt_artifact_text="123456",
        context={"spec_intent": "78901"},
        evidence_sources={},
    ) is False
    assert _anchor_only_catalog_is_bounded(
        prompt_artifact_text="\x00" * 3,
        context={},
        evidence_sources={},
    ) is False


def test_control_character_expansion_degrades_required_anchors_before_render(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _PlanCitationTypoBackend(
        expect_anchor_retry=False,
        valid_first_response=True,
    )
    intent = (
        "The Spec-bound runner, not workflow CLI arguments,\n"
        "owns the fixed targets."
    )

    verdict = review(
        b"register target before creating its source",
        load("plan_critic"),
        backend=backend,
        corpus=corpus,
        context={
            "spec_intent": intent,
            "spec_constraints": "\x00" * 30_000,
            "spec_done_criteria": "The runner assertion must pass.",
        },
        review_anchor_policy="anchors_only",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    assert "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" not in (
        backend.invocations[0].system
    )
    rubric = json.loads(corpus.get_artifact(verdict.rubric_ref))
    assert rubric["response_repair_policy"]["require_source_anchors"] is False
    obligations = json.loads(corpus.get_artifact(
        verdict.artifacts["review_obligations"]
    ))
    assert obligations["obligations"]["anchor_policy_resolution"] == {
        "requested": "anchors_only",
        "effective": "anchors",
        "reason": "json_render_bound",
    }
    assert verdict.findings[0].id != "review_prompt_envelope_failure"


def test_large_single_review_degrades_initial_anchor_requirement(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "crucible.workflows._review._ANCHOR_ONLY_RETAINED_SOURCE_CHARS",
        1,
    )
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorResponseBackend(
        model="kimi-k3.1-pro",
        name="kimi",
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        review_anchor_policy="anchors_only",
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    assert "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" not in (
        backend.invocations[0].system
    )
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["policy"]["require_source_anchors"] is False
    assert report["policy"]["materialize_json_paths"] is True
    assert report["policy"]["materialize_source_anchors"] is True
    obligations = json.loads(corpus.get_artifact(
        verdict.artifacts["review_obligations"]
    ))
    assert obligations["obligations"]["anchor_policy_resolution"] == {
        "requested": "anchors_only",
        "effective": "anchors",
        "reason": "json_render_bound",
    }


def test_finding_required_source_sets_are_hash_bound_and_rendered():
    required_set = ("artifact", "context:spec_intent")
    envelope = _review_obligations_envelope(
        required_source_ids=required_set,
        finding_required_source_sets=(required_set,),
        prior_findings=(),
        severity_floor="Medium",
    )

    assert envelope["obligations"]["finding_required_source_sets"] == [
        ["artifact", "context:spec_intent"],
    ]
    encoded = json.dumps(
        envelope["obligations"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert envelope["obligations_sha256"] == (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    obligated = _obligations_rubric(
        load("deepswe_plan_critic"),
        envelope,
        normalize_coverage=True,
    )
    normalized_system = " ".join(obligated.system_prompt.split())
    assert (
        "each finding must include every source in each required "
        "paired-source set (artifact + context:spec_intent)"
        in normalized_system
    )
    assert '"finding_required_source_sets"' in obligated.user_template


def test_review_renders_manifest_finding_source_sets(tmp_path):
    invocations: list[Invocation] = []

    def invoke(invocation: Invocation) -> InvocationResult:
        invocations.append(invocation)
        response = json.dumps({
            "schema_version": "assay.review.v2",
            "decision": "pass",
            "reasoning": "No current defect remains.",
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": [],
        })
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )

    backend = SimpleNamespace(
        name="manifest-obligation-stub",
        model="manifest-obligation-model",
        capability="text",
        invoke=invoke,
    )
    verdict = review(
        b'{"tiers":[]}',
        load("deepswe_plan_critic"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        context={"spec_intent": "Keep the plan empty."},
        review_anchor_policy="off",
    )

    assert verdict.decision == "pass"
    assert len(invocations) == 1
    envelope_text = invocations[0].user.split(
        "BEGIN_CRUCIBLE_REVIEW_OBLIGATIONS\n",
        1,
    )[1].split("\nEND_CRUCIBLE_REVIEW_OBLIGATIONS", 1)[0]
    envelope = json.loads(envelope_text)
    assert envelope["obligations"]["finding_required_source_sets"] == [
        ["artifact", "context:spec_intent"],
    ]
    assert (
        "each finding must include every source in each required paired-source "
        "set (artifact + context:spec_intent)"
        in " ".join(invocations[0].system.split())
    )


@pytest.mark.parametrize(
    "rubric_name",
    (
        "plan_critic",
        "plan_judge",
        "swebench_plan_critic",
        "deepswe_plan_critic",
    ),
)
def test_plan_anchor_repair_scope_comes_from_manifest_metadata(rubric_name):
    backend = _AnchorResponseBackend(model="another-model")
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name=rubric_name,
        validation_error="unresolved deterministic JSON path",
        response_repair_policy=policy,
    ) is True


@pytest.mark.parametrize(
    "validation_error",
    (
        "invalid review_v2 response: unresolved deterministic source anchor",
        (
            "invalid review_v2 response: declared character span does not "
            "select the exact excerpt"
        ),
    ),
)
def test_plan_anchor_only_repair_accepts_exact_live_assay_suffixes(
    validation_error,
):
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name="plan_judge",
        validation_error=validation_error,
        response_repair_policy=policy,
    ) is True


@pytest.mark.parametrize(
    "validation_error",
    (
        (
            "invalid review_v2 response: findings[0].evidence[2].excerpt: "
            "is not a verbatim substring of 'artifact'"
        ),
        (
            "invalid review_v2 response: "
            "criterion_results[12].evidence[0].excerpt: "
            "is not a verbatim substring of 'context:spec_intent'"
        ),
    ),
)
def test_plan_anchor_only_repair_accepts_exact_nonverbatim_evidence_error(
    validation_error,
):
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name="plan_judge",
        validation_error=validation_error,
        response_repair_policy=policy,
    ) is True


@pytest.mark.parametrize(
    "validation_error",
    (
        (
            "findings[0].evidence[0].excerpt: is not a verbatim substring "
            "of 'artifact'"
        ),
        (
            "invalid review_v2 response: finding[0].evidence[0].excerpt: "
            "is not a verbatim substring of 'artifact'"
        ),
        (
            "invalid review_v2 response: findings[x].evidence[0].excerpt: "
            "is not a verbatim substring of 'artifact'"
        ),
        (
            "invalid review_v2 response: findings[0].evidence[0].location: "
            "is not a verbatim substring of 'artifact'"
        ),
        (
            "invalid review_v2 response: findings[0].evidence[0].excerpt: "
            "is not a verbatim substring of 'artifact'; please retry"
        ),
    ),
)
def test_plan_nonverbatim_anchor_fallback_rejects_near_misses(
    validation_error,
):
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name="plan_judge",
        validation_error=validation_error,
        response_repair_policy=policy,
    ) is False


@pytest.mark.parametrize(
    ("model", "rubric_name", "expected"),
    (
        ("another-model", "plan_judge", True),
        ("nvidia/zai-org/glm-5.3", "spec_clarity", False),
    ),
)
def test_nonverbatim_anchor_fallback_is_backend_neutral_and_plan_scoped(
    model,
    rubric_name,
    expected,
):
    backend = _AnchorResponseBackend(model=model)
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )
    error = (
        "invalid review_v2 response: findings[0].evidence[0].excerpt: "
        "is not a verbatim substring of 'artifact'"
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name=rubric_name,
        validation_error=error,
        response_repair_policy=policy,
    ) is expected


def test_nonverbatim_anchor_fallback_retains_non_chunk_policy_gate():
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")
    chunk_policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=False,
        materialize_json_paths=False,
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name="plan_judge",
        validation_error=(
            "invalid review_v2 response: findings[0].evidence[0].excerpt: "
            "is not a verbatim substring of 'artifact'"
        ),
        response_repair_policy=chunk_policy,
    ) is False


@pytest.mark.parametrize(
    ("model", "rubric_name", "validation_error", "expected"),
    (
        (
            "another-model",
            "plan_judge",
            "unresolved deterministic JSON path",
            True,
        ),
        (
            "nvidia/zai-org/glm-5.3",
            "spec_clarity",
            "unresolved deterministic JSON path",
            False,
        ),
        (
            "nvidia/zai-org/glm-5.3",
            "plan_judge",
            "invalid review_v2 response: missing=['findings']",
            False,
        ),
        (
            "nvidia/zai-org/glm-5.3",
            "plan_judge",
            (
                "invalid review_v2 response: unresolved deterministic source "
                "anchor; please retry"
            ),
            False,
        ),
        (
            "nvidia/zai-org/glm-5.3",
            "plan_judge",
            (
                "invalid review_v2 response: declared character span does not "
                "select the excerpt"
            ),
            False,
        ),
    ),
)
def test_anchor_only_fallback_is_exactly_scoped(
    model,
    rubric_name,
    validation_error,
    expected,
):
    backend = _AnchorResponseBackend(model=model)
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )

    assert _plan_anchor_only_repair(
        backend=backend,
        rubric_name=rubric_name,
        validation_error=validation_error,
        response_repair_policy=policy,
    ) is expected


def test_glm_review_gates_are_exact_to_the_current_gateway_id():
    review_module = importlib.import_module("crucible.workflows._review")
    current = SimpleNamespace(
        name="openai_compat",
        model="nvidia/zai-org/glm-5.3",
    )
    retired = SimpleNamespace(
        name="openai_compat",
        model="nvidia/zai-org/glm-5.2",
    )
    length_verdict = SimpleNamespace(
        decision="inconclusive",
        findings=(SimpleNamespace(
            id="backend_max_turns",
            source="deterministic",
            category="backend_error",
            evidence=(SimpleNamespace(
                excerpt="server returned finish_reason=length",
            ),),
        ),),
    )
    policy = ReviewResponseRepairPolicy(
        materialize_source_anchors=True,
        materialize_json_paths=True,
    )

    def gates(backend):
        return (
            _assay_response_repair_policy(
                backend=backend,
                rollout="auto",
                chunk_protocol=False,
            ) is not None,
            review_module._retryable_output_length_error(
                backend=backend,
                verdict=length_verdict,
            ) is not None,
            review_module._glm_plan_decision_consistency_error(
                backend=backend,
                rubric_name="plan_judge",
                validation_error=(
                    review_module._GLM_PLAN_DECISION_CONSISTENCY_ERROR
                ),
                response_repair_policy=policy,
            ),
        )

    assert review_module._NVIDIA_GLM_MODEL == current.model
    assert gates(current) == (True, True, True)
    # The predecessor still answers on the gateway; it gets no recovery.
    assert gates(retired) == (False, False, False)


def test_glm_review_enables_assay_anchor_materialization(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "revise"
    assert "ASSAY DETERMINISTIC EVIDENCE REFERENCES" in (
        backend.invocations[0].system
    )
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "artifact"
    assert evidence.excerpt == _ARTIFACT.decode()
    assert evidence.location == f"chars:0-{len(_ARTIFACT)}"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert report["policy"] == {
        "derive_decision_from_criterion_results": True,
        "materialize_json_paths": True,
        "materialize_source_anchors": True,
        "require_source_anchors": False,
        "repair_unique_exact_excerpt_char_spans": True,
        "repair_unique_json_quote_escaped_excerpt_char_spans": False,
        "repair_unique_whitespace_normalized_excerpt_char_spans": True,
        "repair_unresolved_source_anchor_from_unique_exact_excerpt": True,
        "recover_fully_satisfied_revise_to_pass": False,
        "source_anchor_chars": 512,
    }
    assert report["actions"][0]["operation"] == "materialize_source_anchor"


def test_default_backend_does_not_enable_assay_response_repair(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorResponseBackend(model="gpt-5.6-sol")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "inconclusive"
    assert "ASSAY DETERMINISTIC EVIDENCE REFERENCES" not in (
        backend.invocations[0].system
    )
    assert "review_response_repair" not in verdict.artifacts


def test_deterministic_source_evidence_repair_canonicalizes_without_retry(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _JsonQuoteEscapedResponseBackend()

    verdict = review(
        _JSON_ARTIFACT_TEXT.encode(),
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
        deterministic_source_evidence_repair=True,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    assert "ASSAY DETERMINISTIC EVIDENCE REFERENCES" not in (
        backend.invocations[0].system
    )
    evidence = verdict.findings[0].evidence[0]
    assert evidence.source_id == "artifact"
    assert evidence.excerpt == _JSON_ARTIFACT_TEXT
    assert evidence.location == f"chars:0-{len(_JSON_ARTIFACT_TEXT)}"
    assert "review_retry_manifest" not in verdict.artifacts
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["policy"][
        "repair_unique_json_quote_escaped_excerpt_char_spans"
    ] is True
    assert repair["policy"]["materialize_source_anchors"] is False
    assert repair["policy"]["materialize_json_paths"] is False
    assert repair["policy"][
        "derive_decision_from_criterion_results"
    ] is False
    telemetry = json.loads(corpus.get_artifact(
        verdict.artifacts["review_call_telemetry"]
    ))
    assert telemetry["terminal_reason"] == "complete_first_attempt"
    assert len(telemetry["attempts"]) == 1


def test_breach_judge_repairs_r5_shaped_medium_target_closure_once(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    artifact = _JSON_ARTIFACT_TEXT.encode()
    artifact_ref = corpus.put_artifact(
        artifact,
        kind="reviewed_artifacts",
        name_hint="breaker-input.json",
    )
    transcript_ref = corpus.put_artifact(
        b"[]",
        kind="transcripts",
        name_hint="breaker.json",
    )
    breaker_findings = tuple(
        Finding(
            id=f"attack-{index}",
            severity="Medium",
            category="correctness",
            evidence=(Evidence(
                source_id="artifact",
                kind="source_quote",
                location="post-fix records",
                excerpt=_JSON_ARTIFACT_TEXT,
            ),),
            suggested_action=None,
            source="llm",
            criterion_id="correctness",
            confidence=0.9,
            disposition="suspected",
            targets=(),
            proposal=None,
            attack_spec=AttackSpec(
                setup=f"prepare attack {index}",
                stimulus=f"run attack {index}",
                oracle=f"observe attack {index}",
            ),
        )
        for index in range(3)
    )
    breaker_verdict = Verdict(
        rubric="impl_breaker",
        decision="pass",
        findings=breaker_findings,
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="The candidates are useful non-blocking attacks.",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name="breaker-stub",
        transcript_ref=transcript_ref,
        cost_usd=0.0,
        execution=None,
    )
    judge_input = handoff_artifact(
        artifact,
        (("breaker", breaker_verdict),),
        heading="BREACH BREAKER FINDINGS",
    )
    parsed = parse_review_handoff(judge_input)
    assert parsed is not None
    prior_refs = tuple(item["ref"] for item in parsed.prior_findings)
    backend = _BreachJudgeJsonQuoteEscapedBackend(prior_refs=prior_refs)

    verdict = review(
        judge_input,
        load("breach_judge"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
        deterministic_source_evidence_repair=True,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 1
    assert tuple(
        finding.targets[0] for finding in verdict.findings
    ) == prior_refs
    assert all(
        finding.severity == "Medium"
        and finding.disposition == "addressed"
        and finding.evidence[0].excerpt == _JSON_ARTIFACT_TEXT
        for finding in verdict.findings
    )
    assert "review_retry_manifest" not in verdict.artifacts
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert sum(
        action["operation"]
        == "repair_unique_json_quote_escaped_excerpt_char_span"
        for action in repair["actions"]
    ) == 3
    assert repair["policy"]["materialize_source_anchors"] is False
    assert repair["policy"]["materialize_json_paths"] is False
    telemetry = json.loads(corpus.get_artifact(
        verdict.artifacts["review_call_telemetry"]
    ))
    assert telemetry["terminal_reason"] == "complete_first_attempt"
    assert len(telemetry["attempts"]) == 1


def test_deterministic_source_evidence_repair_requires_a_bool(tmp_path):
    backend = _JsonQuoteEscapedResponseBackend()

    with pytest.raises(
        TypeError,
        match="deterministic_source_evidence_repair must be a bool",
    ):
        review(
            _JSON_ARTIFACT_TEXT.encode(),
            load("spec_clarity"),
            backend=backend,
            corpus=Corpus(tmp_path / "corpus"),
            deterministic_source_evidence_repair=1,
        )

    assert backend.invocations == []


def test_off_policy_cannot_be_bypassed_with_direct_assay_policy(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorResponseBackend(model="nvidia/zai-org/glm-5.3")

    with pytest.raises(TypeError, match="response_repair_policy is adapter-owned"):
        review(
            _ARTIFACT,
            load("spec_clarity"),
            backend=backend,
            corpus=corpus,
            review_anchor_policy="off",
            response_repair_policy=ReviewResponseRepairPolicy(
                materialize_source_anchors=True,
            ),
        )

    assert backend.invocations == []


def test_backend_capability_can_opt_in_to_assay_response_repair(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorResponseBackend(
        model="custom-review-model",
        supports_review_evidence_anchors=True,
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == "revise"
    assert "review_response_repair" in verdict.artifacts


def test_retry_manifest_retains_each_assay_response_repair_ref(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorResponseBackend(
        model="nvidia/zai-org/glm-5.3",
        invalid_first_contract=True,
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    manifest = json.loads(corpus.get_artifact(
        verdict.artifacts["review_retry_manifest"]
    ))
    assert all(
        attempt["response_repair_ref"].startswith("sha256:")
        for attempt in manifest["attempts"]
    )
    for attempt in manifest["attempts"]:
        raw_ref = attempt["raw_response_ref"]
        assert raw_ref.startswith("sha256:")
        raw_response = corpus.get_artifact(raw_ref)
        assert raw_ref == "sha256:" + hashlib.sha256(raw_response).hexdigest()
        repair_report = json.loads(corpus.get_artifact(
            attempt["response_repair_ref"]
        ))
        assert repair_report["raw_response_ref"] == raw_ref
        assert repair_report["raw_response_sha256"] == raw_ref


def test_target_alias_expansion_is_audited_and_never_deduplicates(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    finding_ref = (
        "prior:critic:0:sha256-" + "a" * 64 + ":ambiguous-limit"
    )
    prior = ({
        "ref": finding_ref,
        "source_id": "context:prior_critic_findings",
        "severity": "High",
    },)
    envelope = _review_obligations_envelope(
        required_source_ids=("artifact",),
        prior_findings=prior,
        severity_floor="High",
        include_material_priors=True,
        include_target_aliases=True,
    )
    alias_record = envelope["obligations"]["material_prior_aliases"][0]
    alias = alias_record["alias"]
    encoded = json.dumps(
        envelope["obligations"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert envelope["obligations_sha256"] == (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )

    artifact_ref = corpus.put_artifact(_ARTIFACT, kind="test", name_hint="input")
    transcript_ref = corpus.put_artifact(b"[]", kind="test", name_hint="raw")
    finding = Finding(
        id="adjudication",
        severity="High",
        category="ambiguity",
        evidence=(Evidence(
            source_id="artifact",
            kind="source_quote",
            location="artifact",
            excerpt=_ARTIFACT.decode(),
        ),),
        suggested_action="clarify",
        source="llm",
        criterion_id="intent_clarity",
        confidence=0.9,
        disposition="suspected",
        targets=(alias, alias),
        proposal=Proposal(action="clarify", replacement=None),
    )
    verdict = Verdict(
        rubric="spec_clarity",
        decision="revise",
        findings=(finding,),
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning="needs clarification",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name="stub",
        transcript_ref=transcript_ref,
        cost_usd=0.0,
        execution=None,
    )

    expanded = _expand_material_prior_aliases(
        corpus=corpus,
        verdict=verdict,
        aliases={alias: finding_ref},
        attempt=0,
    )

    assert expanded.findings[0].targets == (finding_ref, finding_ref)
    assert expanded.findings[0].disposition == "suspected"
    ledger = _material_prior_target_repair_ledger(
        verdict=expanded,
        aliases={alias: finding_ref},
    )
    assert ledger is not None
    assert ledger["duplicate_observations"] == [{
        "alias": alias,
        "occurrences": [
            {
                "path": "findings[0].targets[0]",
                "disposition": "suspected",
            },
            {
                "path": "findings[0].targets[1]",
                "disposition": "suspected",
            },
        ],
        "same_terminal_disposition": False,
    }]
    manifest = json.loads(corpus.get_artifact(
        expanded.artifacts["review_target_alias_expansion"]
    ))
    assert [action["operation"] for action in manifest["actions"]] == [
        "expand_material_prior_alias",
        "expand_material_prior_alias",
    ]


def test_nonclosing_obligations_do_not_expose_material_prior_protocol():
    finding_ref = "prior:critic:0:sha256-" + "a" * 64 + ":blocker"
    envelope = _review_obligations_envelope(
        required_source_ids=("artifact",),
        prior_findings=({
            "ref": finding_ref,
            "source_id": "context:prior_critic_findings",
            "severity": "High",
            "disposition": "confirmed",
        },),
        severity_floor="High",
    )

    assert envelope["obligations"] == {
        "required_source_ids": ["artifact"],
    }
    with pytest.raises(ValueError, match="material-prior closure"):
        _review_obligations_envelope(
            required_source_ids=("artifact",),
            prior_findings=(),
            severity_floor="High",
            include_target_aliases=True,
        )
